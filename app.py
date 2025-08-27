import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                           QComboBox, QSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import yaml
from shapely.geometry import Polygon
from datetime import datetime
import time
from norfair import Detection, Tracker
from collections import defaultdict
import numpy as np

class VehicleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Detection System")
        self.setGeometry(100, 100, 1280, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Video display area with fixed size matching our processing resolution
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setMaximumSize(640, 360)
        layout.addWidget(self.video_label)

        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel)

        # Add controls
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        control_layout.addWidget(self.load_video_btn)

        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_detection)
        control_layout.addWidget(self.stop_btn)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        control_layout.addWidget(self.results_text)

        # Initialize variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.load_models()
        self.load_config()
        self.setup_tracker()
        self.tracked_ids = set()
        self.unique_vehicle_counts = defaultdict(int)
        self.track_positions = {}  # Store previous positions of tracked objects
        self.counted_ids = set()   # IDs that have been counted

    def load_config(self):
        with open("config2.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.DEVICE_ID = self.config.get("device_id", "UNKNOWN")
        self.roi_coords = self.config["roi_polygon"]
        # Set fixed processing frame size to 640x360 as requested
        self.PROCESSING_WIDTH = 640
        self.PROCESSING_HEIGHT = 360
        # Use the same dimensions for output to maintain consistency
        self.OUTPUT_WIDTH = 640
        self.OUTPUT_HEIGHT = 360
        self.NEW_PT_CONFIDENCE = 0.2
        self.YOLOV8N_CONFIDENCE = 0.4
        self.IOU_THRESHOLD = 0.3
        
        # Check if we have a counting zone (polygon) or counting line
        if "counting_zone" in self.config:
            self.counting_zone = self.config["counting_zone"]
            self.use_counting_zone = True
            print(f"Loaded counting zone from config with {len(self.counting_zone)} points")
        elif "counting_line" in self.config:
            self.line_start = self.config["counting_line"]["start"]
            self.line_end = self.config["counting_line"]["end"]
            self.use_counting_zone = False
            print(f"Loaded counting line from config: {self.line_start} to {self.line_end}")
        else:
            # Default: horizontal line in the middle
            self.line_start = [0, self.PROCESSING_HEIGHT // 2]
            self.line_end = [self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT // 2]
            self.use_counting_zone = False
            print(f"Using default counting line: {self.line_start} to {self.line_end}")
            
        # Initialize variables for zone-based counting
        self.object_in_zone = set()  # Track IDs of objects currently in the counting zone

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0
    
    def point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using the ray casting algorithm.
        
        Args:
            point: Tuple (x, y) representing the point
            polygon: List of tuples [(x1, y1), (x2, y2), ...] representing the polygon
            
        Returns:
            True if the point is inside the polygon, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersect:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    def load_models(self):
        self.results_text.append("Loading models...")
        self.yolov8_model = YOLO("yolov8s_openvino_model/")
        self.custom_model = YOLO("v5_pretrained_openvino_model/")
        self.results_text.append("Models loaded successfully!")

    def setup_tracker(self):
        def euclidean_distance(detection, tracked_obj):
            return np.linalg.norm(detection.points - tracked_obj.estimate)
        self.tracker = Tracker(distance_threshold=30, distance_function=euclidean_distance)

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", 
                                                "Video Files (*.mp4 *.avi)")
        if filename:
            self.cap = cv2.VideoCapture(filename)
            self.results_text.append(f"Loaded video: {filename}")
            
    def start_detection(self):
        if self.cap is None:
            self.results_text.append("Please load a video first!")
            return
        self.timer.start(30)  # Update every 30ms
        self.results_text.append("Detection started...")

    def stop_detection(self):
        self.timer.stop()
        self.results_text.append("Detection stopped.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.results_text.append("Video finished.")
            return

        # Immediately resize frame to our fixed 640x360 resolution
        frame = cv2.resize(frame, (self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT))
        # Use the same frame for processing (no separate scaling)
        processing_frame = frame.copy()
        
        # Run both models
        yolov8_dets = []
        custom_dets = []
        
        # YOLOv8 detections
        results = self.yolov8_model(processing_frame, conf=self.YOLOV8N_CONFIDENCE, 
                                  iou=self.IOU_THRESHOLD, device="intel:gpu")
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolov8_model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolov8_dets.append((cls_name, (x1, y1, x2, y2), box))

        # Custom model detections
        results = self.custom_model(processing_frame, conf=self.NEW_PT_CONFIDENCE, 
                                  iou=self.IOU_THRESHOLD, device="intel:gpu")
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.custom_model.names[cls_id]
                if cls_name in {'person', 'bus'}:  # Skip these classes
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                custom_dets.append((cls_name, (x1, y1, x2, y2), box))

        # Deduplicate detections
        final_dets = yolov8_dets + [
            (cls, box, b) for cls, box, b in custom_dets
            if all(self.iou(box, ybox) <= 0.5 for _, ybox, _ in yolov8_dets)
        ]

        # Convert to Norfair detections
        norfair_detections = []
        det_classes = []
        for cls_name, bbox, _ in final_dets:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            norfair_detections.append(Detection(points=np.array([cx, cy])))
            det_classes.append(cls_name)

        # Update tracker
        tracked_objects = self.tracker.update(detections=norfair_detections)

        # No scaling factors needed since we process and display at the same resolution (640x360)
        width_scale_factor = 1.0
        height_scale_factor = 1.0
        
        # Print debug info only once at the beginning
        if not hasattr(self, 'debug_printed'):
            print(f"DEBUG: Fixed processing resolution: {self.PROCESSING_WIDTH}x{self.PROCESSING_HEIGHT}")
            print(f"DEBUG: Using fixed frame size 640x360 for all videos")
            self.debug_printed = True
        
        if self.use_counting_zone:
            # With fixed 640x360 frame, no scaling is needed for the counting zone
            scaled_polygon = []
            for point in self.counting_zone:
                # No scaling needed since we're at fixed resolution
                x = int(point[0]) 
                y = int(point[1])
                scaled_polygon.append((x, y))
            
            # Draw the polygon
            if len(scaled_polygon) >= 3:
                polygon_array = np.array(scaled_polygon, np.int32)
                polygon_array = polygon_array.reshape((-1, 1, 2))
                cv2.polylines(frame, [polygon_array], isClosed=True, color=(0, 255, 255), thickness=2)
                
                # Fill with transparent color
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_array], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # 20% opacity
                
                # Add label
                centroid_x = sum(p[0] for p in scaled_polygon) // len(scaled_polygon)
                centroid_y = sum(p[1] for p in scaled_polygon) // len(scaled_polygon)
                cv2.putText(frame, "Counting Zone", (centroid_x - 40, centroid_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Draw traditional counting line - no scaling needed with fixed frame size
            start_x = int(self.line_start[0]) 
            start_y = int(self.line_start[1])
            end_x = int(self.line_end[0])
            end_y = int(self.line_end[1])
            
            # Make the line with reduced thickness
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # Reduced thickness
            # Add smaller circle endpoints
            cv2.circle(frame, (start_x, start_y), 3, (255, 0, 0), -1)  # Blue start point, smaller
            cv2.circle(frame, (end_x, end_y), 3, (0, 255, 0), -1)      # Green end point, smaller
            # Add smaller label
            cv2.putText(frame, "Counting Line", (start_x + 10, start_y - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw tracked objects and count
        for track in tracked_objects:
            track_id = track.id
            point = track.estimate[0]

            # Find closest detection
            distances = [np.linalg.norm(point - d.points[0]) for d in norfair_detections]
            if distances:
                min_idx = np.argmin(distances)
                cls_name = det_classes[min_idx]
                bbox = final_dets[min_idx][1]
            else:
                continue

            # No need to scale since we're using the same resolution for processing and display
            x1, y1, x2, y2 = bbox
            x1o = int(x1)
            y1o = int(y1)
            x2o = int(x2)
            y2o = int(y2)
            
            # Calculate center point
            center_x = (x1o + x2o) // 2
            center_y = (y1o + y2o) // 2
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), color, 1)
            cv2.putText(frame, f"{cls_name} ID:{track_id}", (x1o, y1o-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Get current position of the object
            current_position = (center_x, center_y)
            
            if self.use_counting_zone:
                # For polygon-based counting
                # Scale the counting zone to match the original frame size
                scaled_polygon = []
                for point in self.counting_zone:
                    x = int(point[0] * width_scale_factor)
                    y = int(point[1] * height_scale_factor)
                    scaled_polygon.append((x, y))
                
                # Check if the object is inside the polygon
                is_in_zone = self.point_in_polygon(current_position, scaled_polygon)
                
                # Count the object if it wasn't previously in the zone but is now
                if is_in_zone:
                    if track_id not in self.object_in_zone:
                        self.object_in_zone.add(track_id)
                        # Only count if not already counted
                        if track_id not in self.counted_ids:
                            self.unique_vehicle_counts[cls_name] += 1
                            self.counted_ids.add(track_id)
                            # Highlight the object when counted
                            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 0, 255), 2)
                else:
                    # Object left the zone
                    if track_id in self.object_in_zone:
                        self.object_in_zone.remove(track_id)
            else:
                # Traditional line crossing detection
                if track_id in self.track_positions:
                    previous_position = self.track_positions[track_id]
                    # Check if object crossed the line
                    if self.line_crossed(previous_position, current_position, 
                                        (start_x, start_y), (end_x, end_y)) and track_id not in self.counted_ids:
                        self.unique_vehicle_counts[cls_name] += 1
                        self.counted_ids.add(track_id)
                        # Highlight the object when counted
                        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 0, 255), 2)
            
            # Update position history
            self.track_positions[track_id] = current_position

        # Display counts on frame
        y_offset = 25  # Reduced starting offset
        for cls, count in self.unique_vehicle_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Smaller font and thinner text
            y_offset += 20  # Reduced spacing
            # Update counts in text display
            self.results_text.setText('\n'.join([f"{k}: {v}" for k, v in self.unique_vehicle_counts.items()]))

        # We've replaced the ROI polygon with a counting line in the middle of the screen

        # No need to resize again since we're already at our target resolution (640x360)
        # Just convert to RGB for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def line_crossed(self, p1, p2, l1, l2):
        """Check if line segment p1-p2 crosses line segment l1-l2"""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        # Check if line segments intersect
        return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)
            
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VehicleDetectionApp()
    window.show()
    sys.exit(app.exec_())
