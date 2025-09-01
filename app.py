# GitHub Repository: https://github.com/sajadulakash/polygon-based-traffic-monitor
# 
# Polygon-based Traffic Monitor
# ===========================
# This application provides a traffic monitoring system that uses polygon-based counting zones
# to track and count vehicles and other objects passing through predefined areas.
# The system supports video files and RTSP streams, and includes tools for defining custom counting zones.
#
# Features:
# - Object detection using YOLOv8 and custom models with OpenVINO acceleration
# - Real-time tracking and counting of objects within user-defined polygon zones
# - CSV logging of object entry/exit times with custom ID system
# - RTSP stream support for live monitoring
# - Integrated coordinate selection tool for easy zone definition
# 
# Author: sajadulakash
# License: MIT
# Version: 1.0

import sys
import cv2
import threading
import queue
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                           QComboBox, QSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import yaml
from shapely.geometry import Polygon
from datetime import datetime
import time
from norfair import Detection, Tracker
from collections import defaultdict
import numpy as np

class ProcessingWorker(QObject):
    """Worker thread for video capture and processing to avoid blocking the GUI thread."""
    # Signal to send processed results back to the main thread
    result_ready = pyqtSignal(dict)
    
    def __init__(self, cap, yolov8_model, custom_model, config, is_rtsp=False):
        super().__init__()
        self.cap = cap
        self.yolov8_model = yolov8_model
        self.custom_model = custom_model
        self.config = config
        self.is_rtsp = is_rtsp
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
        self.result_queue = queue.Queue(maxsize=2)  # Keep limited results
        self.YOLOV8N_CONFIDENCE = 0.4
        self.NEW_PT_CONFIDENCE = 0.2
        self.IOU_THRESHOLD = 0.3
        self.PROCESSING_WIDTH = 640
        self.PROCESSING_HEIGHT = 360
        
        # Frame rate control
        if not is_rtsp and cap is not None and cap.isOpened():
            self.original_fps = cap.get(cv2.CAP_PROP_FPS)
            # If FPS is not available or unreasonable, default to 30 FPS
            if self.original_fps <= 0 or self.original_fps > 120:
                self.original_fps = 30.0
            self.frame_delay = 1.0 / self.original_fps
        else:
            self.original_fps = 30.0
            self.frame_delay = 1.0 / 30.0
        
    def start(self):
        """Start the worker threads"""
        self.running = True
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop(self):
        """Stop all worker threads"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
    
    def _capture_loop(self):
        """Thread that continuously captures frames from the video source at the original video's frame rate"""
        frame_count = 0
        end_of_video_reported = False
        last_frame_time = 0
        
        while self.running:
            # Check if video capture is valid
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            # Get the original video's frame rate for regular videos (not RTSP)
            if not self.is_rtsp and not hasattr(self, 'original_fps'):
                self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
                # If FPS is not available or unreasonable, default to 30 FPS
                if self.original_fps <= 0 or self.original_fps > 120:
                    self.original_fps = 30.0
                self.frame_delay = 1.0 / self.original_fps
                print(f"Video FPS: {self.original_fps}, frame delay: {self.frame_delay:.4f} seconds")
            
            # For non-RTSP videos, control the frame rate to match the original video
            if not self.is_rtsp:
                # Calculate time since last frame
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Sleep to maintain original video frame rate
                if elapsed < self.frame_delay:
                    time.sleep(max(0, self.frame_delay - elapsed))
                
                # Update last frame time
                last_frame_time = time.time()
                
            # For RTSP streams, clear the buffer before reading
            if self.is_rtsp:
                # Skip frames to get to the most recent one
                for _ in range(2):  # Skip buffered frames
                    self.cap.grab()
            
            # Read the actual frame
            ret, frame = self.cap.read()
            if not ret:
                # Handle end of video
                if not end_of_video_reported and not self.is_rtsp:
                    print("End of video reached or frame read error")
                    end_of_video_reported = True
                    
                    # For regular video files, rewind to start for looping playback
                    if self.cap is not None:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
                time.sleep(0.1)
                continue
                
            # Reset the end of video flag if we successfully read a frame
            end_of_video_reported = False
            frame_count += 1
            
            # Log occasional frame progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
                
            try:
                # Resize the frame to our processing resolution
                frame = cv2.resize(frame, (self.PROCESSING_WIDTH, self.PROCESSING_HEIGHT))
                
                # Put in queue, dropping old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Discard old frame
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame, block=False)
            except Exception as e:
                print(f"Error in capture loop: {str(e)}")
                time.sleep(0.1)
            except queue.Full:
                pass  # Skip this frame if queue is still full
    
    def _process_loop(self):
        """Thread that processes frames"""
        while self.running:
            try:
                # Get the latest frame
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process the frame
                start_time = time.time()
                result = self._process_frame(frame)
                processing_time = time.time() - start_time
                
                # Add processing time to the result
                result['processing_time'] = processing_time
                result['frame'] = frame
                
                # Emit the result
                self.result_ready.emit(result)
                
            except queue.Empty:
                continue
    
    def _process_frame(self, frame):
        """Process a single frame with both models"""
        yolov8_dets = []
        custom_dets = []
        
        try:
            # Try with Intel GPU first
            device = "intel:gpu"
            try:
                # Process with YOLOv8
                results = self.yolov8_model(frame, conf=self.YOLOV8N_CONFIDENCE, 
                                          iou=self.IOU_THRESHOLD, device=device)
            except (RuntimeError, Exception) as e:
                # Fall back to CPU if GPU fails
                print(f"GPU inference failed, falling back to CPU: {str(e)}")
                device = "cpu"
                results = self.yolov8_model(frame, conf=self.YOLOV8N_CONFIDENCE, 
                                          iou=self.IOU_THRESHOLD, device=device)
                
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolov8_model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolov8_dets.append((cls_name, (x1, y1, x2, y2), box))

            # Only run custom model if we have enough time (not for RTSP real-time)
            # or if we're processing regular video files
            if not self.is_rtsp:
                # Custom model detections using the same device that worked for YOLOv8
                results = self.custom_model(frame, conf=self.NEW_PT_CONFIDENCE, 
                                          iou=self.IOU_THRESHOLD, device=device)
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.custom_model.names[cls_id]
                        if cls_name in {'person', 'bus'}:  # Skip these classes
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        custom_dets.append((cls_name, (x1, y1, x2, y2), box))
        
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            # Return empty detections instead of crashing
            return {'detections': []}

        # Deduplicate detections
        final_dets = yolov8_dets + [
            (cls, box, b) for cls, box, b in custom_dets
            if all(self.iou(box, ybox) <= 0.5 for _, ybox, _ in yolov8_dets)
        ]
        
        return {
            'detections': final_dets
        }
        
    def iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0

class VehicleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainCount")
        self.setGeometry(100, 100, 1280, 800)  # Slightly wider to accommodate controls

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
        
        # RTSP Link section
        rtsp_layout = QHBoxLayout()
        self.rtsp_input = QTextEdit()
        self.rtsp_input.setPlaceholderText("Enter RTSP URL (e.g., rtsp://username:password@ip:port/stream)")
        self.rtsp_input.setMaximumHeight(50)
        rtsp_layout.addWidget(self.rtsp_input)
        
        self.rtsp_btn = QPushButton("Connect RTSP")
        self.rtsp_btn.clicked.connect(self.connect_rtsp)
        rtsp_layout.addWidget(self.rtsp_btn)
        control_layout.addLayout(rtsp_layout)

        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_detection)
        control_layout.addWidget(self.stop_btn)
        
        # Add button to launch get_coordinates.py
        self.coords_btn = QPushButton("Define Counting Zone")
        self.coords_btn.clicked.connect(self.launch_coordinate_selector)
        self.coords_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        control_layout.addWidget(self.coords_btn)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        control_layout.addWidget(self.results_text)

        # Initialize variables
        self.cap = None
        self.worker = None
        self.is_rtsp = False
        self.latest_frame = None
        self.latest_detections = []
        self.processing_fps = 0
        self.display_fps = 0
        self.last_frame_time = time.time()
        
        # Setup timer for UI updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Load models and setup
        self.load_models()
        self.load_config()
        self.setup_tracker()
        self.tracked_ids = set()
        self.unique_vehicle_counts = defaultdict(int)
        self.track_positions = {}  # Store previous positions of tracked objects
        self.counted_ids = set()   # IDs that have been counted
        
        # Object tracking for CSV logging
        self.object_entry_times = {}  # Track when objects enter the zone {track_id: (timestamp, class_name)}
        self.object_exit_times = {}   # Track when objects exit the zone {track_id: timestamp}
        self.logged_objects = set()   # Track IDs that have already been logged to CSV
        
        # Custom ID system
        self.session_id = datetime.now().strftime("%Y%m%d")
        self.custom_id_counter = defaultdict(int)  # Counter per class type
        self.track_id_to_custom = {}  # Maps Norfair track.id to our custom ID format
        
        # Try to load previous counter state if available
        self.load_counter_state()
        
        # Initialize CSV logging
        self.setup_csv_logging()

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
        
        # Load the counting zone (polygon) from the config
        if "counting_zone" in self.config:
            self.counting_zone = self.config["counting_zone"]
            print(f"Loaded counting zone from config with {len(self.counting_zone)} points")
        else:
            # Default: create a default counting zone in the center of the frame
            center_x, center_y = self.PROCESSING_WIDTH // 2, self.PROCESSING_HEIGHT // 2
            offset_x, offset_y = self.PROCESSING_WIDTH // 4, self.PROCESSING_HEIGHT // 4
            self.counting_zone = [
                [center_x - offset_x, center_y - offset_y],
                [center_x + offset_x, center_y - offset_y],
                [center_x + offset_x, center_y + offset_y],
                [center_x - offset_x, center_y + offset_y]
            ]
            print(f"Using default counting zone with 4 points")
            
        # Initialize variables for zone-based counting
        self.object_in_zone = set()  # Track IDs of objects currently in the counting zone
        
    def get_custom_id(self, track_id, cls_name):
        """Generate or retrieve a custom ID for a track"""
        if track_id in self.track_id_to_custom:
            return self.track_id_to_custom[track_id]
            
        # Create new custom ID
        self.custom_id_counter[cls_name] += 1
        custom_id = f"{cls_name}-{self.session_id}-{self.custom_id_counter[cls_name]:06d}"
        
        # Store mapping
        self.track_id_to_custom[track_id] = custom_id
        print(f"Assigned custom ID {custom_id} to track {track_id}")
        
        # Save updated counter state
        self.save_counter_state()
        
        return custom_id
        
    def save_counter_state(self):
        """Save counter state to disk for persistence across sessions"""
        try:
            # Create data directory if it doesn't exist
            if not os.path.exists('data'):
                os.makedirs('data')
                
            state_file = os.path.join('data', 'counter_state.yaml')
            state_data = {
                'session_id': self.session_id,
                'counters': dict(self.custom_id_counter)
            }
            
            with open(state_file, 'w') as f:
                yaml.dump(state_data, f)
        except Exception as e:
            print(f"Error saving counter state: {str(e)}")
    
    def load_counter_state(self):
        """Load counter state from disk if available"""
        try:
            state_file = os.path.join('data', 'counter_state.yaml')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = yaml.safe_load(f)
                
                # If it's the same day, use the saved counters
                if state_data.get('session_id') == self.session_id:
                    for cls_name, count in state_data.get('counters', {}).items():
                        self.custom_id_counter[cls_name] = count
                    print(f"Loaded previous counter state for session {self.session_id}")
        except Exception as e:
            print(f"Error loading counter state: {str(e)}")
    
    def setup_csv_logging(self):
        """Set up CSV logging for object detection data"""
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Generate filename with current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.csv_filename = os.path.join('data', f'data_{current_date}.csv')
        
        # Check if file exists and create with headers if it doesn't
        file_exists = os.path.isfile(self.csv_filename)
        
        if not file_exists:
            with open(self.csv_filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Object_ID', 'Class_Name', 'Zone_In_Timestamp', 'Zone_Out_Timestamp'])
                self.results_text.append(f"Created new CSV log file: {self.csv_filename}")
        else:
            self.results_text.append(f"Using existing CSV log file: {self.csv_filename}")
            
    def log_object_to_csv(self, track_id, class_name, entry_time, exit_time):
        """Log an object's entry and exit times to the CSV file"""
        try:
            # Get or create custom ID for this track
            custom_id = self.get_custom_id(track_id, class_name)
            
            with open(self.csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Format timestamps for better readability
                entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                exit_time_str = exit_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                csv_writer.writerow([custom_id, class_name, entry_time_str, exit_time_str])
                
            # Add to logged objects set to avoid duplicate logging
            self.logged_objects.add(track_id)
            print(f"Logged object with custom ID {custom_id} ({class_name}) to CSV")
        except Exception as e:
            print(f"Error logging to CSV: {str(e)}")
            
    def check_for_completed_tracks(self):
        """Check for objects that have both entry and exit times and log them"""
        # Find objects that have both entered and exited the zone
        completed_objects = []
        
        for track_id, (entry_time, class_name) in self.object_entry_times.items():
            if track_id in self.object_exit_times and track_id not in self.logged_objects:
                exit_time = self.object_exit_times[track_id]
                # Only log if the object was in the zone for a reasonable time (to filter false positives)
                time_in_zone = (exit_time - entry_time).total_seconds()
                if time_in_zone > 0.1:  # At least 100ms in zone
                    self.log_object_to_csv(track_id, class_name, entry_time, exit_time)
                    completed_objects.append(track_id)

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
            # Stop any existing processing
            self.stop_detection()
            self.results_text.append(f"Loading video: {filename}...")
            
            # Open the video file
            self.cap = cv2.VideoCapture(filename)
            
            # Check if the video was opened successfully
            if not self.cap.isOpened():
                self.results_text.append(f"ERROR: Could not open video file: {filename}")
                self.cap = None
                return
                
            # Try to read the first frame to verify the video is readable
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.results_text.append(f"ERROR: Could not read frames from video file: {filename}")
                self.cap.release()
                self.cap = None
                return
                
            # Reset position to the beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            self.is_rtsp = False  # Reset RTSP flag for video files
            self.results_text.append(f"Successfully loaded video: {filename}")
    
    def connect_rtsp(self):
        rtsp_url = self.rtsp_input.toPlainText().strip()
        if not rtsp_url:
            self.results_text.append("Please enter an RTSP URL!")
            return
        
        # Try to connect to the RTSP stream with optimized settings
        self.results_text.append(f"Connecting to RTSP stream: {rtsp_url}")
        
        # Use FFMPEG backend which handles RTSP better
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # RTSP-specific optimizations
        # Set minimal buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set RTSP transport to TCP (more reliable than UDP)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        # Set flag to indicate this is an RTSP stream
        self.is_rtsp = True
        
        # Check if connection was successful
        if not self.cap.isOpened():
            self.results_text.append("Failed to connect to RTSP stream. Please check the URL.")
            self.is_rtsp = False
            return
            
        self.results_text.append("Successfully connected to RTSP stream!")
        self.results_text.append("RTSP optimizations applied: Using worker thread and minimal buffering")
            
    def start_detection(self):
        if self.cap is None:
            self.results_text.append("Please load a video or connect to an RTSP stream first!")
            return
        
        if not self.cap.isOpened():
            self.results_text.append("Error: Video source is not open or accessible!")
            return
            
        # Check if we can read frames
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.results_text.append("Error: Cannot read frames from video source!")
            return
            
        # Reset position to the beginning of the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Stop any existing worker
        self.stop_detection()
        
        # Clear previous results
        self.latest_frame = None
        self.latest_detections = []
        
        try:
            # Get video properties
            if not self.is_rtsp:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = frame_count / fps if fps > 0 else 0
                minutes = int(duration_sec / 60)
                seconds = int(duration_sec % 60)
                
                self.results_text.append(f"Video info: {fps:.2f} FPS, {frame_count} frames")
                self.results_text.append(f"Duration: {minutes}m {seconds}s")
                self.results_text.append(f"Playback will match original speed ({fps:.2f} FPS)")
            
            # Create and start worker thread
            self.results_text.append("Initializing detection...")
            self.worker = ProcessingWorker(self.cap, self.yolov8_model, self.custom_model, 
                                         self.config, is_rtsp=self.is_rtsp)
            self.worker.result_ready.connect(self.process_result)
            self.worker.start()
            
            # Start the UI update timer
            # For both RTSP and regular videos, update the display at the original video frame rate
            if self.is_rtsp:
                self.timer.start(15)  # Update every 15ms for smoother RTSP display
            else:
                # For regular videos, get the timer interval based on the video's frame rate
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 120:
                    fps = 30.0  # Default to 30 FPS if invalid
                # Calculate milliseconds per frame for the timer
                timer_interval = int(1000 / fps)
                self.timer.start(timer_interval)
                self.results_text.append(f"Display timer set to {timer_interval}ms per frame")
                
            self.results_text.append("Detection started in worker thread...")
            
        except Exception as e:
            self.results_text.append(f"Error starting detection: {str(e)}")
            if self.worker:
                self.worker.stop()
                self.worker = None

    def stop_detection(self):
        # Stop timer
        if self.timer.isActive():
            self.timer.stop()
        
        # Stop worker thread if it exists
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            
        self.results_text.append("Detection stopped.")
        
    def process_result(self, result):
        """Handle results from the worker thread"""
        if 'frame' in result:
            self.latest_frame = result['frame']
        if 'detections' in result:
            self.latest_detections = result['detections']
        if 'processing_time' in result:
            # Calculate processing FPS
            self.processing_fps = 1.0 / max(result['processing_time'], 0.001)  # Avoid division by zero

    def update_frame(self):
        """Update the UI with the latest processed frame - called by the timer"""
        # If no frame is available yet or worker is not running, return
        if self.latest_frame is None and (self.worker is None or not self.worker.running):
            return
            
        # If we haven't received a processed frame yet but worker is running,
        # we might be waiting for first result - just return
        if self.latest_frame is None:
            return
            
        # Use the latest frame from the worker thread
        frame = self.latest_frame.copy()
        
        # Calculate display FPS
        current_time = time.time()
        self.display_fps = 1.0 / max(current_time - self.last_frame_time, 0.001)  # Avoid division by zero
        self.last_frame_time = current_time
        
        # Use the latest detections from the worker thread
        final_dets = self.latest_detections

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

            # Get or create custom ID for this track
            custom_id = self.get_custom_id(track_id, cls_name)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), color, 1)
            # Display the custom ID instead of the track ID
            cv2.putText(frame, f"{cls_name} ID:{custom_id}", (x1o, y1o-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Get current position of the object
            current_position = (center_x, center_y)
            
            # For polygon-based counting
            # Scale the counting zone to match the original frame size
            scaled_polygon = []
            for point in self.counting_zone:
                x = int(point[0] * width_scale_factor)
                y = int(point[1] * height_scale_factor)
                scaled_polygon.append((x, y))
            
            # Check if the object is inside the polygon
            is_in_zone = self.point_in_polygon(current_position, scaled_polygon)
            
            # Get or ensure we have a custom ID for this track
            custom_id = self.get_custom_id(track_id, cls_name)
            
            # Check if the object is entering or leaving the zone
            if is_in_zone:
                if track_id not in self.object_in_zone:
                    # Object is entering the zone
                    self.object_in_zone.add(track_id)
                    current_time = datetime.now()
                    # Store entry time and class
                    self.object_entry_times[track_id] = (current_time, cls_name)
                    
                    # Only count if not already counted
                    if track_id not in self.counted_ids:
                        self.unique_vehicle_counts[cls_name] += 1
                        self.counted_ids.add(track_id)
                        # Highlight the object when counted
                        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 0, 255), 2)
                        
                        # Display the entry time and custom ID
                        time_str = current_time.strftime('%H:%M:%S')
                        short_id = custom_id.split('-')[-1]  # Just show the numerical part for cleaner display
                        cv2.putText(frame, f"IN: {time_str} ({short_id})", (x1o, y1o - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Object left the zone
                if track_id in self.object_in_zone:
                    self.object_in_zone.remove(track_id)
                    current_time = datetime.now()
                    # Store exit time
                    self.object_exit_times[track_id] = current_time
                    
                    # Display the exit time and custom ID
                    if track_id in self.object_entry_times:
                        time_str = current_time.strftime('%H:%M:%S')
                        short_id = custom_id.split('-')[-1]  # Just show the numerical part
                        cv2.putText(frame, f"OUT: {time_str} ({short_id})", (x1o, y1o - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
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
            
        # Add FPS and processing info
        cv2.putText(frame, f"Display FPS: {self.display_fps:.1f}", (10, frame.shape[0] - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Processing FPS: {self.processing_fps:.1f}", (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display original video frame rate
        if hasattr(self.worker, 'original_fps'):
            cv2.putText(frame, f"Original Video FPS: {self.worker.original_fps:.1f}", 
                       (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   
        # Add playback mode status
        if self.is_rtsp:
            cv2.putText(frame, "RTSP Mode: Optimized", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Playback Mode: Real-time speed", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # We've replaced the ROI polygon with a counting line in the middle of the screen

        # Check for objects that have completed their tracking cycle and log them
        self.check_for_completed_tracks()
        
        # Display CSV logging status
        if hasattr(self, 'csv_filename'):
            filename = os.path.basename(self.csv_filename)
            cv2.putText(frame, f"Logging to: {filename}", (10, 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Display custom ID tracking info
        if hasattr(self, 'custom_id_counter') and self.custom_id_counter:
            total_ids = sum(self.custom_id_counter.values())
            cv2.putText(frame, f"Tracking with custom IDs: {total_ids} objects", (10, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # No need to resize again since we're already at our target resolution (640x360)
        # Just convert to RGB for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    # Removed line_crossed method as we're only using polygon-based counting
    
    def launch_coordinate_selector(self):
        """Launch the get_coordinates.py script as a separate process"""
        try:
            # Create a notification that the app has been launched
            self.results_text.append("Launching Coordinate Selector tool...")
            
            # Use subprocess to run the get_coordinates.py script
            import subprocess
            
            # Run the script using the current Python interpreter
            python_executable = sys.executable
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_coordinates.py")
            
            # Run the process detached from this one
            if os.name == 'nt':  # Windows
                subprocess.Popen([python_executable, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Linux/Mac
                subprocess.Popen([python_executable, script_path], start_new_session=True)
            
            self.results_text.append("Coordinate Selector launched in a new window.")
            self.results_text.append("After saving coordinates, restart this application or click 'Reload Config' to apply changes.")
            
            # Add reload config button if it doesn't exist yet
            if not hasattr(self, 'reload_config_btn'):
                self.reload_config_btn = QPushButton("Reload Config")
                self.reload_config_btn.clicked.connect(self.reload_config)
                self.reload_config_btn.setStyleSheet("background-color: #2196F3; color: white;")
                # Find where the coords_btn is in the layout and insert after it
                layout = self.coords_btn.parent().layout()
                layout.insertWidget(layout.indexOf(self.coords_btn) + 1, self.reload_config_btn)
                
        except Exception as e:
            self.results_text.append(f"Error launching Coordinate Selector: {str(e)}")
    
    def reload_config(self):
        """Reload the configuration file to apply any changes made by the coordinate selector"""
        try:
            # Load the config file
            with open("config2.yaml", "r") as f:
                self.config = yaml.safe_load(f)
                
            # Update counting zone if it exists in the config
            if "counting_zone" in self.config:
                self.counting_zone = self.config["counting_zone"]
                self.results_text.append(f"Successfully updated counting zone with {len(self.counting_zone)} points")
            
            # Notify the user
            self.results_text.append("Configuration reloaded successfully!")
            
        except Exception as e:
            self.results_text.append(f"Error reloading configuration: {str(e)}")
    
    def closeEvent(self, event):
        # Final check for any objects that need to be logged
        current_time = datetime.now()
        
        # For objects that entered but never exited, log them with current time as exit time
        for track_id, (entry_time, class_name) in list(self.object_entry_times.items()):
            if track_id not in self.object_exit_times and track_id not in self.logged_objects:
                self.object_exit_times[track_id] = current_time
                self.log_object_to_csv(track_id, class_name, entry_time, current_time)
        
        # Check for any remaining completed tracks
        self.check_for_completed_tracks()
        
        # Log the total counts to the results
        self.results_text.append("\nFinal vehicle counts:")
        for cls_name, count in self.unique_vehicle_counts.items():
            self.results_text.append(f"{cls_name}: {count}")
        self.results_text.append(f"\nDetection data logged to {self.csv_filename}")
        
        # Save final counter state
        self.save_counter_state()
        self.results_text.append(f"Custom ID counters saved")
        
        # Stop detection and worker thread
        self.stop_detection()
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VehicleDetectionApp()
    window.show()
    sys.exit(app.exec_())
