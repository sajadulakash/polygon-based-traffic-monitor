# GitHub Repository: https://github.com/sajadulakash/polygon-based-traffic-monitor
# 
# Coordinate Selector Tool
# =======================
# This utility allows users to define polygon counting zones for the traffic monitoring system.
# It supports both video files and RTSP streams, and provides an interactive interface for
# selecting coordinates and saving them to the configuration file.
#
# Author: sajadulakash
# License: MIT
# Version: 1.0

import cv2
import numpy as np
import sys
import os
import yaml
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QListWidget,
                           QFileDialog, QListWidgetItem, QGroupBox, QMessageBox,
                           QSplitter, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QSize

# Set your video path here - edit this to use a specific video file
VIDEO_PATH = "dumy.mp4"  # Change this to your desired video file

class CoordinateSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordinate Selector")
        self.setGeometry(100, 100, 1200, 700)
        
        # Initialize variables
        self.points = []
        self.frame = None
        self.processed_frame = None
        self.original_frame = None
        self.is_drawing = False
        self.video_path = None
        self.is_rtsp = False
        self.width = 640
        self.height = 360
        
        # Create the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create a splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Left panel for image display
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_layout.addWidget(self.image_label)
        
        # Buttons below image
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_btn)
        
        self.clear_btn = QPushButton("Clear Points")
        self.clear_btn.clicked.connect(self.clear_points)
        button_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Save Coordinates")
        self.save_btn.clicked.connect(self.save_coordinates)
        button_layout.addWidget(self.save_btn)
        
        self.image_layout.addLayout(button_layout)
        
        # RTSP section
        rtsp_layout = QHBoxLayout()
        self.rtsp_input = QTextEdit()
        self.rtsp_input.setPlaceholderText("Enter RTSP URL (e.g., rtsp://username:password@ip:port/stream)")
        self.rtsp_input.setMaximumHeight(50)
        rtsp_layout.addWidget(self.rtsp_input)
        
        self.rtsp_btn = QPushButton("Connect RTSP")
        self.rtsp_btn.clicked.connect(self.connect_rtsp)
        rtsp_layout.addWidget(self.rtsp_btn)
        
        self.image_layout.addLayout(rtsp_layout)
        
        # Right panel for displaying points
        self.points_panel = QWidget()
        self.points_layout = QVBoxLayout(self.points_panel)
        
        # Add title
        points_title = QLabel("Selected Points")
        points_title.setFont(QFont("Arial", 12, QFont.Bold))
        points_title.setAlignment(Qt.AlignCenter)
        self.points_layout.addWidget(points_title)
        
        # List to display points
        self.points_list = QListWidget()
        self.points_list.setFont(QFont("Courier New", 10))
        self.points_list.setAlternatingRowColors(True)
        self.points_layout.addWidget(self.points_list)
        
        # Instructions group box
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions = QLabel(
            "1. Load a video, connect to RTSP stream, or use the default\n"
            "2. Click on the image to add points\n"
            "3. Points form a polygon in sequence\n"
            "4. Minimum 3 points required\n"
            "5. Click 'Save Coordinates' when done\n"
            "6. Copy the coordinates to config2.yaml\n"
        )
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        self.points_layout.addWidget(instructions_group)
        
        # Add panels to splitter
        self.splitter.addWidget(self.image_panel)
        self.splitter.addWidget(self.points_panel)
        self.splitter.setSizes([700, 300])
        
        # Try to load the default video
        if VIDEO_PATH:
            self.video_path = VIDEO_PATH
            self.load_first_frame()
    def load_video(self):
        """Open file dialog to select a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if file_path:
            self.video_path = file_path
            self.is_rtsp = False
            self.load_first_frame()
            
    def connect_rtsp(self):
        """Connect to RTSP stream and get first frame"""
        rtsp_url = self.rtsp_input.toPlainText().strip()
        if not rtsp_url:
            QMessageBox.warning(self, "Warning", "Please enter an RTSP URL!")
            return
        
        # Set the video path to the RTSP URL
        self.video_path = rtsp_url
        self.is_rtsp = True
        
        QMessageBox.information(self, "RTSP Connection", 
                               f"Attempting to connect to RTSP stream:\n{rtsp_url}\n\nThis may take a few moments...")
        
        self.load_first_frame()
    
    def load_first_frame(self):
        """Load the first frame from the video file or RTSP stream"""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "No video path or RTSP URL selected")
            return
        
        # For RTSP streams, use FFMPEG backend for better compatibility
        if hasattr(self, 'is_rtsp') and self.is_rtsp:
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            
            # RTSP-specific optimizations
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set minimal buffer size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # Set RTSP transport to TCP
            
            # Give more time for RTSP connection
            connect_timeout = 10  # seconds
            start_time = time.time()
            connected = False
            
            # Try to connect with a timeout
            while time.time() - start_time < connect_timeout:
                if cap.isOpened():
                    connected = True
                    break
                time.sleep(0.5)  # Wait a bit before trying again
                
            if not connected:
                QMessageBox.critical(self, "Error", f"Could not connect to RTSP stream after {connect_timeout} seconds")
                return
                
            # Try to grab multiple frames to stabilize the connection
            for _ in range(5):
                cap.grab()
        else:
            # Regular video file
            cap = cv2.VideoCapture(self.video_path)
            
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video/stream: {self.video_path}")
            return
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            QMessageBox.critical(self, "Error", "Could not read frame from video or stream")
            return
            
        # Resize to our fixed 640x360 resolution
        self.frame = cv2.resize(frame, (self.width, self.height))
        self.original_frame = self.frame.copy()
        self.update_display_frame()
        
        # Set window title based on source type
        if hasattr(self, 'is_rtsp') and self.is_rtsp:
            self.setWindowTitle(f"Coordinate Selector - RTSP Stream")
        else:
            self.setWindowTitle(f"Coordinate Selector - {os.path.basename(self.video_path)}")
                
    def update_display_frame(self):
        """Update the image with grid, points, and polygon"""
        if self.frame is None:
            return
            
        # Create a copy to work with
        self.processed_frame = self.original_frame.copy()
        
        # Draw grid
        for x in range(0, self.width, 50):
            color = (100, 100, 100) if x % 100 != 0 else (150, 150, 150)
            cv2.line(self.processed_frame, (x, 0), (x, self.height), color, 1)
            if x % 100 == 0:
                cv2.putText(self.processed_frame, str(x), (x, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
        for y in range(0, self.height, 50):
            color = (100, 100, 100) if y % 100 != 0 else (150, 150, 150)
            cv2.line(self.processed_frame, (0, y), (self.width, y), color, 1)
            if y % 100 == 0:
                cv2.putText(self.processed_frame, str(y), (5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw frame border
        cv2.rectangle(self.processed_frame, (0, 0), (self.width-1, self.height-1),
                     (0, 0, 255), 2)
        cv2.putText(self.processed_frame, f"640x360", (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                   
        # Add source type indicator
        source_text = "RTSP Stream" if self.is_rtsp else "Video File"
        cv2.putText(self.processed_frame, source_text, (self.width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        # Draw polygon if we have points
        if len(self.points) > 0:
            # Draw all points
            for i, point in enumerate(self.points):
                # Draw point with number
                cv2.circle(self.processed_frame, point, 5, (0, 255, 0), -1)
                cv2.circle(self.processed_frame, point, 7, (255, 255, 255), 1)
                cv2.putText(self.processed_frame, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw lines between points
            for i in range(len(self.points) - 1):
                cv2.line(self.processed_frame, self.points[i], self.points[i+1],
                        (0, 255, 255), 2)
            
            # Close the polygon if we have 3+ points
            if len(self.points) >= 3:
                cv2.line(self.processed_frame, self.points[-1], self.points[0],
                        (0, 255, 255), 2)
                
                # Fill polygon with semi-transparent color
                overlay = self.processed_frame.copy()
                points_array = np.array(self.points, np.int32)
                cv2.fillPoly(overlay, [points_array], (0, 255, 0, 128))
                cv2.addWeighted(overlay, 0.3, self.processed_frame, 0.7, 0, self.processed_frame)
        
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Set the image
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(w, h)

    def mouse_press_event(self, event):
        """Handle mouse clicks on the image"""
        if self.frame is None:
            return
            
        # Get the position where user clicked
        x, y = event.x(), event.y()
        
        # Make sure click is within the image bounds
        if 0 <= x < self.width and 0 <= y < self.height:
            # Add the point
            self.points.append((x, y))
            
            # Update the display
            self.update_display_frame()
            
            # Update the points list
            self.update_points_list()
            
    def update_points_list(self):
        """Update the list of points in the sidebar"""
        self.points_list.clear()
        
        for i, point in enumerate(self.points):
            item = QListWidgetItem(f"Point {i+1}: ({point[0]}, {point[1]})")
            self.points_list.addItem(item)
            
    def clear_points(self):
        """Clear all selected points"""
        if not self.points:
            return
            
        self.points = []
        self.update_display_frame()
        self.update_points_list()
        
        QMessageBox.information(
            self,
            "Points Cleared",
            "All points have been cleared."
        )
        
    def save_coordinates(self):
        """Save the selected points to output"""
        if len(self.points) < 3:
            QMessageBox.warning(
                self,
                "Not Enough Points",
                "You need at least 3 points to form a valid polygon."
            )
            return
            
        # Print points to console (for legacy compatibility)
        print("\nSaved Points:")
        for i, point in enumerate(self.points):
            print(f"Point {i+1}: {point}")
        
        # Format for config.yaml as a counting zone polygon
        print("\nFor counting_zone in config2.yaml:")
        print("counting_zone:")
        print("  # Coordinates at fixed 640x360 resolution:")
        yaml_output = "counting_zone:\n"
        for point in self.points:
            print(f"  - [{point[0]}, {point[1]}]")
            yaml_output += f"  - [{point[0]}, {point[1]}]\n"
        
        # Also show as regular ROI polygon if needed
        print("\nFor roi_polygon in config2.yaml (if needed):")
        print("roi_polygon:")
        for point in self.points:
            print(f"  - [{point[0]}, {point[1]}]")
        
        # Show success message with copy instructions
        QMessageBox.information(
            self,
            "Coordinates Saved",
            f"Successfully saved {len(self.points)} points!\n\n"
            "The coordinates have been printed to the console.\n"
            "Copy them from there into your config2.yaml file."
        )
        
        # Also save directly to config file
        reply = QMessageBox.question(
            self,
            "Save to Config File?",
            "Do you want to directly update config2.yaml with these coordinates?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.save_to_config_file()
        
    def save_to_config_file(self):
        """Directly save points to config file"""
        try:
            # Try to load the existing config
            config_path = "config2.yaml"
            
            if os.path.exists(config_path):
                # Load existing config
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                    
                if config_data is None:
                    config_data = {}
            else:
                config_data = {}
                
            # Update with new counting zone
            config_data["counting_zone"] = [[p[0], p[1]] for p in self.points]
            
            # Remove counting_line if it exists (replacing with counting zone)
            if "counting_line" in config_data:
                del config_data["counting_line"]
                
            # Save back to file
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            QMessageBox.information(
                self, 
                "Config Updated",
                f"Successfully updated {config_path} with new counting zone."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Config",
                f"Failed to save to config file: {str(e)}"
            )
            

def main():
    app = QApplication(sys.argv)
    window = CoordinateSelector()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
