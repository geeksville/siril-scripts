# (c) Cyril Richard from Franklin Marek SAS code (2025)
# NBtoRGBstars for Siril - Ported from tkinter to PyQt6
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 2.0.0

import sirilpy as s
s.ensure_installed("PyQt6", "pillow", "numpy", "astropy")

import os
import sys
import numpy as np
from PIL import Image
from astropy.io import fits

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton,
                            QGroupBox, QSplitter, QScrollArea, QFrame, QFileDialog,
                            QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter

VERSION = "2.0.0"
# 1.0.1 CR: using tkfilebrowser for linux OS
# 1.0.2 CR: fixing script due to API changes
# 1.0.3 CM: remove unnecessary import
# 1.0.4 CR: fixing flipping issue
# 2.0.0 CR: using PyQt6 instead of tkinter and adding a Clear images button

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.parent_interface = None

    def set_parent_interface(self, parent_interface):
        self.parent_interface = parent_interface

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_display()

    def update_display(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                int(self.original_pixmap.width() * self.scale_factor),
                int(self.original_pixmap.height() * self.scale_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)

    def set_scale_factor(self, factor):
        self.scale_factor = factor
        self.update_display()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.parent_interface and self.original_pixmap:
            # Get wheel delta (positive = zoom in, negative = zoom out)
            delta = event.angleDelta().y()
            
            if delta > 0:
                self.parent_interface.zoom_in()
            elif delta < 0:
                self.parent_interface.zoom_out()
        
        super().wheelEvent(event)

class NBtoRGBstarsInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"NB to RGB Stars - v{VERSION}")
        self.setGeometry(100, 100, 1200, 800)
        
        self.siril = s.SirilInterface()
        
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            self.show_error_message("Failed to connect to Siril")
            self.close()
            return
            
        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.show_error_message("Siril version requirement not met")
            self.close()
            return
        
        # Initialize image variables
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        self.osc_image = None
        self.combined_image = None
        self.is_mono = False
        
        # Filenames
        self.ha_filename = None
        self.oiii_filename = None
        self.sii_filename = None
        self.osc_filename = None
        
        self.original_header = None
        self.original_header_string = None
        self.bit_depth = "Unknown"
        
        # Set up zoom
        self.zoom_factor = 1.0
        
        # Create the UI
        self.init_ui()
    
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel for image preview
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
    
    def create_left_panel(self):
        left_widget = QWidget()
        left_widget.setFixedWidth(400)
        layout = QVBoxLayout(left_widget)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("NB to RGB Stars")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        
        instructions_text = QLabel("""1. Select Ha, OIII, and SII (optional) narrowband images, or an OSC stars-only image.
   Note: Images must be pre-aligned on stars before processing.
2. Adjust the Ha to OIII Ratio if needed.
3. Preview the combined result.
4. Send Preview to Siril.""")
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        layout.addWidget(instructions_group)
        
        # Image selection
        image_group = QGroupBox("Image Selection")
        image_layout = QVBoxLayout(image_group)
        
        # Ha Image
        ha_layout = QHBoxLayout()
        self.ha_button = QPushButton("Ha Image")
        self.ha_button.clicked.connect(lambda: self.load_image('Ha'))
        self.ha_label = QLabel("No Ha image selected")
        ha_layout.addWidget(self.ha_button)
        ha_layout.addWidget(self.ha_label)
        image_layout.addLayout(ha_layout)
        
        # OIII Image
        oiii_layout = QHBoxLayout()
        self.oiii_button = QPushButton("OIII Image")
        self.oiii_button.clicked.connect(lambda: self.load_image('OIII'))
        self.oiii_label = QLabel("No OIII image selected")
        oiii_layout.addWidget(self.oiii_button)
        oiii_layout.addWidget(self.oiii_label)
        image_layout.addLayout(oiii_layout)
        
        # SII Image
        sii_layout = QHBoxLayout()
        self.sii_button = QPushButton("SII Image (Optional)")
        self.sii_button.clicked.connect(lambda: self.load_image('SII'))
        self.sii_label = QLabel("No SII image selected")
        sii_layout.addWidget(self.sii_button)
        sii_layout.addWidget(self.sii_label)
        image_layout.addLayout(sii_layout)
        
        # OSC Image
        osc_layout = QHBoxLayout()
        self.osc_button = QPushButton("OSC Stars Image (Optional)")
        self.osc_button.clicked.connect(lambda: self.load_image('OSC'))
        self.osc_label = QLabel("No OSC image selected")
        osc_layout.addWidget(self.osc_button)
        osc_layout.addWidget(self.osc_label)
        image_layout.addLayout(osc_layout)
        
        # Clear All button
        clear_layout = QHBoxLayout()
        self.clear_all_button = QPushButton("Clear All Images")
        self.clear_all_button.clicked.connect(self.clear_all_images)
        self.clear_all_button.setStyleSheet("QPushButton { background-color: #d32f2f; color: white; }")
        clear_layout.addWidget(self.clear_all_button)
        clear_layout.addStretch()
        image_layout.addLayout(clear_layout)
        
        layout.addWidget(image_group)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Ha to OIII Ratio
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Ha to OIII Ratio:"))
        
        self.ha_to_oiii_slider = QSlider(Qt.Orientation.Horizontal)
        self.ha_to_oiii_slider.setMinimum(0)
        self.ha_to_oiii_slider.setMaximum(100)
        self.ha_to_oiii_slider.setValue(30)
        self.ha_to_oiii_slider.valueChanged.connect(self.update_ratio_display)
        
        self.ha_to_oiii_label = QLabel("0.30")
        self.ha_to_oiii_label.setFixedWidth(40)
        
        ratio_layout.addWidget(self.ha_to_oiii_slider)
        ratio_layout.addWidget(self.ha_to_oiii_label)
        params_layout.addLayout(ratio_layout)
        
        # Star Stretch
        self.enable_star_stretch = QCheckBox("Enable Star Stretch")
        self.enable_star_stretch.setChecked(True)
        self.enable_star_stretch.toggled.connect(self.toggle_stretch_controls)
        params_layout.addWidget(self.enable_star_stretch)
        
        # Stretch Factor
        self.stretch_layout = QHBoxLayout()
        self.stretch_layout.addWidget(QLabel("Stretch Factor:"))
        
        self.stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.stretch_slider.setMinimum(0)
        self.stretch_slider.setMaximum(80)
        self.stretch_slider.setValue(50)
        self.stretch_slider.valueChanged.connect(self.update_stretch_display)
        
        self.stretch_label = QLabel("5.0")
        self.stretch_label.setFixedWidth(40)
        
        self.stretch_layout.addWidget(self.stretch_slider)
        self.stretch_layout.addWidget(self.stretch_label)
        params_layout.addLayout(self.stretch_layout)
        
        # Metadata
        self.copy_metadata = QCheckBox("Copy Metadata from Source Image")
        self.copy_metadata.setChecked(True)
        params_layout.addWidget(self.copy_metadata)
        
        layout.addWidget(params_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Preview Combined Image")
        self.preview_button.clicked.connect(self.preview_combine)
        buttons_layout.addWidget(self.preview_button)
        
        self.send_to_siril_button = QPushButton("Send Preview to Siril")
        self.send_to_siril_button.clicked.connect(self.send_to_siril_preview)
        buttons_layout.addWidget(self.send_to_siril_button)
        
        layout.addLayout(buttons_layout)
        
        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Footer
        footer_label = QLabel("Written by Franklin Marek\nSiril port by Cyril Richard\nwww.setiastro.com")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer_label)
        
        layout.addStretch()
        
        return left_widget
    
    def create_right_panel(self):
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("Fit to Preview")
        fit_btn.clicked.connect(self.fit_to_preview)
        zoom_layout.addWidget(fit_btn)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_label = ImageLabel()
        self.image_label.set_parent_interface(self)
        self.image_label.setText("No preview available")
        scroll_area.setWidget(self.image_label)
        
        layout.addWidget(scroll_area)
        
        return right_widget
    
    def update_ratio_display(self):
        value = self.ha_to_oiii_slider.value() / 100.0
        self.ha_to_oiii_label.setText(f"{value:.2f}")
    
    def update_stretch_display(self):
        value = self.stretch_slider.value() / 10.0
        self.stretch_label.setText(f"{value:.1f}")
    
    def toggle_stretch_controls(self):
        enabled = self.enable_star_stretch.isChecked()
        self.stretch_slider.setEnabled(enabled)
        self.stretch_label.setEnabled(enabled)
    
    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()
    
    def show_warning_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText(message)
        msg_box.exec()
    
    def load_image(self, image_type):
        """Load a FITS image from file using Astropy"""
        try:
            # Get the current working directory from Siril
            current_wd = self.siril.get_siril_wd() or os.path.expanduser("~")
            
            # Open file dialog
            filename, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {image_type} FITS Image File",
                current_wd,
                "FITS files (*.fits *.fit *.fts)"
            )
     
            if not filename:
                return  # User canceled
            
            # Open the FITS file with Astropy
            with fits.open(filename) as image:
                # Get the image data
                image_data = image[0].data
                
                # Get the header
                header = image[0].header
                
                # Convert header to string for later use with metadata copying
                header_string = header.tostring(sep='\n')

                # Normalize data type to float32 if needed
                if image_data.dtype not in (np.uint16, np.float32):
                    image_data = image_data.astype(np.float32)
                
                # Debug print to understand the data structure
                print(f"Image data shape: {image_data.shape}")
                print(f"Image data type: {image_data.dtype}")

                # Ensure the data is in a 2D or 3D format
                if image_data.ndim == 2:
                    # Mono image: add channel dimension
                    image_data = image_data[np.newaxis, :, :]
                elif image_data.ndim == 3:
                    # Check if channels are first or last
                    if image_data.shape[0] in [1, 3]:
                        # Channels first - keep as is
                        pass
                    elif image_data.shape[2] in [1, 3]:
                        # Channels last - transpose to channels first
                        image_data = np.transpose(image_data, (2, 0, 1))
                
                if image_data.shape[0] in [1, 3]:
                    image_data = np.transpose(image_data, (1, 2, 0))
                
                # Determine if it's mono
                is_mono = image_data.ndim == 2 or (image_data.ndim == 3 and image_data.shape[2] == 1)
                
                # If mono, ensure 2D
                if is_mono:
                    image_data = image_data.squeeze()
            
            # Store the image data in the appropriate variable
            if image_type == 'Ha':
                self.ha_image = image_data
                self.ha_filename = filename
                self.ha_label.setText(f"Loaded: {os.path.basename(filename)}")
                
                # Store metadata from Ha image
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono
            
            elif image_type == 'OIII':
                self.oiii_image = image_data
                self.oiii_filename = filename
                self.oiii_label.setText(f"Loaded: {os.path.basename(filename)}")

                # If Ha not loaded yet, use OIII metadata as source
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono

            elif image_type == 'SII':
                self.sii_image = image_data
                self.sii_filename = filename
                self.sii_label.setText(f"Loaded: {os.path.basename(filename)}")

                # If no metadata source yet, use SII
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono

            elif image_type == 'OSC':
                self.osc_image = image_data
                self.osc_filename = filename
                self.osc_label.setText(f"Loaded: {os.path.basename(filename)}")

                # If no metadata source yet, use OSC
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono
            
            self.status_label.setText(f"{image_type} FITS image loaded successfully")
        
        except Exception as e:
            self.show_error_message(f"Failed to load {image_type} FITS image: {str(e)}")
            print(f"Error loading {image_type} FITS image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def clear_all_images(self):
        """Clear all loaded images and reset the interface"""
        # Clear all image data
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        self.osc_image = None
        self.combined_image = None
        
        # Clear filenames
        self.ha_filename = None
        self.oiii_filename = None
        self.sii_filename = None
        self.osc_filename = None
        
        # Clear metadata
        self.original_header = None
        self.original_header_string = None
        self.bit_depth = "Unknown"
        self.is_mono = False
        
        # Reset labels
        self.ha_label.setText("No Ha image selected")
        self.oiii_label.setText("No OIII image selected")
        self.sii_label.setText("No SII image selected")
        self.osc_label.setText("No OSC image selected")
        
        # Clear preview
        self.image_label.clear()
        self.image_label.setText("No preview available")
        
        # Reset zoom
        self.zoom_factor = 1.0
        
        # Update status
        self.status_label.setText("All images cleared")
    
    def preview_combine(self):
        """Generate a preview of the combined image"""
        # Check if required images are loaded
        if not ((self.ha_image is not None and self.oiii_image is not None) or (self.osc_image is not None)):
            self.show_warning_message("Please load Ha and OIII images, or an OSC image")
            return
        
        # Update status
        self.status_label.setText("Processing image... Please wait.")
        QApplication.processEvents()
        
        try:
            # Get parameters
            ha_to_oiii_ratio = self.ha_to_oiii_slider.value() / 100.0
            enable_star_stretch = self.enable_star_stretch.isChecked()
            stretch_factor = self.stretch_slider.value() / 10.0
            
            # Process the image
            combined_image = self.process_image(
                self.ha_image, 
                self.oiii_image, 
                self.sii_image, 
                self.osc_image, 
                ha_to_oiii_ratio,
                enable_star_stretch,
                stretch_factor
            )
            
            # Store the result
            self.combined_image = combined_image
            
            # Update preview
            self.update_preview(combined_image)
            
            # Update status
            self.status_label.setText("Preview generated successfully")
        
        except Exception as e:
            self.show_error_message(f"Error processing image: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def process_image(self, ha_image, oiii_image, sii_image, osc_image, ha_to_oiii_ratio, enable_star_stretch, stretch_factor):
        """Process the images to create a combined RGB image"""
        # Function to preprocess and ensure narrowband images are properly formatted
        def preprocess_narrowband(img):
            if img is None:
                return None
                
            if img.dtype in (np.uint16, np.int16):
                img_normalized = img.astype(np.float32) / 65535.0
                print(f"Normalized 16-bit image to float32 (0-1) range")
            else:
                img_normalized = img.astype(np.float32)
                img_normalized = np.clip(img_normalized, 0, 1)

            if isinstance(img_normalized, np.ndarray) and img_normalized.ndim == 3 and img_normalized.shape[2] == 3:
                # Convert to grayscale using luminance formula
                return 0.299 * img_normalized[..., 0] + 0.587 * img_normalized[..., 1] + 0.114 * img_normalized[..., 2]
            
            return img_normalized
        
        # Preprocess images
        ha_processed = preprocess_narrowband(ha_image)
        oiii_processed = preprocess_narrowband(oiii_image)
        sii_processed = preprocess_narrowband(sii_image)
        
        if osc_image is not None:
            if osc_image.dtype in (np.uint16, np.int16):
                osc_processed = osc_image.astype(np.float32) / 65535.0
                print(f"Normalized 16-bit OSC image to float32 (0-1) range")
            else:
                osc_processed = osc_image.astype(np.float32)
                osc_processed = np.clip(osc_processed, 0, 1)

            # Use OSC image as base, enhance with narrowband data if available
            r_channel = osc_processed[..., 0]
            g_channel = osc_processed[..., 1]
            b_channel = osc_processed[..., 2]
            
            # Enhance with narrowband if available
            r_combined = 0.5 * r_channel + 0.5 * (sii_processed if sii_processed is not None else r_channel)
            g_combined = ha_to_oiii_ratio * (ha_processed if ha_processed is not None else g_channel) + \
                        (1 - ha_to_oiii_ratio) * g_channel
            b_combined = oiii_processed if oiii_processed is not None else b_channel
        else:
            # Using narrowband images only
            r_combined = 0.5 * ha_processed + 0.5 * (sii_processed if sii_processed is not None else ha_processed)
            g_combined = ha_to_oiii_ratio * ha_processed + (1 - ha_to_oiii_ratio) * oiii_processed
            b_combined = oiii_processed
        
        # Normalize combined channels
        r_combined = np.clip(r_combined, 0, 1)
        g_combined = np.clip(g_combined, 0, 1)
        b_combined = np.clip(b_combined, 0, 1)
        
        # Stack channels to create RGB image
        combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)
        
        # Apply star stretch if enabled
        if enable_star_stretch:
            combined_image = self.apply_star_stretch(combined_image, stretch_factor)
        
        # Apply SCNR (remove green cast)
        combined_image = self.apply_scnr(combined_image)

        return combined_image
    
    def apply_star_stretch(self, image, stretch_factor):
        """Apply non-linear stretch to enhance stars"""
        # Ensure input is in [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Apply the formula: (a^b * x) / ((a^b - 1) * x + 1)
        # where a=3, b=stretch_factor
        a = 3.0
        b = stretch_factor
        stretched = ((a ** b) * image) / (((a ** b) - 1) * image + 1)
        
        return np.clip(stretched, 0, 1)
    
    def apply_scnr(self, image):
        """Apply SCNR (Subtractive Chromatic Noise Reduction) to remove green cast"""
        # Extract channels
        r_channel = image[..., 0]
        g_channel = image[..., 1]
        b_channel = image[..., 2]
        
        # Apply average-neutral SCNR
        max_rb = np.maximum(r_channel, b_channel)
        mask = g_channel > max_rb
        g_channel[mask] = max_rb[mask]
        
        # Update green channel in the image
        image[..., 1] = g_channel
        
        return image
    
    def update_preview(self, image):
        """Update the preview display with the processed image"""
        if image is None:
            return
        
        # Flip the image for display purposes only
        display_image = np.flipud(image)
            
        # Convert to 8-bit for display
        preview_image = (display_image * 255).astype(np.uint8)
        
        # Create QImage
        height, width, channels = preview_image.shape
        bytes_per_line = channels * width
        q_image = QImage(preview_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # Set the image in the label
        self.image_label.set_image(pixmap)
        
        # Store original dimensions for zoom calculations
        self.original_width = width
        self.original_height = height
    
    def zoom_in(self):
        """Increase zoom level"""
        if self.zoom_factor < 20.0:
            self.zoom_factor *= 1.25
            self.image_label.set_scale_factor(self.zoom_factor)
    
    def zoom_out(self):
        """Decrease zoom level"""
        if self.zoom_factor > 0.1:
            self.zoom_factor /= 1.25
            self.image_label.set_scale_factor(self.zoom_factor)
    
    def fit_to_preview(self):
        """Fit image to preview window"""
        if self.combined_image is None or not hasattr(self, 'original_width'):
            return
            
        # Get available space
        available_width = self.image_label.parent().width() - 20
        available_height = self.image_label.parent().height() - 20
        
        # Calculate zoom factor to fit
        width_ratio = available_width / self.original_width
        height_ratio = available_height / self.original_height
        
        # Use the smaller ratio to ensure image fits completely
        self.zoom_factor = min(width_ratio, height_ratio, 1.0)
        
        # Update preview
        self.image_label.set_scale_factor(self.zoom_factor)
    
    def send_to_siril_preview(self):
        """Send the combined image to Siril's preview window directly"""
        if self.combined_image is None:
            self.show_warning_message("No combined image to send. Please generate a preview first.")
            return
        
        try:
            # Get image dimensions
            height, width = self.combined_image.shape[:2]

            # Ensure the image is in the correct format
            combined_data = self.combined_image
            if combined_data.dtype in (np.uint16, np.int16):
                combined_data = (combined_data / 65535.0).astype(np.float32)
            else:
                combined_data = combined_data.astype(np.float32)
            print("Converting image to 32-bit float32 for Siril")

            print(f"Output image data type: {combined_data.dtype}")
            
            # Create an empty image with the correct dimensions
            self.siril.cmd("new", f"{width}", f"{height}", "3", "RGB")

            # Transpose the image back to planar format (channels, height, width)
            siril_image_data = np.transpose(combined_data, (2, 0, 1))
            siril_image_data = np.ascontiguousarray(siril_image_data)

            # Get the processing thread
            with self.siril.image_lock():
                try:
                    # Set the pixel data directly 
                    self.siril.set_image_pixeldata(siril_image_data)
                    
                    # Apply metadata if enabled and available
                    if self.copy_metadata.isChecked() and self.original_header_string is not None:
                        try:
                            self.siril.set_image_metadata_from_header_string(self.original_header_string)
                            self.status_label.setText("Image with metadata sent to Siril preview")
                            self.siril.log("Metadata copied from source image")
                        except Exception as metadata_err:
                            self.status_label.setText("Image sent, but metadata copy failed")
                            print(f"Metadata copy error: {str(metadata_err)}")
                    else:
                        self.status_label.setText("Image sent to Siril preview window")
                    
                    # log to Siril console
                    self.siril.log("NBtoRGB stars combined image loaded in Siril preview")
                
                except Exception as e:
                    print(f"Error in send_to_siril_preview: {str(e)}")

        except Exception as e:
            self.show_error_message(f"Failed to apply image to Siril: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    try:
        # Create Siril interface first to determine mode
        siril = s.SirilInterface()
        
        if siril.is_cli():
            print("CLI mode not supported for this script")
            return
        
        # GUI mode
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setApplicationName("NBtoRGBstars")
        window = NBtoRGBstarsInterface()
        window.show()
        sys.exit(app.exec())
            
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
