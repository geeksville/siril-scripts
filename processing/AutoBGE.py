# (c) Adrian Knagg-Baugh from Franklin Marek SAS code (2025)
# AutoBGE for Siril - Converted to PyQt6
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 2.0.0
# 1.0.0 Initial release
# 1.0.1 Clear rectangular selection after setting exclusion area
# 1.0.2 Mono images remain mono after processing
# 1.0.3 Fix copypasta error that meant RGB background couldn't be
#       shown with the "Show Gradient Removed" button.
# 1.0.4 Fix CLI mode so the script can be used with pyscript
# 2.0.0 Converted to PyQt6

"""
Auto Background Extraction script for Siril
===========================================

This script ports the SetiAstro AutoDBE script to interface
with Siril and uses some features of the sirilpy module
for setting exclusion areas and showing the optimized sample
points used for each stage of the processing.

Use
---
To use the script, select the polynomial degree and RBF
smoothness you want to use. The defaults work well in many cases.

Exclusion areas are useful to prevent placement of sample points in
genuinely darker-than-average areas such as dark nebulae.
If you wish to set exclusion areas, make selections in Siril and
click "Add Exclusion Area" for each. You may add as many as you like.

Click "Apply" to apply the background extraction.

Once the extraction is done you can toggle between the corrected
image and the computed background.

Command-Line Use
----------------
To use the script with pyscript, call it as:

pyscript AutoBGE.py [-npoints] [-polydegree] [-rbfsmooth]

Exclusion areas are not supported via the pyscript interface as
they generally need to be set visually, however if you have a
programmatic way of generating exclusion areas you can do so
using SirilInterface.overlay_add_polygon() before calling this
script with pyscript.
"""

import sys
import argparse
import sirilpy as s
s.ensure_installed("opencv-python", "scipy", "PyQt6")
import numpy as np
import math
import cv2
from scipy.interpolate import Rbf

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QSpinBox, QDoubleSpinBox, QSlider,
                            QCheckBox, QPushButton, QGroupBox, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

VERSION = "2.0.0"

if not s.check_module_version(">=0.7.41"):
    print("Error: requires sirilpy version 0.7.41 or higher")
    sys.exit(1)

class ProcessingThread(QThread):
    """Thread for processing the image to avoid blocking the UI"""
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def run(self):
        try:
            self.processor.process_image_threaded(self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class GradientRemovalInterface:
    def __init__(self, siril: s.SirilInterface, app=None, cli_args=None):
        self.siril = siril
        self.app = app

        self.cli_call = self.siril.is_cli()
        # If no CLI args, create a default namespace with defaults
        if self.cli_call and cli_args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-npoints", type=int, default=100)
            parser.add_argument("-polydegree", type=int, default=2)
            parser.add_argument("-rbfsmooth", type=float, default=0.1)
            cli_args = parser.parse_args([])

        self.cli_args = cli_args

        # Ensure target_median is between 0.01 and 0.99
        if hasattr(cli_args, 'median'):
            cli_args.median = max(0.01, min(0.99, cli_args.median))

        if not self.siril.connected:
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                if self.cli_call:
                    QMessageBox.critical(None, "Error", "Failed to connect to Siril")
                    self.close_dialog()
                else:
                    print("Error: failed to connect to Siril", file=sys.stderr)
                return

        try:
            self.siril.cmd("requires", "1.4.0-beta3")
        except s.CommandError:
            if not self.cli_call:
                QMessageBox.critical(None, "Error", "Siril version requirement not met")
                self.close_dialog()
            else:
                print("Error: Siril version requirements not met", file=sys.stderr)
            return

        # Initialize parameters with default values
        self.num_sample_points = 100
        self.poly_degree = 2
        self.rbf_smooth = 0.1
        self.show_gradient = False

        # Polygons
        self.siril.overlay_clear_polygons()
        self.exclusion_polygons = []

        # Downsample scale factor
        self.downsample_scale = 4

        # Image
        self.originally_16bit = False
        self.image = self.siril.get_image_pixeldata()
        if self.image is None:
            if not self.cli_call:
                QMessageBox.critical(None, "Error", "Could not get image")
                self.close_dialog()
            else:
                print("Error: could not get image", file=sys.stderr)
            return
        self.corrected_image = None
        self.gradient_background = None

        # ADBE uses CV so we do everything in HWC format
        if len(self.image.shape) == 3:
            self.image = self.image.transpose(1, 2, 0)
        self.originally_mono = len(self.image.shape) == 2
        self.originally_16bit = self.image.dtype == np.uint16
        if self.originally_16bit:
            self.image = self.image.astype(np.float32) / 65535

        # Processing thread
        self.processing_thread = None

        if self.cli_call:
            # Configure parameters
            self.num_sample_points = self.cli_args.npoints
            self.poly_degree = self.cli_args.polydegree
            self.rbf_smooth = self.cli_args.rbfsmooth
            # Process the image
            self.siril.log(f"AutoBGE: processing image with {self.num_sample_points} "
                           f"sample points, polynomial degree {self.poly_degree} and "
                           f"RBF smoothness {self.rbf_smooth}")
            self.process_image()
        elif app:
            # Store app for GUI mode
            self.create_widgets()
        else:
            print("Error: not called from the CLI and no app created")

    def close_dialog(self):
        """Close the dialog and disconnect from Siril"""
        try:
            self.siril.overlay_clear_polygons()
            self.siril.disconnect()
        except Exception:
            pass
        if hasattr(self, 'window'):
            self.window.close()

    def floor_value(self, value, decimals=2):
        """Floor a value to the specified number of decimal places"""
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def update_rbf_smooth_display(self):
        """Update the displayed RBF smooth value"""
        value = self.rbf_smooth_slider.value() / 100.0  # Convert from 0-1000 to 0-10.0
        self.rbf_smooth = value
        rounded_value = self.floor_value(value, 2)
        self.rbf_smooth_label.setText(f"{rounded_value:.2f}")

    def create_widgets(self):
        # Create main window
        self.window = QMainWindow()
        self.window.setWindowTitle(f"Automatic Background Extraction - v{VERSION}")
        self.window.setFixedSize(600, 550)

        # Store the base window flags
        self.base_flags = self.window.windowFlags()
        self.window.setWindowFlags(self.base_flags | Qt.WindowType.WindowStaysOnTopHint)

        # Create central widget
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Gradient Removal")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Instructions group
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)

        instructions_text = QLabel("""
1. Ensure an image is loaded in Siril.
2. Optionally, draw exclusion zones on the image in the main Siril window. Exclusion \
zones are useful to prevent over-correction of truly dark features like dark nebulae. \
The \"Always on Top\" checkbox can be used to prevent the script going behind Siril \
when drawing exclusion areas.
3. Adjust parameters as needed.
4. Process the image to remove gradients.
        """)
        instructions_text.setWordWrap(True)
        instructions_layout.addWidget(instructions_text)
        main_layout.addWidget(instructions_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Number of sample points
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Number of Sample Points:"))
        self.sample_points_spinbox = QSpinBox()
        self.sample_points_spinbox.setRange(10, 1000)
        self.sample_points_spinbox.setSingleStep(10)
        self.sample_points_spinbox.setValue(self.num_sample_points)
        self.sample_points_spinbox.setToolTip("Number of sample points for gradient estimation")
        sample_layout.addWidget(self.sample_points_spinbox)
        params_layout.addLayout(sample_layout)

        # Polynomial degree
        poly_layout = QHBoxLayout()
        poly_layout.addWidget(QLabel("Polynomial Degree:"))
        self.poly_degree_spinbox = QSpinBox()
        self.poly_degree_spinbox.setRange(1, 10)
        self.poly_degree_spinbox.setValue(self.poly_degree)
        self.poly_degree_spinbox.setToolTip("Degree of polynomial for gradient fitting")
        poly_layout.addWidget(self.poly_degree_spinbox)
        params_layout.addLayout(poly_layout)

        # RBF smoothing
        rbf_layout = QHBoxLayout()
        rbf_layout.addWidget(QLabel("RBF Smoothness:"))
        self.rbf_smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.rbf_smooth_slider.setRange(0, 1000)  # 0 to 10.0 with 0.01 precision
        self.rbf_smooth_slider.setValue(int(self.rbf_smooth * 100))
        self.rbf_smooth_slider.setToolTip("RBF smoothing parameter")
        self.rbf_smooth_slider.valueChanged.connect(self.update_rbf_smooth_display)
        rbf_layout.addWidget(self.rbf_smooth_slider)

        self.rbf_smooth_label = QLabel(f"{self.floor_value(self.rbf_smooth):.2f}")
        self.rbf_smooth_label.setMinimumWidth(50)
        rbf_layout.addWidget(self.rbf_smooth_label)
        params_layout.addLayout(rbf_layout)

        # Show gradient checkbox
        self.show_gradient_checkbox = QCheckBox("Show Gradient Removed")
        self.show_gradient_checkbox.setChecked(self.show_gradient)
        self.show_gradient_checkbox.setEnabled(False)
        self.show_gradient_checkbox.setToolTip("Display the computed gradient (only available after processing)")
        self.show_gradient_checkbox.toggled.connect(self.toggle_output)
        params_layout.addWidget(self.show_gradient_checkbox)

        main_layout.addWidget(params_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.add_exclusion_button = QPushButton("Add Exclusion Area")
        self.add_exclusion_button.setToolTip("Add an exclusion area. If a selection exists, the exclusion will be set to match the selection, otherwise the user can draw a freehand exclusion on the Siril image.")
        self.add_exclusion_button.clicked.connect(self.add_exclusion)
        button_layout.addWidget(self.add_exclusion_button)

        self.clear_exclusion_button = QPushButton("Clear Exclusion Areas")
        self.clear_exclusion_button.setToolTip("Clear all drawn exclusion areas.")
        self.clear_exclusion_button.clicked.connect(self.clear_exclusion_areas)
        button_layout.addWidget(self.clear_exclusion_button)

        button_layout.addStretch()

        self.process_button = QPushButton("Process")
        self.process_button.setToolTip("Process the image to remove gradients")
        self.process_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.process_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_dialog)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setFrameStyle(QFrame.Shape.StyledPanel)
        self.status_label.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.status_label)

        # Always on top toggle
        always_on_top_layout = QHBoxLayout()
        self.always_on_top_checkbox = QCheckBox("Always on Top")
        self.always_on_top_checkbox.setToolTip("Make the script stay on top. This helps "
            "with selecting exclusions. If your desktop doesn't support this "
            "properly (e.g. Linux / Wayland) you may need to use other methods, such "
            "as window settings provided by the window manager.")
        self.always_on_top_checkbox.setChecked(True)  # Start with it on
        self.always_on_top_checkbox.toggled.connect(self.toggle_always_on_top)
        always_on_top_layout.addWidget(self.always_on_top_checkbox)
        always_on_top_layout.addStretch()
        main_layout.addLayout(always_on_top_layout)

        # Show the window
        self.window.show()

    def toggle_always_on_top(self):
        """Toggle the always on top window flag"""
        # Hide the window first
        self.window.hide()

        if self.always_on_top_checkbox.isChecked():
            self.window.setWindowFlags(self.base_flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.window.setWindowFlags(self.base_flags)

        # Show the window again to apply changes
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()  # and focused if possible

    def add_exclusion(self):
        selection = self.siril.get_siril_selection()
        if selection is not None:
            selection_poly = s.Polygon.from_rectangle(selection, color=0xFF000080, fill=True)
            self.siril.overlay_add_polygon(selection_poly)
            self.siril.set_siril_selection(0, 0, 0, 0)
        else:
            try:
                self.siril.overlay_draw_polygon(color=0xFF000080, fill=True)
            except s.MouseModeError:
                QMessageBox.warning(self.window, "Warning", "Mouse must be in normal drag/select mode to draw a polygon")

    def clear_exclusion_areas(self):
        """Clears all drawn exclusion polygons."""
        self.exclusion_polygons = []
        self.current_polygon = []
        self.siril.overlay_clear_polygons()

    def toggle_output(self):
        show_bg = self.show_gradient_checkbox.isChecked()
        if show_bg:
            with self.siril.image_lock():
                output_image = self.gradient_background[0] if self.originally_mono else self.gradient_background
                self.siril.set_image_pixeldata(output_image)
        else:
            with self.siril.image_lock():
                output_image = self.corrected_image[0] if self.originally_mono else self.corrected_image
                self.siril.set_image_pixeldata(output_image)

    def start_processing(self):
        """Start processing in a separate thread"""
        # Update parameters from GUI
        self.num_sample_points = self.sample_points_spinbox.value()
        self.poly_degree = self.poly_degree_spinbox.value()

        # Disable UI during processing
        self.process_button.setEnabled(False)
        self.add_exclusion_button.setEnabled(False)
        self.clear_exclusion_button.setEnabled(False)

        # Start processing thread
        self.processing_thread = ProcessingThread(self)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.progress.connect(self.update_status)
        self.processing_thread.error.connect(self.processing_error)
        self.processing_thread.start()

    def processing_finished(self):
        """Called when processing is complete"""
        self.show_gradient_checkbox.setEnabled(True)
        self.process_button.setEnabled(True)
        self.add_exclusion_button.setEnabled(True)
        self.clear_exclusion_button.setEnabled(True)
        self.status_label.setText("Processing Complete")

    def processing_error(self, error_msg):
        """Called when processing encounters an error"""
        self.process_button.setEnabled(True)
        self.add_exclusion_button.setEnabled(True)
        self.clear_exclusion_button.setEnabled(True)
        self.status_label.setText("Error occurred during processing")
        QMessageBox.critical(self.window, "Processing Error", f"An error occurred during processing:\n{error_msg}")

    def update_status(self, message):
        """Update the status label"""
        self.status_label.setText(message)
        QApplication.processEvents()  # Allow GUI to update

    def process_image_threaded(self, progress_callback):
        """Threaded version of process_image with progress callbacks"""
        # Stretch the image before processing
        progress_callback("Normalizing image for processing...")
        stretched_image = self.stretch_image(self.image)

        # Check if the image is color
        is_color = len(stretched_image.shape) == 3

        # Store original median
        original_median = np.median(stretched_image)

        # Create exclusion mask
        exclusion_mask = None
        self.exclusion_polygons = self.siril.overlay_get_polygons_list()
        if self.exclusion_polygons:
            exclusion_mask = self.create_exclusion_mask(stretched_image.shape, self.exclusion_polygons)

        # First Stage: Polynomial Gradient Removal
        progress_callback("Step 1: Polynomial Gradient Removal")
        small_image_poly = self.downsample_image(stretched_image, self.downsample_scale)

        if exclusion_mask is not None:
            small_exclusion_mask_poly = self.downsample_image(exclusion_mask.astype(np.float32), self.downsample_scale) >= 0.5
        else:
            small_exclusion_mask_poly = None

        poly_sample_points = self.generate_sample_points(
            small_image_poly, num_points=self.num_sample_points, exclusion_mask=small_exclusion_mask_poly
        )

        if is_color:
            poly_background = np.zeros_like(stretched_image)
            for channel in range(3):
                poly_bg_channel = self.fit_polynomial_gradient(
                    small_image_poly[:, :, channel], poly_sample_points, degree=self.poly_degree
                )
                poly_background[:, :, channel] = self.upscale_background(poly_bg_channel, stretched_image.shape[:2])
        else:
            poly_background_small = self.fit_polynomial_gradient(small_image_poly, poly_sample_points, degree=self.poly_degree)
            poly_background = self.upscale_background(poly_background_small, stretched_image.shape[:2])

        image_after_poly = stretched_image - poly_background
        image_after_poly = self.normalize_image(image_after_poly, original_median)
        image_after_poly = np.clip(image_after_poly, 0, 1)

        # Second Stage: RBF Gradient Removal
        progress_callback("Step 2: RBF Gradient Removal")
        small_image_rbf = self.downsample_image(image_after_poly, self.downsample_scale)

        if exclusion_mask is not None:
            small_exclusion_mask_rbf = self.downsample_image(exclusion_mask.astype(np.float32), self.downsample_scale) >= 0.5
        else:
            small_exclusion_mask_rbf = None

        rbf_sample_points = self.generate_sample_points(
            small_image_rbf, num_points=self.num_sample_points, exclusion_mask=small_exclusion_mask_rbf
        )

        if is_color:
            rbf_background = np.zeros_like(stretched_image)
            for channel in range(3):
                rbf_bg_channel = self.fit_background(
                    small_image_rbf[:, :, channel], rbf_sample_points, smooth=self.rbf_smooth, patch_size=15
                )
                rbf_background[:, :, channel] = self.upscale_background(rbf_bg_channel, stretched_image.shape[:2])
        else:
            rbf_background_small = self.fit_background(small_image_rbf, rbf_sample_points, smooth=self.rbf_smooth, patch_size=15)
            rbf_background = self.upscale_background(rbf_background_small, stretched_image.shape[:2])

        corrected_image = image_after_poly - rbf_background
        corrected_image = self.normalize_image(corrected_image, original_median)
        corrected_image = np.clip(corrected_image, 0, 1)

        # Unstretch both images
        progress_callback("De-Normalizing the processed images...")
        corrected_image = self.unstretch_image(corrected_image)
        total_background = poly_background + rbf_background
        gradient_background = self.unstretch_image(total_background)

        # Ensure both images are 3-channel RGB
        corrected_image = self.ensure_rgb(corrected_image)
        gradient_background = self.ensure_rgb(gradient_background)
        if self.originally_16bit:
            corrected_image = (corrected_image * 65535).astype(np.uint16)
            gradient_background = (gradient_background * 65535).astype(np.uint16)

        # Convert back to CHW for returning to Siril
        self.corrected_image = corrected_image.transpose(2, 0, 1)
        self.gradient_background = gradient_background.transpose(2, 0, 1)

        # Update the image in Siril
        self.siril.undo_save_state(f"Auto BGE ({self.num_sample_points} points, "
                                   f"poly degree={self.poly_degree}, "
                                   f"RBF smoothness={self.rbf_smooth})")
        with self.siril.image_lock():
            output_image = self.corrected_image[0] if self.originally_mono else self.corrected_image
            self.siril.set_image_pixeldata(output_image)

        self.siril.clear_image_bgsamples()

    def process_image(self):
        """
        Processes the image to subtract the background in two stages:
        1. Polynomial gradient removal.
        2. RBF gradient removal.
        """
        self.process_image_threaded(lambda msg: None)  # CLI version without progress callbacks

    # All the image processing helper methods remain the same
    def ensure_rgb(self, image):
        """
        Ensures the given image is 3-channel RGB.
        Args:
            image: The input NumPy array (can be 2D or 3D with a single channel).
        Returns:
            A 3D NumPy array with shape (height, width, 3).
        """
        if image.ndim == 2:  # Grayscale image
            return np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 1:  # Single-channel image with an extra dimension
            return np.repeat(image, 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 3:  # Already RGB
            return image
        raise ValueError(f"Unexpected image shape: {image.shape}")

    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image.
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # Check if the image is single-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel by duplicating

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Initialize lists to store per-channel minima and medians
        self.stretch_original_mins = []
        self.stretch_original_medians = []

        # Initialize stretched_image as a copy of the input image
        stretched_image = image.copy()

        # Define the target median for stretching
        target_median = 0.25

        # Apply the stretch for each channel independently
        for c in range(3):
            # Record the minimum of the current channel
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)

            # Subtract the channel's minimum to shift the image
            stretched_image[..., c] -= channel_min

            # Record the median of the shifted channel
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)

            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        # Clip stretched image to [0, 1] range
        stretched_image = np.clip(stretched_image, 0.0, 1.0)

        # Store stretch parameters
        self.was_single_channel = was_single_channel

        return stretched_image

    def unstretch_image(self, image):
        """
        Undo the unlinked linear stretch to return the image to its original state.
        Each channel is unstretched independently by reverting the stretch formula
        using the stored medians and adding back the individual channel minima.
        Returns the unstretched image.
        """
        original_mins = self.stretch_original_mins
        original_medians = self.stretch_original_medians
        was_single_channel = self.was_single_channel

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Apply the unstretch for each channel independently
        for c in range(3):
            channel_median = np.median(image[..., c])
            original_median = original_medians[c]
            original_min = original_mins[c]

            if channel_median != 0 and original_median != 0:
                numerator = (channel_median - 1) * original_median * image[..., c]
                denominator = (
                    channel_median * (original_median + image[..., c] - 1)
                    - original_median * image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median or original median is zero. Skipping unstretch.")

            # Add back the channel's original minimum
            image[..., c] += original_min

        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)

        # If the image was originally single-channel, convert back to single-channel
        if was_single_channel:
            image = np.mean(image, axis=2, keepdims=True)  # Convert back to single-channel

        return image

    def downsample_image(self, image, scale=4):
        """
        Downsamples the image by the specified scale factor using area interpolation.

        Args:
            image: 2D/3D NumPy array of the image.
            scale: Downsampling scale factor.

        Returns:
            downsampled_image: Downsampled image.
        """
        new_size = (max(1, image.shape[1] // scale), max(1, image.shape[0] // scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def upscale_background(self, background, original_shape):
        """
        Upscales the background model to the original image size.

        Args:
            background: 2D NumPy array (single-channel background model).
            original_shape: Tuple of (height, width) for the target size.

        Returns:
            upscaled_background: Upscaled 2D background model.
        """
        if background.ndim == 2:
            # Single-channel (grayscale) input
            return cv2.resize(background, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if background.ndim == 3 and background.shape[2] == 1:
            # Ensure input shape is reduced to 2D for single-channel data
            background = background.squeeze()  # Remove singleton dimension

        return cv2.resize(background, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LANCZOS4)

    def divide_into_quartiles(self, image):
        """
        Divides the image into four quartiles.

        Args:
            image: 2D/3D NumPy array of the image.

        Returns:
            quartiles: Dictionary containing quartile images.
        """
        h, w = image.shape[:2]
        half_h, half_w = h // 2, w // 2
        return {
            'top_left': image[:half_h, :half_w],
            'top_right': image[:half_h, half_w:],
            'bottom_left': image[half_h:, :half_w],
            'bottom_right': image[half_h:, half_w:],
        }

    def exclude_bright_regions(self, quartile, exclusion_fraction=0.5):
        """
        Excludes the brightest regions in a quartile based on the exclusion fraction.

        Args:
            quartile: 2D/3D NumPy array of the quartile image.
            exclusion_fraction: Fraction of the brightest pixels to exclude.

        Returns:
            mask: Boolean mask where True indicates eligible pixels.
        """
        flattened = quartile.flatten()
        threshold = np.percentile(flattened, 100 * (1 - exclusion_fraction))
        mask = quartile < threshold
        return mask

    def gradient_descent_to_dim_spot(self, image, x, y, max_iterations=100, patch_size=15):
        """
        Moves a point to a dimmer spot using gradient descent, considering the median of a patch.

        Args:
            image: 2D/3D NumPy array of the image.
            x, y: Initial coordinates of the point.
            max_iterations: Maximum number of descent steps.
            patch_size: Size of the square patch (e.g., 15 for a 15x15 patch).

        Returns:
            (x, y): Coordinates of the dimmest local spot found.
        """
        half_patch = patch_size // 2

        # Get image dimensions and convert to luminance if color
        if len(image.shape) == 3:
            h, w, _ = image.shape
            luminance = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            h, w = image.shape
            luminance = image

        for _ in range(max_iterations):
            # Define patch around the current point
            xmin, xmax = max(0, x - half_patch), min(w, x + half_patch + 1)
            ymin, ymax = max(0, y - half_patch), min(h, y + half_patch + 1)
            patch = luminance[ymin:ymax, xmin:xmax]
            current_value = np.median(patch)

            # Define a 3x3 window around the point
            neighbors = [
                (nx, ny) for nx in range(max(0, x - 1), min(w, x + 2))
                          for ny in range(max(0, y - 1), min(h, y + 2))
                          if (nx, ny) != (x, y)
            ]

            # Find the dimmest neighbor using patch medians
            def patch_median(coord):
                nx, ny = coord
                xmin_n, xmax_n = max(0, nx - half_patch), min(w, nx + half_patch + 1)
                ymin_n, ymax_n = max(0, ny - half_patch), min(h, ny + half_patch + 1)
                neighbor_patch = luminance[ymin_n:ymax_n, xmin_n:xmax_n]
                return np.median(neighbor_patch)

            dimmest_neighbor = min(neighbors, key=patch_median)
            dimmest_value = patch_median(dimmest_neighbor)

            # If the current point is already the dimmest, stop
            if dimmest_value >= current_value:
                break

            # Move to the dimmest neighbor
            x, y = dimmest_neighbor

        return x, y

    def fit_polynomial_gradient(self, image, sample_points, degree=2, patch_size=15):
        """
        Fits a polynomial gradient (up to the specified degree) to the image using sample points.

        Args:
            image: 2D/3D NumPy array of the image.
            sample_points: Array of (x, y) sample point coordinates.
            degree: Degree of the polynomial (e.g., 1 for linear, 2 for quadratic).
            patch_size: Size of the square patch for median calculation.

        Returns:
            background: The polynomial gradient model across the image.
        """
        h, w = image.shape[:2]
        half_patch = patch_size // 2

        x, y = sample_points[:, 0].astype(np.int32), sample_points[:, 1].astype(np.int32)
        valid_indices = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x[valid_indices], y[valid_indices]

        if len(image.shape) == 3:  # Color image
            background = np.zeros_like(image)
            for channel in range(image.shape[2]):  # Process each channel separately
                z = []
                for xi, yi in zip(x, y):
                    xmin, xmax = max(0, xi - half_patch), min(w, xi + half_patch + 1)
                    ymin, ymax = max(0, yi - half_patch), min(h, yi + half_patch + 1)
                    patch = image[ymin:ymax, xmin:xmax, channel]
                    z.append(np.median(patch))
                z = np.array(z, dtype=np.float64)

                # Fit polynomial model for this channel
                terms = []
                for i in range(degree + 1):
                    for j in range(degree + 1 - i):
                        terms.append((x**i) * (y**j))
                A = np.column_stack(terms)
                coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

                # Generate polynomial model
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                terms = []
                for i in range(degree + 1):
                    for j in range(degree + 1 - i):
                        terms.append((xx**i) * (yy**j))
                terms = np.array(terms)
                background[:, :, channel] = np.sum(coeffs[:, None, None] * terms, axis=0)
            return background
        return self.fit_polynomial_gradient(image[:, :, np.newaxis], sample_points, degree, patch_size)

    def generate_sample_points(self, image, num_points=100, exclusion_mask=None):
        """
        Generates sample points for gradient fitting, avoiding exclusion zones.

        Args:
            image: 2D/3D NumPy array of the image.
            num_points: Total number of sample points to generate.
            exclusion_mask: 2D boolean NumPy array where False indicates exclusion.

        Returns:
            points: NumPy array of shape (N, 2) with (x, y) coordinates.
        """
        h, w = image.shape[:2]
        points = []

        # Add border points: 1 in each corner and 5 along each border
        border_margin = 10

        # Corner points
        corners = [
            (border_margin, border_margin),                # Top-left
            (w - border_margin - 1, border_margin),        # Top-right
            (border_margin, h - border_margin - 1),        # Bottom-left
            (w - border_margin - 1, h - border_margin - 1) # Bottom-right
        ]
        for x, y in corners:
            if exclusion_mask is not None and not exclusion_mask[y, x]:
                continue
            x_new, y_new = self.gradient_descent_to_dim_spot(image, x, y)
            if exclusion_mask is not None and not exclusion_mask[y_new, x_new]:
                continue
            points.append((x_new, y_new))

        # Top and bottom borders
        for x in np.linspace(border_margin, w - border_margin, 5, dtype=int):
            # Top border
            if exclusion_mask is not None and not exclusion_mask[border_margin, x]:
                continue
            x_top, y_top = self.gradient_descent_to_dim_spot(image, x, border_margin)
            if exclusion_mask is not None and not exclusion_mask[y_top, x_top]:
                continue
            points.append((x_top, y_top))
            # Bottom border
            if exclusion_mask is not None and not exclusion_mask[h - border_margin - 1, x]:
                continue
            x_bottom, y_bottom = self.gradient_descent_to_dim_spot(image, x, h - border_margin - 1)
            if exclusion_mask is not None and not exclusion_mask[y_bottom, x_bottom]:
                continue
            points.append((x_bottom, y_bottom))

        # Left and right borders
        for y in np.linspace(border_margin, h - border_margin, 5, dtype=int):
            # Left border
            if exclusion_mask is not None and not exclusion_mask[y, border_margin]:
                continue
            x_left, y_left = self.gradient_descent_to_dim_spot(image, border_margin, y)
            if exclusion_mask is not None and not exclusion_mask[y_left, x_left]:
                continue
            points.append((x_left, y_left))
            # Right border
            if exclusion_mask is not None and not exclusion_mask[y, w - border_margin - 1]:
                continue
            x_right, y_right = self.gradient_descent_to_dim_spot(image, w - border_margin - 1, y)
            if exclusion_mask is not None and not exclusion_mask[y_right, x_right]:
                continue
            points.append((x_right, y_right))

        # Add random points in eligible areas (using quartiles)
        quartiles = self.divide_into_quartiles(image)
        for key, quartile in quartiles.items():
            # Determine the coordinates of the quartile in the full image
            h_quart, w_quart = quartile.shape[:2]
            if "top" in key:
                y_start = 0
            else:
                y_start = h // 2
            if "left" in key:
                x_start = 0
            else:
                x_start = w // 2

            # Create local exclusion mask for the quartile
            if exclusion_mask is not None:
                quart_exclusion_mask = exclusion_mask[y_start:y_start + h_quart, x_start:x_start + w_quart]
            else:
                quart_exclusion_mask = None

            # Convert quartile to grayscale if it has multiple channels
            if quartile.ndim == 3:
                # Assuming the color channels are last, convert to luminance
                quartile_gray = np.dot(quartile[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                quartile_gray = quartile

            # Exclude bright regions
            mask = self.exclude_bright_regions(quartile_gray, exclusion_fraction=0.5)
            if quart_exclusion_mask is not None:
                mask &= quart_exclusion_mask

            eligible_indices = np.argwhere(mask)

            if len(eligible_indices) == 0:
                continue  # Skip if no eligible points in this quartile

            # Ensure we don't request more points than available
            num_points_in_quartile = min(len(eligible_indices), num_points // 4)
            selected_indices = eligible_indices[np.random.choice(len(eligible_indices), num_points_in_quartile, replace=False)]

            for idx in selected_indices:
                y_idx, x_idx = idx  # Unpack row to y, x
                y_coord = y_start + y_idx
                x_coord = x_start + x_idx

                # Apply gradient descent to move to a dimmer spot
                x_new, y_new = self.gradient_descent_to_dim_spot(image, x_coord, y_coord)

                # Check if the new point is in exclusion
                if exclusion_mask is not None and not exclusion_mask[y_new, x_new]:
                    continue  # Skip points in exclusion areas

                points.append((x_new, y_new))

        factor = self.downsample_scale
        scaled_points = [(x * factor, (h - y) * factor) for x, y in points]
        self.siril.set_image_bgsamples(scaled_points, show_samples = True)
        return np.array(points)

    def fit_background(self, image, sample_points, smooth=0.1, patch_size=15):
        """
        Fits a background model using RBF interpolation.

        Args:
            image: 2D/3D NumPy array of the image.
            sample_points: Array of (x, y) sample point coordinates.
            smooth: Smoothness parameter for the RBF fitting.
            patch_size: Size of the square patch for median calculation.

        Returns:
            background: The RBF-based background model.
        """
        h, w = image.shape[:2]
        half_patch = patch_size // 2

        x, y = sample_points[:, 0].astype(np.int32), sample_points[:, 1].astype(np.int32)
        valid_indices = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = x[valid_indices], y[valid_indices]

        if len(image.shape) == 3:  # Color image
            background = np.zeros_like(image)
            for channel in range(image.shape[2]):  # Process each channel separately
                z = []
                for xi, yi in zip(x, y):
                    xmin, xmax = max(0, xi - half_patch), min(w, xi + half_patch + 1)
                    ymin, ymax = max(0, yi - half_patch), min(h, yi + half_patch + 1)
                    patch = image[ymin:ymax, xmin:xmax, channel]
                    z.append(np.median(patch))
                z = np.array(z, dtype=np.float64)

                # Fit RBF for this channel
                rbf = Rbf(x, y, z, function='multiquadric', smooth=smooth, epsilon=1.0)
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                background[:, :, channel] = rbf(grid_x, grid_y)
            return background
        return self.fit_background(image[:, :, np.newaxis], sample_points, smooth, patch_size)

    def calculate_median(self, values):
        """
        Calculates the median of the given values.

        Args:
            values: NumPy array of values.

        Returns:
            median: Median value.
        """
        return np.median(values)

    def calculate_mad(self, values, median):
        """
        Calculates the Median Absolute Deviation (MAD).

        Args:
            values: NumPy array of values.
            median: Median of the values.

        Returns:
            mad: Median Absolute Deviation.
        """
        deviations = np.abs(values - median)
        return np.median(deviations)

    def calculate_noise_weight(self, median, mad):
        """
        Calculates the noise weight based on median and MAD.

        Args:
            median: Median value.
            mad: Median Absolute Deviation.

        Returns:
            noise_weight: Noise weight (0.0 to 1.0).
        """
        if median == 0:
            median = 1e-6  # Avoid division by zero
        noise_factor = 1.0 - (mad / median)
        return max(0.0, min(1.0, noise_factor))

    def calculate_brightness_weight(self, avg_brightness, median_brightness):
        """
        Calculates the brightness weight based on average and median brightness.

        Args:
            avg_brightness: Average brightness of the patch.
            median_brightness: Median brightness of the patch.

        Returns:
            brightness_weight: Brightness weight (0.8 to 1.0).
        """
        if median_brightness == 0:
            median_brightness = 1e-6  # Avoid division by zero
        weight = 1.0 - abs(avg_brightness - median_brightness) / median_brightness
        return max(0.8, min(1.0, weight))  # Limit range for stability

    def calculate_spatial_weight(self, x, y, width, height):
        """
        Calculates the spatial weight based on the position of the point.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
            width: Image width.
            height: Image height.

        Returns:
            spatial_weight: Spatial weight (0.95 to 1.0).
        """
        center_x = width / 2
        center_y = height / 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        normalized_distance = distance / max_distance
        return 0.95 + 0.05 * normalized_distance

    def create_exclusion_mask(self, image_shape, exclusion_polygons):
        """
        Creates a boolean mask with False in exclusion areas and True elsewhere.

        Args:
            image_shape: Shape of the image (height, width, channels).
            exclusion_polygons: List of Polygon objects.

        Returns:
            mask: 2D boolean NumPy array.
        """
        h = image_shape[0]
        mask = np.ones(image_shape[:2], dtype=bool)  # Initialize all True

        if not exclusion_polygons:
            return mask  # No exclusions

        # Prepare polygons for OpenCV
        polygons = []
        for polygon in exclusion_polygons:
            if len(polygon.points) < 3:
                continue
            points = []
            for point in polygon.points:
                # Scale back to original image coordinates
                points.append([int(point.x), int(h - point.y)])
            polygons.append(np.array(points, dtype=np.int32))

        # Create a single-channel mask
        exclusion_mask = np.zeros(image_shape[:2], dtype=np.uint8)

        # Fill the polygons on the exclusion mask
        cv2.fillPoly(exclusion_mask, polygons, 1)  # 1 inside polygons

        # Update the main mask: False inside exclusion polygons
        mask[exclusion_mask == 1] = False
        return mask

    def normalize_image(self, image, target_median):
        """
        Normalizes the image so that its median matches the target median.

        Args:
            image: 2D/3D NumPy array of the image.
            target_median: The desired median value.

        Returns:
            normalized_image: The median-normalized image.
        """
        current_median = np.median(image)
        median_diff = target_median - current_median
        normalized_image = image + median_diff
        return normalized_image


def main():
    try:
        # Initialize Qt application
        app = QApplication(sys.argv)

        # Launch Interface to determine if we are in CLI or GUI mode and to init connection
        siril = s.SirilInterface()
        try:
            siril.connect()
        except s.SirilConnectionError:
            if not siril.is_cli():
                QMessageBox.critical(None, "Error", "Failed to connect to Siril")
            else:
                print("Failed to connect to Siril")
            return

        if siril.is_cli():
            # CLI mode
            parser = argparse.ArgumentParser(description="Automatic Background Extraction")
            parser.add_argument("-npoints", type=int, default=100,
                                help="Number of interior background samples")
            parser.add_argument("-polydegree", type=int, default=2,
                                help="Polynomial degree for initial poly fit")
            parser.add_argument("-rbfsmooth", type=float, default=0.1,
                                help="Smoothing amount for refinement RBF fit")

            args = parser.parse_args()
            processor = GradientRemovalInterface(siril, app, cli_args=args)
        else:
            # GUI mode
            processor = GradientRemovalInterface(siril, app)
            app.exec()

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
