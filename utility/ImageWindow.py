"""
Siril Image Window - A pseudo-MDI GUI script for storing and swapping images with Siril
"""

# (c) Adrian Knagg-Baugh 2025
# Blink Comparator for Siril
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.0
#

import io
import sys
import math
import time
import threading
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass

import sirilpy as s
from sirilpy import STFType
# Declare non-core modules after importing sirilpy
s.ensure_installed("PyQt6", "Pillow", "numba")

import numba
from numba import jit, prange, vectorize

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QMessageBox, QScrollArea,
    QSizePolicy, QFrame, QComboBox, QDialog, QTextEdit,
    QDialogButtonBox, QSlider, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF, QObject, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent
from PIL import Image, ImageCms

class InstructionsDialog(QDialog):
    """Popup dialog to show usage instructions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instructions - Blink / Browse / Filter / Sort")
        self.setModal(True)
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        instructions = QTextEdit(self)
        instructions.setReadOnly(True)
        instructions.setPlainText(
            "This script provides a pseudo-MDI (Multiple Document Interface) "
            "for Siril. At startup, the current image loaded in Siril is copied to "
            "the script, with the same display mode as in Siril and the same colour managed "
            "display. You can pan and zoom the image independently of Siril and change its "
            "display mode: you can also sync the script with Siril so that either the display "
            "mode, pan and zoom in the script follows Siril or the display mode, pan and zoom "
            "in Siril follows the script.\n\n"
            "Siril remains a SDI (Single Document Interface) program, however you can use "
            "the script to swap images and all their metadata between the script window and "
            "Siril. This allows all the workflow options of a MDI interface, you just need "
            "to switch each image into Siril as you wish to work on it. Multiple copies of the "
            "script may be run simultaneously to hold several different images at once, swapping "
            "each into Siril as required.\n\n"
            "Note that the image in the script window is not updated by work done on its copy "
            "in Siril: it is a true independent copy. This means you can also use it as a "
            "checkpoint: if you wish to experiment with a lengthy workflow you can start the "
            "script, carry out many operations in Siril, and then if you decide you don't like "
            "the result simply copy the script's copy of the image back into Siril as an instant "
            "multi-undo.\n\n"
            "Notes\n\n1. When copying an image back from the script into Siril or swapping images "
            "between the script and Siril your Siril undo history will reset: the complexity of "
            "managing multple separate undo histories is not manageable at this point in time.\n\n"
            "2. Overlays and annotations are not available in the script window: only the image "
            "is shown.\n\n"
            "3. Currently, toggling linked autostretch in the script does not update it in Siril "
            "because there is a missing setter method. This will be fixed in Siril 1.4.0-rc1.\n\n"
        )
        layout.addWidget(instructions)

        # OK button
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
        )
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

class ZoomableImageLabel(QLabel):
    """Custom QLabel that supports pan and zoom functionality"""

    # Signal to notify when pan/zoom changes
    pan_changed = pyqtSignal()
    zoom_changed = pyqtSignal()
    zoomed_to_fit = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid gray;")

        self._pixmap: Optional[QPixmap] = None
        self._scale_factor = 1.0

        # Pan state stored in **image coordinates**
        self._image_pan_x = 0.0
        self._image_pan_y = 0.0

        # Mouse drag bookkeeping
        self._last_pan_point = None

        self._last_zoom_time = 0
        self._zoom_rate_limit = 0.02

        self.setMouseTracking(True)

    def set_image_data(self, image_data: np.ndarray, preserve_view=False):
        """Convert numpy array to QPixmap and display

        Args:
            image_data: The image data as numpy array
            preserve_view: If True, preserve current zoom/pan; if False, scale to fit
        """
        if image_data is None:
            self.clear()
            self._pixmap = None
            return

        # Handle different image formats
        if len(image_data.shape) == 2:
            # Grayscale image (hw)
            height, width = image_data.shape
            channels = 1
        elif len(image_data.shape) == 3 and image_data.shape[0] in [1, 3]:
            # CHW format
            channels, height, width = image_data.shape
            if channels == 1:
                image_data = image_data[0]  # Remove channel dimension
            else:
                # Convert CHW to HWC
                image_data = np.transpose(image_data, (1, 2, 0))
        elif len(image_data.shape) == 3 and image_data.shape[2] in [1, 3]:
            # HWC format (used for preview)
            height, width, channels = image_data.shape
        else:
            raise ValueError(f"Unsupported image shape: {image_data.shape}")

        # Normalize data for display
        if image_data.dtype == np.float32:
            # Assume float32 is in range [0, 1] or normalize it
            if image_data.max() <= 1.0:
                display_data = (image_data * 255).astype(np.uint8)
            else:
                # Normalize to 0-255
                display_data = ((image_data - image_data.min()) /
                               (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        elif image_data.dtype == np.uint16:
            # Convert 16-bit to 8-bit for display
            display_data = (image_data / 256).astype(np.uint8)
        else:
            display_data = image_data.astype(np.uint8)

        # Create QImage
        if len(display_data.shape) == 2:
            # Grayscale
            bytes_per_line = width
            qimage = QImage(display_data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # RGB
            bytes_per_line = width * 3
            qimage = QImage(display_data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Flip vertically to match FITS ROWORDER convention (like Siril does)
        qimage = qimage.mirrored(False, True)

        # Create new pixmap
        self._pixmap = QPixmap.fromImage(qimage)

        # Only scale to fit for new images, not for display mode changes
        if not preserve_view:
            self.scale_to_fit()
        else:
            self.update_display()

            # Emit signal for sync purposes
            self.zoom_changed.emit()
            self.pan_changed.emit()

    def scale_to_fit(self):
        """Scale image to fit within the widget while maintaining aspect ratio"""
        if self._pixmap is None:
            return

        widget_size = self.size()
        widget_width = widget_size.width()
        widget_height = widget_size.height()

        image_width = self._pixmap.width()
        image_height = self._pixmap.height()

        # Calculate uniform scale to fit
        scale_x = widget_width / image_width
        scale_y = widget_height / image_height
        self._scale_factor = min(scale_x, scale_y)

        # Reset pan to centred in image coordinates
        # (0,0 means the image is exactly centred once scaled)
        self._image_pan_x = 0.0
        self._image_pan_y = 0.0

        self.update_display()

        # Notify sync users (Siril) that zoom/pan changed
        self.zoomed_to_fit.emit()
        self.pan_changed.emit()

    def update_display(self):
        """Update the displayed image with current scale and pan"""
        if self._pixmap is None:
            self.setText("No image loaded")
            return

        widget_size = self.size()
        display_pixmap = QPixmap(widget_size)
        display_pixmap.fill(Qt.GlobalColor.black)

        painter = QPainter(display_pixmap)

        # Convert stored image pan (image pixels) to widget offsets
        pan_x = int(self._image_pan_x * self._scale_factor)
        pan_y = int(self._image_pan_y * self._scale_factor)

        painter.translate(
            (widget_size.width() - self._pixmap.width() * self._scale_factor) // 2 + pan_x,
            (widget_size.height() - self._pixmap.height() * self._scale_factor) // 2 + pan_y
        )
        painter.scale(self._scale_factor, self._scale_factor)

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()

        self.setPixmap(display_pixmap)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out around the mouse position"""
        if self._pixmap is None:
            return

        current_time = time.time()
        if current_time - self._last_zoom_time < self._zoom_rate_limit:
            return
        self._last_zoom_time = current_time

        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 1.0 / 1.1
        new_scale = self._scale_factor * zoom_factor

        if new_scale != self._scale_factor:
            # Adjust pan so the image point under the mouse stays fixed
            mouse_pos = event.position()
            mouse_x = mouse_pos.x()
            mouse_y = mouse_pos.y()
            widget_size = self.size()

            old_scale = self._scale_factor
            self._scale_factor = new_scale

            # Find image coordinates of the mouse position before zoom
            old_pan_x = self._image_pan_x * old_scale
            old_pan_y = self._image_pan_y * old_scale
            img_x = (mouse_x - (widget_size.width() - self._pixmap.width() * old_scale) / 2 - old_pan_x) / old_scale
            img_y = (mouse_y - (widget_size.height() - self._pixmap.height() * old_scale) / 2 - old_pan_y) / old_scale

            # Recompute pan so that same image coords are under cursor after zoom
            new_pan_x = mouse_x - (widget_size.width() - self._pixmap.width() * new_scale) / 2 - img_x * new_scale
            new_pan_y = mouse_y - (widget_size.height() - self._pixmap.height() * new_scale) / 2 - img_y * new_scale

            self._image_pan_x = new_pan_x / new_scale
            self._image_pan_y = new_pan_y / new_scale

            self.update_display()
            self.zoom_changed.emit()

    def mousePressEvent(self, event: QMouseEvent):
        """Start panning on mouse press"""
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._last_pan_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Pan by dragging"""
        if (self._last_pan_point is not None and
            event.buttons() & Qt.MouseButton.LeftButton and
            self._pixmap is not None):

            delta = event.pos() - self._last_pan_point
            self._last_pan_point = event.pos()

            # Convert from widget pixels to image coords
            self._image_pan_x += delta.x() / self._scale_factor
            self._image_pan_y += delta.y() / self._scale_factor

            self.update_display()
            self.pan_changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End panning on mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pan_point = None

    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        if self._pixmap is not None:
            self.update_display()


class SyncWorker(QObject):
    """Worker object that runs in a separate thread for polling Siril"""

    # Signals to update UI from worker thread
    stf_changed = pyqtSignal(int, bool)
    zoom_pan_changed = pyqtSignal(float, float, float)  # zoom, xoff, yoff from Siril
    script_zoom_pan_changed = pyqtSignal(float, float, float)  # zoom, xoff, yoff to Siril

    def __init__(self, siril_interface, siril_mutex, poll_interval=100):
        super().__init__()
        self.siril = siril_interface
        self.siril_mutex = siril_mutex
        self.poll_interval = poll_interval / 1000.0  # Convert to seconds
        self.running = False

        # Track previous values to detect changes
        self.last_zoom = None
        self.last_xoff = None
        self.last_yoff = None

    def start_polling(self):
        """Start the polling loop"""
        self.running = True
        self.poll()

    def stop_polling(self):
        """Stop the polling loop"""
        self.running = False

    def poll(self):
        """Main polling loop - runs in the worker thread"""
        while self.running:
            try:
                # Use mutex to ensure thread-safe access to Siril
                if self.siril_mutex.acquire(timeout=0.1):  # Non-blocking acquire with timeout
                    try:
                        if self.siril and self.siril.is_image_loaded():
                            # Get STF
                            stf = self.siril.get_siril_stf()
                            linked = self.siril.get_siril_stf_linked()

                            # Emit signal to update UI on main thread
                            self.stf_changed.emit(stf, linked)

                            # Get pan and zoom
                            xoff, yoff, zoom = self.siril.get_siril_panzoom()

                            # Only emit signal if values have changed
                            if (self.last_zoom != zoom or
                                self.last_xoff != xoff or
                                self.last_yoff != yoff):

                                # Emit signal to update zoom/pan on main thread
                                # NOTE: emit (zoom, xoff, yoff) so the slot can convert image-space -> widget-space
                                self.zoom_pan_changed.emit(zoom, xoff, yoff)

                                # Update tracked values
                                self.last_zoom = zoom
                                self.last_xoff = xoff
                                self.last_yoff = yoff

                    except Exception as e:
                        print(f"Polling error: {str(e)}")
                    finally:
                        self.siril_mutex.release()

            except Exception as e:
                print(f"Polling thread error: {str(e)}")

            # Sleep for the specified interval
            time.sleep(self.poll_interval)


class SirilImageHolder(QMainWindow):
    """Main application window for the Siril Image Holder"""

    def __init__(self):
        super().__init__()
        self.siril = None
        self.stored_image = None
        self.stored_metadata = None
        self.stored_filename = "No image"
        self.display_mode = "Linear"  # Current display mode
        self.link = False  # this gets overridden when getting an image

        # Thread synchronization
        self.siril_mutex = threading.Lock()
        self.sync_thread = None
        self.sync_worker = None
        self.sync_mode = "None"  # None, Follow Siril, Follow Script
        self.calc_thread = None # MTF parameter precalculation thread

        # Track previous Siril pan/zoom for relative movement
        self.prev_siril_zoom = None
        self.prev_siril_xoff = None
        self.prev_siril_yoff = None

        # Track previous Script pan/zoom for relative movement (for Follow Script mode)
        self.prev_script_zoom = None
        self.prev_script_xoff = None
        self.prev_script_yoff = None

        # Sliders
        self.hi = 1.0
        self.lo = 0.0

        self.init_ui()
        self.connect_to_siril()

        # Check if Siril has an image or sequence loaded and auto-copy it
        if self.siril:
            with self.siril_mutex:
                is_sequence = self.siril.is_sequence_loaded()
                is_image = self.siril.is_image_loaded()

                if is_sequence or is_image:
                    stf = self.siril.get_siril_stf()
                    self.display_combo.setCurrentIndex(stf) # Match Siril STF
                    # Get the Siril display ICC profile
                    self.display_icc_profile = self.siril.get_siril_display_iccprofile()

            if is_sequence or is_image:
                self.copy_from_siril()

        # Connect image label signals
        self.image_label.pan_changed.connect(self.on_script_pan_changed)
        self.image_label.zoom_changed.connect(self.on_script_zoom_changed)
        self.image_label.zoomed_to_fit.connect(self.on_script_zoom_to_fit) # Direct call to handle zoom-to-fit

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Siril Image Holder - No image")
        self.setGeometry(100, 100, 800, 600)
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Main layout
        layout = QVBoxLayout(central_widget)
        # Image display area
        scroll_area = QScrollArea()
        self.image_label = ZoomableImageLabel()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Control buttons (Copy/Swap row)
        button_layout = QHBoxLayout()
        self.copy_from_siril_btn = QPushButton("Copy from Siril")
        self.copy_from_siril_btn.clicked.connect(self.copy_from_siril)
        self.copy_from_siril_btn.setEnabled(False)
        button_layout.addWidget(self.copy_from_siril_btn)
        self.swap_with_siril_btn = QPushButton("Swap with Siril")
        self.swap_with_siril_btn.clicked.connect(self.swap_with_siril)
        self.swap_with_siril_btn.setEnabled(False)
        button_layout.addWidget(self.swap_with_siril_btn)
        self.copy_to_siril_btn = QPushButton("Copy to Siril")
        self.copy_to_siril_btn.clicked.connect(self.copy_to_siril)
        self.copy_to_siril_btn.setEnabled(False)
        button_layout.addWidget(self.copy_to_siril_btn)
        self.open_from_file_btn = QPushButton("Open from file")
        self.open_from_file_btn.clicked.connect(self.open_from_file)
        self.open_from_file_btn.setEnabled(False)
        button_layout.addWidget(self.open_from_file_btn)
        layout.addLayout(button_layout)

        # Bottom section: sliders on left, controls on right
        bottom_layout = QHBoxLayout()

        # Left side: Sliders in a vertical box
        slider_layout = QVBoxLayout()

        # Max slider
        max_layout = QHBoxLayout()
        max_label = QLabel("Max:")
        max_layout.addWidget(max_label)
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.setValue(int(self.hi * 1000))
        self.max_slider.valueChanged.connect(self.on_max_slider_changed)
        self.max_slider.sliderReleased.connect(self.on_minmax_slider_released)
        max_layout.addWidget(self.max_slider)
        slider_layout.addLayout(max_layout)

        # Min slider
        min_layout = QHBoxLayout()
        min_label = QLabel("Min:")
        min_layout.addWidget(min_label)
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(int(self.lo * 1000))
        self.min_slider.valueChanged.connect(self.on_min_slider_changed)
        self.min_slider.sliderReleased.connect(self.on_minmax_slider_released)
        min_layout.addWidget(self.min_slider)
        slider_layout.addLayout(min_layout)

        bottom_layout.addLayout(slider_layout, stretch=1)

        # Right side: Controls in two rows
        controls_layout = QVBoxLayout()

        # Top row: Scale to Fit, Sync mode dropdown, Help
        top_row = QHBoxLayout()
        top_row.addStretch()  # Push everything to the right
        self.scale_to_fit_btn = QPushButton("Scale to Fit")
        self.scale_to_fit_btn.clicked.connect(self.image_label.scale_to_fit)
        top_row.addWidget(self.scale_to_fit_btn)

        sync_label = QLabel("Sync mode:")
        top_row.addWidget(sync_label)
        self.sync_combo = QComboBox()
        self.sync_combo.addItems(["None", "Follow Siril", "Follow Script"])
        self.sync_combo.setToolTip("Synchronize pan / zoom / display mode between Siril and script")
        self.sync_combo.currentTextChanged.connect(self.on_sync_mode_changed)
        self.sync_combo.setEnabled(False)
        top_row.addWidget(self.sync_combo)

        self.instructions_btn = QPushButton("Help")
        self.instructions_btn.clicked.connect(self.show_instructions)
        top_row.addWidget(self.instructions_btn)

        controls_layout.addLayout(top_row)

        # Bottom row: Display mode and Linked checkbox
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()  # Push everything to the right
        mode_label = QLabel("Display:")
        bottom_row.addWidget(mode_label)
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Linear", "Logarithm", "Square Root", "Squared", "Asinh", "Autostretch", "Histogram"])
        self.display_combo.currentTextChanged.connect(self.on_display_mode_changed)
        self.display_combo.setEnabled(False)
        self.display_combo.setToolTip("Set the display mode")
        bottom_row.addWidget(self.display_combo)

        self.linked_checkbox = QCheckBox("Linked STF")
        self.linked_checkbox.setChecked(self.link)
        self.linked_checkbox.stateChanged.connect(self.on_linked_changed)
        self.linked_checkbox.setToolTip("Set whether the Autostretch channels are linked or unlinked")
        bottom_row.addWidget(self.linked_checkbox)

        controls_layout.addLayout(bottom_row)

        bottom_layout.addLayout(controls_layout, stretch=1)

        layout.addLayout(bottom_layout)

    def open_from_file(self):
        """Open an image from a file"""
        if self.siril is None:
            self.show_error("Not connected to Siril")
            return

        try:
            # Open file dialog
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Open Image File",
                "",
                "Image Files (*.fit *.fits *.fts *.bmp *.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)"
            )

            if not filename:
                return  # User cancelled

            with self.siril_mutex:
                # Load the image
                img = self.siril.load_image_from_file(filename)

                if img is None:
                    self.show_error(f"Failed to load image from {filename}")
                    return

                # Get autostretch preview if in Autostretch mode
                preview_data = None
                if self.display_mode == "Autostretch":
                    preview_data = self.siril.load_image_from_file(filename, preview=True, linked=self.link).data

            # Store the data and metadata
            self.stored_image = img
            self.stored_metadata = img.header
            self.stored_filename = filename.split('/')[-1]  # Get just the filename
            self.linked_params = None  # Reset cached MTF params
            self.unlinked_params = None  # Reset cached MTF params

            # Update display
            if preview_data is not None:
                # Use the preview directly for Autostretch
                self.image_label.set_image_data(preview_data, preserve_view=False)
            else:
                self.update_image_display(initial_load=True)

            self.setWindowTitle(f"Siril Image Holder - {self.stored_filename}")

            # Enable other buttons
            self.swap_with_siril_btn.setEnabled(True)
            self.copy_to_siril_btn.setEnabled(True)
            self.display_combo.setEnabled(True)

            # Precalculate MTF parameters in background
            self.precalculate_mtf_params()

        except Exception as e:
            self.show_error(f"Failed to open file: {str(e)}")

    def on_linked_changed(self, state):
        """Handle changes to the linked checkbox"""
        self.link = state == Qt.CheckState.Checked.value
        if "Autostretch" in self.display_mode:
            self.update_image_display(False)

    def update_linked_checkbox(self):
        """Update the linked checkbox to match self.link"""
        self.linked_checkbox.setChecked(self.link)

    def on_max_slider_changed(self, value: int):
        # Ensure max slider cannot go below min slider
        min_value = self.min_slider.value()
        if value < min_value:
            self.max_slider.blockSignals(True)
            self.max_slider.setValue(min_value)
            self.max_slider.blockSignals(False)
            value = min_value

        self.hi = value / 1000.0

    def on_minmax_slider_released(self):
        self.update_image_display(False)

    def on_min_slider_changed(self, value: int):
        # Ensure min slider cannot go above max slider
        max_value = self.max_slider.value()
        if value > max_value:
            self.min_slider.blockSignals(True)
            self.min_slider.setValue(max_value)
            self.min_slider.blockSignals(False)
            value = max_value

        self.lo = value / 1000.0

    def show_instructions(self) -> None:
        """Show the instructions dialog."""
        dialog = InstructionsDialog(self)
        dialog.exec()

    def connect_to_siril(self):
        """Connect to Siril"""
        try:
            with self.siril_mutex:
                self.siril = s.SirilInterface()
                self.siril.connect()
            self.copy_from_siril_btn.setEnabled(True)
            self.open_from_file_btn.setEnabled(True)
            self.sync_combo.setEnabled(True)

        except Exception as e:
            self.show_error(f"Failed to connect to Siril: {str(e)}")
            self.siril = None

    def check_siril_connection(self):
        """Periodically check if Siril is still connected"""
        if self.siril is None:
            self.connect_to_siril()
        # Note: sirilpy doesn't seem to have a direct connection check method
        # The connection will be tested when we try to use it

    def on_sync_mode_changed(self, mode_text):
        """Handle sync mode change from combo box"""
        self.sync_mode = mode_text

        if self.sync_mode != "None" and self.siril:
            # Reset previous values
            self.prev_siril_zoom = None
            self.prev_siril_xoff = None
            self.prev_siril_yoff = None
            self.prev_script_zoom = None
            self.prev_script_xoff = None
            self.prev_script_yoff = None

            # Initialise sync state to avoid jumps
            try:
                with self.siril_mutex:
                    if self.sync_mode == "Follow Siril":
                        # Get current Siril view for tracking initialization only
                        # Don't change the script view - let the sync worker handle it
                        xoff, yoff, zoom = self.siril.get_siril_panzoom()

                        # Initialize tracking with both Siril and Script current positions
                        self.prev_siril_zoom = zoom
                        self.prev_siril_xoff = xoff
                        self.prev_siril_yoff = yoff
                        self.prev_script_zoom = self.image_label._scale_factor
                        self.prev_script_xoff = self.image_label._image_pan_x
                        self.prev_script_yoff = self.image_label._image_pan_y

                    elif self.sync_mode == "Follow Script":
                        # Push current script view into Siril immediately
                        zoom = self.image_label._scale_factor
                        xoff = self.image_label._image_pan_x
                        yoff = self.image_label._image_pan_y
                        self.siril.set_siril_zoom(zoom)
                        self.siril.set_siril_pan(xoff, yoff)
                        self.prev_script_zoom = zoom
                        self.prev_script_xoff = xoff
                        self.prev_script_yoff = yoff
                        self.prev_siril_zoom = zoom
                        self.prev_siril_xoff = xoff
                        self.prev_siril_yoff = yoff
            except Exception as e:
                print(f"Failed to initialise sync: {str(e)}")

            # Start sync thread
            self.start_sync_thread()
        else:
            self.stop_sync_thread()

    def start_sync_thread(self):
        """Start the synchronization thread"""
        if self.sync_thread is not None:
            self.stop_sync_thread()

        # Create worker and thread
        self.sync_worker = SyncWorker(self.siril, self.siril_mutex)
        self.sync_thread = threading.Thread(target=self.sync_worker.start_polling, daemon=True)

        # Connect worker signals
        self.sync_worker.stf_changed.connect(self.update_display_combo_from_sync)
        self.sync_worker.zoom_pan_changed.connect(self.update_zoom_pan_from_sync)
        self.sync_worker.script_zoom_pan_changed.connect(self.update_siril_zoom_pan_from_script)

        # Start thread
        self.sync_thread.start()

    def stop_sync_thread(self):
        """Stop the synchronization thread"""
        if self.sync_worker:
            self.sync_worker.stop_polling()

        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=1.0)  # Wait up to 1 second

        self.sync_worker = None
        self.sync_thread = None

    def on_script_zoom_to_fit(self):
        """Handles the zoom-to-fit action and sends the special command to Siril"""
        if self.sync_mode == "Follow Script" and self.siril and self.stored_image:
            try:
                with self.siril_mutex:
                    # The special value -1.0 tells Siril to perform its own zoom-to-fit.
                    self.siril.set_siril_zoom(-1.0)

                # Update tracking variables to reflect the new state after zoom-to-fit
                self.prev_script_zoom = self.image_label._scale_factor
                self.prev_siril_zoom = -1.0 # Use a sentinel value to indicate zoom-to-fit state
                # print("INFO: Zoom-to-fit triggered. Sending -1.0 to Siril.")

            except Exception as e:
                print(f"Failed to sync zoom-to-fit to Siril: {str(e)}")

    def on_script_pan_changed(self):
            """Send script pan to Siril, handling independent Siril changes smoothly"""
            if self.sync_mode == "Follow Script" and self.siril and self.stored_image:
                try:
                    # Get current Siril pan and zoom to check for independent changes
                    with self.siril_mutex:
                        current_siril_xoff, current_siril_yoff, current_siril_zoom = self.siril.get_siril_panzoom()

                    # --- Handle independent Siril zoom change ---
                    if (self.prev_siril_zoom is not None and
                        abs(current_siril_zoom - self.prev_siril_zoom) > 0.01):
                        # Siril's zoom has changed. Reset tracking to avoid pan jumps.
                        self.prev_siril_zoom = current_siril_zoom
                        self.prev_siril_xoff = current_siril_xoff
                        self.prev_siril_yoff = current_siril_yoff
                        self.prev_script_zoom = self.image_label._scale_factor
                        self.prev_script_xoff = self.image_label._image_pan_x
                        self.prev_script_yoff = self.image_label._image_pan_y
                        # print("INFO: Detected independent Siril zoom change. Resetting sync state to avoid pan jumps.")
                        return # Exit to prevent an immediate pan jump

                    if (self.prev_siril_xoff is not None and
                        (abs(current_siril_xoff - self.prev_siril_xoff) > 0.5 or
                        abs(current_siril_yoff - self.prev_siril_yoff) > 0.5)):
                        # Siril's pan has changed. Re-baseline the sync state to Siril's current view.
                        self.prev_siril_xoff = current_siril_xoff
                        self.prev_siril_yoff = current_siril_yoff
                        self.prev_script_xoff = self.image_label._image_pan_x
                        self.prev_script_yoff = self.image_label._image_pan_y
                        # print("INFO: Detected independent Siril pan change. Resetting sync state to avoid pan jumps.")
                        return # Exit to prevent an immediate pan jump

                    # Proceed with a normal pan sync, calculating the delta
                    script_pan_delta_x = self.image_label._image_pan_x - self.prev_script_xoff
                    script_pan_delta_y = self.image_label._image_pan_y - self.prev_script_yoff

                    # Apply this delta to Siril's current pan position
                    new_siril_xoff = self.prev_siril_xoff + (script_pan_delta_x * self.prev_siril_zoom)
                    new_siril_yoff = self.prev_siril_yoff + (script_pan_delta_y * self.prev_siril_zoom)

                    # Send the new pan to Siril if there was a significant change
                    if (abs(script_pan_delta_x) > 0.5 or
                        abs(script_pan_delta_y) > 0.5):
                        with self.siril_mutex:
                            self.siril.set_siril_pan(new_siril_xoff, new_siril_yoff)

                        # Update our tracking variables for the next event
                        self.prev_script_xoff = self.image_label._image_pan_x
                        self.prev_script_yoff = self.image_label._image_pan_y
                        self.prev_siril_xoff = new_siril_xoff
                        self.prev_siril_yoff = new_siril_yoff

                except Exception as e:
                    print(f"Failed to sync script pan to Siril: {str(e)}")

    def on_script_zoom_changed(self):
            """Send script zoom to Siril, handling independent Siril changes smoothly"""
            if self.sync_mode == "Follow Script" and self.siril and self.stored_image:
                try:
                    # Get current Siril zoom
                    with self.siril_mutex:
                        _, _, current_siril_zoom = self.siril.get_siril_panzoom()

                    # Check if Siril's zoom has been changed independently
                    if (self.prev_siril_zoom is not None and
                        abs(current_siril_zoom - self.prev_siril_zoom) > 0.01):
                        # Independent change detected. Reset the tracking state.
                        self.prev_siril_zoom = current_siril_zoom
                        self.prev_script_zoom = self.image_label._scale_factor
                        # print("INFO: Detected independent Siril zoom change. Resetting zoom sync state.")
                        return  # Exit to prevent an immediate jump

                    # Calculate the proportional change in script zoom
                    script_zoom_ratio = self.image_label._scale_factor / self.prev_script_zoom

                    # Apply this ratio to Siril's current zoom level
                    new_siril_zoom = self.prev_siril_zoom * script_zoom_ratio

                    # Check for significant change before sending the update
                    if abs(new_siril_zoom - self.prev_siril_zoom) > 0.01:
                        with self.siril_mutex:
                            self.siril.set_siril_zoom(new_siril_zoom)

                        # Update our tracking variables for the next event
                        self.prev_script_zoom = self.image_label._scale_factor
                        self.prev_siril_zoom = new_siril_zoom

                except Exception as e:
                    print(f"Failed to sync script zoom to Siril: {str(e)}")

    @pyqtSlot(float, float, float)
    def update_siril_zoom_pan_from_script(self, zoom, xoff, yoff):
        """Update Siril's zoom and pan from script changes (for Follow Script modes)"""
        if self.sync_mode == "Follow Script" and self.siril:
            try:
                with self.siril_mutex:
                    # This slot is present for an alternate pathway if needed.
                    self.siril.set_siril_zoom(zoom)
                    self.siril.set_siril_pan(xoff, yoff)
            except Exception as e:
                print(f"Failed to sync zoom/pan to Siril: {str(e)}")

    @pyqtSlot(float, float, float)
    def update_zoom_pan_from_sync(self, zoom, xoff, yoff):
        """Apply Siril pan/zoom to script (Follow Siril) using relative changes"""
        if self.sync_mode == "Follow Siril" and self.stored_image is not None:
            if self.image_label._pixmap is not None:
                try:
                    # Detect if this is the first sync after mode change
                    if self.prev_siril_zoom is None:
                        # Initialize tracking - set script to match Siril exactly on first sync
                        self.image_label.blockSignals(True)
                        self.image_label._scale_factor = zoom
                        self.image_label._image_pan_x = xoff
                        self.image_label._image_pan_y = yoff
                        self.image_label.update_display()
                        self.image_label.blockSignals(False)

                        self.prev_siril_zoom = zoom
                        self.prev_siril_xoff = xoff
                        self.prev_siril_yoff = yoff
                        self.prev_script_zoom = zoom
                        self.prev_script_xoff = xoff
                        self.prev_script_yoff = yoff
                        return

                    # Calculate relative changes in Siril
                    zoom_ratio = zoom / self.prev_siril_zoom if self.prev_siril_zoom != 0 else 1.0
                    pan_delta_x = xoff - self.prev_siril_xoff
                    pan_delta_y = yoff - self.prev_siril_yoff

                    changed_zoom = abs(zoom_ratio - 1.0) > 0.001
                    changed_pan = abs(pan_delta_x) > 0.5 or abs(pan_delta_y) > 0.5

                    if changed_zoom or changed_pan:
                        self.image_label.blockSignals(True)

                        if changed_zoom:
                            # Apply relative zoom change to script
                            self.image_label._scale_factor *= zoom_ratio

                        if changed_pan:
                            # Apply relative pan change to script (accounting for script's current zoom)
                            # The pan delta from Siril is in Siril's zoom space, so we need to convert it
                            # to the script's zoom space
                            scale_adjustment = self.image_label._scale_factor / zoom if zoom != 0 else 1.0
                            self.image_label._image_pan_x += pan_delta_x * scale_adjustment
                            self.image_label._image_pan_y += pan_delta_y * scale_adjustment

                        self.image_label.update_display()
                        self.image_label.blockSignals(False)

                        # Update tracking variables
                        self.prev_siril_zoom = zoom
                        self.prev_siril_xoff = xoff
                        self.prev_siril_yoff = yoff
                        self.prev_script_zoom = self.image_label._scale_factor
                        self.prev_script_xoff = self.image_label._image_pan_x
                        self.prev_script_yoff = self.image_label._image_pan_y

                except Exception as e:
                    print(f"Failed to apply Siril zoom/pan to script: {str(e)}")

    @pyqtSlot(int, bool)
    def update_display_combo_from_sync(self, stf_value, linked):
        """Update display combo box from sync thread (slot for thread-safe signal)"""
        if self.sync_mode == "Follow Siril":
            # Block signals temporarily to avoid triggering on_display_mode_changed
            self.display_combo.blockSignals(True)
            self.display_combo.setCurrentIndex(stf_value)
            self.display_combo.blockSignals(False)
            self.linked_checkbox.blockSignals(True)
            old_link = self.link
            self.link = linked
            self.linked_checkbox.setChecked(linked)
            self.linked_checkbox.blockSignals(False)

            # Update display mode and refresh image
            mode_text = self.display_combo.currentText()
            if self.display_mode != mode_text or old_link != self.link:
                self.display_mode = mode_text
                self.update_image_display()

    def copy_from_siril(self):
        """Copy current image or sequence frame from Siril to the holder"""
        if self.siril is None:
            self.show_error("Not connected to Siril")
            return

        try:
            with self.siril_mutex:
                is_sequence = self.siril.is_sequence_loaded()
                is_image = self.siril.is_image_loaded()

                if not is_sequence and not is_image:
                    self.show_error("No image or sequence loaded in Siril")
                    return

                # Get image object with data and metadata
                if is_sequence:
                    seq = self.siril.get_seq()
                    current_frame = seq.current
                    img = self.siril.get_seq_frame(current_frame)
                    filename_suffix = f" (frame {current_frame})"
                else:
                    img = self.siril.get_image()
                    filename_suffix = ""

                if img is None:
                    self.show_error("Failed to get image data from Siril")
                    return

                self.lo, self.hi, slidermode = self.siril.get_siril_slider_state(float_range=True)

                # Get autostretch link state
                self.link = self.siril.get_siril_stf_linked()
                self.update_linked_checkbox()

                # Get filename if available
                try:
                    if is_sequence:
                        filename = self.siril.get_seq_frame_filename(current_frame+1)
                        # (corrects for 0-based vs 1-based sequence frame counting)
                        if not filename:
                            filename = f"Sequence frame {current_frame}"
                    else:
                        filename = self.siril.get_image_filename()
                        if not filename:
                            filename = "Siril Image"
                except:
                    filename = f"Sequence frame {current_frame}" if is_sequence else "Siril Image"

                # Get STF to match Siril if sync is not Follow Script
                if self.sync_mode not in ["Follow Script"]:
                    stf = self.siril.get_siril_stf()
                    self.display_combo.setCurrentIndex(stf) # Match Siril STF

                # If in Autostretch mode, get preview directly
                if self.display_mode == "Autostretch":
                    if is_sequence:
                        preview_data = self.siril.get_seq_frame(current_frame, preview=True, linked=self.link).data
                    else:
                        preview_data = self.siril.get_image_pixeldata(preview=True, linked=self.link).data
                else:
                    preview_data = None

            # Store the data, metadata, and preview
            self.stored_image = img  # Store the FFit object directly
            self.stored_metadata = img.header  # Header metadata
            self.stored_filename = filename
            self.linked_params = None # Reset any cached MTF params
            self.unlinked_params = None # Reset any cached MTF params

            # Update display based on current mode (initial load - scale to fit)
            if preview_data is not None:
                # Use the preview directly for Autostretch
                self.image_label.set_image_data(preview_data, preserve_view=False)
            else:
                self.update_image_display(initial_load=True)

            self.setWindowTitle(f"Siril Image Holder - {self.stored_filename}")

            # Enable other buttons
            self.swap_with_siril_btn.setEnabled(not is_sequence)  # Disable swap for sequences
            self.copy_to_siril_btn.setEnabled(True)
            self.display_combo.setEnabled(True)

            # Precalculate MTF parameters in background
            self.precalculate_mtf_params()

        except Exception as e:
            self.show_error(f"Failed to copy from Siril: {str(e)}")

    def swap_with_siril(self):
        """Swap the stored image with the current Siril image"""
        if self.siril is None:
            self.show_error("Not connected to Siril")
            return

        if self.stored_image is None:
            self.show_error("No image stored in holder")
            return

        try:
            with self.siril_mutex:
                # Get current Siril image object
                siril_img = self.siril.get_image()
                siril_metadata = None
                siril_filename = "Siril Image"
                siril_preview = None
                self.link = self.siril.get_siril_stf_linked()
                self.update_linked_checkbox()
                # Swap slider values
                lo, hi, slidermode = self.siril.get_siril_slider_state(float_range=True)
                self.lo = lo
                self.hi = hi
                lo = int(lo * 65535)
                hi = int(hi * 65535)
                self.siril.set_siril_slider_lohi(lo, hi)

                # Set STF to match Siril if sync is not Follow Script
                if self.sync_mode not in ["Follow Script"]:
                    siril_stf = self.siril.get_siril_stf()
                    our_stf_index = self.display_combo.currentIndex()
                    our_stf = STFType(our_stf_index)  # Convert index to STFType enum
                    self.display_combo.setCurrentIndex(siril_stf) # Match Siril STF
                else:
                    our_stf_index = self.display_combo.currentIndex()
                    our_stf = STFType(our_stf_index)  # Convert index to STFType enum

                if siril_img is not None:
                    siril_metadata = siril_img.header
                    # Get preview if we're in Autostretch mode
                    if self.display_mode == "Autostretch":
                        siril_preview = self.siril.get_image_pixeldata(preview=True, linked=self.link)
                    try:
                        siril_filename = self.siril.get_image_filename()
                        if not siril_filename:
                            siril_filename = "Siril Image"
                    except:
                        siril_filename = "Siril Image"

                # Send stored image to Siril (requires image lock)
                with self.siril.image_lock():
                    self.siril.set_image_pixeldata(self.stored_image.data)
                    if self.stored_metadata:
                        self.siril.set_image_metadata_from_header_string(self.stored_metadata)
                    # Set ICC profile if available
                    if hasattr(self.stored_image, 'icc_profile') and self.stored_image.icc_profile is not None:
                        self.siril.set_image_iccprofile(self.stored_image.icc_profile)
                    self.siril.set_siril_stf(our_stf)

                # Clear undo history after sending image
                self.siril.clear_undo_history()

            # Store what was in Siril
            if siril_img is not None:
                self.stored_image = siril_img
                self.linked_params = None # reset cached MTF params
                self.unlinked_params = None # reset cached MTF params
                self.stored_metadata = siril_metadata
                old_filename = self.stored_filename
                self.stored_filename = siril_filename

                # Update display with new image (important for size changes) - initial load
                if siril_preview is not None:
                    # Use the preview directly for Autostretch
                    self.image_label.set_image_data(siril_preview, preserve_view=False)
                else:
                    self.update_image_display(initial_load=True)

                self.setWindowTitle(f"Siril Image Holder - {self.stored_filename}")

                # Precalculate MTF parameters in background
                self.precalculate_mtf_params()

            else:
                # No image was in Siril, just clear our storage
                self.stored_image = None
                self.stored_metadata = None
                self.stored_filename = "No image"

                self.image_label.set_image_data(None)
                self.setWindowTitle("Siril Image Holder - No image")

                self.swap_with_siril_btn.setEnabled(False)
                self.copy_to_siril_btn.setEnabled(False)
                self.display_combo.setEnabled(False)

        except Exception as e:
            self.show_error(f"Failed to swap with Siril: {str(e)}")

    def copy_to_siril(self):
        """Copy the stored image to Siril"""
        if self.siril is None:
            self.show_error("Not connected to Siril")
            return

        if self.stored_image is None:
            self.show_error("No image stored in holder")
            return

        try:
            with self.siril_mutex:
                # Check if Siril has an image
                if self.siril.is_image_loaded():
                    current_siril_img = self.siril.get_image()

                    if current_siril_img is not None:
                        # Show confirmation dialog
                        reply = QMessageBox.question(
                            self,
                            'Confirm Copy',
                            'Siril already has an image open. Are you sure you want to replace it?',
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )

                        if reply != QMessageBox.StandardButton.Yes:
                            return
                else:
                    # No image is open. We need to create a new one to copy into
                    self.siril.cmd("new", "1", "1", "1", "new empty image")

                # Send image to Siril (requires image lock)
                with self.siril.image_lock():
                    self.siril.set_image_pixeldata(self.stored_image.data)
                    if self.stored_metadata:
                        self.siril.set_image_metadata_from_header_string(self.stored_metadata)
                    # Set ICC profile if available
                    if hasattr(self.stored_image, 'icc_profile') and self.stored_image.icc_profile is not None:
                        self.siril.set_image_iccprofile(self.stored_image.icc_profile)


                our_stf_index = self.display_combo.currentIndex()
                our_stf = STFType(our_stf_index)  # Convert index to STFType enum
                self.siril.set_siril_stf(our_stf)

                # Clear undo history after sending image
                self.siril.clear_undo_history()

            # print(f"INFO: Copied '{self.stored_filename}' to Siril")

        except Exception as e:
            self.show_error(f"Failed to copy to Siril: {str(e)}")

    def on_display_mode_changed(self, mode_text):
        """Handle display mode change from combo box"""
        self.display_mode = mode_text
        self.update_image_display()

        # Disable sliders in autostretch mode
        autostretch = "Autostretch" in self.display_mode
        histogram = "Histogram" in self.display_mode
        self.max_slider.setEnabled(not autostretch and not histogram)
        self.min_slider.setEnabled(not autostretch and not histogram)
        self.linked_checkbox.setEnabled(autostretch)

        # If sync is enabled, also update Siril's STF
        if self.sync_mode == "Follow Script" and self.siril:
            try:
                with self.siril_mutex:
                    current_index = self.display_combo.currentIndex()
                    stf_type = STFType(current_index)
                    self.siril.set_siril_stf(stf_type)
                    # Next line commented out till RC1 - I forgot to add the setter...
                    # self.siril.set_siril_stf_linked(self.link)
            except Exception as e:
                print(f"Failed to sync STF to Siril: {str(e)}")

    def apply_display_transform(self, image_data, mode):
        """Apply the specified display transform to image data"""
        if image_data is None:
            return None

        # Work with a copy to avoid modifying original data
        data = image_data.copy()
        # Convert uint16 to float in range 0-1 if needed
        if data.dtype == np.uint16:
            data = data.astype(np.float32) / 65535.0
        elif data.dtype != np.float32:
            # Assume other types are already in 0-1 range or convert them
            data = data.astype(np.float32)
            if data.max() > 1.0:
                data = data / data.max()

        # Apply ICC profile transformation for Linear mode if profile exists
        if (mode == "Linear" and
            hasattr(self.stored_image, 'icc_profile') and
            self.stored_image.icc_profile is not None and
            self.stored_image.icc_profile != self.display_icc_profile):
            try:
                # Create ICC profile objects from bytes
                source_profile = ImageCms.ImageCmsProfile(io.BytesIO(self.stored_image.icc_profile))
                display_profile = ImageCms.ImageCmsProfile(io.BytesIO(self.display_icc_profile))

                # Convert float32 [0,1] to uint8 [0,255] for Pillow
                temp_data = (np.clip(data, 0, 1) * 255).astype(np.uint8)

                # Handle different image dimensions
                if len(temp_data.shape) == 2:
                    pil_image = Image.fromarray(temp_data)
                elif len(temp_data.shape) == 3 and temp_data.shape[2] == 3:
                    pil_image = Image.fromarray(temp_data)
                elif len(temp_data.shape) == 3 and temp_data.shape[0] == 3:
                    temp_data = np.transpose(temp_data, (1, 2, 0))
                    pil_image = Image.fromarray(temp_data)
                elif len(temp_data.shape) == 3 and temp_data.shape[0] == 1:
                    temp_data = temp_data[0]
                    pil_image = Image.fromarray(temp_data)
                else:
                    raise ValueError(f"Unsupported image format for ICC transform: {temp_data.shape}")

                input_mode = 'RGB' if pil_image.mode == 'RGB' else 'L'
                output_mode = input_mode

                transform = ImageCms.buildTransform(
                    source_profile, display_profile,
                    input_mode, output_mode,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
                )

                transformed_pil = ImageCms.applyTransform(pil_image, transform)
                data = np.array(transformed_pil).astype(np.float32) / 255.0

            except Exception as e:
                print(f"WARNING: Failed to apply ICC profile transformation: {str(e)}")

        # Apply the transform based on mode
        if mode == "Histogram":
            data = apply_histogram_equalization(data)
        else:
            # Apply the transform based on mode using numba-accelerated functions
            delta = self.hi - self.lo
            if delta == 0:
                # When delta is 0, all normalized values would be 1.0, so result is all 255
                data = np.full_like(data, 255, dtype=np.uint8)
            else:
                if mode == "Linear":
                    data = self.linear_transform_combined(data, self.lo, delta)
                elif mode == "Square Root":
                    data = self.sqrt_transform_combined(data, self.lo, delta)
                elif mode == "Squared":
                    data = self.squared_transform_combined(data, self.lo, delta)
                elif mode == "Logarithm":
                    data = self.numba_log_transform_combined(data, self.lo, delta)
                elif mode == "Asinh":
                    data = self.asinh_transform_combined(data, self.lo, delta)
                elif mode == "Autostretch":
                    # Wait for background calculation if it's still running
                    if self.calc_thread is not None and self.calc_thread.is_alive():
                        self.calc_thread.join(timeout=5.0)  # Wait up to  seconds

                    if self.link:
                        if not self.linked_params:
                            self.linked_params = find_linked_midtones_balance(data, -2.8, 0.25)
                        data = MTFProcessor.apply_linked_mtf(data, self.linked_params, copy=False)
                    else:
                        if not self.unlinked_params:
                            self.unlinked_params = find_unlinked_midtones_balance(data, -2.8, 0.25)
                        data = MTFProcessor.apply_unlinked_mtf(data, self.unlinked_params, copy=False)
        return data

    @numba.vectorize([numba.uint8(numba.float32, numba.float32, numba.float32)],
                    target='parallel', fastmath=True, cache=True, nopython=True)
    def linear_transform_combined(pixel, lo, delta):
        """Combined linear transform with clipping and scaling to uint8"""
        # Normalize: (pixel - lo) / delta
        f1 = numba.float32(1.0)
        f0 = numba.float32(0.0)
        f255 = numba.float32(255.0)
        normalized = (pixel - lo) / delta
        # Clip to [0, 1] and scale to [0, 255]
        clipped = max(f0, min(f1, normalized))
        return numba.uint8(clipped * f255)

    @numba.vectorize([numba.uint8(numba.float32, numba.float32, numba.float32)],
                    target='parallel', fastmath=True, cache=True, nopython=True)
    def sqrt_transform_combined(pixel, lo, delta):
        """Combined sqrt transform with clipping and scaling to uint8"""
        f1 = numba.float32(1.0)
        f0 = numba.float32(0.0)
        f255 = numba.float32(255.0)
        sqrt_val = math.sqrt(max(pixel, f0))
        normalized = (sqrt_val - lo) / delta
        clipped = max(f0, min(f1, normalized))
        return numba.uint8(clipped * f255)

    @numba.vectorize([numba.uint8(numba.float32, numba.float32, numba.float32)],
                    target='parallel', fastmath=True, cache=True, nopython=True)
    def squared_transform_combined(pixel, lo, delta):
        """Combined squared transform with clipping and scaling to uint8"""
        f1 = numba.float32(1.0)
        f0 = numba.float32(0.0)
        f255 = numba.float32(255.0)
        squared_val = pixel * pixel
        normalized = (squared_val - lo) / delta
        clipped = max(f0, min(f1, normalized))
        return numba.uint8(clipped * f255)

    @numba.vectorize([numba.uint8(numba.float32, numba.float32, numba.float32)],
                    target='parallel', fastmath=True, cache=True, nopython=True)
    def asinh_transform_combined(pixel, lo, delta):
        """Combined asinh transform with clipping and scaling to uint8"""
        f1 = numba.float32(1.0)
        f0 = numba.float32(0.0)
        f255 = numba.float32(255.0)
        scale_val = numba.float32(65.535)
        asinh_scale = math.asinh(scale_val)
        asinh_val = math.asinh(pixel * scale_val) / asinh_scale
        normalized = (asinh_val - lo) / delta
        clipped = max(f0, min(f1, normalized))
        return numba.uint8(clipped * f255)

    @numba.vectorize([numba.uint8(numba.float32, numba.float32, numba.float32)],
                    target='parallel', fastmath=True, cache=True, nopython=True)
    def numba_log_transform_combined(pixel, lo, delta):
        """Combined log transform with clipping and scaling to uint8"""
        f1 = numba.float32(1.0)
        f0 = numba.float32(0.0)
        f255 = numba.float32(255.0)
        log_val = numba.float32(1.5259022e-4)
        log_inv_scale = numba.float32(8.7877545)
        if pixel > log_val:
            transformed_val = math.log(pixel / log_val) / log_inv_scale
        else:
            transformed_val = pixel
        normalized = (transformed_val - lo) / delta
        clipped = max(f0, min(f1, normalized))
        return numba.uint8(clipped * f255)

    def update_image_display(self, initial_load=False):
        """Update the image display based on current display mode

        Args:
            initial_load: If True, scale to fit; if False, preserve current view
        """
        if self.stored_image is None:
            self.image_label.set_image_data(None)
            return

        # Determine whether to preserve view or scale to fit
        preserve_view = not initial_load

        transformed_data = self.apply_display_transform(self.stored_image.data, self.display_mode)
        self.image_label.set_image_data(transformed_data, preserve_view=preserve_view)

    def show_error(self, message: str):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        print(f"ERROR: {message}")

    def show_info(self, message: str):
        """Show info message"""
        QMessageBox.information(self, "Information", message)
        # print(f"INFO: {message}")

    def closeEvent(self, event):
        """Handle application close"""
        # Stop sync thread before closing
        self.stop_sync_thread()

        if self.siril:
            try:
                with self.siril_mutex:
                    self.siril.disconnect()
            except:
                pass
        event.accept()

    def precalculate_mtf_params(self):
        """Calculate MTF parameters in a background thread"""
        if self.stored_image is None:
            return

        # If a calculation is already running, don't start another
        if self.calc_thread is not None and self.calc_thread.is_alive():
            return

        def calculate_params():
            try:
                data = self.stored_image.data.copy()
                # Convert to float32 if needed
                if data.dtype == np.uint16:
                    data = data.astype(np.float32) / 65535.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)
                    if data.max() > 1.0:
                        data = data / data.max()

                # Calculate in order of priority based on link state
                if self.link:
                    # Calculate linked first if linked mode is active
                    self.linked_params = find_linked_midtones_balance(data, -2.8, 0.25)
                    self.unlinked_params = find_unlinked_midtones_balance(data, -2.8, 0.25)
                else:
                    # Calculate unlinked first if unlinked mode is active
                    self.unlinked_params = find_unlinked_midtones_balance(data, -2.8, 0.25)
                    self.linked_params = find_linked_midtones_balance(data, -2.8, 0.25)

            except Exception as e:
                print(f"Failed to precalculate MTF parameters: {str(e)}")
            finally:
                # Reset thread reference when done
                self.calc_thread = None

        # Start calculation in a daemon thread
        self.calc_thread = threading.Thread(target=calculate_params, daemon=True)
        self.calc_thread.start()

## Optimized autostretch implementation

# MAD normalization constant (approximation for normal distribution)
MAD_NORM = 1.4826

@dataclass
class MTFParams:
    """Structure to hold MTF (Midtone Transfer Function) parameters"""
    shadows: float
    midtones: float
    highlights: float

class MTFProcessor:
    """High-performance MTF processor for image enhancement"""

    @staticmethod
    def apply_linked_mtf(
        image: np.ndarray,
        params: MTFParams,
        copy: bool = True
    ) -> np.ndarray:
        """
        Apply linked MTF transformation to an image using vectorized operations.

        Args:
            image: Input image as float32 array (c, h, w) with values in [0, 1]
            params: MTF parameters
            copy: If True, create a copy; if False, modify in place

        Returns:
            Transformed image
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Handle single channel case
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]

        c, h, w = image.shape

        if copy:
            result = np.empty_like(image)
        else:
            result = image

        for ch in range(c):
            result[ch, :, :] = _apply_mtf_vectorized(
                image[ch, :, :],
                params.shadows,
                params.midtones,
                params.highlights
            )

        return result.squeeze() if result.shape[0] == 1 else result

    @staticmethod
    def apply_unlinked_mtf(
        image: np.ndarray,
        params_list: list[MTFParams],
        copy: bool = True
    ) -> np.ndarray:
        """
        Apply unlinked MTF transformation to an image using vectorized operations.

        Args:
            image: Input image as float32 array (c, h, w) with values in [0, 1]
            params_list: List of MTF parameters, one per channel
            copy: If True, create a copy; if False, modify in place

        Returns:
            Transformed image
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Handle single channel case
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]

        c, h, w = image.shape

        if copy:
            result = np.empty_like(image)
        else:
            result = image

        # Ensure we have enough parameters
        if len(params_list) < c:
            default_param = MTFParams(shadows=0.0, midtones=0.5, highlights=1.0)
            params_list = list(params_list) + [default_param] * (c - len(params_list))

        for ch in range(c):
            params = params_list[ch]
            result[ch, :, :] = _apply_mtf_vectorized(
                image[ch, :, :],
                params.shadows,
                params.midtones,
                params.highlights
            )

        return result.squeeze() if result.shape[0] == 1 else result

@jit(nopython=True, fastmath=True, cache=True)
def _find_percentile(data: np.ndarray, percentile: float) -> float:
    """
    Fast histogram-based percentile computation matching the original C implementation
    """
    size = data.size
    if size == 0:
        return 0.0

    # Find min and max values
    min_val = np.min(data)
    max_val = np.max(data)

    if abs(max_val - min_val) == 0.0:
        return min_val

    # Use smaller histogram for small data sizes, max 65536
    histo_size = min(65536, size)

    # Calculate scale factor to use full range of histogram
    scale = (histo_size - 1) / (max_val - min_val)

    # Create histogram
    histo = np.zeros(histo_size, dtype=np.uint32)

    # Fill histogram
    for i in range(size):
        # Scale data to histogram range
        bin_idx = int(scale * (data.flat[i] - min_val))
        # Ensure we don't exceed histogram bounds
        bin_idx = min(bin_idx, histo_size - 1)
        histo[bin_idx] += 1

    # Find percentile
    thresh = percentile * size
    count = 0
    k = 0

    # Find the bin where we exceed the threshold
    while count < thresh and k < histo_size:
        count += histo[k]
        k += 1

    if k > 0:
        # Interpolate for more accurate result
        count_prev = count - histo[k - 1]
        c0 = count - thresh
        c1 = thresh - count_prev

        if c0 + c1 > 0:
            result = (c1 * k + c0 * (k - 1)) / (c0 + c1)
        else:
            result = k - 1
    else:
        result = 0

    # Convert back to original range
    result = result / scale + min_val

    # Clamp to valid range
    return max(min_val, min(max_val, result))

@jit(nopython=True, fastmath=True, cache=True)
def _compute_histogram_median(data: np.ndarray) -> float:
    """Fast histogram-based median computation using the original algorithm"""
    return _find_percentile(data, 0.5)

@jit(nopython=True, fastmath=True, cache=True)
def _compute_mad(data: np.ndarray, median: float) -> float:
    """Compute Median Absolute Deviation using histogram method"""
    abs_devs = np.abs(data - median)
    return _compute_histogram_median(abs_devs)

@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _compute_channel_stats(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute median and MAD for each channel in parallel"""
    c, h, w = image.shape
    medians = np.zeros(c, dtype=np.float32)
    mads = np.zeros(c, dtype=np.float32)

    for ch in prange(c):
        channel_data = image[ch, :, :].ravel()
        median = _compute_histogram_median(channel_data)
        mad = _compute_mad(channel_data, median)
        medians[ch] = median
        mads[ch] = mad

    return medians, mads

@jit(nopython=True, fastmath=True, cache=True)
def _mtf_function(x: float, m: float, lo: float, hi: float) -> float:
    """Midtone Transfer Function - original implementation"""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0

    xp = (x - lo) / (hi - lo)
    return ((m - 1.0) * xp) / (((2.0 * m - 1.0) * xp) - m)

# Vectorized MTF application for maximum performance
@vectorize(['float32(float32, float32, float32, float32)'], target='parallel', cache=True, fastmath=True,
           nopython=True)
def _apply_mtf_vectorized(x, shadows, midtones, highlights):
    """Vectorized MTF application using numba's parallel vectorization"""
    return _mtf_function(x, midtones, shadows, highlights)

def find_linked_midtones_balance(
    image: np.ndarray,
    shadows_clipping: float,
    target_bg: float
) -> Union[MTFParams, None]:
    """
    Find linked midtones balance parameters for all channels together.

    Args:
        image: Float32 numpy array of shape (c, h, w) with values in [0, 1]
        shadows_clipping: Shadow clipping parameter
        target_bg: Target background level

    Returns:
        MTFParams object with shadows, midtones, highlights values or None on error
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Handle single channel case
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]

    c, h, w = image.shape

    try:
        # Compute statistics for all channels
        medians, mads = _compute_channel_stats(image)

        # Check for inverted channels (median > 0.5)
        inverted_channels = np.sum(medians > 0.5)

        if inverted_channels < c:
            # Normal image processing
            c0 = 0.0
            m = 0.0

            for i in range(c):
                median = medians[i]
                mad = mads[i] * MAD_NORM

                # Guard against breakdown point
                if mad == 0.0:
                    mad = 0.001

                c0 += median + shadows_clipping * mad
                m += median

            c0 /= c
            c0 = max(0.0, c0)  # Clamp to >= 0

            m2 = m / c - c0

            return MTFParams(
                shadows=c0,
                midtones=_mtf_function(m2, target_bg, 0.0, 1.0),
                highlights=1.0
            )
        else:
            # Inverted image processing
            c1 = 0.0
            m = 0.0

            for i in range(c):
                median = medians[i]
                mad = mads[i] * MAD_NORM

                # Guard against breakdown point
                if mad == 0.0:
                    mad = 0.001

                m += median
                c1 += median - shadows_clipping * mad

            c1 /= c
            c1 = min(1.0, c1)  # Clamp to <= 1

            m2 = c1 - m / c

            return MTFParams(
                shadows=0.0,
                midtones=1.0 - _mtf_function(m2, target_bg, 0.0, 1.0),
                highlights=c1
            )

    except Exception:
        # Return default values on error
        return MTFParams(shadows=0.0, midtones=0.2, highlights=1.0)

def find_unlinked_midtones_balance(
    image: np.ndarray,
    shadows_clipping: float,
    target_bg: float
) -> Union[list[MTFParams], None]:
    """
    Find unlinked midtones balance parameters for each channel independently.

    Args:
        image: Float32 numpy array of shape (c, h, w) with values in [0, 1]
        shadows_clipping: Shadow clipping parameter
        target_bg: Target background level

    Returns:
        List of MTFParams objects, one per channel, or None on error
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Handle single channel case
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]

    c, h, w = image.shape

    try:
        # Compute statistics for all channels
        medians, mads = _compute_channel_stats(image)

        # Check for inverted channels
        inverted_channels = np.sum(medians > 0.5)

        results = []

        if inverted_channels < c:
            # Normal image processing
            for i in range(c):
                median = medians[i]
                mad = mads[i] * MAD_NORM

                # Guard against breakdown point
                if mad == 0.0:
                    mad = 0.001

                c0 = median + shadows_clipping * mad
                c0 = max(0.0, c0)  # Clamp to >= 0

                m2 = median - c0

                results.append(MTFParams(
                    shadows=c0,
                    midtones=_mtf_function(m2, target_bg, 0.0, 1.0),
                    highlights=1.0
                ))
        else:
            # Inverted image processing
            for i in range(c):
                median = medians[i]
                mad = mads[i] * MAD_NORM

                # Guard against breakdown point
                if mad == 0.0:
                    mad = 0.001

                c1 = median - shadows_clipping * mad
                c1 = min(1.0, c1)  # Clamp to <= 1

                m2 = c1 - median

                results.append(MTFParams(
                    shadows=0.0,
                    midtones=1.0 - _mtf_function(m2, target_bg, 0.0, 1.0),
                    highlights=c1
                ))

        return results

    except Exception:
        # Return default values on error
        return [MTFParams(shadows=0.0, midtones=0.2, highlights=1.0) for _ in range(c)]

@jit(nopython=True, cache=True)
def _compute_histogram_bins(data: np.ndarray, num_bins: int = 65536) -> np.ndarray:
    """Compute histogram bins for float32 data in range [0, 1]"""
    histogram = np.zeros(num_bins, dtype=np.uint32)

    for i in range(data.size):
        pixel = data.flat[i]
        # Clamp to [0, 1] range
        if pixel < 0.0:
            pixel = 0.0
        elif pixel > 1.0:
            pixel = 1.0

        # Convert to bin index
        bin_idx = int(pixel * (num_bins - 1))
        histogram[bin_idx] += 1

    return histogram

@jit(nopython=True, cache=True)
def _build_equalization_lut(histogram: np.ndarray, total_pixels: int) -> np.ndarray:
    """Build lookup table for histogram equalization"""
    num_bins = histogram.shape[0]
    lut = np.zeros(num_bins, dtype=np.uint8)

    hist_sum = 0
    for i in range(num_bins):
        hist_sum += histogram[i]
        # Map cumulative distribution to output range [0, 255]
        lut[i] = np.uint8(round((hist_sum / total_pixels) * 255.0))

    return lut

@jit(nopython=True, cache=True, fastmath=True)
def _apply_histogram_equalization(channel_data, lut):
    h, w = channel_data.shape
    result = np.empty((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            pixel = channel_data[i, j]
            if pixel < 0.0:
                pixel = 0.0
            elif pixel > 1.0:
                pixel = 1.0

            bin_idx = int(pixel * (len(lut) - 1))
            result[i, j] = lut[bin_idx]

    return result

def apply_histogram_equalization(data: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to image data

    Args:
        data: Float32 array in range [0, 1], shape (h, w) or (c, h, w)

    Returns:
        uint8 array with histogram equalization applied
    """
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Handle single channel case
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]

    c, h, w = data.shape
    result = np.empty((c, h, w), dtype=np.uint8)

    # Process each channel independently
    for ch in range(c):
        channel_data = data[ch, :, :]

        # Compute histogram
        histogram = _compute_histogram_bins(channel_data)

        # Build equalization lookup table
        total_pixels = h * w
        lut = _build_equalization_lut(histogram, total_pixels)

        # Apply equalization
        result[ch, :, :] = _apply_histogram_equalization(channel_data, lut)

    return result.squeeze() if result.shape[0] == 1 else result

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Siril Image Holder")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = SirilImageHolder()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
