#
# ***********************************************
#
# Copyright (C) 2025 - Carlo Mollicone - AstroBOH
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The author of this script is Carlo Mollicone (CarCarlo147) and can be reached at:
# https://www.astroboh.it
# https://www.facebook.com/carlo.mollicone.9
#
# ***********************************************
#
# --------------------------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------------------------
#
# Description:
# This Siril script provides an interface to remove satellite trails from images.
# Trails can be filled with black pixels or with pixels from a reference image.
#
#
# Features:
# - Spline tracing with multiple management.
# - Stretched preview from Siril.
# - Mask with adjustable blur.
# - Progressive blend (soft edges) with controllable strength.
# - Filling with:
#       Black
#       Background median
#       Reference image
# - Visual overlay for mask confirmation.
#
#
# Versions:
# 0.0.1 - Initial release
# 0.0.5 - Stop manually handling the offset and rely completely on tkinter's internal methods
#         for managing coordinates on the Canvas, which take both zooming and panning into account.
#       - Create "extra space" so that the drag area is larger than the image itself,
#         so you can move the image all the way to the edges of the preview pane and still have room to maneuver.
#       - Fixed fit_to_preview because it didn't explicitly recenter the view within this new spacing
#         and the image was not centered in the canvas.
#       - Added instructions and improved the UI.
# 0.0.6 - Integration of the DAOPHOT MMM (Mean, Median, Mode) algorithm,
#         a robust astronomical standard for estimating the sky background while avoiding outliers, via the library: photutils
#       - Change "Black fill" with full black no blend
# 0.1.0 - Many improvements, New Features, Bug Fixes & Improvements
# 0.1.1 - Minor additions
# 0.2.0 - Big improvements, new features, bug fixes
#       - add auto-stretch for detection
#       - add AI detection button
#       - add Canny Edge Detection
#       - add Hough Transform for trail detection
#       - add debug parameters for canny and hough
#       - add DBSCAN clustering for trail grouping
# 0.2.1 - Bug Fixes
# 0.2.2 - Converting instructions to labels for a smoother GUI.
#         Update the instructions. Clearer and more complete.
# 1.0.0 - Better filedialog for Linux
# 1.0.1 - Added contact information
# 1.0.2 - Various improvements
# 1.0.3 - New process_detected_lines with:
#           - Optional corner filtering (uses segment orientation to avoid merging between segments with very different directions),
#           - DBSCAN grouping using position + direction
#           - New edge stretching.
#           - Added Very Low AI sensitivity parameter
# 1.0.4 - Adds the ability to use the script with a loaded sequence
# 1.0.5 - Various improvements
#           The introduction of the new pre-filtering function, which uses morphological operations and contour analysis
#           to filter out stars before running the Hough transform step makes the entire detection pipeline
#           much more robust and less prone to being confused by dense star fields.
# 2.0.0 - Version PyQt6
# 2.0.1 - Added Icon App
# 2.0.2 - Fixed CFA pattern detection
# 2.0.3 - Improved UI layout for Mode and Reference file display
# 2.0.4 - (AKB) updated set_seq_frame_pixeldata() call for compatibility with sirilpy update
#

VERSION = "2.0.3"

# --- Core Imports ---
import sys
import math
import os
import shutil
import base64

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=0.6.37'):
        print("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    # Import Siril GUI related components
    from sirilpy import SirilError
    from sirilpy.models import FPoint

    s.ensure_installed("PyQt6", "numpy", "astropy", "opencv-python", "photutils", "pillow", "scikit-learn")

    # --- PyQt6 Imports ---
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QMainWindow, QDialog, QMessageBox, QFileDialog, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QStyle, QSizePolicy, QSlider, QListWidget, QRadioButton, QButtonGroup,
        QCheckBox, QTextEdit, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QGraphicsPixmapItem,
        QGraphicsEllipseItem, QSplitter, QScrollArea, QGroupBox, QSpinBox, QDoubleSpinBox, QGridLayout,
        QLineEdit, QFrame
    )
    from PyQt6.QtGui import (
        QPixmap, QIcon, QImage, QPainterPath, QAction, QPainter, QBrush, QPen, QColor, QFont, QCloseEvent,
        QMouseEvent, QWheelEvent, QKeyEvent
    )
    from PyQt6.QtCore import (
        Qt, QPointF, QRectF, QTimer, pyqtSignal, QLineF
    )

    # --- Imports for Image Processing ---
    import cv2
    import numpy as np

    from astropy.io import fits
    from photutils.background import MMMBackground, StdBackgroundRMS
    from PIL import Image, ImageDraw, ImageFilter
    from PIL.ImageQt import ImageQt
    from sklearn.cluster import DBSCAN

except ImportError:
    print("Warning: sirilpy not found. The script is not running in the Siril environment.")

# --- Interpolation Function ---
def get_curve_points(points, tension=0.5, num_of_segments=16):
    """
    Calculates the points of a cardinal spline passing through the given points.
    """
    if not points or len(points) < 2:
        return []

    pts = [coord for p in points for coord in p]
    res = []

    _pts = pts[:]
    _pts.insert(0, pts[1])
    _pts.insert(0, pts[0])
    _pts.append(pts[-2])
    _pts.append(pts[-1])

    for i in range(2, len(_pts) - 4, 2):
        for t in range(num_of_segments + 1):
            st = t / num_of_segments

            t1x = (_pts[i+2] - _pts[i-2]) * tension
            t2x = (_pts[i+4] - _pts[i]) * tension
            t1y = (_pts[i+3] - _pts[i-1]) * tension
            t2y = (_pts[i+5] - _pts[i+1]) * tension

            c1 = 2 * (st**3) - 3 * (st**2) + 1
            c2 = -2 * (st**3) + 3 * (st**2)
            c3 = (st**3) - 2 * (st**2) + st
            c4 = (st**3) - (st**2)

            x = c1 * _pts[i] + c2 * _pts[i+2] + c3 * t1x + c4 * t2x
            y = c1 * _pts[i+1] + c2 * _pts[i+3] + c3 * t1y + c4 * t2y
            res.append((x, y))

    return res

def compute_mmm_background(image):
    """
    Computes the sky background value using the DAOPHOT MMM method.
    If the image has 3 channels (RGB), returns an array for each channel.
    """
    mmm = MMMBackground()   # use its default parameters for box_size, sigma_clip, etc., as per the documentation
    
    if image.ndim == 3 and image.shape[2] == 3:  # RGB
        return np.array([mmm(image[..., c]) for c in range(3)])
    else:  # Monochrome or single channel
        return mmm(image)

def pack_rgba_color(r, g, b, a):
    """
    Packs RGBA values into a single 32-bit integer.
    The byte order required by Siril is (A << 24) | (B << 16) | (G << 8) | R.
    """
    return (a << 24) | (b << 16) | (g << 8) | r

def create_thick_line_polygon(points, width):
    """
    Creates a list of FPoint objects that define a closed polygon
    representing a thick line that follows the given points.
    """
    if len(points) < 2:
        return []

    polygon_points = []
    half_width = width / 2.0

    # Calculate the outline points
    left_side = []
    right_side = []

    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])

        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue

        perpendicular = np.array([-direction[1], direction[0]])
        normal = perpendicular / norm

        if i == 0:
            left_side.append(p1 + normal * half_width)
            right_side.append(p1 - normal * half_width)

        left_side.append(p2 + normal * half_width)
        right_side.append(p2 - normal * half_width)

    # Combine the two sides and convert to FPoint objects
    full_outline = left_side + right_side[::-1]

    # Return a list of FPoint objects, as required by sirilpy
    return [FPoint(x=p[0], y=p[1]) for p in full_outline]

# --- Trail Management Classes ---
class trail:
    def __init__(self, trail_id):
        self.id = trail_id
        self.points = []
        self.line_width = 30    # Default Trail width in pixels
        self.spline_points = []
        self.color = "#FF0000"  # Red for the active trail

    def add_point(self, x, y, is_drawing_new=False):
        self.points.append((x, y))
        if not is_drawing_new:
            self.reorder_points()
        self.update_spline()

    def remove_point_at(self, index):
        if 0 <= index < len(self.points):
            del self.points[index]
            self.update_spline()

    def update_spline(self):
        self.spline_points = get_curve_points(self.points)

    def reorder_points(self):
        if len(self.points) < 3:
            return

        # Find the two furthest points
        max_dist = 0
        p1_idx, p2_idx = -1, -1
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                dist = math.hypot(self.points[i][0] - self.points[j][0], self.points[i][1] - self.points[j][1])
                if dist > max_dist:
                    max_dist = dist
                    p1_idx, p2_idx = i, j

        start_point = self.points[p1_idx]
        self.points.sort(key=lambda p: math.hypot(p[0] - start_point[0], p[1] - start_point[1]))

class TrailCollection:
    def __init__(self):
        self.trail = []
        self.active_trail_idx = -1
        self._next_id = 0

    def add_trail(self):
        new_trail = trail(self._next_id)
        self.trail.append(new_trail)
        self._next_id += 1
        self.set_active(len(self.trail) - 1)
        return new_trail

    def remove_trail(self, index):
        if 0 <= index < len(self.trail):
            del self.trail[index]
            if self.active_trail_idx >= index:
                self.active_trail_idx -= 1
            if self.active_trail_idx < 0 and self.trail:
                self.active_trail_idx = 0
            self.update_trail_colors()

    def set_active(self, index):
        """
        Sets the active trail by its index. If index is -1, no trail is active.
        """
        if -1 <= index < len(self.trail): # Allow -1 to mean no active trail
            self.active_trail_idx = index
            self.update_trail_colors()
        else: # Handle out-of-bounds or invalid indices for safety
            self.active_trail_idx = -1
            self.update_trail_colors() # Ensure colors are updated even if invalid index was passed

    def get_active_trail(self):
        if self.active_trail_idx != -1 and self.trail:
            return self.trail[self.active_trail_idx]
        return None

    def update_trail_colors(self):
        # Active Red, Inactive Yellow
        for i, trail in enumerate(self.trail):
            # If no trail is active (self.active_trail_idx is -1), all will become yellow
            trail.color = "#FF0000" if i == self.active_trail_idx else "#FFFF00"

class DeselectableListWidget(QListWidget):
    """
    A custom QListWidget that deselects the current item when an empty area is clicked.
    """
    def mousePressEvent(self, event: QMouseEvent):
        # The itemAt(position) method returns the item under the cursor, or None if there is no item.
        item = self.itemAt(event.pos())
        if item is None:
            self.setCurrentRow(-1)
        # It is essential to call the original base class method
        # so as not to lose standard behavior (e.g., selecting an item).
        super().mousePressEvent(event)

class ZoomPanGraphicsView(QGraphicsView):
    """
    QGraphicsView subclass that supports mouse-wheel zoom centered under the mouse
    and basic click/drag callbacks into the owning app.
    """
    def __init__(self, scene, owner):
        super().__init__(scene)
        self.owner = owner
        # Zoom rendering and anchoring settings
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setMouseTracking(True)

        # Disable native pan to manage it manually
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

        # State variables for manual panning
        self._is_panning = False
        self._pan_start_pos = QPointF()

    def wheelEvent(self, event: QWheelEvent):
        """ Zoom with the mouse wheel """
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        # If right click is pressed, cancel any drawing operation
        if event.button() == Qt.MouseButton.RightButton:
            self.owner.cancel_drawing()
            event.accept() # Indicates that we have handled the event
            return
        
        # Start a mouse action only if it is the left button
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan_start_pos = event.position()
            # Pass the event to the main handler to start moving a control point
            # or to prepare for a click
            self.owner.on_view_mouse_press(event, scene_pos)
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        scene_pos = self.mapToScene(event.position().toPoint())
        
        # Always notify the main application of the mouse position
        # This is necessary to update the rubberband in real time.
        self.owner.on_view_mouse_move(event, scene_pos)

        # If the left button is pressed, we handle the movement
        if event.buttons() & Qt.MouseButton.LeftButton:
            # If we are already moving a control point, do not start the pan
            if self.owner.is_moving_point():
                return # Let the main handler move the point

            # If we are not moving, we check if the distance is enough to start the pan
            if not self._is_panning and (event.position() - self._pan_start_pos).manhattanLength() > QApplication.startDragDistance():
                self._is_panning = True
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            
            if self._is_panning:
                # Use QGraphicsView.translate for smooth panning
                delta = event.position() - self._pan_start_pos
                self.translate(delta.x(), delta.y())
                # Update the starting position for smoother panning
                self._pan_start_pos = event.position()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # If we were panning, we reset the state
            if self._is_panning:
                self._is_panning = False
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            else:
                # If it wasn't a pan (the distance traveled was minimal),
                # then it was a click or the release of a control point.
                scene_pos = self.mapToScene(event.position().toPoint())
                self.owner.on_view_mouse_release(event, scene_pos)
        
        super().mouseReleaseEvent(event)

class HelpWindow(QDialog):
    """
    A dialog box that displays formatted help text.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Program Instructions")
        self.resize(750, 600)

        instructions_text = (
            f"<h2>Satellite Trail Removal Tool v{VERSION} - (c) Carlo Mollicone AstroBOH</h2>"
            "<h3>Workflow and Usage</h3>"
            "<p>Trail removal can be performed on CFA or RGB images.<br>"
            "It is highly recommended to perform trail removal during the image registration/alignment phase.<br>"
            "This is for two reasons:<br>"
            "- First, with the new algorithm, it is easier to recognize and distinguish satellite trails on RGB images.<br>"
            "- Second, if you wish to fill the trail with pixels from the previous or next frame, the images will be perfectly aligned, resulting in a perfect correction.</p>"
            
            "<h3>Trail Management</h3>"
            "<p><b>Add</b>: Creates a new manual trail and activates <b>drawing mode</b>, allowing you to add points directly on the canvas:<br>"
            "- <b>1. Place the first point</b>: Click on the image where you want the trail to start.<br>"
            "- <b>2. Add subsequent points</b>: A dashed guide line will now follow your mouse cursor. Click again to add the second point, third, and so on, defining the curve of the trail. The spline will update in real-time.<br>"
            "- <b>3. Finalize drawing</b>: To exit drawing mode, press the <b>right mouse button</b> or the <b>ESC key</b> on your keyboard.<br><br>"
            "<b>Duplicate</b>: Select a trail and click to create an identical copy, useful for working on variations.<br>"
            "<b>Remove</b>: Select a trail from the list to delete it.<br>"
            "<b>Deselect</b>: Removes the selection from any active trails.</p>"
            "<p><b>Line Width (px)</b>: Sets the thickness of the stroke. When a stroke is created by the AI, this value is calculated automatically, but can be changed manually by selecting the stroke and clicking <b>Update Width</b>.</p>"

            "<h3>Control Point Management</h3>"
            "<p>After adding or selecting a Trail, you can:<br>"
            "<b>Add Points</b>: Click at the desired position along the existing line to insert a new control point.<br>"
            "<b>Move Points</b>: Click and Drag an existing point to reposition it.<br>"
            "<b>Remove Points</b>: Hold down <b>Ctrl</b> and click on a point to remove it.</p>"
            "<p>Note: The script uses a <b>cardinal spline</b> to create a smooth curve between points.</p>"

            "<h3>Automatic Detection (AI)</h3>"
            "<p><b>Find Trail (AI)</b>: Starts the automatic detection algorithm. <b>Warning: All existing trails will be deleted.</b> The script will analyze the image and automatically create trails for each detected satellite or aircraft.</p>"
            "<p><b>AI Sensitivity</b>: These presets adjust the algorithm's core parameters. Due to the wide variety of astronomical images (e.g., monochrome vs. RGB, noise levels), there is no single <b>perfect</b> setting. It is recommended to start with a lower sensitivity and increase it if the trails are not detected.<br>"
            "- <b>Very Low</b>: The most conservative setting. Use this as a starting point for very complex or noisy images, such as full-color RGB images with dense star fields. It is designed to minimize false positives.<br>"
            "- <b>Low</b>: A selective setting, also suitable for RGB images or monochrome images with clear, well-defined trails. It offers a good balance between detection and noise rejection.<br>"
            "- <b>Medium</b>: The recommended default for most situations, particularly for monochrome (CFA) images. It is effective at finding moderately bright trails.<br>"
            "- <b>High</b>: The most aggressive setting. Use this when other presets fail to detect very faint or fragmented trails. Warning: This setting is much more likely to produce false positives by misidentifying star alignments or other noise as trails.</p>"
            "<p><b>Note:</b> Not all trails may be recognized, especially extremely faint ones. Experimentation is key to finding the best result.</p>"

            "<p><b>AI Detection Tuning</b>: This section for advanced users allows you to manually adjust key parameters of the Canny (for edges) and Hough (for lines) algorithms, giving you complete control over the detection process. The sensitivity presets automatically adjust these values.</p>"

            "<h3>Parameters for fusion</h3>"
            "<p><b>Blur</b> and <b>Blend</b>: Controls the blur and blend strength of the correction mask.</p>"

            "<h3>Correction Actions</h3>"
            "<p>Once you are satisfied with the trail tracing, you can apply the correction:<br>"
            "<b>Apply (Background)</b>: Fills the trail areas with the <b>DAOPHOT MMM (Mean, Median, Mode) algorithm</b> of the image. Useful for blending the correction with the surrounding star field.<br>"
            "<b>Apply (Black)</b>: Fills the trail areas with <b>black pixels</b>.<br>"
            "<b>Select Ref.</b>: Click this button to select a <b>reference image</b> (a FITS file) to use for filling. The reference image will be brightness-balanced with the current image before application.<br>"
            "<b>Apply (Ref.)</b>: Fills the trail areas using pixels from the <b>selected reference image</b>. The dimensions of the reference image must match the current image in Siril.</p>"
            "<p><b>Important</b>: After clicking one of the <b>Apply</b> buttons, the change will be applied directly to the image in Siril. An <b>undo state</b> will also be saved in Siril (<b>Trail Removal</b>) to allow you to revert the change if needed.</p>"

            "<h3>Trail Preview</h3>"
            "<p><b>Preview Trail</b>: Click this button to display a <b>red transparent overlay</b> of the created mask directly in the Siril interface. This helps you confirm that your trails have been traced correctly before applying the correction.<br>"
            "<b>Clear Overlay</b>: Removes the preview overlay from Siril.</p>"

            "<h3>Restore Backup</h3>"
            "<p>This button is only available when working on an <b>image sequence</b>.<br>"
            "It allows you to undo the last '<b>Apply</b>' correction by restoring the original frame from the backup file (e.g., <b>filename-original_with_trail.fit</b>) that was created automatically.<br>"
            "This function serves as a manual '<b>Undo</b>', since Siril's native Undo operation is not available for sequences.</p>"

            "<h3>Reload Image from Siril</h3>"
            "<p><b>Reload Image from Siril</b>: Click this button to reload the image (and its preview) from Siril. This is useful if you've made changes to the image in Siril after opening the script and want the script to work with the latest version. All traced trails in the script will be cleared as the new image might be different.</p>"

            "<h3>Zoom and Display</h3>"
            "<b>Pan</b>: Click and drag an empty area to move the image.<br>"
            "The <b>Zoom Out</b> and <b>Zoom In</b> buttons allow you to zoom in or out of the image.<br>"
            "The <b>Fit to view</b> button adjusts the image to fit the canvas window size.<br>"
            "You can also use the <b>mouse wheel</b> for zooming (on Windows/macOS) or <b>Button-4 / Button-5</b> (on Linux).</p>"

            "<h3>Closing the Script</h3>"
            "<p>Clicking the <b>X</b> button on the window will close the script. If you sent a preview (<b>Preview Trail</b>) but did not apply a real correction, the preview overlay will be automatically removed from Siril.</p>"
        )

        # Create a QTextEdit widget (equivalent to ScrolledText)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)  # Equivalent to state='disabled'
        self.text_edit.setHtml(instructions_text)

        # Set up a layout to handle scaling
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

class IconTextButton(QPushButton):
    """
    A custom QPushButton that uses an internal layout and has a proper
    size policy to behave correctly within toolbars and other layouts.
    """
    def __init__(self, text, icon=None, parent=None):
        super().__init__(parent)
        
        # Set the size policy to match a standard button.
        # This prevents the button from expanding greedily and compressing others.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # The internal layout for aligning the icon and text
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0) # left, top, right, bottom
        layout.setSpacing(5)

        # Icon Label
        icon_label = QLabel()
        if icon:
            pixmap = icon.pixmap(self.iconSize())
            icon_label.setPixmap(pixmap)
        
        # Text Label
        text_label = QLabel(text)

        # Add widgets to the internal layout
        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(text_label)
        layout.addStretch(1)
        
        self.setLayout(layout)

# --- Main Application Class ---
class TrailRemovalAPP(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Satellite Trail Removal Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        # --- State Variables ---
        self.trail_collection = TrailCollection()
        self.mouse_is_pressed = False
        self.moving_point_index = None
        self.reference_image_path = None
        self.drag_data = {"x": 0, "y": 0, "item": None, "trail": None, "point_idx": -1}
        self.image_stats = None                 # Contains a dictionary e.g.: {'median': [r,g,b], 'sigma': [r,g,b]}
        self.current_seq_name = None            # String with the name of the sequence
        self.current_frame_index = None         # Index of the current frame
        self.is_drawing_new_segment = False
        self.new_segment_anchor = None
        # Debug variables
        self.ui_debug = True
        self.ui_ai_tuning = True
        self.show_track_check = False
        self.show_stretch_check = False
        self.help_window = None # To keep a reference to the help window

        # Internal AI_Sensitivity names/IDs
        self.ai_sensitivity_Verylow = "V"
        self.ai_sensitivity_low = "L"
        self.ai_sensitivity_mid = "M"
        self.ai_sensitivity_max = "H"

        # Associate IDs with names to display to the user.
        self.ai_Sensitivity_display_names = {
            self.ai_sensitivity_Verylow: "V.Low",
            self.ai_sensitivity_low: "Low",
            self.ai_sensitivity_mid: "Mid",
            self.ai_sensitivity_max: "Max"
        }

        # Associate tooltips with IDs, not display text.
        self.ai_Sensitivity_tooltips = {
            self.ai_sensitivity_Verylow: "Extremely selective: ignores most faint trails, ideal for dense star fields or very noisy images",
            self.ai_sensitivity_low: "Only strong trails are detected (very safe)",
            self.ai_sensitivity_mid: "Balanced detection with good noise rejection",
            self.ai_sensitivity_max: "Detects faint trails (may include noise)"
        }

        # --- Siril Connection ---
        # Initialize Siril connection
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            QMessageBox.critical(self, "Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
            QTimer.singleShot(0, self.close)
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            QTimer.singleShot(0, self.close)
            return

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if seq_loaded:
            # If a sequence is loaded, set the current sequence name and frame index
            seq = self.siril.get_seq()
            self.current_seq_name = seq.seqname
            self.current_frame_index = seq.current

            self.siril.log(
                f"Load frame: {self.current_frame_index + 1}/{seq.number} from sequence: '{self.current_seq_name}' ",
                s.LogColor.BLUE
            )
        elif image_loaded:
            self.current_seq_name = None
            self.current_frame_index = None
            self.siril.log("Loaded single image.", s.LogColor.BLUE)
        else:
            self.siril.error_messagebox("No image or sequence loaded")
            # QTimer.singleShot(0, ...) doesn't run the command instantly.
            # Instead, it places the command (self.close) into the event queue to be executed after a 0 millisecond delay.
            # This means it will run as soon as the main event loop (qapp.exec()) starts and is ready to process events.
            QTimer.singleShot(0, self.close) 
            return

        # --- GUI Setup ---
        self.create_widgets()
        self.center_window()
        self.show_loading_message()

        if seq_loaded:
            # --- Ora posso aggiornare le Info Etichetta ---
            self.current_frame_label.setText(f"Sequence Mode - Current Frame {self.current_frame_index + 1} / {seq.number}")
        elif image_loaded:
            self.current_frame_label.setText("Single Image Mode")

        # This does not block __init__ and allows the window to appear immediately.
        QTimer.singleShot(500, self._load_image_data_and_preview)

    def show_loading_message(self):
        """ Show a "Loading..." text in the center of the canvas. """
        if hasattr(self, 'pixmap_item') and self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None
        self.scene.setBackgroundBrush(QBrush(Qt.GlobalColor.black))
        text_item = self.scene.addText("Loading image...\n\nand calculating initial image statistics...", QFont("Arial", 16))
        text_item.setDefaultTextColor(QColor("white"))
        self.view.fitInView(text_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _load_image_data_and_preview(self):
        """ This new feature contains all the slow code that previously blocked startup. """
        try:
            # Load full-depth data for processing
            self.full_image_data = self.siril.get_image_pixeldata(preview=False)
            if self.full_image_data is None:
                raise SirilError("No image loaded in Siril.")

            # Store the original dtype
            self.original_image_dtype = self.full_image_data.dtype
            
            # Convert full_image to HWC if necessary
            if len(self.full_image_data.shape) == 3 and self.full_image_data.shape[0] in [1, 3]:
                self.full_image_data = self.full_image_data.transpose(1, 2, 0)

            self._calculate_image_statistics()
            
            self.siril.log("Generating custom visual preview...", s.LogColor.BLUE)
            # preview_data = self.autostretch(self.full_image_data, detection=self.show_stretch_check.isChecked())

            # self.preview_pil_image = Image.fromarray(preview_data)
            self.preview_pil_image = self._create_visual_preview()

            # Flip vertically to align with Siril's coordinate system
            self.preview_pil_image = self.preview_pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            # initially the canvas points to the preview
            self.update_canvas_image(self.preview_pil_image)
            
            self.siril.overlay_clear_polygons()

            # Defer the first fit until the Canvas has valid dimensions
            QTimer.singleShot(50, self.fit_to_preview)

        except (SirilError) as e:
            self.siril.log(f"Error - Cannot start script: {e}", s.LogColor.RED)
            # QTimer.singleShot(0, ...) doesn't run the command instantly.
            # Instead, it places the command (self.close) into the event queue to be executed after a 0 millisecond delay.
            # This means it will run as soon as the main event loop (qapp.exec()) starts and is ready to process events.
            QTimer.singleShot(0, self.close)
            return

    def create_widgets(self):
        main_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # --- Control Panel (Left) ---
        left_panel_container = QWidget()
        # Vertical layout to organize the controls on the left
        left_panel_layout = QVBoxLayout(left_panel_container)
        left_panel_layout.setContentsMargins(5, 5, 5, 5) # Similar to padding

        # # To make the panel scrollable, QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel_container)

        main_splitter.addWidget(scroll_area)

        self.mode_layout = QGridLayout()
        self.mode_layout.setContentsMargins(0, 0, 0, 5) # Manteniamo il margine inferiore

        label_mode = QLabel("Mode : ")
        self.mode_layout.addWidget(label_mode, 0, 0) # Riga 0, Colonna 0

        self.current_frame_label = QLabel("Single Image Mode")
        self.current_frame_label.setObjectName("current_frame_label")
        self.current_frame_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Aggiunge l'etichetta del frame alla stessa riga (0), colonna 1
        self.mode_layout.addWidget(self.current_frame_label, 0, 1, Qt.AlignmentFlag.AlignLeft)
        self.mode_layout.setColumnStretch(2, 1)  # Colonna 2 prende tutto lo spazio rimanente

        # Aggiungi il layout (invece del groupbox) direttamente al pannello sinistro
        left_panel_layout.addLayout(self.mode_layout)

        # --- Trail Management ---
        # QGroupBox to group the controls
        trail_group = QGroupBox("Trail Management")
        trail_layout = QVBoxLayout(trail_group)
        left_panel_layout.addWidget(trail_group)

        top_container_layout = QHBoxLayout()
        
        self.trail_listbox = DeselectableListWidget()  # Use our new custom class
        self.trail_listbox.setFixedHeight(120)
        self.trail_listbox.currentItemChanged.connect(self.on_trail_select)

        top_container_layout.addWidget(self.trail_listbox, 1) # 1 = stretch factor

        # Container for line width options
        width_options_group = QWidget()
        width_options_layout = QVBoxLayout(width_options_group)
        
        width_options_layout.addWidget(QLabel("Line Width (px):"))
        # QSpinBox for integer values
        self.line_width_spinbox = QSpinBox()
        self.line_width_spinbox.setRange(1, 1000)
        self.line_width_spinbox.setSingleStep(5)
        self.line_width_spinbox.setValue(30) # Valore di default
        width_options_layout.addWidget(self.line_width_spinbox)
        
        update_width_btn = QPushButton("Update Width")
        update_width_btn.setToolTip("Apply the current width to the selected trail.")
        update_width_btn.clicked.connect(self.update_selected_trail_width)
        width_options_layout.addWidget(update_width_btn)
        width_options_layout.addStretch() # Push controls up

        top_container_layout.addWidget(width_options_group)
        trail_layout.addLayout(top_container_layout)

        # Management buttons
        btn_layout = QHBoxLayout()

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_trail)
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self.duplicate_trail)
        rem_btn = QPushButton("Remove")
        rem_btn.clicked.connect(self.remove_trail)
        des_btn = QPushButton("Deselect")
        des_btn.clicked.connect(lambda: self.trail_listbox.setCurrentRow(-1))
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(dup_btn)
        btn_layout.addWidget(rem_btn)
        btn_layout.addWidget(des_btn)
        
        trail_layout.addLayout(btn_layout)

        # AI Detection
        ai_find_frame = QWidget() # QWidget as a simple container
        ai_find_layout = QHBoxLayout(ai_find_frame)
        ai_find_layout.setContentsMargins(0,0,0,0)

        find_button = QPushButton("Find\nTrails (AI)")
        find_button.clicked.connect(self.auto_detect_trails)
        find_button.setToolTip("Automatically detect satellite trails using Hough Transform.")
        find_button.setProperty("class", "accent")
        ai_find_layout.addWidget(find_button)
  
        # AI Sensitivity with RadioButton
        ai_sensitivity_group = QGroupBox("AI Sensitivity")
        ai_sensitivity_layout = QHBoxLayout(ai_sensitivity_group)
        
        self.radio_buttons_ai = {}
        # Dictionary to keep track of radio buttons
        for sensitivity_id, display_name in self.ai_Sensitivity_display_names.items():
            rb = QRadioButton(display_name)
            rb.setToolTip(self.ai_Sensitivity_tooltips.get(sensitivity_id, ""))
            rb.toggled.connect(self.update_ai_tuning_parameters)
            ai_sensitivity_layout.addWidget(rb)
            self.radio_buttons_ai[sensitivity_id] = rb
        
        ai_find_layout.addWidget(ai_sensitivity_group)
        trail_layout.addWidget(ai_find_frame)

        # --- AI Tuning ---
        if self.ui_ai_tuning:
            tuning_group = QGroupBox("AI Detection Tuning")
            tuning_layout = QGridLayout(tuning_group) # QGridLayout for aligned forms
            
            # Helper function to create sliders
            def add_slider(row, text, range_min, range_max, default_val, tooltip):
                label = QLabel(f"{text}:")
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(range_min, range_max)
                slider.setValue(default_val)
                value_label = QLabel(str(default_val))
                value_label.setMinimumWidth(30) # To prevent the UI from "jumping"
                slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
                slider.setToolTip(tooltip)

                tuning_layout.addWidget(label, row, 0)
                tuning_layout.addWidget(slider, row, 1)
                tuning_layout.addWidget(value_label, row, 2)
                return slider
            
            # Creating sliders
            self.canny_thresh1_slider = add_slider(0, "Canny Low", 1, 255, 10, "Lower threshold for Canny edge detection.")
            self.canny_thresh2_slider = add_slider(1, "Canny High", 1, 255, 30, "Upper threshold for Canny edge detection.")
            self.hough_thresh_slider = add_slider(2, "Hough Thresh", 10, 200, 90, "Minimum number of edge points to detect a line.")
            self.hough_min_len_slider = add_slider(3, "Hough MinLen", 10, 200, 90, "Minimum length (in pixels) of a detected line.")
            self.hough_max_gap_slider = add_slider(4, "Hough MaxGap", 1, 100, 30, "Maximum allowed gap (in pixels) between segments.")

            debug_checks_layout = QHBoxLayout()
            self.show_canny_check = QCheckBox("Show Canny Edges")
            self.show_hough_check = QCheckBox("Show Hough Lines")
            self.show_stretch_check = QCheckBox("Show Stretched")
            debug_checks_layout.addWidget(self.show_canny_check)
            debug_checks_layout.addWidget(self.show_hough_check)
            # debug_checks_layout.addWidget(self.show_stretch_check)    # only for debug
            
            # layout_griglia.addLayout(what_to_add, starting_row, starting_column, how_many_rows_to_occupy, how_many_columns_to_occupy)
            tuning_layout.addLayout(debug_checks_layout, 5, 0, 1, 3)
            
            left_panel_layout.addWidget(tuning_group)

            # Set the default radio button
            self.radio_buttons_ai[self.ai_sensitivity_mid].setChecked(True)

        # Actions Frame
        action_group = QGroupBox("Correction Actions")
        action_layout = QVBoxLayout(action_group)
        
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Blur:"))
        self.blur_spinbox = QDoubleSpinBox()
        self.blur_spinbox.setRange(0.0, 10.0)
        self.blur_spinbox.setSingleStep(0.1)
        self.blur_spinbox.setValue(1.0)
        self.blur_spinbox.setToolTip("0 = Minimal blur, 10 = Maximum blur.")
        params_layout.addWidget(self.blur_spinbox)
        
        params_layout.addWidget(QLabel("Blend:"))
        self.blend_spinbox = QDoubleSpinBox()
        self.blend_spinbox.setRange(0.0, 1.0)
        self.blend_spinbox.setSingleStep(0.05)
        self.blend_spinbox.setValue(1.0)
        self.blend_spinbox.setToolTip("0 = Full blend, 1 = No blend.")
        params_layout.addWidget(self.blend_spinbox)
        action_layout.addLayout(params_layout)

        # Apply buttons in a grid for perfect alignment
        fill_btn_layout = QGridLayout()

        btn_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        btn_background = IconTextButton("Apply (Background)", btn_icon, self)
        btn_background.setToolTip("Using the DAOPHOT MMM (Mean, Median, Mode) algorithm,\napply a synthetic background with the same noise intensity.")
        btn_background.clicked.connect(lambda: self.apply_changes(mode="background"))
        
        btn_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        btn_black = IconTextButton("Apply (Black)", btn_icon, self)
        btn_black.setToolTip("Apply pure black to the image to exclude those pixels from stacking\n(you can still use blending with a strength lower than 1 to soften the black).")
        btn_black.clicked.connect(lambda: self.apply_changes(mode="black"))
        
        btn_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogStart)
        ref_button = IconTextButton("Select Ref.", btn_icon, self)
        ref_button.setToolTip("Select a 'previous or 'subsequent' frame that does not contain 'satellite' or 'airplane trails'.\nIf the 'tracking' was good, there should be no 'shift' between adjacent frames, therefore the 'pixel transfer' will be perfect.")
        ref_button.clicked.connect(self.select_reference)
        
        btn_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        apply_ref_button = IconTextButton("Apply (Ref.)", btn_icon, self)
        apply_ref_button.setToolTip("The pixels from the reference image will be inserted within the region defined by the trace,\nrespecting the Blur and Blend settings.")
        apply_ref_button.clicked.connect(self.apply_reference)
        
        fill_btn_layout.addWidget(btn_background, 0, 0)
        fill_btn_layout.addWidget(btn_black, 0, 1)
        fill_btn_layout.addWidget(ref_button, 1, 0)
        fill_btn_layout.addWidget(apply_ref_button, 1, 1)
        action_layout.addLayout(fill_btn_layout)

        info_Reference_layout = QGridLayout()
        info_Reference_layout.addWidget(QLabel("Reference File:"), 1, 0)
        self.ref_file_label = QLineEdit("None")
        self.ref_file_label.setReadOnly(True)
        self.ref_file_label.setToolTip("No reference file selected.")
        self.ref_file_label.setFrame(False)
        self.ref_file_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        info_Reference_layout.addWidget(self.ref_file_label, 1, 1)
        
        action_layout.addLayout(info_Reference_layout)
        
        left_panel_layout.addWidget(action_group)

        # --- Trail Preview ---
        preview_group = QGroupBox("Trail Preview")
        preview_layout = QHBoxLayout(preview_group)

        preview_trail_button = QPushButton("Preview Trail")
        preview_trail_button.setToolTip("Send the trail drawing to siril so that we can evaluate its correct positioning.")
        preview_trail_button.clicked.connect(self.send_preview_overaly_to_siril)

        clear_overlay_button = QPushButton("Clear Overlay")
        clear_overlay_button.setToolTip("Deletes the overlay sent to siril. Does not delete traces drawn on the canvas.")
        clear_overlay_button.clicked.connect(self.clear_preview_to_siril)
        
        preview_layout.addWidget(preview_trail_button)
        preview_layout.addWidget(clear_overlay_button)
        left_panel_layout.addWidget(preview_group)

        # Add a spacer at the end to push everything up
        left_panel_layout.addStretch(1)

        # --- Image Panel (Right) ---
        right_panel_widget = QWidget()
        right_layout = QVBoxLayout(right_panel_widget)
        main_splitter.addWidget(right_panel_widget)

        # Zoom controls and actions
        top_right_layout = QHBoxLayout()

        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        fit_btn = QPushButton("Fit to view")
        fit_btn.clicked.connect(self.fit_to_preview)
        top_right_layout.addWidget(zoom_in_btn)
        top_right_layout.addWidget(zoom_out_btn)
        top_right_layout.addWidget(fit_btn)
        top_right_layout.addStretch(1) # Push the next buttons to the right

        restore_button = QPushButton("   Restore Backup   ")
        restore_button.setToolTip("Restores the original frame from the backup file\ncreated during the 'Apply' process\n\n(for sequences only).")
        restore_button.clicked.connect(self.restore_backup_frame)
        # restore_button.setObjectName("restoreButton")
        restore_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOkButton))
        
        reload_button = QPushButton("   Reload Image from Siril   ")
        reload_button.clicked.connect(self.load_new_image_from_siril)
        # reload_button.setObjectName("reloadButton")
        reload_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        
        help_button = QPushButton("   HELP   ")
        help_button.clicked.connect(self.show_help)
        help_button.setObjectName("helpButton")
        help_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogHelpButton))
        
        top_right_layout.addWidget(restore_button)
        top_right_layout.addWidget(reload_button)
        top_right_layout.addWidget(help_button)
        
        right_layout.addLayout(top_right_layout)

        # Replacing the Canvas with QGraphicsView for advanced zoom/pan
        self.scene = QGraphicsScene(self)
        self.view = ZoomPanGraphicsView(self.scene, self)
        self.view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )
        # Improve smoothness during pan/zoom
        self.view.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) # Force pan mode
        self.view.setBackgroundBrush(QBrush(Qt.GlobalColor.black))
        right_layout.addWidget(self.view)
        
        # --- Creating the Rubber Band ---
        self.rubber_band_line = QGraphicsLineItem()
        # Let's create a pen for the line: white, 1 pixel thick, dashed style
        pen = QPen(Qt.GlobalColor.white, 5, Qt.PenStyle.DotLine)
        pen.setCosmetic(True) # Maintains 1px thickness regardless of zoom
        self.rubber_band_line.setPen(pen)
        self.rubber_band_line.setZValue(100) # Make sure it's always on top
        self.rubber_band_line.setVisible(False) # Initially invisible
        self.scene.addItem(self.rubber_band_line)

        # Set the splitter aspect ratio
        main_splitter.setSizes([450, 800])

    def center_window(self):
        """ Center window using PyQt methods """
        screen_geometry = self.screen().availableGeometry()
        self.resize(1200, 750)
        self.move(
            int((screen_geometry.width() - self.width()) / 2),
            int((screen_geometry.height() - self.height()) / 2)
        )

    def show_help(self):
        """
        Create and display the help window.
        We keep a reference to it to prevent it from being deleted by the garbage collector.
        """
        # Check if a help window is already open
        if self.help_window is None:
            self.help_window = HelpWindow(self)
            # When the window is closed, reset the reference
            self.help_window.finished.connect(self.on_help_window_closed)
        
        self.help_window.show()
        self.help_window.activateWindow() # Bring window to front
    
    def on_help_window_closed(self):
        """ Slot to reset the reference to the help window. """
        self.help_window = None
        
    def _calculate_image_statistics(self):
        """
        Calculates and stores statistics (median, sigma) for the loaded image.
        It handles both color and monochrome images.

        Compute robust sky background statistics using photutils MMMBackground for each channel
        Why It's an Improvement
        Robustness against Outliers: The current method, np.median(data), calculates the median over the entire image. If the image contains large nebulae or bright galaxies,
        these objects will heavily influence the median value, distorting the estimate of the "true" sky background. The photutils library is specifically designed
        for astrophotography, and its algorithms are much more robust, as they are able to ignore (or "clip") outliers such as stars and other bright objects during the calculation.

        More Accurate Noise Estimation: As a result, StdBackgroundRMS provides a much more accurate estimate of the standard deviation (sigma) than background noise alone.
        The manual method based on MAD (Median Absolute Deviation) is good, but less precise when applied globally to a complex astronomical image.

        Better Contrast for Detection: The goal of the autostretch function is to maximize the contrast of trails.
        Using a more precise median and sigma that are more representative of the sky background, the black and white points of the stretch will be set much more effectively.
        This will make the faint trails "emerge" better from the background, improving the effectiveness of the Canny and Hough algorithms that follow.
        """
        self.siril.log("Calculating initial image statistics...", s.LogColor.BLUE)
        if self.full_image_data is None:
            self.image_stats = None
            return

        data_for_stats = self.full_image_data.astype(np.float32)

        bkg_estimator = MMMBackground()
        bkg_rms = StdBackgroundRMS()

        # Check if the image is in color (RGB)
        if data_for_stats.ndim == 3 and data_for_stats.shape[2] == 3:
            # Calculates median and sigma for each channel and stores them
            medians = [bkg_estimator(data_for_stats[..., c]) for c in range(3)]
            sigmas = [bkg_rms(data_for_stats[..., c]) for c in range(3)]
            self.image_stats = {'median': np.array(medians), 'sigma': np.array(sigmas)}
        else:
            # The image is monochrome
            median = bkg_estimator(data_for_stats)
            sigma = bkg_rms(data_for_stats)
            self.image_stats = {'median': median, 'sigma': sigma}
        
        stats = self.image_stats
        formatted_stats = "\n\nImage statistics calculated:\n"

        def format_array(arr: np.ndarray, decimals: int = 8):
            return '[' + ', '.join(f"{x:.{decimals}g}" for x in arr) + ']'

        if isinstance(stats['median'], np.ndarray):
            formatted_stats += f"median: array{format_array(stats['median'])}, dtype={stats['median'].dtype}\n"
            formatted_stats += f"sigma:  array{format_array(stats['sigma'])}, dtype={stats['sigma'].dtype}"
        else:
            formatted_stats += f"median: {stats['median']:.8g}\n"
            formatted_stats += f"sigma:  {stats['sigma']:.8g}"

        self.siril.log(formatted_stats + "\n", s.LogColor.GREEN)

    def autostretch(self, image_data, detection=False):
        """
        Performs a robust autostretch on the linear image to create
        a balanced visual preview. For color images, it performs
        an unlinked stretch to maintain proper color balance.

        Performs robust autostretch.
        - If detection=False, creates a balanced visual preview.
        - If detection=True, performs advanced preprocessing on the linear data to isolate satellite trails before returning an 8-bit image.
        """
        # Work on float data for calculations
        data = image_data.copy().astype(np.float32)

        stats_are_per_channel = isinstance(self.image_stats['median'], np.ndarray)

        # --- Detection Section ---
        if detection:
            if data.ndim == 3 and data.shape[2] == 3:   # Color image (RGB)
                self.siril.log("RGB IMAGE for Detection Section", s.LogColor.RED)

                stretched_channels = []
                for i in range(3):  # Iterate over R, G, B channels
                    channel = data[..., i]
                    median = self.image_stats['median'][i]
                    sigma = self.image_stats['sigma'][i]
                    black_point = median + 1.0 * sigma  # Set black point above median to suppress background
                    white_point = median + 10.0 * sigma
                    channel = np.clip(channel, black_point, white_point)
                    median_filtered = cv2.medianBlur(channel, 5)
                    stretched_channel = cv2.normalize(median_filtered, None, 0, 255, cv2.NORM_MINMAX)
                    stretched_channels.append(stretched_channel)

                # Recombines channels into a temporary color image
                color_image = cv2.merge(stretched_channels).astype(np.uint8)
                # Convert the final color image to black and white (grayscale)
                stretched_8bit = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
                
            else:   # Monochrome image
                self.siril.log("MONO IMAGE for Detection Section", s.LogColor.RED)

                # Apply stretch to monochrome image
                median = self.image_stats['median']
                sigma = self.image_stats['sigma']
                black_point = median - 0.5 * sigma
                white_point = median + 5.0 * sigma
                data = np.clip(data, black_point, white_point)
                median_filtered = cv2.medianBlur(data, 5)
                stretched_8bit = cv2.normalize(median_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return stretched_8bit

        # --- Visual Preview Section ---
        else:
            if data.ndim == 3 and data.shape[2] == 3:   # Color image (RGB)
                self.siril.log("RGB IMAGE", s.LogColor.BLUE)
                stretched_channels = []
                for i in range(3):  # Iterate over R, G, B channels
                    channel = data[..., i]

                    # If stats_are_per_channel is True (native RGB image), perform a proper unlinked stretch using separate statistics for each channel (median[i], sigma[i]).
                    # If stats_are_per_channel is False (demosaiced CFA image), effectively perform a linked stretch, because it uses the same median and sigma scalar values ​​for all three channels.
                    median = self.image_stats['median'][i] if stats_are_per_channel else self.image_stats['median']
                    sigma = self.image_stats['sigma'][i] if stats_are_per_channel else self.image_stats['sigma']

                    # Applica lo stretch (identico a quello MONO, ma per canale)
                    # Imposta il nero sotto la mediana, il fondo cielo (che è a median) non sarà nero (0), ma un grigio scuro.
                    # Questo è fondamentale per vedere il segnale debole che emerge dal rumore.
                    black_point = median - 1.5 * sigma

                    # Questo crea uno stretch molto più "dolce".
                    # Mappa la gamma da (median - 2.5*s) a (median + 8.0*s) sull'intero display 0-255.
                    # Questo dà contrasto e dettaglio alle strutture deboli (la traccia) senza appiattirle a bianco puro.
                    white_point = median + 8.0 * sigma

                    # Apply the stretch to a single channel
                    # "Clipping" the values to isolate the range of interest
                    channel = np.clip(channel, black_point, white_point)
                    
                    median_filtered = cv2.medianBlur(channel, 5)

                    # Normalizes the channel to use the entire 0-255 range
                    stretched_channel = cv2.normalize(median_filtered, None, 0, 255, cv2.NORM_MINMAX)
                    stretched_channels.append(stretched_channel)

                stretched_8bit = cv2.merge(stretched_channels).astype(np.uint8)

            else:   # Monochrome image
                # Apply stretch to monochrome image
                self.siril.log("MONO IMAGE", s.LogColor.BLUE)
                median = self.image_stats['median']
                sigma = self.image_stats['sigma']
                black_point = median - 1.5 * sigma
                white_point = median + 5.0 * sigma
                data = np.clip(data, black_point, white_point)
                median_filtered = cv2.medianBlur(data, 5)
                stretched_8bit = cv2.normalize(median_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            return stretched_8bit

    def _create_visual_preview(self):
        """
        Generates a high-quality 8-bit preview image. Normalizes input data format
        and, if the source is a CFA image, performs on-the-fly demosaicing,
        white balancing, AND a custom stretch for correct visualization.
        """
        if self.full_image_data.ndim == 3 and self.full_image_data.shape[0] in [1, 3]:
            if self.full_image_data.shape[0] == 1:
                image_for_stretch = self.full_image_data[0]
            else:
                image_for_stretch = self.full_image_data.transpose(1, 2, 0)
        else:
            image_for_stretch = self.full_image_data

        # --- Demosaicing Logic (if needed) ---
        # Now let's check the image shape *after* normalization
        if image_for_stretch.ndim == 2:
            cfa_pattern = None
            header = None
            try:
                self.siril.log(f"Reading header from header FITS", s.LogColor.BLUE)
                if self.current_seq_name is not None:
                    # We are in sequence mode
                    header = self.siril.get_seq_frame_header(self.current_frame_index, return_as='dict')
                else:
                    # We are in single image mode
                    header = self.siril.get_image_fits_header(return_as='dict')

                if header:
                    # 1. Ottieni il valore grezzo (potrebbe essere 'RGGB ' o None)
                    cfa_pattern_raw = header.get('BAYERPAT') or header.get('CFAIMAG')

                    # 2. Controlla SE hai ottenuto un valore prima di pulirlo
                    if cfa_pattern_raw:
                        cfa_pattern = cfa_pattern_raw.strip() # Ora è sicuro chiamare .strip()
                    else:
                        cfa_pattern = None # Era None e rimane None

            except (s.SirilError, s.NoImageError) as e:
                # If we can't read the header, we treat it as mono
                self.siril.log(f"Could not retrieve FITS header: {e}", s.LogColor.RED)
                cfa_pattern = None
            
            bayer_patterns = {
                "RGGB": cv2.COLOR_BAYER_RG2RGB,
                "GBRG": cv2.COLOR_BAYER_GB2RGB,
                "GRBG": cv2.COLOR_BAYER_GR2RGB,
                "BGGR": cv2.COLOR_BAYER_BG2RGB
            }
            
            # --- CASO 1: Immagine CFA (da 2D a 3D) ---
            # If we found a valid CFA pattern, demosaice
            if cfa_pattern and cfa_pattern.upper() in bayer_patterns:
                self.siril.log(f"CFA image detected ({cfa_pattern}). Demosaicing for preview.", s.LogColor.GREEN)
                
                # Create a copy so as not to modify the original data
                data_to_preview = self.full_image_data.copy()
                
                # Convert the data to 16-bit if it is a float for the demosaicing function
                if np.issubdtype(data_to_preview.dtype, np.floating):
                    self.siril.log("Converting float32 data to uint16 (linear scaling)...", s.LogColor.BLUE)
                    data_to_preview = cv2.normalize(data_to_preview, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

                conversion_code = bayer_patterns[cfa_pattern.upper()]
                # Run demosaicing to get an RGB image
                demosaiced_bgr = cv2.cvtColor(data_to_preview, conversion_code)
                # Convert from BGR (OpenCV's default) to RGB (Pillow's default)
                demosaiced_rgb = cv2.cvtColor(demosaiced_bgr, cv2.COLOR_BGR2RGB)

                # --- White Balance for Preview ---
                # We work on a float copy to apply scaling factors
                demosaiced_float = demosaiced_rgb.astype(np.float32)
                try:
                    # Calculate the median of each channel of the demosaiced image
                    # medians = [bkg_estimator(demosaiced_float[..., c]) for c in range(3)]
                    # Calculate the median of each channel with NumPy's fast function
                    medians = [np.median(demosaiced_float[..., c]) for c in range(3)]
                    
                    # --- Find the brightest channel to use as reference ---
                    max_median = max(medians)
                    
                    # Avoid division by zero if a channel is black
                    if max_median == 0:
                        max_median = 1.0

                    # Calculate scaling factors to bring all channels up to the level of the brightest one
                    scale_r = max_median / medians[0] if medians[0] > 0 else 1.0
                    scale_g = max_median / medians[1] if medians[1] > 0 else 1.0
                    scale_b = max_median / medians[2] if medians[2] > 0 else 1.0

                    self.siril.log(f"Preview white balance factors -> R: {scale_r:.2f}, G: {scale_g:.2f}, B: {scale_b:.2f}", s.LogColor.BLUE)

                    # Apply the scaling factors to each channel
                    demosaiced_float[..., 0] *= scale_r # Red channel
                    demosaiced_float[..., 1] *= scale_g # Green channel
                    demosaiced_float[..., 2] *= scale_b # Blue channel
                    
                    image_for_stretch = demosaiced_float
                except Exception as e:
                    self.siril.log(f"Could not perform preview white balance: {e}. Preview may have a color cast.", s.LogColor.ORANGE)
                    image_for_stretch = demosaiced_float
                
                # --- STRETCH unico sia per detection che visuale solo per immagini CFA ---
                # Eseguiamo lo stretch QUI, sull'immagine 3D bilanciata,
                # ricalcolando le statistiche per ogni canale.
                stretched_channels = []
                bkg_estimator = MMMBackground()
                bkg_rms = StdBackgroundRMS()

                for i in range(3):
                    channel_data = demosaiced_float[..., i]
                    try:
                        # Calcola NUOVE statistiche sul canale 3D bilanciato
                        median = bkg_estimator(channel_data)
                        sigma = bkg_rms(channel_data)
                    except Exception:
                        median = np.median(channel_data) # Fallback
                        sigma = np.std(channel_data)

                    # Applica lo stretch (identico a quello MONO, ma per canale)
                    # Imposta il nero sotto la mediana, il fondo cielo (che è a median) non sarà nero (0), ma un grigio scuro.
                    # Questo è fondamentale per vedere il segnale debole che emerge dal rumore.
                    black_point = median - 2.5 * sigma

                    # Questo crea uno stretch molto più "dolce".
                    # Mappa la gamma da (median - 2.5*s) a (median + 8.0*s) sull'intero display 0-255.
                    # Questo dà contrasto e dettaglio alle strutture deboli (la traccia) senza appiattirle a bianco puro.
                    white_point = median + 15.0 * sigma
                    
                    channel_stretched = np.clip(channel_data, black_point, white_point)
                    channel_8bit = cv2.normalize(channel_stretched, None, 0, 255, cv2.NORM_MINMAX)
                    stretched_channels.append(channel_8bit)
                
                preview_data_8bit = cv2.merge(stretched_channels).astype(np.uint8)

            # --- CASO 2: Immagine MONO (vera) ---
            else:
                # È un'immagine mono. Usa autostretch, che userà
                # le statistiche scalari corrette calcolate all'inizio.
                self.siril.log("True MONO image detected. Using standard autostretch.", s.LogColor.BLUE)
                preview_data_8bit = self.autostretch(image_for_stretch, detection=self.show_stretch_check.isChecked())

        # --- CASO 3: Immagine RGB (nativa) ---
        else:
            # È un'immagine 3D nativa. Usa autostretch, che userà
            # le statistiche per-canale corrette calcolate all'inizio.
            self.siril.log("Native RGB image detected. Using standard autostretch.", s.LogColor.BLUE)
            preview_data_8bit = self.autostretch(image_for_stretch, detection=self.show_stretch_check.isChecked())
       
        return Image.fromarray(preview_data_8bit)

    def update_ai_tuning_parameters(self):
        """
        Updates the Canny threshold variables based on the selected AI sensitivity.
        This function is called whenever a sensitivity radio button is clicked.
        """
        selected_id = None
        for sid, rb in self.radio_buttons_ai.items():
            if rb.isChecked():
                selected_id = sid
                break

        if selected_id is None:
            selected_id = self.ai_sensitivity_mid

        # Set slider values accordingly
        if selected_id == self.ai_sensitivity_max:
            # "Max" sensitivity = Lower Canny thresholds (detects even faint edges)
            self.canny_thresh1_slider.setValue(5)
            self.canny_thresh2_slider.setValue(15)
            
            self.hough_thresh_slider.setValue(60)
            self.hough_min_len_slider.setValue(50)
            self.hough_max_gap_slider.setValue(15)
        elif selected_id == self.ai_sensitivity_mid:
            # "Mid" sensitivity = Balanced thresholds
            self.canny_thresh1_slider.setValue(20)
            self.canny_thresh2_slider.setValue(60)
            
            self.hough_thresh_slider.setValue(70)
            self.hough_min_len_slider.setValue(50)
            self.hough_max_gap_slider.setValue(5)
        elif selected_id == self.ai_sensitivity_low:
            # "Low" sensitivity = Higher Canny thresholds (detects only strong edges)
            self.canny_thresh1_slider.setValue(50)
            self.canny_thresh2_slider.setValue(150)
            
            self.hough_thresh_slider.setValue(90)
            self.hough_min_len_slider.setValue(110)
            self.hough_max_gap_slider.setValue(5)
        elif selected_id == self.ai_sensitivity_Verylow:
            # "Extremely selective: ignores most faint trails, ideal for dense star fields or very noisy RGB images"
            self.canny_thresh1_slider.setValue(80)
            self.canny_thresh2_slider.setValue(240)
            
            self.hough_thresh_slider.setValue(90)
            self.hough_min_len_slider.setValue(120)
            self.hough_max_gap_slider.setValue(10)

        self.siril.log(f"AI sensitivity set to {selected_id}", s.LogColor.BLUE)

    def load_new_image_from_siril(self):
        """
        Reloads the full-resolution and preview image data from Siril,
        updates the display, and clears existing trails.
        """
        try:
            self.siril.log("Reloading image from Siril...", s.LogColor.BLUE)

            if self.siril.is_sequence_loaded():
                # If a sequence is loaded, refresh the sequence name and current frame index
                seq = self.siril.get_seq()
                self.current_seq_name = seq.seqname
                self.current_frame_index = seq.current
                self.current_frame_label.setText(f"Sequence Mode - Current Frame {self.current_frame_index + 1} / {seq.number}")

                self.siril.log(
                    f"Reload frame: {self.current_frame_index + 1}/{seq.number} from sequence: '{self.current_seq_name}' ",
                    s.LogColor.BLUE
                )
            elif self.siril.is_image_loaded():
                self.current_seq_name = None
                self.current_frame_index = None
                self.current_frame_label.setText("Single Image Mode")

                self.siril.log("Reloaded single image.", s.LogColor.BLUE)
            else:
                self.siril.error_messagebox("No image or sequence loaded")
                return
            
            # Clear existing trails (as the image might have changed significantly)
            # This is generally a good idea when reloading an image to avoid misaligned trails.
            self.trail_collection = TrailCollection()
            self.trail_listbox.clear()
            self.siril.overlay_clear_polygons()

            self.show_loading_message()
            QTimer.singleShot(500, self._perform_image_reload_work)

        except (SirilError, Exception) as e:
            self.siril.log(f"Error while reloading image: {e}", s.LogColor.RED)
            QMessageBox.critical(self, "Image Reload Error", f"An error occurred while reloading the image: {e}")

    def _perform_image_reload_work(self):
        """ This new feature contains all the slow code that previously blocked message 'Reloading image...' """
        try:
            # Update full-resolution image data
            new_full_image_data = self.siril.get_image_pixeldata(preview=False)
            if new_full_image_data is None:
                self.siril.log("No image loaded in Siril or unable to retrieve full-resolution data.", s.LogColor.RED)
                return

            # Update the original dtype
            self.original_image_dtype = new_full_image_data.dtype
            
            # Ensure HWC format if necessary (channels last)
            if len(new_full_image_data.shape) == 3 and new_full_image_data.shape[0] in [1, 3]:
                self.full_image_data = new_full_image_data.transpose(1, 2, 0)
            else:
                self.full_image_data = new_full_image_data

            # Recalculate statistics for the NEW image before using them.
            self._calculate_image_statistics()

            # Generate the new preview using our custom autostretch logic
            self.siril.log("Generating new custom visual preview...", s.LogColor.BLUE)
            # new_preview_data = self.autostretch(self.full_image_data, detection=self.show_stretch_check.isChecked())
            
            # self.preview_pil_image = Image.fromarray(new_preview_data)
            self.preview_pil_image = self._create_visual_preview()

            # Flip vertically to align with Siril's coordinate system
            self.preview_pil_image = self.preview_pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            self.update_canvas_image(self.preview_pil_image)

            # Refresh the canvas display and reset zoom
            self.fit_to_preview()
            self.siril.log("Image reloaded and display updated.", s.LogColor.GREEN)

        except (SirilError, Exception) as e:
            self.siril.log(f"Error while reloading image: {e}", s.LogColor.RED)
            QMessageBox.critical(self, "Image Reload Error", f"An error occurred while reloading the image: {e}")

    # --- Trail Management Methods ---
    def add_trail(self):
        """
        Adds a new, empty trail and enters drawing mode.

        This function is triggered by the 'Add' button. It performs the following actions:
        1. Creates a new trail object in the data collection and sets it as active.
        2. Assigns the current line width from the spinbox to the new trail.
        3. Adds and selects a new item in the UI listbox.
        4. Activates the 'drawing mode' to allow the user to add points on the canvas.
        """
        trail = self.trail_collection.add_trail()
        # The new trail gets its current width from the spinbox
        trail.line_width = self.line_width_spinbox.value()
        self.trail_listbox.addItem( f"Trail {trail.id}")
        self.trail_listbox.setCurrentRow(self.trail_listbox.count() - 1)

        # --- ACTIVATE DRAWING MODE ---
        self.is_drawing_new_segment = True
        self.new_segment_anchor = None # The anchor has not yet been placed
        self.siril.log("Entered drawing mode for new trail. Click on canvas to add points.", s.LogColor.GREEN)
        
        # Let's update the view
        self.redraw_canvas_overlays()

    def duplicate_trail(self):
        """
        Duplicates the currently selected trail, creating a new, independent trail
        with the same points and width.
        """
        # Get the source trail (the currently active one)
        source_trail = self.trail_collection.get_active_trail()

        # Check if a trail is actually selected
        if not source_trail:
            self.siril.log("No trail selected to duplicate.", s.LogColor.RED)
            return

        # Use the existing method to create a new trail.
        # This already sets it as the active trail.
        new_trail = self.trail_collection.add_trail()

        # Copy the important properties from the source trail to the new one
        # It is FUNDAMENTAL to use .copy() to create a new independent list of points.
        new_trail.points = source_trail.points.copy()
        new_trail.line_width = source_trail.line_width

        # Apply a small offset for visibility
        # Define a shift of 15 pixels (in screen coordinates)
        SHIFT_PIXELS = 15
        # Convert the shift to scene coordinates, taking the zoom into account
        offset = SHIFT_PIXELS / self.view.transform().m11()
        # Apply the shift to each point of the new track
        new_trail.points = [(p[0] + offset, p[1] + offset) for p in new_trail.points]

        # Update the spline for the new trail based on the newly copied points
        new_trail.update_spline()

        # Update the Listbox in the user interface
        self.trail_listbox.addItem( f"Trail {new_trail.id} (copy)")
        self.trail_listbox.setCurrentRow(self.trail_listbox.count() - 1) # Select the new duplicated trail

        # Redraw the canvas to show the new trail
        self.redraw_canvas_overlays()

        self.siril.log(f"Duplicated Trail {source_trail.id} to new Trail {new_trail.id}.", s.LogColor.BLUE)

    def update_selected_trail_width(self):
        """
        Applies the value from the line width spinbox to the currently selected trail.
        """
        active_trail = self.trail_collection.get_active_trail()
        if not active_trail:
            self.siril.log("No trail selected to update.", s.LogColor.RED)
            return

        new_width = self.line_width_spinbox.value()
        active_trail.line_width = new_width
        self.redraw_canvas_overlays()
        self.siril.log(f"Trail {active_trail.id} width updated to {new_width}px.", s.LogColor.BLUE)

    def remove_trail(self):
        """
        Removes the currently selected trail.

        This method deletes the trail from both the data collection and the UI listbox.
        After removing the item, it automatically selects the next appropriate trail
        in the list to ensure a smooth workflow.
        """
        row = self.trail_listbox.currentRow(); selected_indices = [row] if row >= 0 else []
        if not selected_indices:
            self.siril.log(f"No trail selected.", s.LogColor.RED)
            return

        active_trail = self.trail_collection.get_active_trail()
        try:
            idx = selected_indices[0]
            self.trail_collection.remove_trail(idx)
            self.trail_listbox.takeItem(idx)
            self.siril.log(f"Trail {active_trail.id} deleted", s.LogColor.RED)

            if self.trail_collection.trail:
                new_selection = min(idx, len(self.trail_collection.trail) - 1)
                self.trail_listbox.setCurrentRow(new_selection)
                self.trail_collection.set_active(new_selection)

            self.redraw_canvas_overlays()
        except Exception as e:
            self.siril.log(f"Error: {e}", s.LogColor.RED)

    def on_trail_select(self, event):
        """
        Handles the selection change event for the trail listbox.

        This slot is connected to the listbox's `currentItemChanged` signal. It syncs the
        application's state with the user's selection by:
        1. Exiting any active drawing mode.
        2. Setting the newly selected trail as the active one in the data collection.
        3. Updating the 'Line Width' spinbox to reflect the active trail's width.
        4. Redrawing the canvas overlays to update trail colors.
        """
        # Exit drawing mode when changing selection
        self.cancel_drawing()

        row = self.trail_listbox.currentRow()
        if row < 0:
            self.trail_collection.set_active(-1)
        else:
            self.trail_collection.set_active(row)
        active_trail = self.trail_collection.get_active_trail()
        if active_trail:
            self.line_width_spinbox.setValue(int(active_trail.line_width))
        self.redraw_canvas_overlays()

    def auto_detect_trails(self):
        """
        Performs automatic detection of trails and adds them to the list.
        """
        self.siril.log("Starting automatic trail detection...", s.LogColor.BLUE)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # Clear existing trails (as the image might have changed significantly)
        # This is generally a good idea when reloading an image to avoid misaligned trails.
        self.trail_collection = TrailCollection()
        self.trail_listbox.clear() # Clear listbox display
        
        canny_is_checked = self.show_canny_check.isChecked()
        hough_is_checked = self.show_hough_check.isChecked()
        # If no debug checkbox is active, the first thing to do is make sure
        # the view returns to the normal view. This handles the case where the user
        # has just deselected a debug option.
        if not canny_is_checked and not hough_is_checked:
            self.update_canvas_image(self.preview_pil_image)

        # Use the full-depth data and apply autostretch
        self.siril.log("Performing custom autostretch for detection...", s.LogColor.BLUE)
        # The autostretch function handles both mono and color, returning an 8-bit image
        stretched_image_for_detection = self.autostretch(self.full_image_data, detection=True)
        
        if self.ui_ai_tuning:
            # --- High-Reliability Detection in DEBUG MODE ---
            # If debug mode is active, we use the parameters from the UI controls
            self.siril.log("Reads parameters from the UI controls", s.LogColor.BLUE)
            hough_params = {
                'canny_low': self.canny_thresh1_slider.value(),
                'canny_high': self.canny_thresh2_slider.value(),
                'hough': {
                    'threshold': self.hough_thresh_slider.value(),
                    'minLineLength': self.hough_min_len_slider.value(),
                    'maxLineGap': self.hough_max_gap_slider.value()
                }
            }
        else:
            # --- High-Reliability Detection ---
            self.siril.log("Pass 1: Detecting high-confidence trails...", s.LogColor.BLUE)
            hough_params = {
                'canny_low': 10,
                'canny_high': 30,
                'hough': {
                    'threshold': 90,
                    'minLineLength': 90,
                    'maxLineGap': 20
                }
            }

        self.siril.log("Using detection parameters:", s.LogColor.BLUE)
        self.siril.log(f"  - Canny low: {hough_params['canny_low']}", s.LogColor.BLUE)
        self.siril.log(f"  - Canny high: {hough_params['canny_high']}", s.LogColor.BLUE)

        h = hough_params['hough']
        self.siril.log("  - Hough parameters:", s.LogColor.BLUE)
        self.siril.log(f"      * threshold: {h['threshold']}", s.LogColor.BLUE)
        self.siril.log(f"      * minLineLength: {h['minLineLength']}", s.LogColor.BLUE)
        self.siril.log(f"      * maxLineGap: {h['maxLineGap']}", s.LogColor.BLUE)

        # Execute detection
        hough_lines, edges_image  = self.find_trails_with_hough(stretched_image_for_detection, hough_params)

        final_debug_image = None
        # If debug mode is active, display the Canny edge map and stop
        # Determine which image to use as a base
        if canny_is_checked:
            self.siril.log("Debug mode: Displaying Canny edge map.", s.LogColor.BLUE)
            # The basis is the Canny edge map
            # The preview image is flipped, so the debug image must also be flipped for correct display
            base_image_for_debug = Image.fromarray(edges_image).transpose(Image.FLIP_TOP_BOTTOM)
        else:
            # The base is the original preview
            base_image_for_debug = self.preview_pil_image
            
        if hough_is_checked:
            self.siril.log(f"Debug mode: Drawing {len(hough_lines)} raw Hough lines for debug.", s.LogColor.BLUE)
            # Take the image CURRENTLY on the canvas as a base.
            # If Canny is active, it will be the edge map. Otherwise, it will be the original image.
            debug_hough_image = base_image_for_debug.copy().convert("RGB")
            draw = ImageDraw.Draw(debug_hough_image)

            image_height = debug_hough_image.height

            for x1, y1, x2, y2 in hough_lines:
                # --- Invert the Y-axis to align with the flipped image ---
                y1_flipped = image_height - y1
                y2_flipped = image_height - y2
                draw.line([(x1, y1_flipped), (x2, y2_flipped)], fill="lime", width=2)
        
            final_debug_image = debug_hough_image
        elif canny_is_checked:
            # If only Canny is active, the final image is simply the base (the Canny map)
            final_debug_image = base_image_for_debug

        # If a debug image has been created, view it and stop execution.
        if final_debug_image is not None:
            self.update_canvas_image(final_debug_image)
            QApplication.restoreOverrideCursor()
            return
            
        if not hough_lines:
            self.siril.log("No high-confidence trails found. Detection stopped.", s.LogColor.RED)
            QApplication.restoreOverrideCursor()
            return

        self.siril.log(f"Found {len(hough_lines)} high-confidence segments.", s.LogColor.BLUE)

        SEGMENT_LIMIT = 2000
        if len(hough_lines) > SEGMENT_LIMIT:
            self.siril.log(f"\nDetection stopped: Too many segments found ({len(hough_lines)} > {SEGMENT_LIMIT})."
                           "\nSuggesting user to increase Canny thresholds or lower sensitivity.", s.LogColor.RED)
            
            message = (
                f"Automatic detection found {len(hough_lines)} line segments, exceeding the safety limit of {SEGMENT_LIMIT}.\n\n"
                "Processing this many segments would be extremely slow.\n\n"
                "**TIP:**\n\nTry using a lower 'AI Sensitivity' or manually increase the 'Canny Low' and 'Canny High' thresholds to reduce noise."
            )
            QMessageBox.warning(self, "Overload Detection", message)

            QApplication.restoreOverrideCursor()
            return

        # --- Final Fusion ---
        # Merge the reliable lines with the validated weak fragments
        final_line_candidates = hough_lines
        self.siril.log(f"Processing {len(final_line_candidates)} total segments...", s.LogColor.BLUE)
        
        processed_trails = self.process_detected_lines(final_line_candidates, stretched_image_for_detection.shape)

        if not processed_trails:
            self.siril.log("No consolidated trails found after processing.", s.LogColor.RED)
            QApplication.restoreOverrideCursor()
            return

        self.siril.log(f"Reconstructed {len(processed_trails)} full trails.", s.LogColor.GREEN)
        
        for trail_data in processed_trails:
            new_trail = self.trail_collection.add_trail()
            self.trail_listbox.addItem( f"Trail {new_trail.id} (AI)")
            
            p1, p2 = trail_data['points']
            
            # Invert Y with respect to the image height
            p1_inverted_y = self.preview_pil_image.height - p1[1]
            p2_inverted_y = self.preview_pil_image.height - p2[1]

            new_trail.add_point(p1[0], p1_inverted_y)
            new_trail.add_point(p2[0], p2_inverted_y)
            
            # Set the dynamically calculated width on the trail
            new_trail.line_width = trail_data['width']
        
        # Update UI
        if processed_trails:
            # self.trail_listbox.setCurrentRow(self.trail_listbox.count() - 1)
            self.trail_collection.set_active(-1)

            # Refresh the canvas display and reset zoom
            self.fit_to_preview()
        self.redraw_canvas_overlays()
        
        QApplication.restoreOverrideCursor()
        self.siril.log(f"Detection Complete: Reconstructed {len(processed_trails)} trails.", s.LogColor.BLUE)

    def find_trails_with_hough(self, stretched_image_8bit, hough_params, auto_thresholding=False):
        """
        Detects lines in an image using the Hough Transform with customizable parameters.
        Returns a list of coordinates (x1, y1, x2, y2) for each line found.
        """
        # Extract Canny thresholds
        canny_low = hough_params['canny_low']
        canny_high = hough_params['canny_high']

        # Extract Hough parameters from the sub-dictionary
        hough = hough_params['hough']
        hough_threshold = hough['threshold']
        min_line_length = hough['minLineLength']
        max_line_gap = hough['maxLineGap']

        # Apply a slight blur to reduce noise
        #blurred = cv2.GaussianBlur(stretched_image_8bit, (5, 5), 0)
        blurred = stretched_image_8bit

        if auto_thresholding:
            # auto_thresholding, if auto_thresholding = True, Enable automatic thresholding for Canny.
            bkg_estimator = MMMBackground()
            bkg_rms = StdBackgroundRMS()
            median = bkg_estimator(blurred)
            sigma = bkg_rms(blurred)
            lower_thresh = int(max(0, (1.0 - sigma) * median))
            upper_thresh = int(min(255, (1.0 + sigma) * median))

            self.siril.log(f"--- DEBUG: canny test for auto_thresholding value ---", s.LogColor.BLUE)
            self.siril.log(f"    lower_thresh = {lower_thresh} --   upper_thresh = {upper_thresh}", s.LogColor.BLUE)
            self.siril.log(f"----------------------------", s.LogColor.BLUE)

        # Step 1: Edge Detection
        # cv2.Canny(image, low_threshold, high_threshold)
        # The two arguments, low_threshold and high_threshold, allow for the isolation of adjacent pixels that follow the most intense gradient.
        # - If the gradient is greater than the upper threshold, it is identified as an edge pixel.
        # - If it is lower than the lower threshold, it is rejected.
        # The gradient between the thresholds is accepted only if it is connected to a strong edge.
        # Standard Canny thresholds: a low-to-high threshold ratio of 1:3.
        edges = cv2.Canny(blurred, canny_low, canny_high)

        view_edges = False

        # 1) (optional) Remove the small star rings -> Cleans up the noise without affecting the trail
        # kill_circles = True
        # if kill_circles:
        #     circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #     closed = cv2.morphologyEx(edges, cv2.MORPH_OPEN, circ, iterations=1)

        # Step 2: Morphological Closing to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=5)

        # Step 3: Create an empty mask
        mask = np.zeros_like(closed)

        # Step 4: Find contours from the "closed" mask
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Analyze contours and filter based on geometric properties to identify likely satellite trails
        # Step 5a: Metrics for filtering
        elongations = []
        circularities = []
    
        debug_mode_draw_filter_contour = False
        if debug_mode_draw_filter_contour:
            if len(stretched_image_8bit.shape) == 2:  # grayscale
                filter_contour_img_display = cv2.cvtColor(stretched_image_8bit, cv2.COLOR_GRAY2BGR)
            else:  # RGB
                filter_contour_img_display = stretched_image_8bit.copy()

        # Step 5b: Collect all metrics
        for cnt in contours:
            # --- Geometric parameters calculation ---
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            # Typical values: of circularity
            # Perfect circle    -> circularity ≈ 1
            # Blobs/stars       -> 0.7–1
            # Lines             -> very low values ​​(typically < 0.3)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)

            # minAreaRect -> elongation calculation
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            # Typical values: of elongation
            # Stars (circles or blobs)  -> elongation ~ 1
            # Trails                    -> elongation >> 1 (usually > 5–10)
            elongation = max(w, h) / (min(w, h) + 1e-5)

            elongations.append(elongation)
            circularities.append(circularity)

        # Step 5c: Adaptive thresholds based on medians
        elong_median = np.median(elongations)
        circ_median  = np.median(circularities)
        # Thresholds relative to the median
        elong_thresh = max(elong_median * 3.0, 5)   # at least 3x more elongated than a typical star
        circ_thresh = min(circ_median * 0.6, 0.4)   # much less circular than the stellar average

        # Informational debugging: distribution statistics
        # self.siril.log(f"--- DEBUG: Metrics distribution summary ---", s.LogColor.BLUE)
        # self.siril.log(f" contours: {len(elongations)}", s.LogColor.BLUE)
        # self.siril.log(f" elongation: median={elong_median:.2f} -> elong_thresh={elong_thresh:.2f}", s.LogColor.BLUE)
        # self.siril.log(f" circularity: median={circ_median:.3f} -> circ_thresh={circ_thresh:.3f}", s.LogColor.BLUE)
        # self.siril.log(f"----------------------------", s.LogColor.BLUE)

        # Step 5d: Classify the contours
        # --- Typical criteria ---
        # Stars/blobs -> elongation ~1      circularity 0.7 – 1
        # Lines       -> elongation >> 1    circularity < 0.3–0.4
        for cnt, elongation, circularity in zip(contours, elongations, circularities):
            if elongation > elong_thresh and circularity < circ_thresh:
                # Very likely satellite trace
                cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)  # Fill
                if debug_mode_draw_filter_contour:
                    cv2.drawContours(filter_contour_img_display, [cnt], -1, (0,255,0), 2)   # green = trail -> 2 pixel thick line
            else:
                # star/noise
                if debug_mode_draw_filter_contour:
                    cv2.drawContours(filter_contour_img_display, [cnt], -1, (0,0,255), 2)   # blue = star -> 2 pixel thick line

        # Now we have a mask with a "full" trail
        closed = mask

        # Last Step - Hough Transform to find lines using the filled mask
        lines = cv2.HoughLinesP(
            closed,
            rho = 1,
            theta = np.pi / 180,
            threshold = hough_threshold,
            minLineLength = min_line_length,
            maxLineGap = max_line_gap
        )

        found_lines = [line[0] for line in lines] if lines is not None else []

        if debug_mode_draw_filter_contour:
            return found_lines, filter_contour_img_display
        elif view_edges:
            return found_lines, edges
        else:
            return found_lines, closed

    def process_detected_lines(self, lines, img_shape, debug_step_1=False, debug_step_2=False):
        """
        Processes Hough lines with robust logic and edge extrapolation.
        Includes two flag-controlled debug steps.

        1. Filters out segments that are too short.
        2. Groups neighboring segments based on their position.
        3. For each group, creates a line containing all the segments.
        4. Extends the final line to the exact edges of the image.

        - debug_step_1=True: Shows only the original Hough segments after length filtering.
        - debug_step_2=True: Shows the merged stripes but BEFORE edge extrapolation.
        """
        img_h, img_w = img_shape[:2]
        img_diag = math.hypot(img_w, img_h)

        # This internal function takes a group of points and handles everything:
        # fit, width calculation, extrapolation, and fallback.
        def _finalize_trail_from_group(points_in_group, extrapolate=True):
            if len(points_in_group) < 2:
                return None

            # Fit line robustly to all points in the cluster
            pts_arr = np.array(points_in_group, dtype=np.float32)

            # Precomputed fallback: furthest pair of points in the group (O(n^2), small n)
            max_pair_dist = 0.0
            p_far_1 = pts_arr[0]
            p_far_2 = pts_arr[1]
            for i in range(len(pts_arr)):
                for j in range(i+1, len(pts_arr)):
                    d = math.hypot(float(pts_arr[i,0]-pts_arr[j,0]),
                                float(pts_arr[i,1]-pts_arr[j,1]))
                    if d > max_pair_dist:
                        max_pair_dist = d
                        p_far_1 = pts_arr[i]
                        p_far_2 = pts_arr[j]

            try:
                fit = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
                vx_fit, vy_fit, x0_fit, y0_fit = map(float, fit.flatten())
            except Exception:
                # Fallback: Use the furthest pair + minimum width proportional to the diagonal
                min_width = int(img_diag * 0.005)
                width = max(min_width, 10) # Use the calculated minimum, but not less than 10
                return {
                    'points': [(int(round(p_far_1[0])), int(round(p_far_1[1]))),
                            (int(round(p_far_2[0])), int(round(p_far_2[1])))],
                    'width': width
                }

            norm_dir = math.hypot(vx_fit, vy_fit)
            if norm_dir == 0:
                # Fallback for null vector
                min_width = int(img_diag * 0.005)
                width = max(min_width, 10)
                return {
                    'points': [(int(round(p_far_1[0])), int(round(p_far_1[1]))),
                            (int(round(p_far_2[0])), int(round(p_far_2[1])))],
                    'width': width
                }
            
            vx_unit = vx_fit / norm_dir
            vy_unit = vy_fit / norm_dir

            # Width calculation logic
            # Perpendicular distances from each point to the fitted line: used to estimate width
            distances = np.abs((pts_arr[:, 0] - x0_fit) * vy_unit - (pts_arr[:, 1] - y0_fit) * vx_unit)
            # Percentile-based width (sensitive to detection accuracy)
            base_width = np.percentile(distances, 95) * 2.0

            # Minimum offset proportional to resolution
            offset = img_diag * 0.01  # ex. 1% = 75px on diagonal 7500

            # Ensure width is at least a minimum size
            width = int(base_width + offset)

            # Absolute minimum proportional to resolution
            min_width = int(img_diag * 0.005)  # ex. 0.5% = 37.5px on diagonal 7500
            if width < min_width:
                width = min_width

            # Projection by extremes (serves both as a fallback and for "no extrapolation")
            t_vals = (pts_arr[:, 0] - x0_fit) * vx_unit + (pts_arr[:, 1] - y0_fit) * vy_unit
            t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
            p1_proj = (x0_fit + vx_unit * t_min, y0_fit + vy_unit * t_min)
            p2_proj = (x0_fit + vx_unit * t_max, y0_fit + vy_unit * t_max)

            if not extrapolate:
                return {
                    'points': [(int(round(p1_proj[0])), int(round(p1_proj[1]))),
                            (int(round(p2_proj[0])), int(round(p2_proj[1])))],
                    'width': width
                }

            # PRODUCTION Logic: Perform extrapolation to the image borders.
            # We compute intersections of the infinite fitted line with the four image borders.
            epsilon = 1e-8
            intersections = []

            # Use borders 0 and (W - 1)/(H - 1) to stay in image after round
            img_wm1 = float(img_w - 1)
            img_hm1 = float(img_h - 1)

            if abs(vx_unit) > epsilon:
                # Check intersections with vertical borders (x = 0 and x = img_w)
                for x_border in (0.0, img_wm1):
                    t = (x_border - x0_fit) / vx_unit
                    y_at_t = y0_fit + vy_unit * t
                    if 0.0 <= y_at_t <= img_hm1:
                        intersections.append((x_border, y_at_t))

            if abs(vy_unit) > epsilon:
                # Check intersections with horizontal borders (y = 0 and y = img_h)
                for y_border in (0.0, img_hm1):
                    t = (y_border - y0_fit) / vy_unit
                    x_at_t = x0_fit + vx_unit * t
                    if 0.0 <= x_at_t <= img_wm1:
                        intersections.append((x_at_t, y_border))

            # Keep only unique intersection points (within a tolerance)
            unique_intersections = []
            for p in intersections:
                is_unique = True
                for q in unique_intersections:
                    if math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-3:
                        is_unique = False
                        break
                if is_unique:
                    unique_intersections.append(p)
            
            if len(unique_intersections) >= 2:
                # Choose the two intersections that are farthest apart
                max_dist_ext = 0
                ext_p1, ext_p2 = unique_intersections[0], unique_intersections[1]
                for i in range(len(unique_intersections)):
                    for j in range(i + 1, len(unique_intersections)):
                        dist = math.hypot(unique_intersections[i][0] - unique_intersections[j][0],
                                        unique_intersections[i][1] - unique_intersections[j][1])
                        if dist > max_dist_ext:
                            max_dist_ext = dist
                            ext_p1, ext_p2 = unique_intersections[i], unique_intersections[j]
                
                # Final clipping for absolute safety
                x1 = int(round(ext_p1[0]))
                y1 = int(round(ext_p1[1]))
                x2 = int(round(ext_p2[0]))
                y2 = int(round(ext_p2[1]))
                
                x1_clipped = max(0, min(img_w - 1, x1))
                y1_clipped = max(0, min(img_h - 1, y1))
                x2_clipped = max(0, min(img_w - 1, x2))
                y2_clipped = max(0, min(img_h - 1, y2))

                return {
                    'points': [(x1_clipped, y1_clipped),
                            (x2_clipped, y2_clipped)],
                    'width': width
                }

            # Fallback: Use projected endpoints
            return {
                'points': [(int(round(p1_proj[0])), int(round(p1_proj[1]))),
                        (int(round(p2_proj[0])), int(round(p2_proj[1])))],
                'width': width
            }

        # --- STEP 1: Adaptive min_length_perc ---
        lengths = [math.hypot(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines]
        if lengths:
            median_len = np.median(lengths)
            perc75_len = np.percentile(lengths, 75)
            max_len = np.max(lengths)
        else:
            # Default value if there are no lines
            median_len, perc75_len, max_len = 0, 0, 1

        # self.siril.log(f"Maximum Line Length: {max_len}", s.LogColor.BLUE)
        # self.siril.log(f"Median of Lines: {median_len}", s.LogColor.BLUE)
        # self.siril.log(f"75th Percentile: {perc75_len}", s.LogColor.BLUE)

        # if the 75th percentile is small (<5% diagonal), we raise the threshold
        if perc75_len < max_len * 0.25: # 75% of the lines are shorter than 1/4 of the longest
            min_length_perc = 0.005 # more selective
        # if the median is high, we lower the threshold to avoid losing real fragments
        elif median_len > max_len * 0.5: # very high median => lower the threshold
                min_length_perc = 0.001
        else:
            min_length_perc = 0.001

        # self.siril.log(f"min_length_perc: {min_length_perc}", s.LogColor.BLUE)

        # Calculate the minimum length threshold based on the image diagonal
        # --- Initial filter of short segments ---
        min_length_threshold = img_diag * min_length_perc

        # self.siril.log(f"min_length_threshold: {min_length_threshold}", s.LogColor.BLUE)

        segments = []
        for i, (x1, y1, x2, y2) in enumerate(lines):
            length = math.hypot(x2 - x1, y2 - y1)
            if length >= min_length_threshold:
                # compute midpoint and angle (0..180 deg)
                dx = x2 - x1
                dy = y2 - y1
                angle = math.degrees(math.atan2(dy, dx)) % 180.0
                midpoint = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                segments.append({
                    'id': i,
                    'p1': np.array([x1, y1]),
                    'p2': np.array([x2, y2]),
                    'points': [np.array([x1, y1]), np.array([x2, y2])],
                    'length': length,
                    'angle': angle,
                    'mid': midpoint
                })

        if not segments:
            return []

        # --- DEBUG CASE 1: Show only the original filtered segments ---
        # If this flag is set, the function stops here and returns the raw (but length-filtered) segments for analysis.
        if debug_step_1:
            print("--- DEBUG STEP 1: Show original segments after length filter ---")
            final_trails = []
            for seg in segments:
                final_trails.append({
                    'points': [(int(seg['p1'][0]), int(seg['p1'][1])),
                            (int(seg['p2'][0]), int(seg['p2'][1]))],
                    'width': 2  # Assign a fixed width for the display
                })
            return final_trails
        
        # If we reach here, we have valid segments to process further.

        # --- STEP 2 & 3: Group and Merge Neighboring Lines using DBSCAN on (x,y,angle) ---
        # Distance threshold used for grouping (image-diagonal relative)
        merge_dist_perc = 0.04      # maximum distance between midpoints for fusion, 4% of the diagonal
        merge_dist_threshold = img_diag * merge_dist_perc

        # Angular tolerance (degrees) used to scale the angle feature so that
        # a difference of `angle_thresh_deg` degrees is roughly comparable
        # to spatial merge_dist_threshold in DBSCAN feature space.
        angle_thresh_deg = 10.0
        if angle_thresh_deg <= 0:
            angle_scale = 1.0
        else:
            angle_scale = merge_dist_threshold / angle_thresh_deg
        
        features = np.array([[seg['mid'][0], seg['mid'][1], seg['angle'] * angle_scale] for seg in segments], dtype=np.float32)
        
        # DBSCAN will cluster segments that are spatially close and have similar angle.
        # min_samples=1 allows singletons to be their own cluster.
        db = DBSCAN(eps=merge_dist_threshold, min_samples=1, metric='euclidean')
        labels = db.fit_predict(features)
        unique_labels = set(labels)

        # --- DEBUG CASE 2 vs PRODUCTION (Extrapolation) ---
        if debug_step_2:
            # If debug flag set, add the merged strip without extrapolating it to edges.
            # Extrapolation is part of the helper, so this debug step shows the result of the FIRST clustering pass, fully extrapolated.
            print("--- DEBUG STEP 2: Show merged strips WITHOUT extrapolation ---")
            trails_dbg = []
            for lbl in unique_labels:
                idxs = np.where(labels == lbl)[0].tolist()
                all_pts = [p for idx in idxs for p in segments[idx]['points']]
                t = _finalize_trail_from_group(all_pts, extrapolate=False)
                if t: trails_dbg.append(t)
            return trails_dbg
        
        initial_trails = []

        for lbl in unique_labels:
            cluster_indices = np.where(labels == lbl)[0].tolist()
            all_points_in_group = [p for idx in cluster_indices for p in segments[idx]['points']]
            
            # --- Use the helper for the first time ---
            trail = _finalize_trail_from_group(all_points_in_group)
            if trail:
                initial_trails.append(trail)
        
        # If I reach here, I have initial trails that are grouped and merged.

        # --- EXTRA STEP: Final merge of overlapping trails ---
        if len(initial_trails) <= 1:
            return initial_trails

        final_merge_dist_perc = 0.01    # maximum distance between midpoints for fusion, 1% of the diagonal
        final_merge_dist_threshold = img_diag * final_merge_dist_perc
        final_angle_thresh_deg = 5.0    # 5° tolerance
        final_angle_scale = final_merge_dist_threshold / final_angle_thresh_deg if final_angle_thresh_deg > 0 else 1.0
        
        final_features = []
        for tr in initial_trails:
            (x1, y1), (x2, y2) = tr['points']
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            final_features.append([mid_x, mid_y, angle * final_angle_scale])

        final_labels = DBSCAN(eps=final_merge_dist_threshold, min_samples=1).fit_predict(np.array(final_features))
        
        final_trails = []
        for lbl in set(final_labels):
            idxs = np.where(final_labels == lbl)[0].tolist()
            if len(idxs) == 1:
                final_trails.append(initial_trails[idxs[0]])
                continue
                
            all_points = [p for i in idxs for p in initial_trails[i]['points']]
            # --- Use the helper for the second time ---
            merged_trail = _finalize_trail_from_group(all_points)
            if merged_trail:
                final_trails.append(merged_trail)

        return final_trails
    
    # --- Canvas Interaction Methods ---
    def is_moving_point(self):
        """ Checks whether the user is currently dragging a control point. """
        return self.mouse_is_pressed and self.moving_point_index is not None

    def on_view_mouse_press(self, event, scene_pos: QPointF):
        """ 
        Handles Left click to add or select/move a point, Ctrl+click to remove a nearby point.

        This function first determines if the click is near an existing control point.
        If so, it either deletes the point (if Ctrl is pressed and more than two
        points exist) or prepares to move it. If the click is on an empty area
        and the application is in 'drawing mode', it adds a new point to the
        active trail and manages the rubber band line's anchor.
        """
        active_trail = self.trail_collection.get_active_trail()
        if not active_trail:
            return

        x = scene_pos.x()
        y = scene_pos.y()

        # Selection radius for control points (about 20 pixels, scaled with zoom)
        # Converted to scene coordinates to be zoom-independent
        handle_radius = 20 / self.view.transform().m11()
        idx_near = -1
        min_dist_sq = float('inf')
        for i, (px, py) in enumerate(active_trail.points):
            dist_sq = (px - x)**2 + (py - y)**2
            if dist_sq < handle_radius**2 and dist_sq < min_dist_sq:
                idx_near = i
                min_dist_sq = dist_sq

        # Click management
        self.mouse_is_pressed = True
        
        # If we press CTRL + Click, we remove the point
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if idx_near != -1:
                # Check if the trail has more than 2 points before deleting.
                if len(active_trail.points) <= 2:
                    self.siril.log("Cannot remove point. A trail must have at least two points.", s.LogColor.GREEN)
                    self.mouse_is_pressed = False # Resets the mouse state
                    return # Stops deletion

                # If the check passes, proceed with deletion
                active_trail.remove_point_at(idx_near)
                self.redraw_canvas_overlays()
                self.mouse_is_pressed = False # The action is completed
            return

        # If we are close to a point, we start moving it
        if idx_near != -1:
            self.moving_point_index = idx_near
        else:
            # Otherwise, we prepare not to move any points
            self.moving_point_index = None

        # Add a point ONLY IF we are in drawing mode.
        if self.is_drawing_new_segment:
            active_trail.add_point(x, y, is_drawing_new=True)
            
            # If this is the first point on the line, there is no anchor yet
            if self.new_segment_anchor is None:
                self.new_segment_anchor = scene_pos
            
            # Update the rubberband anchor to the newly created point
            self.new_segment_anchor = scene_pos
            self.rubber_band_line.setLine(QLineF(self.new_segment_anchor, self.new_segment_anchor))
            self.rubber_band_line.setVisible(True)
            
            self.redraw_canvas_overlays()

    def on_view_mouse_move(self, event, scene_pos: QPointF):
        """ This method is only called if we are actively dragging a point """
        # If we are dragging an existing point, we execute that logic
        if self.is_moving_point():
            active_trail = self.trail_collection.get_active_trail()
            if active_trail:
                active_trail.points[self.moving_point_index] = (scene_pos.x(), scene_pos.y())
                active_trail.update_spline()
                self.redraw_canvas_overlays()
        # Otherwise, if we are in drawing mode, we update the rubber band line
        elif self.is_drawing_new_segment and self.new_segment_anchor is not None:
            self.rubber_band_line.setLine(QLineF(self.new_segment_anchor, scene_pos))
            
    def on_view_mouse_release(self, event, scene_pos: QPointF):
        """
        Handles the mouse release event to finalize user actions.

        If a control point was being dragged, this finalizes its new position.
        If the action was a simple click (not a pan or drag) near an
        existing trail's spline, it adds a new control point to that trail.
        This allows users to refine a curve after its initial creation.
        
        Resets mouse state flags after the action is complete.
        """
        active_trail = self.trail_collection.get_active_trail()
        
        # If we were moving a point, we finalize the operation
        if self.is_moving_point():
            if active_trail:
                active_trail.update_spline()
                self.redraw_canvas_overlays()

        # Otherwise, if it was a simple click (not a pan) on an active trail...
        elif self.mouse_is_pressed and not self.view._is_panning and active_trail:
            # ...and we are NOT in the initial drawing mode of a new segment...
            if not self.is_drawing_new_segment:
                # ...then it's one click to add a point to an EXISTING trail.
                min_dist_sq = float('inf')
                click_pos = (scene_pos.x(), scene_pos.y())

                if active_trail.spline_points:
                    # Iterate through the SEGMENTS of the spline, not the individual points
                    for i in range(len(active_trail.spline_points) - 1):
                        p1 = active_trail.spline_points[i]
                        p2 = active_trail.spline_points[i+1]
                        
                        dist_sq = self._distance_point_to_segment_sq(click_pos, p1, p2)
                        
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq

                # Let's define a click threshold (e.g. 15 pixels, scaled with zoom)
                click_threshold = 15 / self.view.transform().m11()
                # self.siril.log(f"distanza click: {math.sqrt(min_dist_sq)} - Soglia {click_threshold}", s.LogColor.BLUE)

                # If the click is close enough to the line, we add the dot.
                if math.sqrt(min_dist_sq) < click_threshold:
                    self.siril.log("Adding new point to selected trail.", s.LogColor.BLUE)
                    # The add_point function already handles spline reordering and updating
                    active_trail.add_point(scene_pos.x(), scene_pos.y())
                    self.redraw_canvas_overlays()

        # Reset the mouse state at the end of each action.
        self.mouse_is_pressed = False
        self.moving_point_index = None

    def _distance_point_to_segment_sq(self, p, v, w):
        """
        Calculates the squared distance from point 'p' to line segment 'vw'.
        'p', 'v', and 'w' are tuples or arrays of (x, y) coordinates.
        """
        p, v, w = np.array(p), np.array(v), np.array(w)
        l2 = np.sum((v - w)**2)
        if l2 == 0.0:
            return np.sum((p - v)**2)
        
        # Project the point p onto the segment vw
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)
        
        return np.sum((p - projection)**2)

    # --- Display and Zoom Methods ---
    def update_canvas_image(self, image_to_display=None):
        """
        Convert the PIL preview image into a QPixmap and display it in the QGraphicsScene,
        with some padding around the edges to allow panning beyond the bounds.
        """
        if image_to_display is not None:
            img = image_to_display
        else:
            if self.preview_pil_image is None:
                return
            img = self.preview_pil_image

        # If needed, keep the preview as the "current" state
        self.current_canvas_image = img.copy()

        # Make sure the object is a Pillow image before continuing
        if not isinstance(img, Image.Image):
            # If it's another type (probably a NumPy array), we convert it
            try:
                img = Image.fromarray(img)
            except Exception as e:
                self.siril.log(f"Error: Unable to convert object to image for display. {e}", s.LogColor.RED)
                return

        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        # Convert to QImage/QPixmap
        self.qimage = ImageQt(img).copy()
        pixmap = QPixmap.fromImage(self.qimage)

        # Clear old scene
        if hasattr(self, 'pixmap_item') and self.pixmap_item is not None and self.pixmap_item.scene() == self.scene:
            self.scene.removeItem(self.pixmap_item)

        # Add pixmap
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        # Optionally add padding by expanding scene rect
        pad_x = pixmap.width() * 0.5
        pad_y = pixmap.height() * 0.5
        self.scene.setSceneRect(-pad_x, -pad_y, pixmap.width() + 2*pad_x, pixmap.height() + 2*pad_y)
        
        # Redraw overlays on top
        self.redraw_canvas_overlays()

    def redraw_canvas_overlays(self):
        """
        Redraws all overlays (splines and control points) in the scene.
        Fixes line width and dash style.
        """
        # Removes overlays, but IGNORES both the base image AND the rubberband.
        for item in self.scene.items():
            if item != self.pixmap_item and item != self.rubber_band_line:
                self.scene.removeItem(item)

        # Draw the trails
        for i, trail in enumerate(self.trail_collection.trail):
            spline_points = trail.spline_points
            if not spline_points or len(spline_points) < 2:
                continue

            # Create the path for the spline
            path = QPainterPath()
            path.moveTo(QPointF(spline_points[0][0], spline_points[0][1]))
            for x, y in spline_points[1:]:
                path.lineTo(QPointF(x, y))

            # Set the pen (color, width, style)
            pen = QPen(QColor(trail.color))

            # Set the width in scene coordinates. It will scale with zoom.
            pen.setWidthF(trail.line_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)

            # Set the DashLine style for the strokes
            pen.setStyle(Qt.PenStyle.DashLine)
            # This pattern means a dash of 2 * pen_width, followed by a space of 2 * pen_width.
            pen.setDashPattern([2, 2])

            self.scene.addPath(path, pen)

            # Draw control points (handles)
            # Let's define a FIXED radius in screen pixels (e.g. 50 pixels).
            handle_radius = 50
            
            for x, y in trail.points:
                # To center the circle, the top-left point of its bounding rectangle
                # must be at (x - radius, y - radius).
                top_left_x = x - handle_radius
                top_left_y = y - handle_radius

                # The width and height of the rectangle must equal the diameter (radius * 2).
                diameter = handle_radius * 2

                # Determines the fill color based on the stroke state.
                # - If the stroke is active, the dots are BLUE.
                # - If the stroke is not active, the dots are YELLOW.
                fill_color = Qt.GlobalColor.blue if i == self.trail_collection.active_trail_idx else Qt.GlobalColor.white

                # Create a specific pen for the outline
                outline_pen = QPen(Qt.GlobalColor.white)
                outline_pen.setWidth(5) # Set the thickness to 5 pixels
                # Set it to "cosmetic" to make it zoom independent
                outline_pen.setCosmetic(True)

                self.scene.addEllipse(
                    top_left_x, top_left_y, 
                    diameter, diameter,
                    QPen(Qt.GlobalColor.white), QBrush(fill_color)
                )

    def cancel_drawing(self):
        """
        Cancel and exit drawing mode.
        """
        if self.is_drawing_new_segment:
            self.siril.log("Exited drawing mode.", s.LogColor.GREEN)
            # Reset state variables
            self.is_drawing_new_segment = False
            self.new_segment_anchor = None
            # Hide the rubber band line
            self.rubber_band_line.setVisible(False)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handles keypress events for the main window.
        """
        # If the Escape key is pressed WHILE drawing, cancel.
        if event.key() == Qt.Key.Key_Escape and self.is_drawing_new_segment:
            self.cancel_drawing()
        else:
            # Pass the event to the parent for any other default behaviors
            super().keyPressEvent(event)

    def zoom_in(self):
        """ Zoom in around the view center """
        self.view.scale(1.3, 1.3)

    def zoom_out(self):
        """ Zoom out around the view center """
        self.view.scale(1/1.3, 1/1.3)

    def fit_to_preview(self):
        """ Fit the entire image in view while preserving aspect ratio """
        if hasattr(self, 'pixmap_item') and self.pixmap_item:
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_preview_to_siril(self):
        """ Remove previous overlays for a clean preview """
        self.siril.overlay_clear_polygons()

    def send_preview_overaly_to_siril(self):
        valid_trails = [t for t in self.trail_collection.trail if len(t.points) >= 2]
        if not valid_trails:
            self.siril.log("Warning - No valid trails (with at least 2 points) for preview.", s.LogColor.RED)
            return

        # Remove previous overlays for a clean preview
        self.siril.overlay_clear_polygons()

        # Calculate scale factors and get total image height
        img_h, img_w = self.full_image_data.shape[:2]
        prev_w, prev_h = self.preview_pil_image.size

        scale_x = img_w / prev_w
        scale_y = img_h / prev_h

        for trail in valid_trails:
            if not trail.spline_points:
                continue

            # Apply scaling to spline points to map them to the real image
            scaled_spline_points = [(p[0] * scale_x, p[1] * scale_y) for p in trail.spline_points]

            # Calculate line width in real image pixels
            scaled_width = max(1, int(trail.line_width * (scale_x + scale_y) / 2))

            # Create a list of FPoint objects for the polygon outline
            poly_points_list = create_thick_line_polygon(scaled_spline_points, scaled_width)

            # Invert the Y coordinate of each point to align with Siril's system.
            #flipped_poly_points = [FPoint(x=p.x, y=img_h - p.y) for p in poly_points_list]

            if len(poly_points_list) >= 3:  # if you wanted to use inverted coordinates, replace with: flipped_poly_points
                # Pack the RGBA color into a single integer
                packed_color = pack_rgba_color(255, 0, 0, 128) # Semi-transparent red

                # Create the sirilpy Polygon object using points with the correct Y
                polygon_overlay = s.models.Polygon(
                    points=poly_points_list, # if you wanted to use inverted coordinates, replace with: flipped_poly_points
                    color=packed_color,
                    fill=False
                )
                self.siril.overlay_add_polygon(polygon_overlay)

        self.siril.log("Trail preview sent to Siril overlay.", s.LogColor.GREEN)

    def select_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Image Files (*.fits *.fit *.fts);;All Files (*)")
        if not file_path:
            return

        self.reference_image_path = file_path
        self.siril.log(f"Selected reference image: {self.reference_image_path}", s.LogColor.GREEN)

        self.ref_file_label.setText(os.path.basename(file_path))
        self.ref_file_label.setToolTip(file_path)

    def apply_reference(self):
        if not self.reference_image_path:
            self.siril.log("Warning - No reference image selected.", s.LogColor.RED)
            return
        self.apply_changes(mode="reference")

    def restore_backup_frame(self):
        """
        Restores the original frame from the backup file if working in a sequence.
        """
        # Preliminary Checks
        if not self.siril.is_sequence_loaded():
            self.siril.log("No sequence loaded. Restore operation cancelled.", s.LogColor.RED)
            return
        
        seq = self.siril.get_seq()
        curr = seq.current

        if curr < 0 or curr >= seq.number:
            self.siril.log("Invalid current frame index. Cannot restore.", s.LogColor.RED)
            return

        # Building File Names and Checking Backup Existence
        try:
            orig_file = self.siril.get_seq_frame_filename(curr)
            base, ext = os.path.splitext(orig_file)
            backup_file = base + "-original_with_trail" + ext
        except Exception as e:
            self.siril.log(f"Could not determine filenames. Error: {e}", s.LogColor.RED)
            return
        
        if not os.path.exists(backup_file):
            self.siril.log(f"No backup found for frame {curr + 1} at '{backup_file}'.", s.LogColor.RED)
            return

        message = (
            f"You are about to restore the original data for frame {curr + 1} from the backup file.\n\n"
            f"This will overwrite any changes made to:\n'{os.path.basename(orig_file)}'\n\n"
            "IMPORTANT: After restoring, you must manually re-select the frame in Siril's sequence control panel to refresh the view.\n\n"
            "Are you sure you want to proceed?"
        )
        reply = QMessageBox.question(
            self,                                  # Parent widget
            "Confirm Restore",                     # Window title
            message,                               # Message text
            QMessageBox.StandardButton.Yes |       # Buttons to show (Yes and No)
            QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No          # Default button (No)
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.siril.log("User cancelled the restore operation.", s.LogColor.RED)
            return
        
        # Performing the Restore
        try:
            # # Load data from the backup
            # with fits.open(backup_file, mode="readonly") as hdul:
            #     backup_data = hdul[0].data.copy()
            # # Rewrite the current frame with the backup data
            # self.siril.set_seq_frame_pixeldata(curr, backup_data, prefix=None)

            # Use shutil.copy to overwrite the original file with the backup.
            # This is safer than reading and rewriting data with set_seq_frame_pixeldata.
            self.siril.log(f"Attempting to restore '{orig_file}' from '{backup_file}'...", s.LogColor.BLUE)
            shutil.copy(backup_file, orig_file)
            self.siril.log(f"Frame {curr + 1} file successfully restored on disk.\n\n"
                           "IMPORTANT: After restoring, you must manually re-select the frame in Siril's sequence control panel to refresh the view.\n\n",
                           s.LogColor.GREEN)
            
            # --- Eliminazione del file di backup ---
            try:
                os.remove(backup_file)
                self.siril.log(f"Backup file '{os.path.basename(backup_file)}' successfully deleted.", s.LogColor.GREEN)
            except Exception as e_del:
                self.siril.log(f"Could not delete backup file: {e_del}", s.LogColor.RED)

        except Exception as e:
            self.siril.log(f"Error restoring backup: {str(e)}", s.LogColor.RED)

    def apply_changes(self, mode):
        valid_trails = [t for t in self.trail_collection.trail if len(t.points) >= 2]
        if not valid_trails:
            self.siril.log("Warning - No valid trails (with at least 2 points) to apply.", s.LogColor.RED)
            return

        try:
            self.siril.update_progress("Process started.", 0.0)
            self.siril.overlay_clear_polygons()

            # Create mask
            img_h, img_w = self.full_image_data.shape[:2]
            prev_w, prev_h = self.preview_pil_image.size

            # Scale trail coordinates from preview to real size before creating the mask
            scale_x = img_w / prev_w
            scale_y = img_h / prev_h

            # Create empty mask
            mask = Image.new('L', (img_w, img_h), 0)
            draw = ImageDraw.Draw(mask)
            
            for trail in valid_trails:
                # Apply scaling and a vertical flip to the points, bringing the Y back to the correct system for the full-resolution image in Siril
                scaled_points = [(x * scale_x, img_h - (y * scale_y)) for (x, y) in trail.spline_points]
                scaled_width = max(1, int(trail.line_width * (scale_x + scale_y) / 2))
                
                # Dynamically extending mask endpoints
                border_threshold = 20.0  # Pixel tolerance for considering a point "on edge"
                extension_pixels = 500.0 # How many pixels to extend the line beyond the edge?

                # Create an editable copy of the point list
                extended_scaled_points = list(scaled_points)

                if len(extended_scaled_points) >= 2:
                    # --- Check the starting point of the trail ---
                    p0 = extended_scaled_points[0]
                    is_on_border_start = (p0[0] < border_threshold or 
                                        p0[0] > img_w - border_threshold or 
                                        p0[1] < border_threshold or 
                                        p0[1] > img_h - border_threshold)

                    if is_on_border_start:
                        p1 = extended_scaled_points[1]
                        # Calculate the direction vector (from point 0 to point 1)
                        dx = p1[0] - p0[0]
                        dy = p1[1] - p0[1]
                        length = math.hypot(dx, dy)
                        if length > 0:
                            # Move point p0 backwards along the vector
                            new_x = p0[0] - (dx / length) * extension_pixels
                            new_y = p0[1] - (dy / length) * extension_pixels
                            extended_scaled_points[0] = (new_x, new_y)

                    # --- Check the end point of the trail ---
                    pN = extended_scaled_points[-1]
                    is_on_border_end = (pN[0] < border_threshold or 
                                        pN[0] > img_w - border_threshold or 
                                        pN[1] < border_threshold or 
                                        pN[1] > img_h - border_threshold)
                    
                    if is_on_border_end:
                        p_prev = extended_scaled_points[-2]
                        # Calculate the direction vector (from the penultimate to the last point)
                        dx = pN[0] - p_prev[0]
                        dy = pN[1] - p_prev[1]
                        length = math.hypot(dx, dy)
                        if length > 0:
                            # Move the point pN forward along the vector
                            new_x = pN[0] + (dx / length) * extension_pixels
                            new_y = pN[1] + (dy / length) * extension_pixels
                            extended_scaled_points[-1] = (new_x, new_y)

                # Use potentially extended points to draw the line
                draw.line(extended_scaled_points, fill=255, width=scaled_width, joint='curve')

            #mask_np = np.array(mask) > 0 # Boolean mask

            self.siril.update_progress("Step: Blurring mask...", 0.1)

            # Get user parameters
            blur_radius = self.blur_spinbox.value()
            blend_strength = self.blend_spinbox.value()
            noise_factor = 0.8

            # Apply blur and normalize mask between 0.0 and 1.0
            blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask_np = np.array(blurred_mask).astype(np.float32) / 255.0
            mask_np *= blend_strength

            self.siril.update_progress("Step: Applying correction...", 0.2)

            # Apply changes
            with self.siril.image_lock():
                # Save undo state.
                if self.siril.is_image_loaded():
                    self.siril.undo_save_state("Trail Removal")
                elif self.siril.is_sequence_loaded():
                    self.siril.log("Warning: A sequence is loaded, cannot save an UNDO state.", s.LogColor.RED)

                    seq = self.siril.get_seq()
                    curr = seq.current
                    tot  = seq.number

                    # Get the name of the current FITS file
                    orig_file = self.siril.get_seq_frame_filename(self.current_frame_index)
                    current_frame_filename = os.path.basename(orig_file)
                    base, ext = os.path.splitext(current_frame_filename)
                    backup_filename_base = f"{base}-original_with_trail{ext}"
                    
                    # Build a new name (pp_00002.fit -> pp_00002-original_with_trail.fit)
                    backup_file_path = os.path.join(os.path.dirname(orig_file), backup_filename_base)

                    if (self.current_seq_name is not None and self.current_frame_index is not None):
                        if seq.seqname != self.current_seq_name or curr != self.current_frame_index:
                            message = (
                                f"The sequence or frame currently selected in Siril:\n"
                                f"({seq.seqname}, frame {curr + 1})\ndo not match those loaded in the script:\n"
                                f"({self.current_seq_name}, frame {self.current_frame_index + 1}).\n\n"
                                "Please use 'Reload image from Siril' before applying corrections."
                            )
                            QMessageBox.critical(self, "Sequence/frame mismatch", message)
                            
                            self.siril.log("Mismatch detected: operation aborted.", s.LogColor.RED)
                            return

                    message = (
                        f"You are about to modify a frame from a sequence:\n\n"
                        f"File: {current_frame_filename} (Frame {curr + 1}/{tot})\n\n"
                        "This operation will permanently overwrite the original file. "
                        "The standard 'Undo' is not available for sequences.\n\n"
                        f"A backup of the original frame will be created as:\n'{backup_filename_base}'\n\n"
                        "Do you want to proceed?"
                    )
                    reply = QMessageBox.question(
                        self,
                        "Confirm Frame Overwrite",
                        message,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        self.siril.log("User cancelled the frame overwrite operation.", s.LogColor.RED)
                        return
                    
                    self.siril.log(f"Current frame: {curr + 1}/{tot}", s.LogColor.GREEN) 
                    self.siril.log(f"Current file: {self.siril.get_seq_frame_filename(curr)}", s.LogColor.GREEN)
                
                img_np = self.full_image_data.copy().astype(np.float32) # Work with floats for precision

                # ... blending logic for "black", "background", "reference" ...
                # All these operations will promote img_np to a float type (float32 or float64)

                if mode == "black":
                    self.siril.update_progress("Step: Apply black...", 0.3)
                    # Blend with black (0) (no blend)
                    # binary_mask = mask_np > 0

                    # if img_np.ndim == 3:    # RGB o multi-canale
                    #     binary_mask = binary_mask[..., None]  # per broadcasting
                    # img_np[binary_mask] = 0

                    # old with blend
                    if len(img_np.shape) == 3:  # RGB or 3-channels
                        mask_np = mask_np[..., None]  # expand for broadcasting
                        black = np.zeros_like(img_np)
                        img_np = img_np * (1.0 - mask_np) + black * mask_np
                    else:                       # mono image
                        img_np = img_np * (1.0 - mask_np)
                    
                    self.siril.update_progress("Step: Apply black...", 0.6)

                elif mode == "background":
                    self.siril.update_progress("Step: Apply background...", 0.3)
                    # --- IMPROVED BACKGROUND LOGIC FOR CFA AND MONO/RGB ---
                    is_color = img_np.ndim == 3 and img_np.shape[2] == 3
                    is_mono = img_np.ndim == 2

                    if is_color:
                        self.siril.log("RGB image detected. Processing standard background.", s.LogColor.BLUE)
                        fill_value = self.image_stats['median']
                        fill_image = np.ones_like(img_np) * fill_value
                        mask_np_expanded = mask_np[..., None]
                        
                        try:
                            bkg_rms_estimator = StdBackgroundRMS()
                            noise_std_dev = np.array([bkg_rms_estimator(self.full_image_data[..., c]) for c in range(3)])
                            self.siril.log(f"Estimated Noise (RMS per channel): {noise_std_dev}", s.LogColor.BLUE)
                            noise = np.random.normal(loc=0.0, scale=noise_std_dev * noise_factor, size=fill_image.shape)
                            fill_image += noise
                        except Exception as e:
                            self.siril.log(f"Warning: Could not add background noise. Error: {e}", s.LogColor.RED)
                        
                        self.siril.update_progress("Step: Apply background...", 0.4)
                        img_np = img_np * (1.0 - mask_np_expanded) + fill_image * mask_np_expanded
                    
                    elif is_mono:
                        cfa_pattern = None
                        header = None
                        try:
                            self.siril.log(f"Reading header from header FITS", s.LogColor.BLUE)

                            if self.current_seq_name is not None:
                                # We are in sequence mode
                                header = self.siril.get_seq_frame_header(self.current_frame_index, return_as='dict')
                            else:
                                # We are in single image mode
                                header = self.siril.get_image_fits_header(return_as='dict')

                            if header:
                                # 1. Ottieni il valore grezzo (potrebbe essere 'RGGB ' o None)
                                cfa_pattern_raw = header.get('BAYERPAT') or header.get('CFAIMAG')

                                # 2. Controlla SE hai ottenuto un valore prima di pulirlo
                                if cfa_pattern_raw:
                                    cfa_pattern = cfa_pattern_raw.strip() # Ora è sicuro chiamare .strip()
                                else:
                                    cfa_pattern = None # Era None e rimane None

                                bitpix = header.get('BITPIX')
                            else:
                                self.siril.log("Impossibile recuperare l'header FITS.", s.LogColor.RED)

                            if bitpix:
                                bit_depth_map = {
                                    8: "8-bit Integer",
                                    16: "16-bit Integer",
                                    32: "32-bit Integer",
                                    -32: "32-bit Float",
                                    -64: "64-bit Float"
                                }
                                bit_depth_str = bit_depth_map.get(bitpix, f"Unknown ({bitpix})")
                                self.siril.log(f"Image Bit Depth (BITPIX): {bit_depth_str}", s.LogColor.BLUE)
                            
                            self.siril.update_progress("Step: Apply background...", 0.4)
                        except Exception:
                            self.siril.log("Could not read FITS header. Assuming true monochrome.", s.LogColor.RED)

                        # List of supported CFA patterns
                        supported_patterns = ["RGGB", "GBRG", "GRBG", "BGGR"]

                        if cfa_pattern and cfa_pattern.upper() in supported_patterns:
                            # --- Logic for CFA images ---
                            self.siril.log(f"CFA image detected (Pattern={cfa_pattern}). Processing per-channel background.", s.LogColor.BLUE)
                            
                            # Create masks for individual colors
                            h, w = img_np.shape
                            r_mask, g_mask, b_mask = (np.zeros_like(img_np, dtype=bool) for _ in range(3))
                            p = cfa_pattern.upper()
                            
                            if p == "RGGB":
                                r_mask[0::2, 0::2] = True
                                g_mask[0::2, 1::2] = True
                                g_mask[1::2, 0::2] = True
                                b_mask[1::2, 1::2] = True
                            elif p == "GBRG":
                                g_mask[0::2, 0::2] = True
                                b_mask[0::2, 1::2] = True
                                r_mask[1::2, 0::2] = True
                                g_mask[1::2, 1::2] = True
                            elif p == "GRBG":
                                g_mask[0::2, 0::2] = True
                                r_mask[0::2, 1::2] = True
                                b_mask[1::2, 0::2] = True
                                g_mask[1::2, 1::2] = True
                            elif p == "BGGR":
                                b_mask[0::2, 0::2] = True
                                g_mask[0::2, 1::2] = True
                                g_mask[1::2, 0::2] = True
                                r_mask[1::2, 1::2] = True
                            else:
                                self.siril.log(f"CFA pattern '{p}' not supported. Treating as Mono.", s.LogColor.RED)
                                cfa_pattern = None # Return to mono logic
                            
                            # Calculate the background and noise for each channel
                            fill_r = compute_mmm_background(img_np[r_mask])
                            fill_g = compute_mmm_background(img_np[g_mask])
                            fill_b = compute_mmm_background(img_np[b_mask])
                            
                            noise_r = StdBackgroundRMS()(img_np[r_mask])
                            noise_g = StdBackgroundRMS()(img_np[g_mask])
                            noise_b = StdBackgroundRMS()(img_np[b_mask])

                            # Create the synthetic fill image with the CFA pattern
                            fill_image = np.zeros_like(img_np, dtype=np.float32)
                            fill_image[r_mask] = fill_r
                            fill_image[g_mask] = fill_g
                            fill_image[b_mask] = fill_b

                            # Add per-channel noise
                            fill_image[r_mask] += np.random.normal(0.0, noise_r * noise_factor, np.count_nonzero(r_mask))
                            fill_image[g_mask] += np.random.normal(0.0, noise_g * noise_factor, np.count_nonzero(g_mask))
                            fill_image[b_mask] += np.random.normal(0.0, noise_b * noise_factor, np.count_nonzero(b_mask))
                            
                            self.siril.update_progress("Step: Apply background...", 0.5)
                            img_np = img_np * (1.0 - mask_np) + fill_image * mask_np

                        else: # If it's a CFA image
                            if cfa_pattern: # Log if pattern was not supported
                                self.siril.log(f"CFA pattern '{cfa_pattern}' not supported. Treating as Mono.", s.LogColor.RED)

                            self.siril.log("True Monochrome image detected. Processing standard background.", s.LogColor.BLUE)

                            # Use global statistics already calculated at startup
                            fill_value = self.image_stats['median']
                            fill_image = np.ones_like(img_np) * fill_value
                            try:
                                data_for_stats = self.full_image_data.astype(np.float32)
                                noise_std_dev = StdBackgroundRMS()(data_for_stats)
                                noise = np.random.normal(loc=0.0, scale=noise_std_dev * noise_factor, size=fill_image.shape)
                                fill_image += noise
                            except Exception as e:
                                self.siril.log(f"Warning: Could not add background noise. Error: {e}", s.LogColor.ORANGE)
                            
                            self.siril.update_progress("Step: Apply background...", 0.5)
                            img_np = img_np * (1.0 - mask_np) + fill_image * mask_np

                elif mode == "reference":
                    try:
                        self.siril.update_progress("Step: Apply reference...", 0.3)

                        # Instantiate the robust background estimator from photutils
                        bkg_estimator = MMMBackground()

                        with fits.open(self.reference_image_path) as hdul:
                            ref_data = hdul[0].data.astype(np.float32)  # Read how to float for calculations

                        # If the reference image is CHW, convert it to HWC
                        if len(ref_data.shape) == 3 and ref_data.shape[0] in [1,3]:
                            ref_data = ref_data.transpose(1, 2, 0)

                        if ref_data.shape[:2] != img_np.shape[:2]:
                            self.siril.log(f"Error - Reference image dimensions do not match", s.LogColor.RED)
                            return
                        
                        self.siril.update_progress("Step: Balancing reference brightness...", 0.4)

                        # Calculate background using photutils for a more accurate brightness scaling
                        if len(img_np.shape) == 3: # Color Image
                            self.siril.log("Calculating background for RGB images...", s.LogColor.BLUE)
                            # Calculate background per channel using photutils
                            img_background = np.array([bkg_estimator(img_np[..., c]) for c in range(3)])
                            ref_background = np.array([bkg_estimator(ref_data[..., c]) for c in range(3)])
                            self.siril.log(f"Current img background (RGB): {np.round(img_background, 2)}", s.LogColor.BLUE)
                            self.siril.log(f"Reference img background (RGB): {np.round(ref_background, 2)}", s.LogColor.BLUE)

                            # Calculate scale factor and avoid division by zero
                            scale = np.divide(img_background, ref_background, out=np.ones_like(img_background), where=ref_background != 0)
                            self.siril.log(f"Calculated scale factors (RGB): {np.round(scale, 3)}", s.LogColor.GREEN)
                            ref_data *= scale

                        else: # Immagine è 2D (Mono o CFA)
                            # Dobbiamo leggere l'header QUI per capire se è Mono o CFA
                            cfa_pattern = None
                            header = None
                            try:
                                self.siril.log(f"Reading header from header FITS (for reference matching)", s.LogColor.BLUE)
                                header = None
                                if self.current_seq_name is not None:
                                    header = self.siril.get_seq_frame_header(self.current_frame_index, return_as='dict')
                                else:
                                    header = self.siril.get_image_fits_header(return_as='dict')

                                if header:
                                    cfa_pattern_raw = header.get('BAYERPAT') or header.get('CFAIMAG')
                                    if cfa_pattern_raw:
                                        cfa_pattern = cfa_pattern_raw.strip()
                            except Exception:
                                self.siril.log("Could not read FITS header. Assuming true monochrome.", s.LogColor.RED)

                            supported_patterns = ["RGGB", "GBRG", "GRBG", "BGGR"]

                            if cfa_pattern and cfa_pattern.upper() in supported_patterns:
                                # --- CASO CFA (2D) ---
                                self.siril.log(f"Calculating background for CFA (Pattern={cfa_pattern})...", s.LogColor.BLUE)
                                
                                # Crea maschere per i canali
                                h, w = img_np.shape
                                r_mask, g_mask, b_mask = (np.zeros_like(img_np, dtype=bool) for _ in range(3))
                                p = cfa_pattern.upper()
                                
                                if p == "RGGB":
                                    r_mask[0::2, 0::2] = True
                                    g_mask[0::2, 1::2] = True
                                    g_mask[1::2, 0::2] = True
                                    b_mask[1::2, 1::2] = True
                                elif p == "GBRG":
                                    g_mask[0::2, 0::2] = True
                                    b_mask[0::2, 1::2] = True
                                    r_mask[1::2, 0::2] = True
                                    g_mask[1::2, 1::2] = True
                                elif p == "GRBG":
                                    g_mask[0::2, 0::2] = True
                                    r_mask[0::2, 1::2] = True
                                    b_mask[1::2, 0::2] = True
                                    g_mask[1::2, 1::2] = True
                                elif p == "BGGR":
                                    b_mask[0::2, 0::2] = True
                                    g_mask[0::2, 1::2] = True
                                    g_mask[1::2, 0::2] = True
                                    r_mask[1::2, 1::2] = True

                                # Calcola 3 mediane per l'immagine TARGET
                                img_bkg_r = bkg_estimator(img_np[r_mask])
                                img_bkg_g = bkg_estimator(img_np[g_mask])
                                img_bkg_b = bkg_estimator(img_np[b_mask])
                                
                                # Calcola 3 mediane per l'immagine REFERENCE
                                ref_bkg_r = bkg_estimator(ref_data[r_mask])
                                ref_bkg_g = bkg_estimator(ref_data[g_mask])
                                ref_bkg_b = bkg_estimator(ref_data[b_mask])

                                # Calcola 3 fattori di scala
                                scale_r = img_bkg_r / ref_bkg_r if ref_bkg_r != 0 else 1.0
                                scale_g = img_bkg_g / ref_bkg_g if ref_bkg_g != 0 else 1.0
                                scale_b = img_bkg_b / ref_bkg_b if ref_bkg_b != 0 else 1.0
                                
                                self.siril.log(f"Calculated scale factors (CFA): R={scale_r:.3f}, G={scale_g:.3f}, B={scale_b:.3f}", s.LogColor.GREEN)

                                # Applica i fattori di scala *solo* ai pixel corretti
                                ref_data[r_mask] *= scale_r
                                ref_data[g_mask] *= scale_g
                                ref_data[b_mask] *= scale_b

                            else:
                                # --- CASO MONO PURO (2D) ---
                                if cfa_pattern:
                                    self.siril.log(f"CFA pattern '{cfa_pattern}' not supported. Treating as Mono.", s.LogColor.RED)
                                self.siril.log("Calculating background for mono image...", s.LogColor.BLUE)
                                
                                img_background = bkg_estimator(img_np)
                                ref_background = bkg_estimator(ref_data)
                                self.siril.log(f"Current img background: {img_background:.3f}, Reference img background: {ref_background:.3f}", s.LogColor.BLUE)

                                scale = img_background / ref_background if ref_background != 0 else 1.0
                                self.siril.log(f"Calculated scale factor: {scale:.3f}", s.LogColor.GREEN)
                                ref_data *= scale

                        self.siril.update_progress("Step: Applying reference...", 0.5)

                        # Check the ORIGINAL data type, not the float copy's
                        if np.issubdtype(self.original_image_dtype, np.integer):
                            # Use the ORIGINAL data type information
                            info = np.iinfo(self.original_image_dtype)
                            max_val = info.max
                            ref_data = np.clip(ref_data, 0, max_val)
                        else: # For float data
                            # For floats, clipping isn't strictly necessary if the data is already normalized,
                            ref_data = np.clip(ref_data, 0, None) # Clip only at 0, no upper limit

                        self.siril.update_progress("Step: Blending images...", 0.6)

                        # Ensure data types match for the final blend
                        # La fusione avverrà tra due array float32
                        mask_np_expanded = mask_np[..., None] if len(img_np.shape) == 3 else mask_np
                        
                        # Use blending to combine images
                        img_np = img_np * (1.0 - mask_np_expanded) + ref_data * mask_np_expanded

                    except Exception as e:
                        self.siril.log(f"Error - Could not use reference image: {e}", s.LogColor.RED)
                        return

                self.siril.update_progress("Step: Finalizing image...", 0.8)
                # Final reconversion to the original data type
                # clipping values if converting to an integer type
                if np.issubdtype(self.original_image_dtype, np.integer):
                    info = np.iinfo(self.original_image_dtype)
                    final_image_data = np.clip(img_np, info.min, info.max).astype(self.original_image_dtype)
                else:
                    final_image_data = img_np.astype(self.original_image_dtype)

                # Transpose axes if it's a color image (channels last)
                if final_image_data.ndim == 3:
                    final_image_data = final_image_data.transpose(2, 0, 1)
                
                # Update the image in Siril
                if self.siril.is_image_loaded():
                    self.siril.set_image_pixeldata(final_image_data)
                elif self.siril.is_sequence_loaded():
                    # !!! Siril directly overwrites the FITS of the current frame. !!!
                    # To provide the user with a "fallback," it's a good idea to first save a copy of the original.
                    self.siril.log("Creating backup copy before overwriting frame...", s.LogColor.GREEN)

                    # Difference Between os. rename and shutil. move in Python
                    # https://www.geeksforgeeks.org/python/difference-between-os-rename-and-shutil-move-in-python/
                    # -->> You want to preserve file metadata and attributes.

                    try:
                        # If the backup doesn't already exist, we create it
                        if not os.path.exists(backup_file_path):
                            shutil.copy(orig_file, backup_file_path)
                            self.siril.log(f"Backup created: {backup_file_path}", s.LogColor.GREEN)

                        # Now overwrite the current frame with the changed data
                        # TODO: for compatibility with sirilpy updates and equivalence with v2.0.3 I have set prefix=None
                        # This should be reviewed by the script author to confirm whether overwriting the sequence is optimum
                        # or if he prefers that a new sequence with a different prefix is made.
                        self.siril.set_seq_frame_pixeldata(curr, final_image_data, prefix=None)
                        self.siril.log(f"Frame {curr + 1} overwritten with corrected data.", s.LogColor.BLUE)

                    except Exception as e:
                        self.siril.log(f"Error while creating backup or saving frame: {e}", s.LogColor.RED)
                        QMessageBox.critical(self, "Error", f"Error while creating backup or saving frame: {curr}.\n\n{e}")
    
                self.siril.log("Trail removal performed, image updated.", s.LogColor.GREEN)
        finally:
            self.siril.update_progress("Process complete.", 1.0)
            self.siril.reset_progress()

    def closeEvent(self, event: QCloseEvent):
        """
        This event handler is called automatically by PyQt when the user closes the window.
        It handles the cleanup and disconnection from Siril.
        """
        try:
            if self.siril:
                self.siril.overlay_clear_polygons()
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.BLUE)
                self.siril.disconnect()
        except Exception as e:
            print(f"An error occurred during cleanup: {e}")
        
        event.accept()

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
def main():
    try:
        qapp = QApplication(sys.argv)
        qapp.setApplicationName(f"Satellite Trail Removal Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        icon_data = base64.b64decode("""/9j/4AAQSkZJRgABAgAAZABkAAD/7AARRHVja3kAAQAEAAAAZAAA/+4AJkFkb2JlAGTAAAAAAQMAFQQDBgoNAAADDAAACRsAAAsYAAANX//bAIQAAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQICAgICAgICAgICAwMDAwMDAwMDAwEBAQEBAQECAQECAgIBAgIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMD/8IAEQgAQABAAwERAAIRAQMRAf/EALIAAAIDAQADAAAAAAAAAAAAAAAIBgcJBQEDBAEBAQEAAAAAAAAAAAAAAAAAAAECEAACAgICAQMFAAAAAAAAAAAFBgQHAgMQAREgMAgAUGAVNxEAAQQBAwMDAwMCBwAAAAAABAECAwUGERITACEUIhUHEDIjMUFRIENCUjN0JbUWEgEAAAAAAAAAAAAAAAAAAABgEwEAAgICAwEBAQEAAAAAAAABABEhMRAgQVFhcTCRwf/aAAwDAQACEQMRAAAByDAAAsJOQRRQlCNzZpfc0ivtI2Y+Z2FyJrBrGoFzWS4OZ2uEtdKAX0mwusUAqG50uKgDNJpvrFcLciLnKoaqzNAyKaZaxUssppVZa+VWJQbq52HuEemnc1FIlQrOlvUOuTpOUc4+VfKR1f/aAAgBAQABBQL0hVY0f1nBO0EX4ChSrEUFUfY8Msv0XSw6FaXxogQ4CBqXVsI+QkkiocV/oY97fRZJGcj+SmLwxlAyQHQ+1+wgQRRtYzY/lKnFVjtc+SCmYkWxfjMNKMT4/wA4TzTwvsvYRTtexGnszw7Yws0DHFdKnJSHdgsSFtHioROg5ZAzNGsk3O1JYJb31lpjsFdQd2mtrVBCwsnih/62Al2JIIWAkND8o/tNcGw41hEH6mbGlkJvfEAhPFS57w6lYopqZwWmIeOD8o5crEhEC5Utx//aAAgBAgABBQL8A//aAAgBAwABBQL3PPq79nx9ePsn/9oACAECAgY/AgH/2gAIAQMCBj8CAf/aAAgBAQEGPwL+mUiuGjUMaaOAw8osQIEJ0scszXFlFzQxwR8UD3ar/l/nt0fUTTQkyAEOh8kbl8clnZ0REHNHFKsM8Tkc3Vqdl+olLSBTWNoc97BQ4NvJKscT55O73NY1kUMTnucqo1rWqq9uhG2WJgTQPleKTGTb1tkwVk8UkMhsoGPXbreZ1a1/OjI2uVXRom132rALeMNs7F6Ro4vJCr3FZCZl7O8EDlpU4XvX0t/M5NNN6rr1Je4BM+Bo35T6azOYosYfbmOEsy3tfDGFFrJK2d79WIqtdqiNcPja2AJ7LsC6jFuwBbAZMnsijK+vsaqiddVw1Hcywwr4bJYipDNjpGxQwOe/fd2DnsOtKungsgKqb3J5uLKUlLSNryZRYhJxHAkSRQNDsJJ+No+5vpVWx/SlbiZ0NbkDJ5Zq40mdgw8UkI00jmTzSskh4yY2rDteisk5Ni9ndWgZ2MiRZEFUlrKPujs8WMHfdREm2FWDYpLOCW86dnb1/i09XbrbWuOpE2q1kdSbLAAxqoqK1KSbyKB7VRf0eK5Op3CyCk1aQTuLYE4SgMYEyNVJfJWkcmHWkj2rpI9YqxWt/udA2tXkRWVYFVFTQhSpIRBJixp80U0ox1NJNI2rlNmlY7mh1iI3Mfr+SPdX1drbmnh1iSNFYTM+R+yTi0bPK5eQpIGwNbFyK7iYiMbo1ERPrPnlJeVIFljRgEcdSZOvm3MFjzwFxwAsaqkhRsRGz6qzRsmrXI9qdD2NkJkdXbvgjo20kJgJtGQYbNzjkwwKQKa4h0kHH5DoGxwsftc71J1a4vQhrTnjTIFdWFmPVHSTRxRv/wCOEq1fcVI4SoS5XufITLJu+5qJp0tJd2kftd80W8JGEBoA1tdXt8Uq0mpw4CSZWuCZo0lyvakbO2iN/oohPFqDEb7gW6C8CIs69WBVpZT5H1YssE9lNC2LdFCjkR8iJr216wfMMgx1+QEg/Jfs0rYfjh+JWZVcVjFkXFB/55xZslxEBZRxEMfu0dxuYjdUdvw/OoLQM8UXMm42lZd/GI2FFjuu4EVZmQOV7bhjRYlRH/2ZNHJ/i0+dLfIcYosihw+1xQCtBIBghaRy2a+K6wnbG6UlGHlNfLr/AK0TONe3RubYdgmP22V3WfPrrgKrxT3QetpRqMVwwglazyH1le+dEc937vfqrv00y2uoxxxa2EoJ8QwuiDwTlVIBZ0UTG+iJjD55E2Jokf2oiafXFasic4WIkwjUisMmrz4XQV5ZEcgxg6tmHkbJEndvfo7DqXJ/mIHJqf3A+osbzJzCwQ7Wq5Q0NG0trJ0b2c7k3bIXrC9zWua53WB2Hy3kfyZll3k9ZBk1a4K6nKFoBimDzDOi9wsYpYp2Rzt3PjWR6vYuiIm3X5bpbHJsnuapcJhzUN092ayS0ma0uavdkbEckVxMCUF6HSN/ZHIjV7J8o5TX3OQ09tQJQNDkpLs+qilaaW+KZpsQUsSFojft3fb1hBFZERG7JvjrGsqtHFGEHTEW9w+wcaQ6Yl8j/wAnE3snbX64Z/vTf+osOs0Hy/CcdxLF1ociSLJaZkNbZv0dtHfISy4Of6w1fMruOPY9m7VP0X4ZOw+qkvRBcDq6cuYOYbaLYDDhDzwkc00XEkU8T2ucvparF1Xr5ImGUM0rG/iOrjKHlTnE88FLI7xCmNdGr4nxys3t1au137dfLUp9LjVMtZHj0cbccrpq5k6E2THOUpJjTOV0fD6dNumq9YQ8+nsqjxvj3Gawb3GBIPcR66IiGKxD0c7eGQzTaq6O7d0+sJ9YaXXHDKrhzQCZhC4HOY6NzoSR3xzRK5j1Tsqdl6kBtMvyiyCm7TBn5BbGCyondEkHILkifov8p1INSZHfU48zt00FVb2FfDK7TbukiEIiY92n8p0e8C5tQnWsUkNo4SxLGdZQzK5Zoj1hmYpkUqvXc2Tci69G1olnYC11lxe4145pMIVhwO3QeaLHI2Arhd3bvRdq/p0H7nYm2HtwcNcB5hMxPhgD68AY3K93CNDu9LG6NT6f/9oACAEBAwE/IeuS6ex1VplSILoteqGII3mdqbXIoVNtyiKRQgBYlR+PhVyKjhrPbmwJyeCPl4JJv3Bz2hRlKdEVXGVsk7+L1gSLiF1GiXlkcHJSZGYP9A9PYGyIiNb8aYe7Mt6Sj2JSUxqi0bFVDmf1WTQMBm/sWQWM2V42WEkF1cPKL63Z6s9Fm6ZJ48SrXklC5J3t69JMALjovImtBjyST6QlL5Q7vT0dCr0YoKNPZDsFb3IzIr1tXcnTWcwNLAtG/GHTQbjZ7rvNAjN9iuoKWZOcd9BabaZaYit2LqkdKgAcbk/2CO1XilnmPl9JaSmi/CJBDlo2sceSwELJshekIIaDy+CGsK2hiG2mibINYPkbfXOQqUC3pXJbUhrDu1LgJAkMXEi8DGKKqFS7vaGr/wBhWV2tI/JzKfqiNUbLw14eDZRf3L1kLL4Qf2hQEClcsaWE9sRGUKYWWxLbmv7nHGrRqMI99GnAu4XY2zd9tTWby+3lIATxRmPebmwcHH//2gAIAQIDAT8h/lUoidj+Nxep1d8nH7K4eTfPmeP5/wD/2gAIAQMDAT8h/lZLYe0Ycsscj/vV9cCKOjqfJ8nqeYa5dTJmZ8S9R3Dl1MQaXiqYdKO3/9oADAMBAAIRAxEAABAAAdgUqCAYtHgDKIAajugI+OAF+XAD8gj/2gAIAQEDAT8Q62yO6PVzaywbIg+LaxBMsquCLw8uhwFcco6PSGqlIE44cebAic/4qUeqgUTPrPJV2s+Cnx1zCDFD/pjAuRxNagZ52RMmOWo61r8ImwAmMjtxYJJBABJLI5SL4/CBkwDAqD4gXTgxp4KHEL6bTAd1VdceBJXF/RurwXhQ7kVlyiKg2CELyINZEIp4nhQjx5hAHeBdLzc1JQxzaE0nJyCT7/GvFuigKGLYckMYnAxTzaNmMZOeYxOe/vqGg4Of1VcXg6LpSN2Y4wCM+Dlj2lMlzGlBjKlMwmD1RItmlH/LOhlRAhe0xiWynvjQ0QW+kvN4NRp4We+s9kIS9EFrLNvQS6CPE1VdyNSn/dvCGkbvmUIrC60RTTdg0zFI61AwsNaVgvub7YwAoHpUwYeWxxu4DGF8pK7dexHq+LqBEX/xSDfAFEszAEYZDCus3Rq6xEpuw8O3KLyy72Zjwgk8z8bGYYataAOpBzgovj//2gAIAQIDAT8Q60sSmud6loQ8pCgqNV95LvG5RfsqVEQ+S16GMkDuJoivTaYwvuNmf+S9rDVhm56ORbCnGbmKMrmyQ0sFV+dAvyYiKFep5fku1Hx+dLZaS2Xz/9oACAEDAwE/EOqhBsvlQLdRq3F9alzUWW4ka5pWdShvEt8yx/YC0lQAZOlHLTFGtQkthZZvo6gvIxj3Cs08e7geDV3EMFwEai+XSZgsFTJSoE0wLupoHTEt/HnfEwptgWPcqw+soNX5nlXvne5RoIg7JRKNwA1x/9k=""")
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)
        app_icon = QIcon(pixmap)
        qapp.setWindowIcon(app_icon)

        qapp.setStyle("Fusion")

        # Define a Qt Style Sheet (QSS)
        stylesheet = """
            QPushButton[class="accent"] {
                background-color: #3574F0; /* A nice blue color */
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton[class="accent"]:hover {
                background-color: #4E8AFC; /* A slightly lighter blue for hover */
            }

            QPushButton#helpButton {
                background-color: #f0f0f0; /* Light gray */
                color: #005A9C;
                font-weight: bold;
            }
            QPushButton#helpButton:hover {
                background-color: #e0e0e0; /* Darker on hover */
            }
        """
        # Apply the stylesheet to the entire application
        qapp.setStyleSheet(stylesheet)
        # Now that the application context exists, create the main widget.
        app = TrailRemovalAPP()
        app.show()

        sys.exit(qapp.exec())

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
