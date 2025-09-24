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
# This tool provides two powerful methods for correcting defects in astronomical images
# and includes a utility for creating a definitive star mask after cleanup. It is designed
# for realistic, non-destructive results, ideal for 32-bit images.
#
# 1. Patch Inpainting: This method intelligently fills a selected area using a two-phase process.
#    - Structural Correction: The area is smoothly filled using `cv2.inpaint` with the
#      Navier-Stokes algorithm, which propagates information from the edges inward
#      to create a structurally coherent base.
#    - Texture Synthesis: To reintegrate the natural "grain", the power spectrum of a
#      nearby background area is analyzed. A new random texture with the same spectral
#      "footprint" is generated, ensuring the patch's texture matches its context.
#    - Context Factor: This parameter controls the size of the surrounding area used for
#      texture analysis. Higher values (e.g., 3.0) sample from a wider region, useful
#      if the immediate edges are compromised, but be careful not to include unwanted
#      structures. Lower values (e.g., 1.5) use only the closest context, ideal for defects
#      in uniform areas.
#
# 2. Clone Stamp: This tool allows for the seamless cloning of a clean "source" area onto
#    one or more "destination" areas.
#    - Smart Blending: It performs a robust local gain and bias calibration by analyzing
#      pixels in a ring around the destination. This ensures the cloned patch perfectly
#      matches the brightness and color balance of the target location.
#    - Advanced Feathering: A soft, invisible transition is created using a distance
#      transform-based feather with a smooth-step profile to enhance realism.
#
# 3. Star Mask Generation: An automated tool to create the definitive star mask after
#    removing imperfections or stars missed by star-removal tools like Starnet.
#    It uses PixelMath to subtract the cleaned starless image from the original,
#    perfectly isolating the complete star field for later reintegration.
#
# For advanced control in Inpainting mode, users can experiment with different spectral
# analysis windows (e.g., `np.blackman` is used in the code) to finely vary the
# character of the synthesized texture.
#
# Versions:
# 1.0.0 - Initial release with Patch-Based Inpainting
# 1.0.1 - Added HELP button
#         The image must be a RGB image
#         Using this script on non-RGB images is not recommended,
#         as it can produce unnatural color splotches.
#         NOTE:
#         This tool is intended for use on STARLESS images. Although you can use it on images that
#         contain stars, be aware that any star close to your selection will skew the average color
#         calculation. This will cause the inpainting to appear as a 'smudge' of color that is
#         significantly brighter than its surroundings.
# 1.0.2 - Improved shading, added radius control in the clone tool for color calculation
# 1.0.3 - Many improvements
# 1.0.4 - Changed dropdown select to Radiobutton due to display issues when "Keep window on top" is enabled.
# 2.0.0 - Ported to PyQt6
# 2.0.1 - Added Icon App
# 2.0.2 - Added the new Quilt Inpainting method, the best method for realistic results
#

VERSION = "2.0.2"

# --- Core Imports ---
import sys
import os
import base64
import traceback

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s
    
    if not s.check_module_version('>=0.6.37'):
        print("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    # Import Siril GUI related components
    from sirilpy import SirilError

    s.ensure_installed("PyQt6", "numpy", "astropy", "opencv-python")

    # --- PyQt6 Imports ---
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton,
        QPushButton, QMessageBox, QTabWidget, QGroupBox, QSlider, QFileDialog,
        QTextEdit, QDialog, QCheckBox, QGridLayout, QStyle, QSizePolicy
    )
    from PyQt6.QtGui import QFont, QCloseEvent, QIcon, QPixmap
    from PyQt6.QtCore import Qt, QTimer, QSize

    # --- Imports for Image Processing ---
    import cv2
    import numpy as np

    from astropy.io import fits
    # from photutils.background import MMMBackground, StdBackgroundRMS
    # from PIL import Image, ImageTk, ImageDraw, ImageFilter

except ImportError:
    print("Warning: sirilpy not found. The script is not running in the Siril environment.")

class HelpWindow(QDialog):
    """
    A dialog box that displays formatted help text.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Program Instructions")
        self.resize(750, 600)

        instructions_text = (
            f"<h2>Patch Inpainting Tool v{VERSION} - (c) Carlo Mollicone AstroBOH</h2>"
            f"""
            <p>This tool provides two powerful methods for correcting defects in astronomical <b>RGB STARLESS</b> images and includes a utility for creating a definitive star mask after cleanup.<br/>
            It is designed for realistic, non-destructive results, ideal for 32-bit images.</p>

            <p>Using this script on non-RGB images is not recommended, as it can produce unnatural color splotches.</p>
            <p><b>NOTE:</b><br/>
            This tool is intended for use on <b>RGB STARLESS</b> images.<br/>
            Although you can use it on images that contain stars, be aware that any star close to your selection will skew the average color calculation.<br/>
            This will cause the inpainting to appear as a 'smudge' of color that is significantly brighter than its surroundings.</p>
            
            <h3>1 Texture Synthesis Inpainting (Quilt Inpainting):</h3>
            <p>
            This advanced method reconstructs the missing region by synthesizing new texture from the image itself, 
            using an improved <b>patch quilting</b> approach combined with multi-scale blending.
            </p>
            <ul>
                <li><b>Patch-Based Synthesis</b>: The missing area is filled with overlapping patches sampled from a surrounding 
                annular region (controlled by the <b>Context Factor</b>). Each patch is selected using a similarity score that 
                combines intensity matching with gradient consistency, ensuring coherent local structure.</li>

                <li><b>Seamless Patch Fusion</b>: Instead of hard seams, every patch is merged into its neighbors using smoothstep 
                ramps applied on all four sides (left, right, top, bottom). This guarantees soft transitions and avoids visible 
                block boundaries.</li>

                <li><b>Adaptive Normalization</b>: After all patches are placed, their contributions are accumulated and 
                normalized, producing a uniform mosaic. Pixels outside the mask are preserved from the original image.</li>

                <li><b>Pyramid Blending</b>: As a final step, a Laplacian pyramid blend integrates the synthesized mosaic with 
                the original image, correcting residual seams and harmonizing the global color balance.</li>
            </ul>
            <p>The following parameters allow fine-tuning of the inpainting process:</p>
            <ul>
                <li><b>Patch Size</b>: Defines the side length of the square texture patches.<br/>
                    - Larger values capture more structure but reduce flexibility.<br/>
                    - Smaller values adapt better to local variations but may appear noisier.
                </li>
                <li><b>Overlap Divisor</b>: Controls how much patches overlap, expressed as a fraction of patch size.<br/>
                    - <b>Higher overlap</b>: smoother transitions but slower.<br/>
                    - <b>Lower overlap</b>: faster but may leave visible seams.
                </li>
                <li><b>Blend Feather</b>: The feather width (in pixels) applied at patch borders.<br/>
                    - Increase this value if you notice a hard edge on the patch after processing.<br/>
                    - Larger values create a wider, softer feather to smooth the seam, but can slightly blur the texture at the border.
                </li>
                <li><b>Pyramid Levels</b>: Number of scales in the Laplacian pyramid blending.<br/>
                    - <b>More levels</b>: smoother global integration.<br/>
                    - <b>Fewer levels</b>: sharper details but higher risk of seams.<br/>
                    - Recommended range: 3-5 levels.
                </li>
            </ul>

            <h3>2 Patch Inpainting (CV2 Inpainting):</h3>
            <p>This method intelligently fills a selected area using a two-phase process.</p>
            <ul>
                <li><b>Structural Correction</b>: The area is smoothly filled using <b>cv2.inpaint</b> with the <b>Navier-Stokes</b> algorithm, which propagates information from the edges inward to create a structurally coherent base.</li>
                <li><b>Texture Synthesis</b>: To reintegrate the natural 'grain', the power spectrum of a nearby background area is analyzed. A new random texture with the same spectral 'footprint' is generated, ensuring the patch's texture matches its context.</li>
                <li><b>Context Factor</b>: This parameter controls the size of the surrounding area used for texture analysis.<br/>
                -- <b>Higher values</b> (e.g., 3.0) sample from a wider region, useful if the immediate edges are compromised, but be careful not to include unwanted structures.<br/>
                -- <b>Lower values</b> (e.g., 1.5) use only the closest context, ideal for defects in uniform areas.</li>
            </ul>

            <h3>3 Clone Stamp:</h3>
            <p>This tool allows for the seamless cloning of a clean <b>source</b> area onto one or more <b>destination</b> areas.</p>
            <ul>
                <li><b>Smart Blending</b>: It performs a robust local gain and bias calibration by analyzing pixels in a ring around the destination. This ensures the cloned patch perfectly matches the brightness and color balance of the target location.</li>
                <li><b>Advanced Feathering</b>: A soft, invisible transition is created using a distance transform-based feather with a smooth-step profile to enhance realism.</li>
                <li><b>Context Factor for Smart Blending</b>: This slider controls the width (in pixels) of the ring used for color calibration.<br/>
                - <b>Value 0</b>: Disables color calibration. The patch will not be adjusted and may not match the destination.<br/>
                - <b>Values > 0</b>: Use a ring of that pixel width to calculate the blend. Larger values average colors over a wider area, which is useful for gradients.<br/>
                - <b>Recommendation</b>: A value of <b>2</b> provides a reliable and precise blend for most cases.</li>
            </ul>

            <h3>3 Star Mask Generation:</h3>
            <p>An automated tool to create the definitive star mask after removing imperfections or stars missed by star-removal tools like Starnet.<br/>
            It uses <b>PixelMath</b> to subtract the cleaned starless image from the original, perfectly isolating the complete star field for later reintegration.<br/>
            The final star mask will be saved as a new FITS file in your current Siril working directory.<br/>
            Look for a file named:</p>
            <blockquote style="font-family: 'Courier New', monospace;">
                <b>starmask_{{base_name}}_InpaintTool.fits</b>
            </blockquote>
            <p>, which will be ready to use in your processing workflow.</p>
            """
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
class PatchInpaintingTool(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"Patch Inpainting Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        # --- State Variables ---
        self.patch_size_steps = [16, 32, 64, 128, 256]
        self.overlap_divisor_steps = [2, 4, 8, 16, 32]
        self.clone_source_rect = None
        self.clone_source_poly = None
        self.help_window = None # To keep a reference to the help window

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
            self.close()
            return
        
        # Check if an image is loaded
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image loaded")
            QTimer.singleShot(0, self.close)
            return

        shape_image = self.siril.get_image_shape()
        if shape_image[0] != 3:
            self.siril.error_messagebox("The image must be a RGB image.")
            QTimer.singleShot(0, self.close)
            return

        self.siril.overlay_clear_polygons()
        self.create_widgets()
        
    def create_widgets(self):
        """ Create all the elements of the graphical interface. """
        
        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # --- Top bar for 'Keep on top' checkbox ---
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addStretch(1)
        self.on_top_check = QCheckBox("Keep window on top")
        self.on_top_check.toggled.connect(self._toggle_on_top)
        top_bar_layout.addWidget(self.on_top_check)
        main_layout.addLayout(top_bar_layout)
        
        # --- Main Tab Widget ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create pages for the tab widget
        self.correction_tab = QWidget()
        self.starmask_tab = QWidget()
        
        self.tab_widget.addTab(self.correction_tab, "Inpainting / Clone Stamp")
        self.tab_widget.addTab(self.starmask_tab, "Create Star Mask")

        # Populate the tabs
        self.create_correction_tab()
        self.create_starmask_tab()

        # --- Process Buttons and Status Label (common to all tabs) ---
        process_layout = QHBoxLayout()

        # Process button
        process_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        self.process_button = IconTextButton("PROCESS", process_icon, self)
        # self.process_button.setObjectName("helpButton")
        self.process_button.setProperty("class", "accent")
        self.process_button.clicked.connect(self._process_image)

        # HELP button
        # Use the custom IconTextButton instead of QPushButton
        help_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogHelpButton)
        self.help_button = IconTextButton("HELP", help_icon, self)
        self.help_button.setObjectName("helpButton")
        self.help_button.clicked.connect(self.show_help)

        process_layout.addWidget(self.process_button, 1) # <-- ADD , 1
        process_layout.addWidget(self.help_button, 1)    # <-- ADD , 1
        
        main_layout.addLayout(process_layout)
        main_layout.addSpacing(10)

        # Status Label
        self.status_label = QLabel("Ready. Select a method.")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        main_layout.addSpacing(10)
        
        # --- Final UI Setup ---
        # Window size and centering
        self.setFixedSize(500, 570)
        self.center_window()

        self._update_ui_for_mode() # Initial setup

    def create_correction_tab(self):
        layout = QVBoxLayout(self.correction_tab)
        layout.setContentsMargins(10, 10, 10, 10) # Top padding

        # --- Instructions GroupBox ---
        instr_group = QGroupBox("Instructions")
        instr_layout = QVBoxLayout(instr_group)
        self.instructions_label = QLabel()
        self.instructions_label.setWordWrap(True)
        instr_layout.addWidget(self.instructions_label)
        layout.addWidget(instr_group)

        # --- Instructions Texts (same as original) ---
        self.instructions_texts = {
            "Inpainting": (
                "<ol style='margin-left: -25px;'>"
                "<li>Click 'Add Selection' and draw a selection on the image in Siril to define the area to be filled.</li>"
                "<li>While it is possible to select multiple areas, processing them one at a time is recommended.</li>"
                "<li>Set the inpainting parameters.</li>"
                "<li>Click 'Process' to start.</li>"
                "</ol>"
            ),
            "Clone Stamp": (
                "<ol style='margin-left: -25px;'>"
                "<li>In Siril, select the area that will act as the source (choose a region with texture similar to the destination).</li>"
                "<li>Click 'Set Source' to register the chosen source area.</li>"
                "<li>Click 'Add Selection' and draw the target area(s) where the source will be cloned.</li>"
                "<li>Adjust the Feather Radius if needed.</li>"
                "<li>Click 'Process' to apply the clone.</li>"
                "</ol>"
            )
        }
        
        # --- Correction Method GroupBox ---
        method_group = QGroupBox("Correction Method")
        method_layout = QHBoxLayout(method_group)
        self.quilt_inpainting_radio = QRadioButton("Quilt Inpainting")
        self.quilt_inpainting_radio.setChecked(True)
        self.quilt_inpainting_radio.clicked.connect(self._update_ui_for_mode)
        self.inpainting_radio = QRadioButton("CV2 Inpainting")
        self.inpainting_radio.clicked.connect(self._update_ui_for_mode)
        self.clone_stamp_radio = QRadioButton("Clone Stamp")
        self.clone_stamp_radio.clicked.connect(self._update_ui_for_mode)
        method_layout.addStretch()
        method_layout.addWidget(self.quilt_inpainting_radio)
        method_layout.addSpacing(30)
        method_layout.addWidget(self.inpainting_radio)
        method_layout.addSpacing(30)
        method_layout.addWidget(self.clone_stamp_radio)
        method_layout.addStretch()
        layout.addWidget(method_group)

        # --- Parameters GroupBox ---
        param_group = QGroupBox("Parameters")
        param_layout = QGridLayout(param_group)
        param_layout.setColumnStretch(1, 1) # Allow sliders to expand

        # Slider 1 (will be PATCH_SIZE or Context Factor)
        self.param1_label = QLabel("Context Factor:")

        param_layout.addWidget(self.param1_label, 0, 0)
        self.param1_slider = QSlider(Qt.Orientation.Horizontal)
        self.param1_slider.valueChanged.connect(self._update_sliders)
        self.param1_value_label = QLabel()

        param_layout.addWidget(self.param1_slider, 0, 1)
        param_layout.addWidget(self.param1_value_label, 0, 2)

        # Slider 2 (will be OVERLAP or Feather Radius)
        self.param2_label = QLabel("Feather Radius:")

        param_layout.addWidget(self.param2_label, 1, 0)
        self.param2_slider = QSlider(Qt.Orientation.Horizontal)
        self.param2_slider.valueChanged.connect(self._update_sliders)
        self.param2_value_label = QLabel()

        param_layout.addWidget(self.param2_slider, 1, 1)
        param_layout.addWidget(self.param2_value_label, 1, 2)

        # Slider 3 (Blend Feather)
        self.param3_label = QLabel("Blend Feather:")

        param_layout.addWidget(self.param3_label, 2, 0)
        self.param3_slider = QSlider(Qt.Orientation.Horizontal)
        self.param3_slider.valueChanged.connect(self._update_sliders)
        self.param3_value_label = QLabel()
        
        param_layout.addWidget(self.param3_slider, 2, 1)
        param_layout.addWidget(self.param3_value_label, 2, 2)
        
        # Slider 4 (Pyramid Levels)
        self.param4_label = QLabel("Pyramid Levels:")
        
        param_layout.addWidget(self.param4_label, 3, 0)
        self.param4_slider = QSlider(Qt.Orientation.Horizontal)
        self.param4_slider.valueChanged.connect(self._update_sliders)
        self.param4_value_label = QLabel()
        
        param_layout.addWidget(self.param4_slider, 3, 1)
        param_layout.addWidget(self.param4_value_label, 3, 2)
        
        layout.addWidget(param_group)

        # --- Action Buttons ---
        buttons_layout = QGridLayout()
        self.set_source_button = QPushButton("Set Clone Source")
        self.set_source_button.setToolTip("Draw a rectangular selection in Siril on a clean area,\nthen click this button to set it as the source.\nRemove old source if it exists.")
        self.set_source_button.clicked.connect(self.set_clone_source)
        
        self.clear_source_button = QPushButton("Clear Source Area")
        self.clear_source_button.setToolTip("Clears source area selection only.")
        self.clear_source_button.clicked.connect(self.clear_selection_source)

        self.add_button = QPushButton("Add Selection")
        self.add_button.setToolTip("Draw a freehand outline around the area you want to correct, like using Photoshop's Lasso tool. When finished, click 'Process' to apply the fix.")
        self.add_button.clicked.connect(self.add_selection)

        self.clear_button = QPushButton("Clear Selections")
        self.clear_button.setToolTip("Clears all selection areas.")
        self.clear_button.clicked.connect(self.clear_selection_areas)

        buttons_layout.addWidget(self.set_source_button, 0, 0)
        buttons_layout.addWidget(self.clear_source_button, 0, 1)
        buttons_layout.addWidget(self.add_button, 1, 0)
        buttons_layout.addWidget(self.clear_button, 1, 1)
        layout.addLayout(buttons_layout)
        
        layout.addStretch(1) # Pushes content to the top
        
    def create_starmask_tab(self):
        layout = QVBoxLayout(self.starmask_tab)
        layout.setContentsMargins(10, 10, 10, 10) # Top padding

        # --- Instructions GroupBox ---
        instr_group = QGroupBox("Instructions")
        instr_layout = QVBoxLayout(instr_group)
        instructions_text = (
            "<ol style='margin-left: -25px;'>"
            "<li>Correct all the stars 'missed' by Starnet in your starless image.<br/>"
            "<b>IMPORTANT,</b>"
                "<ul style='margin-left: -25px;'>"
                "<li>Save the image and do not close it,</li>"
                "<li>PixelMath will use the file that is now saved to generate the 'starmask'.</li>"
                "</ul>"
            "</li>"
            "<li>Click 'Select' to choose the original image with stars.</li>"
            "<li>Click 'Process' to start the 'starmask' generation.</li>"
            "<li>You can find the filename of the 'starmask' in the LOG of the 'Console' tab.</li>"
            "</ol>"
        )
        instr_label = QLabel(instructions_text)
        instr_label.setWordWrap(True)
        instr_layout.addWidget(instr_label)
        layout.addWidget(instr_group)

        # --- File Selection GroupBox ---
        file_group = QGroupBox("Select source file")
        file_layout = QGridLayout(file_group)
        file_layout.setColumnStretch(1, 1)

        file_layout.addWidget(QLabel("Source File with star:"), 0, 0)
        self.reference_path_label = QLineEdit()
        self.reference_path_label.setReadOnly(True)
        select_file_button = QPushButton("Select...")
        select_file_button.clicked.connect(self.select_file_with_star)
        
        file_layout.addWidget(self.reference_path_label, 0, 1)
        file_layout.addWidget(select_file_button, 0, 2)
        layout.addWidget(file_group)

        layout.addStretch(1) # Pushes content to the top

    def center_window(self):
        """ A function to center the window. """
        screen_geometry = self.screen().availableGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
    
    def _update_sliders(self):
        """ Update slider labels and apply mode-specific rounding. """
        # Update context factor label based on mode
        if self.clone_stamp_radio.isChecked():
            # In Clone Stamp mode, snap to integer
            context_val = round(self.param1_slider.value() / 10.0)
            self.param1_value_label.setText(f"{context_val:.1f}")
            self.param2_value_label.setText(f"{str(self.param2_slider.value())} %")
        
        elif self.quilt_inpainting_radio.isChecked():
            # --- Patch Size ---
            # Read the index from the slider (e.g. 0, 1, 2, or 3)
            raw_patch_index = self.param1_slider.value()
            # Ensure that the index does not exceed the size of the list
            safe_patch_index = min(raw_patch_index, len(self.patch_size_steps) - 1)
            # Get the actual value from the list using the index
            patch_size_val = self.patch_size_steps[safe_patch_index]
            self.param1_value_label.setText(f"{str(patch_size_val)} px")
            
            # --- Overlap Divisor ---
            # Read the index from the slider (e.g. 0, 1, 2, 3, or 4)
            raw_overlap_index = self.param2_slider.value()
            # Ensure that the index does not exceed the size of the list
            safe_overlap_index = min(raw_overlap_index, len(self.overlap_divisor_steps) - 1)

            # Overlap (calculated value)
            overlap_divisor_val = self.overlap_divisor_steps[safe_overlap_index]
            # Safety check to prevent ZeroDivisionError
            if overlap_divisor_val > 0:
                overlap_val = patch_size_val // overlap_divisor_val
                self.param2_value_label.setText(f"1/{overlap_divisor_val}  (= {overlap_val}px)")
            else:
                # If divisor is 0, display an invalid state instead of crashing
                self.param2_value_label.setText("1/-- (Invalid)")
            
            # --- Final Feather Radius ---
            feather_percentage_val = self.param3_slider.value()
            self.param3_value_label.setText(f"{feather_percentage_val} %")

            # Pyramid Levels
            pyramid_levels_val = self.param4_slider.value()
            self.param4_value_label.setText(str(pyramid_levels_val))

        elif self.inpainting_radio.isChecked():
            # In CV2 Inpainting mode, allow half-integer steps
            context_val = self.param1_slider.value() / 10.0
            rounded_val = round(context_val * 2) / 2.0

            self.param1_value_label.setText(f"{rounded_val:.1f}")
            self.param2_value_label.setText(f"{str(self.param2_slider.value())} %")


    def _update_ui_for_mode(self):
        """ Show/hide widgets and update instructions based on the selected correction mode. """
        is_clone_mode = self.clone_stamp_radio.isChecked()
        is_quilt_mode = self.quilt_inpainting_radio.isChecked()
        is_cv2_mode = self.inpainting_radio.isChecked()

        show_blend_params = is_quilt_mode
        self.param3_label.setVisible(show_blend_params)
        self.param3_slider.setVisible(show_blend_params)
        self.param3_value_label.setVisible(show_blend_params)
        self.param4_label.setVisible(show_blend_params)
        self.param4_slider.setVisible(show_blend_params)
        self.param4_value_label.setVisible(show_blend_params)

        self.set_source_button.setEnabled(is_clone_mode)
        self.clear_source_button.setEnabled(is_clone_mode)
        
        if is_clone_mode:
            self.instructions_label.setText(self.instructions_texts["Clone Stamp"])
            # --- Clone Stamp Parameters ---
            self.param1_label.setText("Context Factor:")
            self.param1_slider.setRange(0, 50)  # Range 0 to 5.0
            self.param1_slider.setValue(20)     # Default 2.0
            self.param1_slider.setToolTip("Controls the width of the ring used for color calibration.")

            self.param2_label.setText("Feather Radius:")
            self.param2_slider.setRange(1, 100)
            self.param2_slider.setValue(20)
            self.param2_slider.setToolTip("Controls the feather's extent as a percentage, from the selection edge inward.\n"
                                            "A low value creates a small, soft edge. 100% extends the blend to the center of the mask.\n"
                                            "Increase the value if the correction's edge appears too sharp.")

        elif is_quilt_mode:
            self.instructions_label.setText(self.instructions_texts["Inpainting"])
            # --- Quilt Inpainting Parameters ---
            self.param1_label.setText("Patch Size:")
            # There are 5 options [16, 32, 64, 128, 256], so the range is 0-4
            self.param1_slider.setRange(0, len(self.patch_size_steps) - 1) 
            self.param1_slider.setValue(2) # Set the default to 64 (2nd index)
            self.param1_slider.setToolTip("Patch size used by the algorithm to search around the selection and fill it.\n"
                                          "If you notice contamination from nearby areas, reduce the patch size.\n\n"
                                          "⚠️ Small patches on large regions greatly increase processing time.\n"
                                          "Tip: use small patches only for small selections when nearby objects may interfere.")

            self.param2_label.setText("Overlap Divisor:")
            # There are 5 options [2, 4, 8, 16, 32], so the range is 0-4
            self.param2_slider.setRange(0, len(self.overlap_divisor_steps) - 1)
            self.param2_slider.setValue(0) # Set the default to 2 (1/2 overlap)
            self.param2_slider.setToolTip("Determines patch overlap (steps: 2, 4, 8, 16, 32).")

            # --- Blend Parameters ---
            self.param3_label.setText("Blend Feather:")
            self.param3_slider.setRange(1, 100)  # Range from 1 to 50
            self.param3_slider.setValue(20)      # Default = 10
            self.param3_slider.setToolTip("Controls the feather's extent as a percentage, from the selection edge inward.\n"
                                            "A low value creates a small, soft edge. 100% extends the blend to the center of the mask.\n"
                                            "Increase the value if the correction's edge appears too sharp.")

            self.param4_label.setText("Pyramid Levels:")
            self.param4_slider.setRange(2, 8)   # Range from 2 to 8 levels
            self.param4_slider.setValue(4)      # Default
            self.param4_slider.setToolTip("Controls the depth of the pyramid blend. More levels for smoother transitions.")

        elif is_cv2_mode:
            self.instructions_label.setText(self.instructions_texts["Inpainting"])
            # --- CV2 Inpainting Parameters ---
            self.param1_label.setText("Context Factor:")
            self.param1_slider.setRange(5, 50) # Range 0.5 to 5.0
            self.param1_slider.setValue(20)    # Default 2.0
            self.param1_slider.setToolTip("Controls the size of the area around the selection used to analyze the texture.")

            self.param2_label.setText("Feather Radius:")
            self.param2_slider.setRange(1, 100)
            self.param2_slider.setValue(20)
            self.param2_slider.setToolTip("Controls the feather's extent as a percentage, from the selection edge inward.\n"
                                            "A low value creates a small, soft edge. 100% extends the blend to the center of the mask.\n"
                                            "Increase the value if the correction's edge appears too sharp.")
        
        self._update_sliders() # Trigger a label and value update

    def _toggle_on_top(self):
        """ Enables or disables the 'always on top' attribute of the window based on the checkbox state. """
        is_on_top = self.on_top_check.isChecked()
        if is_on_top:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show() # Re-show the window to apply the flag change

    def update_status(self, message):
        self.status_label.setText(message)
        QApplication.processEvents()

    def select_file_with_star(self):
        """Opens a dialog box to select the source file with star."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select source file with star",
            "", # Start directory
            "FITS Images (*.fit *.fits);;All Files (*.*)"
        )
        if filepath:
            self.reference_path_label.setText(filepath)
            self.siril.log(f"Reference Selected - Selected:\n{os.path.basename(filepath)}", s.LogColor.BLUE)

    def add_selection(self):
        """ Enters free-draw mode to add a new selection polygon in Siril. """
        # selection = self.siril.get_siril_selection()
        # if selection and len(selection) == 4:
        #     # We assume the tuple is (x, y, w, h)
        #     x, y, w, h = selection
        #     if w > 0 and h > 0:
        #         selection_poly = s.Polygon.from_rectangle((x, y, w, h), color=0x00FF0080, fill=False)
        #         self.siril.overlay_add_polygon(selection_poly)
        #         self.siril.set_siril_selection(0, 0, 0, 0) # Deselect
        #         self.update_status(f"Added rectangular selection.")
        #     else:
        #         QMessageBox.critical(self, "Warning", "The selection is not valid (zero size).")
        # else:
        #     # Otherwise, enter free-draw mode
        
        # clear any existing selection, all algorithms work better with a single area
        self.clear_selection_areas()

        # In this case is better to always enter free-draw mode
        self.update_status("Draw a freehand outline around the area you want to correct, like using Photoshop's Lasso tool. When finished, click 'Process' to apply the fix.")
        try:
            self.siril.overlay_draw_polygon(color=0x00FF0040, fill=False)
        except s.MouseModeError:
            # self.siril.log(f"Warning, {str(s.MouseModeError)}", s.LogColor.RED)
            # QMessageBox.critical(self, "Warning", "The mouse must be in 'normal selection' mode to draw a polygon.")
            self.siril.log("Incorrect mouse mode. Attempting to restore connection...", s.LogColor.RED)
            try:
                self.siril.disconnect()
                self.siril.connect()
                self.siril.log("Connection with Siril restored successfully.", s.LogColor.GREEN)
            except s.SirilConnectionError:
                QMessageBox.critical(self, "Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
                self.close()

    def set_clone_source(self):
        """ Sets the clone source from a rectangular selection in Siril. """
        selection = self.siril.get_siril_selection()
        if not (selection and len(selection) == 4):
            QMessageBox.warning(self, "Selection Missing", "Please draw a rectangular selection on the source area in Siril first.")
            return
        
        self.clone_source_rect = selection
        self.siril.log(f"Polygon : {self.clone_source_poly} .", s.LogColor.GREEN)

        # Remove old source overlay if it exists
        if self.clone_source_poly is not None:
            try:
                self.siril.overlay_delete_polygon(self.clone_source_poly.polygon_id)
                self.clone_source_poly = None
            except:
                pass # Ignore errors if polygon no longer exists

        # Add a new overlay for visual confirmation (cyan)
        source_poly_to_add = s.Polygon.from_rectangle(selection, color=0xFFFF00FF, fill=False, legend="SOURCE")

        self.clone_source_poly = self.siril.overlay_add_polygon(source_poly_to_add)
        self.siril.log(f"Added polygon SOURCE with ID: {self.clone_source_poly.polygon_id}", s.LogColor.GREEN)

        self.siril.set_siril_selection(0, 0, 0, 0)
        self.update_status("Clone source set. Now draw destination area(s).")

    def clear_selection_areas(self):
        """ Delete all selection areas except the source, using the "delete all and redraw source" method. """
        try:
            # Check if there is a source polygon to preserve
            if self.clone_source_poly:
                # Save a copy of the source Polygon object
                source_polygon_to_restore = self.clone_source_poly
                
                # Clear ALL polygons from Siril overlay
                self.siril.overlay_clear_polygons()
                
                # Re-add the source polygon. The function will return a new object with a new ID.
                self.clone_source_poly = self.siril.overlay_add_polygon(source_polygon_to_restore)
                
                self.update_status("Destination areas deleted.")
                # self.siril.log(f"Restored source polygon with new ID: {self.clone_source_poly.polygon_id}", s.LogColor.GREEN)
            
            else:
                # If there was no source defined, we just delete everything.
                self.siril.overlay_clear_polygons()
                self.clone_source_rect = None
                self.clone_source_poly = None
                self.update_status("All selections have been cleared.")

        except Exception as e:
            self.siril.log(f"Error deleting areas: {e}", s.LogColor.RED)
            self.update_status("Error deleting.")

    def clear_selection_source(self):
        """ Only clears the source selection for the Clone Stamp. """
        if self.clone_source_poly:
            try:
                self.siril.overlay_delete_polygon(self.clone_source_poly.polygon_id)
                self.update_status("Source selection cleared.")
            except Exception as e:
                self.siril.log(f"Could not remove source overlay: {e}", s.LogColor.RED)
        
        self.clone_source_rect = None
        self.clone_source_poly = None

    def create_mask_from_selections(self, image_shape, selection_polygons):
        """ Creates a boolean mask. 'True' in exclusion areas, 'False' elsewhere. """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not selection_polygons:
            return mask.astype(bool)

        for polygon in selection_polygons:
            if len(polygon.points) < 3:
                continue
            points = np.array([[int(p.x), int(h - p.y)] for p in polygon.points], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)

        return mask.astype(bool)

    def _process_image(self):
        """
        Main dispatcher function called by the 'Process' button.
        It checks the active tab and calls the appropriate processing function.
        """
        # Check the active tab index
        selected_tab_index = self.tab_widget.currentIndex()
        if selected_tab_index == 0:
            if self.quilt_inpainting_radio.isChecked() or self.inpainting_radio.isChecked():
                self.inpainting_method()

            else:
                self.run_clone_stamp()

        elif selected_tab_index == 1:
            self.create_star_mask()

    def create_star_mask(self):
        """ Creates a star mask from the selected reference image. """
        original_image_path = self.reference_path_label.text()
        if not original_image_path or not os.path.isfile(original_image_path):
            QMessageBox.warning(self, "File Missing", "Please select a valid source file with stars first.")
            return

        self.process_button.setEnabled(False)
        self.update_status("Starting star mask generation...")

        try:
            # Use PixelMath to create the star mask
            # Get the path of the starless image currently loaded in Siril
            starless_image_path = self.siril.get_image_filename()
            if not starless_image_path:
                self.siril.log("Error: No active starless image found. Please open your starless image.", s.LogColor.RED)
                return

            # Build the output path for the star mask file
            base_name, ext = os.path.splitext(os.path.basename(original_image_path))
            # Put the output in the same directory as the original image
            # output_dir = os.path.dirname(original_image_path)
            output_dir = self.siril.get_siril_wd()  # Use Siril's working directory
            starmask_output_path = os.path.join(output_dir, f"starmask_{base_name}_InpaintTool.fits")

            # Build the formula for PixelMath
            # The formula is: "Original_Image" - "Starless_Image"
            # Quotation marks are important for handling paths and names with spaces
            pm_formula = f"'${original_image_path}$' - '${starless_image_path}$'"
            
            self.siril.log(f"Executing PixelMath formula: {pm_formula}")
            self.update_status("Applying PixelMath formula...")

            self.update_status("Step 1/2: Applying PixelMath formula...")

            self.siril.cmd("pm", f'"{pm_formula}"')
            self.siril.log("PixelMath formula applied successfully.", s.LogColor.GREEN)

            self.update_status("Step 2/2: Saving the star mask file...")
            self.siril.cmd("save", f'"{starmask_output_path}"')

            self.siril.log(f"Star mask created and saved as: {starmask_output_path}", s.LogColor.GREEN)
            self.update_status("Star mask generation completed successfully.")

        except s.CommandError as e:
            error_message = f"Error during star mask generation: {e}"
            self.update_status(error_message)
            self.siril.log(error_message, s.LogColor.RED)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            self.update_status(error_message)
            self.siril.log(error_message, s.LogColor.RED)
        finally:
            self.process_button.setEnabled(True)

    def run_clone_stamp(self):
        """ Performs the clone stamp operation using the same robust blending logic as inpainting. """
        self.process_button.setEnabled(False)
        self.update_status("Starting Clone Stamp process...")

        try:
            # Load full-depth data for processing
            current_image_data = self.siril.get_image_pixeldata(preview=False)
            if current_image_data is None:
                raise SirilError("Cannot load image data.")
            
            # Determine image properties at runtime
            original_dtype = current_image_data.dtype
            is_mono = len(current_image_data.shape) == 2 or current_image_data.shape[0] == 1
            
            # Convert to HWC format for OpenCV
            if not is_mono and current_image_data.shape[0] in [1, 3]:
                image_to_process = current_image_data.transpose(1, 2, 0)
            elif is_mono and len(current_image_data.shape) == 3:
                image_to_process = current_image_data[0]
            else:
                image_to_process = current_image_data

            image_to_process = image_to_process.astype(np.float32)

            # --- Preparation (same as inpainting) ---
            if self.clone_source_rect is None:
                QMessageBox.critical(self, "Error", "Clone source is not set.")
                self.process_button.setEnabled(True)
                return

            all_polygons = self.siril.overlay_get_polygons_list()
            if all_polygons is None:
                all_polygons = []
            
            # Filter polygons to exclude the source area and get only the destinations
            destination_polygons = []
            if self.clone_source_poly is not None:
                source_id = self.clone_source_poly.polygon_id
                destination_polygons = [p for p in all_polygons if hasattr(p, 'polygon_id') and p.polygon_id != source_id]
            else:
                destination_polygons = all_polygons

            if not destination_polygons:
                QMessageBox.warning(self, "Warning", "No destination area defined.")
                self.process_button.setEnabled(True)
                return

            self.update_status("Creating destination mask...")
            dest_mask = self.create_mask_from_selections(image_to_process.shape, destination_polygons)

            # Calcola le dimensioni effettive della maschera di destinazione
            ys_dest, xs_dest = np.nonzero(dest_mask)
            h_dest_mask = np.max(ys_dest) - np.min(ys_dest)
            w_dest_mask = np.max(xs_dest) - np.min(xs_dest)
            min_dest_dimension = min(h_dest_mask, w_dest_mask)
            
            # Read parameters from the GUI
            feather_radius = self.param2_slider.value()

            # --- Creating content to clone (direct copy) ---
            self.update_status("Cloning pixels...")

            # --- Coordinate calculation ---
            h, w = image_to_process.shape[:2]
            
            # Calculate the center of the target mask
            moments = cv2.moments((dest_mask*255).astype(np.uint8))
            if moments["m00"] == 0: raise ValueError("Destination mask is invalid.")
            dest_center_x = int(moments["m10"] / moments["m00"])
            dest_center_y = int(moments["m01"] / moments["m00"])
            
            # Calculate the center of the cloning source
            sx, sy, sw, sh = self.clone_source_rect
            sy_from_top = h - sy - sh
            source_center_x = sx + sw // 2
            source_center_y = sy_from_top + sh // 2
            
            # Calculate the offset between the two centers
            offset_x = dest_center_x - source_center_x
            offset_y = dest_center_y - source_center_y
            
            # Find the pixel coordinates in the target mask
            dest_y_coords, dest_x_coords = np.where(dest_mask)
            source_x_coords = np.clip(dest_x_coords - offset_x, 0, w - 1)
            source_y_coords = np.clip(dest_y_coords - offset_y, 0, h - 1)
            
            # Build the cloned image
            self.update_status("Building foreground patch...")
            hard_clone_image = image_to_process.copy()
            cloned_pixels = image_to_process[source_y_coords, source_x_coords]

            # --- Local gain/bias calibration on cloned_pixels (annulus around dest) ---
            # Width for OUTER ring (Dilation) - Use full value from GUI
            ring_width_dilate = int(round(self.param1_slider.value() / 10.0))

            # Width for the INNER ring (Erosion) - Limited value
            # Physical limit: 1/4 of the minimum mask size.
            max_possible_erode_width = max(1, min_dest_dimension // 4)

            # The final width for erosion is the smaller of the GUI value and the physical limit.
            # This ensures that even if the user chooses a small value, it won't be enlarged.
            ring_width_erode = min(ring_width_dilate, max_possible_erode_width)

            self.siril.log(f"Clone Stamp Rings -> Dilate Width: {ring_width_dilate}px, Erode Width: {ring_width_erode}px (limited by mask size)", s.LogColor.BLUE)

            # Create two separate kernels with respective sizes
            kern_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width_dilate * 2 + 1, ring_width_dilate * 2 + 1))
            kern_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width_erode * 2 + 1, ring_width_erode * 2 + 1))

            dest_bool = dest_mask.astype(bool)

            # Outer ring (outside the dest_mask) - Use the expansion kernel (large)
            dilated = cv2.dilate(dest_mask.astype(np.uint8), kern_dilate, iterations=1).astype(bool)
            outer_ring = dilated & (~dest_bool)

            # Inner ring (inside the dest_mask) - Uses erosion kernel (limited and safe)
            eroded = cv2.erode(dest_mask.astype(np.uint8), kern_erode, iterations=1).astype(bool)
            inner_ring = dest_bool & (~eroded)

            # Union of rings = calibration band
            ring_mask = outer_ring | inner_ring

            # Extract ring coordinates
            ring_y, ring_x = np.where(ring_mask)
            if ring_y.size >= 8:
                # Values ​​in the destination ring -->> dest_ring_vals.shape == (N,) # N = number of pixels in the ring
                dest_ring_vals = image_to_process[ring_y, ring_x].astype(np.float32)  # (N,) mono or (N,3) RGB

                # Source coordinates corresponding to the ring
                ring_src_x = np.clip(ring_x - offset_x, 0, w - 1)
                ring_src_y = np.clip(ring_y - offset_y, 0, h - 1)

                # Values ​​in the source ring -->> dest_ring_vals.shape == (N, 3) # N = number of pixels in the ring
                src_ring_vals = image_to_process[ring_src_y, ring_src_x].astype(np.float32)  # (N,) mono or (N,3) RGB

                # Calculate a (gain) over luminance, b/b_vec (bias) per channel
                if not is_mono:
                    # Luminance for estimation of 'a'
                    dY = 0.299 * dest_ring_vals[:,0] + 0.587 * dest_ring_vals[:,1] + 0.114 * dest_ring_vals[:,2]
                    sY = 0.299 * src_ring_vals[:,0]  + 0.587 * src_ring_vals[:,1]  + 0.114 * src_ring_vals[:,2]
                    mean_s, mean_d = float(np.mean(sY)), float(np.mean(dY))
                    var_s = float(np.var(sY))
                    if var_s < 1e-8:
                        a = 1.0
                    else:
                        cov = float(np.mean((sY - mean_s) * (dY - mean_d)))
                        a = cov / (var_s + 1e-12)

                    # Clamp for safety
                    a = float(np.clip(a, 0.7, 1.3))

                    # Bias per channel to align the media
                    mean_src_vec = np.mean(src_ring_vals, axis=0)  # (3,)
                    mean_dst_vec = np.mean(dest_ring_vals, axis=0) # (3,)
                    b_vec = mean_dst_vec - a * mean_src_vec        # (3,)

                    # Clamp bias per channel
                    max_b_vec = np.maximum(np.abs(mean_dst_vec) * 0.5, 1e-3)
                    b_vec = np.clip(b_vec, -max_b_vec, max_b_vec)

                    # Apply a/b to cloned pixels
                    cloned_pixels = a * cloned_pixels + b_vec  # broadcasting su (N,3)

                else:
                    # Mono
                    dY = dest_ring_vals
                    sY = src_ring_vals
                    mean_s, mean_d = float(np.mean(sY)), float(np.mean(dY))
                    var_s = float(np.var(sY))
                    if var_s < 1e-8:
                        a = 1.0
                        b = mean_d - mean_s
                    else:
                        cov = float(np.mean((sY - mean_s) * (dY - mean_d)))
                        a = cov / (var_s + 1e-12)
                        b = mean_d - a * mean_s

                    a = float(np.clip(a, 0.7, 1.3))
                    max_b = max(abs(mean_d) * 0.5, 1e-3)
                    b = float(np.clip(b, -max_b, max_b))

                    cloned_pixels = a * cloned_pixels + b
            # --- End calibration ---

            # Copy the patch pixels to the resulting image only where the mask is active
            hard_clone_image[dest_mask] = cloned_pixels

            # Creating Alpha Mask With Distance Transform
            self.update_status("Creating feathered alpha mask...")
            if feather_radius > 0:
                # Calculate the distance of each pixel in the mask from the nearest edge
                dist = cv2.distanceTransform(dest_mask.astype(np.uint8), cv2.DIST_L2, 3)

                # --- Raggio sfumatura in percentuale ---
                # 1. Trova la distanza massima (dal bordo al punto più centrale della selezione)
                max_dist = dist.max()
                if max_dist == 0: max_dist = 1 # Evita divisione per zero in maschere di 1px
                # 2. Calcola il "raggio effettivo" in pixel basato sulla percentuale data dallo slider.
                #    Es: slider a 100 -> 1.0; slider a 50 -> 0.5
                effective_radius = max_dist * (feather_radius / 100.0)
                # 3. Normalizza la distanza usando questo raggio effettivo e dinamico.
                #    Se la percentuale è 100, effective_radius = max_dist, quindi il centro avrà valore 1.0.
                alpha_mask_lineare = np.clip(dist / (effective_radius + 1e-8), 0.0, 1.0)

                # Gradient profile
                # --------------------------------------------------------------------------
                # alpha_mask = alpha_mask_lineare  # Linear Profile (default)
                # alpha_mask = np.power(alpha_mask_lineare, 2.0)  # Convex Profile (Ease-in)
                # alpha_mask = np.power(alpha_mask_lineare, 0.5)  # Concave Profile (Ease-out)
                alpha_mask = 0.5 * (1 - np.cos(alpha_mask_lineare * np.pi)) # S-profile (SmoothStep)
                # --------------------------------------------------------------------------

                # Optional: add subtle noise ONLY in the transition zone (0 < alpha < 1)
                # noise_strength = 0.5  # tunable
                # transition_zone = (alpha_mask_lineare > 0) & (alpha_mask_lineare < 1)
                # noise = np.random.normal(0, noise_strength, alpha_mask.shape).astype(np.float32)
                # # Apply noise only to the feather region
                # alpha_mask[transition_zone] = np.clip(alpha_mask[transition_zone] + noise[transition_zone], 0.0, 1.0)

                # I apply a light blur to soften the ramp
                # k_size_blur = int(max(3, feather_radius / 4))
                # if k_size_blur % 2 == 0:
                #     k_size_blur += 1
                alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)
            else:
                # If the radius is zero, the mask is sharp
                alpha_mask = dest_mask.astype(np.float32)

            # If the image is in color, we expand the mask size
            if not is_mono:
                alpha_mask = alpha_mask[..., np.newaxis]

            self.update_status("Blending final image...")

            # Final image fusion with alpha blending
            # result_image = image_to_process * (1.0 - alpha_mask) + hard_clone_image * alpha_mask

            # --- Replacing Alpha Blending with Weighted Averaging ---
            # The calculated alpha_mask acts as the weight for the patch (weight_patch)
            weight_patch = alpha_mask
            # The weight for the original image (background) is its complement
            weight_original = 1.0 - weight_patch
            # Sum of weights with an epsilon to prevent division by zero.
            # In a perfect feathering zone, this sum is always close to 1.0.
            total_weight = weight_patch + weight_original + 1e-8
            # Perform the normalized weighted average, combining the calibrated patch
            # (hard_clone_image) and the original background (image_to_process).
            result_image = (hard_clone_image * weight_patch + image_to_process * weight_original) / total_weight

            # --- Update in Siril ---
            self.update_status("Updating image in Siril...")

            final_image = result_image.astype(original_dtype)
            if not is_mono and final_image.ndim == 3:
                final_image = final_image.transpose(2, 0, 1)

            self.siril.undo_save_state("Clone Stamp Tool")
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(final_image)

            self.update_status("Process completed successfully!")
            # self.siril.overlay_clear_polygons()

        except Exception as e:
            self.update_status(f"Error: {e}")
            self.siril.log(f"Error during clone process: {str(e)}", s.LogColor.RED)
        finally:
            self.process_button.setEnabled(True)

    def inpainting_method(self):
        """
        Manages the inpainting process.
        Prepares the data, selects the user-selected method (OpenCV or Texture Synthesis), and applies the result.
        """
        self.process_button.setEnabled(False)
        self.update_status("Starting the fix process...")
        try:
            # Load full-depth data for processing
            current_image_data = self.siril.get_image_pixeldata(preview=False)
            if current_image_data is None:
                raise SirilError("Could not load current image data.")
            
            # Determine image properties at runtime
            original_dtype = current_image_data.dtype
            is_mono = len(current_image_data.shape) == 2 or current_image_data.shape[0] == 1
            
            # Convert to HWC format for OpenCV
            if not is_mono and current_image_data.shape[0] in [1, 3]:
                image_to_process = current_image_data.transpose(1, 2, 0)
            elif is_mono and len(current_image_data.shape) == 3:
                image_to_process = current_image_data[0]
            else:
                image_to_process = current_image_data

            image_to_process = image_to_process.astype(np.float32)

            # Read parameters from the GUI
            inpaint_radius = 2 # Radius of a circular neighborhood of each point inpainted that is considered by the algorithm
            feather_radius = self.param2_slider.value()
            # Getting value from slider (with 0.5 steps)
            context_factor = round(self.param1_slider.value() / 10.0 * 2) / 2.0

            selection_polygons = self.siril.overlay_get_polygons_list()
            if selection_polygons is None:
                selection_polygons = []

            # Exclude clone source, if it exists
            if self.clone_source_poly is not None:
                source_id = self.clone_source_poly.polygon_id
                selection_polygons = [p for p in selection_polygons if p.polygon_id != source_id]

            if not selection_polygons:
                QMessageBox.warning(self, "Warning", "No selection area defined.")
                self.process_button.setEnabled(True)
                return

            self.update_status("Creating mask from selections...")
            mask_to_fill = self.create_mask_from_selections(image_to_process.shape, selection_polygons)
            if not np.any(mask_to_fill):
                QMessageBox.warning(self, "Warning", "The selection mask is empty.")
                self.process_button.setEnabled(True)
                return

            # --- Method Selection and Execution ---
            result_image = None
            if self.quilt_inpainting_radio.isChecked():
                self.siril.log("Using Patch-Based Texture Synthesis Inpainting.", s.LogColor.BLUE)
                self.update_status("Running Patch-Based Texture Synthesis Inpainting...")
                result_image = self.run_texture_synthesis_inpainting(image_to_process, is_mono, mask_to_fill, context_factor)
            
            elif self.inpainting_radio.isChecked():
                self.siril.log("Using OpenCV Inpainting (Navier-Stokes).", s.LogColor.BLUE)
                self.update_status("Running OpenCV Inpainting...")
                self.siril.log(f"Context factor: {context_factor}")
                self.siril.log(f"Feather radius: {feather_radius}")
                result_image = self.run_opencv_inpainting(image_to_process, is_mono, mask_to_fill, context_factor, feather_radius, inpaint_radius)

            if result_image is not None:
                final_image = result_image.astype(original_dtype)
                
                if not is_mono and final_image.ndim == 3:
                    final_image = final_image.transpose(2, 0, 1)

                self.siril.undo_save_state("Patch Inpainting Tool")
                with self.siril.image_lock():
                    self.siril.set_image_pixeldata(final_image)

                # self.siril.overlay_clear_polygons()
            else:
                self.update_status("No operation was performed.")
                self.siril.log("No operation was performed.", s.LogColor.RED)

        except Exception as e:
            self.update_status(f"Error during processing: {e}")
            self.siril.log(f"Error during processing: {str(e)}", s.LogColor.RED)
        finally:
            self.process_button.setEnabled(True)

    def run_texture_synthesis_inpainting(self, target_image, is_mono, mask, context_factor):
        """
        Perform patch-based texture synthesis inpainting using an improved quilting approach.

        This algorithm fills the region specified by `mask` with synthesized texture
        patches sampled from the surrounding context. It works in several stages:

        1. **Preparation**
        - Define patch size, stride, and overlap.
        - Build an annular region around the mask (controlled by `context_factor`)
            from which source patches are extracted.
        - Collect a random subset of patches from this region for efficiency.

        2. **Patch Placement**
        - Slide across the masked region with overlapping square patches.
        - For each position:
            * Select the best matching source patch using a similarity metric
            (SSD on intensity + gradient consistency).
            * Construct a dynamic blending mask (smoothstep ramps) on all
            four edges (left, right, top, bottom) to softly merge the patch
            into already filled neighbors.
            * Accumulate the weighted contribution of the patch into an
            image accumulator (`accum_image`) and its weight map (`accum_weight`).

        3. **Normalization**
        - After all patches are placed, normalize by dividing the accumulated
            image by the accumulated weights to produce a seamless mosaic.
        - For pixels with no patch contributions, fall back to the original
            target image.

        4. **Final Pyramid Blending**
        - Perform Laplacian pyramid blending between the synthesized mosaic
            and the original background image using the inpainting mask.
        - This final step smooths transitions at the global level and ensures
            seamless integration of the inpainted area with the surrounding image.

        Parameters
        ----------
        target_image : np.ndarray
            The input image (grayscale or RGB, float32).
        mask : np.ndarray of bool
            Boolean mask where True indicates the region to be inpainted.
        context_factor : float
            Controls the width of the annulus (search region) around the mask
            from which source patches are sampled.

        Returns
        -------
        final_image : np.ndarray
            The completed image with the masked region inpainted using
            patch-based synthesis and pyramid blending.
        """
        self.update_status("Starting Pyramid-Based Synthesis...")

        # Calculate the dimensions of the rectangle containing the mask
        ys, xs = np.nonzero(mask)
        h_mask = np.max(ys) - np.min(ys)
        w_mask = np.max(xs) - np.min(xs)
        min_dimension = min(h_mask, w_mask)

        # --- Parameters from GUI ---
        # Reads the index from the slider and converts it to the real value
        patch_size_index = self.param1_slider.value()
        PATCH_SIZE = self.patch_size_steps[patch_size_index]

        overlap_divisor_index = self.param2_slider.value()
        OVERLAP_DIVISOR = self.overlap_divisor_steps[overlap_divisor_index]
        
        OVERLAP = PATCH_SIZE // OVERLAP_DIVISOR
        # Ensure OVERLAP is at least 1 pixel
        if OVERLAP == 0:
            OVERLAP = 1
            
        patch_stride = PATCH_SIZE - OVERLAP

        # --- Preparation ---
        fill_mask = mask.copy()
        result_mosaic = target_image.copy() # Here we build the mosaic with sharp edges
        h, w = target_image.shape[:2]
        
        # --- Definition of the Patch Collection and Research Ring ---
        self.update_status("Analyzing local texture...")

        # Use Distance Transform to create a perfect sampling ring, regardless of the mask shape.
        # Let's calculate the distance map. Each pixel outside the mask will have a value equal to its distance from the closest pixel in the mask.
        # Pixels inside the mask will have a distance of 0.
        dist_transform = cv2.distanceTransform(
            (1 - mask.astype(np.uint8)) * 255, # Let's invert the mask
            cv2.DIST_L2, 
            3
        )

        # We define two rings: one outside the patch (stronger weight) and one inside (weaker weight).
        ring_out = max(64, PATCH_SIZE)

        # Calculating the width of the INNER ring (with fallback)
        # Let's start with an "ideal" width, for example, half the width of the outer one.
        ideal_inner_width = ring_out // 2

        # Calcoliamo il "limite fisico" dell'anello interno.
        # Partendo dal bordo della maschera verso l'interno, non può essere più largo di 1/4
        # della dimensione minima della maschera stessa, altrimenti si può contaminare con la luminosità di una stella.
        # Usiamo max(1, ...) per evitare che diventi zero su maschere molto piccole.
        max_possible_inner_width = max(1, min_dimension * 0.25)

        # The final width of the inner ring is the SMALLEST value
        ring_in = min(ideal_inner_width, max_possible_inner_width)

        self.siril.log(f"Search rings -> Out: {ring_out}px, In: {ring_in}px (physical limit was {max_possible_inner_width}px)", s.LogColor.BLUE)

        # Define the two zones for reference color
        mask_out = (dist_transform > 0) & (dist_transform < ring_out)
        mask_in  = (mask > 0) & (dist_transform < ring_in)

        # Calculate average reference color and adaptive alpha
        if np.any(mask_out) or np.any(mask_in):
            ref_mean = None
            annulus_pixels = None

            # Extract pixels from both zones
            if np.any(mask_out):
                pixels_out = target_image[mask_out]
            else:
                pixels_out = np.empty((0,), dtype=target_image.dtype)

            if np.any(mask_in):
                pixels_in = target_image[mask_in]
            else:
                pixels_in = np.empty((0,), dtype=target_image.dtype)

            # Concatenation → all pixels have the same weight
            annulus_pixels = np.vstack([p for p in (pixels_out, pixels_in) if p.size > 0])

            # Single average of all pixels between inner and outer ring
            ref_mean = np.mean(annulus_pixels, axis=0)
            
            # We calculate the variance on the luminance for a robust estimate of the structure
            if not is_mono and annulus_pixels.ndim > 1 and annulus_pixels.shape[1] == 3:
                luminance_pixels = (
                    0.299 * annulus_pixels[:, 0] +
                    0.587 * annulus_pixels[:, 1] +
                    0.114 * annulus_pixels[:, 2]
                )
            else:
                luminance_pixels = annulus_pixels.flatten()
            
            local_variance = np.var(luminance_pixels)

            self.siril.log(f"Pixel range: {luminance_pixels.min():.3f} - {luminance_pixels.max():.3f}", s.LogColor.BLUE)
            self.siril.log(f"Local variance raw: {local_variance:.2e}", s.LogColor.BLUE)

            # Let's calculate the mean and standard deviation for the CV
            local_mean = np.mean(luminance_pixels)
            local_std = np.std(luminance_pixels)
            
            # Calculate the Coefficient of Variation (CV)
            # Add an epsilon to avoid division by zero in completely black areas
            coefficient_of_variation = local_std / (local_mean + 1e-9)

            # We define robust thresholds based on CV, which is a relative measure
            low_cv_threshold = 0.05  # Corresponds to a very smooth sky (5% variation)
            high_cv_threshold = 0.20 # Corresponds to a well-structured nebula (20% variation)

            # We calculate the linear scale as before
            scale = (coefficient_of_variation - low_cv_threshold) / (high_cv_threshold - low_cv_threshold + 1e-9)
            scale = np.clip(scale, 0.0, 1.0) # Let's make sure it's between 0 and 1

            # We apply an "ease-out" curve to make the response more aggressive
            # This causes the alpha to rise much faster in the beginning.
            # eased_scale = 1 - (1 - scale)**2  # <-- "ease-out" disattivato

            min_alpha = 0.3
            max_alpha = 1.0 # We use 1.0 to give more weight in case of high structure
            # adaptive_alpha = min_alpha + eased_scale * (max_alpha - min_alpha)  # <-- "ease-out" disattivato
            adaptive_alpha = min_alpha + scale * (max_alpha - min_alpha)          # <-- lineare
            
            self.siril.log(f"Mean: {local_mean:.2e}, StdDev: {local_std:.2e}, CV: {coefficient_of_variation:.3f} -> Adaptive Alpha: {adaptive_alpha:.2f}", s.LogColor.BLUE)

        else: # Fallback in case the ring is empty (very small image)
            ref_mean = np.mean(target_image[~mask], axis=0)
            adaptive_alpha = 0.5 # A default average value
            # self.siril.log(f"Warning: Empty annulus. Using default alpha: {adaptive_alpha:.2f}", s.LogColor.RED)
        
        # Before we begin, we erase the mask area by filling it with the average color of the ring.
        # This prevents stars or other artifacts under the mask from contaminating subsequent calculations.

        # For RGB images, ref_mean is an array (R,G,B) and is passed correctly to all pixels in the mask.
        result_mosaic[mask] = ref_mean

        # We scan the ring and collect the patches
        coords = np.argwhere(mask_out)

        # We choose a random subset of coordinates to create the patches
        num_patches_to_sample = min(5000, len(coords))
        source_patches = []

        # Random sampling of patches
        if num_patches_to_sample > 0:
            sample_indices = np.random.choice(len(coords), num_patches_to_sample, replace=False)
            half_patch = PATCH_SIZE // 2

            for idx in sample_indices:
                y, x = coords[idx]
                if (y > half_patch and y < h - half_patch and
                    x > half_patch and x < w - half_patch):

                    patch = target_image[y-half_patch:y+half_patch, x-half_patch:x+half_patch]
                    patch_mask = mask[y-half_patch:y+half_patch, x-half_patch:x+half_patch]

                    # We only accept patches that are completely outside the mask
                    if np.any(patch_mask):
                        continue  

                    source_patches.append(patch)

        # Check if we have enough patches
        if len(source_patches) < 20:
            self.siril.log(f"Warning: Only {len(source_patches)} patches found in the annulus. Try changing patch size", s.LogColor.RED)
            if len(source_patches) == 0:
                self.siril.log("Error: No source patches found. Aborting inpainting. Try changing patch size", s.LogColor.RED)
                return None

        # --- Fast Filling with Smart Stitching ---
        pixels_to_fill_total = np.sum(mask)
        pixels_filled_count = 0

        # --- Initializing accumulators ---
        if is_mono:
            accum_image = np.zeros((h, w), dtype=np.float32)
            accum_weight = np.zeros((h, w), dtype=np.float32)
        else:
            accum_image = np.zeros((h, w, 3), dtype=np.float32)
            accum_weight = np.zeros((h, w, 1), dtype=np.float32)

        # Pixels outside the mask retain their original value
        outside = (~mask).astype(bool)
        if is_mono:
            accum_image[outside] = target_image[outside]
            accum_weight[outside] = 1.0
        else:
            accum_image[outside, :] = target_image[outside, :]
            accum_weight[outside, 0] = 1.0

        # We explicitly calculate starting positions to ensure complete coverage
        # with fixed-size patches (your correct analysis).
        y_positions = list(range(0, max(1, h - PATCH_SIZE + 1), patch_stride))
        x_positions = list(range(0, max(1, w - PATCH_SIZE + 1), patch_stride))

        # We add the last possible position if the standard iteration does not cover the edge.
        if not y_positions or y_positions[-1] < h - PATCH_SIZE:
            y_positions.append(h - PATCH_SIZE)
        if not x_positions or x_positions[-1] < w - PATCH_SIZE:
            x_positions.append(w - PATCH_SIZE)
        
        # We remove any duplicates and sort for safety
        y_positions = sorted(list(set(y_positions)))
        x_positions = sorted(list(set(x_positions)))

        is_first_patch = True

        for y in y_positions:
            # self.siril.log(f"Remaining pixels to fill: {np.sum(fill_mask)}", s.LogColor.BLUE)
            for x in x_positions:
                # Let's calculate the actual size of the patch (useful for edges)
                yy0, yy1 = y, y + PATCH_SIZE
                xx0, xx1 = x, x + PATCH_SIZE
                
                # We use the safe coordinates to extract the mask
                patch_mask_in_fill = fill_mask[yy0:yy1, xx0:xx1]

                # Skip this patch ONLY IF it doesn't touch even one pixel of the original area we needed to fix
                if not np.any(mask[yy0:yy1, xx0:xx1]):
                    continue

                if is_first_patch:
                    # For the first patch, let's use our initial guess
                    current_patch = result_mosaic[yy0:yy1, xx0:xx1]
                    is_first_patch = False  # Uncheck the flag for all future iterations
                else:
                    # For subsequent patches, we use the dynamic, updated result
                    eps = 1e-8
                    normalized = accum_image / (accum_weight + eps)
                    current_patch = normalized[yy0:yy1, xx0:xx1]

                # --- Find the best patch ---
                min_ssd = np.inf
                best_patch = None

                num_samples = min(500, len(source_patches))
                indices = np.random.choice(len(source_patches), num_samples, replace=False)
                
                valid_mask = ~patch_mask_in_fill
                
                if not is_mono:
                    valid_mask_3d = np.stack([valid_mask]*3, axis=-1)
                else:
                    valid_mask_3d = valid_mask

                masked_target = current_patch * valid_mask_3d

                for i in indices:
                    source_patch = source_patches[i]
                    
                    # --- SSD calculation with gradient component ---
                    masked_source = source_patch * valid_mask_3d

                    # Intensity
                    ssd_intensity = np.sum((masked_target - masked_source)**2)

                    # Gradients (we use Sobel for dx/dy)
                    grad_t_x = cv2.Sobel(masked_target, cv2.CV_32F, 1, 0, ksize=3)
                    grad_t_y = cv2.Sobel(masked_target, cv2.CV_32F, 0, 1, ksize=3)
                    grad_s_x = cv2.Sobel(masked_source, cv2.CV_32F, 1, 0, ksize=3)
                    grad_s_y = cv2.Sobel(masked_source, cv2.CV_32F, 0, 1, ksize=3)

                    # Gradient consistency
                    # We sum the squared differences of gradients only on valid pixels
                    ssd_grad = np.sum((grad_t_x - grad_s_x)**2 + (grad_t_y - grad_s_y)**2)

                    # Normalization by number of valid pixels
                    ssd_grad /= (np.sum(valid_mask) + 1e-6)

                    # --- Weighted combination WITH ALPHA ADAPTIVE ---
                    # gradient weight, to be adjusted (0.3–1.0)
                    ssd = ssd_intensity + adaptive_alpha * ssd_grad
                    
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_patch = source_patch
                
                if best_patch is None:
                    best_patch = source_patches[0]

                # --- Color normalization step ---
                # We extract the valid (already known) pixels in the target area.
                # These pixels are located in the overlap region.
                valid_pixels_in_destination = current_patch[valid_mask]
                num_valid_pixels = valid_pixels_in_destination.size

                min_pixels_for_local_mean = (PATCH_SIZE * PATCH_SIZE) * 0.25

                # If there are enough valid pixels for a reliable local estimate...
                if num_valid_pixels > min_pixels_for_local_mean: # Minimum threshold to avoid statistical errors
                    # ...we calculate the local average color that our patch must match.
                    local_target_mean = np.mean(valid_pixels_in_destination, axis=0)
                else:
                    # Otherwise (e.g., for the first patch at the center of a large hole),
                    # we use the global ring mean as a fallback.
                    local_target_mean = ref_mean

                # We calculate a "weight" based on the number of available local pixels.
                # This weight determines how much we trust the local context (even though
                # our local context is currently just another reference to the global mean).
                patch_height, patch_width, _ = current_patch.shape
                patch_area = patch_height * patch_width
                max_reliable_pixels = patch_area # * 0.3 # Set the threshold to 30% of the total pixels in the patch

                # "CONFIDENCE METER" that decides how much to trust the color of nearby pixels (local context)
                # versus the average color of the entire reference area (global context).
                # The weight increases linearly from 0.1 (low confidence) to 0.9 (high confidence).
                
                # example:
                # The weight increases from 0.1 (low confidence) to 0.7 (high confidence).
                # The influence of the global average will never fall below 30%.
                # local_weight = 0.1 + 0.6 * np.clip(num_valid_pixels / max_reliable_pixels, 0, 1)
                local_weight = 0.1 + 0.9 * np.clip(num_valid_pixels / max_reliable_pixels, 0, 1)

                # The "final target mean" is a weighted average of the local and global targets.
                # This makes it stable and accurate.
                final_target_mean = (local_target_mean * local_weight) + (ref_mean * (1.0 - local_weight))

                # Calculate the mean of the best source patch.
                patch_mean = np.mean(best_patch, axis=(0, 1))

                # Calculate the scale factor to match the patch's mean to our final target mean.
                # Since final_target_mean is always equal or very close to ref_mean, this ensures
                # all patches are consistently normalized to the global context.
                scale = (final_target_mean + 1e-8) / (patch_mean + 1e-8)

                # self.siril.log(f"num_valid_pixels: {num_valid_pixels}", s.LogColor.GREEN)
                # self.siril.log(f"local_target_mean: {local_target_mean}", s.LogColor.GREEN)
                # self.siril.log(f"local_weight: {local_weight:.3f}", s.LogColor.GREEN)
                # self.siril.log(f"final_target_mean: {final_target_mean}", s.LogColor.GREEN)
                # self.siril.log(f"patch_mean: {patch_mean}", s.LogColor.GREEN)
                # self.siril.log(f"ref_mean: {ref_mean}", s.LogColor.GREEN)
                # self.siril.log(f"scale: {scale}", s.LogColor.GREEN)

                # --- Scale Factor Clamping (VERY CONSERVATIVE) ---
                # The scale factor is heavily restricted to prevent almost any change.
                # This allows only a minuscule correction (±2%), making the algorithm
                # extremely stable and preventing any significant color shifting.
                scale = np.clip(scale, 0.50, 1.50)

                # self.siril.log(f"Clamped scale: {scale}", s.LogColor.GREEN)

                # Let's apply the correction.
                best_patch = best_patch * scale

                # Clamp for safety
                best_patch = np.clip(best_patch, 0, 1)

                # --- Building a 4-Sided Dynamic Blend Mask ---
                # The blend_mask is now always fixed size
                blend_mask = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

                # We create the ramp only once for efficiency
                ramp_forward = self.smoothstep_ramp(OVERLAP) # Using the smoothstep function
                
                # --- LEFT SIDE ---
                if np.any(valid_mask[:, 0:OVERLAP]):
                    ramp_h = ramp_forward[np.newaxis, :]
                    ramp_h = cv2.GaussianBlur(ramp_h, (5, 1), 0)
                    blend_mask[:, 0:OVERLAP] = np.minimum(blend_mask[:, 0:OVERLAP], ramp_h)
                
                # --- RIGHT SIDE ---
                if np.any(valid_mask[:, -OVERLAP:]):
                    ramp_h_inv = ramp_forward[::-1][np.newaxis, :] # Inverted ramp
                    ramp_h_inv = cv2.GaussianBlur(ramp_h_inv, (5, 1), 0)
                    blend_mask[:, -OVERLAP:] = np.minimum(blend_mask[:, -OVERLAP:], ramp_h_inv)

                # --- TOP SIDE ---
                if np.any(valid_mask[0:OVERLAP, :]):
                    ramp_v = ramp_forward[:, np.newaxis]
                    ramp_v = cv2.GaussianBlur(ramp_v, (1, 5), 0)
                    blend_mask[0:OVERLAP, :] = np.minimum(blend_mask[0:OVERLAP, :], ramp_v)
                    
                # --- BOTTOM SIDE ---
                if np.any(valid_mask[-OVERLAP:, :]):
                    ramp_v_inv = ramp_forward[::-1][:, np.newaxis] # Inverted ramp
                    ramp_v_inv = cv2.GaussianBlur(ramp_v_inv, (1, 5), 0)
                    blend_mask[-OVERLAP:, :] = np.minimum(blend_mask[-OVERLAP:, :], ramp_v_inv)

                # We update 'result_mosaic' to guide the next iteration.
                # Without it, it would be a "blind" algorithm.
                if is_mono:
                    # Simple blending for the guide image
                    current_content = result_mosaic[yy0:yy1, xx0:xx1]
                    blended_patch = current_content * (1 - blend_mask) + best_patch * blend_mask
                    result_mosaic[yy0:yy1, xx0:xx1] = blended_patch
                else:
                    # Broadcast Blending for RGB
                    blend_mask_3d = blend_mask[..., None]
                    current_content = result_mosaic[yy0:yy1, xx0:xx1]
                    blended_patch = current_content * (1 - blend_mask_3d) + best_patch * blend_mask_3d
                    result_mosaic[yy0:yy1, xx0:xx1] = blended_patch

                # --- Accumulation with weights ---
                bp = best_patch.astype(np.float32)
                ap = blend_mask.astype(np.float32)  # 'ap' (alpha patch) always has shape (P, P)

                # A factor of 1.0 is the default behavior.
                # Higher values ​​(e.g., 1.5 or 2.0) force the new patch to have a greater impact.
                correction_factor = 1.5  # Example: 50% stronger correction
                weighted_ap = ap * correction_factor

                if is_mono:
                    # Monochrome Case: Shapes are simple and straightforward.
                    # accum_image[slice] is (P, P), bp is (P, P), ap is (P, P)
                    # accum_weight[slice] is (P, P)
                    accum_image[yy0:yy1, xx0:xx1] += bp * weighted_ap
                    accum_weight[yy0:yy1, xx0:xx1] += weighted_ap
                else:
                    # RGB case: We make broadcasting explicit for both operations.
                    # 'ap' is expanded to (P, P, 1) to be compatible with 3-channel arrays.
                    ap_broadcast = weighted_ap[..., None]  # Shape: (P, P, 1)

                    # accum_image[slice] is (P, P, 3), bp is (P, P, 3)
                    # The bp * ap_broadcast operation works correctly thanks to broadcasting.
                    accum_image[yy0:yy1, xx0:xx1] += bp * ap_broadcast

                    # accum_weight[slice] is (P, P, 1) and ap_broadcast is (P, P, 1).
                    # The shapes match perfectly, without the need for slicing or conditionals.
                    accum_weight[yy0:yy1, xx0:xx1] += ap_broadcast
            
                # Update the counter with the newly filled pixels
                pixels_filled_count += np.sum(patch_mask_in_fill)
                
                # Update the fill_mask to indicate that this area has been filled.
                # This is ESSENTIAL for the next patch to create the correct blend.
                fill_mask[yy0:yy1, xx0:xx1] = False

                progress = (pixels_filled_count / pixels_to_fill_total) * 100 if pixels_to_fill_total > 0 else 0
                self.update_status(f"Building mosaic... {progress:.0f}%")

        # Final normalization (safe division)
        eps = 1e-8
        normalized = accum_image / (accum_weight + eps)

        # To be safe, where accum_weight is zero (no patch) we keep the original target
        no_contrib = (accum_weight[..., 0] if (not is_mono and accum_weight.ndim==3) else accum_weight) <= eps
        if is_mono:
            normalized[no_contrib] = target_image[no_contrib]
        else:
            normalized[no_contrib, :] = target_image[no_contrib, :]

        # final image is normalized
        result_mosaic = normalized

        # --- Final Fusion with Pyramid Blending ---
        
        # Blending parameters from the GUI
        feather_radius = self.param3_slider.value()

        # If the defect is small, use a simpler fusion
        if h_mask < 64 and w_mask < 64:
            self.update_status("Small defect detected. Using direct feather blend instead of pyramid blend...")
            self.siril.log("Small defect detected. Using direct feather blend instead of pyramid blend.", s.LogColor.BLUE)

            self.update_status("Calibrating patch color...")
            self.siril.log("Performing local gain/bias calibration on the synthesized patch.", s.LogColor.BLUE)

            # The ring width is proportional to the size of the defect,
            # with a minimum limit of 5px and a maximum of 20px to ensure robustness.
            ring_width = int(np.clip(min_dimension / 3, 5, 20))
            self.siril.log(f"Adaptive ring width set to: {ring_width} px", s.LogColor.BLUE)

            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))

            # We use the main mask 'mask' to define the ring
            mask_bool = mask.astype(bool)

            # Outer ring (in the original image)
            dilated = cv2.dilate(mask.astype(np.uint8), kern, iterations=1).astype(bool)
            outer_ring = dilated & (~mask_bool)

            # Inner ring (in the generated patch)
            eroded = cv2.erode(mask.astype(np.uint8), kern, iterations=1).astype(bool)
            inner_ring = mask_bool & (~eroded)

            # Union of the rings = calibration band
            ring_mask = outer_ring | inner_ring

            # Extract the ring coordinates
            ring_y, ring_x = np.where(ring_mask)
            
            if ring_y.size >= 20: # We use a minimum pixel threshold for robust estimation
                # Values ​​in the ring from the ORIGINAL IMAGE (our "destination" or ground truth)
                dest_ring_vals = target_image[ring_y, ring_x].astype(np.float32)

                # Values ​​in the ring from the SYNTHESIZED PATCH (our "source")
                src_ring_vals = result_mosaic[ring_y, ring_x].astype(np.float32)

                # Calculate a (gain) and b (bias)
                if not is_mono:
                    # Luminance for estimating 'a'
                    dY = 0.299 * dest_ring_vals[:,0] + 0.587 * dest_ring_vals[:,1] + 0.114 * dest_ring_vals[:,2]
                    sY = 0.299 * src_ring_vals[:,0]  + 0.587 * src_ring_vals[:,1]  + 0.114 * src_ring_vals[:,2]
                    mean_s, mean_d = float(np.mean(sY)), float(np.mean(dY))
                    var_s = float(np.var(sY))
                    if var_s < 1e-8:
                        a = 1.0
                    else:
                        cov = float(np.mean((sY - mean_s) * (dY - mean_d)))
                        a = cov / (var_s + 1e-12)

                    a = float(np.clip(a, 0.7, 1.3)) # Clamp for safety

                    # Bias per channel to align averages
                    mean_src_vec = np.mean(src_ring_vals, axis=0)
                    mean_dst_vec = np.mean(dest_ring_vals, axis=0)
                    b_vec = mean_dst_vec - a * mean_src_vec

                    max_b_vec = np.maximum(np.abs(mean_dst_vec) * 0.5, 1e-3)
                    b_vec = np.clip(b_vec, -max_b_vec, max_b_vec) # Clamp for safety

                    # Apply the correction to ALL pixels INSIDE the patch
                    pixels_to_correct = result_mosaic[mask]
                    corrected_pixels = a * pixels_to_correct + b_vec
                    result_mosaic[mask] = corrected_pixels

                else: # MONO
                    mean_s, mean_d = float(np.mean(src_ring_vals)), float(np.mean(dest_ring_vals))
                    var_s = float(np.var(src_ring_vals))
                    if var_s < 1e-8:
                        a = 1.0
                        b = mean_d - mean_s
                    else:
                        cov = float(np.mean((src_ring_vals - mean_s) * (dest_ring_vals - mean_d)))
                        a = cov / (var_s + 1e-12)
                        b = mean_d - a * mean_s

                    a = float(np.clip(a, 0.7, 1.3))
                    max_b = max(abs(mean_d) * 0.5, 1e-3)
                    b = float(np.clip(b, -max_b, max_b))

                    # Apply the correction to ALL pixels INSIDE the patch
                    pixels_to_correct = result_mosaic[mask]
                    corrected_pixels = a * pixels_to_correct + b
                    result_mosaic[mask] = corrected_pixels

            final_image = self._feather_blend(target_image, result_mosaic, mask, radius=feather_radius)

        else:
            # Otherwise, proceed with the pyramid merger
            self.update_status("Applying final pyramid blend...")
        
            # Calculate the distance from every white pixel to the nearest black pixel
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)

            # --- Raggio sfumatura in percentuale ---
            # 1. Trova la distanza massima (dal bordo al punto più centrale della selezione)
            max_dist = dist.max()
            if max_dist == 0: max_dist = 1 # Evita divisione per zero in maschere di 1px
            # 2. Calcola il "raggio effettivo" in pixel basato sulla percentuale data dallo slider.
            #    Es: slider a 100 -> 1.0; slider a 50 -> 0.5
            effective_radius = max_dist * (feather_radius / 100.0)
            # 3. Normalizza la distanza usando questo raggio effettivo e dinamico.
            #    Se la percentuale è 100, effective_radius = max_dist, quindi il centro avrà valore 1.0.
            alpha_mask_lineare = np.clip(dist / (effective_radius + 1e-8), 0.0, 1.0)

            # Gradient profile
            # --------------------------------------------------------------------------
            # alpha = alpha_mask_lineare  # Linear Profile (default)
            # alpha = np.power(alpha_mask_lineare, 2.0)  # Convex Profile (Ease-in)
            # alpha = np.power(alpha_mask_lineare, 0.5)  # Concave Profile (Ease-out)
            alpha = 0.5 * (1 - np.cos(alpha_mask_lineare * np.pi)) # S-profile (SmoothStep)
            # --------------------------------------------------------------------------

            # Optional: add subtle noise ONLY in the transition zone (0 < alpha < 1)
            # noise_strength = 0.2  # tunable
            # transition_zone = (alpha_mask_lineare > 0) & (alpha_mask_lineare < 1)
            # noise = np.random.normal(0, noise_strength, alpha.shape).astype(np.float32)
            # # Apply noise only to the feather region
            # alpha[transition_zone] = np.clip(alpha[transition_zone] + noise[transition_zone], 0.0, 1.0)

            # We calculate a blur kernel size that is proportional to the OVERLAP.
            # blur_ksize = 3
            # blur_ksize = max(3, self.param2_slider.value() // 4) | 1
            blur_ksize = int(OVERLAP) # We use the width of the overlap as a guide
            if blur_ksize < 3:
                blur_ksize = 3
            if blur_ksize % 2 == 0:
                blur_ksize += 1 # Make sure the kernel is odd

            # Apply a light blur to avoid “steps”
            alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)

            # Calculate the maximum number of levels theoretically possible.
            # Each level halves the size, so we use the logarithm to the base 2.
            # We stop when the size drops below a threshold (e.g., 8 pixels).
            max_levels = int(np.log2(min_dimension / 8)) if min_dimension > 8 else 1
            max_levels = max(1, max_levels) # Let's make sure it's at least 1

            # Read the value from the UI
            user_levels = self.param4_slider.value()

            # Use the smaller value between the user's choice and the calculated maximum
            num_levels = min(user_levels, max_levels)

            if num_levels < user_levels:
                self.siril.log(f"User requested {user_levels} levels, but dynamically limited to {num_levels} for stability.", s.LogColor.BLUE)
            else:
                self.siril.log(f"Pyramid levels set to: {user_levels}", s.LogColor.BLUE)

            # Now we perform the pyramid blending.
            # 'target image' is the background, 'result mosaic' is the foreground (our patch).
            # The (slightly blurred) 'mask' tells us where to apply the blending.
            tonally_corrected_image = self._pyramid_blend(result_mosaic, target_image, alpha, num_levels=num_levels)
           
            # !!! Devo aggiunge alla GUI un CheckBox per applicare o meno la feather blend finale
            #     per evitare contaminazioni di colori oltre la maschera.
            final_image = self._feather_blend(target_image, tonally_corrected_image, mask, radius=feather_radius)

        self.update_status("Texture synthesis completed.")
        return final_image

    def smoothstep_ramp(self, size):
        """
        Generate a blending ramp of length `size`.

        Different ramp profiles are available.
        To switch between them, just comment/uncomment the return line you want.

        - Linear: straight ramp [0 → 1]
        - Smoothstep (cosine): soft S-curve
        - Ease-in (convex): starts slow, ends fast
        - Ease-out (concave): starts fast, ends slow
        - S-curve strong: more pronounced smoothstep
        """
        t = np.linspace(0.0, 1.0, size)

        # --- Linear profile ---
        # return t

        # --- Smoothstep (cosine-based, default) ---
        return (1 - np.cos(t * np.pi)) / 2

        # --- Ease-in (convex, quadratic) ---
        # return t**2

        # --- Ease-out (concave, square root) ---
        # return np.sqrt(t)

        # --- Strong S-curve (cosine applied twice) ---
        # return (1 - np.cos((t**2) * np.pi)) / 2

    def _pyramid_blend(self, foreground, background, mask, num_levels=4):
        """
        Blends a foreground image onto a background image using a mask and Laplacian pyramids.
        The mask defines where the foreground is visible (values close to 1).
        """
        # Create Gaussian pyramids for the foreground, background and mask
        G_fg = foreground.copy()
        G_bg = background.copy()
        G_mask = mask.astype(np.float32)

        gp_fg = [G_fg]
        gp_bg = [G_bg]
        gp_mask = [G_mask]

        for i in range(num_levels):
            G_fg = cv2.pyrDown(G_fg)
            G_bg = cv2.pyrDown(G_bg)
            G_mask = cv2.pyrDown(G_mask)
            gp_fg.append(G_fg)
            gp_bg.append(G_bg)
            gp_mask.append(G_mask)

        # Create the Laplacian pyramids
        lp_fg = [gp_fg[num_levels]]
        lp_bg = [gp_bg[num_levels]]

        for i in range(num_levels, 0, -1):
            h, w = gp_fg[i-1].shape[:2]
            L_fg = cv2.subtract(gp_fg[i-1], cv2.pyrUp(gp_fg[i], dstsize=(w, h)))
            L_bg = cv2.subtract(gp_bg[i-1], cv2.pyrUp(gp_bg[i], dstsize=(w, h)))
            lp_fg.append(L_fg)
            lp_bg.append(L_bg)

        # Merge the Laplacian pyramids at each level
        LS = []
        for lf, lb, gm in zip(lp_fg, lp_bg, gp_mask[::-1]):
            # Ensures the mask is broadcastable on color channels
            if lf.ndim == 3 and gm.ndim == 2:
                gm = np.stack([gm]*3, axis=-1)
            
            ls = lf * gm + lb * (1.0 - gm)
            LS.append(ls)

        # Reconstruct the final image by collapsing the pyramid
        blended_image = LS[0]
        for i in range(1, num_levels + 1):
            h, w = LS[i].shape[:2]
            blended_image = cv2.pyrUp(blended_image, dstsize=(w, h))
            blended_image = cv2.add(blended_image, LS[i])
            
        return blended_image

    def run_opencv_inpainting(self, image_to_process, is_mono, mask_to_fill, context_factor, feather_radius, inpaint_radius):
        """
        Performs inpainting in two stages:
        1. Structural filling with OpenCV's Navier-Stokes algorithm.
        2. Spectral texture synthesis to restore the natural grain.
        """
        # Structural inpainting to create a homogeneous base
        self.update_status("Basic inpainting execution (Navier-Stokes)...")
        
        ys, xs = np.nonzero(mask_to_fill)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        H_hole, W_hole = y1 - y0, x1 - x0

        # h_mask = np.max(ys) - np.min(ys)
        # w_mask = np.max(xs) - np.min(xs)
        # min_dimension = min(h_mask, w_mask)

        # Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted
        mask_cv2_inpaint = (mask_to_fill.astype(np.uint8) * 255)

        # Inpainting to create a smooth base
        # https://docs.opencv.org/4.10.0/d7/d8b/group__photo__inpaint.html
        self.update_status("Executing base inpainting...")
        if is_mono:
            filled_image = cv2.inpaint(image_to_process, mask_cv2_inpaint, inpaint_radius, cv2.INPAINT_NS)
        else:
            filled_image = image_to_process.copy()
            for c in range(3):
                filled_image[..., c] = cv2.inpaint(image_to_process[..., c], mask_cv2_inpaint, inpaint_radius, cv2.INPAINT_NS)

        # result_image = image_to_process.copy()
        # # Copy the patch pixels to the resulting image only where the mask is active
        # if is_mono:
        #     result_image[mask_to_fill] = filled_image[mask_to_fill]
        # else:
        #     # For RGB, the mask must be expanded to cover all 3 channels
        #     result_image[mask_to_fill] = filled_image[mask_to_fill]

        # Apply a Gaussian Blur to the entire filled image.
        self.update_status("Smoothing inpainted area...")

        k_size = 55
        blurred_filled_image = cv2.GaussianBlur(filled_image, (k_size, k_size), 0)
        # Use the mask to apply the blur ONLY to the area we filled.
        # Copy the blurred pixels onto the original filled image, but only where the mask is active.
        filled_image_smoothed = filled_image.copy()
        # Copy the blurred pixels onto the 'smoothed' image only where the mask is active.
        filled_image_smoothed[mask_to_fill] = blurred_filled_image[mask_to_fill]
        # Blends the patch with the original for a seamless color transition
        result_image = self._feather_blend(image_to_process, filled_image_smoothed, mask_to_fill, radius=feather_radius)

        # Add texture with spectral synthesis ---
        self.update_status("Synthesizing spectral texture...")

        # --- START LOGIC: FIND A CLEAN "DONOR" PATCH ---        
        # 1. Let's define the size of the donor we are looking for.
        #    Ideally at least as big as the hole to capture the texture well.
        donor_h = max(32, H_hole)
        donor_w = max(32, W_hole)
        donor_patch = None
        
        # 2. Let's create a list of possible positions to search (above, below, left, right).
        #    The 'gap' moves the start of the search slightly away from the edge of the hole.
        gap = 5 
        search_positions = [
            (y0 - donor_h - gap, x0),                  # Sopra
            (y1 + gap, x0),                            # Sotto
            (y0, x0 - donor_w - gap),                  # Sinistra
            (y0, x1 + gap),                            # Destra
            (y0 - donor_h - gap, x0 - donor_w - gap),  # Sopra-Sinistra
            (y0 - donor_h - gap, x1 + gap),            # Sopra-Destra
            (y1 + gap, x0 - donor_w - gap),            # Sotto-Sinistra
            (y1 + gap, x1 + gap),                      # Sotto-Destra
        ]

        h_img, w_img = image_to_process.shape[:2]

        # 3. We cycle through the positions and take the first valid one we find.
        for top, left in search_positions:
            y_start, y_end = top, top + donor_h
            x_start, x_end = left, left + donor_w
            
            # Let's check if the patch is within the image bounds
            if y_start >= 0 and y_end < h_img and x_start >= 0 and x_end < w_img:
                # Check if the patch overlaps the area to be filled
                if not np.any(mask_to_fill[y_start:y_end, x_start:x_end]):
                    # Found! We extract the donor and exit the loop.
                    donor_patch = image_to_process[y_start:y_end, x_start:x_end]
                    self.siril.log(f"Found a clean donor patch at: ({y_start}, {x_start})", s.LogColor.GREEN)
                    break 
        
        # 4. Fallback: If we don't find any clean donors, we revert to the original method (hole window).
        if donor_patch is None:
            self.siril.log("Could not find a clean donor patch, falling back to original windowed method.", s.LogColor.RED)
            pad = max(8, int(min(H_hole, W_hole) * context_factor))
            y0p = max(0, y0 - pad); y1p = min(h_img, y1 + pad)
            x0p = max(0, x0 - pad); x1p = min(w_img, x1 + pad)
            
            texture_sample = image_to_process[y0p:y1p, x0p:x1p]
            valid_mask_for_synth = ~mask_to_fill[y0p:y1p, x0p:x1p]
        else:
            # We have a donor! The texture sample is the donor himself.
            texture_sample = donor_patch
            # The valid mask is simply an array of True, because the donor is all valid.
            valid_mask_for_synth = np.ones_like(donor_patch[..., 0] if not is_mono else donor_patch, dtype=bool)
            # Define y0p, x0p etc. to match the hole for the blending step later
            y0p, y1p = y0, y1
            x0p, x1p = x0, x1

        Hbox, Wbox = y1 - y0, x1 - x0 # The size of the output is always that of the hole
            
        # --- Blending Mask Alpha Patch ---
        # --- Make the alpha_patch mask blurred (Distance Transform Method) ---
        if feather_radius > 0:
            # Extract the local mask patch to work on
            mask_patch = mask_to_fill[y0:y1, x0:x1] # Use the exact hole for the alpha mask base

            # I create the precise ramp with Distance Transform on the patch
            dist = cv2.distanceTransform(mask_patch.astype(np.uint8), cv2.DIST_L2, 3)

            # --- Raggio sfumatura in percentuale ---
            # 1. Trova la distanza massima (dal bordo al punto più centrale della selezione)
            max_dist = dist.max()
            if max_dist == 0: max_dist = 1 # Evita divisione per zero in maschere di 1px
            # 2. Calcola il "raggio effettivo" in pixel basato sulla percentuale data dallo slider.
            #    Es: slider a 100 -> 1.0; slider a 50 -> 0.5
            effective_radius = max_dist * (feather_radius / 100.0)
            # 3. Normalizza la distanza usando questo raggio effettivo e dinamico.
            #    Se la percentuale è 100, effective_radius = max_dist, quindi il centro avrà valore 1.0.
            alpha_mask_lineare = np.clip(dist / (effective_radius + 1e-8), 0.0, 1.0)

            # Gradient profile
            # --------------------------------------------------------------------------
            # alpha_patch = alpha_mask_lineare  # Linear Profile (default)
            # alpha_patch = np.power(alpha_mask_lineare, 2.0)  # Convex Profile (Ease-in)
            # alpha_patch = np.power(alpha_mask_lineare, 0.5)  # Concave Profile (Ease-out)
            alpha_patch = 0.5 * (1 - np.cos(alpha_mask_lineare * np.pi)) # S-profile (SmoothStep)
            # --------------------------------------------------------------------------

            # Optional: add subtle noise ONLY in the transition zone (0 < alpha < 1)
            # noise_strength = 0.5  # tunable
            # transition_zone = (alpha_mask_lineare > 0) & (alpha_mask_lineare < 1)
            # noise = np.random.normal(0, noise_strength, alpha_patch.shape).astype(np.float32)
            # # Apply noise only to the feather region
            # alpha_patch[transition_zone] = np.clip(alpha_patch[transition_zone] + noise[transition_zone], 0.0, 1.0)

            # I apply a light blur to soften the ramp
            alpha_patch = cv2.GaussianBlur(alpha_patch, (3, 3), 0)
        else:
            # If the radius is zero, the mask is sharp (but must be a patch)
            alpha_patch = mask_to_fill[y0:y1, x0:x1].astype(np.float32)

        # --- MONO case ---
        if is_mono:
            sigma_est = self._estimate_local_noise_std(image_to_process, mask_to_fill)
            noise_patch = self._synthesize_noise_with_power_spectrum(texture_sample, valid_mask_for_synth, out_shape=(Hbox, Wbox), sigma=sigma_est)

            # Blend correctly with the edge
            base = result_image[y0:y1, x0:x1]
            filled = base + noise_patch
            
            # Alpha blending
            # result_image[y0:y1, x0:x1] = base * (1 - alpha_patch) + filled * alpha_patch

            # --- Replacing Alpha Blending with Weighted Averaging ---
            weight_patch = alpha_patch
            weight_original = 1.0 - weight_patch
            total_weight = weight_patch + weight_original + 1e-8
            result_image[y0:y1, x0:x1] = (filled * weight_patch + base * weight_original) / total_weight

        # --- RGB case ---
        else:
            # Use a single channel (e.g., green or luminance) to estimate the TEXTURE 
            green_sample = texture_sample[:, :, 1]

            # Estimate the sigma of the master channel and synthesize the base texture
            sigma_master = self._estimate_local_noise_std(image_to_process[:, :, 1], mask_to_fill)
            
            # Sintesi del pattern di rumore usando il NUOVO campione puro del canale verde
            noise_master_pattern = self._synthesize_noise_with_power_spectrum(green_sample, valid_mask_for_synth, out_shape=(Hbox, Wbox), sigma=sigma_master)

            # Apply the master pattern to each channel, scaled correctly
            for c in range(3):
                self.update_status(f"Applying texture to channel {c+1}/3...")

                # Estimate the specific sigma of the current channel
                sigma_c = self._estimate_local_noise_std(image_to_process[:, :, c], mask_to_fill)
                # Calculate the scale factor with respect to the master sigma
                scale = (sigma_c / (sigma_master + 1e-8)) if sigma_master > 0 else 1.0
                original_scale = scale # Save the original value for the log
                
                # Clamp the scale factor to avoid extreme values ---
                min_scale = 0.7  # Minimum limit to avoid over-suppression
                max_scale = 1.5  # Maximum limit to avoid over-amplification
                scale = np.clip(scale, min_scale, max_scale)
                
                # self.siril.log(f"Scale (original: {original_scale:.2f} -> clamped: {scale:.2f}) - Channel {c}", s.LogColor.BLUE)

                # Apply the scaled master pattern
                noise_c = noise_master_pattern * scale

                # Perform the local blend as before
                base = result_image[y0:y1, x0:x1, c]
                filled = base + noise_c
                
                # Alpha blending
                # result_image[y0:y1, x0:x1, c] = base * (1 - alpha_patch) + filled * alpha_patch

                # --- Replacing Alpha Blending with Weighted Averaging ---
                weight_patch = alpha_patch
                weight_original = 1.0 - weight_patch
                total_weight = weight_patch + weight_original + 1e-8
                result_image[y0:y1, x0:x1, c] = (filled * weight_patch + base * weight_original) / total_weight

        self.update_status("Texture synthesis completed.")
        return result_image

    def _feather_blend(self, I, filled, M, radius=2):
        """Blends 'filled' onto 'I' using 'M' mask with DistanceTransform-based feathering."""
        alpha = M.astype(np.uint8)

        if radius > 0:
            # Calculate the distance from the edges inside the mask
            dist = cv2.distanceTransform(alpha, cv2.DIST_L2, 3)
            
            # --- Raggio sfumatura in percentuale ---
            # 1. Trova la distanza massima (dal bordo al punto più centrale della selezione)
            max_dist = dist.max()
            if max_dist == 0: max_dist = 1 # Evita divisione per zero in maschere di 1px
            # 2. Calcola il "raggio effettivo" in pixel basato sulla percentuale data dallo slider.
            #    Es: slider a 100 -> 1.0; slider a 50 -> 0.5
            effective_radius = max_dist * (radius / 100.0)
            # 3. Normalizza la distanza usando questo raggio effettivo e dinamico.
            #    Se la percentuale è 100, effective_radius = max_dist, quindi il centro avrà valore 1.0.
            alpha_mask_lineare = np.clip(dist / (effective_radius + 1e-8), 0.0, 1.0)

            # Gradient profile
            # --------------------------------------------------------------------------
            # alpha = alpha_mask_lineare  # Linear Profile (default)
            # alpha = np.power(alpha_mask_lineare, 2.0)  # Convex Profile (Ease-in)
            # alpha = np.power(alpha_mask_lineare, 0.5)  # Concave Profile (Ease-out)
            alpha = 0.5 * (1 - np.cos(alpha_mask_lineare * np.pi)) # S-profile (SmoothStep)
            # --------------------------------------------------------------------------

            # Optional: add subtle noise ONLY in the transition zone (0 < alpha < 1)
            # noise_strength = 0.2  # tunable
            # transition_zone = (alpha_mask_lineare > 0) & (alpha_mask_lineare < 1)
            # noise = np.random.normal(0, noise_strength, alpha.shape).astype(np.float32)
            # # Apply noise only to the feather region
            # alpha[transition_zone] = np.clip(alpha[transition_zone] + noise[transition_zone], 0.0, 1.0)

            # Apply a light blur to avoid “steps”
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        else:
            alpha = alpha.astype(np.float32)

        # If the image is in color, extend the mask to 3 channels
        if I.ndim == 3:
            alpha = alpha[:, :, np.newaxis]

        # Alpha blending
        # out = I * (1.0 - alpha) + filled * alpha

        # --- Replacing Alpha Blending with Weighted Averaging ---
        # The calculated alpha acts as the weight for the patch (weight_patch)
        weight_patch = alpha
        # The weight for the original image (background) is its complement
        weight_original = 1.0 - weight_patch
        # Sum of weights with an epsilon to prevent division by zero.
        total_weight = weight_patch + weight_original + 1e-8
        # Perform the normalized weighted average
        out = (filled * weight_patch + I * weight_original) / total_weight

        return out

    def _estimate_local_noise_std(self, image, mask, ring_width=500):
        """
        Robustly estimates the standard deviation of the background noise in a ring around the mask.
        """
        # Create a kernel for dilation to define the ring
        kernel = np.ones((ring_width, ring_width), np.uint8)
        
        # Dilate the mask to get the outer area of the ring
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
        # The ring is the dilated area minus the original mask
        ring_mask = (dilated_mask > 0) & (~mask)
        
        # Extract pixels from the ring
        valid_pixels = image[ring_mask]
        
        # If the local estimation doesn't have enough pixels, use a global fallback
        if valid_pixels.size < 50:
            self.siril.log("Local estimation failed (<50 pixels), falling back to global noise estimation.", s.LogColor.RED)
            
            # Calculate the robust estimation on the entire image as a fallback
            median_global = np.median(image)
            mad_global = np.median(np.abs(image - median_global))
            sigma_global = 1.4826 * mad_global
            return float(sigma_global)
                    
        # Calculate MAD (Median Absolute Deviation) - very robust to outliers (stars)
        median = np.median(valid_pixels)
        mad = np.median(np.abs(valid_pixels - median))
        
        # Convert MAD to a standard deviation estimate (sigma)
        # The factor 1.4826 is a constant for normally distributed data
        sigma = 1.4826 * mad
        return sigma

    def _synthesize_noise_with_power_spectrum(self, sample_window, valid_mask, out_shape=None, sigma=None, random_seed=None):
        """
        Synthesizes noise having the same power-spectrum (amplitude) as the sample region.

        Parameters
        ----------
        sample_window : 2D ndarray float32
            Sample region (e.g., area around the patch, size Hs x Ws).
            Must be float type (e.g., image_to_process[y0p:y1p, x0p:x1p]).
        valid_mask : 2D bool ndarray (same shape as sample_window)
            True = valid pixels (to include in spectrum estimation); False = pixels to ignore
            (typically corresponds to ~mask_to_fill within the window).
        out_shape : tuple (Hout, Wout) or None
            Size of the noise to return. If None, the shape of sample_window is used.
            N.B.: for best results, pass a sample window of a similar size to out_shape.
        sigma : float or None
            Desired standard deviation of the synthesized noise. If None, it is estimated from the valid pixels.
        random_seed : int or None
            Seed for reproducibility.

        Returns
        -------
        noise_patch : 2D float32 ndarray, shape out_shape
            Synthesized noise with a power-spectrum similar to the sample region,
            mean ~0 and standard deviation = sigma (if possible).
        """
        # Convert types & shapes
        samp = np.asarray(sample_window, dtype=np.float32)
        vm = np.asarray(valid_mask, dtype=bool)
        Hs, Ws = samp.shape
        if out_shape is None:
            Hout, Wout = Hs, Ws
        else:
            Hout, Wout = int(out_shape[0]), int(out_shape[1])

        # Extract valid pixels for robust calculations
        valid_pixels = samp[vm]
        if valid_pixels.size < 16:
            # Not enough pixels: fallback to white noise with sigma (or 0)
            fallback_sigma = float(sigma) if sigma is not None else (float(np.std(valid_pixels)) if valid_pixels.size > 0 else 0.0)
            rng = np.random.default_rng(random_seed)
            return (rng.standard_normal((Hout, Wout)).astype(np.float32) * fallback_sigma)

        # Robust level estimation (we use the mean of valid pixels to remove DC offset)
        mean_valid = float(np.mean(valid_pixels))
        samp0 = samp - mean_valid

        # Fill invalid pixels with zero (after subtracting the mean)
        samp0_filled = samp0.copy()
        samp0_filled[~vm] = 0.0

        # Apply a 2D window (Hann) to limit spectral leakage at the edges
        # https://www.tek.com/en/blog/window-functions-spectrum-analyzers
        # https://numpy.org/devdocs/reference/generated/numpy.blackman.html
        wy = np.blackman(Hs) if Hs > 1 else np.ones(1, dtype=np.float32)
        wx = np.blackman(Ws) if Ws > 1 else np.ones(1, dtype=np.float32)
        win2d = np.outer(wy, wx).astype(np.float32)
        samp_win = samp0_filled * win2d

        # 2D FFT and amplitude (magnitude)
        F = np.fft.fft2(samp_win)
        amplitude = np.abs(F)

        # If out_shape differs, resize the spectrum
        if (Hout, Wout) != (Hs, Ws):
            amp_resized = cv2.resize(amplitude, (Wout, Hout), interpolation=cv2.INTER_CUBIC)
        else:
            amp_resized = amplitude

        # Generate a random field in space, calculate its FFT
        rng = np.random.default_rng(random_seed)
        z = rng.standard_normal((Hout, Wout)).astype(np.float32)
        Z = np.fft.fft2(z)
        magZ = np.abs(Z)
        eps = 1e-12
        magZ += eps  # avoid division by zero

        # Scale the spectrum of the random noise
        S = amp_resized / magZ
        Zs = Z * S

        # Return to spatial domain with IFFT
        noise = np.real(np.fft.ifft2(Zs)).astype(np.float32)
        noise -= float(np.mean(noise))

        # If sigma is not given, estimate it
        if sigma is None:
            sigma_est = float(np.std(valid_pixels - mean_valid))
            sigma = sigma_est if sigma_est > 0 else 0.0
        else:
            sigma = float(sigma)

        cur_std = float(np.std(noise))
        if cur_std > 1e-12 and sigma > 0:
            noise = noise / cur_std * sigma
        elif sigma > 0:
            noise = (rng.standard_normal((Hout, Wout)).astype(np.float32) * sigma)

        return noise

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
        qapp.setApplicationName(f"Patch Inpainting Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        icon_data = base64.b64decode("""/9j/4AAQSkZJRgABAgAAZABkAAD/7AARRHVja3kAAQAEAAAAZAAA/+4AJkFkb2JlAGTAAAAAAQMAFQQDBgoNAAADDAAACRsAAAsYAAANX//bAIQAAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQICAgICAgICAgICAwMDAwMDAwMDAwEBAQEBAQECAQECAgIBAgIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMD/8IAEQgAQABAAwERAAIRAQMRAf/EALIAAAIDAQADAAAAAAAAAAAAAAAIBgcJBQEDBAEBAQEAAAAAAAAAAAAAAAAAAAECEAACAgICAQMFAAAAAAAAAAAFBgQHAgMQAREgMAgAUGAVNxEAAQQBAwMDAwMCBwAAAAAABAECAwUGERITACEUIhUHEDIjMUFRIENCUjN0JbUWEgEAAAAAAAAAAAAAAAAAAABgEwEAAgICAwEBAQEAAAAAAAABABEhMRAgQVFhcTCRwf/aAAwDAQACEQMRAAAByDAAAsJOQRRQlCNzZpfc0ivtI2Y+Z2FyJrBrGoFzWS4OZ2uEtdKAX0mwusUAqG50uKgDNJpvrFcLciLnKoaqzNAyKaZaxUssppVZa+VWJQbq52HuEemnc1FIlQrOlvUOuTpOUc4+VfKR1f/aAAgBAQABBQL0hVY0f1nBO0EX4ChSrEUFUfY8Msv0XSw6FaXxogQ4CBqXVsI+QkkiocV/oY97fRZJGcj+SmLwxlAyQHQ+1+wgQRRtYzY/lKnFVjtc+SCmYkWxfjMNKMT4/wA4TzTwvsvYRTtexGnszw7Yws0DHFdKnJSHdgsSFtHioROg5ZAzNGsk3O1JYJb31lpjsFdQd2mtrVBCwsnih/62Al2JIIWAkND8o/tNcGw41hEH6mbGlkJvfEAhPFS57w6lYopqZwWmIeOD8o5crEhEC5Utx//aAAgBAgABBQL8A//aAAgBAwABBQL3PPq79nx9ePsn/9oACAECAgY/AgH/2gAIAQMCBj8CAf/aAAgBAQEGPwL+mUiuGjUMaaOAw8osQIEJ0scszXFlFzQxwR8UD3ar/l/nt0fUTTQkyAEOh8kbl8clnZ0REHNHFKsM8Tkc3Vqdl+olLSBTWNoc97BQ4NvJKscT55O73NY1kUMTnucqo1rWqq9uhG2WJgTQPleKTGTb1tkwVk8UkMhsoGPXbreZ1a1/OjI2uVXRom132rALeMNs7F6Ro4vJCr3FZCZl7O8EDlpU4XvX0t/M5NNN6rr1Je4BM+Bo35T6azOYosYfbmOEsy3tfDGFFrJK2d79WIqtdqiNcPja2AJ7LsC6jFuwBbAZMnsijK+vsaqiddVw1Hcywwr4bJYipDNjpGxQwOe/fd2DnsOtKungsgKqb3J5uLKUlLSNryZRYhJxHAkSRQNDsJJ+No+5vpVWx/SlbiZ0NbkDJ5Zq40mdgw8UkI00jmTzSskh4yY2rDteisk5Ni9ndWgZ2MiRZEFUlrKPujs8WMHfdREm2FWDYpLOCW86dnb1/i09XbrbWuOpE2q1kdSbLAAxqoqK1KSbyKB7VRf0eK5Op3CyCk1aQTuLYE4SgMYEyNVJfJWkcmHWkj2rpI9YqxWt/udA2tXkRWVYFVFTQhSpIRBJixp80U0ox1NJNI2rlNmlY7mh1iI3Mfr+SPdX1drbmnh1iSNFYTM+R+yTi0bPK5eQpIGwNbFyK7iYiMbo1ERPrPnlJeVIFljRgEcdSZOvm3MFjzwFxwAsaqkhRsRGz6qzRsmrXI9qdD2NkJkdXbvgjo20kJgJtGQYbNzjkwwKQKa4h0kHH5DoGxwsftc71J1a4vQhrTnjTIFdWFmPVHSTRxRv/wCOEq1fcVI4SoS5XufITLJu+5qJp0tJd2kftd80W8JGEBoA1tdXt8Uq0mpw4CSZWuCZo0lyvakbO2iN/oohPFqDEb7gW6C8CIs69WBVpZT5H1YssE9lNC2LdFCjkR8iJr216wfMMgx1+QEg/Jfs0rYfjh+JWZVcVjFkXFB/55xZslxEBZRxEMfu0dxuYjdUdvw/OoLQM8UXMm42lZd/GI2FFjuu4EVZmQOV7bhjRYlRH/2ZNHJ/i0+dLfIcYosihw+1xQCtBIBghaRy2a+K6wnbG6UlGHlNfLr/AK0TONe3RubYdgmP22V3WfPrrgKrxT3QetpRqMVwwglazyH1le+dEc937vfqrv00y2uoxxxa2EoJ8QwuiDwTlVIBZ0UTG+iJjD55E2Jokf2oiafXFasic4WIkwjUisMmrz4XQV5ZEcgxg6tmHkbJEndvfo7DqXJ/mIHJqf3A+osbzJzCwQ7Wq5Q0NG0trJ0b2c7k3bIXrC9zWua53WB2Hy3kfyZll3k9ZBk1a4K6nKFoBimDzDOi9wsYpYp2Rzt3PjWR6vYuiIm3X5bpbHJsnuapcJhzUN092ayS0ma0uavdkbEckVxMCUF6HSN/ZHIjV7J8o5TX3OQ09tQJQNDkpLs+qilaaW+KZpsQUsSFojft3fb1hBFZERG7JvjrGsqtHFGEHTEW9w+wcaQ6Yl8j/wAnE3snbX64Z/vTf+osOs0Hy/CcdxLF1ociSLJaZkNbZv0dtHfISy4Of6w1fMruOPY9m7VP0X4ZOw+qkvRBcDq6cuYOYbaLYDDhDzwkc00XEkU8T2ucvparF1Xr5ImGUM0rG/iOrjKHlTnE88FLI7xCmNdGr4nxys3t1au137dfLUp9LjVMtZHj0cbccrpq5k6E2THOUpJjTOV0fD6dNumq9YQ8+nsqjxvj3Gawb3GBIPcR66IiGKxD0c7eGQzTaq6O7d0+sJ9YaXXHDKrhzQCZhC4HOY6NzoSR3xzRK5j1Tsqdl6kBtMvyiyCm7TBn5BbGCyondEkHILkifov8p1INSZHfU48zt00FVb2FfDK7TbukiEIiY92n8p0e8C5tQnWsUkNo4SxLGdZQzK5Zoj1hmYpkUqvXc2Tci69G1olnYC11lxe4145pMIVhwO3QeaLHI2Arhd3bvRdq/p0H7nYm2HtwcNcB5hMxPhgD68AY3K93CNDu9LG6NT6f/9oACAEBAwE/IeuS6ex1VplSILoteqGII3mdqbXIoVNtyiKRQgBYlR+PhVyKjhrPbmwJyeCPl4JJv3Bz2hRlKdEVXGVsk7+L1gSLiF1GiXlkcHJSZGYP9A9PYGyIiNb8aYe7Mt6Sj2JSUxqi0bFVDmf1WTQMBm/sWQWM2V42WEkF1cPKL63Z6s9Fm6ZJ48SrXklC5J3t69JMALjovImtBjyST6QlL5Q7vT0dCr0YoKNPZDsFb3IzIr1tXcnTWcwNLAtG/GHTQbjZ7rvNAjN9iuoKWZOcd9BabaZaYit2LqkdKgAcbk/2CO1XilnmPl9JaSmi/CJBDlo2sceSwELJshekIIaDy+CGsK2hiG2mibINYPkbfXOQqUC3pXJbUhrDu1LgJAkMXEi8DGKKqFS7vaGr/wBhWV2tI/JzKfqiNUbLw14eDZRf3L1kLL4Qf2hQEClcsaWE9sRGUKYWWxLbmv7nHGrRqMI99GnAu4XY2zd9tTWby+3lIATxRmPebmwcHH//2gAIAQIDAT8h/lUoidj+Nxep1d8nH7K4eTfPmeP5/wD/2gAIAQMDAT8h/lZLYe0Ycsscj/vV9cCKOjqfJ8nqeYa5dTJmZ8S9R3Dl1MQaXiqYdKO3/9oADAMBAAIRAxEAABAAAdgUqCAYtHgDKIAajugI+OAF+XAD8gj/2gAIAQEDAT8Q62yO6PVzaywbIg+LaxBMsquCLw8uhwFcco6PSGqlIE44cebAic/4qUeqgUTPrPJV2s+Cnx1zCDFD/pjAuRxNagZ52RMmOWo61r8ImwAmMjtxYJJBABJLI5SL4/CBkwDAqD4gXTgxp4KHEL6bTAd1VdceBJXF/RurwXhQ7kVlyiKg2CELyINZEIp4nhQjx5hAHeBdLzc1JQxzaE0nJyCT7/GvFuigKGLYckMYnAxTzaNmMZOeYxOe/vqGg4Of1VcXg6LpSN2Y4wCM+Dlj2lMlzGlBjKlMwmD1RItmlH/LOhlRAhe0xiWynvjQ0QW+kvN4NRp4We+s9kIS9EFrLNvQS6CPE1VdyNSn/dvCGkbvmUIrC60RTTdg0zFI61AwsNaVgvub7YwAoHpUwYeWxxu4DGF8pK7dexHq+LqBEX/xSDfAFEszAEYZDCus3Rq6xEpuw8O3KLyy72Zjwgk8z8bGYYataAOpBzgovj//2gAIAQIDAT8Q60sSmud6loQ8pCgqNV95LvG5RfsqVEQ+S16GMkDuJoivTaYwvuNmf+S9rDVhm56ORbCnGbmKMrmyQ0sFV+dAvyYiKFep5fku1Hx+dLZaS2Xz/9oACAEDAwE/EOqhBsvlQLdRq3F9alzUWW4ka5pWdShvEt8yx/YC0lQAZOlHLTFGtQkthZZvo6gvIxj3Cs08e7geDV3EMFwEai+XSZgsFTJSoE0wLupoHTEt/HnfEwptgWPcqw+soNX5nlXvne5RoIg7JRKNwA1x/9k=""")
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)
        app_icon = QIcon(pixmap)
        qapp.setWindowIcon(app_icon)

        qapp.setStyle("Fusion")

        # Define a Qt Style Sheet (QSS)
        stylesheet = """
            QPushButton[class="accent"] {
                background-color: #3574F0;  /* A nice blue color */
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
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton#helpButton:hover {
                background-color: #e0e0e0; /* Darker on hover */
                border: 1px solid #bbb;
            }
            
            /* Style specifically for the TEXT inside the CUSTOM HELP button */
            QPushButton#helpButton QLabel {
                color: #005A9C; /* A professional dark blue for readability */
                font-weight: bold;
                background-color: transparent; /* Ensure label background is clear */
                border: none;
            }
        """
        # Apply the stylesheet to the entire application
        qapp.setStyleSheet(stylesheet)

        # Now that the application context exists, create the main widget.
        app = PatchInpaintingTool()
        app.show()
        
        sys.exit(qapp.exec())

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()