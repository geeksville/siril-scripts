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
# This script allows you to insert a signature/logo (PNG file with transparency) onto the current image in Siril.
# You can save and manage multiple signature profiles through a graphical interface.
#
# Version History
# 1.0.0 - Initial release
# 1.0.1 - Add undo_save_state
#         Add handling of files with different bit depths
# 1.0.2 - Missing ensure_installed components
# 1.0.3 - Better filedialog for Linux
# 1.0.4 - Added contact information
# 2.0.0 - Ported to PyQt6
# 2.0.1 - CMD requires
# 2.0.2 - Added Icon App
# 2.0.3 - Minor Fix:
#         Added support for logo images without a transparency channel.
#         Improved positioning logic to correctly handle logos placed partially outside image boundaries.
#

VERSION = "2.0.3"
CONFIG_FILENAME = "SignatureTool.conf"

# --- Core Imports ---
import sys
import os
import base64
import traceback
import configparser

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s

    if not s.check_module_version('>=0.6.37'):
        print("Error: requires sirilpy module >= 0.6.37 (Siril >= 1.4.0 Beta 2)")
        sys.exit(1)

    # Import Siril GUI related components
    from sirilpy import SirilError
    
    s.ensure_installed("PyQt6", "numpy", "astropy", "opencv-python")

    # --- PyQt6 Imports ---
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QMessageBox, QGroupBox, QComboBox, QLineEdit, QSlider, QRadioButton,
        QGridLayout, QInputDialog, QFileDialog, QButtonGroup, QSizePolicy, QStyle
    )
    from PyQt6.QtGui import QIcon, QPixmap
    from PyQt6.QtCore import Qt, QTimer

    # --- Imports for Image Processing ---
    import cv2
    import numpy as np

    from astropy.io import fits
    # from photutils.background import MMMBackground, StdBackgroundRMS
    # from PIL import Image, ImageTk, ImageDraw, ImageFilter

except ImportError:
    print("Warning: sirilpy not found. The script is not running in the Siril environment.")

# --- Main Application Class ---
class SignatureApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"Signature Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")
        
        # --- Siril Connection ---
        # Initialize Siril connection
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            QMessageBox.critical(self, "Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
            QTimer.singleShot(0, self.close)
            return
 
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image is loaded")
            QTimer.singleShot(0, self.close)
            return

        shape_image = self.siril.get_image_shape()
        if shape_image[0] != 3:
            self.siril.error_messagebox("The image must be a RGB image.")
            QTimer.singleShot(0, self.close)
            return

        # --- State and configuration variables ---
        self.profiles = {}
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join(self.siril.get_siril_configdir(), CONFIG_FILENAME)

        self.load_config()
        self.create_widgets()
        self.center_window()
        
    def load_config(self):
        """ Load profiles from the configuration file. """
        self.profiles = {}
        self.config.read(self.config_path)

        # Get the list of sections from the config file and sort it alphabetically
        sorted_sections = sorted(self.config.sections())

        for section in sorted_sections:
            self.profiles[section] = {
                'path': self.config.get(section, 'path', fallback=''),
                'size': self.config.getint(section, 'size', fallback=5),
                'margin': self.config.getint(section, 'margin', fallback=2),
                'position': self.config.get(section, 'position', fallback='Bottom_Center'),
                'opacity': self.config.getint(section, 'opacity', fallback=100),
            }
        self.siril.log(f"Loaded {len(self.profiles)} profiles from {self.config_path}", s.LogColor.BLUE)

    def save_config(self):
        """ Save all profiles in the configuration file. """
        for profile_name, settings in self.profiles.items():
            if not self.config.has_section(profile_name):
                self.config.add_section(profile_name)
            self.config.set(profile_name, 'path', str(settings['path']))
            self.config.set(profile_name, 'size', str(settings['size']))
            self.config.set(profile_name, 'margin', str(settings['margin']))
            self.config.set(profile_name, 'position', str(settings['position']))
            self.config.set(profile_name, 'opacity', str(settings['opacity']))

        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)
        self.siril.log(f"Profiles saved in {self.config_path}", s.LogColor.GREEN)

    def create_widgets(self):
        """ Create all the elements of the graphical interface. """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Profiles Section ---
        profiles_group = QGroupBox("Signature Profiles")
        profiles_layout = QVBoxLayout(profiles_group)
        main_layout.addWidget(profiles_group)

        self.profile_combo = QComboBox()
        self.profile_combo.setPlaceholderText("--- Select a Profile ---")
        self.profile_combo.addItems(self.profiles.keys())
        # Set the initial index to -1 to show the placeholder text
        self.profile_combo.setCurrentIndex(-1)
        self.profile_combo.currentTextChanged.connect(self.on_profile_selected)
        profiles_layout.addWidget(self.profile_combo)

        buttons_sublayout = QHBoxLayout()
        save_new_button = QPushButton("Save New...")
        save_new_button.clicked.connect(self.save_current_profile)
        update_button = QPushButton("Update Selected")
        update_button.clicked.connect(self.update_current_profile)
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_current_profile)
        buttons_sublayout.addWidget(save_new_button)
        buttons_sublayout.addWidget(update_button)
        buttons_sublayout.addWidget(delete_button)
        profiles_layout.addLayout(buttons_sublayout)

        # --- Settings Section ---
        settings_group = QGroupBox("Current Settings")
        settings_layout = QGridLayout(settings_group)
        main_layout.addWidget(settings_group)

        # File Logo
        settings_layout.addWidget(QLabel("File Logo:"), 0, 0)
        self.logo_path_entry = QLineEdit()
        self.logo_path_entry.setReadOnly(True)
        select_file_button = QPushButton("Select...")
        select_file_button.clicked.connect(self.select_logo_file)
        settings_layout.addWidget(self.logo_path_entry, 0, 1)
        settings_layout.addWidget(select_file_button, 0, 2)

        # Size Slider
        self.size_label = QLabel("5")
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 100)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(lambda v: self.size_label.setText(str(v)))
        settings_layout.addWidget(QLabel("Size (%):"), 1, 0)
        settings_layout.addWidget(self.size_slider, 1, 1)
        settings_layout.addWidget(self.size_label, 1, 2)

        # Margin Slider
        self.margin_label = QLabel("2")
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(0, 50)
        self.margin_slider.setValue(2)
        self.margin_slider.valueChanged.connect(lambda v: self.margin_label.setText(str(v)))
        settings_layout.addWidget(QLabel("Margin (%):"), 2, 0)
        settings_layout.addWidget(self.margin_slider, 2, 1)
        settings_layout.addWidget(self.margin_label, 2, 2)

        # Opacity Slider
        self.opacity_label = QLabel("100")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(str(v)))
        settings_layout.addWidget(QLabel("Opacity (%):"), 3, 0)
        settings_layout.addWidget(self.opacity_slider, 3, 1)
        settings_layout.addWidget(self.opacity_label, 3, 2)
        
        settings_layout.setColumnStretch(1, 1)

        # --- Position Grid ---
        position_group = QGroupBox("Position")
        position_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        position_layout = QGridLayout(position_group)
        position_layout.setContentsMargins(50, 10, 10, 10) 
        main_layout.addWidget(position_group)
        
        self.position_button_group = QButtonGroup(self)
        self.position_radios = {}

        # Define the positions in a 2D grid to make it easier to create
        positions_grid = [
            ["Top_Left",    "Top_Center",    "Top_Right"],
            ["Middle_Left", "Middle_Center", "Middle_Right"],
            ["Bottom_Left", "Bottom_Center", "Bottom_Right"]
        ]

        # Loop to create and position the 9 radio buttons in the grid
        for r, row_list in enumerate(positions_grid):
            for c, position_value in enumerate(row_list):
                radio = QRadioButton(position_value.replace('_', ' '))
                self.position_button_group.addButton(radio)
                self.position_radios[position_value] = radio
                position_layout.addWidget(radio, r, c, Qt.AlignmentFlag.AlignLeft)
        
        self.position_radios["Bottom_Center"].setChecked(True)

        main_layout.addSpacing(10)
        
        # --- Actions Section ---
        action_layout = QHBoxLayout()
        # action_layout.addStretch(1) # Pushes the button to the right
        apply_button = QPushButton("     Apply Signature     ")
        apply_button.clicked.connect(self.on_apply)
        apply_button.setProperty("class", "accent")
        apply_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        action_layout.addWidget(apply_button)
        main_layout.addLayout(action_layout)

    def center_window(self):
        """ Center window and set fixed size. """
        self.setFixedSize(550, 480)
        screen_geometry = self.screen().availableGeometry()
        self.move(
            int((screen_geometry.width() - self.width()) / 2),
            int((screen_geometry.height() - self.height()) / 2)
        )

    def on_profile_selected(self, profile_name):
        """ Update the UI when a profile is selected from the dropdown. """
        if profile_name in self.profiles:
            settings = self.profiles[profile_name]
            self.logo_path_entry.setText(settings.get('path', ''))
            self.size_slider.setValue(settings.get('size', 5))
            self.margin_slider.setValue(settings.get('margin', 2))
            self.opacity_slider.setValue(settings.get('opacity', 100))
            
            position = settings.get('position', 'Bottom_Center')
            if position in self.position_radios:
                self.position_radios[position].setChecked(True)

            self.siril.log(f"Load profile settings '{profile_name}'", s.LogColor.GREEN)

    def select_logo_file(self):
        """ Opens a dialog box to select the logo file. """
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select PNG file",
            "", # Start directory
            "PNG files (*.png);;All files (*.*)"
        )
        if filepath:
            self.logo_path_entry.setText(filepath)

    def save_current_profile(self):
        """ Save the current settings as a new profile or update an existing one. """
        # QInputDialog, which returns text and a boolean
        profile_name, ok = QInputDialog.getText(self, "Save Profile", "Enter a name for this profile:")
        
        if not ok or not profile_name:
            return

        current_settings = {
            'path': self.logo_path_entry.text(),
            'size': self.size_slider.value(),
            'margin': self.margin_slider.value(),
            'position': self.position_button_group.checkedButton().text().replace(' ', '_'),
            'opacity': self.opacity_slider.value(),
        }
        
        # Validation
        if not os.path.exists(current_settings['path']):
            QMessageBox.critical(self, "Error", "The logo file path is invalid.")
            return

        self.profiles[profile_name] = current_settings
        self.save_config()
        
        # Update the Combobox by clearing and re-adding all profiles sorted
        self.profile_combo.blockSignals(True)  # Block signals to prevent unwanted triggers
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(self.profiles.keys()))
        self.profile_combo.blockSignals(False) # Re-enable signals
        
        # Set the newly created profile as the current one
        self.profile_combo.setCurrentText(profile_name)
        self.siril.log(f"Profile '{profile_name}' saved successfully.", s.LogColor.GREEN)

    def update_current_profile(self):
        """ Updates the currently selected profile with the current settings. """
        profile_name = self.profile_combo.currentText()
        if not profile_name:
            QMessageBox.warning(self, "No Selection", "Please select a profile to update.")
            return

        current_settings = {
            'path': self.logo_path_entry.text(),
            'size': self.size_slider.value(),
            'margin': self.margin_slider.value(),
            'position': self.position_button_group.checkedButton().text().replace(' ', '_'),
            'opacity': self.opacity_slider.value(),
        }

        self.profiles[profile_name] = current_settings
        self.save_config()
        self.siril.log(f"Profile '{profile_name}' updated successfully.", s.LogColor.GREEN)

    def delete_current_profile(self):
        """ Delete the currently selected profile. """
        profile_name = self.profile_combo.currentText()
        if not profile_name:
            QMessageBox.warning(self, "No Selection", "Please select a profile to delete.")
            return

        reply = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete the profile\n\n'{profile_name}'?")
        
        if reply == QMessageBox.StandardButton.Yes:
            # Remove from dictionary and configuration
            del self.profiles[profile_name]
            self.config.remove_section(profile_name)

            self.save_config() # Save changes to the file
            
            # Update the interface
            current_index = self.profile_combo.findText(profile_name)
            self.profile_combo.removeItem(current_index)
            
            # Clear the fields
            self.logo_path_entry.clear()
            self.size_slider.setValue(5)
            self.margin_slider.setValue(2)
            self.opacity_slider.setValue(100)
            self.position_radios['Bottom_Center'].setChecked(True)

            self.siril.log(f"Profile '{profile_name}' deleted.", s.LogColor.GREEN)

    def on_apply(self):
        """ Main function that applies the logo to the image. """
        try:
            # Load logo with OPENCV
            logo_path = self.logo_path_entry.text()
            if not os.path.exists(logo_path):
                raise FileNotFoundError(f"Logo file not found at path: {logo_path}")
            
            # Upload the image including the alpha channel (transparency)
            logo_rgba = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_rgba is None:
                raise IOError(f"Unable to read logo file. It may be corrupt.")

            # Apply changes
            with self.siril.image_lock():
                self.siril.log("Starting signature application process...", s.LogColor.GREEN) 
                self.siril.undo_save_state("Apply Signature")

                # Get image data from Siril
                background_data = self.siril.get_image_pixeldata(preview=False) 
                if background_data is None:
                    raise SirilError("Failed to get image data from Siril.")

                # Converts from CHW (Siril) to HWC (OpenCV)
                background_hwc = background_data.transpose(1, 2, 0)
                
                # Flip vertically to align with Siril's coordinate system
                background_hwc = cv2.flip(background_hwc, 0)

                # Converts from RGB to BGR, the standard OpenCV color format.
                background_bgr = cv2.cvtColor(background_hwc, cv2.COLOR_RGB2BGR)
                
                # --- ADAPTIVE SCALING LOGIC ---
                # ANALYZES THE BACKGROUND AND SCALE THE LOGO ACCORDINGLY
                
                # Find the maximum real value of the background to define the white point
                max_bg_value = background_hwc.max()
                self.siril.log(f"Maximum background brightness: {max_bg_value}. Scaling the logo to match.", s.LogColor.BLUE)
                
                # Calculate the scale factor to bring the logo (0-255) to the background range
                scaling_factor = max_bg_value / 255.0

                # Resize the logo geometry
                doc_height, doc_width, _ = background_bgr.shape

                # Resize the logo
                size_percent = self.size_slider.value()
                target_logo_h = int(doc_height * (size_percent / 100.0))
                logo_h, logo_w, _ = logo_rgba.shape

                aspect_ratio = logo_w / logo_h
                target_logo_w = int(target_logo_h * aspect_ratio)
                
                resized_logo_rgba = cv2.resize(logo_rgba, (target_logo_w, target_logo_h), interpolation=cv2.INTER_AREA)

                # Separate logo channels and handle images with or without an alpha channel
                if resized_logo_rgba.shape[2] == 4:
                    # The image has an alpha channel, we extract it
                    self.siril.log("Logo has an alpha channel (transparency).", s.LogColor.BLUE)
                    logo_rgb_8bit = resized_logo_rgba[:, :, :3]
                    alpha_mask_8bit = resized_logo_rgba[:, :, 3]
                elif resized_logo_rgba.shape[2] == 3:
                    # The image does not have an alpha channel, we create a fully opaque mask
                    self.siril.log("Logo does not have an alpha channel. Treating as fully opaque.", s.LogColor.BLUE)
                    logo_rgb_8bit = resized_logo_rgba
                    # Create a mask of all 255s (fully opaque)
                    alpha_mask_8bit = np.full((resized_logo_rgba.shape[0], resized_logo_rgba.shape[1]), 255, dtype=np.uint8)
                else:
                    # Manages unexpected cases, for example grayscale images
                    raise ValueError(f"Unsupported number of channels in logo image: {resized_logo_rgba.shape[2]}")

                # Apply the scale factor to the brightness of the logo
                logo_rgb_scaled = logo_rgb_8bit.astype(np.float32) * scaling_factor

                # Blending with consistent data
                resized_h, resized_w, _ = resized_logo_rgba.shape
                margin_percent = self.margin_slider.value()
                margin_px = int(doc_height * (margin_percent / 100.0))
                position = self.position_button_group.checkedButton().text().replace(' ', '_')
                
                x, y = 0, 0
                if 'Left' in position:
                    x = margin_px
                elif 'Center' in position:
                    x = (doc_width - resized_w) // 2
                elif 'Right' in position:
                    x = doc_width - resized_w - margin_px
                
                if 'Top' in position:
                    y = margin_px
                elif 'Middle' in position:
                    y = (doc_height - resized_h) // 2
                elif 'Bottom' in position:
                    y = doc_height - resized_h - margin_px
                
                # Define the region of interest (ROI) on the background
                # Make sure the logo doesn't go off the edges
                roi_start_x = max(0, x)
                roi_start_y = max(0, y)
                roi_end_x = min(x + resized_w, doc_width)
                roi_end_y = min(y + resized_h, doc_height)

                # Calculate the size of this intersection
                roi_h = roi_end_y - roi_start_y
                roi_w = roi_end_x - roi_start_x
                
                # Check for a valid ROI size before proceeding
                if roi_h <= 0 or roi_w <= 0:
                    self.siril.log("Logo is entirely outside the image boundaries. Skipping.", s.LogColor.RED)
                    return # Exit if there is nothing to blend

                # Get the ROI from the background
                roi = background_bgr[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

                # Calculate the corresponding part of the LOGO to crop
                logo_crop_start_x = max(0, -x)
                logo_crop_start_y = max(0, -y)

                # Crop the logo and alpha mask to match the ROI size exactly
                logo_to_blend = logo_rgb_scaled[logo_crop_start_y : logo_crop_start_y + roi_h, logo_crop_start_x : logo_crop_start_x + roi_w]
                alpha_to_blend = alpha_mask_8bit[logo_crop_start_y : logo_crop_start_y + roi_h, logo_crop_start_x : logo_crop_start_x + roi_w]
                
                # Read the opacity factor from GUI
                opacity_factor = self.opacity_slider.value() / 100.0
                # Convert the mask to float (0-1) and 3 channels for blending
                alpha_mask_float = ((alpha_to_blend * opacity_factor) / 255.0)[:, :, np.newaxis]
                
                # Blending formula (shapes are now guaranteed to be compatible)
                blended_roi = roi.astype(np.float32) * (1 - alpha_mask_float) + logo_to_blend * alpha_mask_float

                # Place the modified ROI back onto the background using the corrected coordinates
                background_bgr[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = np.clip(blended_roi, 0, 65535).astype(background_bgr.dtype)
                
                # Flip vertically 
                background_bgr = cv2.flip(background_bgr, 0)
                
                # Converts from BGR (OpenCV) to RGB
                final_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
                
                # Reconverts from HWC (OpenCV) to CHW (Siril)
                final_chw = final_rgb.transpose(2, 0, 1)

                # Send new image data to Siril
                self.siril.set_image_pixeldata(final_chw)
                self.siril.log("Signature applied successfully!", s.LogColor.GREEN)
                
        except Exception as e:
            self.siril.log(f"Error applying signature: {e}", s.LogColor.RED)
            # QMessageBox.critical(self, "Error", f"An error occurred while applying the signature:\n{e}")

    def closeEvent(self, event):
        """
        This event handler is called automatically by PyQt when the user closes the window.
        It handles the cleanup and disconnection from Siril.
        """
        try:
            if self.siril:
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.BLUE)
                self.siril.disconnect()
        except Exception:
            pass
        event.accept()

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
def main():
    try:
        qapp = QApplication(sys.argv)
        qapp.setApplicationName(f"Signature Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        icon_data = base64.b64decode("""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""")
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)
        app_icon = QIcon(pixmap)
        qapp.setWindowIcon(app_icon)

        qapp.setStyle("Fusion")

        # Define a Qt Style Sheet (QSS)
        stylesheet = """
            QPushButton[class="accent"] {
                background-color: #3574F0;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton[class="accent"]:hover {
                background-color: #4E8AFC;
            }
        """
        # Apply the stylesheet to the entire application
        qapp.setStyleSheet(stylesheet)

        app = SignatureApp()
        app.show()
        
        sys.exit(qapp.exec())

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()