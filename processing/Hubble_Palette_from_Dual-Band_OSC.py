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
# This Python script allows you to create different "Hubble-like" palettes from your OSC (One-Shot Color) images
# acquired with a dual-band Ha/OIII filter, all through a convenient graphical interface in Siril.
#
# Features:
# 1. User-friendly GUI with radio buttons for palette selection.
# 2. In-GUI instructions for image preparation.
# 3. Automatic generation of Ha, OIII, and synthetic S-II channels.
# 4. Supports various Hubble-like palette combinations (HSO, SHO, OSH, OHS, HOS, HOO).
# 5. Integrates robust Siril image handling (undo/redo, image locking).
# 6. Reuses intermediate channels for rapid testing of different palettes.
#
# Version History
# 1.0.0 - Initial release
# 1.0.1 - Added state management to reuse channels and a Reset button for faster iterations.
# 1.0.2 - Reset button now reloads the original source image.
# 1.1.0 - Added "Custom" palette with editable PixelMath formulas and config file.
# 1.1.1 - Minor fix: adjusted GUI layout
# 1.1.2 - Minor fix: Center window on open, updated instructions
# 1.1.3 - Minor fix:
#         File extension management
#         Updated instructions and UI to clarify image preparation steps
# 1.1.4 - Improved handling of custom palettes
# 1.1.5 - Minor fix:
#         Handle custom temporary file names.
#         Fix handling of -out= option in Siril rgbcomp command to support space in file names
# 1.1.6 - Minor fix:
#         Fixed handling of custom formulas in the GUI
# 1.1.7 - Added more pixelmath formulas for S2 and OIII
#         Minor fix 
# 1.1.8 - Added contact information
# 1.1.9 - Minor fix
# 2.0.0 - Complete porting to PyQt6
# 2.0.1 - CMD requires
# 2.0.2 - Added Icon App
# 2.0.3 - Minor fix:
#         Fixed file extension handling to support various FITS formats (e.g., .fit, .fit.fz).
#         Enhanced logging for the final image composition to improve user feedback.
#

VERSION = "2.0.3"
CONFIG_FILENAME = "Hubble-Palette-from-Dual-Band-OSC.conf"

# Core module imports
import os
import glob
import sys
import base64
import traceback
import configparser # For config file management

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s

    if not s.check_module_version('>=0.6.37'):
        print("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    # Import Siril GUI related components
    from sirilpy import SirilError

    s.ensure_installed("PyQt6", "astropy")

    # --- PyQt6 Imports ---
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
        QPushButton, QMessageBox, QGroupBox, QComboBox, QLineEdit, QStyle
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

def copy_fits_file(source_path, destination_path):
    """
    Copies a FITS file from source_path to destination_path using astropy.
    """
    try:
        # Open the source FITS file
        with fits.open(source_path) as hdul:
            # Create a new HDUList and save it to the destination
            hdul.writeto(destination_path, overwrite=True)
        return True
    except Exception as e:
        print(f"Error copying FITS file from {source_path} to {destination_path}: {e}")
        return False

def delete_file_if_exists(path, log_func=None):
    """
    Deletes a file if it exists, handling cases where the path might not include an extension.
    It will attempt to find and delete all files matching the base name, regardless of extension.
    """
    has_explicit_extension = '.' in os.path.basename(path) and os.path.basename(path).split('.')[-1] != ''

    if has_explicit_extension:
        # If the path already includes an extension, try to delete that specific file.
        files_to_check = [path]
    else:
        # If no explicit extension, use glob to find all files matching the base name.
        files_to_check = glob.glob(f"{path}.*")
        
    if not files_to_check:
        # If glob found nothing or the specific path didn't exist
        if log_func:
            log_func(f"File(s) not found for removal: {path}", s.LogColor.RED)
        return

    for file_to_delete in files_to_check:
        try:
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                if log_func:
                    log_func(f"Temporary file removed: {file_to_delete}", s.LogColor.GREEN)
            else:
                # This case might be hit if glob found a path but it disappeared between glob and os.path.exists
                if log_func:
                    log_func(f"File disappeared before removal: {file_to_delete}", s.LogColor.RED)
        except Exception as e:
            if log_func:
                log_func(f"Error removing {file_to_delete}: {e}", s.LogColor.RED)

class HubblePaletteApp(QWidget):
    """
    Main class that handles the GUI and Siril script execution.
    """
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(f"'Hubble-like' palettes from your OSC v{VERSION} - (c) Carlo Mollicone AstroBOH")

        # --- State Variables ---
        self.channels_generated = False
        self.source_image_name = None
        self.base_file_name = None
        self.temp_file_name = "Temporary_Image"  # Default temp file name
	
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
        
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image is loaded")
            QTimer.singleShot(0, self.close)
            return
        
        shape_image = self.siril.get_image_shape()
        if shape_image[0] != 3:
            self.siril.error_messagebox("The image must be a RGB image.")
            QTimer.singleShot(0, self.close)
            return

        # Internal palette names/IDs (fixed in code)
        self.PALETTE_ID_HSO = "hso_id"
        self.PALETTE_ID_SHO = "sho_id"
        self.PALETTE_ID_OSH = "osh_id"
        self.PALETTE_ID_OHS = "ohs_id"
        self.PALETTE_ID_HOS = "hos_id"
        self.PALETTE_ID_HOO = "hoo_id"

        # Dictionary that maps internal IDs to display names
        # These names can be changed without altering the internal logic
        self.display_names = {
            self.PALETTE_ID_HSO: "HSO",
            self.PALETTE_ID_SHO: "SHO",
            self.PALETTE_ID_OSH: "OSH",
            self.PALETTE_ID_OHS: "OHS",
            self.PALETTE_ID_HOS: "HOS",
            self.PALETTE_ID_HOO: "HOO"
        }

        # I bind tooltips to stable IDs. This way, the tooltip text is tied to the internal logic, not the display name.
        self.palette_tooltips = {
            self.PALETTE_ID_HSO: "HSO Palette (Ha->Red, Sii->Green, Oiii->Blue)",
            self.PALETTE_ID_SHO: "Standard 'Hubble' SHO Palette (Sii->Red, Ha->Green, Oiii->Blue)",
            self.PALETTE_ID_OSH: "OSH Palette (Oiii->Red, Sii->Green, Ha->Blue)",
            self.PALETTE_ID_OHS: "OHS Palette (Oiii->Red, Ha->Green, Sii->Blue)",
            self.PALETTE_ID_HOS: "HOS Palette (Ha->Red, Oiii->Green, Sii->Blue)",
            self.PALETTE_ID_HOO: "HOO Palette (Ha->Red, Oiii->Green, Oiii->Blue)"
        }

        # Dictionary for preset formulas
        self.preset_formulas = {
            'Classic': {
                'HA': 'R',
                'OIII': '(G + B) * 0.5',
                'S2': '(HA + OIII) * 0.5'
            },
            'Improved': {
                'HA': 'R',
                'OIII': '(G + B - 0.3 * R) * 0.5',
                'S2': 'R * 0.3'
            },
            'Advanced': {
                'HA': 'R',
                'OIII': 'max((G + B - 0.3 * R) * 0.5, 0)',
                'S2': '(R * 0.4 + ((G + B) * 0.1)) * 0.6'
            },
            'NonLinear S2': {
                'HA': 'R',
                'OIII': '(G + B) * 0.5',
                'S2': 'pow(R, 1.4) * 0.3'
            }
        }

        # Descriptions for preset formulas
        self.preset_descriptions = {
            'Classic': "Simple and direct\nSimulates a basic SHO palette with a synthetic SII created from the average of Ha and OIII.",
            'Improved': "With OIII channel cleaning\nOIII is filtered from Ha contamination. The SII channel is synthesized as an attenuated version of Ha.",
            'Advanced': "With dynamic compression and weighted mix\nOIII is 'denoised', and SII is a pseudo-spectral mix calibrated to improve contrast between Ha/OIII regions.",
            'NonLinear S2': "SII Curve (with pow)\nSII is simulated with a non-linear expression that emphasizes brighter areas of Ha."
        }
                
        # Load custom formulas from config file
        self.custom_formulas = self.load_config_file()

        self.create_widgets()
        self.center_window()
        self.update_ui_state() # Set initial UI state

    def create_widgets(self):
        """ Create all GUI widgets for the PyQt6 interface. """
        # --- Main Layout (3 columns) ---
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 0, 10, 10)

        # --- Left Column (Instructions) ---
        left_column_widget = QWidget()
        left_layout = QVBoxLayout(left_column_widget)
        
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_text = (
            "<p><b>Before running the script:</b><br>"
            "Your OSC image must be RGB and acquired using narrowband filters.</p>"
            "<p>It is recommended to follow the following workflow:</p>"
            "<ul style='margin-left: -25px;'>"
            "<li>Stacking, cropping, gradient removal,</li>"
            "<li>Other processes such as plate solve and color calibration</li>"
            "<li>Deconvolution, denoising.</li>"
            "</ul>"
            "<p>And finally, very importantly, the image must be <b>star-free</b>.</p>"
            "<p><b>Standard mapping:</b><br>"
            "H = from the Red channel<br>"
            "S = (Ha + OIII) / 2 &nbsp;&nbsp; [synthetic Green]<br>"
            "O = (G + B) / 2</p>"
            "<p><b>Recognized variables:</b><br>"
            "R, G, B - derived from the RGB split of the OSC image.<br>"
            "HA, OIII - intermediate results, usable in the SII (green) formula.</p>"
        )
        instructions_label = QLabel(instructions_text)
        instructions_label.setWordWrap(True)
        instructions_layout.addWidget(instructions_label)
        left_layout.addWidget(instructions_group)
        # left_layout.addStretch(1) # Pushes content up

        # --- Middle Column (Formulas) ---
        middle_column_widget = QWidget()
        middle_layout = QVBoxLayout(middle_column_widget)
        
        step1_group = QGroupBox("First Step")
        step1_layout = QVBoxLayout(step1_group)
        instructions_step1_text = (
            "<p>This is the first step of the script:</p>"
            "<ul style='margin-left: -25px;'>"
            "<li>The image is split into RGB</li>"
            "<li>Then the following formulas are applied to create: <b>Ha, Oiii, and S2</b></li>"
            "</ul>"
        )
        step1_label = QLabel(instructions_step1_text)
        step1_label.setWordWrap(True)
        step1_layout.addWidget(step1_label)
        middle_layout.addWidget(step1_group)

        middle_layout.addSpacing(10)

        middle_layout.addWidget(QLabel("<b>PixelMath Formula:</b>"))
        middle_layout.addWidget(QLabel("Formula Presets:"))
        self.preset_combobox = QComboBox()
        self.preset_combobox.addItems(self.preset_formulas.keys())
        self.preset_combobox.currentTextChanged.connect(self.on_preset_selected)
        middle_layout.addWidget(self.preset_combobox)
        
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        middle_layout.addWidget(self.description_label)

        middle_layout.addSpacing(10)

        self.ha_formula_entry = QLineEdit()
        middle_layout.addWidget(QLabel("RED : (it will be H)"))
        middle_layout.addWidget(self.ha_formula_entry)

        self.s2_formula_entry = QLineEdit()
        middle_layout.addWidget(QLabel("GREEN : (it will be S)"))
        middle_layout.addWidget(self.s2_formula_entry)

        self.oiii_formula_entry = QLineEdit()
        middle_layout.addWidget(QLabel("BLUE : (it will be O)"))
        middle_layout.addWidget(self.oiii_formula_entry)

        middle_layout.addSpacing(10)

        custom_buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Custom Formulas")
        self.load_button.clicked.connect(self.load_custom_formulas_from_file)
        self.save_button = QPushButton("Save Custom Formulas")
        self.save_button.clicked.connect(self.save_config_file)
        custom_buttons_layout.addWidget(self.load_button)
        custom_buttons_layout.addWidget(self.save_button)
        middle_layout.addLayout(custom_buttons_layout)
        # middle_layout.addStretch(1)

        # --- Right Column (Palettes) ---
        right_column_widget = QWidget()
        right_layout = QVBoxLayout(right_column_widget)

        step2_group = QGroupBox("Final Step")
        step2_layout = QVBoxLayout(step2_group)
        instructions_step2_text = (
            "This is the final step:<br><br>"
            "H - O - S files will be combined in the following ways"
        )
        step2_label = QLabel(instructions_step2_text)
        step2_label.setWordWrap(True)
        step2_layout.addWidget(step2_label)
        right_layout.addWidget(step2_group)

        right_layout.addSpacing(20)

        right_layout.addWidget(QLabel("<b>Select the Hubble Palette type:</b>"))

        # Store radio buttons in a dict to retrieve selection easily
        self.radio_buttons = {}
        for palette_id, display_name in self.display_names.items():
            radio_button = QRadioButton(display_name)
            radio_button.setToolTip(self.palette_tooltips.get(palette_id, "No description"))
            right_layout.addWidget(radio_button)
            self.radio_buttons[palette_id] = radio_button
        
        self.radio_buttons[self.PALETTE_ID_HSO].setChecked(True) # Default

        right_layout.addSpacing(20)

        buttons_layout = QHBoxLayout()
        apply_button = QPushButton("   Apply   ")
        apply_button.clicked.connect(self.on_apply)
        apply_button.setProperty("class", "accent")
        apply_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))

        reset_button = QPushButton("   Reset   ")
        reset_button.clicked.connect(self.reset_process)
        reset_button.setToolTip("Delete all files, even the combinations already produced and reload the original image")
        reset_button.setProperty("class", "reset")
        reset_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogDiscardButton))
        
        buttons_layout.addWidget(apply_button)
        buttons_layout.addWidget(reset_button)
        right_layout.addLayout(buttons_layout)
        right_layout.addStretch(1)
        
        # Add columns to main layout
        main_layout.addWidget(left_column_widget, 2)
        main_layout.addWidget(middle_column_widget, 3)
        main_layout.addWidget(right_column_widget, 2)
        
        # Set initial preset
        self.preset_combobox.setCurrentText('Classic')
        self.on_preset_selected('Classic')

    def center_window(self):
        """ Center window using PyQt methods """
        self.resize(850, 520)
        screen_geometry = self.screen().availableGeometry()
        self.move(
            int((screen_geometry.width() - self.width()) / 2),
            int((screen_geometry.height() - self.height()) / 2)
        )
        # self.setFixedSize(self.size()) # Make window not resizable

    def update_ui_state(self):
        """ Enable/disable and populate formula fields. """
        self.on_preset_selected(self.preset_combobox.currentText())

    def on_preset_selected(self, preset_name):
        """ Updates formula text fields based on combobox selection. """
        custom_item_text = "Custom Loaded"

        # If a standard preset is chosen, remove the temporary "Custom" item
        # to keep the list clean.
        if preset_name and preset_name != custom_item_text:
            index = self.preset_combobox.findText(custom_item_text)
            if index != -1: # If the custom item exists
                # Temporarily block signals to prevent this function from firing again
                # when the item is removed and the index changes.
                self.preset_combobox.blockSignals(True)
                self.preset_combobox.removeItem(index)
                self.preset_combobox.blockSignals(False)

        formulas = self.preset_formulas.get(preset_name)
        description = self.preset_descriptions.get(preset_name, "")
        
        if formulas:
            self.ha_formula_entry.setText(formulas['HA'])
            self.s2_formula_entry.setText(formulas['S2'])
            self.oiii_formula_entry.setText(formulas['OIII'])
            self.description_label.setText(description)

    def load_custom_formulas_from_file(self):
        """ Loads formulas from the config file and updates the text fields. """
        self.custom_formulas = self.load_config_file()
        self.ha_formula_entry.setText(self.custom_formulas['HA'])
        self.s2_formula_entry.setText(self.custom_formulas['S2'])
        self.oiii_formula_entry.setText(self.custom_formulas['OIII'])
        
        # Add "Custom Loaded" to the list if it doesn't already exist
        custom_item_text = "Custom Loaded"
        if self.preset_combobox.findText(custom_item_text) == -1: # -1 means not found
            self.preset_combobox.addItem(custom_item_text)

        # Now that the item exists in the list, this command will work correctly
        self.preset_combobox.setCurrentText(custom_item_text)

        self.description_label.setText("Custom formulas loaded from your saved configuration file.")
        self.siril.log("Loaded custom formulas from file.", s.LogColor.BLUE)

    def load_config_file(self):
        """ Method to load custom formulas from a config file """
        config = configparser.ConfigParser()
        
        # Use the 'Classic' preset as the default fallback value.
        default_fallback = self.preset_formulas['Classic'].copy()
        formulas = default_fallback.copy()
        
        try:
            config_dir = self.siril.get_siril_configdir()
            config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
            
            if os.path.exists(config_file_path):
                config.read(config_file_path)
                if 'CustomFormulas' in config:
                    # When loading from file, use the 'Classic' values as a fallback if a specific key is missing.
                    formulas['HA'] = config['CustomFormulas'].get('HA_Formula', default_fallback['HA'])
                    formulas['S2'] = config['CustomFormulas'].get('S2_PROXY_Formula', default_fallback['S2'])
                    formulas['OIII'] = config['CustomFormulas'].get('OIII_Formula', default_fallback['OIII'])
        except Exception as e:
            self.siril.log(f"Error loading config file: {str(e)}", s.LogColor.RED)
        
        return formulas

    def save_config_file(self):
        """ Method to save custom formulas to a config file """
        config = configparser.ConfigParser()
        config['CustomFormulas'] = {
            'HA_Formula': self.ha_formula_entry.text(),
            'S2_PROXY_Formula': self.s2_formula_entry.text(),
            'OIII_Formula': self.oiii_formula_entry.text()
        }
        
        try:
            config_dir = self.siril.get_siril_configdir()
            config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
            with open(config_file_path, 'w') as configfile:
                config.write(configfile)
            
            self.custom_formulas['HA'] = self.ha_formula_entry.text()
            self.custom_formulas['S2'] = self.s2_formula_entry.text()
            self.custom_formulas['OIII'] = self.oiii_formula_entry.text()

            self.siril.log("Success: Custom formulas saved successfully.", s.LogColor.BLUE)
            # QMessageBox.information(self, "Success", "Custom formulas saved successfully.")
        except Exception as e:
            self.siril.log(f"Error saving config file: {str(e)}", s.LogColor.RED)
            # QMessageBox.critical(self, "Error", f"Error saving config file: {str(e)}")

    def on_apply(self):
        """
        Identifies which palette radio button is selected, gets the current
        formula preset name, and then triggers the main processing logic.
        """
        chosen_palette_id = ""
        for pid, radio_button in self.radio_buttons.items():
            if radio_button.isChecked():
                chosen_palette_id = pid
                break
        
        chosen_display_name = self.display_names.get(chosen_palette_id, "UnknownID")
        formula_name = self.preset_combobox.currentText()

        self.run_hubble_palette_logic(chosen_palette_id, chosen_display_name, formula_name)

    def reset_process(self):
        """
        Resets the script to its initial state. This method handles the 'Reset'
        button click. It performs three main actions:
        1. Deletes all temporary channel files (e.g., HA.fit, S2_PROXY.fit).
        2. Attempts to reload the original source image into Siril.
        3. Resets internal state variables to allow for a fresh start.
        """
        if not self.siril: return
        self.siril.log("Resetting process. Cleaning up intermediate files.", s.LogColor.BLUE)
        self.cleanup_temp_files()
        
        if self.source_image_name and os.path.exists(self.source_image_name):
            try:
                self.siril.log(f"Reloading original image: {self.source_image_name}", s.LogColor.BLUE)
                self.siril.cmd("load", "\"" + self.source_image_name + "\"")
            except SirilError as e:
                self.siril.log(f"Failed to reload original image: {e}", s.LogColor.RED)
                QMessageBox.warning(self, "Reset", "Process reset, but failed to reload the original image.")
        else:
             self.siril.log("Reset: Process has been reset. Load an image to begin.", s.LogColor.BLUE)

        # Reset state variables
        self.channels_generated = False
        self.source_image_name = None
        self.base_file_name = None
        self.siril.log("Reset complete. You can now start with a new image.", s.LogColor.GREEN)

    def cleanup_temp_files(self):
        """Cleans up all generated intermediate and result files."""
        # Base names of files to be deleted, without extension
        base_names_to_delete = ["HA", "OIII", "S2_PROXY"]
        
        base_names_to_delete.extend([
            f"{self.temp_file_name}_r",
            f"{self.temp_file_name}_g",
            f"{self.temp_file_name}_b"
        ])
        
        log_func = self.siril.log if self.siril else print

        # Iterate over the base names and use glob to find all extensions
        for base_name in base_names_to_delete:
            # Use glob.glob to find all files starting with the base name,
            # followed by a dot and any extension.
            # For example, for "HA", it will search for "HA.fit", "HA.fts", "HA.tiff", etc.
            files_to_remove = glob.glob(f"{base_name}.*")
            for file_path in files_to_remove:
                delete_file_if_exists(file_path, log_func)

    def run_hubble_palette_logic(self, palette_id, palette_name, formula_name):
        """
        Main function that executes Siril commands to create the palette.
        """
        try:
            # Define output filenames for derived channels, no extensions here
            # Siril will append the user's preferred FITS extension automatically
            HA_OUT = "Ha"               # Red
            S2_PROXY_OUT = "S2_synt"    # Green synthetic
            OIII_OUT = "Oiii"           # Blue

            # Base names without extension (for split)
            R_CHAN = f"{self.temp_file_name}_r"     # Temporary_Image_r.fit or Temporary_Image_r.fit.fz
            G_CHAN = f"{self.temp_file_name}_g"     # Temporary_Image_g.fit or Temporary_Image_g.fit.fz
            B_CHAN = f"{self.temp_file_name}_b"     # Temporary_Image_b.fit or Temporary_Image_b.fit.fz
            
            # --- PHASE 1: Generate channels ---
            if not self.channels_generated:
                #self.siril.undo_save_state("Generate Intermediate Channels (Ha, OIII, S2)")
                self.siril.log("First run: generating intermediate channels...", s.LogColor.BLUE)
        
                # Get the thread and current image
                with self.siril.image_lock():
                    # Get current image and ensure data type
                    fit = self.siril.get_image()
                    fit.ensure_data_type(np.float32)

                # Get the filename of the active image
                current_image = self.siril.get_image_filename()
                if not current_image:
                    self.siril.log("Error: No active image found. Please open your image.", s.LogColor.RED)
                    return
                
                # Set the source image name only on the very first run
                if self.source_image_name is None:
                    self.source_image_name = current_image
                    self.siril.log(f"Source image set to: {self.source_image_name}", s.LogColor.GREEN)

                # Get base name of current image
                # Extract just the filename without path and extension
                base_name = os.path.basename(self.source_image_name)
                file_name, extension = os.path.splitext(base_name)
                # Save base name for later use
                self.base_file_name = file_name 

                # Use the updated 'split' command with output file names
                # Ex: split "myimage_r.fit" "myimage_g.fit" "myimage_b.fit"
                self.siril.log("Splitting RGB channels from source image...", s.LogColor.GREEN)
                self.siril.cmd("split", f'"{R_CHAN}"', f'"{G_CHAN}"', f'"{B_CHAN}"', f'-from="{self.source_image_name}"')

                self.channels_generated = True
                self.siril.log("R - G - B channels split successfully.", s.LogColor.GREEN)

            # Dynamic formula processing
            self.siril.log("Deriving channels using formulas from the GUI...", s.LogColor.GREEN)

            ha_formula = self.ha_formula_entry.text()
            s2_formula = self.s2_formula_entry.text()
            oiii_formula = self.oiii_formula_entry.text()
            
            # Create a dictionary to replace placeholders with actual filenames
            placeholders = {
                'R': f'${R_CHAN}$', 'G': f'${G_CHAN}$', 'B': f'${B_CHAN}$',
                'HA': f'${HA_OUT}$', 'OIII': f'${OIII_OUT}$'
            }

            def format_formula(formula, current_placeholders):
                for key, value in current_placeholders.items():
                    # Use a regex-like approach to replace whole words only
                    formula = formula.replace(key, value)
                return formula

            #pm_ha_formula = format_formula(ha_formula, {'R': f'${R_CHAN}$', 'G': f'${G_CHAN}$', 'B': f'${B_CHAN}$'})
            
            # Process Ha
            self.siril.log(f"Apply {HA_OUT} formula = {ha_formula}", s.LogColor.GREEN)
            pm_ha_formula = format_formula(ha_formula, placeholders)
            self.siril.cmd("pm", f'"{pm_ha_formula}"')
            self.siril.cmd("save", f'"{HA_OUT}"')

            # Process OIII
            self.siril.log(f"Apply {OIII_OUT} formula = {oiii_formula}", s.LogColor.GREEN)
            pm_oiii_formula = format_formula(oiii_formula, placeholders)
            self.siril.cmd("pm", f'"{pm_oiii_formula}"')
            self.siril.cmd("save", f'"{OIII_OUT}"')

            # Process S2
            self.siril.log(f"Apply {S2_PROXY_OUT} formula = {s2_formula}", s.LogColor.GREEN)
            pm_s2_formula = format_formula(s2_formula, placeholders)
            self.siril.cmd("pm", f'"{pm_s2_formula}"')
            self.siril.cmd("save", f'"{S2_PROXY_OUT}"')

            self.siril.log("Intermediate channels generated successfully.", s.LogColor.GREEN)
            # Clean up the split files now, as they are no longer needed
            # for path in [R_CHAN, G_CHAN, B_CHAN]:
            #     delete_file_if_exists(path, self.siril.log)

            # --- PHASE 2: Compose the selected palette ---
            self.siril.log(f"Composing the {palette_name} palette...", s.LogColor.GREEN)

            # don't need to include filename extensions here: Siril will add them automatically when saving, according to the user's preferred FITS extension
            output_filename = f"{self.base_file_name}_result_{palette_name}_{formula_name}"

            # Check if intermediate files exist, just in case
            # We use glob to find files with any extension (e.g., .fit, .fit.fz)
            # glob.glob(f + ".*") -->> Search for all files that start with "HA." followed by anything (e.g., HA.fit, HA.fit.fz).
            if not all(glob.glob(f + ".*") for f in [HA_OUT, OIII_OUT, S2_PROXY_OUT]):
                 self.siril.log("Error: Intermediate files (HA, OIII, S2_PROXY) not found. Please Reset.", s.LogColor.RED)
                 QMessageBox.critical(self, "Error", "Missing intermediate files. Please use the Reset button and try again.")
                 return

            # Simplified map, as Custom is handled by the dynamic formulas
            palette_map = {
                self.PALETTE_ID_HSO: f'rgbcomp "{HA_OUT}" "{S2_PROXY_OUT}" "{OIII_OUT}"',
                self.PALETTE_ID_SHO: f'rgbcomp "{S2_PROXY_OUT}" "{HA_OUT}" "{OIII_OUT}"',
                self.PALETTE_ID_OSH: f'rgbcomp "{OIII_OUT}" "{S2_PROXY_OUT}" "{HA_OUT}"',
                self.PALETTE_ID_OHS: f'rgbcomp "{OIII_OUT}" "{HA_OUT}" "{S2_PROXY_OUT}"',
                self.PALETTE_ID_HOS: f'rgbcomp "{HA_OUT}" "{OIII_OUT}" "{S2_PROXY_OUT}"',
                self.PALETTE_ID_HOO: f'rgbcomp "{HA_OUT}" "{OIII_OUT}" "{OIII_OUT}"'
            }
            
            command = palette_map.get(palette_id)
            if not command:
                self.siril.log(f"Error: Invalid palette ID specified: {palette_id}", s.LogColor.RED)
                return

            # Creating a clearer final log
            # The command is 'rgbcomp "CHANNEL_R" "CHANNEL_G" "CHANNEL_B"'
            # Dividing by the quotes, the names are in positions 1, 3, and 5
            parts = command.split('"')
            r_channel = parts[1]
            g_channel = parts[3]
            b_channel = parts[5]
            
            self.siril.log("-------------------------------------------------", s.LogColor.BLUE)
            self.siril.log(f"Composing '{palette_name}' palette with mapping:", s.LogColor.BLUE)
            self.siril.log(f"  > Red:   {r_channel}", s.LogColor.GREEN)
            self.siril.log(f"  > Green: {g_channel}", s.LogColor.GREEN)
            self.siril.log(f"  > Blue:  {b_channel}", s.LogColor.GREEN)
            self.siril.log("-------------------------------------------------", s.LogColor.BLUE)

            self.siril.cmd(command, f'"-out={output_filename}"', "-nosum")

            self.siril.cmd("load", "\"" + output_filename + "\"")
            self.siril.log(f"Successfully generated and loaded the {palette_name} palette!", s.LogColor.GREEN)

        except SirilError as e:
            self.siril.log(f"Siril error during execution: {e}", s.LogColor.RED)
            QMessageBox.critical(self, "Siril Error", str(e))
        except Exception as e:
            self.siril.log(f"An unexpected error occurred: {e}", s.LogColor.RED)
            QMessageBox.critical(self, "Generic Error", str(e))

    def closeEvent(self, event):
        """
        This event handler is called automatically by PyQt when the user closes the window.
        It handles the cleanup and disconnection from Siril.
        """
        try:
            if self.siril:
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.BLUE)
                self.cleanup_temp_files()
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
        qapp.setApplicationName(f"'Hubble-like' palettes from your OSC v{VERSION} - (c) Carlo Mollicone AstroBOH")

        icon_data = base64.b64decode("""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""")
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

            QPushButton[class="reset"] {
                background-color: #D32F2F; /* Un bel rosso scuro */
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton[class="reset"]:hover {
                background-color: #E57373; /* Un rosso più chiaro per l'hover */
            }
        """
        # Apply the stylesheet to the entire application
        qapp.setStyleSheet(stylesheet)
        
        app = HubblePaletteApp()
        app.show()
        
        sys.exit(qapp.exec())
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()