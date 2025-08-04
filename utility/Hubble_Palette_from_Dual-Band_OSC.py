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
# 1.0.0 Initial release
# 1.0.1 Added state management to reuse channels and a Reset button for faster iterations.
# 1.0.2 Reset button now reloads the original source image.
# 1.1.0 Added "Custom" palette with editable PixelMath formulas and config file.
# 1.1.1 Minor fix: adjusted GUI layout
# 1.1.2 Minor fix: Center window on open, updated instructions
# 1.1.3 Minor fix:
#           File extension management
#           Updated instructions and UI to clarify image preparation steps
# 1.1.4 Improved handling of custom palettes
# 1.1.5 Minor fix:
#       Handle custom temporary file names.
#       Fix handling of -out= option in Siril rgbcomp command to support space in file names
# 1.1.6 Minor fix:
#       Fixed handling of custom formulas in the GUI
# 1.1.7 Added more pixelmath formulas for S2 and OIII
#       Minor fix 
# 1.1.8 Added contact information
#

VERSION = "1.1.8"
CONFIG_FILENAME = "Hubble-Palette-from-Dual-Band-OSC.conf"

# Core module imports
import os
import glob
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np # Import numpy for image data manipulation
import configparser # For config file management

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
# This allows the script to be run externally for testing (with limited functionality).
try:
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=0.6.37'):
        messagebox.showerror("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    SIRIL_ENV = True

    # Import Siril GUI related components
    from sirilpy import tksiril, SirilError

    # Ensure ttkthemes is installed for better looking GUI
    s.ensure_installed("ttkthemes", "astropy")

    from astropy.io import fits
    from ttkthemes import ThemedTk
except ImportError:
    SIRIL_ENV = False
    messagebox.showerror("Warning: sirilpy not found. The script is not running in the Siril environment.")

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

class HubblePaletteApp:
    """
    Main class that handles the GUI and Siril script execution.
    """
    def __init__(self, root):
        self.root = root
        self.root.title(f"'Hubble-like' palettes from your OSC v{VERSION} - (c) Carlo Mollicone AstroBOH")

        #setting window size And Center the window
        width=850
        height=520
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)

        self.root.resizable(False, False)
        self.style = tksiril.standard_style()

        # State management
        self.channels_generated = False
        self.source_image_name = None
        self.base_file_name = None
        self.temp_file_name = "Temporary_Image"  # Default temp file name
	
        # Initialize Siril connection
        self.siril = None # Initialize to None
        if SIRIL_ENV:
            self.siril = s.SirilInterface()
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                messagebox.showerror("Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
                self.close_dialog()
                return
        
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image is loaded")
            self.close_dialog()
            return
        
        shape_image = self.siril.get_image_shape()
        if shape_image[0] != 3:
            self.siril.error_messagebox("The image must be a RGB image.")
            self.close_dialog()
            return

        tksiril.match_theme_to_siril(self.root, self.siril)

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
            self.PALETTE_ID_HOS: "HOS Palette (Ha->Red, Sii->Green, Sii->Blue)",
            self.PALETTE_ID_HOO: "HOO Palette (Ha->Red, Sii->Green, Oiii->Blue)"
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

        # --- Frame for layout (simulating columns) ---
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=0, pady=0, fill="both", expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", padx=5, pady=5, fill="both", expand=True)

        middle_frame = ttk.Frame(main_frame, width=300)
        middle_frame.pack(side="left", padx=5, pady=5, fill="y")
        middle_frame.pack_propagate(False)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)

        # Instructions frame
        Instructions_frame = ttk.LabelFrame(left_frame, text="Instructions", padding=10)
        Instructions_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        Instructions_frame.columnconfigure(1, weight=1) # use all space in col 1

        # --- Left Column (Instructions) ---
        instructions_text = (
            "Before running the script:\n"
            "Your OSC image must be RGB and acquired using narrowband filters.\n\n"
            "It is recommended to follow the following workflow:\n"
            "- Stacking, cropping, gradient removal,\n"
            "- Other processes such as plate solve and color calibration\n"
            "- Deconvolution, denoising.\n"
            "And finally, very importantly, the image must be star-free.\n\n"
            "Standard mapping:\n"
            "H = from the Red channel\n"
            "S = (Ha + OIII) / 2   [synthetic Green]\n"
            "O = (G + B) / 2\n\n"
            "Recognized variables:\n"
            "R, G, B - derived from the RGB split of the OSC image.\n"
            "HA, OIII - intermediate results, usable in the SII (green) formula."
        )
        instructions_label = ttk.Label(Instructions_frame, text=instructions_text, wraplength=220, justify="left", font=("TkDefaultFont", 9))
        instructions_label.pack(fill="both", expand=True, pady=(0, 10))

        # --- Middle Column ---
        # Instructions Frame_step 1
        Frame_step1 = ttk.LabelFrame(middle_frame, text="First Step", padding=10)
        Frame_step1.pack(fill=tk.BOTH, padx=5, pady=5)
        Frame_step1.columnconfigure(1, weight=1)

        instructions_step1 = (
            "This is the first step of the script:\n"
            "   - The image is split into RGB\n"
            "   - Then the following formulas are applied to create:\n"
            "       Ha, Oiii, and S2\n"
        )
        instructions_step1 = ttk.Label(Frame_step1, text=instructions_step1, wraplength=220, justify="left", font=("TkDefaultFont", 9))
        instructions_step1.pack(fill="both", expand=True, pady=(0, 10))

        label_PixelMath = ttk.Label(middle_frame, text="PixelMath Formula:")
        label_PixelMath.pack(pady=5)
        
        ttk.Label(middle_frame, text="Formula Presets:").pack(anchor="w", pady=(0, 2))
        self.preset_combobox = ttk.Combobox(middle_frame, values=list(self.preset_formulas.keys()), width=33)
        self.preset_combobox.pack(anchor="w", fill="x")
        self.preset_combobox.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        self.description_label = ttk.Label(middle_frame, text="", justify="left")
        self.description_label.pack(fill="both", expand=False, pady=(5, 10))
        self.description_label.bind("<Configure>", self.on_label_resize)

        self.ha_formula_var = tk.StringVar()
        ttk.Label(middle_frame, text="RED : (it will be H)").pack(anchor="w")
        self.ha_formula_entry = ttk.Entry(middle_frame, textvariable=self.ha_formula_var, width=35)
        self.ha_formula_entry.pack(anchor="w", fill="x")

        self.s2_formula_var = tk.StringVar()
        ttk.Label(middle_frame, text="GREEN : (it will be S)").pack(anchor="w", pady=(10, 0))
        self.s2_formula_entry = ttk.Entry(middle_frame, textvariable=self.s2_formula_var, width=35)
        self.s2_formula_entry.pack(anchor="w", fill="x")

        self.oiii_formula_var = tk.StringVar()
        ttk.Label(middle_frame, text="BLUE : (it will be O)").pack(anchor="w", pady=(10, 0))
        self.oiii_formula_entry = ttk.Entry(middle_frame, textvariable=self.oiii_formula_var, width=35)
        self.oiii_formula_entry.pack(anchor="w", fill="x")

        custom_buttons_frame = ttk.Frame(middle_frame)
        custom_buttons_frame.pack(pady=20)
        
        self.load_button = ttk.Button(custom_buttons_frame, text="Load Custom Formulas", command=self.load_custom_formulas_from_file)
        self.load_button.pack(side="left", padx=5)

        self.save_button = ttk.Button(custom_buttons_frame, text="Save Custom Formulas", command=self.save_config_file)
        self.save_button.pack(side="left", padx=5)

        # --- Right Column ---
        right_content_frame = right_frame
        right_content_frame.pack(fill="both", expand=True)

        # Instructions Frame_step 2
        Frame_step2 = ttk.LabelFrame(right_content_frame, text="Final Step", padding=10)
        Frame_step2.pack(fill=tk.BOTH, padx=5, pady=5)
        Frame_step2.columnconfigure(1, weight=1)

        instructions_step2 = (
            "This is the final step:\n\n"
            "H - O - S files will be combined in the following ways\n"
        )
        instructions_step2 = ttk.Label(Frame_step2, text=instructions_step2, wraplength=220, justify="left", font=("TkDefaultFont", 9))
        instructions_step2.pack(fill="both", expand=True, pady=(0, 10))


        label_select = ttk.Label(right_content_frame, text="Select the Hubble Palette type:")
        label_select.pack(pady=5)

        # palette options Combination
        # La variabile di controllo ora memorizza l'ID ("hso_id", "sho_id", etc.), non il nome visualizzato.
        self.selected_palette_id = tk.StringVar(root)
        self.selected_palette_id.set(self.PALETTE_ID_HSO) # Imposta il default usando l'ID

        for palette_id, display_name in self.display_names.items():
            radio_button = ttk.Radiobutton(
                right_content_frame,
                text=display_name,                       # Testo per l'utente
                variable=self.selected_palette_id,       # Variabile di controllo (che contiene l'ID)
                value=palette_id,                        # Il valore di questo radio button è il suo ID univoco
                command=self.update_ui_state             # Comando da eseguire al click
            )
            radio_button.pack(anchor="w", padx=10, pady=2)

            # Associa il tooltip recuperandolo tramite l'ID
            tooltip_text = self.palette_tooltips.get(palette_id, "Nessuna descrizione")
            tksiril.create_tooltip(radio_button, tooltip_text)

        buttons_frame = ttk.Frame(right_content_frame)
        buttons_frame.pack(pady=10, anchor="w", padx=10)

        apply_button = ttk.Button(buttons_frame, text="Apply", command=self.on_apply)
        apply_button.pack(side="left", padx=5)

        reset_button = ttk.Button(buttons_frame, text="Reset", command=self.reset_process)
        reset_button.pack(side="left", padx=5)
        tksiril.create_tooltip(reset_button, "Delete all files, even the combinations already produced and reload the original image")
        
        # Imposta 'Classic' come preset visibile all'avvio
        self.preset_combobox.set('Classic')

        # Set initial UI state
        self.update_ui_state()
        
        # Handle window closing event
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # Method to update UI based on palette selection
    def update_ui_state(self):
        """Enable/disable and populate formula fields based on palette selection."""
        # # Get the ID directly from the control variable.
        # chosen_palette_id = self.selected_palette_id.get()
        # is_custom = (chosen_palette_id == self.PALETTE_ID_CUSTOM)

        # # Enable/disable the entire formula panel based on selection
        # state = "normal" if is_custom else "disabled"
        # #self.preset_combobox.config(state=state)
        # self.ha_formula_entry.config(state=state)
        # self.s2_formula_entry.config(state=state)
        # self.oiii_formula_entry.config(state=state)
        # self.load_button.config(state=state)
        # self.save_button.config(state=state)

        # # Populate text fields
        # if is_custom:
        #     # If custom mode is activated, load user's saved formulas
        #     self.load_custom_formulas_from_file()
        # else:
        #     # If a standard palette is chosen, display the formulas from the selected preset
        #     self.on_preset_selected()
        self.on_preset_selected()

    def on_label_resize(self, event):
        """ Dynamically adjusts the wraplength to the current label width. """
        # event.width contains the new width of the widget in pixels
        self.description_label.config(wraplength=event.width)

    # New method to handle preset selection
    def on_preset_selected(self, event=None):
        """Updates formula text fields based on combobox selection."""
        preset_name = self.preset_combobox.get()
        formulas = self.preset_formulas.get(preset_name)
        description = self.preset_descriptions.get(preset_name, "")
        
        if formulas:
            self.ha_formula_var.set(formulas['HA'])
            self.s2_formula_var.set(formulas['S2'])
            self.oiii_formula_var.set(formulas['OIII'])
            self.description_label.config(text=description)

    # New method for the "Load Custom" button
    def load_custom_formulas_from_file(self):
        """Loads formulas from the config file and updates the text fields."""
        self.custom_formulas = self.load_config_file()
        self.ha_formula_var.set(self.custom_formulas['HA'])
        self.s2_formula_var.set(self.custom_formulas['S2'])
        self.oiii_formula_var.set(self.custom_formulas['OIII'])
        
        # Visually indicate that custom formulas are loaded
        self.preset_combobox.set("Custom Loaded")
        self.description_label.config(text="Custom formulas loaded from your saved configuration file.")
        self.siril.log("Loaded custom formulas from file.", s.LogColor.BLUE)

    def load_config_file(self):
        """ Method to load custom formulas from a config file """
        config = configparser.ConfigParser()
        
        # Use the 'Classic' preset as the default fallback value.
        default_fallback = self.preset_formulas['Classic'].copy()
        formulas = default_fallback.copy()
        
        if SIRIL_ENV:
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
        if not SIRIL_ENV:
            messagebox.showerror("Error", "Cannot save config file outside of Siril environment.")
            return

        config = configparser.ConfigParser()
        config['CustomFormulas'] = {
            'HA_Formula': self.ha_formula_var.get(),
            'S2_PROXY_Formula': self.s2_formula_var.get(),
            'OIII_Formula': self.oiii_formula_var.get()
        }
        
        try:
            config_dir = self.siril.get_siril_configdir()
            config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
            with open(config_file_path, 'w') as configfile:
                config.write(configfile)
            
            # Update the stored custom formulas in memory
            self.custom_formulas['HA'] = self.ha_formula_var.get()
            self.custom_formulas['S2'] = self.s2_formula_var.get()
            self.custom_formulas['OIII'] = self.oiii_formula_var.get()

            self.siril.log("Success: Custom formulas saved successfully.", s.LogColor.BLUE)
            #messagebox.showinfo("Success", "Custom formulas saved successfully.")
        except Exception as e:
            self.siril.log(f"Error saving config file: {str(e)}", s.LogColor.RED)
            #messagebox.showerror("Error", f"Error saving config file: {str(e)}")

    def close_dialog(self):
        """
        Helper to safely close the Tkinter window.
        Close the dialog and disconnect from Siril
        """
        try:
            if SIRIL_ENV and self.siril and self.siril.is_connected():
                # self.siril.log("Window closed. Script cancelled by user.", s.LogColor.RED)
                self.siril.disconnect()
            self.root.quit()
        except Exception:
            pass
        self.root.destroy()

    def on_apply(self):
        """Called when the 'Apply' button is pressed."""
        # Get the ID directly from the control variable
        chosen_palette_id = self.selected_palette_id.get()
        chosen_display_name = self.display_names.get(chosen_palette_id, "ID Sconosciuto")

        formula_name = self.preset_combobox.get()

        self.run_hubble_palette_logic(chosen_palette_id, chosen_display_name, formula_name)

    def on_closing(self):
        """
        Handle dialog close - Called when the window is closed via the 'X' button.
        Close the dialog and disconnect from Siril
        """
        if SIRIL_ENV and self.siril:
            self.siril.log("Window closed. Cleaning up temporary files.", s.LogColor.BLUE)
        self.cleanup_temp_files()
        self.close_dialog()

    def reset_process(self):
        if not self.siril: return
        self.siril.log("Resetting process. Cleaning up intermediate files.", s.LogColor.BLUE)
        self.cleanup_temp_files()
        
        if self.source_image_name and os.path.exists(self.source_image_name):
            try:
                self.siril.log(f"Reloading original image: {self.source_image_name}", s.LogColor.BLUE)
                self.siril.cmd("load", "\"" + self.source_image_name + "\"")
            except SirilError as e:
                self.siril.log(f"Failed to reload original image: {e}", s.LogColor.RED)
                messagebox.showwarning("Reset", "Process reset, but failed to reload the original image.")
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

        log_func = self.siril.log if SIRIL_ENV and self.siril else print
        
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
        if not SIRIL_ENV or not self.siril:
            messagebox.showerror("Error", "Script is not running in Siril environment or connection failed.")
            return

        try:
            # Define output filenames for derived channels
            HA_OUT = "HA.fit"               # Red
            S2_PROXY_OUT = "S2_PROXY.fit"   # Green synthetic
            OIII_OUT = "OIII.fit"           # Blue

            # Base names without extension (for split)
            R_CHAN = f"{self.temp_file_name}_r"
            G_CHAN = f"{self.temp_file_name}_g"
            B_CHAN = f"{self.temp_file_name}_b"
            
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

            ha_formula = self.ha_formula_var.get()
            s2_formula = self.s2_formula_var.get()
            oiii_formula = self.oiii_formula_var.get()
            
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

            # Process Ha
            # Ha component (from Red channel) - using Python copy function
            if ha_formula.strip() == 'R':
                # We use glob to find the R_CHAN.* file
                found_r_files = glob.glob(f"{R_CHAN}.*")
                if not found_r_files:
                    self.siril.log(f"Error: R channel file not found after split: {R_CHAN}.*", s.LogColor.RED)
                    return
                
                # Take the first (and should be the only) file found
                actual_r_chan_path = found_r_files[0]

                # Now pass the full path with extension to the copy_fits_file function
                if not copy_fits_file(actual_r_chan_path, HA_OUT):
                    self.siril.log(f"Error copying {actual_r_chan_path} to {HA_OUT}", s.LogColor.RED)
                    return
            else:
                #pm_ha_formula = format_formula(ha_formula, {'R': f'${R_CHAN}$', 'G': f'${G_CHAN}$', 'B': f'${B_CHAN}$'})
                pm_ha_formula = format_formula(ha_formula, placeholders)
                self.siril.cmd("pm", f'"{pm_ha_formula}"')
                self.siril.cmd("save", f'"{HA_OUT}"')

            # Process OIII
            pm_oiii_formula = format_formula(oiii_formula, placeholders)
            self.siril.cmd("pm", f'"{pm_oiii_formula}"')
            self.siril.cmd("save", f'"{OIII_OUT}"')

            # Process S2
            pm_s2_formula = format_formula(s2_formula, placeholders)
            self.siril.cmd("pm", f'"{pm_s2_formula}"')
            self.siril.cmd("save", f'"{S2_PROXY_OUT}"')

            self.siril.log("Intermediate channels generated successfully.", s.LogColor.GREEN)
            # Clean up the split files now, as they are no longer needed
            # for path in [R_CHAN, G_CHAN, B_CHAN]:
            #     delete_file_if_exists(path, self.siril.log)

            # --- PHASE 2: Compose the selected palette ---
            #self.siril.undo_save_state(f"Compose Palette: {palette_name}")
            self.siril.log(f"Composing the {palette_name} palette...", s.LogColor.GREEN)

            # don't need to include filename extensions here: Siril will add them automatically when saving, according to the user's preferred FITS extension
            output_filename = f"{self.base_file_name}_result_{palette_name}_{formula_name}"

            # Check if intermediate files exist, just in case
            if not all(os.path.exists(f) for f in [HA_OUT, OIII_OUT, S2_PROXY_OUT]):
                 self.siril.log("Error: Intermediate files not found. Please Reset.", s.LogColor.RED)
                 messagebox.showerror("Error", "Missing intermediate files. Please use the Reset button and try again.")
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

            self.siril.cmd(command, f'"-out={output_filename}"', "-nosum")

            self.siril.cmd("load", "\"" + output_filename + "\"")
            self.siril.log(f"Successfully generated and loaded the {palette_name} palette!", s.LogColor.GREEN)

        except SirilError as e:
            self.siril.log(f"Siril error during execution: {e}", s.LogColor.RED)
            messagebox.showerror("Siril Error", str(e))
        except Exception as e:
            self.siril.log(f"An unexpected error occurred: {e}", s.LogColor.RED)
            messagebox.showerror("Generic Error", str(e))

def main():
    try:
        # Create the main GUI window
        #root = ThemedTk(theme="adapta") # Try a modern theme if ttkthemes is available
        root = ThemedTk() if SIRIL_ENV else tk.Tk()

        app = HubblePaletteApp(root)

        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
if __name__ == '__main__':
    main()