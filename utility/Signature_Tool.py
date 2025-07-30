# (c) Carlo Mollicone - AstroBOH (2025)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Description:
# This script allows you to insert a signature/logo (PNG file with transparency) onto the current image in Siril.
# You can save and manage multiple signature profiles through a graphical interface.
#
# Version History
# 1.0.0 Initial release
# 1.0.1 Add undo_save_state
#       Add handling of files with different bit depths
# 1.0.2 Missing ensure_installed components
#       


VERSION = "1.0.2"
CONFIG_FILENAME = "SignatureTool.conf"

# --- Core Imports ---
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import configparser

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
# This allows the script to be run externally for testing (with limited functionality).
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=0.6.37'):
        messagebox.showerror("Error", "requires sirilpy module >= 0.6.37 (Siril >= 1.4.0 Beta 2)")
        sys.exit(1)

    SIRIL_ENV = True

    # Import Siril GUI related components
    from sirilpy import tksiril, SirilError
    from sirilpy.models import FPoint

    s.ensure_installed("ttkthemes", "numpy", "astropy", "opencv-python")
    from ttkthemes import ThemedTk

    # --- Imports for Image Processing ---
    import cv2
    import numpy as np

except ImportError:
    SIRIL_ENV = False

# --- Main Application Class ---
class SignatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"AstroBOH Signature Tool v{VERSION} - (c) Carlo Mollicone")
        
        # --- Siril Connection ---
        # Initialize Siril connection
        self.siril = None # Initialize to None
        if SIRIL_ENV:
            self.siril = s.SirilInterface()
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                messagebox.showerror("Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
                self.on_closing()
                return
 
        else:
            messagebox.showerror("Invalid environment", "This script must be run from within Siril.")
            self.on_closing()
            return

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image is loaded")
            self.on_closing()
            return

        shape_image = self.siril.get_image_shape()
        if shape_image[0] != 3:
            self.siril.error_messagebox("The image must be a RGB image.")
            self.on_closing()
            return

        tksiril.match_theme_to_siril(self.root, self.siril)
        
        # --- State and configuration variables ---
        self.profiles = {}
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join(self.siril.get_siril_configdir(), CONFIG_FILENAME)

        # Load configurations before creating widgets
        self.load_config()

        # Creating the graphical interface
        self.create_widgets()
        
        # Setting window size and centering
        width, height = 550, 480
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = f'{width}x{height}+{(screenwidth - width) // 2}+{(screenheight - height) // 2}'
        self.root.geometry(alignstr)
        self.root.resizable(False, False)
        
        # Window closing management
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_config(self):
        """Load profiles from the configuration file."""
        self.profiles = {}
        self.config.read(self.config_path)
        for section in self.config.sections():
            self.profiles[section] = {
                'path': self.config.get(section, 'path', fallback=''),
                'size': self.config.getint(section, 'size', fallback=5),
                'margin': self.config.getint(section, 'margin', fallback=2),
                'position': self.config.get(section, 'position', fallback='Bottom_Center'),
                'opacity': self.config.getint(section, 'opacity', fallback=100),
            }
        self.siril.log(f"Loaded {len(self.profiles)} profiles from {self.config_path}", s.LogColor.BLUE)

    def save_config(self):
        """Save all profiles in the configuration file."""
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
        """Create all the elements of the graphical interface."""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Profiles Section ---
        profiles_frame = ttk.LabelFrame(main_frame, text="Signature Profiles", padding=10)
        profiles_frame.pack(fill=tk.X, pady=(0, 10))
        profiles_frame.columnconfigure(1, weight=1)

        # Profile Dropdown
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(profiles_frame, textvariable=self.profile_var, state="readonly")
        self.profile_combo['values'] = list(self.profiles.keys())
        self.profile_combo.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_selected)

        # Frame for management buttons
        buttons_subframe = ttk.Frame(profiles_frame)
        buttons_subframe.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5,0))
        buttons_subframe.columnconfigure((0, 1, 2), weight=1)

        # Buttons
        save_new_button = ttk.Button(buttons_subframe, text="Save New...", command=self.save_current_profile)
        save_new_button.grid(row=0, column=0, sticky="ew", padx=2)

        update_button = ttk.Button(buttons_subframe, text="Update Selected", command=self.update_current_profile)
        update_button.grid(row=0, column=1, sticky="ew", padx=2)

        delete_button = ttk.Button(buttons_subframe, text="Delete Selected", command=self.delete_current_profile)
        delete_button.grid(row=0, column=2, sticky="ew", padx=2)

        # --- Settings Section ---
        settings_frame = ttk.LabelFrame(main_frame, text="Current Settings", padding=10)
        settings_frame.pack(fill=tk.X, expand=True)
        settings_frame.columnconfigure(1, weight=1)

        # File Logo
        ttk.Label(settings_frame, text="File Logo:").grid(row=0, column=0, sticky="w", pady=2)
        self.logo_path_var = tk.StringVar()
        logo_path_entry = ttk.Entry(settings_frame, textvariable=self.logo_path_var, state="readonly")
        logo_path_entry.grid(row=0, column=1, sticky="ew", padx=5)
        select_file_button = ttk.Button(settings_frame, text="Select...", command=self.select_logo_file)
        select_file_button.grid(row=0, column=2)

        # Size
        ttk.Label(settings_frame, text="Size (%):").grid(row=1, column=0, sticky="w", pady=2)
        self.size_var = tk.IntVar(value=5)
        # Size slider
        size_slider = ttk.Scale(
            settings_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.size_var,
            command=lambda value: self.size_var.set(round(float(value)))
        )
        size_slider.grid(row=1, column=1, sticky="ew", padx=5)
        # Label showing the current value of the slider
        size_label = ttk.Label(settings_frame, textvariable=self.size_var, width=4)
        size_label.grid(row=1, column=2, padx=(0, 5))

        # Margin
        ttk.Label(settings_frame, text="Margin (%):").grid(row=2, column=0, sticky="w", pady=2)
        self.margin_var = tk.IntVar(value=2)
        # Margin slider
        margin_slider = ttk.Scale(
            settings_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.margin_var,
            command=lambda value: self.margin_var.set(round(float(value)))
        )
        margin_slider.grid(row=2, column=1, sticky="ew", padx=5)
        # Label showing the current value
        margin_label = ttk.Label(settings_frame, textvariable=self.margin_var, width=4)
        margin_label.grid(row=2, column=2, padx=(0, 5))

        # Opacity
        ttk.Label(settings_frame, text="Opacity (%):").grid(row=3, column=0, sticky="w", pady=2)
        self.opacity_var = tk.IntVar(value=100)
        # Opacity slider
        opacity_slider = ttk.Scale(
            settings_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.opacity_var,
            command=lambda value: self.opacity_var.set(round(float(value)))
        )
        opacity_slider.grid(row=3, column=1, sticky="ew", padx=5)
        # Label showing the current value
        opacity_label = ttk.Label(settings_frame, textvariable=self.opacity_var, width=4)
        opacity_label.grid(row=3, column=2, padx=(0, 5))

        # --- Position Grid ---
        position_frame = ttk.LabelFrame(settings_frame, text="Position")
        position_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 2), padx=5)
        position_frame.columnconfigure((0, 1, 2), weight=1) 

        # The control variable is the same, so the rest of the code doesn't change
        self.position_var = tk.StringVar(value="Bottom_Center")

        # I define the positions in a 2D grid to make it easier to create
        positions_grid = [
            ["Top_Left",    "Top_Center",    "Top_Right"],
            ["Middle_Left", "Middle_Center", "Middle_Right"],
            ["Bottom_Left", "Bottom_Center", "Bottom_Right"]
        ]

        # Loop to create and position the 9 radio buttons in the grid
        for r, row_list in enumerate(positions_grid):
            for c, position_value in enumerate(row_list):
                radio = ttk.Radiobutton(
                    position_frame,
                    text=position_value.replace('_', ' '),
                    variable=self.position_var,
                    value=position_value,
                    width=19
                )
                radio.grid(row=r, column=c, padx=10, pady=5, sticky="w")
                # Add a tooltip to show the location name on mouseover
                tksiril.create_tooltip(radio, position_value.replace('_', ' '))

        # --- Actions Section ---
        action_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
        action_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        apply_button = ttk.Button(action_frame, text="Apply Signature", command=self.on_apply)
        apply_button.pack(side=tk.RIGHT)

    def on_profile_selected(self, event=None):
        """Update the UI when a profile is selected from the dropdown."""
        profile_name = self.profile_var.get()
        if profile_name in self.profiles:
            settings = self.profiles[profile_name]
            self.logo_path_var.set(settings.get('path', ''))
            self.size_var.set(settings.get('size', 5.0))
            self.margin_var.set(settings.get('margin', 2.0))
            self.position_var.set(settings.get('position', 'Bottom_Center'))
            self.opacity_var.set(settings.get('opacity', 100))

            self.siril.log(f"Load profile settings '{profile_name}'", s.LogColor.GREEN)

    def select_logo_file(self):
        """Opens a dialog box to select the logo file."""
        filepath = filedialog.askopenfilename(
            title="Select PNG file",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filepath:
            self.logo_path_var.set(filepath)

    def save_current_profile(self):
        """Save the current settings as a new profile or update an existing one."""
        profile_name = simpledialog.askstring("Save Profile", "Enter a name for this profile:", parent=self.root)

        if not profile_name:
            return

        current_settings = {
            'path': self.logo_path_var.get(),
            'size': self.size_var.get(),
            'margin': self.margin_var.get(),
            'position': self.position_var.get(),
            'opacity': self.opacity_var.get(),
        }
        
        # Validation
        if not os.path.exists(current_settings['path']):
            messagebox.showerror("Error", "The logo file path is invalid.")
            return

        self.profiles[profile_name] = current_settings
        self.save_config() # Save all changes to file
        
        # Update the Combobox
        self.profile_combo['values'] = list(self.profiles.keys())
        self.profile_var.set(profile_name)
        self.siril.log(f"Profile '{profile_name}' saved successfully.", s.LogColor.GREEN)

    def update_current_profile(self):
        """Updates the currently selected profile with the current settings."""
        profile_name = self.profile_var.get()
        if not profile_name:
            messagebox.showwarning("No Selection", "Please select a profile to update.")
            return

        current_settings = {
            'path': self.logo_path_var.get(),
            'size': self.size_var.get(),
            'margin': self.margin_var.get(),
            'position': self.position_var.get(),
            'opacity': self.opacity_var.get(),
        }

        self.profiles[profile_name] = current_settings
        self.save_config()
        self.siril.log(f"Profile '{profile_name}' updated successfully.", s.LogColor.GREEN)

    def delete_current_profile(self):
        """Delete the currently selected profile."""
        profile_name = self.profile_var.get()
        if not profile_name:
            messagebox.showwarning("No Selection", "Please select a profile to delete.")
            return

        # Ask for confirmation before deleting
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the profile\n\n'{profile_name}'?"):
            # Remove from dictionary and configuration
            del self.profiles[profile_name]
            self.config.remove_section(profile_name)
            
            self.save_config() # Save changes to the file
            
            # Update the interface
            self.profile_combo['values'] = list(self.profiles.keys())
            self.profile_var.set('') # Deselect
            
            # Clear the fields
            self.logo_path_var.set('')
            self.size_var.set(5)
            self.margin_var.set(2)
            self.opacity_var.set(100)
            self.position_var.set('Bottom_Center')

            self.siril.log(f"Profile '{profile_name}' deleted.", s.LogColor.GREEN)

    def on_apply(self):
        """Main function that applies the logo to the image."""
        try:
            # Load logo with OPENCV
            logo_path = self.logo_path_var.get()
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
                size_percent = self.size_var.get()
                target_logo_h = int(doc_height * (size_percent / 100.0))
                logo_h, logo_w, _ = logo_rgba.shape

                aspect_ratio = logo_w / logo_h
                target_logo_w = int(target_logo_h * aspect_ratio)
                
                resized_logo_rgba = cv2.resize(logo_rgba, (target_logo_w, target_logo_h), interpolation=cv2.INTER_AREA)

                # Separate logo channels (which are 8-bit)
                logo_rgb_8bit = resized_logo_rgba[:, :, :3]
                alpha_mask_8bit = resized_logo_rgba[:, :, 3]

                # Apply the scale factor to the brightness of the logo
                logo_rgb_scaled = logo_rgb_8bit.astype(np.float32) * scaling_factor

                # Blending with consistent data
                resized_h, resized_w, _ = resized_logo_rgba.shape
                margin_percent = self.margin_var.get()
                margin_px = int(doc_height * (margin_percent / 100.0))
                position = self.position_var.get()
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
                x_end = min(x + resized_w, doc_width)
                y_end = min(y + resized_h, doc_height)

                roi = background_bgr[y:y_end, x:x_end]
                
                # Fit logo and mask to ROI if they go outside the borders
                logo_to_blend = logo_rgb_scaled[:y_end-y, :x_end-x]
                alpha_to_blend = alpha_mask_8bit[:y_end-y, :x_end-x]

                opacity_factor = self.opacity_var.get() / 100.0

                # Convert the mask to float (0-1) and 3 channels for blending
                alpha_mask_float = ((alpha_to_blend * opacity_factor) / 255.0)[:, :, np.newaxis]

                roi_float = roi.astype(np.float32)

                # Blending formula: background*(1-alpha) + foreground*alpha
                blended_roi = roi_float * (1 - alpha_mask_float) + logo_to_blend * alpha_mask_float
                
                # Place the modified ROI on the background image
                background_bgr[y:y_end, x:x_end] = np.clip(blended_roi, 0, 65535).astype(background_bgr.dtype)

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

    def on_closing(self):
        """
        Handle dialog close - Called when the window is closed via the 'X' button.
        Close the dialog and disconnect from Siril
        """
        try:
            if SIRIL_ENV and self.siril:
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.BLUE)
            self.siril.disconnect()
            self.root.quit()
        except Exception:
            pass
        self.root.destroy()

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
def main():
    try:
        if SIRIL_ENV:
            # Create the main GUI window
            #root = ThemedTk(theme="adapta") # Try a modern theme if ttkthemes is available
            root = ThemedTk()
        else:
            root = tk.Tk()

        # Create an instance of our application
        app = SignatureApp(root)
        # Start the GUI event loop, which keeps it running
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
if __name__ == '__main__':
    main()