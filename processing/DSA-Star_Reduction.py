# Feb. 18, 2025
# (c) Rich Stevenson - Deep Space Astro
# SPDX-License-Identifier: GPL-3.0-or-later
# Script for reducing stars using pixel math.
# 
# YouTube https://www.youtube.com/@DeepSpaceAstro
# Instagram https://www.instagram.com/deepspaceastro_official/
# FaceBook https://www.facebook.com/DeepSpaceAstro/
# TikTok https://www.tiktok.com/@DeepSpaceAstro
# Discord https://discord.gg/eK9nEWy2Wf

import sirilpy as s
s.ensure_installed("ttkthemes")

import os
import sys
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from sirilpy import LogColor, tksiril

VERSION = "1.0.2"

# 1.0.0 Original release by Rich Stevenson
# 1.0.1 Minor edit by Adrian Knagg-Baugh to reflect changes to LogColor names
# 1.0.2 AKB: convert "requires" to use exception handling
# 1.0.3 AKB: remove topmot window settings (now part of match_theme_to_siril())

class StarReductionInterface:
    def __init__(self, root):
        self.root = root
        self.overwrite_checkbox_state = tk.IntVar(value=1)
        self.root.geometry("350x270")
        self.root.resizable(False, False)
        self.root.title(f"DSA-Star Reduction - v{VERSION}")

        # Connect to Siril & match theme
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            self.siril.log("Connection failed: {e}", color=LogColor.RED)
            sys.exit()

        tksiril.match_theme_to_siril(self.root, self.siril)
        self.siril.log("Connected successfully!", color=LogColor.GREEN)

        # Initial checks
        if not self.initial_checks():
            self.root.withdraw()
            sys.exit()

        # Get image info
        self.setup_image_info()
        
        # Create GUI
        self.create_widgets()
        
        self.root.protocol("WM_DELETE_WINDOW", self.run_close)

    def initial_checks(self):
    
        # Check Siril version
        require_version = "1.3.6"
        try:
            self.siril.cmd("requires", require_version)
        except:
            self.siril.error_messagebox(f"This script requires Siril version {require_version} or later!")
            return False
            
        # Check Starnet configuration
        starnet_path = self.siril.get_siril_config("core", "starnet_exe")
        if not starnet_path or not os.path.isfile(starnet_path) or not os.access(starnet_path, os.X_OK):
            self.siril.error_messagebox("Starnet Command Line Tool was not found or is not configured!")
            return False
        
        # Check an image is loaded
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("Open a FITS image before running Star Reduction!")
            return False

        # Check current image file extension
        path = self.siril.get_image_filename()
        basename = os.path.basename(path)
        get_extension = os.path.splitext(basename)[1]
        if get_extension not in (".fit", ".fits"):
            self.siril.error_messagebox(f"The image that is open is a {get_extension} and is not supported.\nPlease open a FITS file and run the script again.")
            return False

        # Check if already reduced
        if "_ReducedStars" in basename:
            self.siril.error_messagebox("This image has already had star reduction applied."
            "\n\nOpen an image that has not yet undergone star reduction.")
            return False

        return True

    def setup_image_info(self):
        # Get current image filename & set working directory to opened image directory.
        path = self.siril.get_image_filename()
        self.img_name = os.path.basename(path)
        self.img_dir = os.path.dirname(os.path.abspath(path))
        self.siril.cmd("cd", f'"{self.img_dir}"')
        os.chdir(self.img_dir)
        self.img_name_pm = f"${self.img_name}$" # Wrap in $ for PixelMath use
        self.get_extension = os.path.splitext(self.img_name)[1] # Get file extension
        self.file_name_without_ext = os.path.splitext(os.path.basename(path))[0] # Get filename without extension

    def create_widgets(self):
        self.resolution = 0.01
     
        # Create Frame
        frame = ttk.Frame(self.root)
        frame.pack(expand=True, fill="both")

        # Title
        title_label = ttk.Label(frame, text="Star Reduction")
        title_label.pack(pady=(10, 10))

        # Reminder Label
        reminder_label = ttk.Label(frame, text="Image must already be stretched!", font=("Arial", 10, "bold"))
        reminder_label.pack(pady=(0,10))

        # Create Frame for Parameters
        slider_frame = ttk.LabelFrame(frame, text="Parameters", padding=10)
        slider_frame.pack(fill="x", padx=10, pady=(0, 5))

        # Slider Text Label
        slider_text_label = ttk.Label(slider_frame, text="Value: ")
        slider_text_label.pack(side="left", padx=(0, 5))

        # Slider setup
        self.slider_value = tk.DoubleVar(value=0.01)  

        # Create Slider
        self.slider = ttk.Scale(slider_frame, from_=round(0.0 / self.resolution) * self.resolution, 
                    to=round(0.99 / self.resolution) * self.resolution, 
                    orient="horizontal", variable=self.slider_value)
        self.slider.pack(side="left", expand=True, fill="x")

        # Slider Value Label
        self.slider_value_label = ttk.Label(slider_frame, text="0.2")
        self.slider_value_label.pack(side="left", padx=(5, 0))

        # Set Slider value & update
        self.slider.set(0.2)
        self.slider_value.trace_add("write", self.update_slider_value)

        # Create Frame for Options
        overwrite_checkbox_frame = ttk.LabelFrame(frame, text="Options", padding=10)
        overwrite_checkbox_frame.pack(fill="x", padx=10, pady=(5, 10), anchor="w")

        # Overwrite Checkbox
        overwrite_checkbox = ttk.Checkbutton(overwrite_checkbox_frame, text="Overwrite Output File", variable=self.overwrite_checkbox_state)
        overwrite_checkbox.pack(side="left")
        
        # Buttons
        submit_button = ttk.Button(frame, text="Apply", width=7, command=self.run_reduction)
        submit_button.pack(padx=5, side="right")

        close_button = ttk.Button(frame, text="Close", width=7, command=self.run_close)
        close_button.pack(padx=5, side="right")
        
        help_button = ttk.Button(frame, text="Help", width=7, command=self.show_help)
        help_button.pack(padx=10, side="left")

    def update_slider_value(self, *args):
        rounded_value = round(self.slider_value.get() / self.resolution) * self.resolution
        self.slider_value.set(rounded_value)
        self.slider_value_label.config(text=f"{rounded_value:.2f}")

    def show_help(self):
        self.siril.info_messagebox("Reduces the size of the stars"
        " in an image based on the value selected. The lower the number, the"
        " stronger the reduction. You can also increase the size of the stars by"
        " raising the value. A new file named {image_name}_ReducedStars.fit"
        " will be created in the same directory as the original image."
        "\n\n-StarNet Command Line Tool must be installed and configured in Siril."
        "\n\n-Image must be in FITS format and stretched (non-linear)."
        "\n\n-Set the value using the slider, and click Apply. The lower the value, the stronger the"
        " reduction. Value of 0.5 will have no affect, while a value higher than 0.5 will"
        " actually increase star size."
        "\n\n-Uncheck 'Overwrite Output File' to save a new image each time Apply is clicked."
        " Each image will have the selected value included in it's filename. Leaving this option"
        " checked will only create one file that will be overwritten on each run."
        "\n\n-Click Close to exit the script", True)
        
    def run_reduction(self):
        try:
            filename_overwrite = self.overwrite_checkbox_state.get()
            star_reduction_value = round(self.slider.get(), 2)

            default_ext = self.siril.get_siril_config("core", "extension")  # Default FITS extension
            img_name_default_ext = f"{self.file_name_without_ext}{default_ext}"

            # Check if Apply was previously clicked by checking for starless file. If so, no need to run Starnet again
            if os.path.exists(f"starless_{img_name_default_ext}"):
                self.siril.log("Previous star reduction was detected.", color=LogColor.GREEN)
                starless = f"$starless_{img_name_default_ext}$"  # Wrap in $ for PixelMath
            else:
                self.siril.cmd("starnet", "-nostarmask")
                path = self.siril.get_image_filename()
                starless = f"${os.path.basename(path)}$"  # Wrap in $ for PixelMath

            # Pixel math for star reduction using selected value
            self.siril.cmd("pm", f"\"~((~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless}))*~{starless})\"")

            current_img = self.img_name.replace(self.get_extension, "")
            
            if filename_overwrite == 1:
                self.siril.cmd("save", f"\"{current_img}_ReducedStars{default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars{default_ext}\"")
            else:
                print("overwrite no")
                self.siril.cmd("save", f"\"{current_img}_ReducedStars_{star_reduction_value}{default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars_{star_reduction_value}{default_ext}\"")

            self.siril.log("Star Reduction is complete!", color=LogColor.GREEN)
            return True
        except Exception as e:
            self.siril.log(f"Error in run_reduction: {str(e)}", color=LogColor.RED)
            return False

    def run_close(self):
        self.root.destroy()

def main():
    root = ThemedTk()
    app = StarReductionInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()
