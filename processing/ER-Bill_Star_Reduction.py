# Jun. 16, 2025
# (c) Eduardo Ramírez - @edramigon
# SPDX-License-Identifier: GPL-3.0-or-later
# Script for reducing stars using pixel math.
# 
# Instagram https://www.instagram.com/edramigon/
# TikTok https://www.tiktok.com/@edramigon

import sirilpy as s
s.ensure_installed("ttkthemes")

import os
import sys
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from sirilpy import LogColor, tksiril

VERSION = "1.0.1"

# 1.0.0 Original release by Eduardo Ramírez
# 1.0.1 Add siril.update_progress() calls

class StarReductionInterface:
    def __init__(self, root):
        self.root = root
        self.overwrite_checkbox_state = tk.IntVar(value=1)
        self.root.geometry("350x410")
        self.root.resizable(False, False)
        self.root.title(f"ER-Bill_Star Reduction - v{VERSION}")

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
        self.enable_widget(self.mode_frame, False)
        self.enable_widget(self.interactions_frame, False)
        self.reduction_method_var.trace_add('write', self.update_widgets)
        self.root.protocol("WM_DELETE_WINDOW", self.run_close)

    def initial_checks(self):
    
        # Check Siril version
        require_version = "1.3.6"
        self.default_ext = self.siril.get_siril_config("core", "extension")  # Default FITS extension
        self.final_name = "0"
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
        if get_extension.lower() not in (".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz"):
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
        title_label = ttk.Label(frame, text="Bill Blanshan's Star Reduction by @edramigon")
        title_label.pack(pady=(10, 10))

        # Reminder Label
        reminder_label = ttk.Label(frame, text="Image must already be stretched!", font=("Arial", 10, "bold"))
        reminder_label.pack(pady=(0,10))

        # Create Frame for Method
        method_frame = ttk.LabelFrame(frame, text="Reduction method", padding=10)
        method_frame.pack(fill="x", padx=10, pady=(0,5))
        
        # Create buttons for Method
        self.reduction_method_var = tk.StringVar(value="Transfer")
        reduction_methods = ["Transfer", "Halo", "Star"]
        for method in reduction_methods:
            ttk.Radiobutton(
                method_frame,
                text=method,
                variable=self.reduction_method_var,
                value=method
            ).pack(side="left", expand=True, fill="x")


        # Create Frame for Parameters
        self.parameters_frame = ttk.LabelFrame(frame, text="Parameters", padding=10)
        self.parameters_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        # Create Frame for Stretch factor
        self.stretch_frame = ttk.Frame(self.parameters_frame)
        self.stretch_frame.pack(fill="x", pady=(0, 5))

        # Slider Text Label
        self.slider_text_label = ttk.Label(self.stretch_frame, text="Stretch factor: ")
        self.slider_text_label.pack(side="left", padx=(0, 5))

        # Slider setup
        self.slider_value = tk.DoubleVar(value=0.01)  

        # Create Slider
        self.slider = ttk.Scale(self.stretch_frame, from_=round(0.0 / self.resolution) * self.resolution, 
                    to=round(0.99 / self.resolution) * self.resolution, 
                    orient="horizontal", variable=self.slider_value)
        self.slider.pack(side="left", expand=True, fill="x")

        # Slider Value Label
        self.slider_value_label = ttk.Label(self.stretch_frame, text="0.2")
        self.slider_value_label.pack(side="left", padx=(5, 0))

        # Set Slider value & update
        self.slider.set(0.2)
        self.slider_value.trace_add("write", self.update_slider_value)
        
        # Create Frame for Mode
        self.mode_frame = ttk.Frame(self.parameters_frame)
        self.mode_frame.pack(fill="x", pady=(0, 5))
        
        # Mode Text Label
        mode_text_label = ttk.Label(self.mode_frame, text="Mode: ")
        mode_text_label.pack(side="left", padx=(0, 5))
        
        self.modes_var = tk.StringVar(value="Moderate")
        reduction_modes = ["Strong", "Moderate", "Soft"]
        for mode in reduction_modes:
            ttk.Radiobutton(
                self.mode_frame,
                text=mode,
                variable=self.modes_var,
                value=mode
            ).pack(side="left", expand=True, fill="x")

        # Create Frame for Interactions
        self.interactions_frame = ttk.Frame(self.parameters_frame)
        self.interactions_frame.pack(fill="x", pady=(0, 5))
        
        # Interactions Text Label
        interactions_text_label = ttk.Label(self.interactions_frame, text="Interactions: ")
        interactions_text_label.pack(side="left", padx=(0, 5))
        
        # Interactions spinbox
        self.interactions_spin = ttk.Spinbox(self.interactions_frame, from_=1, to=3)
        self.interactions_spin.pack(side="left")
        self.interactions_spin.set("2")
        
        # Create Frame for Options
        overwrite_checkbox_frame = ttk.LabelFrame(frame, text="Options", padding=10)
        overwrite_checkbox_frame.pack(fill="x", padx=10, pady=(0, 5), anchor="w")

        # Overwrite Checkbox
        overwrite_checkbox = ttk.Checkbutton(overwrite_checkbox_frame, text="Overwrite Output File", variable=self.overwrite_checkbox_state)
        overwrite_checkbox.pack(side="left")
        
        # Buttons
        submit_button = ttk.Button(frame, text="Apply", width=7, command=self.run_reduction)
        submit_button.pack(padx=5, side="right")

        close_button = ttk.Button(frame, text="Save&Close", width=14, command=self.run_close)
        close_button.pack(padx=5, side="right")
        
        help_button = ttk.Button(frame, text="Help", width=7, command=self.show_help)
        help_button.pack(padx=10, side="left")
        
    def enable_children(self, parent, enabled=True):
        for child in parent.winfo_children():
            wtype = child.winfo_class()
            if wtype not in ('Frame', 'Labelframe', 'TFrame', 'TLabelframe'):
                child.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            else:
                enable_children(child, enabled)

    def enable_widget(self, widget, enabled=True):
        wtype = widget.winfo_class()
        if wtype not in ('Frame', 'Labelframe', 'TFrame', 'TLabelframe'):
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        else:
            self.enable_children(widget, enabled)
    
    def update_widgets(self, *args):
        if self.reduction_method_var.get() == "Halo":
            self.enable_widget(self.stretch_frame)
            self.enable_widget(self.mode_frame, False)
            self.enable_widget(self.interactions_frame, False)
        elif self.reduction_method_var.get() == "Star":
            self.enable_widget(self.stretch_frame, False)
            self.enable_widget(self.mode_frame)
            self.enable_widget(self.interactions_frame)            
        elif self.reduction_method_var.get() == "Transfer":
            self.enable_widget(self.stretch_frame)
            self.enable_widget(self.mode_frame, False)
            self.enable_widget(self.interactions_frame, False)            
    
    def update_slider_value(self, *args):
        rounded_value = round(self.slider_value.get() / self.resolution) * self.resolution
        self.slider_value.set(rounded_value)
        self.slider_value_label.config(text=f"{rounded_value:.2f}")

    def show_help(self):
        self.siril.info_messagebox("Reduces the size of the stars"
        " in an image based on the selected method and parameters. There are 3 methods:"
        "\n\nTransfer: The lower the number, the stronger the reduction. You can also increase"
        " the size of the stars by raising the value. A value of 0.5 will have no affect; values below"
        " 0.5 reduce the stars, and values above increase their size."
        "\n\nHalo: Works the same as the Transfer method but preserves the original halo around the stars"
        "\n\nStar:\nStrong Mode: Produces smaller, sharper stars while removing the tiny ones"
        "\nModerate Mode: Keeps stars sharp and includes some small stars"
        "\nSoft Mode: Applies a simple reduction to the stars in the original image"
        "\n\n-StarNet Command Line Tool must be installed and configured in Siril."
        "\n\n-Image must be in FITS format and stretched (non-linear)."
        "\n\n-Set the value using the slider, and click Apply."
        "\n\n-Uncheck 'Overwrite Output File' to save a new image each time."
        " Each image will include the selected value in its filename. Leaving this option checked will overwrite the file on each run"
        " with a file named {image_name}_ReducedStars.fit"
        "\n\n-Once you're satisfied with the result, click 'Save & Close' to save your work and exit the script", True)
        tksiril.elevate(self.root)
        
    def run_reduction(self):
        try:
            star_reduction_value = round(self.slider.get(), 2)
            
            img_name_default_ext = f"{self.file_name_without_ext}{self.default_ext}"

            # Check if Apply was previously clicked by checking for starless file. If so, no need to run Starnet again
            if os.path.exists(f"starless_{img_name_default_ext}"):
                self.siril.log("Previous star reduction was detected.", color=LogColor.GREEN)
                starless = f"$starless_{img_name_default_ext}$"  # Wrap in $ for PixelMath
            else:
                self.siril.cmd("starnet", "-nostarmask")
                path = self.siril.get_image_filename()
                starless = f"${os.path.basename(path)}$"  # Wrap in $ for PixelMath

            # Pixel math for star reduction using selected value
            
            if self.reduction_method_var.get() == "Halo":
                self.siril.update_progress("Generating PixelMath expressions",0.0)
                h1=f"((~(~{self.img_name_pm}/~{starless})-~(~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless})))*~{starless})"
                h2=f"(~(~{self.img_name_pm}/~{starless})-~(~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless})))"
                self.siril.update_progress("Processing image data with PixelMath",0.1)
                self.siril.cmd("pm", f"\"{self.img_name_pm}*~(({h1}+{h2})/2)\"")
                self.final_name = "Halo_{star_reduction_value}"
                self.siril.update_progress("Star Reduction is complete",1.0)
                
            elif self.reduction_method_var.get() == "Star":
                self.siril.update_progress("Generating PixelMath expressions",0.0)
                s1=f"({self.img_name_pm}*~(~(max(0,min(1,{starless}/{self.img_name_pm})))*~{self.img_name_pm}))"
                s2=f"(max({s1},({self.img_name_pm}*{s1})+({s1}*~{s1})))"
                s3=f"({s1}*~(~(max(0,min(1,{starless}/{s1})))*~{s1}))"
                s4=f"(max({s3},({self.img_name_pm}*{s3})+({s3}*~{s3})))"
                s5=f"({s3}*~(~(max(0,min(1,{starless}/{s3})))*~{s3}))"
                s6=f"(max({s5},({self.img_name_pm}*{s5})+({s5}*~{s5})))"
                self.siril.update_progress("Processing image data with PixelMath",0.1)
                interactions = self.interactions_spin.get()
                if self.modes_var.get() == "Strong":
                    self.siril.cmd("pm", f"\"(iif({interactions}==1,{s1},iif({interactions}==2,{s3},{s5})))\"")
                    self.final_name = "Star_Strong_{interactions}interactions"
                elif self.modes_var.get() == "Moderate":
                    self.siril.cmd("pm", f"\"(iif({interactions}==1,{s2},iif({interactions}==2,{s4},{s6})))\"")
                    self.final_name = "Star_Moderate_{interactions}interactions"
                elif self.modes_var.get() == "Soft":
                    self.siril.cmd("pm", f"\"((({self.img_name_pm}-({self.img_name_pm}-iif({interactions}==1,{s2},iif({interactions}==2,{s4},{s6}))))+({self.img_name_pm}*~({self.img_name_pm}-iif({interactions}==1,{s2},iif({interactions}==2,{s4},{s6})))))/2)\"")
                    self.final_name = "Star_Soft_{interactions}interactions"
                self.siril.update_progress("Star Reduction is complete",1.0)    
                
            elif self.reduction_method_var.get() == "Transfer":
                self.siril.update_progress("Generating PixelMath expressions",0.0)
                self.siril.update_progress("Processing image data with PixelMath",0.1)
                self.siril.cmd("pm", f"\"~((~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless}))*~{starless})\"")
                self.final_name = "Transfer_{star_reduction_value}"
                self.siril.update_progress("Star Reduction is complete",1.0) 
            
            self.siril.log("Star Reduction is complete!", color=LogColor.GREEN)
            return True
        except Exception as e:
            self.siril.log(f"Error in run_reduction: {str(e)}", color=LogColor.RED)
            return False

    def run_close(self):
        filename_overwrite = self.overwrite_checkbox_state.get()
        current_img = self.img_name.replace(self.get_extension, "")
        if self.final_name != "0":
            if filename_overwrite == 1:
                self.siril.cmd("save", f"\"{current_img}_ReducedStars{self.default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars{self.default_ext}\"")
            else:
                print("overwrite no")
                self.siril.cmd("save", f"\"{current_img}_ReducedStars_{self.final_name}{self.default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars_{self.final_name}{self.default_ext}\"")

        self.root.destroy()

def main():
    root = ThemedTk()
    app = StarReductionInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()