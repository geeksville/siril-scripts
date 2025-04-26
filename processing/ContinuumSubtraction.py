# Narrowband Continuum Subtraction script
# SPDX-License-Identifier: GPL-3.0
# Author: Adrian Knagg-Baugh, (c) 2025

"""
This script provides continuum subtraction for narrowband images.
It uses the currently loaded narrowband image in Siril and allows the user to select 
a continuum image, then automatically determines the optimal scaling factor for subtraction
by minimizing the noise in a user-selected region using AAD (Average Absolute Deviation).
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import sirilpy as s
from sirilpy import tksiril, SirilError
s.ensure_installed("ttkthemes", "scipy", "matplotlib")
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ttkthemes import ThemedTk

### This code uses tkfilebrowser, which currently doesn't work properly in
# a flatpak environment. Commented out for the time being as I hope to make
# it work again in the future.

#filefilter = []
# This makes the filechooser much nicer on Linux
# Leaving the standard tk filedialog on other OSes as the native 
# file dialog is used, which in all other cases is fine
#if sys.platform.startswith("linux"):
#    s.ensure_installed("tkfilebrowser")
#    import tkfilebrowser as filedialog
#    filefilter=[("FITS files", "*.fit|*.fits"), ("All files", "*.*")]
#else:
#    from tkinter import filedialog
#    filefilter=[("FITS files", "*.fit *.fits"), ("All files", "*.*")]
from tkinter import filedialog
filefilter=[("FITS files", "*.fit *.fits"), ("All files", "*.*")]

VERSION = "1.0.1"

def aad(data):
    """
    Calculate Average Absolute Deviation
    """
    mean = np.mean(data)
    return np.mean(np.abs(data - mean))

def find_min(nb, co, c_median, siril):
    """
    Find the approximate scaling factor using a coarse search

    Returns: min_val
    """
    # Sample AAD curve
    scale_factors = np.linspace(-1, 5, 12)
    aad_values = []

    for i, s in enumerate(scale_factors):
        value = aad(nb - (co - c_median) * s)
        aad_values.append(value)
        
        # Calculate progress as a float between 0 and 1
        progress = i / (len(scale_factors) - 1)
        siril.update_progress("Coarse bounds check...", progress)
    min_val = scale_factors[np.argmin(aad_values)]
    return min_val

def perform_continuum_subtraction(narrowband_image,
                                  continuum_image,
                                  selection,
                                  c_median,
                                  siril,
                                  plot_optimization=True,
                                  tk_root=None):
    """
    Find the optimal scaling factor for continuum subtraction and apply it.

    Returns: optimal_scale
    """
    x, y, w, h = selection
    def slc(im): return im[y:y+h, x:x+w]
    nb = slc(narrowband_image)
    co = slc(continuum_image)

    # Rough pass to check the bounds are reasonable
    approx_min = find_min(nb, co, c_median, siril)
    max_val = approx_min + 1.0
    min_val = approx_min - 1.0
    
    # Sample AAD curve
    scale_factors = np.linspace(min_val, max_val, 40)
    aad_values = []

    for i, s in enumerate(scale_factors):
        value = aad(nb - (co - c_median) * s)
        aad_values.append(value)
        
        # Calculate progress as a float between 0 and 1
        progress = i / (len(scale_factors) - 1)
        siril.update_progress("Optimizing continuum subtraction...", progress)

    # Define smooth‐V model
    def smooth_v(x, A, s0, eps, B):
        return A * np.sqrt((x - s0)**2 + eps**2) + B

    # Initial guesses
    B0       = np.min(aad_values)
    s0_0     = scale_factors[np.argmin(aad_values)]
    # Approximate slope from tails
    slope_est = (aad_values[-1] - aad_values[0]) / (scale_factors[-1] - scale_factors[0])
    A0       = slope_est
    eps0     = 0.01

    p0    = [A0, s0_0, eps0, B0 ]
    lb    = [ -1.0,  0.00,  0.0, 0.00]
    ub    = [ np.inf,2*max_val, np.inf, np.inf]

    # Fit
    popt, pcov = curve_fit(smooth_v, scale_factors, aad_values,
                           p0=p0, bounds=(lb, ub))
    A_opt, s0_opt, eps_opt, B_opt = popt
    optimal_scale = float(np.clip(s0_opt, 0, 1))

    if plot_optimization and tk_root is not None:
        def show_plot_in_tk():

            # Prepare the figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(scale_factors, aad_values,
                       color='C0', alpha=0.6, label='AAD values')

            fx = np.linspace(min_val, max_val, 500)
            fy = smooth_v(fx, *popt)
            ax.plot(fx, fy, 'C3-', 
                    label=(f'Smooth‐V fit:\n'
                           r'$y=A\sqrt{(s-s_0)^2+\epsilon^2}+B$' '\n'
                           f'→ A={A_opt:.4g}, s₀={s0_opt:.4g}, ε={eps_opt:.4g}, B={B_opt:.4g}'))

            min_aad = smooth_v(optimal_scale, *popt)
            ax.plot([optimal_scale], [min_aad], 'go', ms=10,
                    label=f'Optimal scale = {optimal_scale:.4f}')
            ax.axvline(optimal_scale, color='green', ls='--', alpha=0.5)

            ax.set_title('Optimization for Continuum Subtraction')
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Average Absolute Deviation (AAD)')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
            fig.tight_layout()

            # Create a new window for the plot
            plot_window = tk.Toplevel(tk_root)
            plot_window.title("Continuum Subtraction Optimization")
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Schedule on Tk thread
        tk_root.after(0, show_plot_in_tk)

    return optimal_scale

def aad(array):
    """
    Calculate the Average Absolute Deviation of an array.
    """
    return np.mean(np.abs(array - np.mean(array)))

class ContinuumSubtractionInterface:
    """ This class provides the GUI and related callbacks """
    def __init__(self, root):
        self.root = root
        self.root.title(f"Continuum Subtraction - v{VERSION}")
        self.root.resizable(False, False)
        self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        try:
            self.siril.connect()
        except s.SirilConnectionError:
            self.siril.error_messagebox("Failed to connect to Siril")
            return

        # Check if the version of Siril is high enough
        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            return

        # Check if an image is loaded
        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image loaded. Please load a narrowband image in Siril first.")
            self.root.destroy()
            return
        
        shape = self.siril.get_image_shape()
        if shape[0] != 1:
            self.siril.error_messagebox("Images must be mono. Please load a mono narrowband image in Siril.")
            self.root.destroy()
            return

        # Add image cache to store loaded images
        self.cached_continuum_path = None
        self.cached_continuum_data = None
        self.cached_continuum_median = None
        self.cached_selection = None
        
        # Store the initial narrowband image information
        self.initial_narrowband_filename = self.siril.get_image_filename()
        self.initial_narrowband_data = self.siril.get_image_pixeldata()
        self.narrowband_header = self.siril.get_image_fits_header()
        
        # This flag will help us determine if a new narrowband image has been loaded
        self.user_changed_narrowband = False
        
        # Current narrowband image data and filename (starts as initial)
        self.current_narrowband_data = self.initial_narrowband_data
        self.current_narrowband_filename = self.initial_narrowband_filename

        # Add internal flags to track file selector changes
        self.continuum_path_changed = False

        # Create the UI and match its theme to Siril
        self.create_widgets()
        tksiril.match_theme_to_siril(self.root, self.siril)

    def _browse_continuum_file(self):
        """Browse for continuum file"""
        filename = filedialog.askopenfilename(
            title="Select Continuum Image",
            initialdir=self.siril.get_siril_wd(),
            filetypes=filefilter
        )
        if filename:
            # Mark cache as changed if filename changes
            if self.cached_continuum_path != filename:
                self.cached_continuum_path = filename
                self.cached_continuum_data = None
                self.cached_continuum_median = None
                self.continuum_path_changed = True
            self.continuum_path_var.set(filename)

    def create_widgets(self):
        """Create the GUI's widgets, connect signals etc. """
        # Main frame with no padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Continuum Subtraction",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Image selection frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding=10)
        image_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add a label explaining narrowband is from current image
        nb_info_frame = ttk.Frame(image_frame)
        nb_info_frame.pack(fill=tk.X, pady=5)
        
        # Label showing which narrowband image is being used
        self.narrowband_info_var = tk.StringVar()
        self.narrowband_info_var.set(f"Narrowband Image: {self.initial_narrowband_filename}")
        ttk.Label(nb_info_frame, textvariable=self.narrowband_info_var).pack(side=tk.LEFT)

        # Continuum file selector
        cont_frame = ttk.Frame(image_frame)
        cont_frame.pack(fill=tk.X, pady=5)

        ttk.Label(cont_frame, text="Continuum Image:").pack(side=tk.LEFT)

        self.continuum_path_var = tk.StringVar(self.root, value="")
        # Track changes to entry field
        self.continuum_path_var.trace_add("write", self._on_continuum_path_change)

        continuum_entry = ttk.Entry(
            cont_frame,
            textvariable=self.continuum_path_var,
            width=40
        )
        continuum_entry.pack(side=tk.LEFT, padx=(5, 5), expand=True)

        ttk.Button(
            cont_frame,
            text="Browse",
            command=self._browse_continuum_file,
            style="TButton"
        ).pack(side=tk.LEFT)

        # Use current image button
        use_current_btn = ttk.Button(
            cont_frame,
            text="Reload NB",
            command=self.reset_to_current_image,
            style="TButton"
        )
        use_current_btn.pack(side=tk.LEFT, padx=(5, 0))
        tksiril.create_tooltip(use_current_btn,
                             "Reload the currently loaded image in Siril to use as narrowband "
                             "(otherwise the image that was loaded when the script was started "
                             "will be used)")

        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=10)

        # Subtractionfactor frame - converted from display to editable
        scale_frame = ttk.Frame(options_frame)
        scale_frame.pack(fill=tk.X, pady=5)

        ttk.Label(scale_frame, text="SubtractionFactor:").pack(side=tk.LEFT)

        self.scale_factor_var = tk.StringVar(value="Auto")
        scale_entry = ttk.Entry(
            scale_frame,
            textvariable=self.scale_factor_var,
            width=10
        )
        scale_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(scale_frame,
                 text="(Enter a value or leave as 'Auto' for optimization)").pack(side=tk.LEFT)

        # Show plot option
        self.show_plot_var = tk.BooleanVar(value=True)
        plot_check = ttk.Checkbutton(
            options_frame,
            text="Show optimization plot",
            variable=self.show_plot_var
        )
        plot_check.pack(anchor=tk.W, pady=5)

        # Output type selection
        output_frame = ttk.Frame(options_frame)
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="Output Type:").pack(side=tk.LEFT, padx=(0, 10))

        self.output_type_var = tk.StringVar(value="subtract")
        subtract_radio = ttk.Radiobutton(
            output_frame,
            text="Continuum-subtracted Narrowband",
            variable=self.output_type_var,
            value="subtract"
        )
        subtract_radio.pack(side=tk.LEFT, padx=(0, 10))

        enhance_radio = ttk.Radiobutton(
            output_frame,
            text="Enhanced Continuum",
            variable=self.output_type_var,
            value="enhance"
        )
        enhance_radio.pack(side=tk.LEFT)

        # Enhancement factor (only used if enhance is selected)
        enhance_frame = ttk.Frame(options_frame)
        enhance_frame.pack(fill=tk.X, pady=5)

        ttk.Label(enhance_frame, text="Enhancement Factor:").pack(side=tk.LEFT)

        self.enhance_factor_var = tk.StringVar(value="1.0")
        enhance_entry = ttk.Entry(
            enhance_frame,
            textvariable=self.enhance_factor_var,
            width=10
        )
        enhance_entry.pack(side=tk.LEFT, padx=5)

        # Status message frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=400
        ).pack(fill=tk.X)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(close_btn,
                               "Close the interface and disconnect from Siril. "
                               "No changes will be made to the current image.")

        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self.apply_changes,
            style="TButton"
        )
        apply_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(apply_btn,
                               "Apply continuum subtraction to the current image. "
                               "Changes can be undone using Siril's undo function.")

    def reset_to_current_image(self):
        """Reset to use the currently loaded image as narrowband"""
        # Save current image as new narrowband
        current_filename = self.siril.get_image_filename()
        
        # Only update if the current image is different from what we're already using
        if current_filename != self.current_narrowband_filename:
            # Check if it's different from our continuum image
            if current_filename == self.cached_continuum_path:
                messagebox.showwarning("Warning", 
                                      "The current image appears to be your selected continuum image.\n"
                                      "Using the same image for both narrowband and continuum is not recommended.")
            
            new_narrowband_data = self.siril.get_image_pixeldata()
            shape = self.siril.get_image_shape()
            if shape[0] != 1:
                self.siril.error_messagebox("Images must be mono. Please load a mono "
                                            "narrowband image in Siril.")
                return
            self.user_changed_narrowband = True
            self.current_narrowband_data = new_narrowband_data
            self.narrowband_header = self.siril.get_image_fits_header()
            self.current_narrowband_filename = current_filename
            self.narrowband_info_var.set(f"Narrowband Image: {os.path.basename(current_filename)}")
            self.siril.log(f"Now using current image as narrowband: {os.path.basename(current_filename)}", 
                          s.LogColor.GREEN)
        else:
            self.siril.log("Already using this image as narrowband", s.LogColor.SALMON)

    def _on_continuum_path_change(self, *args):
        """Callback for when continuum path changes in the entry field"""
        current_path = self.continuum_path_var.get()
        if current_path != self.cached_continuum_path:
            self.continuum_path_changed = True
            self.cached_continuum_path = current_path
            self.cached_continuum_data = None
            self.cached_continuum_median = None

    def check_narrowband_image(self):
        """
        Determine which narrowband image to use.
        - If user explicitly changed narrowband, use that
        - If not, use the initial narrowband image
        """
        if self.user_changed_narrowband:
            # User manually selected a different narrowband image
            return self.current_narrowband_data
        else:
            # Use the initial narrowband image (from first run)
            return self.initial_narrowband_data

    def load_image_data(self, image_path):
        """Load continuum image data with caching"""
        # Check if we can use cached data
        if not self.continuum_path_changed and self.cached_continuum_data is not None:
            self.status_var.set("Using cached continuum image")
            return self.cached_continuum_data

        # Before loading continuum, check if current image is different from our narrowband
        # and might be worth saving
        current_filename = self.siril.get_image_filename()
        if (current_filename != self.current_narrowband_filename and 
            current_filename != self.initial_narrowband_filename and
            current_filename != self.cached_continuum_path and
            not self.user_changed_narrowband):
            # User has loaded a different image that might be a new narrowband
            response = messagebox.askyesno("New Image Detected", 
                                         f"The current image '{os.path.basename(current_filename)}' "
                                         f"is different from your narrowband image. "
                                         f"Would you like to use this as your narrowband image instead?")
            if response:
                self.user_changed_narrowband = True
                self.current_narrowband_data = self.siril.get_image().data
                self.current_narrowband_filename = current_filename
                self.narrowband_info_var.set(f"Narrowband Image: {os.path.basename(current_filename)}")
                self.siril.log(f"Using new image as narrowband: {os.path.basename(current_filename)}", 
                              s.LogColor.GREEN)

        # Now load the continuum image
        self.status_var.set("Loading continuum image...")
        self.root.update_idletasks()  # Update the UI to show loading message

        # Remember what narrowband we were using before loading continuum
        temp_nb_file = self.current_narrowband_filename if self.user_changed_narrowband else self.initial_narrowband_filename

        # Load continuum
        self.siril.cmd("load", f"\"{image_path}\"")
        image = self.siril.get_image()
        image_data = image.data
        if image_data.shape != self.current_narrowband_data.shape:
            self.siril.error_messagebox("Image sizes must match. Please load a "
                                        "matching and aligned continuum image in Siril.")
            return
 
        # Cache the loaded data and reset change flags
        self.cached_continuum_path = image_path
        self.cached_continuum_data = image_data
        self.continuum_path_changed = False

        return image_data

    def get_selection_stats_cached(self, selection):
        """Get selection stats with caching for continuum image"""
        if (not self.continuum_path_changed and
            self.cached_selection == selection and
            self.cached_continuum_median is not None and
            self.cached_continuum_data is not None):
            self.status_var.set("Using cached selection stats")
            return self.cached_continuum_median

        # Need to calculate stats
        self.status_var.set("Calculating selection statistics...")
        self.root.update_idletasks()

        c_stats = self.siril.get_selection_stats(selection, 0)
        c_median = c_stats.median

        # Cache the result
        self.cached_selection = selection
        self.cached_continuum_median = c_median

        return c_median

    def apply_changes(self):
        """ Get the necessary variables from the GUI and call the algorithm """
        try:
            if not self.siril.is_image_loaded():
                messagebox.showerror("Error", "No image loaded in Siril")
                return

            # Get the appropriate narrowband image data
            narrowband_data = self.check_narrowband_image()
            
            # Check if continuum path is provided
            continuum_path = self.continuum_path_var.get()
            if not continuum_path:
                messagebox.showerror("Error", "Please select a continuum image")
                return

            # Get the selected region for estimating uniformity
            selection = self.siril.get_siril_selection()
                
            using_whole_image = False
            if selection[2] <= 0 or selection[3] <= 0:
                shape = self.siril.get_image_shape()
                selection = (0, 0, shape[2] - 1, shape[1] - 1)
                using_whole_image = True

            # Load continuum image and get stats (with caching)
            continuum_data = self.load_image_data(continuum_path)

            # Get continuum image stats from the selection (with caching)
            c_median = self.get_selection_stats_cached(selection)

            if not continuum_data.shape == narrowband_data.shape:
                self.siril.error_messagebox("Error: images must be the same size!")
                return

            # Determine subtraction factor
            scale_factor = None
            try:
                # Check if user provided a manual subtraction factor
                scale_str = self.scale_factor_var.get()
                if scale_str.lower() != "auto":
                    scale_factor = float(scale_str)
                    self.siril.log(f"Using manual subtraction factor: {scale_factor}", s.LogColor.GREEN)
                    self.status_var.set(f"Using manual subtraction factor: {scale_factor}")
            except ValueError:
                self.siril.log("Invalid subtraction factor provided, using optimization instead", s.LogColor.SALMON)
                self.status_var.set("Invalid subtraction factor provided, using optimization instead")
                scale_factor = None

            # If no valid subtraction factor was provided, calculate optimal value
            if scale_factor is None:
                # Check if a selection was made or if we are using the entire image
                if using_whole_image:
                    self.siril.log("No selection made: using entire image. This is probably not optimum. "
                        "It is recommended to make a generous selection around the object of interest.", s.LogColor.SALMON)

                self.status_var.set("Optimizing subtraction factor...")
                self.root.update_idletasks()

                show_plot = self.show_plot_var.get()

                scale_factor = perform_continuum_subtraction(
                    narrowband_data,
                    continuum_data,
                    selection,
                    c_median,
                    self.siril,
                    plot_optimization=show_plot,
                    tk_root=self.root
                )
                self.scale_factor_var.set(f"{scale_factor:.4f}")
                self.siril.log(f"Optimized subtraction factor: {scale_factor:.4f}", s.LogColor.GREEN)
                self.status_var.set(f"Optimized subtraction factor: {scale_factor:.4f}")

            self.status_var.set("Processing image...")
            self.root.update_idletasks()

            output_type = self.output_type_var.get()

            if output_type == "subtract":
                # Subtract continuum from narrowband
                result = narrowband_data - (continuum_data - c_median) * scale_factor
                result = np.clip(result, 0, 1)
                message = f"Continuum subtraction completed with subtraction factor {scale_factor:.4f}"
            else:  # enhance
                try:
                    enhance_factor = float(self.enhance_factor_var.get())
                except ValueError:
                    enhance_factor = 1.0
                    self.siril.log("Invalid enhancement factor, using default 1.0", s.LogColor.SALMON)

                # Calculate continuum-subtracted narrowband first
                subtracted = narrowband_data - (continuum_data - c_median) * scale_factor
                subtracted = np.clip(subtracted, 0, 1)

                # Add back to continuum with enhancement factor
                result = continuum_data + subtracted * enhance_factor
                result = np.clip(result, 0, 1)
                message = f"Enhanced continuum created with subtraction factor {scale_factor:.4f} and enhancement {enhance_factor:.2f}"

            # Create a new image (avoids risk of saving over the continuum)
            ext = self.siril.get_siril_config("core", "extension")
            self.siril.cmd("new", "1", "1", "1", f"result{ext}")
            # Set the result as the current image
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(result.astype(np.float32))
            self.siril.set_image_metadata_from_header_string(self.narrowband_header)

            # Update status
            self.status_var.set(message)
            
        except SirilError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Error: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def close_dialog(self):
        """ Close dialog """
        self.root.quit()
        self.root.destroy()

def main():
    """ Main entry point """
    try:
        # Create the GUI interface
        root = ThemedTk()
        ContinuumSubtractionInterface(root)
        root.mainloop()
    except SirilError as e:
        print(f"Error initializing script: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
