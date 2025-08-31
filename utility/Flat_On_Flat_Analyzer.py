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
# Graph idea taken from the Distortion3d.py script by: cec m @cissou8
#
# Description:
# Script to analyze the effectiveness of flat fields using the "flat-on-flat" technique.
# The script combines two flat masters, displays the result as a 3D map, and
# calculates statistics on the "flatness" of the resulting image.
#
# Principle of Using the Flat-on-Flat Technique:
#  - Two master flats are used, taken under similar but not identical conditions (e.g., different telescope positions or different light panels).
#  - They are divided using the formula: (Flat to be verified ÷ Flat Reference) * med(Flat Reference)
#  - The ideal result should be a uniform (flat) image: this means both flats are correcting in the same way → therefore, they are consistent.
#
# This is useful for detecting issues such as:
#  - mechanical flexures
#  - variations in flat field illumination
#  - errors in the master flat (e.g., improperly subtracted bias)
#
# https://arciereceleste.it/articoli/120-amatissimi-flat-field
#
# Features:
# - Loads two flat masters.
# - Generates a combined image using a custom formula (if implemented).
# - 3D visualization of the resulting image surface.
# - Space for statistical analysis.
#
# Versions:
# 1.0.0 - Initial release
# 1.0.1 - Better filedialog for Linux
# 1.0.2 - Added contact information
# 1.0.3 - Bug fix
#         Added a box with a guide to interpreting the results
# 1.0.4 - Several improvements have been made for visual assessment
#

VERSION = "1.0.4"

# --- Core Imports ---
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox

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

    # Better filedialog for Linux
    if s.check_module_version(">=0.6.0") and sys.platform.startswith("linux"):
        import sirilpy.tkfilebrowser as filedialog
    else:
        from tkinter import filedialog

    # Import Siril GUI related components
    from sirilpy import tksiril, SirilError

    s.ensure_installed("ttkthemes", "numpy", "astropy", "matplotlib")
    from ttkthemes import ThemedTk

    # --- Imports for Plotting ---
    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

except ImportError:
    SIRIL_ENV = False

# --- Custom Toolbar Class ---
class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent):
        # Definiamo una vista 3D di default per il reset
        self.default_view = {'elev': 30, 'azim': -60}

        super().__init__(canvas, parent)
        
        # Top view button
        button_top_view = ttk.Button(self, text="Top View", command=self.set_top_view)
        # Inserts the left button into the toolbar
        button_top_view.pack(side=tk.LEFT, padx=5)

        # Side X view button
        button_side_x_view = ttk.Button(self, text="Side X View", command=self.set_side_view_x)
        button_side_x_view.pack(side=tk.LEFT, padx=2)

        # Side Y view button
        button_side_y_view = ttk.Button(self, text="Side Y View", command=self.set_side_view_y)
        button_side_y_view.pack(side=tk.LEFT, padx=2)

    def _get_3d_ax(self):
        """ Trova e restituisce l'asse 3D dalla figura, se esiste. """
        for ax in self.canvas.figure.axes:
            if ax.name == '3d':
                return ax
        # Se non trova nessun asse 3D, non fa nulla
        self.canvas.figure.siril_app.siril.log("Nessun grafico 3D trovato nella figura.", s.LogColor.ORANGE)
        return None

    # Top view method
    def set_top_view(self):
        """ Sets the 3D plot camera to a top view. """
        ax = self._get_3d_ax()
        if ax:
            # Sets the elevation to 90 degrees (top view)
            # and the azimuth to -90 (standard orientation)
            ax.view_init(elev=90, azim=-90)
            # Asks the canvas to redraw itself
            self.canvas.draw_idle()

    # Side view method
    def set_side_view_x(self):
        """ Sets the camera to a view along the X axis. """
        ax = self._get_3d_ax()
        if ax:
            ax.view_init(elev=0, azim=-90)
            self.canvas.draw_idle()

    # Method for the Y-side view
    def set_side_view_y(self):
        """ Sets the camera to a view along the Y-axis. """
        ax = self._get_3d_ax()
        if ax:
            ax.view_init(elev=0, azim=0)
            self.canvas.draw_idle()

    def home(self):
        """ Overrides the 'home' method to also reset the 3D view. """
        # Call the original 'home' function first to reset zoom and pan.
        super().home()
        
        # Now manually set the 3D view to the stored initial view.
        ax = self._get_3d_ax()
        if ax:
            ax.view_init(elev=self.default_view['elev'], azim=self.default_view['azim'])
            
            # Redraw the canvas
            self.canvas.draw_idle()

# --- Main Application Class ---
class FlatOnFlatApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Flat-on-Flat Analyzer v{VERSION} - (c) Carlo Mollicone AstroBOH")
        
        # Variable to control the plot's data source
        self.plot_source_var = tk.StringVar(value="placeholder")
        # Cache to avoid unnecessary reloading of FITS files
        self.image_cache = {}

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

        self.style = tksiril.standard_style()
        tksiril.match_theme_to_siril(self.root, self.siril)
        
        # Center the window
        width = 1280
        height = 720
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = f'{width}x{height}+{int((screenwidth - width) / 2)}+{int((screenheight - height) / 2)}'
        self.root.geometry(alignstr)
        self.root.minsize(800, 600)

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create the main UI widgets."""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # --- Left Column (Controls) ---
        left_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(left_frame, weight=1)

        # --- File Input Frame ---
        file_frame = ttk.LabelFrame(left_frame, text="Input Files", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.master_flat_1_path = tk.StringVar()
        self.master_flat_2_path = tk.StringVar()

        # Master Flat 1 - reference
        ttk.Label(file_frame, text="Master Flat (reference):").grid(row=0, column=0, sticky="w", pady=2)
        entry1 = ttk.Entry(file_frame, textvariable=self.master_flat_1_path)
        entry1.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(file_frame, text="Browse...", command=lambda: self.browse_file(self.master_flat_1_path, "flat1")).grid(row=1, column=1, sticky="ew")

        # Master Flat 2 - to be verified
        ttk.Label(file_frame, text="Master Flat (to be verified):").grid(row=2, column=0, sticky="w", pady=(10, 2))
        entry2 = ttk.Entry(file_frame, textvariable=self.master_flat_2_path)
        entry2.grid(row=3, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(file_frame, text="Browse...", command=lambda: self.browse_file(self.master_flat_2_path, "flat2")).grid(row=3, column=1, sticky="ew")
        
        file_frame.columnconfigure(0, weight=1)

        # --- Processing Frame ---
        processing_frame = ttk.LabelFrame(left_frame, text="Processing", padding=10)
        processing_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(processing_frame, text="Pixel Math Formula:", anchor="w").pack(fill=tk.X)
        
        # Placeholder for the formula
        formula_text = "( Flat_Ver / Flat_Ref ) * med(Flat_Ref)"
        formula_label = ttk.Label(processing_frame, text=formula_text, relief="sunken", padding=5, background="#eee")
        formula_label.pack(fill=tk.X, pady=5)
        
        self.generate_button = ttk.Button(processing_frame, text="Generate Flat-on-Flat Image", command=self.generate_flat_image, style='Accent.TButton')
        self.generate_button.pack(fill=tk.X, pady=(10, 0))
        self.generate_button.config(state=tk.DISABLED)

        # Plot Source Selection Frame
        plot_source_frame = ttk.LabelFrame(left_frame, text="Chart Data Source", padding=10)
        plot_source_frame.pack(fill=tk.X, pady=10)

        self.rb_flat1 = ttk.Radiobutton(plot_source_frame, text="Data from Master Flat Reference", variable=self.plot_source_var, value="flat1", command=self.update_plot_view)
        self.rb_flat1.pack(fill=tk.X, anchor="w")

        self.rb_flat2 = ttk.Radiobutton(plot_source_frame, text="Data from Master Flat to be Verified", variable=self.plot_source_var, value="flat2", command=self.update_plot_view)
        self.rb_flat2.pack(fill=tk.X, anchor="w")
        
        self.rb_result = ttk.Radiobutton(plot_source_frame, text="Data from Flat-on-Flat Result", variable=self.plot_source_var, value="result", command=self.update_plot_view)
        self.rb_result.pack(fill=tk.X, anchor="w")
        
        self.plot_source_var.set("placeholder") # Start without selection

        # Disable radio buttons until files are loaded
        self.rb_flat1.config(state=tk.DISABLED)
        self.rb_flat2.config(state=tk.DISABLED)
        self.rb_result.config(state=tk.DISABLED)

        scrollable_left_frame = tksiril.ScrollableFrame(left_frame)
        scrollable_left_frame.pack(fill=tk.BOTH, expand=True)
        scrollable_left_frame.add_mousewheel_binding()

        # --- Interpretation Frame ---
        self.interpretation_frame = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="Interpretation of Results", padding=10)
        
        interpretation_text = (
            "Use Standard Deviation (Std Dev) to judge the test:\n\n"
            "• Std Dev < 0.0015 (0.15%): Excellent.\n Maximum consistency, stable system.\n\n"
            "• 0.0015 to 0.0030 (0.15% - 0.3%): Good.\n Minimal differences, usable flats.\n\n"
            "• Std Dev > 0.0030 (0.3%): To be investigated.\n Significant difference. Check for flexure,\n lighting, or calibration issues."
        )

        interp_label = ttk.Label(
            self.interpretation_frame,
            text=interpretation_text,
            wraplength=320,
            justify="left",
            anchor="nw",
            font=("TkDefaultFont", 8)
        )
        interp_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 0))

        # I "package" it and immediately hide it.
        # Technically, this is useless because .pack_forget() will cause Tkinter to lose memory of this configuration.
        # Therefore, we will have to re-issue the complete and correct layout instructions when we display it.
        self.interpretation_frame.pack(fill=tk.X, pady=10, padx=(0, 10))
        # It will only be made visible after the calculation.
        self.interpretation_frame.pack_forget()

        # Instructions frame
        self.Instructions_frame = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="Instructions", padding=10)
        self.Instructions_frame.pack(fill=tk.X, pady=10, padx=(0, 10))

        # Instructions
        instructions_text = (
            "Principle of Using the Flat-on-Flat Technique:\n"
            " - Two master flats are used, taken under similar but not identical conditions (e.g., different telescope positions or different light panels).\n"
            " - They are divided using the formula.\n"
            " - The ideal result should be a uniform (flat) image: this means both flats are correcting in the same way → therefore, they are consistent.\n\n"
            
            "This is useful for detecting issues such as:\n"
            " - Mechanical flexures.\n"
            " - Variations in flat field illumination.\n"
            " - Errors in the master flat (e.g., improperly subtracted bias).\n"
        )

        label = ttk.Label(
            self.Instructions_frame,
            text=instructions_text,
            wraplength=320,
            justify="left",
            anchor="nw",
            font=("TkDefaultFont", 8)
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 0))

        # --- Right Column (3D Plot) ---
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        self.fig = plt.Figure(figsize=(8, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Creating the canvas for the chart
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)

        # Creating the toolbar associated with the canvas
        toolbar = CustomToolbar(self.canvas, right_frame)
        toolbar.update()

        # Positioning the elements: first the toolbar at the bottom,
        # then the canvas fills the remaining space at the top.
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Delays drawing the graph by 100ms to give the UI time to settle
        self.root.after(100, self.plot_placeholder) # Display a sample chart at startup

    def browse_file(self, string_var, source_tag):
        """Opens a dialog file, updates the text box, and activates the relevant controls."""
        filetypes = [('FITS files', '*.fits *.fit'), ('All files', '*.*')]
        initial_dir = self.siril.get_siril_wd()

        if source_tag == "flat1":
            filename = filedialog.askopenfilename(title='Select Master Flat (reference)', filetypes=filetypes, initialdir=initial_dir)
        elif source_tag == "flat2":
            filename = filedialog.askopenfilename(title='Select Master Flat (to be verified)', filetypes=filetypes, initialdir=initial_dir)

        if filename:
            string_var.set(filename)

            # Removes data from the previous file from the cache to force reloading.
            self.image_cache.pop(source_tag, None)
            # If a flat is changed, the calculation result is also invalid.
            self.image_cache.pop("result", None)
            # Disables the result radio button because it is no longer valid.
            self.rb_result.config(state=tk.DISABLED)
            # Hides the interpretation pane because the result is no longer valid
            self.interpretation_frame.pack_forget()

            # Enable the corresponding radio button
            if source_tag == "flat1":
                self.rb_flat1.config(state=tk.NORMAL)
            elif source_tag == "flat2":
                self.rb_flat2.config(state=tk.NORMAL)
            
            # If both files are selected, enable the generate button.
            if self.master_flat_1_path.get() and self.master_flat_2_path.get():
                self.generate_button.config(state=tk.NORMAL)

            # Automatically select the radio button to display the newly uploaded file.
            self.plot_source_var.set(source_tag)
            self.update_plot_view()

    def update_plot_view(self):
        """Updates the chart based on the selected radio button."""
        source = self.plot_source_var.get()
        if source == "placeholder": return
        
        if source == "flat1":
            self.siril.log(f"Graphic update for source: Master Flat (reference)", s.LogColor.BLUE)
        elif source == "flat2":
            self.siril.log(f"Graphic update for source: Master Flat (to be verified)", s.LogColor.BLUE)

        data_to_plot = self.image_cache.get(source)
        path = ""
        title = ""

        if data_to_plot is None:
            try:
                if source == "flat1":
                    path = self.master_flat_1_path.get()
                    title = f"Master Flat Reference: {os.path.basename(path)}"
                elif source == "flat2":
                    path = self.master_flat_2_path.get()
                    title = f"Master Flat to be verified: {os.path.basename(path)}"
                elif source == "result":
                    # If the result is not cached, it means it has not been calculated yet.
                    self.siril.log("Result must be generated first.", s.LogColor.RED)
                    self.plot_placeholder("Result not yet generated")
                    return
                
                with fits.open(path) as hdul:
                    data_to_plot = hdul[0].data.astype(np.float32)
                self.image_cache[source] = data_to_plot
                self.siril.log(f"Data loaded and cached by: {path}", s.LogColor.GREEN)

            except Exception as e:
                self.siril.log(f"Error loading FITS file: {e}", s.LogColor.RED)
                self.plot_placeholder(f"Error loading file:\n{os.path.basename(path)}")
                return
        else: # Data already cached
            if source == "flat1": title = f"Master Flat Reference (cache)"
            elif source == "flat2": title = f"Master Flat to be verified (cache)"
            elif source == "result": title = "Flat-on-Flat Result (cache)"

            if source == "flat1":
                self.siril.log(f"Data for 'Master Flat (reference)' read from cache.", s.LogColor.BLUE)
            elif source == "flat2":
                self.siril.log(f"Data for 'Master Flat (to be verified)' read from cache.", s.LogColor.BLUE)

        self.plot_3d_surface(data_to_plot, title)

    def generate_flat_image(self):
        """
        Performs the Flat-on-Flat calculation using Pixel Math and updates the graph.
        """
        self.siril.log("Starting Flat-on-Flat image generation...", s.LogColor.GREEN)

        path1 = self.master_flat_1_path.get() # Flat1 (reference)
        path2 = self.master_flat_2_path.get() # Flat2 (to be verified)

        if not path1 or not path2:
            messagebox.showwarning("Missing Input", "Select both flat master files.")
            return
        
        try:
            # Clears the cache of the previous result
            self.image_cache.pop("result", None)

            # Defines the formula for Pixel Math
            formula = f"(${path2}$ / ${path1}$) * med(${path1}$)"

            # Runs the Pixel Math command
            self.siril.cmd("pm", f'"{formula}"')

            # Retrieves the resulting image data from Siril
            result_data = self.siril.get_image_pixeldata()

            # Caches the result and updates the UI
            self.image_cache['result'] = result_data
            self.siril.log("Result image calculated and cached.", s.LogColor.GREEN)

            # Show the box with the interpretation of the results
            self.interpretation_frame.pack(fill=tk.X, pady=10, padx=(0, 10), before=self.Instructions_frame)

            # After the calculation, enable and select the result radio button
            self.rb_result.config(state=tk.NORMAL)
            self.plot_source_var.set("result")
            # Update the graph to show the result
            self.update_plot_view()

        except SirilError as e: 
            error_message = f"Error executing Siril command: {e}" 
            self.siril.log(error_message, s.LogColor.RED) 
        except Exception as e: 
            error_message = f"An unexpected error occurred: {e}" 
            self.siril.log(error_message, s.LogColor.RED)

    def _rebin(self, arr, factor):
        """Reduces a 2D array by an integer factor using block averaging."""
        if factor <= 1: return arr
        new_h = arr.shape[0] // factor
        new_w = arr.shape[1] // factor
        trimmed_arr = arr[:new_h * factor, :new_w * factor]
        rebinned = trimmed_arr.reshape(new_h, factor, new_w, factor).mean(axis=3).mean(axis=1)
        return rebinned
    
    def plot_3d_surface(self, data, title):
        """Draws the 3D surface + 2D map + histogram."""
        # Pulisce la figura e imposta griglia di subplot
        self.fig.clear()

        # Rebin parameters to lighten the graph
        TARGET_RESOLUTION = 250 # Maximum resolution for the longest side of the plot

        # Handles both color (taking luminance) and monochrome images
        if data.ndim == 3:
            # Simple grayscale conversion if the image is in color
            # The channel order from Siril is C, H, W. For viewing, you may need H, W, C
            if data.shape[0] == 3: # C, H, W
                data = data.transpose(1, 2, 0) # H, W, C
                img_data = 0.299*data[:,:,0] + 0.587*data[:,:,1] + 0.114*data[:,:,2]
            else: # H, W, C
                img_data = 0.299*data[:,:,0] + 0.587*data[:,:,1] + 0.114*data[:,:,2]
        else:
            img_data = data

        h, w = img_data.shape
        if max(h, w) > TARGET_RESOLUTION:
            factor = max(h, w) // TARGET_RESOLUTION
            plot_data = self._rebin(img_data, factor)
            self.siril.log(f"Grouped data for plot with factor {factor}. From {img_data.shape} to {plot_data.shape}", s.LogColor.BLUE)
        else:
            plot_data = img_data
            
        h_plot, w_plot = plot_data.shape
        X, Y = np.meshgrid(np.arange(w_plot), np.arange(h_plot))

        # Normalization on 5–95% percentiles
        norm = Normalize(vmin=np.percentile(plot_data, 5), vmax=np.percentile(plot_data, 95))

        # --- Creazione layout ---
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[3, 1])
        ax2d = self.fig.add_subplot(gs[0, 0])
        ax3d = self.fig.add_subplot(gs[0, 1], projection='3d')
        axhist = self.fig.add_subplot(gs[1, :])

        # --- 3D Surface ---
        surf = ax3d.plot_surface(X, Y, plot_data, cmap=cm.viridis, norm=norm, linewidth=0, antialiased=False, shade=False)
        ax3d.set_title(title, fontsize=12)
        ax3d.set_xlabel('X [pixel]')
        ax3d.set_ylabel('Y [pixel]')
        ax3d.set_zlabel('Normalized value')

        # --- 2D Map ---
        im = ax2d.imshow(plot_data, cmap=cm.viridis, norm=norm, origin='lower')
        ax2d.set_title("2D Map")
        self.fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)

        # --- Histogram ---
        vals = plot_data.ravel()
        axhist.hist(vals, bins=100, color='steelblue', alpha=0.7)
        axhist.set_title("Value Distribution")
        axhist.set_xlabel("Normalized value")
        axhist.set_ylabel("Pixel count")

        # Calculate the statistics
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        p5_val = np.percentile(vals, 5)
        p95_val = np.percentile(vals, 95)

        # Lines for mean and ±1σ
        for xline, col in [(mean_val, 'red'), (mean_val - std_val, 'orange'), (mean_val + std_val, 'orange')]:
            axhist.axvline(xline, color=col, linestyle='--')

        # --- Stats ---
        stats_text = (
            f"Statistics:\n"
            f"-----------------\n"
            f"Mean: {mean_val:.4f}\n"
            f"Std Dev: {std_val:.4f}\n"
            f"5th Percentile: {p5_val:.4f}\n" 
            f"95th Percentile: {p95_val:.4f}"
        )

        # Adds text to the figure, not a specific subplot. This makes the position stable regardless of window resizing.
        # Coordinates (0.01, 0.99) place it in the top-left corner of the entire figure.
        self.fig.text(0.01, 0.99,
                      stats_text,
                      transform=self.fig.transFigure, # Use figure-relative coordinates
                      fontsize=10,
                      verticalalignment='top',
                      horizontalalignment='left',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        self.canvas.draw()

    def plot_placeholder(self, text="Select files to start analysis"):
        """Show a sample surface at startup."""
        w, h = 600, 400
        x = np.linspace(-3, 3, w)
        y = np.linspace(-3, 3, h)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2)) + np.random.normal(0, 0.1, (h,w)) * 0.2
        
        self.plot_3d_surface(Z, "3D Graph (Example just for fun)")

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
        app = FlatOnFlatApp(root)
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