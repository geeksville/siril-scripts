# (c) Cyril Richard from Franklin Marek SAS code (2025)
# NBtoRGBstars for Siril - Ported from PyQt to Siril/tkinter
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.0
#

import sirilpy as s
s.ensure_installed("ttkthemes", "pillow", "numpy", "astropy")

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import numpy as np
from PIL import Image, ImageTk
import astropy
from astropy.io import fits

VERSION = "1.0.0"

class NBtoRGBstarsInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"NB to RGB Stars - v{VERSION}")
        self.root.resizable(True, True)
        
        self.style = tksiril.standard_style()
        
        self.siril = s.SirilInterface()
        
        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            self.siril.error_messagebox("Failed to connect to Siril")
            self.close_dialog()
            return
            
        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            messagebox.showerror("Error", "Siril version requirement not met")
            self.close_dialog()
            return
        
        # Initialize image variables
        self.ha_image = None
        self.oiii_image = None
        self.sii_image = None
        self.osc_image = None
        self.combined_image = None
        self.is_mono = False
        
        # Filenames
        self.ha_filename = None
        self.oiii_filename = None
        self.sii_filename = None
        self.osc_filename = None
        
        self.original_header = None
        self.original_header_string = None
        self.bit_depth = "Unknown"
        
        # Set up zoom
        self.zoom_factor = 1.0
        self.preview_image = None
        
        # Create the UI
        tksiril.match_theme_to_siril(self.root, self.siril)
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame with paned window to allow resizing
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame for controls
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=1)
        
        # Title
        title_label = ttk.Label(
            left_frame,
            text="NB to RGB Stars",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 10))
        
        # Instructions box
        instruction_frame = ttk.LabelFrame(left_frame, text="Instructions", padding=5)
        instruction_frame.pack(fill=tk.X, padx=5, pady=5)
        
        instructions = ttk.Label(
            instruction_frame,
            text="""
1. Select Ha, OIII, and SII (optional) narrowband images, or an OSC stars-only image.
   Note: Images must be pre-aligned on stars before processing.
2. Adjust the Ha to OIII Ratio if needed.
3. Preview the combined result.
4. Send Preview to Siril.
            """,
            wraplength=280
        )
        instructions.pack(fill=tk.X, padx=5, pady=5)
        
        # Image selection frame
        image_select_frame = ttk.LabelFrame(left_frame, text="Image Selection", padding=5)
        image_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ha Image
        ha_frame = ttk.Frame(image_select_frame)
        ha_frame.pack(fill=tk.X, pady=2)
        
        self.ha_button = ttk.Button(
            ha_frame,
            text="Select Ha Image",
            command=lambda: self.load_image('Ha'),
            style="TButton"
        )
        self.ha_button.pack(side=tk.LEFT, padx=5)
        
        self.ha_label = ttk.Label(ha_frame, text="No Ha image selected")
        self.ha_label.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        # OIII Image
        oiii_frame = ttk.Frame(image_select_frame)
        oiii_frame.pack(fill=tk.X, pady=2)
        
        self.oiii_button = ttk.Button(
            oiii_frame, 
            text="Select OIII Image",
            command=lambda: self.load_image('OIII'),
            style="TButton"
        )
        self.oiii_button.pack(side=tk.LEFT, padx=5)
        
        self.oiii_label = ttk.Label(oiii_frame, text="No OIII image selected")
        self.oiii_label.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        # SII Image
        sii_frame = ttk.Frame(image_select_frame)
        sii_frame.pack(fill=tk.X, pady=2)
        
        self.sii_button = ttk.Button(
            sii_frame,
            text="Select SII Image (Optional)",
            command=lambda: self.load_image('SII'),
            style="TButton"
        )
        self.sii_button.pack(side=tk.LEFT, padx=5)
        
        self.sii_label = ttk.Label(sii_frame, text="No SII image selected")
        self.sii_label.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        # OSC Image
        osc_frame = ttk.Frame(image_select_frame)
        osc_frame.pack(fill=tk.X, pady=2)
        
        self.osc_button = ttk.Button(
            osc_frame,
            text="Select OSC Stars Image (Optional)",
            command=lambda: self.load_image('OSC'),
            style="TButton"
        )
        self.osc_button.pack(side=tk.LEFT, padx=5)
        
        self.osc_label = ttk.Label(osc_frame, text="No OSC image selected")
        self.osc_label.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=5)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ha to OIII Ratio slider
        ratio_frame = ttk.Frame(params_frame)
        ratio_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ratio_frame, text="Ha to OIII Ratio:").pack(side=tk.LEFT)
        
        self.ha_to_oiii_ratio = tk.DoubleVar(value=0.3)
        self.ha_to_oiii_slider = ttk.Scale(
            ratio_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.ha_to_oiii_ratio,
            length=150
        )
        self.ha_to_oiii_slider.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.ha_to_oiii_label = ttk.Label(
            ratio_frame,
            textvariable=self.ha_to_oiii_ratio,
            width=5
        )
        self.ha_to_oiii_label.pack(side=tk.LEFT)
        tksiril.create_tooltip(self.ha_to_oiii_slider, "Adjust the ratio of Ha to OIII in the green channel")
        
        # Star Stretch options
        stretch_frame = ttk.Frame(params_frame)
        stretch_frame.pack(fill=tk.X, pady=5)
        
        self.enable_star_stretch = tk.BooleanVar(value=True)
        self.star_stretch_checkbox = ttk.Checkbutton(
            stretch_frame,
            text="Enable Star Stretch",
            variable=self.enable_star_stretch,
            command=self.toggle_stretch_controls,
            style="TCheckbutton"
        )
        self.star_stretch_checkbox.pack(anchor=tk.W)
        tksiril.create_tooltip(self.star_stretch_checkbox, "Apply a non-linear stretch to enhance stars")
        
        # Stretch factor slider
        self.stretch_frame = ttk.Frame(params_frame)
        self.stretch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.stretch_frame, text="Stretch Factor:").pack(side=tk.LEFT)
        
        self.stretch_factor = tk.DoubleVar(value=5.0)
        self.stretch_slider = ttk.Scale(
            self.stretch_frame,
            from_=0.0,
            to=8.0,
            orient=tk.HORIZONTAL,
            variable=self.stretch_factor,
            length=150
        )
        self.stretch_slider.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.stretch_label = ttk.Label(
            self.stretch_frame,
            textvariable=self.stretch_factor,
            width=5
        )
        self.stretch_label.pack(side=tk.LEFT)
        tksiril.create_tooltip(self.stretch_slider, "Adjust the intensity of the star stretch")
        
        # Metadata options
        metadata_frame = ttk.Frame(params_frame)
        metadata_frame.pack(fill=tk.X, pady=5)

        self.copy_metadata = tk.BooleanVar(value=True)
        self.metadata_checkbox = ttk.Checkbutton(
            metadata_frame,
            text="Copy Metadata from Source Image",
            variable=self.copy_metadata,
            style="TCheckbutton"
        )
        self.metadata_checkbox.pack(anchor=tk.W)
        tksiril.create_tooltip(self.metadata_checkbox, "Transfer FITS metadata from source image to combined result")

        # Action buttons frame
        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=10)

        self.preview_button = ttk.Button(
            action_frame,
            text="Preview Combined Image",
            command=self.preview_combine,
            style="TButton"
        )
        self.preview_button.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(self.preview_button, "Generate a preview of the combined image")
        
        self.send_to_siril_button = ttk.Button(
            action_frame,
            text="Send Preview to Siril",
            command=self.send_to_siril_preview,
            style="TButton"
        )
        self.send_to_siril_button.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(self.send_to_siril_button, "Send the combined image to Siril's preview window")
        
        # Status label
        self.status_label = ttk.Label(left_frame, text="")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Footer
        footer_label = ttk.Label(
            left_frame,
            text="Written by Franklin Marek\nSiril port by Cyril Richard\nwww.setiastro.com",
            justify=tk.CENTER
        )
        footer_label.pack(pady=10)
        
        # Right frame for image preview
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # Zoom controls frame
        zoom_frame = ttk.Frame(right_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        zoom_in_btn = ttk.Button(
            zoom_frame,
            text="Zoom In",
            command=self.zoom_in,
            style="TButton"
        )
        zoom_in_btn.pack(side=tk.LEFT, padx=5)
        
        zoom_out_btn = ttk.Button(
            zoom_frame,
            text="Zoom Out",
            command=self.zoom_out,
            style="TButton"
        )
        zoom_out_btn.pack(side=tk.LEFT, padx=5)
        
        fit_btn = ttk.Button(
            zoom_frame,
            text="Fit to Preview",
            command=self.fit_to_preview,
            style="TButton"
        )
        fit_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for image preview with scrollbars
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a canvas with scrollbars
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        # Configure the canvas
        self.canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Set default size
        self.canvas.config(width=600, height=400)
        
        # Pack the canvas and scrollbars
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add mouse event bindings for panning
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.mouse_wheel)    # Linux scroll down

    def toggle_stretch_controls(self):
        if self.enable_star_stretch.get():
            self.stretch_frame.pack(fill=tk.X, pady=5)
        else:
            self.stretch_frame.pack_forget()
    
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
    
    def pan_image(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def mouse_wheel(self, event):
        # Handle zoom with mouse wheel
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom_in()
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom_out()
    
    def load_image(self, image_type):
        """Load a FITS image from file using Astropy"""
        try:
            # Get the current working directory from Siril
            current_wd = self.siril.get_siril_wd() or os.path.expanduser("~")
            
            # Open file dialog in Siril's current working directory
            filename = filedialog.askopenfilename(
                title=f"Select {image_type} FITS Image File",
                initialdir=current_wd,
                filetypes=[("FITS files", "*.fits *.fit *.fts")]
            )
            
            if not filename:
                return  # User canceled
            
            # Open the FITS file with Astropy
            with fits.open(filename) as image:
                
                # Get the image data
                image_data = image[0].data
                
                # Get the header
                header = image[0].header
                
                # Convert header to string for later use with metadata copying
                header_string = header.tostring(sep='\n')

                # Normalize data type to float32 if needed
                if image_data.dtype not in (np.uint16, np.float32):
                    image_data = image_data.astype(np.float32)

                # Flip the image vertically to make it bottom-up
                image_data = np.flipud(image_data)
                
                # Debug print to understand the data structure
                print(f"Image data shape: {image_data.shape}")
                print(f"Image data type: {image_data.dtype}")


                # Ensure the data is in a 2D or 3D format
                if image_data.ndim == 2:
                    # Mono image: add channel dimension
                    image_data = image_data[np.newaxis, :, :]
                elif image_data.ndim == 3:
                    # Check if channels are first or last
                    if image_data.shape[0] in [1, 3]:
                        # Channels first - keep as is
                        pass
                    elif image_data.shape[2] in [1, 3]:
                        # Channels last - transpose to channels first
                        image_data = np.transpose(image_data, (2, 0, 1))
                
                if image_data.shape[0] in [1, 3]:
                    image_data = np.transpose(image_data, (1, 2, 0))
                
                # Determine if it's mono
                is_mono = image_data.ndim == 2 or (image_data.ndim == 3 and image_data.shape[2] == 1)
                
                # If mono, ensure 2D
                if is_mono:
                    image_data = image_data.squeeze()
            
            # Store the image data in the appropriate variable
            if image_type == 'Ha':
                self.ha_image = image_data
                self.ha_filename = filename
                self.ha_label.config(text=f"Loaded: {os.path.basename(filename)}")
                
                # Store metadata from Ha image
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono
            
            elif image_type == 'OIII':
                self.oiii_image = image_data
                self.oiii_filename = filename
                self.oiii_label.config(text=f"Loaded: {os.path.basename(filename)}")

                # If Ha not loaded yet, use OIII metadata as source
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono

            elif image_type == 'SII':
                self.sii_image = image_data
                self.sii_filename = filename
                self.sii_label.config(text=f"Loaded: {os.path.basename(filename)}")

                # If no metadata source yet, use SII
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono

            elif image_type == 'OSC':
                self.osc_image = image_data
                self.osc_filename = filename
                self.osc_label.config(text=f"Loaded: {os.path.basename(filename)}")

                # If no metadata source yet, use OSC
                if self.original_header is None:
                    self.original_header = header
                    self.original_header_string = header_string
                    self.bit_depth = "32-bit"
                    self.is_mono = is_mono
            
            self.status_label.config(text=f"{image_type} FITS image loaded successfully")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {image_type} FITS image: {str(e)}")
            print(f"Error loading {image_type} FITS image: {str(e)}")
            # Include the traceback for more detailed error information
            import traceback
            traceback.print_exc()
    
    def preview_combine(self):
        """Generate a preview of the combined image"""
        # Check if required images are loaded
        if not ((self.ha_image is not None and self.oiii_image is not None) or (self.osc_image is not None)):
            messagebox.showwarning("Missing Images", "Please load Ha and OIII images, or an OSC image")
            return
        
        # Update status
        self.status_label.config(text="Processing image... Please wait.")
        self.root.update_idletasks()
        
        try:
            # Get parameters
            ha_to_oiii_ratio = self.ha_to_oiii_ratio.get()
            enable_star_stretch = self.enable_star_stretch.get()
            stretch_factor = self.stretch_factor.get()
            
            # Process the image
            combined_image = self.process_image(
                self.ha_image, 
                self.oiii_image, 
                self.sii_image, 
                self.osc_image, 
                ha_to_oiii_ratio,
                enable_star_stretch,
                stretch_factor
            )
            
            # Store the result
            self.combined_image = combined_image
            
            # Update preview
            self.update_preview(combined_image)
            
            # Update status
            self.status_label.config(text="Preview generated successfully")
        
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
    
    def process_image(self, ha_image, oiii_image, sii_image, osc_image, ha_to_oiii_ratio, enable_star_stretch, stretch_factor):
        """Process the images to create a combined RGB image"""
        # Function to preprocess and ensure narrowband images are properly formatted
        def preprocess_narrowband(img):
            if img is None:
                return None
                
            if img.dtype in (np.uint16, np.int16):
                img_normalized = img.astype(np.float32) / 65535.0
                print(f"Normalized 16-bit image to float32 (0-1) range")
            else:
                img_normalized = img.astype(np.float32)
                img_normalized = np.clip(img_normalized, 0, 1)

            if isinstance(img_normalized, np.ndarray) and img_normalized.ndim == 3 and img_normalized.shape[2] == 3:
                # Convert to grayscale using luminance formula
                return 0.299 * img_normalized[..., 0] + 0.587 * img_normalized[..., 1] + 0.114 * img_normalized[..., 2]
            
            return img_normalized
        
        # Preprocess images
        ha_processed = preprocess_narrowband(ha_image)
        oiii_processed = preprocess_narrowband(oiii_image)
        sii_processed = preprocess_narrowband(sii_image)
        
        if osc_image is not None:
            if osc_image.dtype in (np.uint16, np.int16):
                osc_processed = osc_image.astype(np.float32) / 65535.0
                print(f"Normalized 16-bit OSC image to float32 (0-1) range")
            else:
                osc_processed = osc_image.astype(np.float32)
                osc_processed = np.clip(osc_processed, 0, 1)

            # Use OSC image as base, enhance with narrowband data if available
            r_channel = osc_processed[..., 0]
            g_channel = osc_processed[..., 1]
            b_channel = osc_processed[..., 2]
            
            # Enhance with narrowband if available
            r_combined = 0.5 * r_channel + 0.5 * (sii_processed if sii_processed is not None else r_channel)
            g_combined = ha_to_oiii_ratio * (ha_processed if ha_processed is not None else g_channel) + \
                        (1 - ha_to_oiii_ratio) * g_channel
            b_combined = oiii_processed if oiii_processed is not None else b_channel
        else:
            # Using narrowband images only
            r_combined = 0.5 * ha_processed + 0.5 * (sii_processed if sii_processed is not None else ha_processed)
            g_combined = ha_to_oiii_ratio * ha_processed + (1 - ha_to_oiii_ratio) * oiii_processed
            b_combined = oiii_processed
        
        # Normalize combined channels
        r_combined = np.clip(r_combined, 0, 1)
        g_combined = np.clip(g_combined, 0, 1)
        b_combined = np.clip(b_combined, 0, 1)
        
        # Stack channels to create RGB image - tout en float32 pour le traitement
        combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)
        
        # Apply star stretch if enabled
        if enable_star_stretch:
            combined_image = self.apply_star_stretch(combined_image, stretch_factor)
        
        # Apply SCNR (remove green cast)
        combined_image = self.apply_scnr(combined_image)

        return combined_image
    
    def apply_star_stretch(self, image, stretch_factor):
        """Apply non-linear stretch to enhance stars"""
        # Ensure input is in [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Apply the formula: (a^b * x) / ((a^b - 1) * x + 1)
        # where a=3, b=stretch_factor
        a = 3.0
        b = stretch_factor
        stretched = ((a ** b) * image) / (((a ** b) - 1) * image + 1)
        
        return np.clip(stretched, 0, 1)
    
    def apply_scnr(self, image):
        """Apply SCNR (Subtractive Chromatic Noise Reduction) to remove green cast"""
        # Extract channels
        r_channel = image[..., 0]
        g_channel = image[..., 1]
        b_channel = image[..., 2]
        
        # Apply average-neutral SCNR
        max_rb = np.maximum(r_channel, b_channel)
        mask = g_channel > max_rb
        g_channel[mask] = max_rb[mask]
        
        # Update green channel in the image
        image[..., 1] = g_channel
        
        return image
    
    def update_preview(self, image):
        """Update the preview display with the processed image"""
        if image is None:
            return
            
        # Convert to 8-bit for display
        preview_image = (image * 255).astype(np.uint8)
        
        # Create PIL image
        pil_image = Image.fromarray(preview_image)
        
        # Store original size
        self.original_width, self.original_height = pil_image.size
        
        # Apply zoom
        zoomed_width = int(self.original_width * self.zoom_factor)
        zoomed_height = int(self.original_height * self.zoom_factor)
        
        if self.zoom_factor != 1.0:
            pil_image = pil_image.resize((zoomed_width, zoomed_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.preview_image = ImageTk.PhotoImage(pil_image)
        
        # Clear previous image
        self.canvas.delete("all")
        
        # Create image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_image)
        
        # Configure scrollregion
        self.canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))
        
        # Force update of the preview
        self.root.update_idletasks()
    
    def zoom_in(self):
        """Increase zoom level"""
        if self.zoom_factor < 20.0:
            self.zoom_factor *= 1.25
            if self.combined_image is not None:
                self.update_preview(self.combined_image)
    
    def zoom_out(self):
        """Decrease zoom level"""
        if self.zoom_factor > 0.1:
            self.zoom_factor /= 1.25
            if self.combined_image is not None:
                self.update_preview(self.combined_image)
    
    def fit_to_preview(self):
        """Fit image to preview window"""
        if self.combined_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate zoom factor to fit
        width_ratio = canvas_width / self.original_width
        height_ratio = canvas_height / self.original_height
        
        # Use the smaller ratio to ensure image fits completely
        self.zoom_factor = min(width_ratio, height_ratio)
        
        # Update preview
        self.update_preview(self.combined_image)
    
    def send_to_siril_preview(self):
        """Send the combined image to Siril's preview window directly"""
        if self.combined_image is None:
            messagebox.showwarning("No Image", "No combined image to send. Please generate a preview first.")
            return
        
        try:
            # Get image dimensions
            height, width = self.combined_image.shape[:2]

            # Ensure the image is in the correct format
            combined_data = self.combined_image
            if combined_data.dtype in (np.uint16, np.int16):
                combined_data = (combined_data / 65535.0).astype(np.float32)
            else:
                combined_data = combined_data.astype(np.float32)
            print("Converting image to 32-bit float32 for Siril")

            print(f"Output image data type: {combined_data.dtype}")
            
            # Create an empty image with the correct dimensions
            # Using cmd to create a new image with 3 channels
            self.siril.cmd("new", f"{width}", f"{height}", "3", "RGB")

            # Transpose the image back to planar format (channels, height, width)
            # since Siril expects images in this format
            siril_image_data = np.transpose(combined_data, (2, 0, 1))
            siril_image_data = np.ascontiguousarray(siril_image_data)
            siril_image_data = siril_image_data[:, ::-1, :]

            # Claim the thread to ensure safe image data transfer
            if self.siril.claim_thread():
                try:
                    # Set the pixel data directly 
                    self.siril.set_image_pixeldata(siril_image_data)
                    
                    # Apply metadata if enabled and available
                    if self.copy_metadata.get() and self.original_header_string is not None:
                        # Copy the metadata from the source image
                        try:
                            self.siril.set_image_metadata_from_header_string(self.original_header_string)
                            self.status_label.config(text=f"Image with metadata sent to Siril preview")
                            self.siril.log("Metadata copied from source image")
                        except Exception as metadata_err:
                            self.status_label.config(text="Image sent, but metadata copy failed")
                            print(f"Metadata copy error: {str(metadata_err)}")
                    else:
                        # Update status
                        self.status_label.config(text=f"Image sent to Siril preview window")
                    
                    # log to Siril console
                    self.siril.log(f"NBtoRGB stars combined image loaded in Siril preview")
                
                finally:
                    # Always release the thread
                    self.siril.release_thread()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply image to Siril: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def close_dialog(self):
        """Close the dialog"""
        
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()

def main():
    try:
        root = ThemedTk()
        app = NBtoRGBstarsInterface(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
