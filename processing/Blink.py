# (c) Adrian Knagg-Baugh 2025
# Blink Comparator for Siril
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.0
#
VERSION = "1.0.0"

import sirilpy as s
s.ensure_installed("ttkthemes", "pillow")

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import numpy as np
from PIL import Image, ImageTk

class BlinkInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Blink Comparator for Siril - v{VERSION}")
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

        if not self.siril.is_sequence_loaded():
            self.siril.error_messagebox("No sequence loaded")
            self.close_dialog()
            return

        self.bit_depth = "Unknown"
        self.blink_speed = 1.0
        self.is_blinking = False

        # Set up zoom
        self.zoom_factor = 1.0
        self.preview_image = None
        self.blink_frames = []
        self.current_frame_index = 0

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
            text="Blink Comparator",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 10))

        # Instructions box
        instruction_frame = ttk.LabelFrame(left_frame, text="Instructions", padding=5)
        instruction_frame.pack(fill=tk.X, padx=5, pady=5)

        instructions = ttk.Label(
            instruction_frame,
            text="""
1. A sequence must be loaded to use this script.
2. Select frames using the Siril frame selector dialog. All selected frames will be blinked, memory permitting.
3. Adjust the blink speed if required.
4. Press Go to blink.
5. Press Stop to end blinking.
            """,
            wraplength=280
        )
        instructions.pack(fill=tk.X, padx=5, pady=5)

        # Parameters frame
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=5)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Blink speed slider
        blink_speed_frame = ttk.Frame(params_frame)
        blink_speed_frame.pack(fill=tk.X, pady=5)

        ttk.Label(blink_speed_frame, text="Blink duration / s:").pack(side=tk.LEFT)

        self.blink_speed = tk.DoubleVar(value=0.3)
        self.blink_speed_slider = ttk.Scale(
            blink_speed_frame,
            from_=0.1,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.blink_speed,
            length=150
        )
        self.blink_speed_slider.pack(side=tk.LEFT, padx=10, expand=True)

        self.blink_speed_label = ttk.Label(
            blink_speed_frame,
            textvariable=self.blink_speed,
            width=5
        )
        self.blink_speed_label.pack(side=tk.LEFT)
        tksiril.create_tooltip(self.blink_speed_slider, "Adjust the duration of "
                               "each frame before blinking to the next one")

        # Action buttons frame
        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=10)

        self.preview_button = ttk.Button(
            action_frame,
            text="Go!",
            command=self.blink_compare,
            style="TButton"
        )
        self.preview_button.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(self.preview_button, "Start blinking")

        self.stop_button = ttk.Button(
            action_frame,
            text="Stop",
            command=self.stop_blinking,
            style="TButton"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(self.stop_button, "Stop blinking")

        # Status label
        self.status_label = ttk.Label(left_frame, text="")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

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

    def zoom_in(self):
        """Increase zoom level"""
        if self.zoom_factor < 20.0:
            self.zoom_factor *= 1.25
            # If we're currently blinking, update the display
            if self.is_blinking and hasattr(self, 'blink_frames') and self.blink_frames:
                self.display_current_frame()

    def zoom_out(self):
        """Decrease zoom level"""
        if self.zoom_factor > 0.1:
            self.zoom_factor /= 1.25
            # If we're currently blinking, update the display
            if self.is_blinking and hasattr(self, 'blink_frames') and self.blink_frames:
                self.display_current_frame()

    def fit_to_preview(self):
        """Fit image to preview window"""
        if not hasattr(self, 'blink_frames') or not self.blink_frames:
            return

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate zoom factor to fit
        width_ratio = canvas_width / self.original_width
        height_ratio = canvas_height / self.original_height

        # Use the smaller ratio to ensure image fits completely
        self.zoom_factor = min(width_ratio, height_ratio)

        # If we're currently blinking, update the display
        if self.is_blinking:
            self.display_current_frame()

    def blink_compare(self):
        # Stop any existing blinking
        self.stop_blinking()
        
        self.status_label.config(text="Preprocessing blink data... Please wait.")
        self.root.update_idletasks()  # Force UI update
        
        frames = build_frames_list(self.siril)
        if not frames:
            self.status_label.config(text="No frames selected for blinking.")
            return
            
        self.status_label.config(text=f"Blinking {len(frames)} frames...")
        self.root.update_idletasks()  # Force UI update
        
        # Store the frames for reference
        self.blink_frames = frames
        self.current_frame_index = 0
        
        # Store original size of first frame
        self.original_width, self.original_height = frames[0].size
        
        # Start the blinking process
        self.is_blinking = True
        self.display_current_frame()
        
        # Schedule the next frame
        self.blink_timer_id = self.root.after(int(self.blink_speed.get() * 1000), self.next_frame)

    def display_current_frame(self):
        """Display the current frame on the canvas with proper zoom"""
        if not hasattr(self, 'blink_frames') or not self.blink_frames:
            return
            
        # Get the current frame
        pil_image = self.blink_frames[self.current_frame_index]
        
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
        
        # Update status label with current frame information
        self.status_label.config(text=f"Blinking: frame {self.current_frame_index + 1} of {len(self.blink_frames)}")
        
        # Force update of the preview
        self.root.update_idletasks()

    def next_frame(self):
        """Display the next frame and schedule the next update"""
        if not self.is_blinking:
            return
            
        # Move to the next frame, cycling back to the start if needed
        self.current_frame_index = (self.current_frame_index + 1) % len(self.blink_frames)
        
        # Display the new current frame
        self.display_current_frame()
        
        # Schedule the next frame update
        self.blink_timer_id = self.root.after(int(self.blink_speed.get() * 1000), self.next_frame)

    def stop_blinking(self):
        """Stop the blinking process"""
        if hasattr(self, 'blink_timer_id'):
            self.root.after_cancel(self.blink_timer_id)
            
        self.is_blinking = False
        self.status_label.config(text="Blinking stopped.")
        
    def close_dialog(self):
        """Clean up resources and close the dialog"""
        # Stop any ongoing blinking
        self.stop_blinking()
        
        # Disconnect from Siril if connected
        if hasattr(self, 'siril') and self.siril:
            try:
                self.siril.disconnect()
            except:
                pass
            
        # Close the window
        self.root.destroy()

# Build a list of the frames, converted to 8-bit Pillow images
def build_frames_list(siril):
    blinkframes = []
    sequence = siril.get_seq()
    if not sequence:
        return blinkframes
        
    for i in range(sequence.number):
        if sequence.imgparam[i].incl:
            try:
                # Get the raw frame data
                frame = siril.get_seq_frame_pixeldata(i)
                
                # Debug information
                print(f"Frame {i} shape: {frame.shape}, dtype: {frame.dtype}")
                
                # Check if we need to reshape the array
                if len(frame.shape) != 3 or frame.shape[0] == 1:
                    # Try to reshape the array based on the sequence information
                    width = sequence.rx
                    height = sequence.ry
                    channels = 3  # Assuming RGB
                    
                    # Try to reshape to proper dimensions (channels, height, width)
                    try:
                        frame = frame.reshape(channels, height, width)
                    except:
                        # If reshaping fails, try another approach
                        try:
                            # Maybe it's a flattened array
                            frame = frame.reshape(height, width, channels)
                            # Convert to channels-first format
                            frame = np.transpose(frame, (2, 0, 1))
                        except:
                            print(f"Could not reshape frame {i}")
                            continue
                
                # Convert to float64 and normalize to [0,1] range
                max_value = 65535.0 if frame.dtype == np.uint16 else 255.0
                frame = frame.astype(np.float64) / max_value
                
                # Apply MTF processing
                m = compute_optimum_m(frame, 0.2)
                apply_mtf_inplace(frame, m)
                
                # Convert back to 8-bit for display
                frame = (frame * 255).astype(np.uint8)
                
                # PIL expects (height, width, channels) for RGB
                if frame.shape[0] == 3:  # If channels-first
                    frame = np.transpose(frame, (1, 2, 0))
                
                # Convert to PIL Image
                frame = Image.fromarray(frame)
                blinkframes.append(frame)
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                continue
    return blinkframes
def compute_optimum_m(image: np.ndarray, target_mean: float) -> float:
    """
    Compute the optimum midtone value 'm' for an image.

    Parameters:
    - image: numpy array of shape (channels, height, width), normalized to [0, 1].
    - target_mean: desired mean brightness, between 0 and 1.

    Returns:
    - optimum midtone value 'm' (float between 0 and 1).
    """
    # Compute current mean across all channels and pixels
    current_mean = np.mean(image)

    # Compute numerator and denominator
    numerator = target_mean
    denominator = current_mean + (target_mean - 2 * current_mean * target_mean)

    if denominator == 0:
        return 0.5  # fallback safe value

    m = numerator / denominator
    # Clamp m to [0, 1] just in case of numerical issues
    return np.clip(m, 0.0, 1.0)

def apply_mtf_inplace(image: np.ndarray, m: float) -> None:
    """
    Apply the Midtone Transfer Function (MTF) to an image, modifying it in-place.

    Parameters:
    - image: numpy array of shape (channels, height, width), normalized to [0, 1].
    - m: midtone value, between 0 and 1.

    Returns:
    - None (the input array is modified directly).
    """
    if m <= 0:
        image.fill(0.0)
        return
    if m >= 1:
        image.fill(1.0)
        return

    # Perform the MTF calculation in-place
    np.divide(image, image * (1.0 - m) + m, out=image)

    # Clamp in-place to [0, 1]
    np.clip(image, 0.0, 1.0, out=image)

def main():
    try:
        root = ThemedTk()
        app = BlinkInterface(root)
        root.protocol("WM_DELETE_WINDOW", app.close_dialog)  # Handle window close event
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
