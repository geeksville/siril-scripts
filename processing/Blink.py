# (c) Adrian Knagg-Baugh 2025
# Blink Comparator for Siril
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.1.0
#
VERSION = "1.1.0"

import sirilpy as s
s.ensure_installed("ttkthemes", "pillow", "psutil")

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import numpy as np
import psutil
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
        self.photo_images = []
        self.canvas_image_ids = []
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
            self.update_zoom()

    def zoom_out(self):
        """Decrease zoom level"""
        if self.zoom_factor > 0.1:
            self.zoom_factor /= 1.25
            self.update_zoom()

    def update_zoom(self):
        """Update all images with the new zoom factor"""
        if not hasattr(self, 'blink_frames') or not self.blink_frames:
            return
            
        # Clear canvas and existing image references
        self.canvas.delete("all")
        self.photo_images = []
        self.canvas_image_ids = []
        
        # Calculate new dimensions
        zoomed_width = int(self.original_width * self.zoom_factor)
        zoomed_height = int(self.original_height * self.zoom_factor)
        
        # Create and add all frames to the canvas
        for i, pil_image in enumerate(self.blink_frames):
            if self.zoom_factor != 1.0:
                zoomed_image = pil_image.resize((zoomed_width, zoomed_height), Image.LANCZOS)
            else:
                zoomed_image = pil_image
                
            # Convert to PhotoImage
            photo_image = ImageTk.PhotoImage(zoomed_image)
            self.photo_images.append(photo_image)
            
            # Add image to canvas with state hidden (except first one)
            state = 'normal' if i == self.current_frame_index else 'hidden'
            image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image, state=state)
            self.canvas_image_ids.append(image_id)
        
        # Configure scrollregion
        self.canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))
        
        # Force update of the preview
        self.root.update_idletasks()

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
        
        # Update all images with new zoom
        self.update_zoom()

    def blink_compare(self):
        # Stop any existing blinking
        self.stop_blinking()
        
        self.status_label.config(text="Preprocessing blink data... Please wait.")
        self.root.update_idletasks()  # Force UI update
        
        frames = build_frames_list(self.siril)
        if not frames:
            self.status_label.config(text="No frames selected for blinking.")
            return
            
        self.status_label.config(text=f"Setting up {len(frames)} frames...")
        self.root.update_idletasks()  # Force UI update
        
        # Store the frames for reference
        self.blink_frames = frames
        self.current_frame_index = 0
        
        # Store original size of first frame
        self.original_width, self.original_height = frames[0].size
        
        # Create the PhotoImages and canvas images
        self.update_zoom()
        
        # Start the blinking process
        self.is_blinking = True
        self.update_display_state()
        
        # Schedule the next frame
        self.blink_timer_id = self.root.after(int(self.blink_speed.get() * 1000), self.next_frame)

    def update_display_state(self):
        """Update which frame is visible based on current_frame_index"""
        if not self.canvas_image_ids:
            return
            
        # Hide all images
        for img_id in self.canvas_image_ids:
            self.canvas.itemconfigure(img_id, state='hidden')
            
        # Show only the current frame
        self.canvas.itemconfigure(self.canvas_image_ids[self.current_frame_index], state='normal')
        
        # Update status label
        self.status_label.config(text=f"Blinking: frame {self.current_frame_index + 1} of {len(self.blink_frames)}")
        
        # Force update of the display
        self.root.update_idletasks()

    def next_frame(self):
        """Display the next frame and schedule the next update"""
        if not self.is_blinking:
            return
            
        # Move to the next frame, cycling back to the start if needed
        self.current_frame_index = (self.current_frame_index + 1) % len(self.blink_frames)
        
        # Update which frame is visible
        self.update_display_state()
        
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
    
    # Memory check
    available_mem = psutil.virtual_memory().available
    mem_per_frame = sequence.rx * sequence.ry * sequence.nb_layers
    tot_mem_needed = mem_per_frame * sequence.selnum
    mem_ratio = siril.get_siril_config("core", "mem_ratio")
    if tot_mem_needed > available_mem * mem_ratio:
        siril.error_messagebox("Error: insufficient memory to blink the number of "
                    "frames currently selected. Reduce the number of frames that "
                    "are included using the Siril frame selector dialog.", True)
        return blinkframes
        
    for i in range(sequence.number):
        if sequence.imgparam[i].incl:
            try:
                # Get the raw frame data
                frame = siril.get_seq_frame_pixeldata(i, preview=True)
                
                # Check if we need to reshape the array and determine if mono or color
                is_mono = False
                
                if len(frame.shape) == 2:
                    # This is a mono image (height, width)
                    is_mono = True
                    height, width = frame.shape
                    # Convert to 3D array with one channel for consistent processing
                    frame = frame.reshape(1, height, width)
                elif len(frame.shape) == 1:
                    # This is a flattened array, we need to determine if it's mono or color
                    width = sequence.rx
                    height = sequence.ry
                    
                    # Check if it's a mono image by looking at the size
                    if frame.size == width * height:
                        is_mono = True
                        frame = frame.reshape(1, height, width)
                    else:
                        # Assume it's color with 3 channels
                        channels = 3
                        try:
                            frame = frame.reshape(channels, height, width)
                        except:
                            # If reshaping fails, try another approach
                            try:
                                frame = frame.reshape(height, width, channels)
                                # Convert to channels-first format
                                frame = np.transpose(frame, (2, 0, 1))
                            except:
                                print(f"Could not reshape frame {i}")
                                continue
                elif len(frame.shape) == 3:
                    # Already a 3D array, but check if it's in the right format
                    if frame.shape[0] != 3 and frame.shape[2] == 3:
                        # Convert from (height, width, channels) to (channels, height, width)
                        frame = np.transpose(frame, (2, 0, 1))
                
                # For PIL, convert to correct format
                if is_mono:
                    # For mono, reshape to 2D
                    pil_image = Image.fromarray(frame[0], 'L')
                else:
                    # For color, convert to (height, width, channels) which PIL expects
                    frame = np.transpose(frame, (1, 2, 0))
                    pil_image = Image.fromarray(frame, 'RGB')
                
                blinkframes.append(pil_image)
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                continue
    return blinkframes

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
