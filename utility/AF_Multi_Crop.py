"""
AF_Multi_Crop.py

SPDX-License-Identifier: MIT
Version: 1.0.0
Author: AstroAF (Doug Reynolds)
Date: 2025-05-22
Project: Siril Multi-Crop Script Contribution

Description:
A user-friendly multi-crop script for Siril that allows:
- Batch cropping of .fit and .fits files from a selected input directory
- Saving and loading crop parameters for repeatable workflows
- A clear GUI interface with directory pickers and validation
- Preservation of original files with new cropped versions
- Optional reuse of crop settings across projects

Designed with astrophotographers in mind to streamline preprocessing of multi-channel or repeatable framing datasets.

Toolset: Python, Tkinter, SirilPy API

Website: https://astroaf.space
YouTube: https://youtube.com/@astroaf
Meta: https://www.facebook.com/AstroAFphotos
IG: https://www.instagram.com/astroaf/
bsky: https://bsky.app/profile/astroaf.space
"""

"""
Usage Guide:

1. Set your Home directory and open a representative image in Siril to establish the working directory.

2. After placing this script in your configured scripts directory, run it from Siril's menu:
   Scripts > AF_Multi_Crop.py

3. In the GUI:
   - Use "Get Crop Frame" to automatically pull the current selection (after drawing a crop frame in Siril).
   - Or manually enter X, Y, Width, and Height in the fields provided.
   - Use "Load Saved Crop" to restore previously saved crop dimensions.
   - Use "Save Crop As" to save the current dimensions to a named file (defaults to crop_settings.txt).
   - Input and output directories are set automatically to the image directory by default.
   - The output directory defaults to a /cropped folder within your image directory, but you may change it to any location.
   - Click "Run Crop" to batch crop all .fit/.fits files in the input folder.

4. During cropping:
   - Invalid files (e.g., images smaller than the crop region) will be skipped and reported.
   - You may cancel processing at any time using the "Cancel" button.

5. Upon completion:
   - A message will confirm the number of files cropped and skipped.
   - Reasons for skipped files are printed to the Siril console.

Notes:
- Original FITS files are never overwritten.
- The output directory must be different from the input directory.
- Crop dimensions are saved in standard CSV format: x,y,w,h
"""


import os
import sirilpy as s
from sirilpy import LogColor, SirilInterface
import sys
import tkinter as tk
from tkinter import ttk, messagebox
if s.check_module_version(">=0.6.47") and sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog
import threading

# Ensure a usable Siril version
required_version = "1.3.6"

# Connect to Siril
siril = SirilInterface()
if not siril.connect():
    raise RuntimeError("Failed to connect to Siril.")

# Check Siril version
try:
    siril.cmd("requires", required_version)
except Exception:
    raise RuntimeError(f"This script requires Siril version {required_version} or later!")

# Get working image directory
try:
    path = siril.get_image_filename()
    img_dir = os.path.dirname(os.path.abspath(path))
except Exception:
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showerror("Error", "Please open an image in Siril first.")
    root.destroy()
    exit(1)

siril.cmd("cd", f'"{img_dir}"')
os.chdir(img_dir)

crop_settings_filename = "crop_settings.txt"

# Initialize GUI
cancel_requested = False
root = tk.Tk()
root.attributes("-topmost", True)
style = ttk.Style()
style.theme_use('clam')  # 'alt', 'default', or 'classic' also work

bg_color = "#2e2e2e"
fg_color = "white"

style.configure('.', background=bg_color, foreground=fg_color)
style.configure('TLabel', background=bg_color, foreground=fg_color)
style.configure('TButton', background=bg_color, foreground=fg_color)
style.configure('TEntry', fieldbackground="#3c3c3c", foreground=fg_color)

root.title("AF Multi Crop")

def try_preload_crop():
    try:
        x, y, w, h = map(int, siril.get_siril_selection())
        x_var.set(str(x))
        y_var.set(str(y))
        w_var.set(str(w))
        h_var.set(str(h))
    except Exception:
        # No image loaded or invalid selection — silently ignore or log
        pass

root.after(100, try_preload_crop)

def load_crop():
    path = filedialog.askopenfilename(initialdir=img_dir, title="Select Crop Settings", filetypes=[("Text files", "*.txt")])
    if path:
        try:
            with open(path) as f:
                parts = f.read().strip().split(",")
                if len(parts) == 4:
                    x_var.set(parts[0])
                    y_var.set(parts[1])
                    w_var.set(parts[2])
                    h_var.set(parts[3])
                    messagebox.showinfo("Loaded", "Crop settings loaded.")
                else:
                    raise ValueError
        except:
            messagebox.showerror("Error", "Failed to load crop settings.")
            
def show_completion_message(msg):
    messagebox.showinfo("Complete", msg)
    root.quit()
            
def is_crop_valid(x, y, w, h, width, height):
    return (
        0 <= x < width and
        0 <= y < height and
        w > 0 and h > 0 and
        x + w <= width and
        y + h <= height
    )

def perform_crop():
    global cancel_requested
    cancel_button.config(state="normal")

    try:
        x = int(x_var.get())
        y = int(y_var.get())
        w = int(w_var.get())
        h = int(h_var.get())
    except ValueError:
        root.after(0, lambda: messagebox.showerror("Error", "Crop fields must contain valid integers."))
        cancel_button.config(state="disabled")
        return

    input_dir = input_dir_var.get()
    output_dir = output_dir_var.get()

    if not os.path.isdir(input_dir):
        root.after(0, lambda: messagebox.showerror("Error", f"Input directory does not exist:\n{input_dir}"))
        cancel_button.config(state="disabled")
        return

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.fit', '.fits'))]

    if not files:
        messagebox.showinfo("No Files", "No FITS files found in the input directory.")
        return

    count = 0
    skipped = 0

    for fname in files:
        if cancel_requested:
            break

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            siril.cmd("load", f'"{in_path}"')
            channels, height, width = siril.get_image_shape()

            if not is_crop_valid(x, y, w, h, width, height):
                print(f"[Warning] Skipping {fname}: crop out of bounds")
                skipped += 1
                continue

            siril.cmd("crop", f"{x} {y} {w} {h}")
            siril.cmd("save", f'"{out_path}"')

            count += 1
            if not cancel_requested:
                root.update()
        except Exception as e:
            print(f"[Warning] Failed to process {fname}: {e}")
            skipped += 1
            continue

    try:
        with open(os.path.join(output_dir, crop_settings_filename), "w") as f:
            f.write(f"{x},{y},{w},{h}")
    except Exception as e:
        print(f"[Warning] Failed to save crop settings: {e}")

    cancel_button.config(state="disabled")

    def show_message_and_quit(msg):
        messagebox.showinfo("Batch Crop", msg)
        root.destroy()

    if cancel_requested:
        msg = f"Crop cancelled after {count} files.\nSkipped: {skipped}"
    else:
        msg = (f"{count} files cropped and saved to:\n{output_dir}\n"
               f"Skipped: {skipped}\nSee log for details.")

    root.after(0, show_message_and_quit, msg)

def run_crop():
    global cancel_requested
    cancel_requested = False
    threading.Thread(target=perform_crop).start()
    
def cancel_crop():
    global cancel_requested
    cancel_requested = True
    messagebox.showinfo("Cancelled", "Crop process will stop after current file.")
    
def set_crop_settings_filename():
    global crop_settings_filename
    filepath = filedialog.asksaveasfilename(
        initialdir=img_dir,
        title="Save Crop Settings As...",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")]
    )
    if filepath:
        crop_settings_filename = os.path.basename(filepath)
        messagebox.showinfo("Filename Set", f"Crop settings will be saved as:\n{crop_settings_filename}")
        
def validate_directories(*args):
    in_dir = os.path.abspath(input_dir_var.get())
    out_dir = os.path.abspath(output_dir_var.get())

    if in_dir and out_dir and in_dir != out_dir:
        run_button.config(state="normal")
    else:
        run_button.config(state="disabled")

# GUI Below
    
# Declare crop coordinate variables before layout
x_var, y_var, w_var, h_var = [tk.StringVar() for _ in range(4)]

# Main Frame
main_frame = ttk.Frame(root, style="Dark.TFrame", padding=10)
main_frame.grid(row=0, column=0, sticky="nsew")

# Crop Coordinates Frame
coord_frame = ttk.Frame(main_frame)
coord_frame.grid(row=3, column=0, columnspan=3, pady=10)

# Button Frame
btn_frame = ttk.Frame(main_frame)
btn_frame.grid(row=5, column=0, columnspan=5, pady=10)

# Header
header = ttk.Label(main_frame, text="AstroAF Multi Crop", font=("Helvetica", 16, "bold"))
header.grid(row=0, column=0, columnspan=3, pady=(10, 15))

# Input Directory
ttk.Label(main_frame, text="Input Directory:").grid(row=1, column=0, sticky="e", padx=10, pady=5)
input_dir_var = tk.StringVar(value=img_dir)
input_dir_var.trace_add("write", validate_directories)
input_entry = ttk.Entry(main_frame, textvariable=input_dir_var, width=50)
input_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
ttk.Button(main_frame, text="Browse...", command=lambda: input_dir_var.set(filedialog.askdirectory(initialdir=img_dir))).grid(row=1, column=2, padx=5)

# Output Directory
ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky="e", padx=10, pady=5)
output_dir_var = tk.StringVar(value=os.path.join(img_dir, "cropped"))
output_dir_var.trace_add("write", validate_directories)
output_entry = ttk.Entry(main_frame, textvariable=output_dir_var, width=50)
output_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
ttk.Button(main_frame, text="Browse...", command=lambda: output_dir_var.set(filedialog.askdirectory(initialdir=img_dir))).grid(row=2, column=2, padx=5)

# Crop Coordinates
# First row: Crop X and Y
ttk.Label(coord_frame, text="Crop X:").grid(row=0, column=0, sticky="e", padx=(0, 5))
ttk.Entry(coord_frame, textvariable=x_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 15))

ttk.Label(coord_frame, text="Crop Y:").grid(row=0, column=2, sticky="e", padx=(0, 5))
ttk.Entry(coord_frame, textvariable=y_var, width=10).grid(row=0, column=3, sticky="w")

# Second row: Width and Height
ttk.Label(coord_frame, text="Width:").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=(5, 0))
ttk.Entry(coord_frame, textvariable=w_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 15), pady=(5, 0))

ttk.Label(coord_frame, text="Height:").grid(row=1, column=2, sticky="e", padx=(0, 5), pady=(5, 0))
ttk.Entry(coord_frame, textvariable=h_var, width=10).grid(row=1, column=3, sticky="w", pady=(5, 0))

# Buttons
ttk.Button(btn_frame, text="Load Saved Crop", command=load_crop).grid(row=5, column=0, padx=10, pady=(10, 10))
ttk.Button(btn_frame, text="Get Crop Frame", command=try_preload_crop).grid(row=5, column=1, padx=10, pady=(10, 10), sticky="w")
ttk.Button(btn_frame, text="Save Crop As", command=set_crop_settings_filename).grid(row=5, column=2, padx=5)
run_button = ttk.Button(btn_frame, text="Run Crop", command=run_crop)
run_button.grid(row=5, column=3, padx=10, pady=(10, 10))
cancel_button = tk.Button(btn_frame, text="Cancel", state="disabled", fg="black", command=lambda: cancel_crop())
cancel_button.grid(row=5, column=4, padx=10, pady=(10, 10))

# Handle exit
root.protocol("WM_DELETE_WINDOW", root.destroy)
root.mainloop()