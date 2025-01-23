# (c) Adrian Knagg-Baugh 2024
# SPDX-License-Identifier: GPL-3.0-or-later

import sirilpy as s
s.ensure_installed("ttkthemes", "tiffile")

import os
import re
import sys
import asyncio
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import numpy as np
import tiffile

VERSION = "1.0.0"

class SirilCosmicClarityInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Cosmic Clarity Sharpening - v{VERSION}")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        if not self.siril.connect():
            self.siril.error_messagebox("Failed to connect to Siril")
            self.close_dialog()
            return

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image loaded")
            self.close_dialog()
            return

        if not self.siril.cmd("requires", "1.3.6"):
            self.close_dialog()
            return

        if not self.siril.cmd("requires", "1.3.6"):
            messagebox.showerror("Error", "Siril version requirement not met")
            self.close_dialog()
            return

        self.config_executable = self.check_config_file()
        tksiril.match_theme_to_siril(self.root, self.siril)

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Cosmic Clarity Sharpening Settings",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Sharpening Mode Frame
        mode_frame = ttk.LabelFrame(main_frame, text="Sharpening Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.sharpening_mode_var = tk.StringVar(value="Stellar Only")
        sharpening_modes = ["Stellar Only", "Non-Stellar Only", "Both"]
        for mode in sharpening_modes:
            ttk.Radiobutton(
                mode_frame,
                text=mode,
                variable=self.sharpening_mode_var,
                value=mode
            ).pack(anchor=tk.W, pady=2)

        # Options Frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # GPU Checkbox
        self.use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Use GPU",
            variable=self.use_gpu_var,
            style="TCheckbutton"
        ).pack(anchor=tk.W, pady=2)

        # Clear Input Directory Checkbox
        self.clear_input_dir_var = tk.BooleanVar(value=False)
        clear_input_check = ttk.Checkbutton(
            options_frame,
            text="Clear input directory",
            variable=self.clear_input_dir_var,
            style="TCheckbutton"
        )
        clear_input_check.pack(anchor=tk.W, pady=2)
        tksiril.create_tooltip(clear_input_check,
            "Delete any TIFF files from the Cosmic Clarity input directory. "
            "If not done, Cosmic Clarity will process all TIFF files in the input "
            "directory, which will take longer and generate potentially unnecessary files.")

        # Stellar Amount
        stellar_amount_frame = ttk.Frame(options_frame)
        stellar_amount_frame.pack(fill=tk.X, pady=5)

        ttk.Label(stellar_amount_frame, text="Stellar Amount:").pack(side=tk.LEFT)
        self.stellar_amount_var = tk.DoubleVar(value=0.5)
        stellar_amount_scale = ttk.Scale(
            stellar_amount_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.stellar_amount_var,
            length=200
        )
        stellar_amount_scale.pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Label(
            stellar_amount_frame,
            textvariable=self.stellar_amount_var,
            width=5
        ).pack(side=tk.LEFT)

        # Non-Stellar Strength
        non_stellar_strength_frame = ttk.Frame(options_frame)
        non_stellar_strength_frame.pack(fill=tk.X, pady=5)

        ttk.Label(non_stellar_strength_frame, text="Non-Stellar Strength:").pack(side=tk.LEFT)
        self.non_stellar_strength_var = tk.IntVar(value=3)
        non_stellar_strength_scale = ttk.Scale(
            non_stellar_strength_frame,
            from_=1,
            to=5,
            orient=tk.HORIZONTAL,
            variable=self.non_stellar_strength_var,
            length=200
        )
        non_stellar_strength_scale.pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Label(
            non_stellar_strength_frame,
            textvariable=self.non_stellar_strength_var,
            width=5
        ).pack(side=tk.LEFT)

        # Executable Selection Frame
        exec_frame = ttk.LabelFrame(main_frame, text="Cosmic Clarity Executable", padding=10)
        exec_frame.pack(fill=tk.X, padx=5, pady=5)

        self.executable_path_var = tk.StringVar(value=self.config_executable or "")
        exec_entry = ttk.Entry(
            exec_frame,
            textvariable=self.executable_path_var,
            width=40
        )
        exec_entry.pack(side=tk.LEFT, padx=(0, 5), expand=True)

        ttk.Button(
            exec_frame,
            text="Browse",
            command=self._browse_executable,
            style="TButton"
        ).pack(side=tk.LEFT)

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT, padx=5)

        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self._on_apply,
            style="TButton"
        )
        apply_btn.pack(side=tk.LEFT, padx=5)

    def _browse_executable(self):
        filename = filedialog.askopenfilename(
            title="Select Cosmic Clarity Executable",
            initialdir=os.path.expanduser("~")
        )
        if filename:
            self.executable_path_var.set(filename)

    def _on_apply(self):
        # Wrap the async method to run in the event loop
        self.root.after(0, self._run_async_task)

    def _run_async_task(self):
        asyncio.run(self._apply_changes())

    def close_dialog(self):
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()

    async def run_cosmic_clarity(self, executable_path, mode, stellar_amount, non_stellar_strength):
        # (Keep the existing implementation)
        try:
            command = [
                executable_path,
                f"--sharpening_mode={mode}",
                f"--stellar_amount={stellar_amount}",
                f"--nonstellar_strength={non_stellar_strength}"
            ]

            if not self.use_gpu_var.get():
                command.append("--disable_gpu")

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
            )

            buffer = ""
            while True:
                chunk = await process.stdout.read(80)
                if not chunk:
                    break

                buffer += chunk.decode()
                lines = buffer.split('\r')

                for line in lines[:-1]:
                    match = re.search(r'(\d+\.\d+)%', line)
                    if match:
                        percentage = float(match.group(1))
                        message = "Seti Astro Cosmic Clarity Sharpen progress..."
                        self.siril.update_progress(message, percentage / 100)
                    else:
                        print(line.strip())

                buffer = lines[-1]

            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read()
                raise subprocess.CalledProcessError(
                    process.returncode,
                    executable_path,
                    stderr.decode()
                )

            return True

        except Exception as e:
            print(f"Error in run_cosmic_clarity: {str(e)}")
            return False

    async def _apply_changes(self):
        try:
            # Claim the processing thread
            if self.siril.claim_thread():
                # Read user input values
                mode = self.sharpening_mode_var.get()
                stellar_amount = self.stellar_amount_var.get()
                non_stellar_strength = self.non_stellar_strength_var.get()
                executable_path = self.executable_path_var.get()
                clear_input = self.clear_input_dir_var.get()

                # Save the executable path to config file if changed
                if executable_path != self.config_executable:
                    config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_sharpen.conf")
                    with open(config_file_path, 'w') as file:
                        file.write(f"{executable_path}\n")

                # Rest of the implementation remains the same as in the original script
                filename = self.siril.get_image_filename()
                directory = os.path.dirname(executable_path)
                basename = os.path.basename(filename)
                print(f"filename: {filename}")
                original_dir = os.getcwd()
                os.chdir(directory)
                os.makedirs("input", exist_ok=True)
                os.makedirs("output", exist_ok=True)

                inputpath = os.path.join(directory, "input")
                inputfilename = os.path.join(inputpath, basename)
                outputpath = os.path.join(directory, "output")
                outputfilename = os.path.join(outputpath, f"{basename}_sharpened.tif")

                if clear_input:
                    tiff_files = Path(inputpath).glob("*.tif*")
                    for tiff_file in tiff_files:
                        try:
                            tiff_file.unlink()
                            print(f"Deleted: {tiff_file}")
                        except Exception as e:
                            print(f"Failed to delete {tiff_file}: {e}")

                self.siril.cmd("savetif32", f"\"{inputfilename}\"")

                print(f"Running sharpening with mode: {mode}, stellar_amount: {stellar_amount}, non_stellar_strength: {non_stellar_strength}")
                self.siril.update_progress("Seti Astro Cosmic Clarity Sharpen starting...", 0)

                success = await self.run_cosmic_clarity(
                    executable_path,
                    mode,
                    stellar_amount,
                    non_stellar_strength
                )

                if success:
                    with tiffile.TiffFile(outputfilename) as tiff:
                        pixel_data = tiff.asarray()
                    pixel_data = np.ascontiguousarray(pixel_data)
                    # Handle both 2D (mono) and 3D (RGB) images
                    if pixel_data.ndim == 2:
                        # For 2D images, add a channel dimension
                        pixel_data = pixel_data[np.newaxis, :, :]
                    elif pixel_data.ndim == 3 and pixel_data.shape[2] == 3:
                        pixel_data = np.transpose(pixel_data, (2, 0, 1))
                        pixel_data = np.ascontiguousarray(pixel_data)
                    pixel_data = pixel_data[:, ::-1, :]
                    force_16bit = self.siril.get_siril_config("core", "force_16bit")
                    if (force_16bit):
                        pixel_data = np.rint(pixel_data * 65536).astype(np.uint16)
                    # Save original image for undo
                    self.siril.undo_save_state(f"Cosmic Clarity sharpen ({mode})")
                    # Update Siril
                    self.siril.set_image_pixeldata(pixel_data)
                    # Reset progress bar and report completion
                    self.siril.reset_progress()
                    self.siril.log("Cosmic Clarity sharpening complete.")

        except Exception as e:
            print(f"Error in apply_changes: {str(e)}")
            self.siril.update_progress(f"Error: {str(e)}", 0)

        finally:
            # Release the thread in the finally: block so that it is guaranteed to be released
            self.siril.release_thread()

    def check_config_file(self):
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, "siril", "sirilcc_sharpen.conf")

        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                executable_path = file.readline().strip()
                if os.path.isfile(executable_path) and os.access(executable_path, os.X_OK):
                    return executable_path

        messagebox.showinfo("Configuration", "Executable not yet configured. Recommended to use Seti Astro Cosmic Clarity v5.4 or higher.")
        return None

def main():
    try:
        root = ThemedTk()
        app = SirilCosmicClarityInterface(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
