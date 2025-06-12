# (c) EnderByBear 2025
# Inspired by CosmicClarity_Sharpen.py developed by Adrian Knagg-Baugh
# SPDX-License-Identifier: GPL-3.0-or-later

# Contact: enderbybear@foxmail.com
# Version: 1.0.0

# Release notes:
# 1.0.0: initial release

import sirilpy as s
s.ensure_installed("ttkthemes")

import os
import re
import sys
import math
import asyncio
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril

VERSION = "1.0.0"

if s.check_module_version(">=0.6.0") and sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog

class SirilCosmicClarityInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Cosmic Clarity Superres - v{VERSION}")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        try:
            self.siril.connect()
        except s.SirilConnectionError:
            self.siril.error_messagebox("Failed to connect to Siril")
            self.close_dialog()
            return

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image loaded")
            self.close_dialog()
            return

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return
            
        self.config_executable = self.check_config_file()
        tksiril.match_theme_to_siril(self.root, self.siril)

    def floor_value(self, value, decimals=2):
        """Floor a value to the specified number of decimal places"""
        factor = 10 ** decimals
        return math.floor(value * factor) / factor
        
    def update_scale_display(self, *args):
        """Update the displayed target median value with floor rounding"""
        value = self.scale_var.get()
        rounded_value = self.floor_value(value)
        self.scale_var.set(f"{rounded_value:.0f}")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Cosmic Clarity Superres Settings",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))
        
        # Options Frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Scale
        scale_frame = ttk.Frame(options_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_frame, text="Upscale Factor:").pack(side=tk.LEFT)
        self.scale_var = tk.IntVar(value=2)
        scale_scale = ttk.Scale(
            scale_frame,
            from_=2,
            to=4,
            orient=tk.HORIZONTAL,
            variable=self.scale_var,
            length=200
        )
        scale_scale.pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Label(
            scale_frame,
            textvariable=self.scale_var,
            width=5,
            style="Value.TLabel"
        ).pack(side=tk.LEFT)
        
        # Load Upscaled Image
        self.load_upscaled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Load Upscaled Image When Finished",
            variable=self.load_upscaled,
            style="TCheckbutton"
        ).pack(anchor=tk.W, pady=2)
        
        # Add trace to update display when slider changes
        self.scale_var.trace_add("write", self.update_scale_display)

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
        self.root.quit()
        self.root.destroy()
        
    async def run_cosmic_clarity(self, executable_path, scale):
        try:
            model_dir = os.path.dirname(executable_path)
            
            command = [
                executable_path,
                f"--input={self.siril.get_image_filename()}",
                f"--output_dir={self.siril.get_siril_wd()}",
                f"--scale={scale}",
                f"--model_dir={model_dir}"
            ]
            
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

                buffer += chunk.decode('utf-8', errors='ignore')
                lines = buffer.split('\n')

                for line in lines[:-1]:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        percentage = int(match.group(1))
                        message = "Seti Astro Cosmic Clarity Superres progress..."
                        self.siril.update_progress(message, percentage / 100)
                    else:
                        print(line.strip())

                buffer = lines[-1]

            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read()
                error_message = stderr.decode('utf-8', errors='ignore')
                raise subprocess.CalledProcessError(
                    process.returncode,
                    executable_path,
                    error_message
                )

            return True

        except Exception as e:
            print(f"Error in run_cosmic_clarity: {str(e)}")
            return False

    async def _apply_changes(self):
        try:
            # Claim the processing thread
            with self.siril.image_lock():
                # Read user input values
                executable_path = self.executable_path_var.get()
                scale = self.scale_var.get()
                load_upscaled = self.load_upscaled.get()
                
                # Save the executable path to config file if changed
                if executable_path != self.config_executable:
                    config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_superres.conf")
                    with open(config_file_path, 'w') as file:
                        file.write(f"{executable_path}\n")

                print(f"Running superres with scale: {scale}")
                self.siril.update_progress("Seti Astro Cosmic Clarity Superres starting...", 0)

                success = await self.run_cosmic_clarity(
                    executable_path,
                    scale
                )

                if success:
                    filename = self.siril.get_image_filename()
                    filename_without_extention = os.path.splitext(filename)[0]
                    outputfilename_fit = f"{filename_without_extention}_upscaled{scale}x.fit"
                    outputfilename_fits = f"{filename_without_extention}_upscaled{scale}x.fits"
                    
                    if(load_upscaled):
                        if os.path.exists(outputfilename_fit):
                            self.siril.log(f"ready to open upscaled image: {outputfilename_fit}")
                            self.siril.cmd(f"load \"{outputfilename_fit}\"")
                        elif os.path.exists(outputfilename_fits):
                            self.siril.log(f"ready to open upscaled image: {outputfilename_fits}")
                            self.siril.cmd(f"load \"{outputfilename_fits}\"")
                        else:
                            self.siril.log("cannot open upscaled image.")
                        
                    self.siril.reset_progress()
                    self.siril.log("Cosmic Clarity superres complete.")

        except Exception as e:
            print(f"Error in apply_changes: {str(e)}")
            self.siril.update_progress(f"Error: {str(e)}", 0)
            
    def check_config_file(self):
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, "sirilcc_superres.conf")

        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                executable_path = file.readline().strip()
                if os.path.isfile(executable_path) and os.access(executable_path, os.X_OK):
                    return executable_path

        messagebox.showinfo("Configuration", "Executable not yet configured. Recommended to use Seti Astro Cosmic Clarity Super-Resolution Upscaling Tool  V1.1 or higher.")
        return None

def main():
    try:
        root = ThemedTk()
        app = SirilCosmicClarityInterface(root)
        app.create_widgets()
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

