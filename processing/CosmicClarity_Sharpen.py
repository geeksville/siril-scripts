# (c) Adrian Knagg-Baugh 2024-2025
# SPDX-License-Identifier: GPL-3.0-or-later

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version: 1.0.7
# 1.0.1: convert "requires" to use exception handling
# 1.0.2: misc updates
# 1.0.3: Use tiffile instead of savetif32 to save the input file
#        This avoids colour shifts if the image profile != the display profile
# 1.0.4: Fix an error in 32-to-16-bit conversion; always save the input file as
#        32-bit to ensure consistent input for CC; add support for PSF auto-
#        detection and non-stellar amount; clear the input directory by default
# 1.0.5: DOn't print empty lines of CC output to the log
# 1.0.6: Implement available option checking so the auto PSF widget is not
#        available if that option is not supported
# 1.0.7: Fix an error that occurred if the config file was missing

import sirilpy as s
s.ensure_installed("ttkthemes", "tiffile")

import os
import re
import sys
import math
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import numpy as np
import tiffile

VERSION = "1.0.7"

if s.check_module_version(">=0.6.0") and sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog

class SirilCosmicClarityInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Cosmic Clarity Sharpening - v{VERSION}")
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

        # Create widgets
        self.create_widgets()

    def get_command_options(self, command_path: str, nonsense_arg: str = "--nonsense") -> List[str]:
        """
        Execute a command with a nonsense argument to trigger usage output,
        then parse the output to extract supported command-line options.

        Args:
            command_path: Path to the command executable
            nonsense_arg: Nonsense argument to trigger usage output (default: "--nonsense")

        Returns:
            List of option names (without leading dashes)

        Raises:
            subprocess.SubprocessError: If the command fails to execute
            ValueError: If no usage information could be parsed
        """
        try:
            # Run the command with a nonsense argument to get usage info
            result = subprocess.run(
                [command_path, nonsense_arg],
                capture_output=True,
                text=True,
                timeout=30  # Prevent hanging
            )

            # Usage info might be in stdout or stderr
            output = result.stdout + result.stderr

            if not output:
                raise ValueError(f"No output received from {command_path}")

            # Extract options from the usage output
            options = []

            # Pattern to match options like [-h], [--option], [--option VALUE], etc.
            # This captures both short (-h) and long (--option) options
            option_patterns = [
                r'\[(-[a-zA-Z])\]',  # Short options like [-h]
                r'\[(--[a-zA-Z_][a-zA-Z0-9_-]*)',  # Long options like [--option]
                r'(--[a-zA-Z_][a-zA-Z0-9_-]*)\s+[A-Z_]+',  # Options with arguments like --option ARGUMENT
                r'(--[a-zA-Z_][a-zA-Z0-9_-]*)\s+\{[^}]+\}',  # Options with choices like --option {choice1,choice2}
            ]

            for pattern in option_patterns:
                matches = re.findall(pattern, output)
                for match in matches:
                    # Remove leading dashes and add to options list
                    option = match.lstrip('-')
                    if option and option not in options:
                        options.append(option)

            # Also look for standalone options mentioned in the usage line
            # Pattern to find options in usage format like "[-h] [--option]"
            usage_line_pattern = r'\[(-{1,2}[a-zA-Z_][a-zA-Z0-9_-]*)'
            usage_matches = re.findall(usage_line_pattern, output)

            for match in usage_matches:
                option = match.lstrip('-')
                if option and option not in options:
                    options.append(option)

            if not options:
                raise ValueError(f"Could not parse any options from output: {output}")

            return sorted(options)  # Return sorted for consistent ordering

        except subprocess.TimeoutExpired:
            raise subprocess.SubprocessError(f"Command {command_path} timed out")
        except subprocess.CalledProcessError as e:
            # This is actually expected - the command should fail with unrecognized argument
            # But we still want to capture the output
            output = e.stdout + e.stderr if hasattr(e, 'stdout') else ""
            if not output:
                raise subprocess.SubprocessError(f"Command failed and produced no output: {e}")
        except FileNotFoundError:
            raise subprocess.SubprocessError(f"Command not found: {command_path}")

    def floor_value(self, value, decimals=2):
        """Floor a value to the specified number of decimal places"""
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def update_stellar_amount_display(self, *args):
        """Update the displayed target median value with floor rounding"""
        value = self.stellar_amount_var.get()
        rounded_value = self.floor_value(value)
        self.stellar_amount_var.set(f"{rounded_value:.2f}")

    def update_non_stellar_amount_display(self, *args):
        """Update the displayed target median value with floor rounding"""
        value = self.non_stellar_amount_var.get()
        rounded_value = self.floor_value(value)
        self.non_stellar_amount_var.set(f"{rounded_value:.2f}")

    def update_non_stellar_strength_display(self, *args):
        """Update the displayed target median value with floor rounding"""
        value = self.non_stellar_strength_var.get()
        rounded_value = self.floor_value(value)
        self.non_stellar_strength_var.set(f"{rounded_value:.0f}")

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
        self.clear_input_dir_var = tk.BooleanVar(value=True)
        clear_input_check = ttk.Checkbutton(
            options_frame,
            text="Clear input directory",
            variable=self.clear_input_dir_var,
            style="TCheckbutton"
        )
        clear_input_check.pack(anchor=tk.W, pady=2)
        tksiril.create_tooltip(clear_input_check,
            "Delete any files from the Cosmic Clarity input directory. "
            "If not done, Cosmic Clarity will process all image files in the input "
            "directory, which will take longer and generate potentially unnecessary files. "
            "WARNING: set this to False if you wish to retain previous content of the "
            "Cosmic Clarity input directory")

        # PSF Autodetection Checkbox
        self.auto_psf_var = tk.BooleanVar(value=True)
        auto_psf_check = ttk.Checkbutton(
            options_frame,
            text="Autodetect PSF",
            variable=self.auto_psf_var,
            style="TCheckbutton"
        )
        auto_psf_check.pack(anchor=tk.W, pady=2)
        tksiril.create_tooltip(auto_psf_check,
            "Automatically measure PSF per chunk and use the two nearest radius models. "
            "NOTE: this option requires at least CosmicClarity Sharpen v6.5. It will be "
            "disabled if not supported.")

## Separate channels functionality is problematic in some modes at present, so is disabled until more stable
        # Separate Channels Checkbox
        self.separate_channels_var = tk.BooleanVar(value=False)
#        separate_channels_check = ttk.Checkbutton(
#            options_frame,
#            text="Sharpen Channels Separately",
#            variable=self.separate_channels_var,
#            style="TCheckbutton"
#        )
#        separate_channels_check.pack(anchor=tk.W, pady=2)
#        tksiril.create_tooltip(separate_channels_check,
#            "Sharpen channels separately")

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

        # Add trace to update display when slider changes
        self.stellar_amount_var.trace_add("write", self.update_stellar_amount_display)

        # Non-stellar Amount
        non_stellar_amount_frame = ttk.Frame(options_frame)
        non_stellar_amount_frame.pack(fill=tk.X, pady=5)

        ttk.Label(non_stellar_amount_frame, text="Non-Stellar Amount:").pack(side=tk.LEFT)
        self.non_stellar_amount_var = tk.DoubleVar(value=0.5)
        non_stellar_amount_scale = ttk.Scale(
            non_stellar_amount_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.non_stellar_amount_var,
            length=200
        )
        non_stellar_amount_scale.pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Label(
            non_stellar_amount_frame,
            textvariable=self.non_stellar_amount_var,
            width=5
        ).pack(side=tk.LEFT)

        # Add trace to update display when slider changes
        self.non_stellar_amount_var.trace_add("write", self.update_non_stellar_amount_display)

        # Non-Stellar Strength
        non_stellar_strength_frame = ttk.Frame(options_frame)
        non_stellar_strength_frame.pack(fill=tk.X, pady=5)

        ttk.Label(non_stellar_strength_frame, text="Non-Stellar Strength:").pack(side=tk.LEFT)
        self.non_stellar_strength_var = tk.IntVar(value=3)
        non_stellar_strength_scale = ttk.Scale(
            non_stellar_strength_frame,
            from_=1,
            to=8,
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

        # Add trace to update display when slider changes
        self.non_stellar_strength_var.trace_add("write", self.update_non_stellar_strength_display)

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

        # FIXED: Check if executable path is not empty AND not None before trying to get command options
        if self.config_executable and self.config_executable.strip():
            try:
                self.command_options = self.get_command_options(self.config_executable)
                if "auto_detect_psf" not in self.command_options:
                    self.auto_psf_var.set(False)
                    auto_psf_check.config(state='disabled')
            except (subprocess.SubprocessError, ValueError) as e:
                print(f"Warning: Could not get command options for {self.config_executable}: {e}")
                self.command_options = []
                self.auto_psf_var.set(False)
                auto_psf_check.config(state='disabled')
        else:
            self.command_options = []
            self.auto_psf_var.set(False)
            auto_psf_check.config(state='disabled')

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
            try:
                self.command_options = self.get_command_options(filename)
                autopsf_available = "normal" if "auto_detect_psf" in self.command_options else "disabled"
            except (subprocess.SubprocessError, ValueError) as e:
                print(f"Warning: Could not get command options for {filename}: {e}")
                self.command_options = []
                autopsf_available = "disabled"

            # Find the auto PSF checkbox and update its state
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.LabelFrame) and "Options" in str(grandchild.cget('text')):
                            for widget in grandchild.winfo_children():
                                if isinstance(widget, ttk.Checkbutton) and "Autodetect PSF" in str(widget.cget('text')):
                                    widget.config(state=autopsf_available)
                                    break

    def _on_apply(self):
        # Wrap the async method to run in the event loop
        self.root.after(0, self._run_async_task)

    def _run_async_task(self):
        asyncio.run(self._apply_changes())

    def close_dialog(self):
        self.root.quit()
        self.root.destroy()

    async def run_cosmic_clarity(self, executable_path, mode, stellar_amount, non_stellar_strength, non_stellar_amount):
        # (Keep the existing implementation)
        try:
            command = [
                executable_path,
                f"--sharpening_mode={mode}",
                f"--stellar_amount={stellar_amount}",
                f"--nonstellar_strength={non_stellar_strength}",
                f"--nonstellar_amount={non_stellar_amount}"
            ]

            if not self.use_gpu_var.get():
                command.append("--disable_gpu")

            if self.auto_psf_var.get():
                command.append("--auto_detect_psf")

            if self.separate_channels_var.get():
                command.append("--sharpen_channels_separately")

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
                lines = buffer.split('\r')

                for line in lines[:-1]:
                    match = re.search(r'(\d+\.\d+)%', line)
                    if match:
                        percentage = float(match.group(1))
                        message = "Seti Astro Cosmic Clarity Sharpen progress..."
                        self.siril.update_progress(message, percentage / 100)
                    else:
                        tmp = line.strip()
                        if tmp != "":
                            print(tmp)

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
                mode = self.sharpening_mode_var.get()
                stellar_amount = self.stellar_amount_var.get()
                non_stellar_strength = self.non_stellar_strength_var.get()
                non_stellar_amount = self.non_stellar_amount_var.get()
                executable_path = self.executable_path_var.get()
                clear_input = self.clear_input_dir_var.get()

                # Save the executable path to config file if changed
                if executable_path != self.config_executable:
                    config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_sharpen.conf")
                    with open(config_file_path, 'w') as file:
                        file.write(f"{executable_path}\n")

                filename = self.siril.get_image_filename()
                directory = os.path.dirname(executable_path)
                basename = os.path.basename(filename)
                print(f"filename: {filename}")
                original_dir = os.getcwd()
                os.chdir(directory)
                os.makedirs("input", exist_ok=True)
                os.makedirs("output", exist_ok=True)

                inputpath = os.path.join(directory, "input")
                inputfilename = os.path.join(inputpath, basename) + str(".tif")
                outputpath = os.path.join(directory, "output")
                outputfilename = os.path.join(outputpath, f"{basename}_sharpened.tif")

                if clear_input:
                    files = Path(inputpath).glob("*.*")
                    for each_file in files:
                        try:
                            each_file.unlink()
                            print(f"Deleted: {each_file}")
                        except Exception as e:
                            print(f"Failed to delete {each_file}: {e}")

                was_16bit = False
                pixels = self.siril.get_image_pixeldata()
                if pixels.dtype == np.uint16:
                    pixels = pixels.astype(np.float32) / 65535.0
                    was_16bit = True

                # Determine photometric and reshape if needed
                if pixels.ndim == 2:
                    # Mono image
                    photometry = 'minisblack'
                elif pixels.ndim == 3 and pixels.shape[0] in (1, 3):
                    # Multi-sample image in (samples, height, width)
                    photometry = 'minisblack' if pixels.shape[0] == 1 else 'rgb'
                    pixels = pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)
                else:
                    raise ValueError(f"Unexpected image shape: {pixels.shape}")

                # Write TIFF without ICC profile
                tiffile.imwrite(inputfilename, pixels, photometric=photometry, planarconfig='contig')

                print(f"Running sharpening with mode: {mode}, stellar_amount: {stellar_amount}, non_stellar_strength: {non_stellar_strength}, "
                      f"non_stellar_amount: {non_stellar_amount}")
                self.siril.update_progress("Seti Astro Cosmic Clarity Sharpen starting...", 0)

                success = await self.run_cosmic_clarity(
                    executable_path,
                    mode,
                    stellar_amount,
                    non_stellar_strength,
                    non_stellar_amount
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
                    #pixel_data = pixel_data[:, ::-1, :]
                    force_16bit = self.siril.get_siril_config("core", "force_16bit")
                    if (was_16bit or force_16bit):
                        pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)
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

    def check_config_file(self):
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, "sirilcc_sharpen.conf")

        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                executable_path = file.readline().strip()
                if os.path.isfile(executable_path) and os.access(executable_path, os.X_OK):
                    return executable_path

        messagebox.showinfo("Configuration", "Executable not yet configured. It is recommended to use Seti Astro Cosmic Clarity v6.5 or higher.")
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
