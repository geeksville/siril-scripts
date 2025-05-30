"""
GraXpert AI Model Interface script
==================================
This script provides a direct interface between Siril and the GraXpert
AI ONNX models.
The script offers a GUI as well as a pyscript-compatible CLI interface
(see `pyscript GraXpert.py --help` for CLI details)
Single image and sequence processing is supported.

As this script offers tighter integration of the dependencies as well
as better performance it is intended to become the primary interface
to GraXpert in the future: if you experience issues with the legacy
GraXpert interface it is recommended to try this script instead.

(c) Adrian Knagg-Baugh 2025
SPDX-License-Identifier: GPL-3.0-or-later

Model inference methods adapt code from GraXpert for Siril data formats
=======================================================================
GraXpert website: https://graxpert.com
GraXpert is (c) the GraXpert Development Team
GraXpert code licensed as GPL-3.0-or-later
Models licensed as CC-BY-NC-SA-4.0
"""

# Version History
# 1.0.0 Initial release
# 1.0.1 Bug fix in handling mono images in BGE; improved fallback behaviour
#       for inferencing runtime errors (try again with CPU backend)
# 1.0.2 Interim fix for MacOS to prevent issues with the CREATE_ML_PROGRAM
#       flag; make the defaults match GraXpert (except smoothing: the
#       default GraXpert smoothing value of 0.0 seems too low so this is
#       set at 0.5)
# 1.0.3 Fix an error with use of the onnx_helper
# 1.0.4 Fix GPU checkbox on MacOS
# 1.0.5 Fallback to CPU is more robust
# 1.0.6 Fix a bug relating to printing the used inference providers
# 1.0.7 More bugfixes
# 1.0.8 Fix interpretation of a TkBool variable as an integer

import os
import re
import sys
import copy
import argparse
import platform
import tempfile
import threading
import subprocess
import tkinter as tk
from time import sleep
from tkinter import ttk, messagebox
from packaging.version import Version, parse

import sirilpy as s
# Check the module version is enough to provide ONNXHelper
if not s.check_module_version('>=0.6.42'):
    print("Error: requires sirilpy module >= 0.6.42")
    sys.exit(1)

from sirilpy import tksiril, SirilError

s.ensure_installed("ttkthemes", "numpy", "astropy", "appdirs",
                   "opencv-python")

import cv2
import numpy as np
from astropy.io import fits
from ttkthemes import ThemedTk
from appdirs import user_data_dir

# Determine the correct onnxruntime package based on OS and hardware,
# and ensure it is installed
onnx_helper = s.ONNXHelper()
onnx_helper.install_onnxruntime()

import onnxruntime

VERSION = "1.0.8"
DENOISE_CONFIG_FILENAME = "graxpert_denoise_model.conf"
BGE_CONFIG_FILENAME = "graxpert_bge_model.conf"
DECONVOLVE_STARS_CONFIG_FILENAME = "graxpert_deconv_stars_model.conf"
DECONVOLVE_OBJECTS_CONFIG_FILENAME = "graxpert_deconv_obj_model.conf"

_graxpert_mutex = threading.Lock()
_graxpert_version = None

def get_executable(siril):
    if siril is None or not siril.connected:
        return None
    return siril.get_siril_config('core', 'graxpert_path')

def check_graxpert_version(executable):
    """
    Check the version of the GraXpert executable.
    Returns the version string if successful, None otherwise.
    """
    version_key = "version: "

    # Check if executable is valid
    if not executable or not executable.strip():
        return

    # Check if the file exists and is executable
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        print("Error: cannot access or execute the GraXpert path", file=sys.stderr)
        return

    with _graxpert_mutex:
        try:
            # Prepare command arguments
            cmd = [executable, "-v"]

            # Run the process with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Wait for process with timeout (200ms)
            try:
                stdout, stderr = process.communicate(timeout=10)
                exit_status = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                print("GraXpert version check timed out")
                return

            # Check for errors
            if exit_status != 0:
                print(f"Spawning GraXpert failed during version check: {stderr}")
                return

            # Process output to extract version
            output = stderr + stdout
            if output:
                # Find the version string
                version_start = output.find(version_key)
                if version_start >= 0:
                    # Move past the key
                    version_start += len(version_key)
                    # Find the end of the version string
                    version_text = output[version_start:]
                    version_end = version_text.find(" ")

                    if version_end >= 0:
                        # Extract just the version number
                        version_string = version_text[:version_end].strip()
                    else:
                        # If no space after version, take the rest of the line
                        version_string = version_text.strip()

                    global _graxpert_version
                    _graxpert_version = version_string

            return

        except Exception as e:
            print(f"Error checking GraXpert version: {str(e)}")
            return

def get_available_local_operations():
    operations = {
            'denoise': 'Denoising',
            'bge': 'Background Extraction'
        }
    # Get the GraXpert directory
    deconvolution_stars_dir = os.path.join(user_data_dir(appname="GraXpert"), 'deconvolution-stars-ai-models')
    deconvolution_obj_dir = os.path.join(user_data_dir(appname="GraXpert"), 'deconvolution-object-ai-models')
    if get_available_local_models(deconvolution_stars_dir):
        operations.update({
            'deconvolution-stars': 'Deconvolution (Stellar)'})
    if get_available_local_models(deconvolution_obj_dir):
        operations.update({
            'deconvolution-object': 'Deconvolution (Objects)'})
    return operations

def get_available_operations():
    version = Version(_graxpert_version)
    # If version check failed or version is less than 3.0.0, abort initialization
    operations = {
            'denoise': 'Denoising',
            'bge': 'Background Extraction'
        }
    if (version.release[0] == 3 and version.release[1] == 1 and
            version.release[2] == 0 and version.is_prerelease):
        operations.update({
            'deconvolution-stars': 'Deconvolution (Stellar)',
            'deconvolution-object': 'Deconvolution (Objects)'
        })
    return operations

def get_available_local_models(subdir : str) -> dict:
    """
    Get a dictionary of available models from the GraXpert directory.
    Returns a dict with model names as keys and paths as values.
    """

    # Get the GraXpert directory
    models_dir = os.path.join(user_data_dir(appname="GraXpert"), subdir)
    model_paths = {}

    # Check if directory exists
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        # Search for model.onnx files in subdirectories
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    # Use subdirectory name as the display name
                    model_paths[subdir] = model_path

    return model_paths

def list_available_models(models_dir):
    """List all available models and exit. For use with CLI interfaces. """
    # Check if directory exists
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        print(f"Models directory not found: {models_dir}")
        sys.exit(1)

    # Find all available models
    available_models = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            model_path = os.path.join(subdir_path, "model.onnx")
            if os.path.exists(model_path) and os.path.isfile(model_path):
                available_models.append(subdir)

    if not available_models:
        print("No models found")
        sys.exit(1)

    # Sort models and print them
    available_models.sort()
    print("Available models:")
    for model in available_models:
        print(f"  {model}")

    # Print highest available model
    print(f"\nLatest available model: {available_models[-1]} (default if no model specified)")
    sys.exit(0)

def get_image_data_from_file(siril, path):
    """
    Load image data from a file.

    Args:
        path: Path to the image file

    Returns:
        Tuple of (data, header) where data is a numpy array and header is a FITS header
    """
    if path.lower().endswith((".fit", ".fits")):
        with fits.open(path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()  # Copy the header
            return data, header
    else:
        try:
            header = None
            self.siril.cmd(f"load {path}")
            header = siril.get_image_fits_header()
        except SirilError as e:
            self.siril.log(f"Error reading file {path}: {e}", s.LogColor.RED)
            return None, None
        return siril.get_image().data, header

def save_fits(data, path, original_header=None, history_text=""):
    """
    Save data to a FITS file.

    Args:
        data: Numpy array to save
        path: Path to save to
        original_header: Optional FITS header to use
        history_text: Text to add to the HISTORY keyword
    """
    if data.dtype not in (np.float32, np.uint16):
        data = data.astype(np.float32)
    # Create a new header if none is provided
    if original_header is None:
        header = fits.Header()
    else:
        try:
            with SuppressedStderr():
                header = fits.Header.fromstring(original_header, sep='\n')
        except Exception:
            header = fits.Header()
    # Add the HISTORY line
    header['HISTORY'] = history_text
    fits.writeto(path, data, header, overwrite=True)

class GraXpertModelManager:
    def __init__(self, parent, siril, callback=None):
        """
        Initialize the GraXpert Model Manager dialog.

        Args:
            parent: The parent window/widget
        """
        self.parent = parent
        self.siril = siril
        self.callback=callback
        self.models_by_operation = {}

        # Check GraXpert version and set up operations accordingly
        self.operations = get_available_operations()

        self.operation_cmd_map = {
            'denoise': 'denoising',
            'bge': 'background-extraction',
            'deconvolution-stars': 'deconv-stellar',
            'deconvolution-object': 'deconv-obj'
        }

    def show_dialog(self):
        """Show the model manager dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("GraXpert Model Manager")
        self.dialog.geometry("600x600")
        self.dialog.minsize(500, 400)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        self.create_widgets()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start refreshing the model list
        self.refresh_models()

        # Position the dialog relative to the parent window
        self.center_dialog()

    def center_dialog(self):
        """Center the dialog on the parent window"""
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()

        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        self.dialog.geometry(f"+{x}+{y}")

    def create_widgets(self):
        """Create the widgets for the dialog"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Operation selection
        op_frame = ttk.LabelFrame(main_frame, text="Operation", padding="5")
        op_frame.pack(fill=tk.X, pady=5)

        self.operation_var = tk.StringVar(value="denoise")

        max_columns = 2
        for i, (op_key, op_name) in enumerate(self.operations.items()):
            row = i // max_columns
            column = i % max_columns
            ttk.Radiobutton(
                op_frame,
                text=op_name,
                value=op_key,
                variable=self.operation_var,
                command=self.on_operation_changed
            ).grid(row=row, column=column, padx=10, pady=5, sticky="w")

        # Center the grid frame in its parent container
        op_frame.grid_columnconfigure(0, weight=1)
        op_frame.grid_columnconfigure(1, weight=1)

        # Model list frame
        model_frame = ttk.LabelFrame(main_frame, text="Models Available Remotely", padding="5")
        model_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a frame with scrollbar for the models list
        list_frame = ttk.Frame(model_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.model_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            font=("TkDefaultFont", 11)
        )
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.model_listbox.yview)

        # Status section
        self.status_var = tk.StringVar(value="Ready")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

        # Buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        self.refresh_btn = ttk.Button(
            buttons_frame,
            text="Refresh",
            command=self.refresh_models
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        self.download_btn = ttk.Button(
            buttons_frame,
            text="Download Selected Model",
            command=self.download_selected_model,
            state=tk.DISABLED
        )
        self.download_btn.pack(side=tk.RIGHT, padx=5)

        self.close_btn = ttk.Button(
            buttons_frame,
            text="Close",
            command=self.on_close
        )
        self.close_btn.pack(side=tk.RIGHT, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            mode="indeterminate"
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

    def on_operation_changed(self):
        """Handle operation change event"""
        operation = self.operation_var.get()
        self.update_model_list(operation)

    def update_model_list(self, operation):
        """Update the model list based on the selected operation"""
        self.model_listbox.delete(0, tk.END)
        self.download_btn.config(state=tk.DISABLED)

        if operation in self.models_by_operation:
            models = self.models_by_operation[operation]
            if models:
                for model in models:
                    self.model_listbox.insert(tk.END, model)
                self.model_listbox.selection_set(0)
                self.download_btn.config(state=tk.NORMAL)
            else:
                self.model_listbox.insert(tk.END, "No models available")
        else:
            self.model_listbox.insert(tk.END, "Click Refresh to check available models")

    def refresh_models(self):
        """Refresh the available models list"""
        operation = self.operation_var.get()
        self.status_var.set(f"Checking available models for {self.operations[operation]}...")
        self.progress_bar.start()
        self.refresh_btn.config(state=tk.DISABLED)
        self.download_btn.config(state=tk.DISABLED)

        # Start a thread to avoid blocking the UI
        threading.Thread(target=self._fetch_models_thread, args=(operation,), daemon=True).start()

    def _fetch_models_thread(self, operation):
        """Background thread to fetch models"""
        models = self.check_ai_versions(operation)

        # Update UI in the main thread
        self.dialog.after(0, lambda: self._update_after_fetch(operation, models))

    def _update_after_fetch(self, operation, models):
        """Update UI after fetching models"""
        if self.callback:
            self.dialog.after(0, self.callback())
        self.progress_bar.stop()
        self.refresh_btn.config(state=tk.NORMAL)

        if models:
            self.models_by_operation[operation] = models
            self.status_var.set(f"Found {len(models)} models for {self.operations[operation]}")
            self.update_model_list(operation)
        else:
            self.models_by_operation[operation] = []
            self.status_var.set("Failed to retrieve models. Check GraXpert installation.")
            self.model_listbox.delete(0, tk.END)
            self.model_listbox.insert(tk.END, "Error retrieving models")

    def download_selected_model(self):
        """Download the selected model"""
        operation = self.operation_var.get()
        selection = self.model_listbox.curselection()

        if not selection:
            messagebox.showerror("Selection Error", "Please select a model to download")
            return

        model_version = self.model_listbox.get(selection[0])

        self.status_var.set(f"Downloading {model_version}...")
        self.progress_bar.start()
        self.download_btn.config(state=tk.DISABLED)
        self.refresh_btn.config(state=tk.DISABLED)

        # Start a thread to download the model
        threading.Thread(
            target=self._download_model_thread,
            args=(operation, model_version),
            daemon=True
        ).start()

    def _download_model_thread(self, operation, version):
        """Background thread to download model"""
        success = self.download_model(operation, version)

        # Update UI in the main thread
        self.dialog.after(0, lambda: self._update_after_download(success, version))

    def _update_after_download(self, success, version):
        """Update UI after downloading model"""
        self.progress_bar.stop()
        self.refresh_btn.config(state=tk.NORMAL)
        self.download_btn.config(state=tk.NORMAL)

        if success:
            self.status_var.set(f"Successfully downloaded model {version}")
            messagebox.showinfo("Download Complete", f"Model {version} has been downloaded successfully.")
        else:
            self.status_var.set("Failed to download model")
            messagebox.showerror("Download Failed", "Failed to download the selected model. Check the logs for details.")

    def on_close(self):
        """Handle dialog close"""
        self.dialog.grab_release()
        self.dialog.destroy()

    def check_ai_versions(self, operation):
        """
        Check available AI model versions for the specified GraXpert operation.

        Args:
            operation (str): One of 'denoise', 'bge', 'deconv_star', or 'deconv_obj'

        Returns:
            list: List of available AI model versions, or None if check fails
        """
        # Map operation names to GraXpert command arguments
        operation_map = self.operation_cmd_map

        if operation not in operation_map:
            print(f"Invalid operation: {operation}")
            return None

        try:
            executable = get_executable(self.siril)
            with _graxpert_mutex:
                if not executable:
                    return None

                # Prepare command arguments
                cmd_args = [
                    executable,
                    "-cmd",
                    operation_map[operation],
                    "--help"
                ]

                try:
                    # Execute the command with a timeout
                    process = subprocess.Popen(
                        cmd_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        universal_newlines=True
                    )

                    # Wait for the process with timeout (500ms in the C code)
                    try:
                        stdout, stderr = process.communicate(timeout=10)
                        output = stdout + stderr  # Combine both stdout and stderr as the help text might be in either
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                        output = stdout + stderr
                        print("GraXpert process timed out")
                        return None
                    # Check if version information was found
                    return self.parse_ai_versions(output)

                except Exception as e:
                    print(f"Error executing GraXpert: {str(e)}")
                    return None
        except Exception as e:
            print(f"Error in check_ai_versions: {str(e)}")
            return None

    def parse_ai_versions(self, output):
        """
        Parse the output of GraXpert to extract available AI model versions.

        Args:
            output (str): The output from GraXpert command

        Returns:
            list: List of available AI model versions, or None if parsing fails
        """
        # Preprocess the output to simplify the parsing
        # 1. Split into lines
        lines = output.split('\n')

        # 2. For each line, strip everything up to "root INFO" and any leading/trailing whitespace
        cleaned_lines = []
        for line in lines:
            root_info_pos = line.find("root INFO")
            if root_info_pos != -1:
                # Extract content after "root INFO" and strip whitespace
                cleaned_line = line[root_info_pos + len("root INFO"):].strip()
                if cleaned_line:  # Only add non-empty lines
                    cleaned_lines.append(cleaned_line)

        # 3. Join the cleaned lines with spaces
        cleaned_output = ' '.join(cleaned_lines)

        # 4. Replace multiple spaces with a single space
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output)

        # Now we can use a simple regex to find and extract the version list
        match = re.search(r'available remotely:\s*\[(.*?)\]', cleaned_output)
        if not match:
            print("No 'available remotely:' list found")
            return None

        # Extract the list content
        list_content = match.group(1).strip()

        # Split by commas and extract versions
        versions = []
        for part in list_content.split(','):
            part = part.strip()
            # Skip empty parts
            if not part:
                continue
            # Extract version patterns (like 1.0.0, 2.3.4, etc.)
            version_match = re.search(r'\d+\.\d+\.\d+', part)
            if version_match:
                versions.append(version_match.group(0))

        if not versions:
            print("No valid versions found in list")
            return None
        return versions
    
    def download_model(self, operation, version):
        """
        Download a specific AI model version.

        Args:
            operation (str): Operation type ('denoise', 'bge', 'deconv_star', 'deconv_obj')
            version (str): Model version to download

        Returns:
            bool: True if download was successful, False otherwise
        """
        executable = get_executable(self.siril)
        if not executable:
            return False

        command = self.operation_cmd_map[operation]

        # Create temporary FITS file
        with _graxpert_mutex:
            try:
                with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp_file:
                    temp_fits_path = tmp_file.name

                # Create a minimal FITS file
                data = np.zeros((256, 256), dtype=np.float32)
                hdu = fits.PrimaryHDU(data)
                hdu.writeto(temp_fits_path, overwrite=True)

                # Prepare download command
                cmd_args = [
                    executable,
                    "-cli",
                    "-cmd",
                    command,
                    "-ai_version",
                    version,
                    "-output",
                    temp_fits_path.rstrip('.fits'),
                    temp_fits_path
                ]

                print(f"Running download command: {' '.join(cmd_args)}")

                # Execute the command
                process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True
                )

                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    print(f"Error downloading model: {stderr}")
                    return False

                # Check if download was successful by looking for success indicators in output
                success = ("download successful" in stdout or
                        process.returncode == 0)

                if not success:
                    # Handle case where success isn't explicitly stated but download was successful
                    # For example, some tools may not output success messages
                    success = "error" not in stdout.lower() and "failed" not in stdout.lower()

                if success:
                    print("Download succeeded")
                    self.dialog.after(0, self.callback)
                    return True
                else:
                    print("Download failed")
                    return False

            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                return False
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_fits_path):
                        os.unlink(temp_fits_path)
                except:
                    pass

class DenoiserProcessing:
    """Class encapsulating the core image processing functionality for GraXpert AI Denoise."""

    def __init__(self, siril):
        """
        Initialize the processing class.

        Args:
            siril: SirilInterface instance, which must already be connected.
        """
        if not siril:
            raise ValueError("No SirilInterface provided to DenoiserProcessing()")

        self.siril = siril

        # Cache for processed images
        self.cached_processed_image = None
        self.cached_original_image = None

        self.config_dir = self.siril.get_siril_configdir() if self.siril else None

    def reset_cache(self):
        """Reset the cached denoised image."""
        self.cached_processed_image = None
        self.cached_original_image = None

    def check_config_file(self):
        """
        Check for a saved model path in the configuration file.
        Returns model_path or None if not found.
        """
        if not self.config_dir:
            return None

        config_file_path = os.path.join(self.config_dir, DENOISE_CONFIG_FILENAME)
        model_path = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
        return model_path

    def save_config_file(self, model_path):
        """
        Save the selected model path to the configuration file.
        """
        if not self.config_dir:
            return

        config_file_path = os.path.join(self.config_dir, DENOISE_CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(model_path + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def apply_blend(self, strength):
        """
        Apply blending with cached denoised image.

        Args:
            strength: Blending strength (0-1)

        Returns:
            Blended image as numpy array
        """
        try:
            if self.cached_processed_image is None or self.cached_original_image is None:
                print("No cached images for blending")
                return None

            if strength != 1.0:
                original_dtype = self.cached_processed_image.dtype
                blended = self.cached_processed_image * strength + \
                        self.cached_original_image * (1 - strength)
                if blended.dtype != original_dtype:
                    blended = blended.astype(original_dtype)
            else:
                blended = self.cached_processed_image

            if self.siril.is_image_loaded():
                try:
                    with self.siril.image_lock():
                        if not self.siril.is_cli():
                            self.siril.undo_save_state(f"GraXpert AI denoise: strength {strength:.2f}")
                        self.siril.set_image_pixeldata(blended)
                except s.ImageDialogOpenError:
                    messagebox.showerror("Image dialog open", "An image dialog is open: please close it and try again.")
                    self.siril.reset_progress()
                except s.ProcessingThreadBusyError:
                    messagebox.showerror("Thread busy", "The processing thread is busy. Please wait for it to finish "
                        "and try again.")
                    self.siril.reset_progress()

            self.siril.reset_progress()
            return blended

        except Exception as e:
            print(f"Error in blending: {str(e)}")
            return None

    def denoise(self, image, ai_path, batch_size=4, progress_callback=None,
                ai_gpu_acceleration=True, window_size=256, stride=128):
        """
        Apply AI-based denoising to an image.

        Args:
            image: Input image as numpy array
            ai_path: Path to ONNX model
            batch_size: Number of patches to process at once
            window_size: Size of patch window
            stride: Stride between patches
            progress_callback: Function to call with progress updates
            ai_gpu_acceleration: Whether to use GPU acceleration

        Returns:
            Denoised image as numpy array
        """
        print("Starting denoising")

        # Handle planar format (c, h, w) -> (h, w, c)
        if image.shape[0] < image.shape[1] and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
            planar_format = True
        else:
            planar_format = False

        # Sanitize batch size
        if batch_size < 1:
            print(f"mapping batch_size of {batch_size} to 1")
            batch_size = 1
        elif batch_size > 32:
            print(f"mapping batch_size of {batch_size} to 32")
            batch_size = 32
        elif batch_size & batch_size - 1 != 0:  # check if batch_size is power of two
            print(f"mapping batch_size of {batch_size} to {2 ** (batch_size).bit_length() // 2}")
            batch_size = 2 ** (batch_size).bit_length() // 2  # map batch_size to power of two

        # Calculate median and median absolute deviation (MAD)
        median = np.median(image[::4, ::4, :], axis=[0, 1])
        mad = np.median(np.abs(image[::4, ::4, :] - median), axis=[0, 1])

        # Set model threshold based on version
        if "1.0.0" in ai_path or "1.1.0" in ai_path:
            model_threshold = 1.0
        else:
            model_threshold = 10.0

        # Handle grayscale images
        num_colors = image.shape[-1]
        if num_colors == 1:
            image = np.array([image[:, :, 0], image[:, :, 0], image[:, :, 0]])
            image = np.moveaxis(image, 0, -1)

        H, W, _ = image.shape
        offset = int((window_size - stride) / 2)

        # Calculate number of patches
        h, w, _ = image.shape
        ith = int(h / stride) + 1
        itw = int(w / stride) + 1

        # Pad image
        dh = ith * stride - h
        dw = itw * stride - w
        image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
        image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)

        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset):, :, :]), axis=0)
        image = np.concatenate((image[:offset, :, :], image), axis=0)
        image = np.concatenate((image, image[:, (w - offset):, :]), axis=1)
        image = np.concatenate((image[:, :offset, :], image), axis=1)

        output = copy.deepcopy(image)

        print(f"Available inference providers: {onnxruntime.get_available_providers()}")

        # Initialize ONNX runtime session
        providers = []
        if platform.system().lower() == 'darwin':
            if ai_gpu_acceleration is True:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        else:
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)

        try:
            session = onnxruntime.InferenceSession(ai_path, providers=providers)
        except Exception as err:
            error_message = str(err)
            print("Warning: falling back to CPU.")
            if "cudaErrorNoKernelImageForDevice" in error_message \
                or "Error compiling model" in error_message:
                print("ONNX cannot build an inferencing kernel for this GPU.")
            # Retry with CPU only
            providers = ['CPUExecutionProvider']
            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except ONNXRuntimeError as err:
                messagebox.showerror("Error", "Cannot build an inference model on this device")
                return

        print(f"Using inference providers: {session.get_providers()}")

        print(f"Used inference providers: {session.get_providers()}")

        # Process image in batches
        cancel_flag = False
        last_progress = 0

        for b in range(0, ith * itw + batch_size, batch_size):
            if cancel_flag:
                print("Denoising cancelled")
                return None

            input_tiles = []
            input_tile_copies = []
            for t_idx in range(0, batch_size):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j

                tile = image[x:x + window_size, y:y + window_size, :]
                tile = (tile - median) / mad * 0.04
                input_tile_copies.append(np.copy(tile))
                tile = np.clip(tile, -model_threshold, model_threshold)

                input_tiles.append(tile)

            if not input_tiles:
                continue

            input_tiles = np.array(input_tiles)

            # Run inference
            output_tiles = []

            try:
                session_result = session.run(None, {"gen_input_image": input_tiles})[0]
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                providers = ['CPUExecutionProvider']
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
                session_result = session.run(None, {"gen_input_image": input_tiles})[0]

            for e in session_result:
                output_tiles.append(e)

            output_tiles = np.array(output_tiles)

            # Place denoised tiles back into output image
            for t_idx, tile in enumerate(output_tiles):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j
                tile = np.where(input_tile_copies[t_idx] < model_threshold, tile, \
                                input_tile_copies[t_idx])
                tile = tile / 0.04 * mad + median
                tile = tile[offset:offset + stride, offset:offset + stride, :]
                output[x + offset:stride * (i + 1) + offset, y + offset:stride * \
                                           (j + 1) + offset, :] = tile

            # Update progress
            p = int(b / (ith * itw + batch_size) * 100)
            if p > last_progress:
                if progress_callback is not None:
                    progress_callback(f"Processing: {p}%", p/100)
                else:
                    print(f"Progress: {p}%")
                last_progress = p

        # Crop output back to original size
        output = output[offset:H + offset, offset:W + offset, :]

        # Handle grayscale output
        if num_colors == 1:
            output = np.array([output[:, :, 0]])
            output = np.moveaxis(output, 0, -1)

        # Convert back to planar format if needed
        if planar_format:
            output = np.transpose(output, (2, 0, 1))

        # Cache denoised image for future blending
        self.cached_processed_image = output

        print("Finished denoising")
        return output

    def process_image(self, model_path, strength=1.0, batch_size=4,
                    gpu_acceleration=True, progress_callback=None):
        """
        Process an image with denoising and blending.

        Args:
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Processed image as numpy array
        """
        try:
            if progress_callback:
                progress_callback("Fetching image data...")

            # Get original shape and data format
            original_shape = self.cached_original_image.shape
            is_planar = len(original_shape) == 3 and original_shape[0] <= 4

            # Normalize if pixel values exceed [0,1]
            pixel_data = self.cached_original_image
            original_dtype = pixel_data.dtype
            if original_dtype == np.uint16:
                pixel_data = pixel_data.astype(np.float32) / 65535.0

            # Process the image
            if progress_callback:
                progress_callback("Starting denoising process...")

            denoised = self.denoise(
                pixel_data,
                model_path,
                batch_size,
                progress_callback,
                gpu_acceleration
            )

            if denoised is None:
                if progress_callback:
                    progress_callback("Processing cancelled.")
                return None

            # Scale back if normalized
            if original_dtype == np.uint16:
                denoised = denoised * 65535.0
                denoised = denoised.astype(np.uint16)

            # Ensure the output has the same shape as input
            if denoised.shape != original_shape:
                if progress_callback:
                    progress_callback("Reshaping output to match input dimensions...")
                if len(original_shape) == 2:
                    # Handle special case for grayscale
                    if len(denoised.shape) == 3:
                        denoised = denoised[:, :, 0]
                elif is_planar and len(denoised.shape) == 3 and denoised.shape[2] <= 4:
                    # If input was planar (c,h,w) and output is (h,w,c)
                    denoised = np.transpose(denoised, (2, 0, 1))

            # Cache the denoised image for future blending
            self.cached_processed_image = denoised

            # Apply blending
            denoised = self.apply_blend(strength)

            if progress_callback:
                progress_callback("Processing complete.", 1.0)
            return denoised

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_sequence(self, sequence_name, model_path, strength=1.0, batch_size=4,
                       gpu_acceleration=True,
                       progress_callback=None):
        """
        Process a sequence with denoising and blending.

        Args:
            sequence_name: Name of the sequence to process
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Processing sequence {sequence_name}...")

            if not self.siril.is_sequence_loaded():
                # Try to load the sequence
                try:
                    self.siril.cmd("load_seq", f"\"{sequence_name}\"")
                except SirilError:
                    if progress_callback:
                        progress_callback(f"Failed to load sequence {sequence_name}")
                    return False

            sequence = self.siril.get_seq()
            input_seqname = sequence.seqname
            output_seqname = 'denoised_' + input_seqname

            # Get files to process
            files = [self.siril.get_seq_frame_filename(i) for i in range(sequence.number) \
                    if sequence.imgparam[i].incl]

            total_files = len(files)
            if total_files == 0:
                if progress_callback:
                    progress_callback("No files to process in sequence")
                return False

            for i, f in enumerate(files):
                # Reset the cached image
                self.reset_cache()

                file_progress = i / total_files
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}", file_progress)

                # Get the pixel data and FITS header
                self.cached_original_image, header = get_image_data_from_file(self.siril, f)
                if self.cached_original_image is None:
                    print(f"Error loading file {f}, skipping this file...")
                    continue

                # Reshape mono images to 3D with a channels size of 1
                if self.cached_original_image.ndim == 2:
                    self.cached_original_image = self.cached_original_image[np.newaxis, ...]

                # Define a callback to maintain overall progress
                # pylint: disable=cell-var-from-loop
                def file_progress_callback(msg, p = None):
                    if progress_callback:
                        if p:
                            overall_progress = file_progress + p / total_files
                            progress_callback(f"File {i+1}/{total_files}: {msg}", overall_progress)
                        else:
                            progress_callback(msg)

                # Process the image
                denoised = self.process_image(
                    model_path,
                    strength,
                    batch_size,
                    gpu_acceleration,
                    file_progress_callback
                )

                if denoised is None:
                    continue

                # Save the processed image
                output_path = os.path.join(self.siril.get_siril_wd(),
                                           f"{output_seqname}{(i+1):05d}.fit")
                print(f"Saving frame as {output_path}")
                save_fits(denoised, output_path, original_header=header,
                            history_text=f"GraXpert denoise (strength {strength:.2f})")

            # Create the new sequence
            self.siril.create_new_seq(output_seqname)

            # On completion, load the new sequence
            self.siril.cmd("load_seq", f"\"{output_seqname}\"")

            if progress_callback:
                progress_callback(f"Sequence processing complete: {output_seqname}", 1.0)
                sleep(1)
                self.siril.reset_progress()

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return False

class DeconvolutionProcessing:
    """
    Class encapsulating the core image processing functionality
    for GraXpert AI Deconvolution.
    """

    def __init__(self, siril):
        """
        Initialize the processing class.

        Args:
            siril: SirilInterface instance, which must already be connected.
        """
        if not siril:
            raise ValueError("No SirilInterface provided to DeconvolutionProcessing()")

        self.siril = siril

        # Cache for processed images
        self.cached_processed_image = None
        self.cached_original_image = None

        self.config_dir = self.siril.get_siril_configdir() if self.siril else None

    def reset_cache(self):
        """Reset the cached deconvolved image."""
        self.cached_processed_image = None
        self.cached_original_image = None

    def check_config_file(self, operation):
        """
        Check for a saved model path in the configuration file.
        Returns model_path or None if not found.
        """
        if not self.config_dir:
            return None

        config_file_path = None
        if operation == "deconvolution-stars":
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_STARS_CONFIG_FILENAME)
        else:
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_OBJECTS_CONFIG_FILENAME)

        model_path = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
        return model_path

    def save_config_file(self, model_path):
        """
        Save the selected model path to the configuration file.
        """
        if not self.config_dir:
            return

        if 'stars' in model_path:
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_STARS_CONFIG_FILENAME)
        else:
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_OBJECTS_CONFIG_FILENAME)

        try:
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(model_path + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def deconvolve(self, image, ai_path, strength, psfsize, batch_size=4,
                  window_size=512, stride=448, progress_callback=None, ai_gpu_acceleration=True):
        """
        Apply AI-based deconvolution to an image.

        Args:
            image: Input image as numpy array
            ai_path: Path to ONNX model
            strength: Deconvolution strength (0-1)
            psfsize: Size of the PSF (Point Spread Function)
            batch_size: Number of patches to process at once
            window_size: Size of patch window
            stride: Stride between patches
            progress_callback: Function to call with progress updates
            ai_gpu_acceleration: Whether to use GPU acceleration

        Returns:
            Deconvolved image as numpy array
        """
        print("Starting deconvolution")

        # Handle planar format (c, h, w) -> (h, w, c)
        if image.shape[0] < image.shape[1] and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
            planar_format = True
        else:
            planar_format = False

        # Determine deconvolution type based on model path
        if "stars" in ai_path:
            deconv_type = "Stellar"
        elif "obj" in ai_path:
            deconv_type = "Obj"
        else:
            deconv_type = "Unknown"
            print(f"Unknown model type: {ai_path}, assuming Object type")

        # Adjust strength (as per original function)
        strength = 0.95 * strength  # TODO: strength of exactly 1.0 brings no results, to fix

        # Normalize PSF size according to the model type
        if deconv_type == "Stellar":
            psfsize = np.clip((psfsize / 2.355 - 1.5) / 3.0, 0.05, 0.95)  # Stellar
        else:
            if "1.0.0" in ai_path:
                psfsize = np.clip((psfsize / 2.355 - 1.0) / 5.0, 0.05, 0.95)  # Object v1.0.0
            else:
                psfsize = np.clip((psfsize / 2.355 - 0.5) / 5.5, 0.05, 0.95)  # Object v1.0.1

        print(f"Calculated normalized PSFsize value: {psfsize}")

        # Sanitize batch size
        if batch_size < 1:
            print(f"mapping batch_size of {batch_size} to 1")
            batch_size = 1
        elif batch_size > 32:
            print(f"mapping batch_size of {batch_size} to 32")
            batch_size = 32
        elif not (batch_size & (batch_size - 1) == 0):  # check if batch_size is power of two
            print(f"mapping batch_size of {batch_size} to {2 ** (batch_size).bit_length() // 2}")
            batch_size = 2 ** (batch_size).bit_length() // 2  # map batch_size to power of two

        # Adjust batch size for RGB images
        if batch_size >= 4 and image.shape[-1] == 3:
            batch_size = batch_size // 4

        num_colors = image.shape[-1]

        H, W, _ = image.shape
        offset = int((window_size - stride) / 2)

        # Calculate padding and pad the image
        h, w, _ = image.shape
        ith = int(h / stride) + 1
        itw = int(w / stride) + 1

        dh = ith * stride - h
        dw = itw * stride - w

        image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
        image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)

        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset):, :, :]), axis=0)
        image = np.concatenate((image[:offset, :, :], image), axis=0)
        image = np.concatenate((image, image[:, (w - offset):, :]), axis=1)
        image = np.concatenate((image[:, :offset, :], image), axis=1)

        output = copy.deepcopy(image)

        # Initialize ONNX runtime session
        providers = []
        if platform.system().lower() == 'darwin':
            if ai_gpu_acceleration is True:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        else:
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)

        try:
            session = onnxruntime.InferenceSession(ai_path, providers=providers)
        except Exception as err:
            error_message = str(err)
            print("Warning: falling back to CPU.")
            if "cudaErrorNoKernelImageForDevice" in error_message \
                or "Error compiling model" in error_message:
                print("ONNX cannot build an inferencing kernel for this GPU.")
            # Retry with CPU only
            providers = ['CPUExecutionProvider']
            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except ONNXRuntimeError as err:
                messagebox.showerror("Error", "Cannot build an inference model on this device")
                return

        print(f"Available inference providers: {onnxruntime.get_available_providers()}")
        print(f"Using inference providers: {session.get_providers()}")

        # Process image in batches
        cancel_flag = False
        last_progress = 0

        for b in range(0, ith * itw + batch_size, batch_size):
            if cancel_flag:
                print("Deconvolution cancelled")
                return None

            input_tiles = []
            input_tile_copies = []
            params = []
            for t_idx in range(0, batch_size):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j

                tile = image[x:x + window_size, y:y + window_size, :]

                # Logarithmic preprocessing
                _min = np.min(tile, axis=(0, 1))
                tile = tile - _min + 1e-5
                tile = np.log(tile)

                _mean = tile.mean()
                _std = tile.std()
                _mean, _std = _mean.astype(np.float32), _std.astype(np.float32)
                tile = (tile - _mean) / _std * 0.1
                params.append([_mean, _std, _min])

                input_tile_copies.append(np.copy(tile))
                input_tiles.append(tile)

            if not input_tiles:
                continue

            input_tiles = np.array(input_tiles)
            input_tiles = np.moveaxis(input_tiles, -1, 1)
            input_tiles = np.reshape(input_tiles, [input_tiles.shape[0] * num_colors, 1, window_size, window_size])

            # Run inference
            output_tiles = []
            sigma = np.full(shape=(input_tiles.shape[0], 1), fill_value=psfsize, dtype=np.float32)
            strenght_p = np.full(shape=(input_tiles.shape[0], 1), fill_value=strength, dtype=np.float32)
            conds = np.concatenate([sigma, strenght_p], axis=-1)

            if deconv_type == "Obj" and "1.0.0" in ai_path:
                try:
                    session_result = session.run(None, {"gen_input_image": input_tiles, "sigma": sigma, "strenght": strenght_p})[0]
                except Exception as err:
                    error_message = str(err)
                    print("Warning: falling back to CPU.")
                    error_patterns = ("cudaErrorNoKernelImageForDevice",
                                      "Error compiling model")
                    if any(pattern in error_message for pattern in error_patterns):
                        print("ONNX cannot build an inferencing kernel for this GPU.")
                    # Retry with CPU only
                    print("Falling back to GPU")
                    # Retry with CPU only
                    providers = ['CPUExecutionProvider']
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                    session_result = session.run(None, {"gen_input_image": input_tiles, "sigma": sigma, "strenght": strenght_p})[0]
            else:
                try:
                    session_result = session.run(None, {"gen_input_image": input_tiles, "params": conds})[0]
                except Exception as err:
                    error_message = str(err)
                    print("Warning: falling back to CPU.")
                    error_patterns = ("cudaErrorNoKernelImageForDevice",
                                      "Error compiling model")
                    if any(pattern in error_message for pattern in error_patterns):
                        print("ONNX cannot build an inferencing kernel for this GPU.")
                    # Retry with CPU only
                    providers = ['CPUExecutionProvider']
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                    session_result = session.run(None, {"gen_input_image": input_tiles, "sigma": sigma, "strenght": strenght_p})[0]

            for e in session_result:
                output_tiles.append(e)

            output_tiles = np.array(output_tiles)
            output_tiles = input_tiles - output_tiles
            output_tiles = np.reshape(output_tiles, [output_tiles.shape[0] // num_colors, num_colors, window_size, window_size])
            output_tiles = np.moveaxis(output_tiles, 1, -1)

            # Post-process tiles
            for idx in range(len(params)):
                output_tiles[idx] = output_tiles[idx] * params[idx][1] / 0.1 + params[idx][0]
                output_tiles[idx] = np.exp(output_tiles[idx])
                output_tiles[idx] = output_tiles[idx] + params[idx][2] - 1e-5

            # Place deconvolved tiles back into output image
            for t_idx, tile in enumerate(output_tiles):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j
                tile = tile[offset:offset + stride, offset:offset + stride, :]
                output[x + offset:stride * (i + 1) + offset, y + offset:stride * (j + 1) + offset, :] = tile

            # Update progress
            p = int(b / (ith * itw + batch_size) * 100)
            if p > last_progress:
                if progress_callback is not None:
                    progress_callback(f"Processing: {p}%", p/100)
                else:
                    print(f"Progress: {p}%")
                last_progress = p

        # Crop output back to original size and clip values
        output = output[offset:H + offset, offset:W + offset, :]
        output = np.clip(output, 0.0, 1.0)

        # Convert back to planar format if needed
        if planar_format:
            output = np.transpose(output, (2, 0, 1))

        # Cache deconvolved image for future blending
        self.cached_processed_image = output

        print("Finished deconvolution")
        return output

    def process_image(self, model_path, strength=1.0, psfsize=2.5, batch_size=4,
                     gpu_acceleration=True, progress_callback=None):
        """
        Process an image with deconvolution and blending.

        Args:
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            psfsize: Size of the PSF (Point Spread Function)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Processed image as numpy array
        """
        try:
            if progress_callback:
                progress_callback("Fetching image data...")

            # Get original shape and data format
            original_shape = self.cached_original_image.shape
            is_planar = len(original_shape) == 3 and original_shape[0] <= 4

            # Normalize if pixel values exceed [0,1]
            pixel_data = self.cached_original_image
            original_dtype = pixel_data.dtype
            if original_dtype == np.uint16:
                pixel_data = pixel_data.astype(np.float32) / 65535.0

            # Process the image
            if progress_callback:
                progress_callback("Starting deconvolution process...")

            deconvolved = self.deconvolve(
                pixel_data,
                model_path,
                strength,
                psfsize,
                batch_size,
                progress_callback=progress_callback,
                ai_gpu_acceleration=gpu_acceleration
            )

            if deconvolved is None:
                if progress_callback:
                    progress_callback("Processing cancelled.")
                return None

            # Scale back if normalized
            if original_dtype == np.uint16:
                deconvolved = deconvolved * 65535.0
                deconvolved = deconvolved.astype(np.uint16)

            # Ensure the output has the same shape as input
            if deconvolved.shape != original_shape:
                if progress_callback:
                    progress_callback("Reshaping output to match input dimensions...")
                if len(original_shape) == 2:
                    # Handle special case for grayscale
                    if len(deconvolved.shape) == 3:
                        deconvolved = deconvolved[:, :, 0]
                elif is_planar and len(deconvolved.shape) == 3 and deconvolved.shape[2] <= 4:
                    # If input was planar (c,h,w) and output is (h,w,c)
                    deconvolved = np.transpose(deconvolved, (2, 0, 1))

            # Cache the deconvolved image for future blending
            self.cached_processed_image = deconvolved

            # Update image
            if self.siril.is_image_loaded():
                try:
                    with self.siril.image_lock():
                        if not self.siril.is_cli():
                            self.siril.undo_save_state(f"GraXpert AI deconvolve: strength {strength:.2f}")
                        self.siril.set_image_pixeldata(deconvolved)
                except s.ImageDialogOpenError:
                    messagebox.showerror("Image dialog open", "An image dialog is open: please close it and try again.")
                    self.siril.reset_progress()
                except s.ProcessingThreadBusyError:
                    messagebox.showerror("Thread busy", "The processing thread is busy. Please wait for it to finish "
                        "and try again.")
                    self.siril.reset_progress()

            if progress_callback:
                progress_callback("Processing complete.", 1.0)
            return deconvolved

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_sequence(self, sequence_name, model_path, strength=1.0, psfsize=2.5, batch_size=4,
                       gpu_acceleration=True,
                       progress_callback=None):
        """
        Process a sequence with deconvolution and blending.

        Args:
            sequence_name: Name of the sequence to process
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            psfsize: Size of the PSF (Point Spread Function)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Processing sequence {sequence_name}...")

            if not self.siril.is_sequence_loaded():
                # Try to load the sequence
                try:
                    self.siril.cmd("load_seq", f"\"{sequence_name}\"")
                except SirilError:
                    if progress_callback:
                        progress_callback(f"Failed to load sequence {sequence_name}")
                    return False

            sequence = self.siril.get_seq()
            input_seqname = sequence.seqname
            print(model_path)
            output_seqname = 'deconv_obj_' + input_seqname \
                if 'deconvolution-object' in model_path else \
                'deconv_stellar_' + input_seqname

            # Get files to process
            files = [self.siril.get_seq_frame_filename(i) for i in range(sequence.number) \
                    if sequence.imgparam[i].incl]

            total_files = len(files)
            if total_files == 0:
                if progress_callback:
                    progress_callback("No files to process in sequence")
                return False

            for i, f in enumerate(files):
                # Reset the cached image
                self.reset_cache()

                file_progress = i / total_files
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}", file_progress)

                # Get the pixel data and FITS header
                self.cached_original_image, header = get_image_data_from_file(self.siril, f)  # Assuming this function exists
                if self.cached_original_image is None:
                    print(f"Error loading file {f}, skipping this file...")
                    continue

                # Reshape mono images to 3D with a channels size of 1
                if self.cached_original_image.ndim == 2:
                    self.cached_original_image = self.cached_original_image[np.newaxis, ...]

                # Define a callback to maintain overall progress
                def file_progress_callback(msg, p=None):
                    if progress_callback:
                        if p:
                            overall_progress = file_progress + p / total_files
                            progress_callback(f"File {i+1}/{total_files}: {msg}", overall_progress)
                        else:
                            progress_callback(msg)

                # Process the image
                deconvolved = self.process_image(
                    model_path,
                    strength,
                    psfsize,
                    batch_size,
                    gpu_acceleration,
                    file_progress_callback
                )

                if deconvolved is None:
                    continue

                # Save the processed image
                output_path = os.path.join(self.siril.get_siril_wd(),
                                           f"{output_seqname}{(i+1):05d}.fit")
                print(f"Saving frame as {output_path}")
                save_fits(deconvolved, output_path, original_header=header,
                            history_text=f"GraXpert deconvolve (strength {strength:.2f}, psfsize {psfsize:.2f})")

            # Create the new sequence
            self.siril.create_new_seq(output_seqname)

            # On completion, load the new sequence
            self.siril.cmd("load_seq", f"\"{output_seqname}\"")

            if progress_callback:
                progress_callback(f"Sequence processing complete: {output_seqname}", 1.0)
                sleep(1)
                self.siril.reset_progress()

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return False

class BGEProcessing:
    """Class encapsulating the core image processing functionality for GraXpert Background Extraction."""

    def __init__(self, siril):
        """
        Initialize the processing class.

        Args:
            siril: SirilInterface instance, which must already be connected.
        """
        if not siril:
            raise ValueError("No SirilInterface provided to BGEProcessing()")

        self.siril = siril

        # Cache for processed images
        self.cached_background_image = None
        self.cached_original_image = None
        self.cached_processed_image = None

        self.config_dir = self.siril.get_siril_configdir() if self.siril else None

    def reset_cache(self):
        """Reset the cached images."""
        self.cached_background_image = None
        self.cached_original_image = None
        self.cached_processed_image = None

    def check_config_file(self):
        """
        Check for a saved model path in the configuration file.
        Returns model_path or None if not found.
        """
        if not self.config_dir:
            return None

        config_file_path = os.path.join(self.config_dir, BGE_CONFIG_FILENAME)
        model_path = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
        return model_path

    def save_config_file(self, model_path):
        """
        Save the selected model path to the configuration file.
        """
        if not self.config_dir:
            return

        config_file_path = os.path.join(self.config_dir, BGE_CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(model_path + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def gaussian_kernel(self, sigma):
        """Calculate appropriate kernel size for Gaussian blur based on sigma"""
        size = int(8.0 * sigma + 1.0)
        if size % 2 == 0:
            size += 1
        return (size, size)

    def apply_correction(self, image, background, correction_type):
        """
        Apply correction using the cached background image.

        Args:
            correction_type: Type of correction ('subtraction' or 'division')

        Returns:
            Corrected image as numpy array
        """
        try:
            if image is None or image is None:
                print("No image for correction")
                return None

            # Create a copy of the original image to work with
            corrected = copy.deepcopy(image)

            # Apply the correction based on the selected type
            if correction_type == "subtraction":
                mean = np.mean(background)
                corrected = corrected - background + mean
            elif correction_type == "division":
                # Handle each channel separately for division
                num_colors = 3 if len(corrected.shape) > 2 else 1
                if num_colors == 1 and len(corrected.shape) == 2:
                    # Handle grayscale as a special case
                    mean = np.mean(corrected)
                    corrected = corrected / background * mean
                else:
                    for c in range(num_colors):
                        mean = np.mean(corrected[c, :, :])
                        corrected[c, :, :] = corrected[c, :, :] / background[c, :, :] * mean

            # Clip the result to valid range
            corrected = np.clip(corrected, 0.0, 1.0)

            # Cache the corrected image
            self.cached_processed_image = corrected
            self.siril.reset_progress()
            return corrected

        except Exception as e:
            print(f"Error in correction: {str(e)}")
            return None

    def extract_background_ai(self, image, ai_path, smoothing=0,
                              ai_gpu_acceleration=True, progress_callback=None):
        """
        Apply AI-based background extraction to an image.

        Args:
            image: Input image as numpy array
            ai_path: Path to ONNX model
            smoothing: Amount of smoothing to apply (0-1)
            ai_gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Background image as numpy array
        """
        print("Starting background extraction")

        # Handle different image formats
        was_mono = False
        if len(image.shape) == 2:
            # Handle grayscale image
            was_mono = True
            image = np.expand_dims(image, -1)
        # Convert to hwc format if needed:
        was_planar = False
        if image.shape[0] < 4 and len(image.shape) == 3 and image.shape[0] < image.shape[1] \
                              and image.shape[0] < image.shape[2]:
            was_planar = True
            image = np.transpose(image, (1, 2, 0))

        # Store original shape for later reshaping
        original_shape = image.shape
        num_colors = image.shape[-1]
        if num_colors == 1:
            was_mono = True
        # Shrink and pad to avoid artifacts on borders
        padding = 8
        if progress_callback:
            progress_callback("Preparing image...", 0.05)
        # Resize to a standard size for the AI model
        imarray_shrink = cv2.resize(image, dsize=(256 - 2*padding, 256 - 2*padding),
                                    interpolation=cv2.INTER_LINEAR)

        if len(imarray_shrink.shape) == 2:
            imarray_shrink = np.expand_dims(imarray_shrink, -1)
        # Pad the image to avoid edge artifacts
        imarray_shrink = np.pad(imarray_shrink, ((padding, padding), (padding, padding), (0, 0)),
                               mode="edge")
        if progress_callback:
            progress_callback("Computing image statistics...", 0.1)

        # Calculate median and median absolute deviation for each channel
        median = []
        mad = []
        for c in range(num_colors):
            median.append(np.median(imarray_shrink[:, :, c]))
            mad.append(np.median(np.abs(imarray_shrink[:, :, c] - median[c])))
        if progress_callback:
            progress_callback("Normalizing image...", 0.15)

        # Normalize the image for the AI model
        imarray_shrink = (imarray_shrink - median) / mad * 0.04
        imarray_shrink = np.clip(imarray_shrink, -1.0, 1.0)
        # For grayscale, convert to RGB for the AI model
        if num_colors == 1:
            imarray_shrink = np.array([imarray_shrink[:, :, 0],
                                       imarray_shrink[:, :, 0],
                                       imarray_shrink[:, :, 0]])
            imarray_shrink = np.moveaxis(imarray_shrink, 0, -1)

        if progress_callback:
            progress_callback("Initializing ONNX runtime...", 0.25)

        # Initialize ONNX runtime session
        providers = []
        if platform.system().lower() == 'darwin':
            if ai_gpu_acceleration is True:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        else:
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)

        try:
            session = onnxruntime.InferenceSession(ai_path, providers=providers)
        except Exception as err:
            error_message = str(err)
            print("Warning: falling back to CPU.")
            if "cudaErrorNoKernelImageForDevice" in error_message \
                or "Error compiling model" in error_message:
                print("ONNX cannot build an inferencing kernel for this GPU.")
            # Retry with CPU only
            providers = ['CPUExecutionProvider']
            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except ONNXRuntimeError as err:
                messagebox.showerror("Error", "Cannot build an inference model on this device")
                return

        print(f"Using inference providers: {session.get_providers()}")

        if progress_callback:
            progress_callback("Running inference...", 0.4)

        # Run inference
        try:
            background = session.run(None, {"gen_input_image": np.expand_dims(imarray_shrink, axis=0)})[0][0]
        except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException as err:
            error_message = str(err)
            print("Warning: falling back to CPU.")
            error_patterns = ("cudaErrorNoKernelImageForDevice",
                              "Error compiling model")
            if any(pattern in error_message for pattern in error_patterns):
                print("ONNX cannot build an inferencing kernel for this GPU.")
            # Retry with CPU only
            print("Falling back to GPU")
            providers = ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(ai_path, providers=providers)
            background = session.run(None, {"gen_input_image": np.expand_dims(imarray_shrink, axis=0)})[0][0]

        if progress_callback:
            progress_callback("Post-processing...", 0.6)

        # Denormalize the background
        background = background / 0.04 * mad + median

        # Apply smoothing if requested
        if smoothing != 0:
            sigma = smoothing * 20
            kernel = self.gaussian_kernel(sigma)
            background = cv2.GaussianBlur(background, ksize=kernel, sigmaX=sigma, sigmaY=sigma)

        if progress_callback:
            progress_callback("Finalizing background...", 0.8)

        # Remove padding
        if padding != 0:
            background = background[padding:-padding, padding:-padding, :]

        # Apply additional smoothing for better results
        sigma = 3.0
        kernel = self.gaussian_kernel(sigma)
        background = cv2.GaussianBlur(background, ksize=kernel, sigmaX=sigma, sigmaY=sigma)

        # Resize back to original dimensions
        background = cv2.resize(background, dsize=(original_shape[1], original_shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        if was_planar:
            background = np.transpose(background, (2, 0, 1))

        # Ensure output has the same shape as input
        if len(background.shape) == 2 and len(original_shape) == 3:
            background = np.expand_dims(background, -1)
        elif was_mono and len(background.shape) == 3:
            background = background[0, :, :]

        # Cache the extracted background
        self.cached_background_image = background

        if progress_callback:
            progress_callback("Background extraction completed", 1.0)

        print("Finished background extraction")
        return background

    def process_image(self, model_path, correction_type="subtraction", smoothing=0,
                    keep_bg=False, filename=None, header=None, gpu_acceleration=True,
                    progress_callback=None):
        """
        Process an image with background extraction and correction.

        Args:
            model_path: Path to the ONNX model
            correction_type: Type of correction ('subtraction' or 'division')
            smoothing: Amount of smoothing to apply (0-1)
            keep_bg: whether to save the extracted background
            filename: filename (for use with keep_bg)
            header: original header (for use with keep_bg)
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Processed image as numpy array
        """
        try:
            if progress_callback:
                progress_callback("Fetching image data...")

            # Ensure we have an original image
            if self.cached_original_image is None:
                if progress_callback:
                    progress_callback("No image loaded.")
                return None

            # Get original shape and data format
            original_shape = self.cached_original_image.shape
            original_dtype = self.cached_original_image.dtype

            # Normalize if needed
            pixel_data = self.cached_original_image
            if original_dtype == np.uint16:
                pixel_data = pixel_data.astype(np.float32) / 65535.0

            # Process the image
            if progress_callback:
                progress_callback("Starting background extraction...")

            # Extract the background
            background = self.extract_background_ai(
                pixel_data,
                model_path,
                smoothing,
                gpu_acceleration,
                progress_callback
            )

            if background is None:
                if progress_callback:
                    progress_callback("Processing cancelled.")
                return None

            # Apply correction
            if progress_callback:
                progress_callback("Applying correction...")

            corrected = self.apply_correction(pixel_data, background, correction_type)

            if corrected is None:
                if progress_callback:
                    progress_callback("Correction failed.")
                return None

            # Scale back if needed
            if original_dtype == np.uint16:
                corrected = corrected * 65535.0
                corrected = corrected.astype(np.uint16)
                if self.cached_background_image is not None:
                    self.cached_background_image = self.cached_background_image * 65535.0
                    self.cached_background_image = self.cached_background_image.astype(np.uint16)

            if keep_bg:
                extension = self.siril.get_siril_config("core", "extension")
                output_path = os.path.splitext(filename)[0] + "_bg" + extension
                print(f"Saving background as {output_path}")
                save_fits(self.cached_background_image, path=output_path, original_header=header,
                          history_text="Extracted background")

            # Update the display if an image is loaded
            if self.siril.is_image_loaded():
                try:
                    with self.siril.image_lock():
                        if not self.siril.is_cli():
                            self.siril.undo_save_state(f"GraXpert AI BGE: {correction_type}")
                        self.siril.set_image_pixeldata(corrected)
                except s.ImageDialogOpenError:
                    messagebox.showerror("Image dialog open", "An image dialog is open: please close it and try again.")
                    self.siril.reset_progress()
                except s.ProcessingThreadBusyError:
                    messagebox.showerror("Thread busy", "The processing thread is busy. Please wait for it to finish "
                        "and try again.")
                    self.siril.reset_progress()

            if progress_callback:
                progress_callback("Processing complete.", 1.0)
            return corrected

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_sequence(self, sequence_name, model_path, correction_type="subtraction",
                       smoothing=0.5, keep_bg=False, gpu_acceleration=True,
                       progress_callback=None):
        """
        Process a sequence with background extraction and correction.

        Args:
            sequence_name: Name of the sequence to process
            model_path: Path to the ONNX model
            correction_type: Type of correction ('subtraction' or 'division')
            smoothing: Amount of smoothing to apply (0-1)
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Processing sequence {sequence_name}...")

            if not self.siril.is_sequence_loaded():
                # Try to load the sequence
                try:
                    self.siril.cmd("load_seq", f"\"{sequence_name}\"")
                except SirilError:
                    if progress_callback:
                        progress_callback(f"Failed to load sequence {sequence_name}")
                    return False

            sequence = self.siril.get_seq()
            input_seqname = sequence.seqname
            output_seqname = 'bge_' + input_seqname

            # Get files to process
            files = [self.siril.get_seq_frame_filename(i) for i in range(sequence.number)
                    if sequence.imgparam[i].incl]

            total_files = len(files)
            if total_files == 0:
                if progress_callback:
                    progress_callback("No files to process in sequence")
                return False

            for i, f in enumerate(files):
                # Reset the cached image
                self.reset_cache()

                file_progress = i / total_files
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}", file_progress)

                # Get the pixel data and FITS header
                self.cached_original_image, header = get_image_data_from_file(self.siril, f)
                if self.cached_original_image is None:
                    print(f"Error loading file {f}, skipping this file...")
                    continue

                # Define a callback to maintain overall progress
                def file_progress_callback(msg, p=None):
                    if progress_callback:
                        if p is not None:
                            overall_progress = file_progress + p / total_files
                            progress_callback(f"File {i+1}/{total_files}: {msg}", overall_progress)
                        else:
                            progress_callback(msg)

                output_path = os.path.join(self.siril.get_siril_wd(),
                                          f"{output_seqname}{(i+1):05d}.fit")
                header = None
                if keep_bg:
                    header = self.siril.get_seq_frame_header(i)

                # Process the image
                corrected = self.process_image(
                    model_path,
                    correction_type,
                    smoothing,
                    keep_bg,
                    output_path,
                    header,
                    gpu_acceleration,
                    file_progress_callback
                )

                if corrected is None:
                    continue

                # Save the processed image
                print(f"Saving frame as {output_path}")
                save_fits(corrected, output_path, original_header=header,
                             history_text=f"GraXpert BGE ({correction_type})")

            # Create the new sequence
            self.siril.create_new_seq(output_seqname)

            # On completion, load the new sequence
            self.siril.cmd("load_seq", f"\"{output_seqname}\"")

            if progress_callback:
                progress_callback(f"Sequence processing complete: {output_seqname}", 1.0)
                self.siril.reset_progress()

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return False

class GUIInterface:
    """Class providing the GUI interface for GraXpert AI Operations."""

    def __init__(self, root, siril):
        if not siril:
            raise ValueError("No SirilInterface provided to GUIInterface()")
        self.root = root
        self.root.title(f"GraXpert AI - Siril interface v{VERSION}")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        self.siril = siril
        # Get available operations
        self.operations = get_available_local_operations()
        self.selected_operation = tk.StringVar()

        # Initialize processor reference (will be set based on operation)
        self.processor = None

        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()
        if not (image_loaded or seq_loaded):
            self.siril.error_messagebox("No image or sequence loaded")
            self.close_dialog()
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            self.close_dialog()
            return

        # Initialize variables for UI
        self.model_path_var = tk.StringVar(value="")
        self.strength_var = tk.DoubleVar(value=0.5)
        self.smoothing_var = tk.DoubleVar(value=0.5)
        self.psf_size_var = tk.DoubleVar(value=5.0)  # Default PSF size value
        self.batch_size_var = tk.IntVar(value=4)
        self.keep_bg_var = tk.BooleanVar(value=False)
        self.gpu_acceleration_var = tk.BooleanVar(value=True)
        self.model_path_mapping = {}

        tksiril.match_theme_to_siril(self.root, self.siril)

        self.siril.log("This script is under ongoing development. Please report any bugs to "
            "https://gitlab.com/free-astro/siril-scripts. We are also especially keen "
            "for confirmation of success / failure from Linux users with AMD Radeon "
            "or Intel ARC GPUs as we do not have these hardware / OS combinations among "
            "the development team", color=s.LogColor.BLUE)
        # Create widgets
        self.create_widgets()

        # Set default operation to denoise
        if 'denoise' in self.operations:
            self.selected_operation.set('denoise')
            # Set the corresponding display name in the dropdown
            if 'denoise' in self.operations:
                op_display_name = self.operations['denoise']
                self.operation_display_var.set(op_display_name)
            self._on_operation_selected(None)  # Initialize the correct processor

        # Set progress label
        if image_loaded:
            self._update_progress("Single image loaded: will process this image only")
        else:
            self._update_progress("Sequence loaded: will process selected frames of the sequence")

    def create_widgets(self):
        """ Create Tk widgets to provide the script GUI"""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title and version
        title_label = ttk.Label(
            main_frame,
            text="GraXpert AI",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 5))
        version_label = ttk.Label(main_frame, text=f"Script version: {VERSION}")
        version_label.pack(pady=(0, 10))

        # Separator
        sep = ttk.Separator(main_frame, orient='horizontal')
        sep.pack(fill=tk.X, pady=5)

        # Operation selection frame
        op_frame = ttk.Frame(main_frame)
        op_frame.pack(fill=tk.X, pady=5)

        op_label = ttk.Label(op_frame, text="Operation:")
        op_label.pack(side=tk.LEFT, padx=5)

        # Convert operations dict to display names for dropdown
        operation_names = list(self.operations.values())
        operation_keys = list(self.operations.keys())

        # Create a separate variable for the display name
        self.operation_display_var = tk.StringVar()

        self.op_dropdown = ttk.Combobox(
            op_frame,
            textvariable=self.operation_display_var,  # Use display variable instead
            state="readonly",
            values=operation_names,  # Display names, not keys
            width=20
        )
        self.op_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tksiril.create_tooltip(self.op_dropdown, "Select which GraXpert AI operation to perform")

        # Bind operation selection event
        self.op_dropdown.bind("<<ComboboxSelected>>", self._on_operation_selected)

        # Model selection frame
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=5)

        # Model label
        model_label = ttk.Label(model_frame, text="Select Model:")
        model_label.pack(side=tk.LEFT, padx=5)

        # Create a separate variable for the display name
        self.model_name_var = tk.StringVar()

        # Create the model dropdown
        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_name_var,
            state="readonly",
            width=40
        )
        self.model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tksiril.create_tooltip(self.model_dropdown, "Select a GraXpert AI model. "
                "Models can be downloaded using the GraXpert Model Manager button")

        # Parameters Frame - will contain operation-specific parameters
        self.params_frame = ttk.LabelFrame(main_frame, text="Parameters")
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create frames for each operation's parameters (initially hidden)
        self.operation_frames = {}

        # Denoise operation parameters
        denoise_frame = ttk.Frame(self.params_frame)
        self.operation_frames['denoise'] = denoise_frame

        # Strength slider for denoise
        strength_frame = ttk.Frame(denoise_frame)
        strength_frame.pack(fill=tk.X, padx=5, pady=5)
        strength_label = ttk.Label(strength_frame, text="Strength:")
        strength_label.pack(side=tk.LEFT, padx=5)
        strength_slider = ttk.Scale(
            strength_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.strength_var,
            length=200
        )
        strength_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.strength_value = ttk.Label(strength_frame, text="0.5")
        self.strength_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(strength_slider, "Adjust the denoising strength. The "
                "result is a linear blend of the denoised image and the original, "
                "with the mixture of the two images controlled by the strength "
                "parameter.")
        self.strength_var.trace_add("write", self._update_strength_label)

        # Deconvolution operations parameters (for both stars and object)
        deconv_stars_frame = ttk.Frame(self.params_frame)
        self.operation_frames['deconvolution-stars'] = deconv_stars_frame
        deconv_object_frame = ttk.Frame(self.params_frame)
        self.operation_frames['deconvolution-object'] = deconv_object_frame

        # Strength slider for deconvolution (stars) - same as denoise
        deconv_stars_strength_frame = ttk.Frame(deconv_stars_frame)
        deconv_stars_strength_frame.pack(fill=tk.X, padx=5, pady=5)
        deconv_stars_strength_label = ttk.Label(deconv_stars_strength_frame, text="Strength:")
        deconv_stars_strength_label.pack(side=tk.LEFT, padx=5)
        deconv_stars_strength_slider = ttk.Scale(
            deconv_stars_strength_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.strength_var,
            length=200
        )
        deconv_stars_strength_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.deconv_stars_strength_value = ttk.Label(deconv_stars_strength_frame, text="0.5")
        self.deconv_stars_strength_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(deconv_stars_strength_slider, "Adjust the deconvolution strength. The "
                "result is a linear blend of the deconvolved image and the original, "
                "with the mixture of the two images controlled by the strength "
                "parameter.")
        self.strength_var.trace_add("write", self._update_deconv_stars_strength_label)

        # PSF Size slider for deconvolution (stars)
        deconv_stars_psf_frame = ttk.Frame(deconv_stars_frame)
        deconv_stars_psf_frame.pack(fill=tk.X, padx=5, pady=5)
        deconv_stars_psf_label = ttk.Label(deconv_stars_psf_frame, text="PSF Size:")
        deconv_stars_psf_label.pack(side=tk.LEFT, padx=5)
        deconv_stars_psf_slider = ttk.Scale(
            deconv_stars_psf_frame,
            from_=0.1,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.psf_size_var,
            length=200
        )
        deconv_stars_psf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.deconv_stars_psf_value = ttk.Label(deconv_stars_psf_frame, text="5.0")
        self.deconv_stars_psf_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(deconv_stars_psf_slider, "Adjust the PSF (Point Spread Function) size "
                "for star deconvolution. Values from 0.1 to 10.0 are allowed.")
        self.psf_size_var.trace_add("write", self._update_deconv_stars_psf_label)

        # Strength slider for deconvolution (object) - same as denoise
        deconv_object_strength_frame = ttk.Frame(deconv_object_frame)
        deconv_object_strength_frame.pack(fill=tk.X, padx=5, pady=5)
        deconv_object_strength_label = ttk.Label(deconv_object_strength_frame, text="Strength:")
        deconv_object_strength_label.pack(side=tk.LEFT, padx=5)
        deconv_object_strength_slider = ttk.Scale(
            deconv_object_strength_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.strength_var,
            length=200
        )
        deconv_object_strength_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.deconv_object_strength_value = ttk.Label(deconv_object_strength_frame, text="0.5")
        self.deconv_object_strength_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(deconv_object_strength_slider, "Adjust the deconvolution strength. The "
                "result is a linear blend of the deconvolved image and the original, "
                "with the mixture of the two images controlled by the strength "
                "parameter.")
        self.strength_var.trace_add("write", self._update_deconv_object_strength_label)

        # PSF Size slider for deconvolution (object)
        deconv_object_psf_frame = ttk.Frame(deconv_object_frame)
        deconv_object_psf_frame.pack(fill=tk.X, padx=5, pady=5)
        deconv_object_psf_label = ttk.Label(deconv_object_psf_frame, text="PSF Size:")
        deconv_object_psf_label.pack(side=tk.LEFT, padx=5)
        deconv_object_psf_slider = ttk.Scale(
            deconv_object_psf_frame,
            from_=0.1,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.psf_size_var,
            length=200
        )
        deconv_object_psf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.deconv_object_psf_value = ttk.Label(deconv_object_psf_frame, text="5.0")
        self.deconv_object_psf_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(deconv_object_psf_slider, "Adjust the PSF (Point Spread Function) size "
                "for object deconvolution. Values from 0.1 to 10.0 are allowed.")
        self.psf_size_var.trace_add("write", self._update_deconv_object_psf_label)

        # BGE operation parameters
        bge_frame = ttk.Frame(self.params_frame)
        self.operation_frames['bge'] = bge_frame

        # Smoothing slider for BGE
        smoothing_frame = ttk.Frame(bge_frame)
        smoothing_frame.pack(fill=tk.X, padx=5, pady=5)
        smoothing_label = ttk.Label(smoothing_frame, text="Smoothing:")
        smoothing_label.pack(side=tk.LEFT, padx=5)
        smoothing_slider = ttk.Scale(
            smoothing_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.smoothing_var,
            length=200
        )
        smoothing_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.smoothing_value = ttk.Label(smoothing_frame, text="0.5")
        self.smoothing_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(smoothing_slider, "Adjust the background extraction smoothing. "
                "Higher values result in smoother background extraction.")
        self.smoothing_var.trace_add("write", self._update_smoothing_label)

        # Correction type
        correction_frame = ttk.Frame(bge_frame)
        correction_frame.pack(fill=tk.X, padx=5, pady=5)
        combo_label = ttk.Label(correction_frame, text="Correction Type:")
        combo_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # Set up the correction type variable
        self.correction_type = tk.StringVar()
        self.correction_type.set("subtraction")  # Default value

        # Create the combobox
        self.correction_type_combo = ttk.Combobox(
            correction_frame,
            textvariable=self.correction_type,
            values=["subtraction", "division"],
            state="readonly",
            width=15
        )
        self.correction_type_combo.grid(row=0, column=1, padx=5, pady=5)

        # Keep background checkbox
        keepbg_frame = ttk.Frame(bge_frame)
        keepbg_frame.pack(fill=tk.X, padx=5, pady=2)
        keepbg_checkbox = ttk.Checkbutton(
            keepbg_frame,
            text="Keep background",
            variable=self.keep_bg_var
        )
        keepbg_checkbox.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(keepbg_checkbox, "Save the extracted background as well")

        # Advanced parameters (common for all operations)
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)

        # Batch size
        batch_size_frame = ttk.Frame(advanced_frame)
        batch_size_frame.pack(fill=tk.X, padx=5, pady=2)
        batch_size_label = ttk.Label(batch_size_frame, text="Batch Size:")
        batch_size_label.pack(side=tk.LEFT, padx=5)
        batch_size_entry = ttk.Entry(batch_size_frame, textvariable=self.batch_size_var, width=8)
        batch_size_entry.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(batch_size_entry, "AI batch size. It is recommended to leave "
                               "this at the default value")

        # GPU acceleration checkbox
        gpu_frame = ttk.Frame(advanced_frame)
        gpu_frame.pack(fill=tk.X, padx=5, pady=2)
        gpu_checkbox = ttk.Checkbutton(
            gpu_frame,
            text="Use GPU acceleration (if available)",
            variable=self.gpu_acceleration_var
        )
        gpu_checkbox.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(gpu_checkbox, "Controls use of GPU acceleration. Disable "
                               "this if you encounter problems with it enabled.")

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)

        # Apply button that handles single image and sequence
        apply_btn = ttk.Button(
            action_frame,
            text="Apply",
            command=self._on_apply
        )
        apply_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(apply_btn, "Apply the selected operation to the loaded image or sequence.")

        model_manager = GraXpertModelManager(action_frame, self.siril, self.update_dropdowns)

        # Create a button that will open the model manager dialog
        model_button = ttk.Button(
            action_frame,
            text="GraXpert Model Manager",
            command=model_manager.show_dialog
        )
        model_button.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(model_button, "Check and download remote GraXpert models")

        # Close button
        close_btn = ttk.Button(
            action_frame,
            text="Close",
            command=self.close_dialog
        )
        close_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(close_btn, "Close the script.")

        # Progress message label
        self.progress_var = tk.StringVar(value="")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack(pady=5)

    def _update_strength_label(self, *args):
        """Update the strength value label when slider changes"""
        self.strength_value.config(text=f"{self.strength_var.get():.2f}")

    def _update_smoothing_label(self, *args):
        """Update the smoothing value label when slider changes"""
        self.smoothing_value.config(text=f"{self.smoothing_var.get():.2f}")

    def _update_deconv_stars_strength_label(self, *args):
        """Update the deconvolution (stars) strength value label when slider changes"""
        self.deconv_stars_strength_value.config(text=f"{self.strength_var.get():.2f}")

    def _update_deconv_stars_psf_label(self, *args):
        """Update the deconvolution (stars) PSF size value label when slider changes"""
        self.deconv_stars_psf_value.config(text=f"{self.psf_size_var.get():.2f}")

    def _update_deconv_object_strength_label(self, *args):
        """Update the deconvolution (object) strength value label when slider changes"""
        self.deconv_object_strength_value.config(text=f"{self.strength_var.get():.2f}")

    def _update_deconv_object_psf_label(self, *args):
        """Update the deconvolution (object) PSF size value label when slider changes"""
        self.deconv_object_psf_value.config(text=f"{self.psf_size_var.get():.2f}")

    def _on_operation_selected(self, event):
        """Handle operation selection change"""
        # Get the selected display name
        display_name = self.operation_display_var.get()

        # Map back to the operation key
        operation_keys = list(self.operations.keys())
        operation_names = list(self.operations.values())
        try:
            index = operation_names.index(display_name)
            operation = operation_keys[index]
            self.selected_operation.set(operation)
        except ValueError:
            # Fallback in case of error
            operation = self.selected_operation.get() or 'denoise'

        # Update the processor based on the selected operation
        if operation == 'denoise':
            self.processor = DenoiserProcessing(self.siril)
            # Load previously saved model path from configuration
            model_path = self.processor.check_config_file()
            self.model_path_var.set(model_path or "")
        elif operation == 'bge':
            self.processor = BGEProcessing(self.siril)
            # Load previously saved model path from configuration
            model_path = self.processor.check_config_file()
            self.model_path_var.set(model_path or "")
        elif operation == 'deconvolution-stars' or operation == 'deconvolution-object':
            self.processor = DeconvolutionProcessing(self.siril)
            # Load previously saved model path from configuration
            model_path = self.processor.check_config_file(operation)
            self.model_path_var.set(model_path or "")
        # Add additional operation handlers here in future

        # Update model dropdown based on operation
        self._populate_model_dropdown()

        # Show the appropriate parameter frame and hide others
        for op, frame in self.operation_frames.items():
            if op == operation:
                frame.pack(fill=tk.X, padx=5, pady=5)
            else:
                frame.pack_forget()

        # Update the window title to reflect the current operation
        op_display_name = self.operations.get(operation, "Operation")
        self.root.title(f"GraXpert AI {op_display_name} - Siril interface v{VERSION}")

    def update_dropdowns(self):
        self._populate_model_dropdown()
        self._populate_operations_dropdown()

    def _populate_operations_dropdown(self):
        """
        Rescans available operations and updates the operations dropdown.
        Should be called when new operations are downloaded or become available.
        """
        # Get the most up-to-date operations
        self.operations = get_available_local_operations()

        # Convert operations dict to display names for dropdown
        operation_names = list(self.operations.values())
        operation_keys = list(self.operations.keys())

        # Update the dropdown with new values
        self.op_dropdown['values'] = operation_names

        # Get the current selected operation key (if any)
        current_key = self.selected_operation.get()

        if current_key and current_key in self.operations:
            # If current selection is still available, keep it selected
            display_name = self.operations[current_key]
            self.operation_display_var.set(display_name)
        elif operation_keys:
            # Otherwise select the first available operation
            first_key = operation_keys[0]
            first_display_name = operation_names[0]

            self.selected_operation.set(first_key)
            self.operation_display_var.set(first_display_name)

            # Since selection changed, update processor and UI
            self._on_operation_selected(None)
        else:
            # If no operations are available
            self.op_dropdown['values'] = ["No operations available"]
            self.operation_display_var.set("No operations available")
            self.selected_operation.set("")

            # Hide all parameter frames
            for frame in self.operation_frames.values():
                frame.pack_forget()

        # Update window title
        current_key = self.selected_operation.get()
        if current_key and current_key in self.operations:
            op_display_name = self.operations[current_key]
            self.root.title(f"GraXpert AI {op_display_name} - Siril interface v{VERSION}")
        else:
            self.root.title(f"GraXpert AI - Siril interface v{VERSION}")

    def _populate_model_dropdown(self):
        """Populate the model dropdown with available models for the current operation"""
        operation = self.selected_operation.get()

        # Get model directory name based on operation
        model_dir = None
        if operation == 'denoise':
            model_dir = "denoise-ai-models"
        elif operation == 'bge':
            model_dir = "bge-ai-models"
        elif operation == 'deconvolution-stars':
            model_dir = "deconvolution-stars-ai-models"
        elif operation == 'deconvolution-object':
            model_dir = "deconvolution-object-ai-models"
        else:
            # Default to denoise models if operation not recognized
            model_dir = "denoise-ai-models"

        # Dictionary to store model name -> full path mapping
        model_paths = get_available_local_models(model_dir)

        # Update the dropdown values
        if model_paths:
            # Sort model names alphabetically
            model_names = sorted(model_paths.keys())
            self.model_dropdown['values'] = model_names

            # Store the full path mapping for when selection changes
            self.model_path_mapping = model_paths

            # Get the previously selected model path from model_path_var
            current_path = self.model_path_var.get()

            # Find the model name associated with the current path
            selected_model = None
            for name, path in model_paths.items():
                if path == current_path:
                    selected_model = name
                    break

            if selected_model and selected_model in model_names:
                # Set the dropdown to the previously selected model
                self.model_name_var.set(selected_model)
            else:
                # If no match found or no previous selection, select the first model
                self.model_dropdown.current(0)
                selected_model = model_names[0]
                self.model_name_var.set(selected_model)
                # Set the full path in the hidden variable
                self.model_path_var.set(model_paths[selected_model])
        else:
            self.model_dropdown['values'] = ["No models found"]
            self.model_dropdown.current(0)
            self.model_name_var.set("No models found")
            self.model_path_var.set("")

        # Bind the selection event to update the model_path_var
        self.model_dropdown.bind("<<ComboboxSelected>>", self._on_model_selected)

    def _on_model_selected(self, event):
        """Update the model_path_var when a model is selected from dropdown"""
        selected_model = self.model_dropdown.get()
        if selected_model in self.model_path_mapping:
            # Set the full path in the hidden variable
            self.model_path_var.set(self.model_path_mapping[selected_model])
            # The display shows only the model name, which is already handled by
            # self.model_name_var

            # Save config with updated model path
            self.processor.save_config_file(self.model_path_var.get())
            # Reset the cached image to force recalculation
            self.processor.reset_cache()

    def _on_apply(self):
        """
        Handle the 'Apply' button click event.
        This method starts the appropriate processing thread based on the current state and
        uses the processor class methods as thread targets.
        """
        model_path = self.model_path_var.get()
        operation = self.selected_operation.get()

        # Get processing parameters from UI
        batch_size = int(self.batch_size_var.get())
        gpu_acceleration = self.gpu_acceleration_var.get()

        # Validate model path if needed for new processing
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid ONNX model file.")
            return

        if self.siril.is_image_loaded():
            if operation == "bge":
                correction_type = self.correction_type.get()
                smoothing = float(self.smoothing_var.get())

                # Cache the original image
                self.processor.cached_original_image = self.siril.get_image_pixeldata()

                # Reshape mono images to 3D with a channels size of 1
                if self.processor.cached_original_image.ndim == 2:
                    self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                filename = None
                header = None
                keep_bg = self.keep_bg_var.get()
                if keep_bg:
                    filename = self.siril.get_image_filename()
                    header = self.siril.get_image_fits_header()

                # Start image processing thread
                threading.Thread(
                    target=self.processor.process_image,
                    args=(
                        model_path,
                        correction_type,
                        smoothing,
                        keep_bg,
                        filename,
                        header,
                        gpu_acceleration,
                        self._update_progress
                    ),
                    daemon=True
                ).start()

            elif operation == "denoise":
                strength = float(self.strength_var.get())
                if self.processor.cached_processed_image is None:
                    # Cache the original image if this is first-time processing
                    self.processor.cached_original_image = \
                                self.siril.get_image_pixeldata()

                    # Reshape mono images to 3D with a channels size of 1
                    if self.processor.cached_original_image.ndim == 2:
                        self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                    # Start image processing thread using processor's method
                    threading.Thread(
                        target=self.processor.process_image,
                        args=(model_path, strength, batch_size,
                            gpu_acceleration, self._update_progress),
                        daemon=True
                    ).start()
                else:
                    # Apply operation-specific blend
                    threading.Thread(
                        target=lambda: self.processor.apply_blend(strength),
                        daemon=True
                    ).start()
            elif operation == "deconvolution-stars":
                strength = float(self.strength_var.get())
                psf_size = float(self.psf_size_var.get())

                if self.processor.cached_processed_image is None:
                    # Cache the original image if this is first-time processing
                    self.processor.cached_original_image = \
                                self.siril.get_image_pixeldata()

                # Reshape mono images to 3D with a channels size of 1
                if self.processor.cached_original_image.ndim == 2:
                    self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                self._update_progress(f"Processing image with stars deconvolution (PSF size: {psf_size:.2f}, strength: {strength:.2f})")
                threading.Thread(
                    target=self.processor.process_image,
                    args=(model_path, strength, psf_size, batch_size,
                          gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()
            elif operation == "deconvolution-object":
                strength = float(self.strength_var.get())
                psf_size = float(self.psf_size_var.get())

                if self.processor.cached_processed_image is None:
                    # Cache the original image if this is first-time processing
                    self.processor.cached_original_image = \
                                self.siril.get_image_pixeldata()

                # Reshape mono images to 3D with a channels size of 1
                if self.processor.cached_original_image.ndim == 2:
                    self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                self._update_progress(f"Processing image with object deconvolution (PSF size: {psf_size:.2f}, strength: {strength:.2f})")
                threading.Thread(
                    target=self.processor.process_image,
                    args=(model_path, strength, psf_size, batch_size,
                          gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()
        elif self.siril.is_sequence_loaded():
            if operation == "bge":
                correction_type = self.correction_type.get()
                smoothing = float(self.smoothing_var.get())
                # Get current sequence name
                sequence_name = self.siril.get_seq().seqname
                keep_bg = self.keep_bg_var.get()

                # Start sequence processing thread using processor's method
                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, correction_type, smoothing,
                        keep_bg, gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()
            elif operation == "denoise":
                strength = float(self.strength_var.get())
                # Get current sequence name
                sequence_name = self.siril.get_seq().seqname

                # Start sequence processing thread using processor's method
                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, strength, batch_size,
                        gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()
            elif operation == "deconvolution-stars":
                strength = float(self.strength_var.get())
                psf_size = float(self.psf_size_var.get())
                sequence_name = self.siril.get_seq().seqname

                # Placeholder for deconvolution-stars sequence processing
                self._update_progress(f"Processing {sequence_name} with stars deconvolution (PSF size: {psf_size:.2f}, strength: {strength:.2f})")
                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, strength, psf_size, batch_size,
                        gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()

            elif operation == "deconvolution-object":
                strength = float(self.strength_var.get())
                psf_size = float(self.psf_size_var.get())
                sequence_name = self.siril.get_seq().seqname

                self._update_progress(f"Processing {sequence_name} with object deconvolution (PSF size: {psf_size:.2f}, strength: {strength:.2f})")
                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, strength, psf_size, batch_size,
                        gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()

            else:
                print("Operation not handled")

        else:
            messagebox.showerror("Error", "No sequence or image is loaded.")

    def _update_progress(self, message, progress=0):
        self.progress_var.set(message)
        self.siril.update_progress(message, progress)

    def close_dialog(self):
        """ Close the dialog """
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()

class DenoiserCLI:
    """CLI interface for GraXpert AI Denoise."""

    def __init__(self, siril, args=None):
        """ init method """
        # Parse command line arguments
        if not siril:
            raise ValueError("No SirilInterface provided to DenoiserCLI()")

        self.siril = siril
        if args is None:
            args = sys.argv[1:]

        self.args = self.parse_arguments(args)
        if self.args.listmodels:
            models_dir = os.path.join(user_data_dir(appname="GraXpert"), "denoise-ai-models")
            list_available_models(models_dir)
            return

        # Initialize the processing class
        self.processor = DenoiserProcessing(self.siril)

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if not (image_loaded or seq_loaded):
            self.error("No image or sequence loaded")
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            self.error("Error: this script requires Siril 1.4.0-beta2 or higher")
            return

        # Find and set the model path based on arguments
        self.model_path = self.get_model_path()
        if not self.model_path:
            self.error("No valid model found")
            return

        # Start processing
        if image_loaded:
            self.process_image()
        else:
            self.process_sequence()

    def parse_arguments(self, args):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"GraXpert AI Denoise - Siril CLI v{VERSION}")
        parser.add_argument("-strength", type=float, default=0.5, help="Denoising strength (0.0-1.0)")
        parser.add_argument("-batch", type=int, default=4, help="Batch size for processing")
        parser.add_argument("-model", type=str, help="Model name to use (directory name in GraXpert models folder)")

        # Boolean flag for GPU usage - store_true/store_false approach
        parser.add_argument("-gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
        parser.add_argument("-nogpu", action="store_true", default=False, help="Disable GPU acceleration")
        # List models flag
        parser.add_argument("-listmodels", action="store_true", help="List available models and exit")

        return parser.parse_args(args)

    def get_model_path(self):
        """Get the model path based on user choice or highest available version."""
        # Get the GraXpert directory
        models_dir = os.path.join(user_data_dir(appname="GraXpert"), "denoise-ai-models")

        # Check if directory exists
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            self.error(f"Models directory not found: {models_dir}")
            return None

        # Find all available models
        available_models = {}
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    available_models[subdir] = model_path

        if not available_models:
            self.error("No models found")
            return None

        # If model specified, use it if available
        if self.args.model:
            if self.args.model in available_models:
                model_path = available_models[self.args.model]
                print(f"Using specified model: {self.args.model}")
                return model_path
            self.error(f"Specified model '{self.args.model}' not found. Available models: {', '.join(available_models.keys())}")
            return None

        # Otherwise use the highest available version
        model_names = sorted(available_models.keys())
        highest_model = model_names[-1]
        print(f"Using highest available model: {highest_model}")

        # Save the selected model to config
        self.processor.save_config_file(available_models[highest_model])

        return available_models[highest_model]

    def process_image(self):
        """Process a single image."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        print(f"Processing image with strength={self.args.strength}, batch={self.args.batch}, gpu={use_gpu}")

        # Cache the original image
        self.processor.cached_original_image = self.siril.get_image_pixeldata()

        # Reshape mono images to 3D with a channels size of 1
        if self.processor.cached_original_image.ndim == 2:
            self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

        # Start image processing thread
        thread = threading.Thread(
            target=self.processor.process_image,
            args=(
                self.model_path,
                self.args.strength,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Image processing complete")

    def process_sequence(self):
        """Process a sequence."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        sequence_name = self.siril.get_seq().seqname
        print(f"Processing sequence {sequence_name} with strength={self.args.strength}, batch={self.args.batch}, gpu={use_gpu}")

        # Start sequence processing thread
        thread = threading.Thread(
            target=self.processor.process_sequence,
            args=(
                sequence_name,
                self.model_path,
                self.args.strength,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Sequence processing complete")

    def update_progress(self, message, progress=0):
        """Update progress information."""
        self.siril.update_progress(message, progress)

    def error(self, message):
        """Print error message and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        if hasattr(self, 'siril'):
            self.siril.disconnect()
        sys.exit(1)

class DeconvolutionCLI:
    """CLI interface for GraXpert AI Deconvolution."""

    def __init__(self, siril, deconv_type, args=None):
        """ init method """
        # Parse command line arguments
        if not siril:
            raise ValueError("No SirilInterface provided to DeconvolutionCLI()")

        self.siril = siril
        if args is None:
            args = sys.argv[1:]
        self.deconv_obj = True if deconv_type == "deconv_obj" else False
        self.deconv_stellar = True if deconv_type == "deconv_stellar" else False

        self.args = self.parse_arguments(args)
        self.folder = None
        if self.deconv_obj:
            self.folder = "deconvolution-object-ai-models"
        elif self.deconv_stellar:
            self.folder = "deconvolution-stars-ai-models"
        if self.folder is None:
            self.error("Incorrect argument")
            return

        if self.args.listmodels:
            models_dir = os.path.join(user_data_dir(appname="GraXpert"), self.folder)
            list_available_models(models_dir)
            return

        # Initialize the processing class
        self.processor = DeconvolutionProcessing(self.siril)

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if not (image_loaded or seq_loaded):
            self.error("No image or sequence loaded")
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            self.error("Error: this script requires Siril 1.4.0-beta2 or higher")
            return

        # Find and set the model path based on arguments
        self.model_path = self.get_model_path()
        if not self.model_path:
            self.error("No valid model found")
            return

        # Start processing
        if image_loaded:
            self.process_image()
        else:
            self.process_sequence()

    def parse_arguments(self, args):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"GraXpert AI Deconvolution - Siril CLI v{VERSION}")
        parser.add_argument("-strength", type=float, default=0.5, help="Deconvolution strength (0.0-1.0)")
        parser.add_argument("-psfsize", type=float, default=5.0, help="Point Spread Function size")
        parser.add_argument("-batch", type=int, default=4, help="Batch size for processing")
        parser.add_argument("-model", type=str, help="Model name to use (directory name in GraXpert models folder)")

        # Boolean flag for GPU usage - store_true/store_false approach
        parser.add_argument("-gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
        parser.add_argument("-nogpu", action="store_true", default=False, help="Disable GPU acceleration")
        # List models flag
        parser.add_argument("-listmodels", action="store_true", help="List available models and exit")

        return parser.parse_args(args)

    def get_model_path(self):
        """Get the model path based on user choice or highest available version."""
        # Get the GraXpert directory
        models_dir = os.path.join(user_data_dir(appname="GraXpert"), self.folder)

        # Check if directory exists
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            self.error(f"Models directory not found: {models_dir}")
            return None

        # Find all available models
        available_models = {}
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    available_models[subdir] = model_path

        if not available_models:
            self.error("No models found")
            return None

        # If model specified, use it if available
        if self.args.model:
            if self.args.model in available_models:
                model_path = available_models[self.args.model]
                print(f"Using specified model: {self.args.model}")
                return model_path
            self.error(f"Specified model '{self.args.model}' not found. Available models: {', '.join(available_models.keys())}")
            return None

        # Otherwise use the highest available version
        model_names = sorted(available_models.keys())
        highest_model = model_names[-1]
        print(f"Using highest available model: {highest_model}")

        # Save the selected model to config
        self.processor.save_config_file(available_models[highest_model])

        return available_models[highest_model]

    def process_image(self):
        """Process a single image."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        print(f"Processing image with strength={self.args.strength}, psfsize={self.args.psfsize}, batch={self.args.batch}, gpu={use_gpu}")

        # Cache the original image
        self.processor.cached_original_image = self.siril.get_image_pixeldata()

        # Reshape mono images to 3D with a channels size of 1
        if self.processor.cached_original_image.ndim == 2:
            self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

        # Start image processing thread
        thread = threading.Thread(
            target=self.processor.process_image,
            args=(
                self.model_path,
                self.args.strength,
                self.args.psfsize,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Image processing complete")

    def process_sequence(self):
        """Process a sequence."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        sequence_name = self.siril.get_seq().seqname
        print(f"Processing sequence {sequence_name} with strength={self.args.strength}, psfsize={self.args.psfsize}, batch={self.args.batch}, gpu={use_gpu}")

        # Start sequence processing thread
        thread = threading.Thread(
            target=self.processor.process_sequence,
            args=(
                sequence_name,
                self.model_path,
                self.args.strength,
                self.args.psfsize,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Sequence processing complete")

    def update_progress(self, message, progress=0):
        """Update progress information."""
        self.siril.update_progress(message, progress)

    def error(self, message):
        """Print error message and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        if hasattr(self, 'siril'):
            self.siril.disconnect()
        sys.exit(1)

class BackgroundExtractionCLI:
    """CLI interface for GraXpert AI Denoise."""

    def __init__(self, siril, args=None):
        """ init method """
        # Parse command line arguments
        if not siril:
            raise ValueError("No SirilInterface provided to BackgroundExtractionCLI()")

        self.siril = siril
        if args is None:
            args = sys.argv[1:]

        self.args = self.parse_arguments(args)
        if self.args.listmodels:
            models_dir = os.path.join(user_data_dir(appname="GraXpert"), "bge-ai-models")
            list_available_models(models_dir)
            return

        # Initialize the processing class
        self.processor = BGEProcessing(self.siril)

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if not (image_loaded or seq_loaded):
            self.error("No image or sequence loaded")
            return

        # Find and set the model path based on arguments
        self.model_path = self.get_model_path()
        if not self.model_path:
            self.error("No valid model found")
            return

        # Start processing
        if image_loaded:
            self.process_image()
        else:
            self.process_sequence()

    def parse_arguments(self, args):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"GraXpert AI BG Extraction - Siril CLI v{VERSION}")
        parser.add_argument("-correction", type=str, default="subtraction", help="Correction type ('subtraction' (default) or 'division')")
        parser.add_argument("-smoothing", type=float, default=0.5, help="Smoothing (0.0-1.0)")
        parser.add_argument("-model", type=str, help="Model name to use (directory name in GraXpert models folder)")
        parser.add_argument("-keep_bg", action="store_true", help="Keep the extracted background")

        # Boolean flag for GPU usage - store_true/store_false approach
        parser.add_argument("-gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
        parser.add_argument("-nogpu", action="store_true", default=False, help="Disable GPU acceleration")
        # List models flag
        parser.add_argument("-listmodels", action="store_true", help="List available models and exit")

        return parser.parse_args(args)

    def get_model_path(self):
        """Get the model path based on user choice or highest available version."""
        # Get the GraXpert directory
        models_dir = os.path.join(user_data_dir(appname="GraXpert"), "bge-ai-models")

        # Check if directory exists
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            self.error(f"Models directory not found: {models_dir}")
            return None

        # Find all available models
        available_models = {}
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    available_models[subdir] = model_path

        if not available_models:
            self.error("No models found")
            return None

        # If model specified, use it if available
        if self.args.model:
            if self.args.model in available_models:
                model_path = available_models[self.args.model]
                print(f"Using specified model: {self.args.model}")
                return model_path
            self.error(f"Specified model '{self.args.model}' not found. Available models: {', '.join(available_models.keys())}")
            return None

        # Otherwise use the highest available version
        model_names = sorted(available_models.keys())
        highest_model = model_names[-1]
        print(f"Using latest available model: {highest_model}")

        # Save the selected model to config
        self.processor.save_config_file(available_models[highest_model])

        return available_models[highest_model]

    def process_image(self):
        """Process a single image."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        print(f"Processing image with correction={self.args.correction}, smoothing={self.args.smoothing}, keep_bg={self.args.keep_bg}, gpu={use_gpu}")

        # Cache the original image
        self.processor.cached_original_image = self.siril.get_image_pixeldata()

        # Reshape mono images to 3D with a channels size of 1
        if self.processor.cached_original_image.ndim == 2:
            self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

        filename = self.siril.get_image_filename()
        header = self.siril.get_image_fits_header()

        # Start image processing thread
        thread = threading.Thread(
            target=self.processor.process_image,
            args=(
                self.model_path,
                self.args.correction,
                self.args.smoothing,
                self.args.keep_bg,
                filename,
                header,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Image processing complete")

    def process_sequence(self):
        """Process a sequence."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        sequence_name = self.siril.get_seq().seqname
        print(f"Processing sequence {sequence_name} with strength={self.args.strength}, batch={self.args.batch}, gpu={use_gpu}")

        # Start sequence processing thread
        thread = threading.Thread(
            target=self.processor.process_sequence,
            args=(
                sequence_name,
                self.model_path,
                self.args.correction,
                self.args.smoothing,
                self.args.keep_bg,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Sequence processing complete")

    def update_progress(self, message, progress=0):
        """Update progress information."""
        self.siril.update_progress(message, progress)

    def error(self, message):
        """Print error message and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        if hasattr(self, 'siril'):
            self.siril.disconnect()
        sys.exit(1)

def main():
    """ main entry point into the script """
    siril = s.SirilInterface()
    try:
        siril.connect()
        check_graxpert_version(get_executable(siril))

        if siril.is_cli():
            # Top level argument parser
            parser = argparse.ArgumentParser(
                description=f"GraXpert AI Tools - Siril CLI v{VERSION}",
                add_help=False  # Disable the default help so we can handle it ourselves
            )

            # Add help argument manually
            parser.add_argument('-h', '--help', action='store_true',
                               help='Show this help message and exit')

            # Tool selection arguments
            group = parser.add_mutually_exclusive_group()
            group.add_argument("-denoise", action="store_true", help="Use the AI Denoising tool")
            group.add_argument("-deconv_obj", action="store_true", help="Use the AI Object Deconvolution tool")
            group.add_argument("-deconv_stellar", action="store_true", help="Use the AI Stellar Deconvolution tool")
            group.add_argument("-bge", action="store_true", help="Use the AI Background Extraction tool")

            # First parse, just to get the tool type (parse_known_args ignores unknown args)
            args, remaining_args = parser.parse_known_args()

            # If general help is requested, print it and exit
            if not (args.denoise or args.bge or args.deconv_obj or args.deconv_stellar):
                parser.print_help()
                sys.exit(0)

            # Determine which tool to use and handle tool-specific help
            if args.denoise:
                # If help is requested for the denoiser
                if args.help:
                    # Create parser just to show help
                    denoise_parser = argparse.ArgumentParser(
                        description=f"GraXpert AI Denoise - Siril CLI v{VERSION}")
                    # Add denoiser-specific arguments
                    denoise_parser.add_argument("-strength", type=float, default=1.0,
                                               help="Denoising strength (0.0-1.0)")
                    denoise_parser.add_argument("-batch", type=int, default=4,
                                               help="Batch size for processing")
                    denoise_parser.add_argument("-model", type=str,
                                               help="Model name to use (directory name in GraXpert models folder)")
                    denoise_parser.add_argument("-gpu", action="store_true", default=True,
                                               help="Enable GPU acceleration (default)")
                    denoise_parser.add_argument("-nogpu", action="store_true", default=False,
                                               help="Disable GPU acceleration")
                    denoise_parser.add_argument("-listmodels", action="store_true",
                                               help="List available models and exit")
                    denoise_parser.print_help()
                    sys.exit(0)

                # Otherwise proceed with denoiser
                DenoiserCLI(siril, remaining_args)
            elif args.deconv_obj or args.deconv_stellar:
                # If help is requested for the denoiser
                if args.help:
                    # Create parser just to show help
                    denoise_parser = argparse.ArgumentParser(
                        description=f"GraXpert AI Deconvolution - Siril CLI v{VERSION}")
                    # Add denoiser-specific arguments
                    denoise_parser.add_argument("-strength", type=float, default=1.0,
                                               help="Denoising strength (0.0-1.0)")
                    denoise_parser.add_argument("-psfsize", type=float, default=0.3,
                                               help="PSF size: adjust to suit your seeing")
                    denoise_parser.add_argument("-batch", type=int, default=4,
                                               help="Batch size for processing")
                    denoise_parser.add_argument("-model", type=str,
                                               help="Model name to use (directory name in GraXpert models folder)")
                    denoise_parser.add_argument("-gpu", action="store_true", default=True,
                                               help="Enable GPU acceleration (default)")
                    denoise_parser.add_argument("-nogpu", action="store_true", default=False,
                                               help="Disable GPU acceleration")
                    denoise_parser.add_argument("-listmodels", action="store_true",
                                               help="List available models and exit")
                    denoise_parser.print_help()
                    sys.exit(0)

                deconv_type = "deconv_obj" if args.deconv_obj else "deconv_stellar"
                DeconvolutionCLI(siril, deconv_type, remaining_args)
            elif args.bge:
                # If help is requested for background extraction
                if args.help:
                    # Create parser just to show help
                    bge_parser = argparse.ArgumentParser(
                        description=f"GraXpert AI BG Extraction - Siril CLI v{VERSION}")
                    # Add bge-specific arguments
                    bge_parser.add_argument("-correction", type=str, default="subtraction",
                                           help="Correction type ('subtraction' (default) or 'division')")
                    bge_parser.add_argument("-smoothing", type=float, default=1.0,
                                           help="Smoothing (0.0-1.0)")
                    bge_parser.add_argument("-batch", type=int, default=4,
                                           help="Batch size for processing")
                    bge_parser.add_argument("-model", type=str,
                                           help="Model name to use (directory name in GraXpert models folder)")
                    bge_parser.add_argument("-gpu", action="store_true", default=True,
                                           help="Enable GPU acceleration (default)")
                    bge_parser.add_argument("-nogpu", action="store_true", default=False,
                                           help="Disable GPU acceleration")
                    bge_parser.add_argument("-listmodels", action="store_true",
                                           help="List available models and exit")
                    bge_parser.print_help()
                    sys.exit(0)

                # Otherwise proceed with background extraction
                BackgroundExtractionCLI(siril, remaining_args)
            else:
                # No tool specified - show available tools and default to denoiser
                print("No tool specified. Use -denoise, -bge, -deconv-stellar or -deconv-obj to select a tool.")
        else:
            # GUI mode remains unchanged
            root = ThemedTk()
            GUIInterface(root, siril)
            root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
