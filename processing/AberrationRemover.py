# Aberration Remover AI by Riccardo Alberghi
# Script version: 1.0.1
# SPDX-License-Identifier: GPL-3.0-or-later

# 1.0.0 Original release
# 1.0.1 bugfix: fixed incorrect handling of 16 bits images

import sirilpy as s
from sirilpy import tksiril

s.ensure_installed("ttkthemes", "numpy", "requests", "subprocess", "platform")

import platform

# Determine the correct onnxruntime package based on OS and hardware
def detect_nvidia_gpu():
    # Try to detect NVIDIA GPU by checking for nvidia-smi
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False

onnxruntime_pkg = "onnxruntime"
system = platform.system().lower()

if system == "windows":
    if detect_nvidia_gpu():
        onnxruntime_pkg = "onnxruntime-gpu"
    else:
        onnxruntime_pkg = "onnxruntime-directml"
elif system == "linux":
    if detect_nvidia_gpu():
        onnxruntime_pkg = "onnxruntime-gpu"

s.ensure_installed(onnxruntime_pkg)

import os
import sys
import threading
import tkinter as tk
import requests
from tkinter import ttk, filedialog, messagebox
import webbrowser

import numpy as np
import onnxruntime
from ttkthemes import ThemedTk

VERSION = "1.0.1"
CONFIG_FILENAME = "aberration_remover_model.conf"


class DeconvolutionAIInterface:
    def __init__(self, root, github_repo="riccardoalberghi/abberation_models"):
        self.root = root
        self.root.title(f"Aberrations Remover - v{VERSION}")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        # Fetch latest GitHub release version at init time
        self.latest_github_version = self._fetch_latest_github_release_version(github_repo)
        # Optionally, you can print or log this value for debugging
        print(f"Latest GitHub release version for {github_repo}: {self.latest_github_version}")

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

        # Load previously saved model path and max version used from configuration
        model_path, stored_max = self.check_config_file()

        # Initialize model path variable for UI
        self.model_path_var = tk.StringVar(value=model_path or "")

        # Keep stored max version for update comparisons
        self.stored_max = stored_max

        tksiril.match_theme_to_siril(self.root, self.siril)

        # Create widgets
        self.create_widgets()

        # Initial model update check based on saved config
        model_path, stored_max = self.check_config_file()
        if model_path:
            ai_version = None
            try:
                session = onnxruntime.InferenceSession(model_path)
                meta = session.get_modelmeta()
                ai_version = meta.custom_metadata_map.get("ai_version", None)
            except Exception as e:
                print(f"Could not read ai_version from saved model: {e}")
            if ai_version:
                ai_cmp = ai_version.lstrip("vV")
                latest_cmp = (self.latest_github_version or "").lstrip("vV")
                def version_tuple(v):
                    try:
                        return tuple(map(int, v.split(".")))
                    except:
                        return ()
                # Display update if saved model is older than latest GitHub release
                if latest_cmp and version_tuple(ai_cmp) < version_tuple(latest_cmp):
                    self.model_update_var.set("A new model version is available to download.")
                else:
                    self.model_update_var.set("")

    def _fetch_latest_github_release_version(self, repo):
        """
        Fetch the latest release version from the specified GitHub repository.
        :param repo: str, in the form "owner/repo"
        :return: str, version tag or None if not found/error
        """
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("tag_name")
            else:
                print(f"GitHub API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error fetching GitHub release version: {e}")
        return None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title and version
        title_label = ttk.Label(
            main_frame,
            text="Aberrations Remover by Riccardo Alberghi",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 5))
        version_label = ttk.Label(main_frame, text=f"Script version: {VERSION}")
        version_label.pack(pady=(0, 10))

        # Separator
        sep = ttk.Separator(main_frame, orient='horizontal')
        sep.pack(fill=tk.X, pady=5)

        # Model Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        load_model_btn = ttk.Button(
            button_frame,
            text="Load model",
            command=self._browse_model
        )
        load_model_btn.pack(side=tk.LEFT, padx=5)

        download_model_btn = ttk.Button(
            button_frame,
            text="Download model & info",
            command=self._download_model
        )
        download_model_btn.pack(side=tk.RIGHT, padx=5)

        # Display selected model path
        model_entry = ttk.Entry(
            main_frame,
            textvariable=self.model_path_var,
            width=50
        )
        model_entry.pack(fill=tk.X, padx=5, pady=5)

        # Model update message label (for new model available)
        self.model_update_var = tk.StringVar(value="")
        model_update_label = ttk.Label(main_frame, textvariable=self.model_update_var, foreground="red")
        model_update_label.pack(pady=2)

        # Calculate button
        calc_btn = ttk.Button(
            main_frame,
            text="Calculate",
            command=self._on_calculate
        )
        calc_btn.pack(pady=10)

        # Progress message label
        self.progress_var = tk.StringVar(value="")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack(pady=5)

    def _browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select ONNX Model",
            filetypes=[("ONNX Model", "*.onnx")],
            initialdir=os.path.expanduser("~")
        )
        if filename:
            self.model_path_var.set(filename)
            # Read stored max_version_used from config (if any)
            _, stored_max = self.check_config_file()
            # Try to extract ai_version from ONNX model metadata
            ai_version = None
            try:
                session = onnxruntime.InferenceSession(filename)
                meta = session.get_modelmeta()
                ai_version = meta.custom_metadata_map.get("ai_version", None)
            except Exception as e:
                print(f"Could not read ai_version from model: {e}")
            # Compare versions and compute new max_version_used
            show_update = False
            new_max = stored_max
            if ai_version is not None:
                # Normalize version strings
                ai_cmp = ai_version.lstrip("vV")
                latest_cmp = (self.latest_github_version or "").lstrip("vV")
                stored_cmp = (stored_max or "").lstrip("vV")
                # Version comparison helper
                def version_tuple(v):
                    try:
                        return tuple(map(int, v.split(".")))
                    except:
                        return ()
                # Update stored max if this ai_version is newer
                if not stored_cmp or version_tuple(ai_cmp) > version_tuple(stored_cmp):
                    new_max = ai_version
                # Show update if loaded model version is older than latest GitHub release
                if latest_cmp and version_tuple(ai_cmp) < version_tuple(latest_cmp):
                    show_update = True
            # Display update message
            self.model_update_var.set("A new model version is available to download." if show_update else "")
            # Save config with updated max_version_used
            self.save_config_file(filename, new_max)

    def _download_model(self):
        webbrowser.open("https://github.com/riccardoalberghi/abberation_models/releases/latest")

    def _on_calculate(self):
        model_path = self.model_path_var.get()
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid ONNX model file.")
            return
        # Always start the calculation thread; thread safety is handled in _calculate_deconvolution
        threading.Thread(target=self._calculate_deconvolution, daemon=True).start()

    def _calculate_deconvolution(self):
        try:
            with self.siril.image_lock():
                self._update_progress("Loading ONNX model...")
                model_path = self.model_path_var.get()
                session = onnxruntime.InferenceSession(model_path)

                self._update_progress("Fetching image data...")
                # Get the currently loaded image as a NumPy array.
                pixel_data = self.siril.get_image_pixeldata()  # Assumes a NumPy array is returned.

                # Ensure pixel_data is in channels-first format.
                original_format = None
                original_dtype = pixel_data.dtype
                if pixel_data.ndim == 2:
                    # Mono image, shape (H, W)
                    pixel_data = pixel_data[np.newaxis, ...]  # Convert to shape (1, H, W)
                    original_format = "mono"
                elif pixel_data.ndim == 3:
                    # Check if channels are first or last
                    if pixel_data.shape[0] in [1, 3]:
                        # Already channels-first: (C, H, W)
                        original_format = "channels_first"
                    elif pixel_data.shape[2] in [1, 3]:
                        # Channels-last: (H, W, C)
                        pixel_data = np.transpose(pixel_data, (2, 0, 1))
                        original_format = "channels_last"
                    else:
                        raise ValueError("Unsupported image shape for pixel_data: {}".format(pixel_data.shape))
                else:
                    raise ValueError("Unsupported number of dimensions for pixel_data: {}".format(pixel_data.ndim))

                # Normalize if pixel values exceed [0,1]
                if original_dtype == np.uint16:
                    pixel_data = pixel_data.astype(np.float32) / 65535.0

                # Save undo state.
                self.siril.undo_save_state("Aberrations Remover")

                # Define patch parameters.
                patch_size = 512
                overlap = 64
                stride = patch_size - overlap

                # Create a 2D Hann window for smooth blending
                hann1d = np.hanning(patch_size)
                window2d = np.outer(hann1d, hann1d).astype(np.float32)
                window2d = window2d / window2d.max()  # Normalize to max 1.0

                _, H, W = pixel_data.shape

                # If mono (1, H, W), process as before.
                # If RGB (3, H, W) but model expects mono, process each channel independently.
                if pixel_data.shape[0] == 1:
                    # Mono image
                    output_image = np.zeros_like(pixel_data, dtype=np.float32)
                    weight_image = np.zeros_like(pixel_data, dtype=np.float32)

                    h_starts = self._get_patch_indices(H, patch_size, overlap)
                    w_starts = self._get_patch_indices(W, patch_size, overlap)

                    total_patches = len(h_starts) * len(w_starts)
                    patch_count = 0

                    import time
                    self._update_progress("Process started.")
                    start_time = time.time()
                    ten_percent_patch = max(1, int(0.1 * total_patches))
                    for i in h_starts:
                        for j in w_starts:
                            patch_count += 1
                            patch = np.copy(pixel_data[:, i:i + patch_size, j:j + patch_size])
                            input_patch = patch.astype(np.float32)
                            input_batch = np.expand_dims(input_patch, axis=0)
                            inputs = session.get_inputs()
                            input_dict = {inputs[0].name: input_batch}
                            outputs = session.run(None, input_dict)
                            processed_patch = np.squeeze(outputs[0], axis=0)
                            # Apply blending window
                            weighted_patch = processed_patch * window2d
                            output_image[:, i:i + patch_size, j:j + patch_size] += weighted_patch
                            weight_image[:, i:i + patch_size, j:j + patch_size] += window2d
                            self._update_progress(f"Patches done: {patch_count}/{total_patches}", patch_count/total_patches)

                    weight_image[weight_image == 0] = 1.0
                    final_image = output_image / weight_image

                elif pixel_data.shape[0] == 3:
                    # RGB image, process each channel independently
                    output_image = np.zeros_like(pixel_data, dtype=np.float32)
                    weight_image = np.zeros_like(pixel_data, dtype=np.float32)

                    h_starts = self._get_patch_indices(H, patch_size, overlap)
                    w_starts = self._get_patch_indices(W, patch_size, overlap)

                    total_patches = len(h_starts) * len(w_starts)
                    for c in range(3):
                        patch_count = 0
                        import time
                        self._update_progress(f"Process started on channel {c+1}/3.")
                        start_time = time.time()
                        ten_percent_patch = max(1, int(0.1 * total_patches))
                        for i in h_starts:
                            for j in w_starts:
                                patch_count += 1
                                patch = np.copy(pixel_data[c:c+1, i:i + patch_size, j:j + patch_size])  # shape (1, patch_size, patch_size)
                                input_patch = patch.astype(np.float32)
                                input_batch = np.expand_dims(input_patch, axis=0)
                                inputs = session.get_inputs()
                                input_dict = {inputs[0].name: input_batch}
                                outputs = session.run(None, input_dict)
                                processed_patch = np.squeeze(outputs[0], axis=0)  # shape (1, patch_size, patch_size)
                                # Apply blending window
                                weighted_patch = processed_patch * window2d
                                output_image[c:c+1, i:i + patch_size, j:j + patch_size] += weighted_patch
                                weight_image[c:c+1, i:i + patch_size, j:j + patch_size] += window2d
                                self._update_progress(f"Patches done (channel {c+1}/3): {patch_count}/{total_patches}", patch_count/total_patches)

                    weight_image[weight_image == 0] = 1.0
                    final_image = output_image / weight_image

                else:
                    raise ValueError("Only mono (1 channel) or RGB (3 channel) images are supported.")

                # Restore output to original format if needed
                if original_format == "channels_last":
                    # Convert back to (H, W, C)
                    final_image = np.transpose(final_image, (1, 2, 0))
                elif original_format == "mono":
                    # Convert back to (H, W)
                    final_image = np.squeeze(final_image, axis=0)
                # Else, already (C, H, W), no change needed

                # Clip image to range
                final_image = np.clip(final_image, 0.0, 1.0)
                final_image = np.nan_to_num(final_image)

                # Scale back if normalized
                if original_dtype == np.uint16:
                    final_image = final_image * 65535.0
                    final_image = final_image.astype(np.uint16)

                # Update the loaded image.
                self.siril.set_image_pixeldata(final_image)
                self._update_progress("Process complete.")
                self.siril.log("Aberrations Remover processing complete.")
                self.siril.reset_progress()

        except Exception as e:
            self._update_progress(f"Error: {str(e)}")
            print(f"Error: {str(e)}")

    def _get_patch_indices(self, dim_size, patch_size, overlap):
        """Compute start indices for patches with the given overlap.
        This function ensures full coverage even if the image dimensions aren’t exact multiples of the stride.
        """
        stride = patch_size - overlap
        indices = []
        pos = 0
        while True:
            if pos + patch_size >= dim_size:
                indices.append(dim_size - patch_size)
                break
            else:
                indices.append(pos)
                pos += stride
        # Remove any duplicates and return in sorted order.
        return sorted(set(indices))
    
    def check_config_file(self):
        """
        Check for a saved model path and max version used in the configuration file.
        Returns (model_path, max_version_used) or (None, None) if not found.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        model_path = None
        max_version_used = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
                if len(lines) > 1:
                    max_version_used = lines[1].strip()
        return model_path, max_version_used

    def save_config_file(self, model_path, max_version_used=None):
        """
        Save the selected model path and max version used to the configuration file.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w') as file:
                file.write(model_path + "\n")
                if max_version_used is not None:
                    file.write(str(max_version_used) + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def _update_progress(self, message, progress=0):
        self.progress_var.set(message)
        self.siril.update_progress(message, progress)

    def close_dialog(self):
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()


def main():
    try:
        root = ThemedTk()
        app = DeconvolutionAIInterface(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
