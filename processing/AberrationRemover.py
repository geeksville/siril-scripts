# Aberration Remover AI by Riccardo Alberghi
# Script version: 1.0.5
# SPDX-License-Identifier: GPL-3.0-or-later

# 1.0.0 Original release
# 1.0.1 bugfix: fixed incorrect handling of 16 bits images
# 1.0.2 Updates due to API changes
# 1.0.3 Implemented ONNXHelper
# 1.0.4 Fixed a copy/paste bug I overlooked when reviewing (AKB)
# 1.0.5 Refactor & Disabled CoreML acceleration for stability


import sirilpy as s
from sirilpy import tksiril

s.ensure_installed("ttkthemes", "numpy", "requests", "subprocess", "platform")

onnx_helper = s.ONNXHelper()
onnx_helper.install_onnxruntime()

import os
import sys
import threading
import tkinter as tk
import requests
from tkinter import ttk, messagebox
import webbrowser
import numpy as np
from ttkthemes import ThemedTk
import platform

import onnxruntime
if hasattr(onnxruntime, 'preload_dlls'):
    with s.SuppressedStderr(), s.SuppressedStdout():
        onnxruntime.preload_dlls()
onnxruntime.set_default_logger_severity(4)

if s.check_module_version(">=0.6.0") and sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog


VERSION = "1.0.5"
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

        system = platform.system().lower()
        self.use_hwd_acceleration = False if system == "darwin" else True

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
        """Main deconvolution calculation method with improved error handling and structure."""
        try:
            with self.siril.image_lock():
                # Initialize ONNX session
                session = self._initialize_onnx_session()
                if session is None:
                    return

                # Prepare image data
                pixel_data, original_format, original_dtype = self._prepare_image_data()
                if pixel_data is None:
                    return

                # Process the image
                self._update_progress("Saving undo state...")
                self.siril.undo_save_state("Aberrations Remover")

                final_image = self._process_image_patches(session, pixel_data)

                # Restore original format and finalize
                final_image = self._finalize_image(final_image, original_format, original_dtype)

                self.siril.set_image_pixeldata(final_image)
                self._update_progress("Process complete.")
                self.siril.log("Aberrations Remover processing complete.")
                self.siril.reset_progress()

        except Exception as e:
            self._update_progress(f"Error: {str(e)}")
            print(f"Error: {str(e)}")

    def _initialize_onnx_session(self):
        """Initialize ONNX runtime session with proper error handling."""
        self._update_progress("Loading ONNX model...")
        model_path = self.model_path_var.get()

        # Initialize ONNX runtime session with ONNXHelper
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(self.use_hwd_acceleration)

            try:
                session = onnxruntime.InferenceSession(model_path, providers=providers)
                print(f"Used providers: {providers}")
                return session
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")

                # Retry with CPU only
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(model_path, providers=providers)
                    return session
                except onnxruntime.ONNXRuntimeError as err:
                    messagebox.showerror("Error", "Cannot build an inference model on this device")
                    return None

    def _prepare_image_data(self):
        """Prepare image data for processing, handling format conversions and normalization."""
        self._update_progress("Fetching image data...")
        pixel_data = self.siril.get_image_pixeldata()
        original_dtype = pixel_data.dtype

        # Convert to channels-first format
        original_format = None
        if pixel_data.ndim == 2:
            # Mono image: (H, W) -> (1, H, W)
            pixel_data = pixel_data[np.newaxis, ...]
            original_format = "mono"
        elif pixel_data.ndim == 3:
            if pixel_data.shape[0] in [1, 3]:
                # Already channels-first: (C, H, W)
                original_format = "channels_first"
            elif pixel_data.shape[2] in [1, 3]:
                # Channels-last: (H, W, C) -> (C, H, W)
                pixel_data = np.transpose(pixel_data, (2, 0, 1))
                original_format = "channels_last"
            else:
                raise ValueError(f"Unsupported image shape: {pixel_data.shape}")
        else:
            raise ValueError(f"Unsupported number of dimensions: {pixel_data.ndim}")

        # Normalize 16-bit images to [0,1] range
        if original_dtype == np.uint16:
            pixel_data = pixel_data.astype(np.float32) / 65535.0

        return pixel_data, original_format, original_dtype

    def _process_image_patches(self, session, pixel_data):
        """Process image using patch-based approach with proper blending."""
        patch_size = 512
        overlap = 64

        # Create Hann window for smooth blending
        hann1d = np.hanning(patch_size)
        window2d = np.outer(hann1d, hann1d).astype(np.float32)
        window2d = window2d / window2d.max()

        _, H, W = pixel_data.shape

        if pixel_data.shape[0] == 1:
            # Process mono image
            return self._process_mono_image(session, pixel_data, patch_size, overlap, window2d, H, W)
        elif pixel_data.shape[0] == 3:
            # Process RGB image channel by channel
            return self._process_rgb_image(session, pixel_data, patch_size, overlap, window2d, H, W)
        else:
            raise ValueError("Only mono (1 channel) or RGB (3 channel) images are supported.")

    def _process_mono_image(self, session, pixel_data, patch_size, overlap, window2d, H, W):
        """Process a monochrome image using patch-based inference."""
        output_image = np.zeros_like(pixel_data, dtype=np.float32)
        weight_image = np.zeros_like(pixel_data, dtype=np.float32)

        h_starts = self._get_patch_indices(H, patch_size, overlap)
        w_starts = self._get_patch_indices(W, patch_size, overlap)
        total_patches = len(h_starts) * len(w_starts)

        self._update_progress("Process started.")
        patch_count = 0

        for i in h_starts:
            for j in w_starts:
                patch_count += 1
                patch = np.copy(pixel_data[:, i:i + patch_size, j:j + patch_size])

                # Run inference
                processed_patch = self._run_inference(session, patch)
                processed_patch = np.nan_to_num(processed_patch, nan=0.0, posinf=0.0, neginf=0.0)

                # Apply blending and accumulate
                weighted_patch = processed_patch * window2d
                output_image[:, i:i + patch_size, j:j + patch_size] += weighted_patch
                weight_image[:, i:i + patch_size, j:j + patch_size] += window2d

                self._update_progress(
                    f"Patches done: {patch_count}/{total_patches}",
                    patch_count / total_patches
                )

        # Normalize by weights
        weight_image[weight_image == 0] = 1.0
        return output_image / weight_image

    def _process_rgb_image(self, session, pixel_data, patch_size, overlap, window2d, H, W):
        """Process an RGB image channel by channel using patch-based inference."""
        output_image = np.zeros_like(pixel_data, dtype=np.float32)
        weight_image = np.zeros_like(pixel_data, dtype=np.float32)

        h_starts = self._get_patch_indices(H, patch_size, overlap)
        w_starts = self._get_patch_indices(W, patch_size, overlap)
        total_patches = len(h_starts) * len(w_starts)

        for c in range(3):
            patch_count = 0
            self._update_progress(f"Process started on channel {c+1}/3.")

            for i in h_starts:
                for j in w_starts:
                    patch_count += 1
                    # Extract single channel patch
                    patch = np.copy(pixel_data[c:c+1, i:i + patch_size, j:j + patch_size])

                    # Run inference
                    processed_patch = self._run_inference(session, patch)
                    processed_patch = np.nan_to_num(processed_patch, nan=0.0, posinf=0.0, neginf=0.0)

                    # Apply blending and accumulate
                    weighted_patch = processed_patch * window2d
                    output_image[c:c+1, i:i + patch_size, j:j + patch_size] += weighted_patch
                    weight_image[c:c+1, i:i + patch_size, j:j + patch_size] += window2d

                    self._update_progress(
                        f"Patches done (channel {c+1}/3): {patch_count}/{total_patches}",
                        patch_count / total_patches
                    )

        # Normalize by weights
        weight_image[weight_image == 0] = 1.0
        return output_image / weight_image

    def _run_inference(self, session, patch):
        """Run ONNX inference on a single patch."""
        input_patch = patch.astype(np.float32)
        input_batch = np.expand_dims(input_patch, axis=0)

        inputs = session.get_inputs()
        input_dict = {inputs[0].name: input_batch}
        outputs, session = onnx_helper.run(session, self.model_path_var.get(), None, input_dict, return_first_output=True)

        return np.squeeze(outputs, axis=0)

    def _finalize_image(self, final_image, original_format, original_dtype):
        """Restore image to original format and apply final processing."""
        # Restore original format
        if original_format == "channels_last":
            final_image = np.transpose(final_image, (1, 2, 0))
        elif original_format == "mono":
            final_image = np.squeeze(final_image, axis=0)

        # Clip and handle NaN values
        final_image = np.clip(final_image, 0.0, 1.0)
        final_image = np.nan_to_num(final_image)

        # Scale back to original dtype
        if original_dtype == np.uint16:
            final_image = final_image * 65535.0
            final_image = final_image.astype(np.uint16)

        return final_image


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
