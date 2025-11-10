"""
# (c) Adrian Knagg-Baugh 2024-2025
# SPDX-License-Identifier: GPL-3.0-or-later

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version 2.0.2
# 1.0.1: AKB - convert "requires" to use exception handling
# 1.0.2: Miscellaneous fixes
# 1.0.3: Use tiffile instead of savetif32 to save the input file
#        This avoids colour shifts if the image profile != the display profile
# 1.0.4: Fix bug in 1.0.3 when processing mono images
# 1.0.5: Fix bug in 1.0.4 when converting 32-to-16-bit
#        Set "clear input directory" to default to True
# 1.0.6: Don't print empty lines to the log
# 2.0.0: Migrate GUI from tkinter to PyQt6, add image caching and
#        denoise-strength blending.
# 2.0.1: Update script to work correctly with pyscript
# 2.0.2: Fix GPU use when run from inside a flatpak sandbox
"""

import os
import re
import sys
import math
import argparse
import subprocess
from pathlib import Path

import numpy as np

import sirilpy as s

# Ensure dependencies are installed
s.ensure_installed("PyQt6", "tiffile")
import tiffile

VERSION = "2.0.1"

# ------------------------------
# Shared utility functions
# ------------------------------
def run_cosmic_clarity_process(executable_path: str, mode: str, use_gpu: bool, progress_callback=None):
    """
    Run Cosmic Clarity process and report progress.

    Args:
        executable_path: Path to Cosmic Clarity executable
        mode: Denoise mode (luminance, full, separate)
        use_gpu: Whether to use GPU
        progress_callback: Optional callback function(float) for progress updates (0.0 to 1.0)

    Returns:
        int: Process return code
    """
    command = [
        executable_path,
        f"--denoise_mode={mode}",
        f"--denoise_strength=1.0",
    ]
    if not use_gpu:
        command.append("--disable_gpu")

    in_flatpak = os.environ.get("container") == "flatpak"
    if in_flatpak:
        # Run executable on host to access host environment & GPU drivers
        command = ["flatpak-spawn", "--host"] + command
        print("Detected Flatpak sandbox — using flatpak-spawn to run Cosmic Clarity.")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    percent_re = re.compile(r"(\d+\.?\d*)%")
    buffer = ""
    while True:
        chunk = process.stdout.read(80)
        if not chunk:
            break
        buffer += chunk

        # Split on either \r or \n (or both)
        lines = re.split(r'[\r\n]+', buffer)

        for line in lines[:-1]:  # complete lines only
            if not line.strip():
                continue
            m = percent_re.search(line)
            if m:
                try:
                    pct = float(m.group(1)) / 100.0
                    if progress_callback:
                        progress_callback(max(0.0, min(1.0, pct)))
                except Exception:
                    pass
            else:
                print(line)

        buffer = lines[-1]  # keep incomplete line

    ret = process.wait()
    if progress_callback:
        progress_callback(1.0)

    return ret

def check_config_file(siril):
    config_dir = siril.get_siril_configdir()
    config_file_path = os.path.join(config_dir, "sirilcc_denoise.conf")
    if os.path.isfile(config_file_path):
        try:
            with open(config_file_path, "r") as f:
                p = f.readline().strip()
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        except Exception:
            pass
    print("Executable not yet configured. It is recommended to use Seti Astro Cosmic Clarity v5.4 or higher.")
    return None

def save_config_if_changed(siril, new_path: str, current_path: str):
    if new_path and new_path != current_path:
        config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_denoise.conf")
        try:
            with open(config_file_path, "w") as f:
                f.write(new_path + "\n")
        except Exception as e:
            print(f"Failed to write config: {e}")
        return new_path
    return current_path

def ensure_channel_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unexpected image shape: {arr.shape}")

def to_tiff_compatible(arr_cf: np.ndarray):
    if arr_cf.ndim != 3 or arr_cf.shape[0] not in (1, 3):
        raise ValueError(f"Unexpected array shape for TIFF: {arr_cf.shape}")
    if arr_cf.shape[0] == 1:
        return arr_cf[0], 'minisblack'
    else:
        return np.transpose(arr_cf, (1, 2, 0)), 'rgb'

def read_tiff_as_channel_first_float(filename: str) -> np.ndarray:
    with tiffile.TiffFile(filename) as t:
        data = t.asarray()
    data = np.ascontiguousarray(data)
    if data.ndim == 2:
        data_cf = data[np.newaxis, :, :]
    elif data.ndim == 3 and data.shape[2] == 3:
        data_cf = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected TIFF shape: {data.shape}")
    return data_cf.astype(np.float32, copy=False)

# ------------------------------
# CLI mode processing
# ------------------------------
def run_cli_mode(args):
    """Run processing directly from CLI args (headless mode)."""
    # Initialize Siril connection
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        print("Error: Failed to connect to Siril", file=sys.stderr)
        sys.exit(1)

    if not siril.is_image_loaded():
        print("Error: No image loaded", file=sys.stderr)
        sys.exit(1)

    try:
        siril.cmd("requires", "1.3.6")
    except s.CommandError:
        sys.exit(1)

    try:
        mode = args.denoising_mode.lower()
        denoise_strength = args.denoise_strength
        executable_path = args.executable
        use_gpu = args.use_gpu
        clear_input = args.clear_input_dir

        # --- fallback: check config file if no executable provided ---
        if not executable_path:
            executable_path = check_config_file(siril) or ""
        if not executable_path or not os.path.isfile(executable_path):
            print("Error: please provide a valid Cosmic Clarity executable "
                  "(-executable) or configure one via GUI.", file=sys.stderr)
            sys.exit(1)

        # Save executable path if it differs from stored one
        config_executable = check_config_file(siril)
        save_config_if_changed(siril, executable_path, config_executable)

        # Load and prepare image
        pixels = siril.get_image_pixeldata()
        was_16bit = (pixels.dtype == np.uint16)
        if was_16bit:
            pixels = pixels.astype(np.float32) / 65535.0
        else:
            pixels = pixels.astype(np.float32, copy=False)
        pixels_cf = ensure_channel_first(pixels)

        directory = os.path.dirname(executable_path)
        os.makedirs(os.path.join(directory, "input"), exist_ok=True)
        os.makedirs(os.path.join(directory, "output"), exist_ok=True)

        basename = os.path.basename(siril.get_image_filename())
        inputfilename = os.path.join(directory, "input", basename + ".tif")
        outputfilename = os.path.join(directory, "output", f"{basename}_denoised.tif")

        if clear_input:
            for each_file in Path(os.path.join(directory, "input")).glob("*.*"):
                try:
                    each_file.unlink()
                except Exception:
                    pass

        tiff_arr, photometry = to_tiff_compatible(pixels_cf)
        tiffile.imwrite(inputfilename, tiff_arr, photometric=photometry, planarconfig='contig')

        siril.update_progress("Seti Astro Cosmic Clarity Denoise starting...", 0)

        # Run external process with progress reporting
        def progress_callback(pct):
            siril.update_progress("Seti Astro Cosmic Clarity Denoise progress...", pct)

        ret = run_cosmic_clarity_process(executable_path, mode, use_gpu, progress_callback)

        if ret != 0 or not os.path.isfile(outputfilename):
            siril.reset_progress()
            print("Error: Cosmic Clarity process failed.", file=sys.stderr)
            sys.exit(1)

        result_fullstrength = read_tiff_as_channel_first_float(outputfilename)
        ds = float(max(0.0, min(1.0, denoise_strength)))
        blended = ds * result_fullstrength + (1.0 - ds) * pixels_cf
        if was_16bit:
            blended = np.rint(blended * 65535.0).astype(np.uint16)

        with siril.image_lock():
            siril.undo_save_state(f"Cosmic Clarity denoise ({mode}, str={ds})")
            siril.set_image_pixeldata(blended)

        siril.reset_progress()
        print("Cosmic Clarity denoise completed successfully.")
    except Exception as e:
        print(f"Error in CLI apply: {e}", file=sys.stderr)
        sys.exit(1)

# ------------------------------
# GUI mode
# ------------------------------
def run_gui_mode():
    """Run GUI mode."""
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QWidget,
        QLabel,
        QVBoxLayout,
        QHBoxLayout,
        QRadioButton,
        QGroupBox,
        QCheckBox,
        QSlider,
        QLineEdit,
        QPushButton,
        QFileDialog,
        QMessageBox,
        QStyleFactory,
    )

    # ------------------------------
    # Worker to run Cosmic Clarity
    # ------------------------------
    class CosmicClarityWorker(QThread):
        progress = pyqtSignal(float)  # 0..1
        finished_ok = pyqtSignal(bool, str)  # success, outputfilename (if any)

        def __init__(self, executable_path: str, mode: str, use_gpu: bool, inputfilename: str, outputfilename: str):
            super().__init__()
            self.executable_path = executable_path
            self.mode = mode
            self.use_gpu = use_gpu
            self.inputfilename = inputfilename
            self.outputfilename = outputfilename

        def run(self):
            try:
                # Use shared process runner with progress callback
                def progress_callback(pct):
                    self.progress.emit(pct)

                ret = run_cosmic_clarity_process(
                    self.executable_path,
                    self.mode,
                    self.use_gpu,
                    progress_callback
                )

                if ret != 0:
                    self.finished_ok.emit(False, "")
                else:
                    self.finished_ok.emit(True, self.outputfilename)
            except Exception:
                self.finished_ok.emit(False, "")

    # ------------------------------
    # Main UI
    # ------------------------------
    class CosmicClarityInterface(QWidget):
        def __init__(self):
            super().__init__()

            self.setWindowTitle(f"Cosmic Clarity Denoise - v{VERSION}")

            # Initialize Siril connection
            self.siril = s.SirilInterface()
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                self._siril_error("Failed to connect to Siril")
                return

            if not self.siril.is_image_loaded():
                self._siril_error("No image loaded")
                return

            try:
                self.siril.cmd("requires", "1.3.6")
            except s.CommandError:
                self.close()
                return

            # Load config
            self.config_executable = check_config_file(self.siril)

            # Initialize cache
            self.cached_original = None
            self.cached_was_16bit = False
            self.cached_mode_key = None
            self.result_fullstrength = None

            self._build_ui()
            if self.config_executable:
                self.exec_path_edit.setText(self.config_executable)

        # ------------------------------
        # UI
        # ------------------------------
        def _build_ui(self):
            root = QVBoxLayout(self)

            title = QLabel("Cosmic Clarity Denoise Settings")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            title.setProperty("class", "header")
            root.addWidget(title)

            # Mode group
            mode_group = QGroupBox("Denoise Mode")
            mode_layout = QVBoxLayout(mode_group)
            self.mode_lum = QRadioButton("Luminance")
            self.mode_full = QRadioButton("Full")
            self.mode_sep = QRadioButton("Separate")
            self.mode_lum.setChecked(True)
            for rb in (self.mode_lum, self.mode_full, self.mode_sep):
                mode_layout.addWidget(rb)
            root.addWidget(mode_group)

            # Options
            options_group = QGroupBox("Options")
            options_layout = QVBoxLayout(options_group)

            self.chk_gpu = QCheckBox("Use GPU")
            self.chk_gpu.setChecked(True)
            options_layout.addWidget(self.chk_gpu)

            self.chk_clear = QCheckBox("Clear input directory")
            self.chk_clear.setChecked(True)
            self.chk_clear.setToolTip(
                "Delete any files from the Cosmic Clarity input directory.\n"
                "If not done, Cosmic Clarity will process all image files in the input\n"
                "directory, which will take longer and generate potentially unnecessary files.\n"
                "WARNING: set this to False if you wish to retain previous content of the\n"
                "Cosmic Clarity input directory."
            )
            options_layout.addWidget(self.chk_clear)

            # Denoise strength slider + value
            strength_row = QHBoxLayout()
            strength_label = QLabel("Denoise Strength:")
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setSingleStep(1)
            self.slider.setValue(50)
            self.slider.valueChanged.connect(self._update_strength_label)
            self.lbl_strength = QLabel("0.50")
            self.lbl_strength.setFixedWidth(40)

            strength_row.addWidget(strength_label)
            strength_row.addWidget(self.slider, stretch=1)
            strength_row.addWidget(self.lbl_strength)
            options_layout.addLayout(strength_row)

            root.addWidget(options_group)

            # Executable selection
            exec_group = QGroupBox("Cosmic Clarity Executable")
            exec_layout = QHBoxLayout(exec_group)
            self.exec_path_edit = QLineEdit()
            btn_browse = QPushButton("Browse")
            btn_browse.clicked.connect(self._browse_executable)
            exec_layout.addWidget(self.exec_path_edit, stretch=1)
            exec_layout.addWidget(btn_browse)
            root.addWidget(exec_group)

            # Buttons
            btn_row = QHBoxLayout()
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(self.close)
            self.btn_apply = QPushButton("Apply")
            self.btn_apply.clicked.connect(self.apply_clicked)
            btn_row.addWidget(btn_close)
            btn_row.addWidget(self.btn_apply)
            root.addLayout(btn_row)

            self.setLayout(root)

        def _update_strength_label(self):
            v = self.slider.value() / 100.0
            v2 = math.floor(v * 100) / 100.0
            self.lbl_strength.setText(f"{v2:.2f}")

        def _browse_executable(self):
            fn, _ = QFileDialog.getOpenFileName(self, "Select Cosmic Clarity Executable", os.path.expanduser("~"))
            if fn:
                self.exec_path_edit.setText(fn)

        def _siril_error(self, msg: str):
            try:
                self.siril.error_messagebox(msg)
            except Exception:
                QMessageBox.critical(self, "Error", msg)
            self.close()

        # ------------------------------
        # Apply (main logic with caching)
        # ------------------------------
        def apply_clicked(self):
            try:
                mode = self._get_current_mode()
                denoise_strength = math.floor((self.slider.value() / 100.0) * 100) / 100.0
                executable_path = self.exec_path_edit.text().strip()
                use_gpu = self.chk_gpu.isChecked()
                clear_input = self.chk_clear.isChecked()

                if not executable_path or not os.path.isfile(executable_path):
                    QMessageBox.warning(self, "Executable", "Please select a valid Cosmic Clarity executable.")
                    return

                # Persist executable path if changed
                self.config_executable = save_config_if_changed(self.siril, executable_path, self.config_executable)

                # Invalidate cache if mode changed
                if self.cached_mode_key is not None and self.cached_mode_key != mode:
                    self.cached_original = None
                    self.result_fullstrength = None
                    self.cached_was_16bit = False

                if self.cached_original is None:
                    pixels = self.siril.get_image_pixeldata()
                    was_16bit = (pixels.dtype == np.uint16)
                    if was_16bit:
                        pixels = pixels.astype(np.float32) / 65535.0
                    else:
                        pixels = pixels.astype(np.float32, copy=False)
                    pixels_cf = ensure_channel_first(pixels)

                    self.cached_original = pixels_cf
                    self.cached_was_16bit = was_16bit
                    self.cached_mode_key = mode

                    directory = os.path.dirname(executable_path)
                    original_dir = os.getcwd()
                    try:
                        os.chdir(directory)
                        inputpath = os.path.join(directory, "input")
                        outputpath = os.path.join(directory, "output")
                        os.makedirs(inputpath, exist_ok=True)
                        os.makedirs(outputpath, exist_ok=True)

                        basename = os.path.basename(self.siril.get_image_filename())
                        inputfilename = os.path.join(inputpath, basename) + ".tif"
                        outputfilename = os.path.join(outputpath, f"{basename}_denoised.tif")

                        if clear_input:
                            for each_file in Path(inputpath).glob("*.*"):
                                try:
                                    each_file.unlink()
                                    print(f"Deleted: {each_file}")
                                except Exception as e:
                                    print(f"Failed to delete {each_file}: {e}")

                        tiff_arr, photometry = to_tiff_compatible(self.cached_original)
                        tiffile.imwrite(inputfilename, tiff_arr, photometric=photometry, planarconfig='contig')

                        self.siril.update_progress("Seti Astro Cosmic Clarity Denoise starting...", 0)

                        # Disable Apply until worker finishes
                        self.btn_apply.setEnabled(False)

                        self._worker = CosmicClarityWorker(executable_path, mode, use_gpu, inputfilename, outputfilename)
                        self._worker.progress.connect(self._on_progress)
                        self._worker.finished_ok.connect(self._on_full_run_finished)

                        self._pending_blend_strength = denoise_strength
                        self._pending_outputfilename = outputfilename
                        self._worker.start()
                        return
                    finally:
                        os.chdir(original_dir)

                if self.result_fullstrength is not None:
                    self._apply_blend_and_set_image(denoise_strength)
                else:
                    QMessageBox.information(self, "Please wait", "Initial denoise is still in progress.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

        def _on_progress(self, frac: float):
            try:
                self.siril.update_progress("Seti Astro Cosmic Clarity Denoise progress...", float(frac))
            except Exception:
                pass

        def _on_full_run_finished(self, success: bool, outputfilename: str):
            try:
                if not success:
                    self.siril.reset_progress()
                    QMessageBox.critical(self, "Cosmic Clarity", "Processing failed.")
                    self.cached_original = None
                    self.result_fullstrength = None
                    return

                full_cf = read_tiff_as_channel_first_float(self._pending_outputfilename)
                self.result_fullstrength = full_cf
                self._apply_blend_and_set_image(self._pending_blend_strength)
            finally:
                try:
                    self.siril.reset_progress()
                except Exception:
                    pass
                # Re-enable Apply button
                self.btn_apply.setEnabled(True)

        # --------
        # Helpers
        # --------
        def _get_current_mode(self) -> str:
            if self.mode_full.isChecked():
                return "full"
            if self.mode_sep.isChecked():
                return "separate"
            return "luminance"

        def _apply_blend_and_set_image(self, denoise_strength: float):
            try:
                if self.cached_original is None or self.result_fullstrength is None:
                    return
                ds = float(max(0.0, min(1.0, denoise_strength)))
                blended = ds * self.result_fullstrength + (1.0 - ds) * self.cached_original
                blended = np.clip(blended, 0.0, 1.0) if self.cached_was_16bit else blended

                mode = self._get_current_mode()
                self.siril.undo_save_state(f"Cosmic Clarity denoise ({mode}, str={ds})")

                out_cf = blended
                if self.cached_was_16bit:
                    out_cf = np.rint(out_cf * 65535.0).astype(np.uint16)

                out_cf = np.ascontiguousarray(out_cf)
                with self.siril.image_lock():
                    self.siril.set_image_pixeldata(out_cf)

                self.siril.log("Cosmic Clarity denoise complete.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    # Create and run the GUI
    app = QApplication(sys.argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    w = CosmicClarityInterface()
    w.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity Denoise Script")
    parser.add_argument("-denoising_mode", type=str, choices=["luminance", "full", "separate"],
                        help="Denoising mode")
    parser.add_argument("-denoise_strength", type=float,
                        help="Denoise strength (0.0 to 1.0)")
    parser.add_argument("-use_gpu", action="store_true",
                        help="Use GPU")
    parser.add_argument("-no_gpu", action="store_true",
                        help="Disable GPU")
    parser.add_argument("-clear_input_dir", action="store_true",
                        help="Clear input directory")
    parser.add_argument("-no_clear_input", action="store_true",
                        help="Do not clear input directory")
    parser.add_argument("-executable", type=str,
                        help="Path to Cosmic Clarity executable")

    args, unknown = parser.parse_known_args()

    # Detect CLI mode: if ANY argument was provided (check sys.argv)
    # sys.argv[0] is the script name, so check if there are more arguments
    cli_mode = len(sys.argv) > 1 and any(
        arg.startswith('-') for arg in sys.argv[1:]
    )

    try:
        if cli_mode:
            # Process arguments for CLI mode
            # Set defaults for CLI mode
            if args.denoising_mode is None:
                args.denoising_mode = "luminance"
            if args.denoise_strength is None:
                args.denoise_strength = 0.5
            if args.executable is None:
                args.executable = ""

            # Handle GPU flags
            if args.no_gpu:
                args.use_gpu = False
            elif not args.use_gpu:
                args.use_gpu = True  # Default to True if neither flag specified

            # Handle clear input flags
            if args.no_clear_input:
                args.clear_input_dir = False
            elif not args.clear_input_dir:
                args.clear_input_dir = True  # Default to True if neither flag specified

            # Headless CLI run - no GUI imports needed
            run_cli_mode(args)
        else:
            # GUI mode - import GUI components only here
            run_gui_mode()
    except Exception as e:
        print(f"Error initializing application: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
