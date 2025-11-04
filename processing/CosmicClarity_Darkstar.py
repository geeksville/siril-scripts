# Cosmic Clarity - Darkstar wrapper
# SPDX-License-Identifier: GPL-3.0-or-later
# (c) Adrian Knagg-Baugh 2025

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version 1.1.1
# 1.1.0 First stable release
# 1.1.1 Fix issue when running the script using pyscript

import sirilpy as s
s.ensure_installed("PyQt6", "tiffile")

import os
import sys
import re
import argparse
import subprocess
from pathlib import Path

import numpy as np
import tifffile as tiffile

VERSION = "1.1.1"

# ------------------------------
# Shared utility functions
# ------------------------------
def run_cosmic_clarity_darkstar_process(executable_path: str, exe_dir: str,
                                       disable_gpu: bool, star_removal_mode: str,
                                       show_extracted_stars: bool, chunk_size: int,
                                       overlap: int, progress_callback=None):
    """
    Run Cosmic Clarity Darkstar process and report progress.

    Args:
        executable_path: Path to Cosmic Clarity Darkstar executable
        exe_dir: Directory containing the executable (sets working directory)
        disable_gpu: Whether to disable GPU
        star_removal_mode: Mode for star removal (additive or unscreen)
        show_extracted_stars: Whether to produce stars-only image
        chunk_size: Size of processing chunks
        overlap: Overlap between chunks in pixels
        progress_callback: Optional callback function(float) for progress updates (0.0 to 1.0)

    Returns:
        int: Process return code
    """
    cmd = [executable_path]
    if disable_gpu:
        cmd.append("--disable_gpu")
    if star_removal_mode:
        cmd.extend(["--star_removal_mode", str(star_removal_mode)])
    if show_extracted_stars:
        cmd.append("--show_extracted_stars")
    if chunk_size:
        cmd.extend(["--chunk_size", str(int(chunk_size))])
    if overlap is not None:
        cmd.extend(["--overlap", str(int(overlap))])

    process = subprocess.Popen(
        cmd,
        cwd=exe_dir,
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

        lines = re.split(r'[\r\n]+', buffer)

        for line in lines[:-1]:
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

        buffer = lines[-1]

    ret = process.wait()
    if progress_callback:
        progress_callback(1.0)

    return ret

def check_config_file(siril):
    """Check for and load the configured executable path."""
    config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_darkstar.conf")
    if os.path.isfile(config_file_path):
        try:
            with open(config_file_path, "r") as f:
                p = f.readline().strip()
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        except Exception:
            pass
    print("Executable not yet configured.")
    return None

def save_config_if_changed(siril, new_path: str, current_path: str):
    """Save the executable path to config if it has changed."""
    if new_path and new_path != current_path:
        config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_darkstar.conf")
        try:
            with open(config_file_path, "w") as f:
                f.write(new_path + "\n")
        except Exception as e:
            print(f"Failed to write config: {e}")
        return new_path
    return current_path

def prepare_image_for_tiff(pixels):
    """Convert image data to TIFF-compatible format."""
    if pixels.ndim == 2:
        return pixels, "minisblack"
    elif pixels.ndim == 3 and pixels.shape[0] in (1, 3):
        photometry = "minisblack" if pixels.shape[0] == 1 else "rgb"
        output = pixels[0] if pixels.shape[0] == 1 else np.transpose(pixels, (1, 2, 0))
        return output, photometry
    else:
        raise ValueError(f"Unexpected image shape: {pixels.shape}")

def load_result_tiff_as_float(filename: str):
    """Load a TIFF file and convert to channel-first float32 format (0..1)."""
    with tiffile.TiffFile(filename) as t:
        data = t.asarray()
    data = np.ascontiguousarray(data)

    if data.ndim == 2:
        cf = data[np.newaxis, :, :].astype(np.float32, copy=False)
    elif data.ndim == 3 and data.shape[2] == 3:
        cf = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unexpected TIFF shape: {data.shape}")

    # If dtype is integer, normalize to 0..1
    if np.issubdtype(data.dtype, np.integer):
        return cf.astype(np.float32) / 65535.0
    else:
        return cf.astype(np.float32, copy=False)

def ensure_channel_first(pixels_float):
    """Ensure image data is in channel-first format."""
    if pixels_float.ndim == 2:
        return pixels_float[np.newaxis, :, :].astype(np.float32, copy=False)
    elif pixels_float.ndim == 3 and pixels_float.shape[0] in (1, 3):
        return pixels_float
    else:
        return pixels_float.astype(np.float32, copy=False)

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
        executable_path = args.executable

        # Fallback to config file if no executable provided
        if not executable_path:
            executable_path = check_config_file(siril) or ""
        if not executable_path or not os.path.isfile(executable_path):
            print("Error: please provide a valid Cosmic Clarity Darkstar executable "
                  "(-executable) or configure one via GUI.", file=sys.stderr)
            sys.exit(1)

        # Save executable path if it differs from stored one
        config_executable = check_config_file(siril)
        save_config_if_changed(siril, executable_path, config_executable)

        # Compute overlap default if not supplied
        chunk_size = args.chunk_size
        overlap = args.overlap if args.overlap is not None else int(0.125 * chunk_size)

        # Setup directories
        exe_dir = os.path.dirname(executable_path)
        input_dir = os.path.join(exe_dir, "input")
        output_dir = os.path.join(exe_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Clear input if requested
        if args.clear_input and os.path.isdir(input_dir):
            for f in Path(input_dir).glob("*.*"):
                try:
                    f.unlink()
                except Exception:
                    pass

        # Get current Siril image
        pixels = siril.get_image_pixeldata()
        was_16bit = (pixels.dtype == np.uint16)
        if was_16bit:
            pixels_float = pixels.astype(np.float32) / 65535.0
        else:
            pixels_float = pixels.astype(np.float32, copy=False)

        # Prepare for writing
        write_arr, photometry = prepare_image_for_tiff(pixels_float)

        # Generate filenames
        orig_filename = os.path.basename(siril.get_image_filename())
        base_noext = os.path.splitext(orig_filename)[0]
        input_file = os.path.join(input_dir, f"{base_noext}.tif")
        starless_file = os.path.join(output_dir, f"{base_noext}_starless.tif")
        stars_only_file = os.path.join(output_dir, f"{base_noext}_stars_only.tif")

        # Write input TIFF
        tiffile.imwrite(input_file, write_arr, photometric=photometry, planarconfig="contig")

        siril.update_progress("Cosmic Clarity Darkstar starting...", 0)

        # Run external process with progress reporting
        def progress_callback(pct):
            siril.update_progress("Cosmic Clarity Darkstar progress...", pct)

        ret = run_cosmic_clarity_darkstar_process(
            executable_path, exe_dir, args.disable_gpu, args.star_removal_mode,
            args.show_extracted_stars, chunk_size, overlap, progress_callback
        )

        if ret != 0:
            siril.reset_progress()
            print("Error: Cosmic Clarity Darkstar process failed.", file=sys.stderr)
            sys.exit(1)

        # Read outputs
        got_starless = os.path.isfile(starless_file)
        got_stars_only = os.path.isfile(stars_only_file)

        starless_arr = None
        stars_arr = None

        if got_starless:
            starless_arr = load_result_tiff_as_float(starless_file)
        if got_stars_only:
            stars_arr = load_result_tiff_as_float(stars_only_file)

        # Choose which to set
        if args.show_extracted_stars and stars_arr is not None:
            arr = stars_arr
            label = "Darkstar: stars-only result"
        else:
            original_cf = ensure_channel_first(pixels_float)
            arr = starless_arr if starless_arr is not None else original_cf
            label = "Darkstar: starless result" if starless_arr is not None else "Darkstar: original"

        if arr is None:
            siril.reset_progress()
            print("Error: No output image found to set.", file=sys.stderr)
            sys.exit(1)

        # Convert to appropriate dtype
        force_16 = siril.get_siril_config("core", "force_16bit")
        out = arr
        if was_16bit or force_16:
            out = np.rint(out * 65535.0).astype(np.uint16)
        out = np.ascontiguousarray(out)

        with siril.image_lock():
            siril.undo_save_state(label)
            siril.set_image_pixeldata(out)

        siril.reset_progress()
        print("Cosmic Clarity Darkstar processing completed successfully.")
    except Exception as e:
        print(f"Error in CLI apply: {e}", file=sys.stderr)
        sys.exit(1)

# ------------------------------
# GUI mode
# ------------------------------
def run_gui_mode():
    """Run GUI mode."""
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QCheckBox, QLineEdit, QPushButton, QFileDialog, QMessageBox, QGroupBox,
        QComboBox, QSlider, QRadioButton, QButtonGroup
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal

    class CosmicClarityDarkstarWorker(QThread):
        # finished_ok: success (bool), starless_path (str), stars_only_path (str),
        # original_array (object), starless_array (object or None), stars_array (object or None), was_16bit (bool)
        progress = pyqtSignal(float)
        finished_ok = pyqtSignal(bool, str, str, object, object, object, bool)

        def __init__(self, executable_path, exe_dir, disable_gpu, star_removal_mode,
                     show_extracted_stars, chunk_size, overlap, clear_input,
                     input_file, starless_file, stars_only_file,
                     pixels_float, was_16bit):
            super().__init__()
            self.executable_path = executable_path
            self.exe_dir = exe_dir
            self.disable_gpu = disable_gpu
            self.star_removal_mode = star_removal_mode
            self.show_extracted_stars = show_extracted_stars
            self.chunk_size = chunk_size
            self.overlap = overlap
            self.clear_input = clear_input
            self.input_file = input_file
            self.starless_file = starless_file
            self.stars_only_file = stars_only_file
            self.pixels_float = pixels_float
            self.was_16bit = was_16bit

        def run(self):
            try:
                # Clear input if requested
                if self.clear_input:
                    input_dir = os.path.join(self.exe_dir, "input")
                    if os.path.isdir(input_dir):
                        for f in Path(input_dir).glob("*.*"):
                            try:
                                f.unlink()
                            except Exception:
                                pass

                # Write input TIFF
                write_arr, photometry = prepare_image_for_tiff(self.pixels_float)
                tiffile.imwrite(self.input_file, write_arr, photometric=photometry, planarconfig="contig")

                # Use shared process runner with progress callback
                def progress_callback(pct):
                    self.progress.emit(pct)

                ret = run_cosmic_clarity_darkstar_process(
                    self.executable_path, self.exe_dir, self.disable_gpu,
                    self.star_removal_mode, self.show_extracted_stars,
                    self.chunk_size, self.overlap, progress_callback
                )

                if ret != 0:
                    self.finished_ok.emit(False, "", "", None, None, None, False)
                    return

                # Read outputs
                got_starless = os.path.isfile(self.starless_file)
                got_stars_only = os.path.isfile(self.stars_only_file)

                starless_arr = None
                stars_arr = None

                if got_starless:
                    starless_arr = load_result_tiff_as_float(self.starless_file)
                if got_stars_only:
                    stars_arr = load_result_tiff_as_float(self.stars_only_file)

                # Original array in channel-first format
                original_cf = ensure_channel_first(self.pixels_float)

                self.finished_ok.emit(
                    True,
                    self.starless_file if got_starless else "",
                    self.stars_only_file if got_stars_only else "",
                    original_cf,
                    starless_arr,
                    stars_arr,
                    self.was_16bit
                )
            except Exception as e:
                print(f"Error in worker: {e}", file=sys.stderr)
                self.finished_ok.emit(False, "", "", None, None, None, False)

    class SirilCosmicClarityDarkstar(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle(f"Cosmic Clarity - Darkstar - v{VERSION}")
            self.setFixedSize(600, 480)

            # Connect to Siril
            self.siril = s.SirilInterface()
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                self.siril.error_messagebox("Failed to connect to Siril")
                self.close()
                return

            if not self.siril.is_image_loaded():
                self.siril.error_messagebox("No image loaded")
                self.close()
                return

            try:
                self.siril.cmd("requires", "1.3.6")
            except s.CommandError:
                self.close()
                return

            self.config_executable = check_config_file(self.siril)

            # GUI defaults
            self.executable_path = self.config_executable or ""
            self.disable_gpu = False
            self.star_removal_mode = "additive"
            self.show_extracted_stars = False
            self.chunk_size = 512
            self.overlap_pixels = int(0.125 * self.chunk_size)
            self.clear_input = True

            # caches (store channel-first float32 arrays)
            self.cached = {
                "original": None,
                "starless": None,
                "stars": None,
                "was_16bit": False
            }

            self._create_widgets()

        def _create_widgets(self):
            central = QWidget()
            layout = QVBoxLayout(central)

            title = QLabel("Cosmic Clarity - Darkstar (Star removal)")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(title)

            # Executable selection
            exec_group = QGroupBox("Cosmic Clarity Executable")
            exec_layout = QHBoxLayout()
            self.exec_entry = QLineEdit(self.executable_path)
            exec_layout.addWidget(self.exec_entry)
            browse = QPushButton("Browse")
            browse.clicked.connect(self._browse_executable)
            exec_layout.addWidget(browse)
            exec_group.setLayout(exec_layout)
            layout.addWidget(exec_group)

            # Options group
            options_group = QGroupBox("Options")
            options_layout = QVBoxLayout()

            # GPU checkbox
            self.gpu_checkbox = QCheckBox("Use GPU (unchecked => --disable_gpu)")
            self.gpu_checkbox.setChecked(not self.disable_gpu)
            options_layout.addWidget(self.gpu_checkbox)

            # Mode selector
            mode_layout = QHBoxLayout()
            mode_layout.addWidget(QLabel("Star removal mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["additive", "unscreen"])
            self.mode_combo.setCurrentText(self.star_removal_mode)
            mode_layout.addWidget(self.mode_combo)
            options_layout.addLayout(mode_layout)

            # Show extracted stars
            self.show_stars_checkbox = QCheckBox("Produce extracted-stars image (stars-only)")
            self.show_stars_checkbox.setChecked(self.show_extracted_stars)
            self.show_stars_checkbox.toggled.connect(self._toggle_stars_switch_visibility)
            options_layout.addWidget(self.show_stars_checkbox)

            # Result choice group
            self.result_choice_group = QGroupBox("Image to set in Siril (toggle after processing)")
            rc_layout = QHBoxLayout()
            self.rb_original = QRadioButton("Original")
            self.rb_starless = QRadioButton("Starless")
            self.rb_stars = QRadioButton("Stars-only")
            self.rb_original.setChecked(True)
            self.rb_choice_buttons = QButtonGroup(self)
            self.rb_choice_buttons.addButton(self.rb_original)
            self.rb_choice_buttons.addButton(self.rb_starless)
            self.rb_choice_buttons.addButton(self.rb_stars)
            rc_layout.addWidget(self.rb_original)
            rc_layout.addWidget(self.rb_starless)
            rc_layout.addWidget(self.rb_stars)
            self.result_choice_group.setLayout(rc_layout)
            self.result_choice_group.setEnabled(False)
            options_layout.addWidget(self.result_choice_group)

            # chunk size slider
            chunk_layout = QHBoxLayout()
            chunk_layout.addWidget(QLabel("Chunk size:"))
            self.chunk_slider = QSlider(Qt.Orientation.Horizontal)
            self.chunk_slider.setRange(128, 2048)
            self.chunk_slider.setSingleStep(64)
            self.chunk_slider.setValue(self.chunk_size)
            self.chunk_slider.valueChanged.connect(self._on_chunk_changed)
            self.chunk_label = QLabel(str(self.chunk_size))
            chunk_layout.addWidget(self.chunk_slider)
            chunk_layout.addWidget(self.chunk_label)
            options_layout.addLayout(chunk_layout)

            # overlap slider
            overlap_layout = QHBoxLayout()
            overlap_layout.addWidget(QLabel("Overlap (% of chunk):"))
            self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
            self.overlap_slider.setRange(0, 50)
            default_percent = int(12.5)
            self.overlap_slider.setValue(default_percent)
            self.overlap_slider.valueChanged.connect(self._on_overlap_percent_changed)
            self.overlap_label = QLabel(f"{default_percent}%")
            overlap_layout.addWidget(self.overlap_slider)
            overlap_layout.addWidget(self.overlap_label)
            options_layout.addLayout(overlap_layout)

            # Clear input checkbox
            self.clear_checkbox = QCheckBox("Clear input directory before run")
            self.clear_checkbox.setChecked(self.clear_input)
            options_layout.addWidget(self.clear_checkbox)

            options_group.setLayout(options_layout)
            layout.addWidget(options_group)

            # Buttons
            btn_layout = QHBoxLayout()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.close)
            btn_layout.addWidget(close_btn)

            self.apply_btn = QPushButton("Apply")
            self.apply_btn.clicked.connect(self._on_apply)
            btn_layout.addWidget(self.apply_btn)

            # Hook toggles to replace image from cache
            self.rb_original.toggled.connect(lambda checked: checked and self._set_image_from_cache("original"))
            self.rb_starless.toggled.connect(lambda checked: checked and self._set_image_from_cache("starless"))
            self.rb_stars.toggled.connect(lambda checked: checked and self._set_image_from_cache("stars"))

            layout.addLayout(btn_layout)
            self.setCentralWidget(central)

        def _on_chunk_changed(self, val):
            self.chunk_size = int(val)
            self.chunk_label.setText(str(self.chunk_size))
            percent = self.overlap_slider.value()
            self.overlap_pixels = int((percent / 100.0) * self.chunk_size)
            self.overlap_label.setText(f"{percent}%")

        def _on_overlap_percent_changed(self, val):
            self.overlap_label.setText(f"{val}%")
            self.overlap_pixels = int((val / 100.0) * self.chunk_size)

        def _toggle_stars_switch_visibility(self, checked):
            self.result_choice_group.setVisible(True)

        def _browse_executable(self):
            fn, _ = QFileDialog.getOpenFileName(self, "Select Cosmic Clarity darkstar executable", os.path.expanduser("~"))
            if fn:
                self.exec_entry.setText(fn)
                self.executable_path = fn

        def _set_image_from_cache(self, key: str):
            """Convert cached float array to appropriate dtype and set in Siril with undo."""
            try:
                arr = self.cached.get(key)
                if arr is None:
                    return
                was_16bit = self.cached.get("was_16bit", False)
                force_16 = self.siril.get_siril_config("core", "force_16bit")
                out = arr
                if was_16bit or force_16:
                    out = np.rint(out * 65535.0).astype(np.uint16)
                out = np.ascontiguousarray(out)
                with self.siril.image_lock():
                    label = {
                        "original": "Original",
                        "starless": "Starless result",
                        "stars": "Stars-only result"
                    }.get(key, "Darkstar result")
                    self.siril.undo_save_state(f"Darkstar: set {label}")
                    self.siril.set_image_pixeldata(out)
            except Exception as e:
                print(f"Error setting image from cache: {e}", file=sys.stderr)

        def _on_apply(self):
            try:
                executable_path = self.exec_entry.text().strip()

                if not executable_path or not os.path.isfile(executable_path):
                    QMessageBox.critical(self, "Executable", "Please select a valid Cosmic Clarity Darkstar executable.")
                    return

                # Persist executable path if changed
                self.config_executable = save_config_if_changed(self.siril, executable_path, self.config_executable)

                # Setup directories and filenames
                exe_dir = os.path.dirname(executable_path)
                input_dir = os.path.join(exe_dir, "input")
                output_dir = os.path.join(exe_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                orig_filename = os.path.basename(self.siril.get_image_filename())
                base_noext = os.path.splitext(orig_filename)[0]
                input_file = os.path.join(input_dir, f"{base_noext}.tif")
                starless_file = os.path.join(output_dir, f"{base_noext}_starless.tif")
                stars_only_file = os.path.join(output_dir, f"{base_noext}_stars_only.tif")

                # Get image data
                pixels = self.siril.get_image_pixeldata()
                was_16bit = (pixels.dtype == np.uint16)
                if was_16bit:
                    pixels_float = pixels.astype(np.float32) / 65535.0
                else:
                    pixels_float = pixels.astype(np.float32, copy=False)

                self.siril.update_progress("Cosmic Clarity Darkstar starting...", 0)

                # Disable apply while running
                self.apply_btn.setEnabled(False)

                # Start worker
                self._worker = CosmicClarityDarkstarWorker(
                    executable_path, exe_dir,
                    not self.gpu_checkbox.isChecked(),
                    self.mode_combo.currentText(),
                    self.show_stars_checkbox.isChecked(),
                    self.chunk_slider.value(),
                    int((self.overlap_slider.value() / 100.0) * self.chunk_slider.value()),
                    self.clear_checkbox.isChecked(),
                    input_file, starless_file, stars_only_file,
                    pixels_float, was_16bit
                )
                self._worker.progress.connect(self._on_progress)
                self._worker.finished_ok.connect(self._on_finished_worker)
                self._worker.start()

            except Exception as e:
                print(f"Error in apply: {e}")
                try:
                    self.apply_btn.setEnabled(True)
                except Exception:
                    pass
                self.siril.update_progress(f"Error: {str(e)}", 0)

        def _on_progress(self, frac):
            try:
                self.siril.update_progress("Cosmic Clarity Darkstar progress...", float(frac))
            except Exception:
                pass

        def _on_finished_worker(self, success, starless_path, stars_only_path, original_arr, starless_arr, stars_arr, was_16bit):
            try:
                self.apply_btn.setEnabled(True)
            except Exception:
                pass

            try:
                self.siril.reset_progress()
            except Exception:
                pass

            if not success:
                QMessageBox.critical(self, "Cosmic Clarity", "Darkstar failed.")
                return

            # Cache arrays
            self.cached["original"] = original_arr
            self.cached["starless"] = starless_arr
            self.cached["stars"] = stars_arr
            self.cached["was_16bit"] = was_16bit

            # Enable radio group
            self.result_choice_group.setEnabled(True)

            # Default selection
            if starless_arr is not None:
                self.rb_starless.setChecked(True)
                self._set_image_from_cache("starless")
            else:
                self.rb_original.setChecked(True)
                self._set_image_from_cache("original")

            self.siril.log("Cosmic Clarity - Darkstar completed.")

    # Create and run the GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SirilCosmicClarityDarkstar()
    win.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity - Darkstar wrapper")
    parser.add_argument("-executable", type=str,
                        help="Path to setiastrocosmicclarity_darkstar executable")
    parser.add_argument("--disable_gpu", action="store_true",
                        help="Disable GPU (use CPU only)")
    parser.add_argument("--star_removal_mode", choices=["additive", "unscreen"],
                        help="Star removal mode")
    parser.add_argument("--show_extracted_stars", action="store_true",
                        help="Produce stars-only image")
    parser.add_argument("--chunk_size", type=int,
                        help="Processing chunk size")
    parser.add_argument("--overlap", type=int,
                        help="Overlap in pixels (default=0.125*chunk_size)")
    parser.add_argument("--clear_input", action="store_true",
                        help="Clear input directory before run")

    args, unknown = parser.parse_known_args()

    # Detect CLI mode: if ANY argument was provided
    cli_mode = len(sys.argv) > 1 and any(
        arg.startswith('-') or arg.startswith('--') for arg in sys.argv[1:]
    )

    try:
        if cli_mode:
            # Process arguments for CLI mode
            # Set defaults for CLI mode
            if args.executable is None:
                args.executable = ""
            if args.star_removal_mode is None:
                args.star_removal_mode = "additive"
            if args.chunk_size is None:
                args.chunk_size = 512
            # disable_gpu, show_extracted_stars, clear_input default to False
            if not args.disable_gpu:
                args.disable_gpu = False
            if not args.show_extracted_stars:
                args.show_extracted_stars = False
            if not args.clear_input:
                args.clear_input = False
            # overlap can remain None (will be computed as 12.5% of chunk_size)

            # Headless CLI run
            run_cli_mode(args)
        else:
            # GUI mode
            run_gui_mode()
    except Exception as e:
        print(f"Error initializing Darkstar wrapper: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
