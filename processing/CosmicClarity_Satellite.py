"""
# Cosmic Clarity Satellite Removal Script
# SPDX-License-Identifier: GPL-3.0-or-later
# (c) Adrian Knagg-Baugh 2025

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version: 1.2.1
# 1.2.0: First PyQt6 version of the script
# 1.2.1: Updated to work better with pyscript. Note that unlike other CosmicClarity scripts this
#        one shows its own GUI, so it may not work 100% headlessly though it does work in scripts
#        called from Siril's GUI. This is a limitation of CC and not something I can work around.
"""

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

VERSION = "1.2.1"

# ------------------------------
# Shared utility functions
# ------------------------------
def run_cosmic_clarity_satellite_process(executable_path: str, input_dir: str,
                                        output_dir: str, mode: str, sensitivity: float,
                                        use_gpu: bool, monitor: bool, skip_save: bool,
                                        clip_trail: bool, progress_callback=None):
    """
    Run Cosmic Clarity Satellite removal process and report progress.

    Args:
        executable_path: Path to Cosmic Clarity Satellite executable
        input_dir: Input directory path
        output_dir: Output directory path
        mode: Processing mode (full or luminance)
        sensitivity: Detection sensitivity (0.01 to 0.5)
        use_gpu: Whether to use GPU
        monitor: Monitor mode (else batch)
        skip_save: Skip save if no trail detected
        clip_trail: Clip trail to 0.000
        progress_callback: Optional callback function(float) for progress updates (0.0 to 1.0)

    Returns:
        int: Process return code
    """
    cmd = [
        executable_path,
        f"--input={input_dir}",
        f"--output={output_dir}",
        f"--mode={mode}",
        f"--sensitivity={sensitivity}",
    ]
    if use_gpu:
        cmd.append("--use-gpu")
    if monitor:
        cmd.append("--monitor")
    else:
        cmd.append("--batch")
    if skip_save:
        cmd.append("--skip-save")
    if clip_trail:
        cmd.append("--clip-trail")
    else:
        cmd.append("--no-clip-trail")

    process = subprocess.Popen(
        cmd,
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
    config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_satellite.conf")
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
        config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_satellite.conf")
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
        output = pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)
        return output, photometry
    else:
        raise ValueError(f"Unexpected image shape: {pixels.shape}")

def load_result_tiff(filename: str):
    """Load a TIFF file and convert to channel-first format."""
    with tiffile.TiffFile(filename) as tiff:
        pixel_data = tiff.asarray()
    pixel_data = np.ascontiguousarray(pixel_data)

    if pixel_data.ndim == 2:
        pixel_data = pixel_data[np.newaxis, :, :]
    elif pixel_data.ndim == 3 and pixel_data.shape[2] == 3:
        pixel_data = np.transpose(pixel_data, (2, 0, 1))
        pixel_data = np.ascontiguousarray(pixel_data)

    return pixel_data

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
            print("Error: please provide a valid Cosmic Clarity Satellite executable "
                  "(-executable) or configure one via GUI.", file=sys.stderr)
            sys.exit(1)

        # Save executable path if it differs from stored one
        config_executable = check_config_file(siril)
        save_config_if_changed(siril, executable_path, config_executable)

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
            pixels = pixels.astype(np.float32) / 65535.0
        else:
            pixels = pixels.astype(np.float32, copy=False)

        # Prepare for writing
        write_arr, photometry = prepare_image_for_tiff(pixels)

        # Generate filenames
        basename = os.path.basename(siril.get_image_filename())
        input_file = os.path.join(input_dir, basename + ".tif")
        output_file = os.path.join(output_dir, f"{basename}_satellited.tif")

        # Write input TIFF
        tiffile.imwrite(input_file, write_arr, photometric=photometry, planarconfig="contig")

        siril.update_progress("Cosmic Clarity Satellite Removal starting...", 0)

        # Run external process with progress reporting
        def progress_callback(pct):
            siril.update_progress("Cosmic Clarity Satellite Removal progress...", pct)

        ret = run_cosmic_clarity_satellite_process(
            executable_path, input_dir, output_dir, args.mode, args.sensitivity,
            args.use_gpu, args.monitor, args.skip_save, args.clip_trail,
            progress_callback
        )

        if ret != 0:
            siril.reset_progress()
            # If skip_save and no output, it might be intentional (no trail detected)
            if args.skip_save and not os.path.isfile(output_file):
                print("No satellite trail detected (skip-save mode).")
                sys.exit(0)
            print("Error: Cosmic Clarity Satellite removal process failed.", file=sys.stderr)
            sys.exit(1)

        # Check if output exists
        if not os.path.isfile(output_file):
            siril.reset_progress()
            if args.skip_save:
                print("No satellite trail detected (skip-save mode).")
                sys.exit(0)
            print("Error: Output file not created.", file=sys.stderr)
            sys.exit(1)

        # Load result
        pixel_data = load_result_tiff(output_file)

        # Convert to appropriate dtype
        force_16 = siril.get_siril_config("core", "force_16bit")
        if was_16bit or force_16:
            pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)

        with siril.image_lock():
            siril.undo_save_state("Satellite removal")
            siril.set_image_pixeldata(pixel_data)

        siril.reset_progress()
        print(f"Cosmic Clarity Satellite removal completed successfully. Output: {output_file}")
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
        QComboBox, QSlider
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal

    class CosmicClaritySatelliteWorker(QThread):
        progress = pyqtSignal(float)
        finished_ok = pyqtSignal(bool, str)

        def __init__(self, executable_path, exe_dir, mode, sensitivity, use_gpu,
                     monitor, skip_save, clip_trail, clear_input, input_file,
                     output_file, pixels, was_16bit, siril):
            super().__init__()
            self.executable_path = executable_path
            self.exe_dir = exe_dir
            self.mode = mode
            self.sensitivity = sensitivity
            self.use_gpu = use_gpu
            self.monitor = monitor
            self.skip_save = skip_save
            self.clip_trail = clip_trail
            self.clear_input = clear_input
            self.input_file = input_file
            self.output_file = output_file
            self.pixels = pixels
            self.was_16bit = was_16bit
            self.siril = siril

        def run(self):
            try:
                input_dir = os.path.join(self.exe_dir, "input")
                output_dir = os.path.join(self.exe_dir, "output")

                # Clear input if requested
                if self.clear_input and os.path.isdir(input_dir):
                    for f in Path(input_dir).glob("*.*"):
                        try:
                            f.unlink()
                        except Exception:
                            pass

                # Write input TIFF
                write_arr, photometry = prepare_image_for_tiff(self.pixels)
                tiffile.imwrite(self.input_file, write_arr, photometric=photometry, planarconfig="contig")

                # Use shared process runner with progress callback
                def progress_callback(pct):
                    self.progress.emit(pct)

                ret = run_cosmic_clarity_satellite_process(
                    self.executable_path, input_dir, output_dir, self.mode,
                    self.sensitivity, self.use_gpu, self.monitor, self.skip_save,
                    self.clip_trail, progress_callback
                )

                if ret != 0:
                    if self.skip_save and not os.path.isfile(self.output_file):
                        self.finished_ok.emit(True, "no detection: skipped output")
                    else:
                        self.finished_ok.emit(False, "")
                    return

                # Check if output exists
                if not os.path.isfile(self.output_file):
                    if self.skip_save:
                        self.finished_ok.emit(True, "no detection: skipped output")
                    else:
                        self.finished_ok.emit(False, "")
                    return

                # Load result
                pixel_data = load_result_tiff(self.output_file)

                # Convert to appropriate dtype
                force_16 = self.siril.get_siril_config("core", "force_16bit")
                if self.was_16bit or force_16:
                    pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)

                # Save undo + update image
                with self.siril.image_lock():
                    self.siril.undo_save_state("Satellite removal")
                    self.siril.set_image_pixeldata(pixel_data)

                self.finished_ok.emit(True, self.output_file)

            except Exception as e:
                print(f"Error in worker: {e}", file=sys.stderr)
                self.finished_ok.emit(False, "")

    class SirilCosmicClaritySatellite(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle(f"Cosmic Clarity Satellite Removal - v{VERSION}")
            self.setFixedSize(600, 500)

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

            # Defaults
            self.executable_path = self.config_executable or ""
            self.use_gpu = True
            self.mode = "full"
            self.monitor = False
            self.skip_save = False
            self.sensitivity = 0.1
            self.clip_trail = True
            self.clear_input = True

            self._create_widgets()

        def _create_widgets(self):
            central = QWidget()
            layout = QVBoxLayout(central)

            title = QLabel("Cosmic Clarity Satellite Removal Settings")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(title)

            # Mode
            mode_box = QGroupBox("Mode")
            mode_layout = QHBoxLayout()
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["full", "luminance"])
            self.mode_combo.setCurrentText(self.mode)
            mode_layout.addWidget(self.mode_combo)
            mode_box.setLayout(mode_layout)
            layout.addWidget(mode_box)

            # Sensitivity slider
            sens_groupbox = QGroupBox("Sensitivity (0.01 - 0.5)")
            sens_layout = QHBoxLayout()
            self.sens_slider = QSlider(Qt.Orientation.Horizontal)
            self.sens_slider.setRange(1, 50)
            self.sens_slider.setValue(int(self.sensitivity * 100))
            self.sens_slider.valueChanged.connect(self._update_sensitivity)
            self.sens_label = QLabel(str(self.sensitivity))
            sens_layout.addWidget(self.sens_slider)
            sens_layout.addWidget(self.sens_label)
            sens_groupbox.setLayout(sens_layout)
            layout.addWidget(sens_groupbox)

            # Options
            options_box = QGroupBox("Options")
            options_layout = QVBoxLayout()

            self.gpu_checkbox = QCheckBox("Use GPU")
            self.gpu_checkbox.setChecked(self.use_gpu)
            options_layout.addWidget(self.gpu_checkbox)

            self.monitor_checkbox = QCheckBox("Monitor folder (else batch mode)")
            self.monitor_checkbox.setChecked(self.monitor)
            options_layout.addWidget(self.monitor_checkbox)

            self.skip_checkbox = QCheckBox("Skip save if no trail")
            self.skip_checkbox.setChecked(self.skip_save)
            options_layout.addWidget(self.skip_checkbox)

            self.clip_checkbox = QCheckBox("Clip trail to 0.000")
            self.clip_checkbox.setChecked(self.clip_trail)
            options_layout.addWidget(self.clip_checkbox)

            self.clear_checkbox = QCheckBox("Clear input directory before run")
            self.clear_checkbox.setChecked(self.clear_input)
            options_layout.addWidget(self.clear_checkbox)

            options_box.setLayout(options_layout)
            layout.addWidget(options_box)

            # Executable selection
            exec_groupbox = QGroupBox("Executable")
            exec_layout = QHBoxLayout()
            self.exec_entry = QLineEdit(self.executable_path)
            exec_layout.addWidget(self.exec_entry)
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(self._browse_executable)
            exec_layout.addWidget(browse_btn)
            exec_groupbox.setLayout(exec_layout)
            layout.addWidget(exec_groupbox)

            # Buttons
            btn_layout = QHBoxLayout()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.close)
            btn_layout.addWidget(close_btn)

            self.apply_btn = QPushButton("Apply")
            self.apply_btn.clicked.connect(self._on_apply)
            btn_layout.addWidget(self.apply_btn)

            layout.addLayout(btn_layout)
            self.setCentralWidget(central)

        def _update_sensitivity(self, value):
            self.sensitivity = value / 100.0
            self.sens_label.setText(f"{self.sensitivity:.2f}")

        def _browse_executable(self):
            filename, _ = QFileDialog.getOpenFileName(self, "Select Executable", os.path.expanduser("~"))
            if filename:
                self.executable_path = filename
                self.exec_entry.setText(filename)

        def _on_apply(self):
            try:
                executable_path = self.exec_entry.text().strip()

                if not executable_path or not os.path.isfile(executable_path):
                    QMessageBox.critical(self, "Executable", "Please select a valid Cosmic Clarity Satellite executable.")
                    return

                # Persist executable path if changed
                self.config_executable = save_config_if_changed(self.siril, executable_path, self.config_executable)

                # Setup directories and filenames
                exe_dir = os.path.dirname(executable_path)
                input_dir = os.path.join(exe_dir, "input")
                output_dir = os.path.join(exe_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                basename = os.path.basename(self.siril.get_image_filename())
                input_file = os.path.join(input_dir, basename + ".tif")
                output_file = os.path.join(output_dir, f"{basename}_satellited.tif")

                # Get image data
                pixels = self.siril.get_image_pixeldata()
                was_16bit = (pixels.dtype == np.uint16)
                if was_16bit:
                    pixels = pixels.astype(np.float32) / 65535.0
                else:
                    pixels = pixels.astype(np.float32, copy=False)

                self.siril.update_progress("Cosmic Clarity Satellite Removal starting...", 0)

                # Disable apply while running
                self.apply_btn.setEnabled(False)

                # Start worker
                self._worker = CosmicClaritySatelliteWorker(
                    executable_path, exe_dir,
                    self.mode_combo.currentText(),
                    self.sensitivity,
                    self.gpu_checkbox.isChecked(),
                    self.monitor_checkbox.isChecked(),
                    self.skip_checkbox.isChecked(),
                    self.clip_checkbox.isChecked(),
                    self.clear_checkbox.isChecked(),
                    input_file, output_file,
                    pixels, was_16bit, self.siril
                )
                self._worker.progress.connect(self._on_progress)
                self._worker.finished_ok.connect(self._on_finished)
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
                self.siril.update_progress("Cosmic Clarity Satellite Removal progress...", float(frac))
            except Exception:
                pass

        def _on_finished(self, success, outputfile):
            try:
                self.apply_btn.setEnabled(True)
            except Exception:
                pass

            try:
                self.siril.reset_progress()
            except Exception:
                pass

            if not success:
                QMessageBox.critical(self, "Cosmic Clarity", "Satellite removal failed.")
            else:
                self.siril.log(f"Satellite removal complete. Output: {outputfile}")

    # Create and run the GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SirilCosmicClaritySatellite()
    win.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity Satellite Removal Script")
    parser.add_argument("-executable", type=str,
                        help="Path to executable")
    parser.add_argument("-use_gpu", action="store_true",
                        help="Use GPU")
    parser.add_argument("-no_gpu", action="store_true",
                        help="Disable GPU")
    parser.add_argument("-mode", choices=["full", "luminance"],
                        help="Processing mode")
    parser.add_argument("-monitor", action="store_true",
                        help="Monitor mode (else batch)")
    parser.add_argument("-skip_save", action="store_true",
                        help="Skip save if no trail detected")
    parser.add_argument("-sensitivity", type=float,
                        help="Detection sensitivity (0.01 to 0.5)")
    parser.add_argument("-clip_trail", action="store_true",
                        help="Clip trail to 0.000")
    parser.add_argument("-no_clip_trail", action="store_true",
                        help="Don't clip trail")
    parser.add_argument("-clear_input", action="store_true",
                        help="Clear input directory before run")

    args, unknown = parser.parse_known_args()

    # Detect CLI mode: if ANY argument was provided
    cli_mode = len(sys.argv) > 1 and any(
        arg.startswith('-') for arg in sys.argv[1:]
    )

    try:
        if cli_mode:
            # Process arguments for CLI mode
            # Set defaults for CLI mode
            if args.executable is None:
                args.executable = ""
            if args.mode is None:
                args.mode = "full"
            if args.sensitivity is None:
                args.sensitivity = 0.1

            # Handle GPU flags
            if args.no_gpu:
                args.use_gpu = False
            elif not args.use_gpu:
                args.use_gpu = True

            # Handle clip_trail flags
            if args.no_clip_trail:
                args.clip_trail = False
            elif not args.clip_trail:
                args.clip_trail = True

            # Other boolean flags default to False
            if not args.monitor:
                args.monitor = False
            if not args.skip_save:
                args.skip_save = False
            if not args.clear_input:
                args.clear_input = False

            # Headless CLI run
            run_cli_mode(args)
        else:
            # GUI mode
            run_gui_mode()
    except Exception as e:
        print(f"Error initializing application: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
