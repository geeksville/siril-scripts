# (c) Adrian Knagg-Baugh 2024-2025
# SPDX-License-Identifier: GPL-3.0-or-later

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version: 2.1.3
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
# 2.1.0  PyQt6 threaded worker port
# 2.1.1  Fix pyscript operation
# 2.1.2  Fix GPU use when called from inside a flatpak sandbox
# 2.1.3  Improve undo history and log messaging (helps Workflow_Summarizer, closes
#        siril-scripts:#60)

import sirilpy as s
s.ensure_installed("tiffile", "PyQt6")

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import tiffile

VERSION = "2.1.3"

# ------------------------------
# Shared utility functions
# ------------------------------
def run_cosmic_clarity_sharpen_process(executable_path: str, mode: str, stellar_amount: float,
                                       non_stellar_strength: int, non_stellar_amount: float,
                                       use_gpu: bool, auto_psf: bool, separate_channels: bool,
                                       progress_callback=None):
    """
    Run Cosmic Clarity sharpening process and report progress.

    Args:
        executable_path: Path to Cosmic Clarity executable
        mode: Sharpening mode (Stellar Only, Non-Stellar Only, Both)
        stellar_amount: Stellar sharpening amount (0.0 to 1.0)
        non_stellar_strength: Non-stellar strength (1 to 8)
        non_stellar_amount: Non-stellar amount (0.0 to 1.0)
        use_gpu: Whether to use GPU
        auto_psf: Whether to auto-detect PSF
        separate_channels: Whether to sharpen channels separately
        progress_callback: Optional callback function(float) for progress updates (0.0 to 1.0)

    Returns:
        int: Process return code
    """
    command = [
        executable_path,
        f"--sharpening_mode={mode}",
        f"--stellar_amount={stellar_amount}",
        f"--nonstellar_strength={non_stellar_strength}",
        f"--nonstellar_amount={non_stellar_amount}",
    ]
    if not use_gpu:
        command.append("--disable_gpu")
    if auto_psf:
        command.append("--auto_detect_psf")
    if separate_channels:
        command.append("--sharpen_channels_separately")

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
    """Check for and load the configured executable path."""
    config_dir = siril.get_siril_configdir()
    config_file_path = os.path.join(config_dir, "sirilcc_sharpen.conf")
    if os.path.isfile(config_file_path):
        try:
            with open(config_file_path, "r") as f:
                p = f.readline().strip()
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        except Exception:
            pass
    print("Executable not yet configured. It is recommended to use Seti Astro Cosmic Clarity v6.5 or higher.")
    return None

def save_config_if_changed(siril, new_path: str, current_path: str):
    """Save the executable path to config if it has changed."""
    if new_path and new_path != current_path:
        config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_sharpen.conf")
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
        mode = args.sharpening_mode
        stellar_amount = args.stellar_amount
        non_stellar_strength = args.non_stellar_strength
        non_stellar_amount = args.non_stellar_amount
        use_gpu = args.use_gpu
        auto_psf = args.auto_psf
        separate_channels = args.separate_channels
        clear_input = args.clear_input_dir
        executable_path = args.executable

        # Fallback to config file if no executable provided
        if not executable_path:
            executable_path = check_config_file(siril) or ""
        if not executable_path or not os.path.isfile(executable_path):
            print("Error: please provide a valid Cosmic Clarity executable "
                  "(-executable) or configure one via GUI.", file=sys.stderr)
            sys.exit(1)

        # Save executable path if it differs from stored one
        config_executable = check_config_file(siril)
        save_config_if_changed(siril, executable_path, config_executable)

        # Prepare directories and filenames
        filename = siril.get_image_filename()
        directory = os.path.dirname(executable_path)
        os.makedirs(os.path.join(directory, "input"), exist_ok=True)
        os.makedirs(os.path.join(directory, "output"), exist_ok=True)

        basename = os.path.basename(filename)
        inputfilename = os.path.join(directory, "input", basename + ".tif")
        outputfilename = os.path.join(directory, "output", f"{basename}_sharpened.tif")

        if clear_input:
            for each_file in Path(os.path.join(directory, "input")).glob("*.*"):
                try:
                    each_file.unlink()
                except Exception:
                    pass

        # Load and prepare image
        pixels = siril.get_image_pixeldata()
        was_16bit = False
        if pixels.dtype == np.uint16:
            pixels = pixels.astype(np.float32) / 65535.0
            was_16bit = True

        tiff_data, photometry = prepare_image_for_tiff(pixels)
        tiffile.imwrite(inputfilename, tiff_data, photometric=photometry, planarconfig="contig")

        siril.update_progress("Seti Astro Cosmic Clarity Sharpen starting...", 0)

        # Run external process with progress reporting
        def progress_callback(pct):
            siril.update_progress("Seti Astro Cosmic Clarity Sharpen progress...", pct)

        ret = run_cosmic_clarity_sharpen_process(
            executable_path, mode, stellar_amount, non_stellar_strength, non_stellar_amount,
            use_gpu, auto_psf, separate_channels, progress_callback
        )

        if ret != 0 or not os.path.isfile(outputfilename):
            siril.reset_progress()
            print("Error: Cosmic Clarity sharpening process failed.", file=sys.stderr)
            sys.exit(1)

        # Load result and update image
        pixel_data = load_result_tiff(outputfilename)

        force_16bit = siril.get_siril_config("core", "force_16bit")
        if was_16bit or force_16bit:
            pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)

        with siril.image_lock():
            self.siril.undo_save_state(f"CC sharpen ({self.sharpening_mode}, "
                                                     f"Stel_amt={self.stellar_amount}, "
                                                     f"Nstel_str={self.non_stellar_strength}, "
                                                     f"Nstel_amt={self.non_stellar_amount})")
            siril.set_image_pixeldata(pixel_data)

        siril.reset_progress()
        print("Cosmic Clarity sharpening completed successfully ({self.sharpening_mode}, "
                                                     f"Stel_amt={self.stellar_amount}, "
                                                     f"Nstel_str={self.non_stellar_strength}, "
                                                     f"Nstel_amt={self.non_stellar_amount})")
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
        QRadioButton, QButtonGroup, QCheckBox, QSlider, QLineEdit, QPushButton,
        QFileDialog, QMessageBox, QGroupBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal

    class CosmicClaritySharpenWorker(QThread):
        progress = pyqtSignal(float)
        finished_ok = pyqtSignal(bool, str)

        def __init__(self, executable_path, mode, stellar_amount, non_stellar_strength, non_stellar_amount,
                     use_gpu, auto_psf, separate_channels, outputfilename):
            super().__init__()
            self.executable_path = executable_path
            self.mode = mode
            self.stellar_amount = stellar_amount
            self.non_stellar_strength = non_stellar_strength
            self.non_stellar_amount = non_stellar_amount
            self.use_gpu = use_gpu
            self.auto_psf = auto_psf
            self.separate_channels = separate_channels
            self.outputfilename = outputfilename

        def run(self):
            try:
                # Use shared process runner with progress callback
                def progress_callback(pct):
                    self.progress.emit(pct)

                ret = run_cosmic_clarity_sharpen_process(
                    self.executable_path,
                    self.mode,
                    self.stellar_amount,
                    self.non_stellar_strength,
                    self.non_stellar_amount,
                    self.use_gpu,
                    self.auto_psf,
                    self.separate_channels,
                    progress_callback
                )

                self.finished_ok.emit(ret == 0, self.outputfilename)
            except Exception:
                self.finished_ok.emit(False, "")

    class SirilCosmicClarityInterface(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle(f"Cosmic Clarity Sharpening - v{VERSION}")
            self.setFixedSize(500, 600)

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

            # Widget state vars
            self.sharpening_mode = "Stellar Only"
            self.use_gpu = True
            self.clear_input_dir = True
            self.auto_psf = True
            self.separate_channels = False
            self.stellar_amount = 0.5
            self.non_stellar_amount = 0.5
            self.non_stellar_strength = 3
            self.executable_path = self.config_executable or ""

            self._create_widgets()

        def _create_widgets(self):
            central = QWidget()
            layout = QVBoxLayout(central)

            title = QLabel("Cosmic Clarity Sharpening Settings")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(title)

            mode_groupbox = QGroupBox("Sharpening Mode")
            mode_layout = QVBoxLayout()
            self.mode_buttons = QButtonGroup(self)
            for mode in ["Stellar Only", "Non-Stellar Only", "Both"]:
                btn = QRadioButton(mode)
                if mode == self.sharpening_mode:
                    btn.setChecked(True)
                btn.toggled.connect(lambda checked, m=mode: self._set_mode(m, checked))
                self.mode_buttons.addButton(btn)
                mode_layout.addWidget(btn)
            mode_groupbox.setLayout(mode_layout)
            layout.addWidget(mode_groupbox)

            options_groupbox = QGroupBox("Options")
            options_layout = QVBoxLayout()

            self.gpu_checkbox = QCheckBox("Use GPU")
            self.gpu_checkbox.setChecked(self.use_gpu)
            self.gpu_checkbox.toggled.connect(lambda v: setattr(self, "use_gpu", v))
            options_layout.addWidget(self.gpu_checkbox)

            self.clear_checkbox = QCheckBox("Clear input directory")
            self.clear_checkbox.setChecked(self.clear_input_dir)
            self.clear_checkbox.toggled.connect(lambda v: setattr(self, "clear_input_dir", v))
            options_layout.addWidget(self.clear_checkbox)

            self.psf_checkbox = QCheckBox("Autodetect PSF")
            self.psf_checkbox.setChecked(self.auto_psf)
            self.psf_checkbox.toggled.connect(lambda v: setattr(self, "auto_psf", v))
            options_layout.addWidget(self.psf_checkbox)

            stellar_layout = QHBoxLayout()
            stellar_layout.addWidget(QLabel("Stellar Amount:"))
            self.stellar_slider = QSlider(Qt.Orientation.Horizontal)
            self.stellar_slider.setRange(0, 100)
            self.stellar_slider.setValue(int(self.stellar_amount * 100))
            self.stellar_slider.valueChanged.connect(self._update_stellar_amount)
            stellar_layout.addWidget(self.stellar_slider)
            self.stellar_label = QLabel(f"{self.stellar_amount:.2f}")
            stellar_layout.addWidget(self.stellar_label)
            options_layout.addLayout(stellar_layout)

            non_stellar_layout = QHBoxLayout()
            non_stellar_layout.addWidget(QLabel("Non-Stellar Amount:"))
            self.non_stellar_slider = QSlider(Qt.Orientation.Horizontal)
            self.non_stellar_slider.setRange(0, 100)
            self.non_stellar_slider.setValue(int(self.non_stellar_amount * 100))
            self.non_stellar_slider.valueChanged.connect(self._update_non_stellar_amount)
            non_stellar_layout.addWidget(self.non_stellar_slider)
            self.non_stellar_label = QLabel(f"{self.non_stellar_amount:.2f}")
            non_stellar_layout.addWidget(self.non_stellar_label)
            options_layout.addLayout(non_stellar_layout)

            strength_layout = QHBoxLayout()
            strength_layout.addWidget(QLabel("Non-Stellar Strength:"))
            self.strength_slider = QSlider(Qt.Orientation.Horizontal)
            self.strength_slider.setRange(1, 8)
            self.strength_slider.setValue(self.non_stellar_strength)
            self.strength_slider.valueChanged.connect(self._update_non_stellar_strength)
            strength_layout.addWidget(self.strength_slider)
            self.strength_label = QLabel(str(self.non_stellar_strength))
            strength_layout.addWidget(self.strength_label)
            options_layout.addLayout(strength_layout)

            options_groupbox.setLayout(options_layout)
            layout.addWidget(options_groupbox)

            exec_groupbox = QGroupBox("Cosmic Clarity Executable")
            exec_layout = QHBoxLayout()
            self.exec_entry = QLineEdit(self.executable_path)
            exec_layout.addWidget(self.exec_entry)
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(self._browse_executable)
            exec_layout.addWidget(browse_btn)
            exec_groupbox.setLayout(exec_layout)
            layout.addWidget(exec_groupbox)

            btn_layout = QHBoxLayout()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.close)
            btn_layout.addWidget(close_btn)

            self.apply_btn = QPushButton("Apply")
            self.apply_btn.clicked.connect(self._on_apply)
            btn_layout.addWidget(self.apply_btn)

            layout.addLayout(btn_layout)

            self.setCentralWidget(central)

        def _set_mode(self, mode, checked):
            if checked:
                self.sharpening_mode = mode

        def _update_stellar_amount(self, value):
            self.stellar_amount = value / 100.0
            self.stellar_label.setText(f"{self.stellar_amount:.2f}")

        def _update_non_stellar_amount(self, value):
            self.non_stellar_amount = value / 100.0
            self.non_stellar_label.setText(f"{self.non_stellar_amount:.2f}")

        def _update_non_stellar_strength(self, value):
            self.non_stellar_strength = value
            self.strength_label.setText(str(value))

        def _browse_executable(self):
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select Cosmic Clarity Executable", os.path.expanduser("~")
            )
            if filename:
                self.executable_path = filename
                self.exec_entry.setText(filename)

        def _on_apply(self):
            try:
                mode = self.sharpening_mode
                stellar_amount = self.stellar_amount
                non_stellar_strength = self.non_stellar_strength
                non_stellar_amount = self.non_stellar_amount
                executable_path = self.exec_entry.text().strip()
                clear_input = self.clear_input_dir

                if not executable_path or not os.path.isfile(executable_path):
                    QMessageBox.warning(self, "Executable", "Please select a valid Cosmic Clarity executable.")
                    return

                # Persist executable path if changed
                self.config_executable = save_config_if_changed(self.siril, executable_path, self.config_executable)

                filename = self.siril.get_image_filename()
                directory = os.path.dirname(executable_path)
                basename = os.path.basename(filename)
                os.chdir(directory)
                os.makedirs("input", exist_ok=True)
                os.makedirs("output", exist_ok=True)

                inputpath = os.path.join(directory, "input")
                inputfilename = os.path.join(inputpath, basename) + ".tif"
                outputpath = os.path.join(directory, "output")
                outputfilename = os.path.join(outputpath, f"{basename}_sharpened.tif")

                if clear_input:
                    files = Path(inputpath).glob("*.*")
                    for each_file in files:
                        try:
                            each_file.unlink()
                        except Exception as e:
                            print(f"Failed to delete {each_file}: {e}")

                was_16bit = False
                pixels = self.siril.get_image_pixeldata()
                if pixels.dtype == np.uint16:
                    pixels = pixels.astype(np.float32) / 65535.0
                    was_16bit = True

                tiff_data, photometry = prepare_image_for_tiff(pixels)
                tiffile.imwrite(inputfilename, tiff_data, photometric=photometry, planarconfig="contig")

                self.siril.update_progress("Seti Astro Cosmic Clarity Sharpen starting...", 0)

                # Disable Apply button until the worker finishes
                self.apply_btn.setEnabled(False)

                self._worker = CosmicClaritySharpenWorker(
                    executable_path, mode, stellar_amount, non_stellar_strength, non_stellar_amount,
                    self.use_gpu, self.auto_psf, self.separate_channels, outputfilename
                )
                self._worker.progress.connect(self._on_progress)
                self._worker.finished_ok.connect(lambda success, fn: self._on_finished(success, fn, was_16bit, mode))
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
                self.siril.update_progress("Seti Astro Cosmic Clarity Sharpen progress...", float(frac))
            except Exception:
                pass

        def _on_finished(self, success, outputfilename, was_16bit, mode):
            try:
                self.apply_btn.setEnabled(True)
            except Exception:
                pass

            try:
                if not success:
                    self.siril.reset_progress()
                    QMessageBox.critical(self, "Cosmic Clarity", "Sharpening failed.")
                    return

                pixel_data = load_result_tiff(outputfilename)

                force_16bit = self.siril.get_siril_config("core", "force_16bit")
                if was_16bit or force_16bit:
                    pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)

                with self.siril.image_lock():
                    self.siril.undo_save_state(f"CC sharpen ({self.sharpening_mode}, "
                                                     f"Stel_amt={self.stellar_amount}, "
                                                     f"Nstel_str={self.non_stellar_strength}, "
                                                     f"Nstel_amt={self.non_stellar_amount})")
                    self.siril.set_image_pixeldata(pixel_data)

                self.siril.reset_progress()
                print("Cosmic Clarity sharpening completed successfully ({self.sharpening_mode}, "
                                                     f"Stel_amt={self.stellar_amount}, "
                                                     f"Nstel_str={self.non_stellar_strength}, "
                                                     f"Nstel_amt={self.non_stellar_amount})")
            except Exception as e:
                print(f"Error in finished handler: {e}")
                self.siril.reset_progress()

    # Create and run the GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SirilCosmicClarityInterface()
    win.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity Sharpening Script")
    parser.add_argument("-sharpening_mode", type=str, choices=["Stellar Only", "Non-Stellar Only", "Both"],
                        help="Sharpening mode")
    parser.add_argument("-stellar_amount", type=float,
                        help="Stellar sharpening amount (0.0 to 1.0)")
    parser.add_argument("-non_stellar_strength", type=int,
                        help="Non-stellar strength (1 to 8)")
    parser.add_argument("-non_stellar_amount", type=float,
                        help="Non-stellar amount (0.0 to 1.0)")
    parser.add_argument("-use_gpu", action="store_true",
                        help="Use GPU")
    parser.add_argument("-no_gpu", action="store_true",
                        help="Disable GPU")
    parser.add_argument("-auto_psf", action="store_true",
                        help="Auto-detect PSF")
    parser.add_argument("-no_auto_psf", action="store_true",
                        help="Disable auto PSF detection")
    parser.add_argument("-separate_channels", action="store_true",
                        help="Sharpen channels separately")
    parser.add_argument("-executable", type=str,
                        help="Path to Cosmic Clarity executable")
    parser.add_argument("-clear_input_dir", action="store_true",
                        help="Clear input directory")
    parser.add_argument("-no_clear_input", action="store_true",
                        help="Do not clear input directory")

    args, unknown = parser.parse_known_args()

    # Detect CLI mode: if ANY argument was provided
    cli_mode = len(sys.argv) > 1 and any(
        arg.startswith('-') for arg in sys.argv[1:]
    )

    try:
        if cli_mode:
            # Process arguments for CLI mode
            # Set defaults for CLI mode
            if args.sharpening_mode is None:
                args.sharpening_mode = "Stellar Only"
            if args.stellar_amount is None:
                args.stellar_amount = 0.5
            if args.non_stellar_strength is None:
                args.non_stellar_strength = 3
            if args.non_stellar_amount is None:
                args.non_stellar_amount = 0.5
            if args.executable is None:
                args.executable = ""

            # Handle GPU flags
            if args.no_gpu:
                args.use_gpu = False
            elif not args.use_gpu:
                args.use_gpu = True

            # Handle auto PSF flags
            if args.no_auto_psf:
                args.auto_psf = False
            elif not args.auto_psf:
                args.auto_psf = True

            # Handle separate channels (default False)
            if not args.separate_channels:
                args.separate_channels = False

            # Handle clear input flags
            if args.no_clear_input:
                args.clear_input_dir = False
            elif not args.clear_input_dir:
                args.clear_input_dir = True

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
