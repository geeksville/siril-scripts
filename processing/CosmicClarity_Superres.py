"""
# (c) EnderByBear 2025
# Inspired by CosmicClarity_Sharpen.py developed by Adrian Knagg-Baugh
# SPDX-License-Identifier: GPL-3.0-or-later

# Contact: report issues with this script to enderbybear (at) foxmail.com or at
# https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version: 2.0.1
# 2.0.0 PyQt6 threaded worker port by AKB
# 2.0.1 Fixes operation with pyscript
"""

import sirilpy as s
s.ensure_installed("PyQt6")

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

VERSION = "2.0.1"

# ------------------------------
# Shared utility functions
# ------------------------------
def run_cosmic_clarity_superres_process(executable_path: str, input_file: str,
                                        output_dir: str, scale: int,
                                        model_dir: str, progress_callback=None):
    """
    Run Cosmic Clarity superres process and report progress.

    Args:
        executable_path: Path to Cosmic Clarity executable
        input_file: Input file path
        output_dir: Output directory path
        scale: Upscale factor (2, 3, or 4)
        model_dir: Model directory path
        progress_callback: Optional callback function(float) for progress updates (0.0 to 1.0)

    Returns:
        int: Process return code
    """
    command = [
        executable_path,
        f"--input={input_file}",
        f"--output_dir={output_dir}",
        f"--scale={scale}",
        f"--model_dir={model_dir}"
    ]

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
    config_file_path = os.path.join(config_dir, "sirilcc_superres.conf")
    if os.path.isfile(config_file_path):
        try:
            with open(config_file_path, "r") as f:
                p = f.readline().strip()
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        except Exception:
            pass
    print("Executable not yet configured. Recommended to use Seti Astro Cosmic Clarity Super-Resolution Upscaling Tool v1.1 or higher.")
    return None

def save_config_if_changed(siril, new_path: str, current_path: str):
    """Save the executable path to config if it has changed."""
    if new_path and new_path != current_path:
        config_file_path = os.path.join(siril.get_siril_configdir(), "sirilcc_superres.conf")
        try:
            with open(config_file_path, "w") as f:
                f.write(new_path + "\n")
        except Exception as e:
            print(f"Failed to write config: {e}")
        return new_path
    return current_path

def get_output_filename(input_file: str, scale: int):
    """Generate output filename for upscaled image."""
    return f"{os.path.splitext(input_file)[0]}_upscaled{scale}x.fit"

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
        scale = args.scale
        load_upscaled = args.load_upscaled
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

        # Prepare paths
        input_file = siril.get_image_filename()
        output_dir = siril.get_siril_wd()
        output_filename = get_output_filename(input_file, scale)
        model_dir = os.path.dirname(executable_path)

        siril.update_progress("Seti Astro Cosmic Clarity Superres starting...", 0)

        # Run external process with progress reporting
        def progress_callback(pct):
            siril.update_progress("Seti Astro Cosmic Clarity Superres progress...", pct)

        ret = run_cosmic_clarity_superres_process(
            executable_path, input_file, output_dir, scale, model_dir, progress_callback
        )

        if ret != 0 or not os.path.exists(output_filename):
            siril.reset_progress()
            print("Error: Cosmic Clarity superres process failed.", file=sys.stderr)
            sys.exit(1)

        # Optionally load the upscaled image
        if load_upscaled and os.path.exists(output_filename):
            siril.log(f"Loading upscaled image: {output_filename}")
            siril.cmd(f"load \"{output_filename}\"")

        siril.reset_progress()
        print("Cosmic Clarity superres completed successfully.")
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
        QCheckBox, QSlider, QLineEdit, QPushButton,
        QFileDialog, QMessageBox, QGroupBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal

    class CosmicClaritySuperresWorker(QThread):
        progress = pyqtSignal(float)
        finished_ok = pyqtSignal(bool, str)

        def __init__(self, input_file, output_dir, executable_path, scale, model_dir):
            super().__init__()
            self.input_file = input_file
            self.output_dir = output_dir
            self.executable_path = executable_path
            self.scale = scale
            self.model_dir = model_dir

        def run(self):
            try:
                # Use shared process runner with progress callback
                def progress_callback(pct):
                    self.progress.emit(pct)

                output_filename = get_output_filename(self.input_file, self.scale)

                ret = run_cosmic_clarity_superres_process(
                    self.executable_path,
                    self.input_file,
                    self.output_dir,
                    self.scale,
                    self.model_dir,
                    progress_callback
                )

                self.finished_ok.emit(ret == 0, output_filename)
            except Exception as e:
                print(f"Error in worker: {e}")
                self.finished_ok.emit(False, "")

    class SirilCosmicClarityInterface(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle(f"Cosmic Clarity Superres - v{VERSION}")
            self.setFixedSize(500, 400)

            # Initialize Siril connection
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

            # State vars for GUI
            self.scale = 2
            self.load_upscaled = False
            self.executable_path = self.config_executable or ""
            self._create_widgets()

        def _create_widgets(self):
            central = QWidget()
            layout = QVBoxLayout(central)

            title = QLabel("Cosmic Clarity Superres Settings")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(title)

            # Options
            options_groupbox = QGroupBox("Options")
            options_layout = QVBoxLayout()

            scale_layout = QHBoxLayout()
            scale_layout.addWidget(QLabel("Upscale Factor:"))
            self.scale_slider = QSlider(Qt.Orientation.Horizontal)
            self.scale_slider.setRange(2, 4)
            self.scale_slider.setValue(self.scale)
            self.scale_slider.valueChanged.connect(self._update_scale)
            scale_layout.addWidget(self.scale_slider)
            self.scale_label = QLabel(str(self.scale))
            scale_layout.addWidget(self.scale_label)
            options_layout.addLayout(scale_layout)

            self.load_checkbox = QCheckBox("Load Upscaled Image When Finished")
            self.load_checkbox.setChecked(self.load_upscaled)
            self.load_checkbox.toggled.connect(lambda v: setattr(self, "load_upscaled", v))
            options_layout.addWidget(self.load_checkbox)

            options_groupbox.setLayout(options_layout)
            layout.addWidget(options_groupbox)

            # Executable selection
            exec_groupbox = QGroupBox("Cosmic Clarity Executable")
            exec_layout = QHBoxLayout()
            self.exec_entry = QLineEdit(self.executable_path)
            exec_layout.addWidget(self.exec_entry)
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(self._browse_executable)
            exec_layout.addWidget(browse_btn)
            exec_groupbox.setLayout(exec_layout)
            layout.addWidget(exec_groupbox)

            # Action buttons
            btn_layout = QHBoxLayout()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.close)
            btn_layout.addWidget(close_btn)

            self.apply_btn = QPushButton("Apply")
            self.apply_btn.clicked.connect(self._on_apply)
            btn_layout.addWidget(self.apply_btn)

            layout.addLayout(btn_layout)

            self.setCentralWidget(central)

        def _update_scale(self, value):
            self.scale = value
            self.scale_label.setText(str(value))

        def _browse_executable(self):
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select Cosmic Clarity Executable", os.path.expanduser("~")
            )
            if filename:
                self.executable_path = filename
                self.exec_entry.setText(filename)

        def _on_apply(self):
            try:
                executable_path = self.exec_entry.text().strip()
                scale = self.scale
                load_upscaled = self.load_upscaled

                if not executable_path or not os.path.isfile(executable_path):
                    QMessageBox.warning(self, "Executable", "Please select a valid Cosmic Clarity executable.")
                    return

                # Persist executable path if changed
                self.config_executable = save_config_if_changed(self.siril, executable_path, self.config_executable)

                input_file = self.siril.get_image_filename()
                output_dir = self.siril.get_siril_wd()
                model_dir = os.path.dirname(executable_path)

                self.siril.update_progress("Seti Astro Cosmic Clarity Superres starting...", 0)

                # Disable Apply button until finished
                self.apply_btn.setEnabled(False)

                self._worker = CosmicClaritySuperresWorker(
                    input_file, output_dir, executable_path, scale, model_dir
                )
                self._worker.progress.connect(self._on_progress)
                self._worker.finished_ok.connect(
                    lambda success, fn: self._on_finished(success, fn, load_upscaled)
                )
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
                self.siril.update_progress("Seti Astro Cosmic Clarity Superres progress...", float(frac))
            except Exception:
                pass

        def _on_finished(self, success, outputfilename, load_upscaled):
            try:
                # Re-enable Apply button
                self.apply_btn.setEnabled(True)
            except Exception:
                pass

            try:
                if not success:
                    self.siril.reset_progress()
                    QMessageBox.critical(self, "Cosmic Clarity", "Superres failed.")
                    return

                if load_upscaled and os.path.exists(outputfilename):
                    self.siril.log(f"Loading upscaled image: {outputfilename}")
                    self.siril.cmd(f"load \"{outputfilename}\"")

                self.siril.reset_progress()
                self.siril.log("Cosmic Clarity superres complete.")

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
    parser = argparse.ArgumentParser(description="Cosmic Clarity Superres Script")
    parser.add_argument("-scale", type=int, choices=[2, 3, 4],
                        help="Upscale factor (2x, 3x, or 4x)")
    parser.add_argument("-load_upscaled", action="store_true",
                        help="Load upscaled image into Siril after completion")
    parser.add_argument("-executable", type=str,
                        help="Path to Cosmic Clarity executable")

    args, unknown = parser.parse_known_args()

    # Detect CLI mode: if ANY argument was provided
    cli_mode = len(sys.argv) > 1 and any(
        arg.startswith('-') for arg in sys.argv[1:]
    )

    try:
        if cli_mode:
            # Process arguments for CLI mode
            # Set defaults for CLI mode
            if args.scale is None:
                args.scale = 2
            if args.executable is None:
                args.executable = ""
            # load_upscaled defaults to False when not specified
            if not args.load_upscaled:
                args.load_upscaled = False

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
