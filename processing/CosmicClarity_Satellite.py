# Cosmic Clarity Satellite Removal Script
# SPDX-License-Identifier: GPL-3.0-or-later
# (c) Adrian Knagg-Baugh 2025

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QLineEdit, QPushButton, QFileDialog, QMessageBox, QGroupBox, QComboBox, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

VERSION = "1.2.0"


class CosmicClaritySatelliteWorker(QThread):
    progress = pyqtSignal(float)
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, args, siril):
        super().__init__()
        self.args = args
        self.siril = siril

    def run(self):
        try:
            executable_path = self.args.executable
            exe_dir = os.path.dirname(executable_path)
            input_dir = os.path.join(exe_dir, "input")
            output_dir = os.path.join(exe_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Clear input if requested
            if self.args.clear_input and os.path.isdir(input_dir):
                for f in Path(input_dir).glob("*.*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

            # Get current Siril image
            pixels = self.siril.get_image_pixeldata()
            was_16bit = (pixels.dtype == np.uint16)
            if was_16bit:
                pixels = pixels.astype(np.float32) / 65535.0
            else:
                pixels = pixels.astype(np.float32, copy=False)

            # Channel/photometry handling
            if pixels.ndim == 2:
                photometry = "minisblack"
            elif pixels.ndim == 3 and pixels.shape[0] in (1, 3):
                photometry = "minisblack" if pixels.shape[0] == 1 else "rgb"
                pixels = pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)
            else:
                raise ValueError(f"Unexpected image shape: {pixels.shape}")

            basename = os.path.basename(self.siril.get_image_filename())
            input_file = os.path.join(input_dir, basename + ".tif")
            output_file = os.path.join(output_dir, f"{basename}_satellited.tif")

            # Write input
            tiffile.imwrite(input_file, pixels, photometric=photometry, planarconfig="contig")

            # Build command
            cmd = [
                executable_path,
                f"--input={input_dir}",
                f"--output={output_dir}",
                f"--mode={self.args.mode}",
                f"--sensitivity={self.args.sensitivity}",
            ]
            if self.args.use_gpu:
                cmd.append("--use-gpu")
            if self.args.monitor:
                cmd.append("--monitor")
            else:
                cmd.append("--batch")  # default if not monitoring
            if self.args.skip_save:
                cmd.append("--skip-save")
            if self.args.clip_trail:
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

                # Split on either \r or \n (or both)
                lines = re.split(r'[\r\n]+', buffer)

                for line in lines[:-1]:  # complete lines only
                    if not line.strip():
                        continue
                    m = percent_re.search(line)
                    if m:
                        try:
                            pct = float(m.group(1)) / 100.0
                            self.progress.emit(max(0.0, min(1.0, pct)))
                        except Exception:
                            pass
                    else:
                        print(line)
                        pass

                buffer = lines[-1]  # keep incomplete line

            ret = process.wait()
            self.progress.emit(1.0)
            if ret != 0 or not os.path.isfile(output_file):
                if args.skip_save:
                    self.finished_ok.emit(True, "no detection: skipped output")
                else:
                    self.finished_ok.emit(False, "")
                return

            # Read back result
            with tiffile.TiffFile(output_file) as tiff:
                pixel_data = tiff.asarray()
            pixel_data = np.ascontiguousarray(pixel_data)

            if pixel_data.ndim == 2:
                pixel_data = pixel_data[np.newaxis, :, :]
            elif pixel_data.ndim == 3 and pixel_data.shape[2] == 3:
                pixel_data = np.transpose(pixel_data, (2, 0, 1))
                pixel_data = np.ascontiguousarray(pixel_data)

            if was_16bit or self.siril.get_siril_config("core", "force_16bit"):
                pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)

            # Save undo + update image
            with self.siril.image_lock():
                self.siril.undo_save_state("Satellite removal")
                self.siril.set_image_pixeldata(pixel_data)

            self.finished_ok.emit(True, output_file)

        except Exception as e:
            print(f"Error in worker: {e}", file=sys.stderr)
            self.finished_ok.emit(False, "")


class SirilCosmicClaritySatellite(QMainWindow):
    def __init__(self, cli_args=None):
        super().__init__()
        self.cli_args = cli_args

        self.setWindowTitle(f"Cosmic Clarity Satellite Removal - v{VERSION}")
        self.setFixedSize(600, 500)

        # Connect to Siril
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            if cli_args:
                print("Failed to connect to Siril", file=sys.stderr)
            else:
                self.siril.error_messagebox("Failed to connect to Siril")
            self.close()
            return

        if not self.siril.is_image_loaded():
            if cli_args:
                print("No image loaded", file=sys.stderr)
            else:
                self.siril.error_messagebox("No image loaded")
            self.close()
            return

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close()
            return

        self.config_executable = self.check_config_file()

        if cli_args:
            self._apply_cli(cli_args)
            return

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
        args = argparse.Namespace(
            executable=self.exec_entry.text().strip(),
            use_gpu=self.gpu_checkbox.isChecked(),
            mode=self.mode_combo.currentText(),
            monitor=self.monitor_checkbox.isChecked(),
            skip_save=self.skip_checkbox.isChecked(),
            sensitivity=self.sensitivity,
            clip_trail=self.clip_checkbox.isChecked(),
            clear_input=self.clear_checkbox.isChecked(),
        )
        self._apply_cli(args)

    def _apply_cli(self, args):
        try:
            executable_path = args.executable or self.config_executable
            if not executable_path or not os.path.isfile(executable_path):
                print("Error: no valid executable specified.", file=sys.stderr)
                sys.exit(1)

            # Save path if changed
            if executable_path != self.config_executable:
                config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_satellite.conf")
                with open(config_file_path, "w") as f:
                    f.write(executable_path + "\n")

            args.executable = executable_path

            # Worker
            self._worker = CosmicClaritySatelliteWorker(args, self.siril)
            self._worker.progress.connect(lambda frac: self.siril.update_progress("Satellite Removal...", frac))
            self._worker.finished_ok.connect(self._on_finished)
            self._worker.start()

        except Exception as e:
            print(f"Error in CLI apply: {e}", file=sys.stderr)
            sys.exit(1)

    def _on_finished(self, success, outputfile):
        self.apply_btn.setEnabled(True)
        self.siril.reset_progress()
        if not success:
            QMessageBox.critical(self, "Cosmic Clarity", "Satellite removal failed.")
        else:
            self.siril.log(f"Satellite removal complete. Output: {outputfile}")

    def check_config_file(self):
        config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_satellite.conf")
        if os.path.isfile(config_file_path):
            with open(config_file_path, "r") as f:
                exe = f.readline().strip()
                if os.path.isfile(exe) and os.access(exe, os.X_OK):
                    return exe
        return None


def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity Satellite Removal Script")
    parser.add_argument("-executable", type=str, default="", help="Path to executable")
    parser.add_argument("-use_gpu", dest="use_gpu", action="store_true", default=True)
    parser.add_argument("-no_gpu", dest="use_gpu", action="store_false")
    parser.add_argument("-mode", choices=["full", "luminance"], default="full")
    parser.add_argument("-monitor", action="store_true", default=False, help="Monitor mode (else batch)")
    parser.add_argument("-skip_save", action="store_true", default=False)
    parser.add_argument("-sensitivity", type=float, default=0.1)
    parser.add_argument("-clip_trail", dest="clip_trail", action="store_true", default=True)
    parser.add_argument("-no_clip_trail", dest="clip_trail", action="store_false")
    parser.add_argument("-clear_input", action="store_true", default=False)

    args = parser.parse_args()

    cli_mode = any([
        args.executable,
        not args.use_gpu,
        args.mode != "full",
        args.monitor,
        args.skip_save,
        args.sensitivity != 0.1,
        not args.clip_trail,
        args.clear_input
    ])

    try:
        if cli_mode:
            SirilCosmicClaritySatellite(cli_args=args)
        else:
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            win = SirilCosmicClaritySatellite()
            win.show()
            sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing application: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
