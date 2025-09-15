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

# Version: 2.0.0 (PyQt6 threaded worker port by AKB)

import sirilpy as s
s.ensure_installed("PyQt6")

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QSlider, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

VERSION = "2.0.0"

class CosmicClaritySuperresWorker(QThread):
    progress = pyqtSignal(float)
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, siril, executable_path, scale):
        super().__init__()
        self.siril = siril  # Already connected instance
        self.executable_path = executable_path
        self.scale = scale

    def run(self):
        try:
            model_dir = os.path.dirname(self.executable_path)
            input_file = self.siril.get_image_filename()
            output_dir = self.siril.get_siril_wd()
            output_filename = f"{os.path.splitext(input_file)[0]}_upscaled{self.scale}x.fit"

            command = [
                self.executable_path,
                f"--input={input_file}",
                f"--output_dir={output_dir}",
                f"--scale={self.scale}",
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
                            self.progress.emit(max(0.0, min(1.0, pct)))
                        except Exception:
                            pass
                    else:
                        print(line)
                        pass

                buffer = lines[-1]  # keep incomplete line

            ret = process.wait()
            self.progress.emit(1.0)
            self.finished_ok.emit(ret == 0, output_filename)

        except Exception as e:
            print(f"Error in worker: {e}")
            self.finished_ok.emit(False, "")

class SirilCosmicClarityInterface(QMainWindow):
    def __init__(self, cli_args=None):
        super().__init__()
        self.cli_args = cli_args

        self.setWindowTitle(f"Cosmic Clarity Superres - v{VERSION}")
        self.setFixedSize(500, 400)

        # Initialize Siril connection
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
            with self.siril.image_lock():
                executable_path = self.exec_entry.text().strip()
                scale = self.scale
                load_upscaled = self.load_upscaled

                if executable_path != self.config_executable:
                    config_file_path = os.path.join(
                        self.siril.get_siril_configdir(), "sirilcc_superres.conf"
                    )
                    with open(config_file_path, "w") as file:
                        file.write(f"{executable_path}\n")

                self.siril.update_progress("Seti Astro Cosmic Clarity Superres starting...", 0)

                # Disable Apply button until finished
                self.apply_btn.setEnabled(False)

                self._worker = CosmicClaritySuperresWorker(self.siril, executable_path, scale)
                self._worker.progress.connect(self._on_progress)
                self._worker.finished_ok.connect(
                    lambda success, fn: self._on_finished(success, fn, load_upscaled)
                )
                self._worker.start()

        except Exception as e:
            print(f"Error in apply: {e}")
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

            if not success:
                self.siril.reset_progress()
                QMessageBox.critical(self, "Cosmic Clarity", "Superres failed.")
                return

            if load_upscaled and os.path.exists(outputfilename):
                self.siril.log(f"Ready to open upscaled image: {outputfilename}")
                self.siril.cmd(f"load \"{outputfilename}\"")

            self.siril.reset_progress()
            self.siril.log("Cosmic Clarity superres complete.")

        except Exception as e:
            print(f"Error in finished handler: {e}")
            self.siril.reset_progress()

    def check_config_file(self):
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, "sirilcc_superres.conf")
        if os.path.isfile(config_file_path):
            with open(config_file_path, "r") as file:
                executable_path = file.readline().strip()
                if os.path.isfile(executable_path) and os.access(executable_path, os.X_OK):
                    return executable_path
        QMessageBox.information(
            self,
            "Configuration",
            "Executable not yet configured. Recommended to use Seti Astro Cosmic Clarity Super-Resolution Upscaling Tool v1.1 or higher.",
        )
        return None

    def _apply_cli(self, args):
        """Headless run via CLI args."""
        try:
            scale = args.scale
            load_upscaled = args.load_upscaled
            executable_path = args.executable or self.config_executable

            if not executable_path or not os.path.isfile(executable_path):
                print("Error: please specify a valid executable with -executable or configure one via GUI.", file=sys.stderr)
                sys.exit(1)

            if executable_path != self.config_executable:
                config_file_path = os.path.join(
                    self.siril.get_siril_configdir(), "sirilcc_superres.conf"
                )
                with open(config_file_path, "w") as f:
                    f.write(executable_path + "\n")

            input_file = self.siril.get_image_filename()
            output_dir = self.siril.get_siril_wd()
            output_filename = f"{os.path.splitext(input_file)[0]}_upscaled{scale}x.fit"

            cmd = [
                executable_path,
                f"--input={input_file}",
                f"--output_dir={output_dir}",
                f"--scale={scale}",
                f"--model_dir={os.path.dirname(executable_path)}"
            ]

            ret = subprocess.call(cmd)
            if ret != 0 or not os.path.exists(output_filename):
                print("Error: superres process failed.", file=sys.stderr)
                sys.exit(1)

            if load_upscaled and os.path.exists(output_filename):
                self.siril.cmd(f"load \"{output_filename}\"")

            print("Cosmic Clarity superres completed successfully.")

        except Exception as e:
            print(f"Error in CLI apply: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity Superres Script")
    parser.add_argument("-scale", type=int, choices=[2, 3, 4], default=2,
                        help="Upscale factor (2x, 3x, or 4x)")
    parser.add_argument("-load_upscaled", action="store_true", default=False,
                        help="Load upscaled image into Siril after completion")
    parser.add_argument("-executable", type=str, default="",
                        help="Path to Cosmic Clarity executable")

    args = parser.parse_args()

    cli_mode = any([
        args.scale != 2,
        args.load_upscaled,
        args.executable != ""
    ])

    try:
        if cli_mode:
            SirilCosmicClarityInterface(cli_args=args)
        else:
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            win = SirilCosmicClarityInterface()
            win.show()
            sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing application: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
