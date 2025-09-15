# (c) Adrian Knagg-Baugh 2024-2025
# SPDX-License-Identifier: GPL-3.0-or-later

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version: 2.1.0
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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QButtonGroup, QCheckBox, QSlider, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

VERSION = "2.1.0"


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
            command = [
                self.executable_path,
                f"--sharpening_mode={self.mode}",
                f"--stellar_amount={self.stellar_amount}",
                f"--nonstellar_strength={self.non_stellar_strength}",
                f"--nonstellar_amount={self.non_stellar_amount}",
            ]
            if not self.use_gpu:
                command.append("--disable_gpu")
            if self.auto_psf:
                command.append("--auto_detect_psf")
            if self.separate_channels:
                command.append("--sharpen_channels_separately")

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
            self.finished_ok.emit(ret == 0, self.outputfilename)
        except Exception:
            self.finished_ok.emit(False, "")


class SirilCosmicClarityInterface(QMainWindow):
    def __init__(self, cli_args=None):
        super().__init__()
        self.cli_args = cli_args

        self.setWindowTitle(f"Cosmic Clarity Sharpening - v{VERSION}")
        self.setFixedSize(500, 600)

        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            if not cli_args:
                self.siril.error_messagebox("Failed to connect to Siril")
            else:
                print("Failed to connect to Siril", file=sys.stderr)
            self.close()
            return

        if not self.siril.is_image_loaded():
            if not cli_args:
                self.siril.error_messagebox("No image loaded")
            else:
                print("No image loaded", file=sys.stderr)
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
            return  # don't build GUI

        # Widget state vars...
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

        # store apply button as an instance attribute so we can disable/enable it
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

            if executable_path != self.config_executable:
                config_file_path = os.path.join(
                    self.siril.get_siril_configdir(), "sirilcc_sharpen.conf"
                )
                with open(config_file_path, "w") as file:
                    file.write(f"{executable_path}\n")

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

            if pixels.ndim == 2:
                photometry = "minisblack"
            elif pixels.ndim == 3 and pixels.shape[0] in (1, 3):
                photometry = "minisblack" if pixels.shape[0] == 1 else "rgb"
                pixels = pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)
            else:
                raise ValueError(f"Unexpected image shape: {pixels.shape}")

            tiffile.imwrite(inputfilename, pixels, photometric=photometry, planarconfig="contig")

            self.siril.update_progress("Seti Astro Cosmic Clarity Sharpen starting...", 0)

            # Disable Apply button until the worker finishes to avoid overlapping Siril calls
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
            # Ensure Apply button is re-enabled if an error occurs before worker starts
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
        # Re-enable Apply button immediately when worker finishes
        try:
            self.apply_btn.setEnabled(True)
        except Exception:
            pass

        try:
            if not success:
                self.siril.reset_progress()
                QMessageBox.critical(self, "Cosmic Clarity", "Sharpening failed.")
                return

            with tiffile.TiffFile(outputfilename) as tiff:
                pixel_data = tiff.asarray()
            pixel_data = np.ascontiguousarray(pixel_data)
            if pixel_data.ndim == 2:
                pixel_data = pixel_data[np.newaxis, :, :]
            elif pixel_data.ndim == 3 and pixel_data.shape[2] == 3:
                pixel_data = np.transpose(pixel_data, (2, 0, 1))
                pixel_data = np.ascontiguousarray(pixel_data)
            force_16bit = self.siril.get_siril_config("core", "force_16bit")
            if was_16bit or force_16bit:
                pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)
            with self.siril.image_lock():
                self.siril.undo_save_state(f"Cosmic Clarity sharpen ({mode})")
                self.siril.set_image_pixeldata(pixel_data)
            self.siril.reset_progress()
            self.siril.log("Cosmic Clarity sharpening complete.")
        except Exception as e:
            print(f"Error in finished handler: {e}")
            self.siril.reset_progress()

    def check_config_file(self):
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, "sirilcc_sharpen.conf")
        if os.path.isfile(config_file_path):
            with open(config_file_path, "r") as file:
                executable_path = file.readline().strip()
                if os.path.isfile(executable_path) and os.access(executable_path, os.X_OK):
                    return executable_path
        QMessageBox.information(
            self,
            "Configuration",
            "Executable not yet configured. It is recommended to use Seti Astro Cosmic Clarity v6.5 or higher.",
        )
        return None

    def _apply_cli(self, args):
        """Headless apply using CLI arguments."""
        try:
            mode = args.sharpening_mode
            stellar_amount = args.stellar_amount
            non_stellar_strength = args.non_stellar_strength
            non_stellar_amount = args.non_stellar_amount
            use_gpu = args.use_gpu
            auto_psf = args.auto_psf
            separate_channels = args.separate_channels
            clear_input = args.clear_input_dir
            executable_path = args.executable or self.config_executable

            if not executable_path or not os.path.isfile(executable_path):
                print("Error: no valid executable specified or configured.", file=sys.stderr)
                sys.exit(1)

            # Save executable if changed
            if executable_path != self.config_executable:
                config_file_path = os.path.join(
                    self.siril.get_siril_configdir(), "sirilcc_sharpen.conf"
                )
                with open(config_file_path, "w") as f:
                    f.write(executable_path + "\n")

            filename = self.siril.get_image_filename()
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

            pixels = self.siril.get_image_pixeldata()
            was_16bit = False
            if pixels.dtype == np.uint16:
                pixels = pixels.astype(np.float32) / 65535.0
                was_16bit = True

            if pixels.ndim == 2:
                photometry = "minisblack"
            elif pixels.ndim == 3 and pixels.shape[0] in (1, 3):
                photometry = "minisblack" if pixels.shape[0] == 1 else "rgb"
                pixels = pixels[0] if pixels.shape[0] == 1 else pixels.transpose(1, 2, 0)
            else:
                raise ValueError(f"Unexpected image shape: {pixels.shape}")

            tiffile.imwrite(inputfilename, pixels, photometric=photometry, planarconfig="contig")

            # Build command
            cmd = [
                executable_path,
                f"--sharpening_mode={mode}",
                f"--stellar_amount={stellar_amount}",
                f"--nonstellar_strength={non_stellar_strength}",
                f"--nonstellar_amount={non_stellar_amount}",
            ]
            if not use_gpu:
                cmd.append("--disable_gpu")
            if auto_psf:
                cmd.append("--auto_detect_psf")
            if separate_channels:
                cmd.append("--sharpen_channels_separately")

            ret = subprocess.call(cmd)
            if ret != 0 or not os.path.isfile(outputfilename):
                print("Error: sharpening process failed.", file=sys.stderr)
                sys.exit(1)

            with tiffile.TiffFile(outputfilename) as tiff:
                pixel_data = tiff.asarray()
            pixel_data = np.ascontiguousarray(pixel_data)
            if pixel_data.ndim == 2:
                pixel_data = pixel_data[np.newaxis, :, :]
            elif pixel_data.ndim == 3 and pixel_data.shape[2] == 3:
                pixel_data = np.transpose(pixel_data, (2, 0, 1))
                pixel_data = np.ascontiguousarray(pixel_data)
            if was_16bit or self.siril.get_siril_config("core", "force_16bit"):
                pixel_data = np.rint(pixel_data * 65535).astype(np.uint16)

            with self.siril.image_lock():
                self.siril.undo_save_state(f"Cosmic Clarity sharpen ({mode})")
                self.siril.set_image_pixeldata(pixel_data)

            print("Cosmic Clarity sharpening completed successfully.")
        except Exception as e:
            print(f"Error in CLI apply: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity Sharpening Script")
    parser.add_argument("-sharpening_mode", type=str, choices=["Stellar Only", "Non-Stellar Only", "Both"],
                        default="Stellar Only")
    parser.add_argument("-stellar_amount", type=float, default=0.5)
    parser.add_argument("-non_stellar_strength", type=int, default=3)
    parser.add_argument("-non_stellar_amount", type=float, default=0.5)
    parser.add_argument("-use_gpu", action="store_true", default=True)
    parser.add_argument("-no_gpu", dest="use_gpu", action="store_false")
    parser.add_argument("-auto_psf", action="store_true", default=True)
    parser.add_argument("-no_auto_psf", dest="auto_psf", action="store_false")
    parser.add_argument("-separate_channels", action="store_true", default=False)
    parser.add_argument("-executable", type=str, default="")
    parser.add_argument("-clear_input_dir", action="store_true", default=True)
    parser.add_argument("-no_clear_input", dest="clear_input_dir", action="store_false")

    args = parser.parse_args()

    cli_mode = any([
        args.sharpening_mode != "Stellar Only",
        args.stellar_amount != 0.5,
        args.non_stellar_strength != 3,
        args.non_stellar_amount != 0.5,
        not args.use_gpu,
        not args.auto_psf,
        args.separate_channels,
        args.executable != "",
        not args.clear_input_dir,
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
