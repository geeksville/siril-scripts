# Cosmic Clarity - Darkstar wrapper
# SPDX-License-Identifier: GPL-3.0-or-later
# (c) Adrian Knagg-Baugh 2025

# Contact: report issues with this script at https://gitlab.com/free-astro/siril-scripts

# *** Please do not submit bug reports about the Cosmic Clarity interface scripts to SetiAstro. ***
# He has not been involved in writing the scripts and all the scripts do is automate the execution
# of the CosmicClarity programme. If you think you have found a bug in Cosmic Clarity itself you
# *MUST* reproduce the bug either standalone from the commandline or using SetiAstroSuite Pro before
# reporting it.

# Version 1.0.0

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
    QCheckBox, QLineEdit, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QComboBox, QSlider, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

VERSION = "1.1.0"


class CosmicClarityDarkstarWorker(QThread):
    # finished_ok: success (bool), starless_path (str), stars_only_path (str),
    # original_array (object), starless_array (object or None), stars_array (object or None), was_16bit (bool)
    progress = pyqtSignal(float)
    finished_ok = pyqtSignal(bool, str, str, object, object, object, bool)

    def __init__(self, args, siril):
        super().__init__()
        self.args = args
        self.siril = siril

    def run(self):
        try:
            result = self._process(self.args, self.siril)
            if result is None:
                self.finished_ok.emit(False, "", "", None, None, None, False)
            else:
                (starless_file, stars_file, original_f, starless_f, stars_f, was_16bit) = result
                self.finished_ok.emit(True, starless_file or "", stars_file or "", original_f, starless_f, stars_f, was_16bit)
        except Exception as e:
            print(f"Error in worker: {e}", file=sys.stderr)
            self.finished_ok.emit(False, "", "", None, None, None, False)

    def _process(self, args, siril):
        """Performs the same job as CLI synchronous path but returns arrays and paths."""
        executable_path = args.executable
        exe_dir = os.path.dirname(executable_path)
        input_dir = os.path.join(exe_dir, "input")
        output_dir = os.path.join(exe_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Clear input if requested
        if getattr(args, "clear_input", False) and os.path.isdir(input_dir):
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

        # Channel-first -> for writing tiff we want (H, W) or (H, W, C)
        if pixels_float.ndim == 2:
            photometry = "minisblack"
            write_arr = pixels_float
        elif pixels_float.ndim == 3 and pixels_float.shape[0] in (1, 3):
            photometry = "minisblack" if pixels_float.shape[0] == 1 else "rgb"
            write_arr = pixels_float[0] if pixels_float.shape[0] == 1 else np.transpose(pixels_float, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected image shape: {pixels_float.shape}")

        # Avoid double extensions: strip original extension
        orig_filename = os.path.basename(siril.get_image_filename())
        base_noext = os.path.splitext(orig_filename)[0]

        input_file = os.path.join(input_dir, f"{base_noext}.tif")
        starless_file = os.path.join(output_dir, f"{base_noext}_starless.tif")
        stars_only_file = os.path.join(output_dir, f"{base_noext}_stars_only.tif")

        # Write input TIFF
        tiffile.imwrite(input_file, write_arr, photometric=photometry, planarconfig="contig")

        # Build command (tool doesn't show input/output flags; tool operates on exe_dir/input/ output/)
        cmd = [executable_path]
        if getattr(args, "disable_gpu", False):
            cmd.append("--disable_gpu")
        if getattr(args, "star_removal_mode", None):
            cmd.extend(["--star_removal_mode", str(args.star_removal_mode)])
        if getattr(args, "show_extracted_stars", False):
            cmd.append("--show_extracted_stars")
        if getattr(args, "chunk_size", None):
            cmd.extend(["--chunk_size", str(int(args.chunk_size))])
        if getattr(args, "overlap", None) is not None:
            cmd.extend(["--overlap", str(int(args.overlap))])

        # Launch process from exe_dir
        process = subprocess.Popen(
            cmd,
            cwd=exe_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        percent_re = re.compile(r"(\d+\.?\d*)%")

        for line in process.stdout:
            if not line:
                continue
            m = percent_re.search(line)
            if m:
                try:
                    pct = float(m.group(1)) / 100.0
                    self.progress.emit(max(0.0, min(1.0, pct)))
                except Exception:
                    pass
            else:
                tmp = line.strip()
                if tmp != "":
                    print(tmp)

        ret = process.wait()
        self.progress.emit(1.0)
        if ret != 0:
            return None

        # Read outputs (if they exist)
        got_starless = os.path.isfile(starless_file)
        got_stars_only = os.path.isfile(stars_only_file)

        # Convert outputs to channel-first float32 arrays (0..1)
        starless_arr = None
        stars_arr = None

        if got_starless:
            with tiffile.TiffFile(starless_file) as t:
                data = t.asarray()
            data = np.ascontiguousarray(data)
            if data.ndim == 2:
                cf = data[np.newaxis, :, :].astype(np.float32, copy=False)
            elif data.ndim == 3 and data.shape[2] == 3:
                cf = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unexpected starless shape: {data.shape}")
            # If dtype is integer, normalize
            if np.issubdtype(data.dtype, np.integer):
                # assume 16-bit
                starless_arr = cf.astype(np.float32) / 65535.0
            else:
                starless_arr = cf.astype(np.float32, copy=False)

        if got_stars_only:
            with tiffile.TiffFile(stars_only_file) as t:
                data = t.asarray()
            data = np.ascontiguousarray(data)
            if data.ndim == 2:
                cf = data[np.newaxis, :, :].astype(np.float32, copy=False)
            elif data.ndim == 3 and data.shape[2] == 3:
                cf = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unexpected stars-only shape: {data.shape}")
            if np.issubdtype(data.dtype, np.integer):
                stars_arr = cf.astype(np.float32) / 65535.0
            else:
                stars_arr = cf.astype(np.float32, copy=False)

        # original float array (channel-first) for caching
        if pixels_float.ndim == 2:
            original_cf = pixels_float[np.newaxis, :, :].astype(np.float32, copy=False)
        elif pixels_float.ndim == 3 and pixels_float.shape[0] in (1, 3):
            original_cf = pixels_float if pixels_float.shape[0] in (1, 3) else None
            # ensure channel-first
            if original_cf is not None and original_cf.shape[0] not in (1, 3):
                original_cf = np.transpose(original_cf, (2, 0, 1))
        else:
            original_cf = pixels_float.astype(np.float32, copy=False)

        return starless_file if got_starless else "", stars_only_file if got_stars_only else "", original_cf, starless_arr, stars_arr, was_16bit


class SirilCosmicClarityDarkstar(QMainWindow):
    def __init__(self, cli_args=None):
        super().__init__()
        self.cli_args = cli_args

        self.setWindowTitle(f"Cosmic Clarity - Darkstar - v{VERSION}")
        self.setFixedSize(600, 480)

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
            # run synchronous CLI path and exit
            code = self._apply_cli_sync(cli_args)
            # _apply_cli_sync returns exit code; if non-zero, exit with it
            if code != 0:
                sys.exit(code)
            return

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

        # GPU checkbox (program uses --disable_gpu)
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

        # Result choice group (Original / Starless / Stars-only)
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
        # initially disabled until processing occurs
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

        # overlap slider as percent of chunk size
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap (% of chunk):"))
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlap_slider.setRange(0, 50)  # 0% .. 50%
        default_percent = int(12.5)  # ~12.5%
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

        # Hook toggles to replace image from cache when clicked
        self.rb_original.toggled.connect(lambda checked: checked and self._set_image_from_cache("original"))
        self.rb_starless.toggled.connect(lambda checked: checked and self._set_image_from_cache("starless"))
        self.rb_stars.toggled.connect(lambda checked: checked and self._set_image_from_cache("stars"))

        layout.addLayout(btn_layout)
        self.setCentralWidget(central)

    def _on_chunk_changed(self, val):
        self.chunk_size = int(val)
        self.chunk_label.setText(str(self.chunk_size))
        # update overlap percent -> pixel conversion
        percent = self.overlap_slider.value()
        self.overlap_pixels = int((percent / 100.0) * self.chunk_size)
        self.overlap_label.setText(f"{percent}%")

    def _on_overlap_percent_changed(self, val):
        self.overlap_label.setText(f"{val}%")
        self.overlap_pixels = int((val / 100.0) * self.chunk_size)

    def _toggle_stars_switch_visibility(self, checked):
        # only enable the radio group after processing
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
        # Prepare args
        args = argparse.Namespace(
            executable=self.exec_entry.text().strip(),
            disable_gpu=not self.gpu_checkbox.isChecked(),
            star_removal_mode=self.mode_combo.currentText(),
            show_extracted_stars=self.show_stars_checkbox.isChecked(),
            chunk_size=self.chunk_slider.value(),
            overlap=int((self.overlap_slider.value() / 100.0) * self.chunk_slider.value()),
            clear_input=self.clear_checkbox.isChecked(),
        )

        executable_path = args.executable or self.config_executable
        if not executable_path or not os.path.isfile(executable_path):
            QMessageBox.critical(self, "Executable", "Please select a valid Cosmic Clarity Darkstar executable.")
            return

        # Save config if changed
        if executable_path != self.config_executable:
            config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_darkstar.conf")
            try:
                with open(config_file_path, "w") as f:
                    f.write(executable_path + "\n")
            except Exception as e:
                print(f"Failed writing config: {e}", file=sys.stderr)

        args.executable = executable_path

        # disable apply while running
        self.apply_btn.setEnabled(False)

        # Start worker
        self._worker = CosmicClarityDarkstarWorker(args, self.siril)
        self._worker.progress.connect(lambda frac: self.siril.update_progress("Darkstar processing...", float(frac)))
        self._worker.finished_ok.connect(self._on_finished_worker)
        self._worker.start()

    def _on_finished_worker(self, success, starless_path, stars_only_path, original_arr, starless_arr, stars_arr, was_16bit):
        # Re-enable apply
        self.apply_btn.setEnabled(True)
        try:
            self.siril.reset_progress()
        except Exception:
            pass

        if not success:
            QMessageBox.critical(self, "Cosmic Clarity", "Darkstar failed.")
            return

        # Cache arrays (channel-first float32)
        self.cached["original"] = original_arr
        self.cached["starless"] = starless_arr
        self.cached["stars"] = stars_arr
        self.cached["was_16bit"] = was_16bit

        # Enable radio group for toggling
        self.result_choice_group.setEnabled(True)
        # Default selection: starless if exists else original
        if starless_arr is not None:
            self.rb_starless.setChecked(True)
            self._set_image_from_cache("starless")
        else:
            self.rb_original.setChecked(True)
            self._set_image_from_cache("original")

        self.siril.log("Cosmic Clarity - Darkstar completed.")

    def check_config_file(self):
        config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_darkstar.conf")
        if os.path.isfile(config_file_path):
            with open(config_file_path, "r") as f:
                exe = f.readline().strip()
                if os.path.isfile(exe) and os.access(exe, os.X_OK):
                    return exe
        return None

    #
    # CLI synchronous path (blocking). Returns 0 on success, nonzero on error.
    #
    def _apply_cli_sync(self, args):
        executable_path = args.executable or self.config_executable
        if not executable_path or not os.path.isfile(executable_path):
            print("Error: no valid executable specified.", file=sys.stderr)
            return 2

        # persist executable if changed
        if executable_path != self.config_executable:
            config_file_path = os.path.join(self.siril.get_siril_configdir(), "sirilcc_darkstar.conf")
            try:
                with open(config_file_path, "w") as f:
                    f.write(executable_path + "\n")
            except Exception:
                pass

        # compute overlap default if not supplied
        chunk_size = getattr(args, "chunk_size", 512) or 512
        overlap = args.overlap if getattr(args, "overlap", None) is not None else int(0.125 * chunk_size)

        proc_args = argparse.Namespace(
            executable=executable_path,
            disable_gpu=getattr(args, "disable_gpu", False),
            star_removal_mode=getattr(args, "star_removal_mode", "additive"),
            show_extracted_stars=getattr(args, "show_extracted_stars", False),
            chunk_size=chunk_size,
            overlap=overlap,
            clear_input=getattr(args, "clear_input", False),
        )

        try:
            worker = CosmicClarityDarkstarWorker(proc_args, self.siril)
            # Use _process synchronously
            res = worker._process(proc_args, self.siril)
            if res is None:
                print("Darkstar processing failed.", file=sys.stderr)
                return 3
            starless_file, stars_file, original_arr, starless_arr, stars_arr, was_16bit = res

            # Choose which to set: if show_extracted_stars true and stars_arr exists, set stars, else set starless
            if proc_args.show_extracted_stars and stars_arr is not None:
                arr = stars_arr
                label = "Darkstar: stars-only result"
            else:
                arr = starless_arr if starless_arr is not None else original_arr
                label = "Darkstar: starless result" if starless_arr is not None else "Darkstar: original"

            if arr is None:
                print("No output image found to set.", file=sys.stderr)
                return 4

            # Convert to dtype appropriate for Siril
            force_16 = self.siril.get_siril_config("core", "force_16bit")
            out = arr
            if was_16bit or force_16:
                out = np.rint(out * 65535.0).astype(np.uint16)
            out = np.ascontiguousarray(out)
            with self.siril.image_lock():
                self.siril.undo_save_state(label)
                self.siril.set_image_pixeldata(out)

            print("Darkstar processing completed successfully.")
            return 0
        except Exception as e:
            print(f"Error during Darkstar processing: {e}", file=sys.stderr)
            return 5


def main():
    parser = argparse.ArgumentParser(description="Cosmic Clarity - Darkstar wrapper")
    parser.add_argument("-executable", type=str, default="", help="Path to setiastrocosmicclarity_darkstar executable")
    parser.add_argument("--disable_gpu", action="store_true", default=False, help="Disable GPU (use CPU only)")
    parser.add_argument("--star_removal_mode", choices=["additive", "unscreen"], default="additive")
    parser.add_argument("--show_extracted_stars", action="store_true", default=False)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=None, help="Overlap in pixels (default=0.125*chunk_size)")
    parser.add_argument("--clear_input", action="store_true", default=False, help="Clear input directory before run")

    args = parser.parse_args()

    # Decide CLI mode if any non-default arg provided
    cli_mode = any([
        args.executable,
        args.disable_gpu,
        args.star_removal_mode != "additive",
        args.show_extracted_stars,
        args.chunk_size != 512,
        args.overlap is not None,
        args.clear_input
    ])

    try:
        if cli_mode:
            svc = SirilCosmicClarityDarkstar(cli_args=args)
            # constructor runs the synchronous CLI and returns (it will sys.exit if necessary)
            return
        else:
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            win = SirilCosmicClarityDarkstar()
            win.show()
            sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing Darkstar wrapper: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
