# Feb. 18, 2025
# (c) Rich Stevenson - Deep Space Astro
# SPDX-License-Identifier: GPL-3.0-or-later
# Script for reducing stars using pixel math.
#
# Version 2.0.0

import sirilpy as s
s.ensure_installed("PyQt6")

import os
import sys
from sirilpy import LogColor

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
    QHBoxLayout, QSlider, QCheckBox, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt

VERSION = "2.0.0"
# 2.0.0 CR: Using PyQt6 instead of tkinter

class StarReductionInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.overwrite_checkbox_state = True
        self.resolution = 0.01

        self.setWindowTitle(f"DSA-Star Reduction - v{VERSION}")
        self.setFixedSize(350, 270)

        # Connect to Siril
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            self.siril.log(f"Connection failed: {e}", color=LogColor.RED)
            sys.exit()

        self.siril.log("Connected successfully!", color=LogColor.GREEN)

        if not self.initial_checks():
            self.close()
            sys.exit()

        self.setup_image_info()
        self.initUI()

    def initial_checks(self):
        require_version = "1.3.6"
        try:
            self.siril.cmd("requires", require_version)
        except:
            self.siril.error_messagebox(
                f"This script requires Siril version {require_version} or later!"
            )
            return False

        starnet_path = self.siril.get_siril_config("core", "starnet_exe")
        if not starnet_path or not os.path.isfile(starnet_path) or not os.access(starnet_path, os.X_OK):
            self.siril.error_messagebox("Starnet Command Line Tool was not found or is not configured!")
            return False

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("Open a FITS image before running Star Reduction!")
            return False

        path = self.siril.get_image_filename()
        basename = os.path.basename(path)
        get_extension = os.path.splitext(basename)[1]
        if get_extension not in (".fit", ".fits"):
            self.siril.error_messagebox(
                f"The image that is open is a {get_extension} and is not supported.\nPlease open a FITS file."
            )
            return False

        if "_ReducedStars" in basename:
            self.siril.error_messagebox("This image has already had star reduction applied.")
            return False

        return True

    def setup_image_info(self):
        path = self.siril.get_image_filename()
        self.img_name = os.path.basename(path)
        self.img_dir = os.path.dirname(os.path.abspath(path))
        self.siril.cmd("cd", f'"{self.img_dir}"')
        os.chdir(self.img_dir)
        self.img_name_pm = f"${self.img_name}$"
        self.get_extension = os.path.splitext(self.img_name)[1]
        self.file_name_without_ext = os.path.splitext(os.path.basename(path))[0]

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("Star Reduction")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Reminder
        reminder_label = QLabel("Image must already be stretched!")
        reminder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(reminder_label)

        # Parameters group
        param_group = QGroupBox("Parameters")
        param_layout = QHBoxLayout()
        param_group.setLayout(param_layout)

        param_layout.addWidget(QLabel("Value: "))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(0.99 / self.resolution))
        self.slider.setValue(int(0.2 / self.resolution))
        self.slider.valueChanged.connect(self.update_slider_value)
        param_layout.addWidget(self.slider)

        self.slider_value_label = QLabel("0.20")
        param_layout.addWidget(self.slider_value_label)

        layout.addWidget(param_group)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        options_group.setLayout(options_layout)

        self.overwrite_checkbox = QCheckBox("Overwrite Output File")
        self.overwrite_checkbox.setChecked(True)
        options_layout.addWidget(self.overwrite_checkbox)

        layout.addWidget(options_group)

        # Buttons
        button_layout = QHBoxLayout()
        help_button = QPushButton("Help")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.run_close)
        button_layout.addWidget(close_button)

        submit_button = QPushButton("Apply")
        submit_button.clicked.connect(self.run_reduction)
        button_layout.addWidget(submit_button)

        layout.addLayout(button_layout)

    def update_slider_value(self):
        rounded_value = round(self.slider.value() * self.resolution, 2)
        self.slider_value_label.setText(f"{rounded_value:.2f}")

    def show_help(self):
        self.siril.info_messagebox(
            "Reduces the size of the stars in an image based on the value selected.\n"
            "Lower values reduce more, values >0.5 increase size.\n"
            "Creates {image_name}_ReducedStars.fit in the same directory.\n\n"
            "- StarNet must be installed and configured.\n"
            "- Image must be in FITS format and stretched.\n"
            "- Uncheck 'Overwrite' to create new files for each run."
        )

    def run_reduction(self):
        try:
            filename_overwrite = self.overwrite_checkbox.isChecked()
            star_reduction_value = round(self.slider.value() * self.resolution, 2)

            default_ext = self.siril.get_siril_config("core", "extension")
            img_name_default_ext = f"{self.file_name_without_ext}{default_ext}"

            if os.path.exists(f"starless_{img_name_default_ext}"):
                self.siril.log("Previous star reduction detected.", color=LogColor.GREEN)
                starless = f"$starless_{img_name_default_ext}$"
            else:
                self.siril.cmd("starnet", "-nostarmask")
                path = self.siril.get_image_filename()
                starless = f"${os.path.basename(path)}$"

            self.siril.cmd(
                "pm", 
                f"\"~((~mtf(~{star_reduction_value},{self.img_name_pm})/~mtf(~{star_reduction_value},{starless}))*~{starless})\""
            )

            current_img = self.img_name.replace(self.get_extension, "")

            if filename_overwrite:
                self.siril.cmd("save", f"\"{current_img}_ReducedStars{default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars{default_ext}\"")
            else:
                self.siril.cmd("save", f"\"{current_img}_ReducedStars_{star_reduction_value}{default_ext}\"")
                self.siril.cmd("load", f"\"{current_img}_ReducedStars_{star_reduction_value}{default_ext}\"")

            self.siril.log("Star Reduction complete!", color=LogColor.GREEN)
            return True
        except Exception as e:
            self.siril.log(f"Error in run_reduction: {str(e)}", color=LogColor.RED)
            return False

    def run_close(self):
        self.close()


def main():
    app = QApplication(sys.argv)
    window = StarReductionInterface()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
