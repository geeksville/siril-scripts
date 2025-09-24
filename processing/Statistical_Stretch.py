# (c) Cyril Richard 2024
# Code From Seti Astro Statistical Stretch - PyQt Version
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 2.0.1

import sirilpy as s
s.ensure_installed('PyQt6')

import sys
import argparse
import numpy as np
import math

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton,
                            QGroupBox, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

VERSION = "2.0.1"
REQUIRES_SIRILPY = "0.6.10"
# 1.0.1 AKB: convert "requires" to exception handling
# 1.0.2 CR: round down slider values
# 1.0.3 CR: fix division by zero when target_median is 0 or 1.0
# 1.0.4 CM: Better cli/GUI handling
# 2.0.0 CR: Using PyQt6 instead of tkinter
# 2.0.1 AKB: Separate GUI and CLI processing to prevent failure with pyscript

if not s.check_module_version(f'>={REQUIRES_SIRILPY}'):
    print(f"Please install sirilpy version {REQUIRES_SIRILPY} or higher")
    sys.exit(1)

class StatisticalStretchProcessor:
    """Core processing logic separated from GUI"""
    def __init__(self, siril: s.SirilInterface):
        self.siril = siril

    def floor_value(self, value, decimals=2):
        """Floor a value to the specified number of decimal places"""
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def stretch_mono_image(self, fit, target_median, normalize=False, apply_curves=False, curves_boost=0.0):
        # Ensure target_median is between 0.01 and 0.99
        target_median = max(0.01, min(0.99, target_median))

        # Calculate black point
        black_point = max(np.min(fit.data), np.median(fit.data) - 2.7 * np.std(fit.data))

        # Rescale image
        rescaled_image = (fit.data - black_point) / (1 - black_point)
        median_image = np.median(rescaled_image)

        # Stretch image
        stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)

        # Optional normalization
        if normalize:
            stretched_image = stretched_image / np.max(stretched_image)

        # Optional curve boost
        if apply_curves:
            stretched_image = np.clip(stretched_image, 0, None) # Make sure no negative pixels are used
            stretched_image = np.power(stretched_image, 1.0 + curves_boost)

        return stretched_image

    def stretch_color_image(self, fit, target_median, linked=True, normalize=False, apply_curves=False, curves_boost=0.0):
        # Ensure target_median is between 0.01 and 0.99
        target_median = max(0.01, min(0.99, target_median))

        channels, height, width = fit.data.shape

        if linked:
            combined_median = np.median(fit.data)
            combined_std = np.std(fit.data)
            black_point = max(np.min(fit.data), combined_median - 2.7 * combined_std)

            rescaled_image = (fit.data - black_point) / (1 - black_point)
            median_image = np.median(rescaled_image)

            stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)
        else:
            stretched_image = np.zeros_like(fit.data)

            for channel in range(channels):
                channel_data = fit.get_channel(channel)

                black_point = max(
                    np.min(channel_data),
                    np.median(channel_data) - 2.7 * np.std(channel_data)
                )

                rescaled_channel = (channel_data - black_point) / (1 - black_point)

                median_channel = np.median(rescaled_channel)

                stretched_channel = ((median_channel - 1) * target_median * rescaled_channel) / (
                    median_channel * (target_median + rescaled_channel - 1) - target_median * rescaled_channel
                )

                stretched_image[channel, ...] = stretched_channel

        if normalize:
            stretched_image = stretched_image / np.max(stretched_image)

        if apply_curves:
            stretched_image = np.clip(stretched_image, 0, None) # Make sure no negative pixels are used
            stretched_image = np.power(stretched_image, 1.0 + curves_boost)

        return stretched_image

    def apply_stretch(self, target_median, linked_stretch=False, normalize=False, apply_curves=False, curves_boost=0.0):
        """Apply statistical stretch to the current image"""
        try:
            # Get the thread
            with self.siril.image_lock():
                # Get current image
                fit = self.siril.get_image()
                fit.ensure_data_type(np.float32)

                # Save original image for undo
                self.siril.undo_save_state(f"StatStretch: m={target_median:.2f} l={linked_stretch} n={normalize} c={apply_curves} b={curves_boost:.2f}")

                # Apply stretch based on image type
                if fit.data.ndim == 3:
                    stretched_image = self.stretch_color_image(
                        fit, target_median, linked_stretch, normalize, apply_curves, curves_boost
                    )
                else:
                    stretched_image = self.stretch_mono_image(
                        fit, target_median, normalize, apply_curves, curves_boost
                    )

                # Clip and update image data
                fit.data[:] = np.clip(stretched_image, 0, 1)
                self.siril.set_image_pixeldata(fit.data)

            return True

        except Exception as e:
            raise e

class StatisticalStretchInterface(QMainWindow):
    def __init__(self, siril: s.SirilInterface, cli_args=None):
        super().__init__()

        self.siril = siril
        self.processor = StatisticalStretchProcessor(siril)

        # If no CLI args, create a default namespace with defaults
        if cli_args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-median", type=float, default=0.2)
            parser.add_argument("-boost", type=float, default=0.0)
            parser.add_argument("-linked", action="store_true", default=False)
            parser.add_argument("-normalize", action="store_true", default=False)
            cli_args = parser.parse_args([])

        self.cli_args = cli_args

        # Ensure target_median is between 0.01 and 0.99
        if hasattr(cli_args, 'median'):
            cli_args.median = max(0.01, min(0.99, cli_args.median))

        if not self.siril.is_image_loaded():
            self.show_error_message("No image is loaded")
            return

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            return

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"Statistical Stretch Interface - v{VERSION}")
        self.setFixedWidth(400)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        main_layout.addSpacing(10)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QHBoxLayout(params_group)  # Use horizontal layout directly
        params_layout.setContentsMargins(10, 5, 10, 5)  # Reduce vertical margins

        # Target Median
        median_label = QLabel("Target median:")
        params_layout.addWidget(median_label)

        self.target_median_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_median_slider.setMinimum(1)  # 0.01 * 100
        self.target_median_slider.setMaximum(99)  # 0.99 * 100
        initial_value = max(0.01, self.cli_args.median) * 100
        self.target_median_slider.setValue(int(initial_value))
        self.target_median_slider.valueChanged.connect(self.update_target_median_display)
        params_layout.addWidget(self.target_median_slider)

        self.target_median_display = QLabel(f"{self.processor.floor_value(max(0.01, self.cli_args.median)):.2f}")
        self.target_median_display.setMinimumWidth(50)
        params_layout.addWidget(self.target_median_display)

        main_layout.addWidget(params_group)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(10, 5, 10, 5)  # Reduce vertical margins
        options_layout.setSpacing(5)  # Reduce spacing between elements

        # Linked Stretch checkbox
        self.linked_stretch_checkbox = QCheckBox("Linked Stretch")
        self.linked_stretch_checkbox.setChecked(self.cli_args.linked)
        self.linked_stretch_checkbox.setToolTip(
            "When enabled, applies the same stretching parameters to all color channels "
            "simultaneously for color images. When disabled, stretches each color channel independently."
        )
        options_layout.addWidget(self.linked_stretch_checkbox)

        # Normalize checkbox
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.setChecked(self.cli_args.normalize)
        self.normalize_checkbox.setToolTip(
            "Scales the image data to use the full dynamic range from 0 to 1, "
            "ensuring maximum contrast and detail preservation after stretching."
        )
        options_layout.addWidget(self.normalize_checkbox)

        # Apply Curve checkbox
        self.apply_curve_checkbox = QCheckBox("Apply Curves Adjustment")
        self.apply_curve_checkbox.setChecked(self.cli_args.boost > 0)
        self.apply_curve_checkbox.toggled.connect(self.toggle_curves_boost)
        self.apply_curve_checkbox.setToolTip(
            "Enables non-linear curve boosting to enhance image contrast and bring out "
            "finer details by applying a power law transformation."
        )
        options_layout.addWidget(self.apply_curve_checkbox)

        # Curves Boost
        curves_layout = QHBoxLayout()
        curves_label = QLabel("Curves Boost:")
        curves_layout.addWidget(curves_label)

        self.curves_boost_slider = QSlider(Qt.Orientation.Horizontal)
        self.curves_boost_slider.setMinimum(0)
        self.curves_boost_slider.setMaximum(50)  # 0.5 * 100
        self.curves_boost_slider.setValue(int(self.cli_args.boost * 100))
        self.curves_boost_slider.valueChanged.connect(self.update_curves_boost_display)
        self.curves_boost_slider.setToolTip(
            "Controls the intensity of the non-linear curve adjustment. Higher values "
            "increase contrast and emphasize faint details, but can also introduce more noise or artifacts."
        )
        curves_layout.addWidget(self.curves_boost_slider)

        self.curves_boost_display = QLabel(f"{self.processor.floor_value(self.cli_args.boost):.2f}")
        self.curves_boost_display.setMinimumWidth(50)
        curves_layout.addWidget(self.curves_boost_display)

        options_layout.addLayout(curves_layout)

        # Initially disable curves boost if no boost
        if self.cli_args.boost == 0:
            self.curves_boost_slider.setEnabled(False)

        main_layout.addWidget(options_group)

        # Add stretch to push buttons to bottom
        main_layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close_dialog)
        close_btn.setToolTip("Close the Statistical Stretch interface and disconnect from Siril. "
                           "No changes will be made to the current image.")
        button_layout.addWidget(close_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_changes)
        apply_btn.setToolTip("Apply the selected statistical stretch parameters to the current image. "
                           "Changes can be undone using Siril's undo function.")
        button_layout.addWidget(apply_btn)

        main_layout.addLayout(button_layout)

    def update_target_median_display(self):
        """Update the displayed target median value with floor rounding"""
        value = self.target_median_slider.value() / 100.0
        rounded_value = self.processor.floor_value(value)
        self.target_median_display.setText(f"{rounded_value:.2f}")

    def update_curves_boost_display(self):
        """Update the displayed curves boost value with floor rounding"""
        value = self.curves_boost_slider.value() / 100.0
        rounded_value = self.processor.floor_value(value)
        self.curves_boost_display.setText(f"{rounded_value:.2f}")

    def toggle_curves_boost(self):
        enabled = self.apply_curve_checkbox.isChecked()
        self.curves_boost_slider.setEnabled(enabled)

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

    def apply_changes(self):
        try:
            target_median = max(0.01, self.target_median_slider.value() / 100.0)  # Ensure minimum is 0.01
            linked_stretch = self.linked_stretch_checkbox.isChecked()
            normalize = self.normalize_checkbox.isChecked()
            apply_curves = self.apply_curve_checkbox.isChecked()
            curves_boost = self.curves_boost_slider.value() / 100.0 if apply_curves else 0.0

            self.processor.apply_stretch(target_median, linked_stretch, normalize, apply_curves, curves_boost)

        except Exception as e:
            self.show_error_message(str(e))

    def close_dialog(self):
        self.close()

def run_cli_mode(siril, args):
    """Run in CLI mode without creating Qt widgets"""
    if not siril.is_image_loaded():
        print("No image is loaded")
        return

    try:
        siril.cmd("requires", "1.3.6")
    except s.CommandError:
        return

    # Ensure target_median is between 0.01 and 0.99
    args.median = max(0.01, min(0.99, args.median))

    # Create processor and apply stretch
    processor = StatisticalStretchProcessor(siril)

    try:
        target_median = args.median
        linked_stretch = args.linked
        normalize = args.normalize
        apply_curves = args.boost is not None and args.boost > 0
        curves_boost = args.boost or 0.0

        processor.apply_stretch(target_median, linked_stretch, normalize, apply_curves, curves_boost)
        print("Statistical stretch applied successfully.")

    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    try:
        # Launch to Interface to determine if we are in CLI or GUI mode and to init connection
        siril = s.SirilInterface()
        try:
            siril.connect()
        except s.SirilConnectionError:
            if not siril.is_cli():
                app = QApplication(sys.argv)
                app.setStyle('Fusion')
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle("Error")
                msg_box.setText("Failed to connect to Siril")
                msg_box.exec()
                sys.exit(1)
            else:
                print("Failed to connect to Siril")
            return

        if siril.is_cli():
            # CLI mode - no Qt widgets created
            parser = argparse.ArgumentParser(description="Statistical Stretch for Astronomical Images")
            parser.add_argument("-median", type=float, default=0.2,
                                help="Target median value for stretch (0.01 to 0.99)")
            parser.add_argument("-boost", type=float, default=0.0,
                                help="Curves boost value (0.0 to 0.5)")
            parser.add_argument("-linked", action="store_true",
                                help="Use linked stretch for color images")
            parser.add_argument("-normalize", action="store_true",
                                help="Normalize image after stretch")

            args = parser.parse_args()
            run_cli_mode(siril, args)
        else:
            # GUI mode
            app = QApplication(sys.argv)
            app.setStyle('Fusion')
            app.setApplicationName("Statistical Stretch Interface")
            window = StatisticalStretchInterface(siril)
            window.show()
            sys.exit(app.exec())

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
