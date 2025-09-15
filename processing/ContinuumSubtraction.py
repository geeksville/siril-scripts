# Narrowband Continuum Subtraction script - JAX Optimized
# SPDX-License-Identifier: GPL-3.0
# Author: Adrian Knagg-Baugh, (c) 2025

"""
This script provides continuum subtraction for narrowband images.
It uses the currently loaded narrowband image in Siril and allows the user
to select a continuum image, then automatically determines the optimal
scaling factor for subtraction by minimizing the nonuniformity in a
user-selected region using AAD (Average Absolute Deviation).

JAX-optimized version for GPU acceleration.
"""
# Version history
# 1.0.0 Initial release
# 1.0.1 Bug fixes
# 1.0.2 Improve file selection on Linux (use tkfilebrowser)
# 2.0.0 Conversion of GUI to PyQt6

import os
import sys
import math
import numpy as np

import sirilpy as s
from sirilpy import SirilError
s.ensure_installed("scipy", "matplotlib", "PyQt6")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QCheckBox, QRadioButton,
    QGroupBox, QButtonGroup, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

VERSION = "2.0.0"

def aad(data):
    mean = np.mean(data)
    return np.mean(np.abs(data - mean))


def find_min(nb, co, c_median, siril):
    scale_factors = np.linspace(-1, 5, 12)
    aad_values = []

    for i, sf in enumerate(scale_factors):
        value = aad(nb - (co - c_median) * sf)
        aad_values.append(value)
        siril.update_progress("Coarse bounds check...", i / (len(scale_factors) - 1))
    min_val = scale_factors[np.argmin(aad_values)]
    return min_val


def perform_continuum_subtraction(narrowband_image,
                                  continuum_image,
                                  selection,
                                  c_median,
                                  siril,
                                  plot_optimization=True,
                                  qt_parent=None):
    x, y, w, h = selection
    def slc(im): return im[y:y+h, x:x+w]
    nb = slc(narrowband_image)
    co = slc(continuum_image)

    approx_min = find_min(nb, co, c_median, siril)
    max_val = approx_min + 1.0
    min_val = approx_min - 1.0

    scale_factors = np.linspace(min_val, max_val, 40)
    aad_values = []

    for i, sf in enumerate(scale_factors):
        value = aad(nb - (co - c_median) * sf)
        aad_values.append(value)
        siril.update_progress("Optimizing continuum subtraction...", i / (len(scale_factors) - 1))

    def smooth_v(x, A, s0, eps, B):
        return A * np.sqrt((x - s0)**2 + eps**2) + B

    B0 = np.min(aad_values)
    s0_0 = scale_factors[np.argmin(aad_values)]
    slope_est = (aad_values[-1] - aad_values[0]) / (scale_factors[-1] - scale_factors[0])
    A0 = slope_est
    eps0 = 0.01
    p0 = [A0, s0_0, eps0, B0]
    lb = [-1.0, 0.00, 0.0, 0.00]
    ub = [np.inf, 2*max_val, np.inf, np.inf]

    popt, _ = curve_fit(smooth_v, scale_factors, aad_values, p0=p0, bounds=(lb, ub))
    A_opt, s0_opt, eps_opt, B_opt = popt
    optimal_scale = float(np.clip(s0_opt, 0, 1))

    if plot_optimization and qt_parent is not None:
        def show_plot():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(scale_factors, aad_values, color='C0', alpha=0.6, label='AAD values')
            fx = np.linspace(min_val, max_val, 500)
            fy = smooth_v(fx, *popt)
            ax.plot(fx, fy, 'C3-', label="Smooth-V fit")
            min_aad = smooth_v(optimal_scale, *popt)
            ax.plot([optimal_scale], [min_aad], 'go', ms=10, label=f'Optimal scale = {optimal_scale:.4f}')
            ax.axvline(optimal_scale, color='green', ls='--', alpha=0.5)
            ax.set_title('Optimization for Continuum Subtraction')
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('AAD')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')

            plot_window = QMainWindow(qt_parent)
            plot_window.setWindowTitle("Continuum Subtraction Optimization")
            canvas = FigureCanvasQTAgg(fig)
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addWidget(canvas)
            plot_window.setCentralWidget(central)
            plot_window.resize(800, 600)
            plot_window.show()

        QTimer.singleShot(0, show_plot)

    return optimal_scale


class ContinuumSubtractionInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Continuum Subtraction - v{VERSION}")

        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError:
            QMessageBox.critical(self, "Error", "Failed to connect to Siril")
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta3")
        except s.CommandError:
            return

        if not self.siril.is_image_loaded():
            QMessageBox.critical(self, "Error", "No image loaded. Load a narrowband image first.")
            self.close()
            return

        shape = self.siril.get_image_shape()
        if shape[0] != 1:
            QMessageBox.critical(self, "Error", "A mono narrowband image must be loaded.")
            self.close()
            return

        self.cached_continuum_path = None
        self.cached_continuum_data = None
        self.cached_continuum_median = None
        self.cached_selection = None

        self.initial_narrowband_filename = self.siril.get_image_filename()
        self.initial_narrowband_data = self.siril.get_image_pixeldata()
        self.narrowband_header = self.siril.get_image_fits_header()

        self.user_changed_narrowband = False
        self.current_narrowband_data = self.initial_narrowband_data
        self.current_narrowband_filename = self.initial_narrowband_filename

        self.continuum_path_changed = False

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        # Narrowband info
        self.nb_label = QLabel(f"Narrowband Image: {self.initial_narrowband_filename}")
        layout.addWidget(self.nb_label)

        # Continuum selector
        hbox = QHBoxLayout()
        self.cont_path = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_continuum)
        hbox.addWidget(QLabel("Continuum Image:"))
        hbox.addWidget(self.cont_path)
        hbox.addWidget(browse_btn)
        layout.addLayout(hbox)

        # Options
        self.scale_edit = QLineEdit("Auto")
        layout.addWidget(QLabel("Subtraction Factor (Auto or number):"))
        layout.addWidget(self.scale_edit)

        self.show_plot_check = QCheckBox("Show optimization plot")
        self.show_plot_check.setChecked(True)
        layout.addWidget(self.show_plot_check)

        self.output_group = QButtonGroup(self)
        subtract_radio = QRadioButton("Continuum-subtracted Narrowband")
        subtract_radio.setChecked(True)
        enhance_radio = QRadioButton("Enhanced Continuum")
        self.output_group.addButton(subtract_radio, 0)
        self.output_group.addButton(enhance_radio, 1)
        layout.addWidget(subtract_radio)
        layout.addWidget(enhance_radio)

        self.enhance_edit = QLineEdit("1.0")
        layout.addWidget(QLabel("Enhancement Factor:"))
        layout.addWidget(self.enhance_edit)

        self.status = QLabel("")
        layout.addWidget(self.status)

        # Buttons
        hbtn = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_changes)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        hbtn.addWidget(apply_btn)
        hbtn.addWidget(close_btn)
        layout.addLayout(hbtn)

        self.setCentralWidget(central)

    def browse_continuum(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Continuum Image", self.siril.get_siril_wd(),
                                                 "FITS files (*.fit *.fits *.fts);;All files (*)")
        if filename:
            self.cont_path.setText(filename)

    def apply_changes(self):
        try:
            if not self.siril.is_image_loaded():
                QMessageBox.warning(self, "Error", "No image loaded in Siril")
                return

            narrowband_data = self.current_narrowband_data
            continuum_path = self.cont_path.text()
            if not continuum_path:
                QMessageBox.warning(self, "Error", "Select a continuum image")
                return

            using_whole_image = False
            selection = self.siril.get_siril_selection()
            if selection is None or selection[2] <= 0 or selection[3] <= 0:
                shape = self.siril.get_image_shape()
                selection = (0, 0, shape[2] - 1, shape[1] - 1)
                using_whole_image = True

            continuum_data = self.siril.load_image_from_file(f'{continuum_path}').data
            if continuum_data.shape != narrowband_data.shape:
                QMessageBox.critical(self, "Error", "Image sizes must match.")
                return

            c_median = self.siril.get_selection_stats(selection, 0).median

            try:
                scale_str = self.scale_edit.text()
                scale_factor = None if scale_str.lower() == "auto" else float(scale_str)
            except ValueError:
                scale_factor = None

            if scale_factor is None:
                if using_whole_image:
                    self.siril.log("No selection made: using entire image. This is probably not optimum. "
                        "It is recommended to make a generous selection around the object of interest.", s.LogColor.SALMON)
                scale_factor = perform_continuum_subtraction(narrowband_data, continuum_data, selection, c_median,
                                                             self.siril, plot_optimization=self.show_plot_check.isChecked(),
                                                             qt_parent=self)
                self.scale_edit.setText(f"{scale_factor:.4f}")

            if self.output_group.checkedId() == 0:
                result = narrowband_data - (continuum_data - c_median) * scale_factor
                result = np.clip(result, 0, 1)
                message = f"Continuum subtraction completed with subtraction factor {scale_factor:.4f}"
            else:
                try:
                    enhance_factor = float(self.enhance_edit.text())
                except ValueError:
                    enhance_factor = 1.0
                    self.siril.log("Invalid enhancement factor, using default 1.0", s.LogColor.SALMON)
                subtracted = narrowband_data - (continuum_data - c_median) * scale_factor
                subtracted = np.clip(subtracted, 0, 1)
                result = continuum_data + subtracted * enhance_factor
                result = np.clip(result, 0, 1)
                message = f"Enhanced continuum created with subtraction factor {scale_factor:.4f} and enhancement {enhance_factor:.2f}"

            ext = self.siril.get_siril_config("core", "extension")
            self.siril.cmd("new", "1", "1", "1", f"result{ext}")
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(result.astype(np.float32))
            if isinstance(self.narrowband_header, str):
                self.siril.set_image_metadata_from_header_string(self.narrowband_header)

            self.status.setText(message)

        except SirilError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")


def main():
    app = QApplication(sys.argv)
    win = ContinuumSubtractionInterface()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
