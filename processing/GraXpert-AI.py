"""
GraXpert AI Model Interface script
==================================
This script provides a direct interface between Siril and the GraXpert
AI ONNX models.
The script offers a GUI as well as a pyscript-compatible CLI interface
(see `pyscript GraXpert.py --help` for CLI details)
Single image and sequence processing is supported.

As this script offers tighter integration of the dependencies as well
as better performance it is intended to become the primary interface
to GraXpert in the future: if you experience issues with the legacy
GraXpert interface it is recommended to try this script instead.

(c) Adrian Knagg-Baugh 2025
SPDX-License-Identifier: GPL-3.0-or-later

Model inference methods adapt code from GraXpert for Siril data formats
=======================================================================
GraXpert website: https://graxpert.com
GraXpert is (c) the GraXpert Development Team
GraXpert code licensed as GPL-3.0-or-later
Models licensed as CC-BY-NC-SA-4.0
"""

# Version History
# 1.0.0  Initial release
# 1.0.1  Bug fix in handling mono images in BGE; improved fallback behaviour
#        for inferencing runtime errors (try again with CPU backend)
# 1.0.2  Interim fix for MacOS to prevent issues with the CREATE_ML_PROGRAM
#        flag; make the defaults match GraXpert (except smoothing: the
#        default GraXpert smoothing value of 0.0 seems too low so this is
#        set at 0.5)
# 1.0.3  Fix an error with use of the onnx_helper
# 1.0.4  Fix GPU checkbox on MacOS
# 1.0.5  Fallback to CPU is more robust
# 1.0.6  Fix a bug relating to printing the used inference providers
# 1.0.7  More bugfixes
# 1.0.8  Fix interpretation of a TkBool variable as an integer
# 1.0.9  Remove -batch option from -bge -h: this option is not relevant to BG
#        extraction
# 1.0.10 CR: Change operation order
# 1.0.11 Increase timeout on GraXpert version check (required if run offline
#        apparently) and move check to ModelManager __init__ so that there is
#        no delay at startup
# 1.1.0  For beta3+: use ONNXHelper.run(), remove special macOS handling
#        Use CPU ExecutionProvider for BG extraction as this process is not
#        computationally demanding and it causes errors with some EPs
# 1.1.1  Better error messaging if the GraXpert executable isn't set or is
#        invalid
# 1.1.2  Update version string and add DLL preloading, which may improve
#        situations where system NVIDIA libraries can't be found
# 2.0.0  Update GUI to base it on PyQt6
# 2.0.1  Change import order to avoid DLL load errors on Windows

import os
import re
import sys
import copy
import argparse
import platform
import tempfile
import threading
import subprocess
from time import sleep
from packaging.version import Version, parse

import sirilpy as s
# Check the module version is enough to provide ONNXHelper
if not s.check_module_version('>=0.6.42'):
    print("Error: requires sirilpy module >= 0.6.42")
    sys.exit(1)

from sirilpy import SirilError

# Determine the correct onnxruntime package based on OS and hardware,
# and ensure it is installed
onnx_helper = s.ONNXHelper()
onnx_helper.install_onnxruntime()

import onnxruntime
if hasattr(onnxruntime, 'preload_dlls'):
    with s.SuppressedStderr(), s.SuppressedStdout():
        onnxruntime.preload_dlls()
onnxruntime.set_default_logger_severity(4)

s.ensure_installed("numpy", "astropy", "appdirs",
                   "opencv-python", "PyQt6")
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFrame, QListWidget, QScrollArea, QProgressBar, QMessageBox, QDialog,
    QRadioButton, QButtonGroup, QGroupBox, QSizePolicy, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import cv2
import numpy as np
from astropy.io import fits
from appdirs import user_data_dir

VERSION = "2.0.1"
DENOISE_CONFIG_FILENAME = "graxpert_denoise_model.conf"
BGE_CONFIG_FILENAME = "graxpert_bge_model.conf"
DECONVOLVE_STARS_CONFIG_FILENAME = "graxpert_deconv_stars_model.conf"
DECONVOLVE_OBJECTS_CONFIG_FILENAME = "graxpert_deconv_obj_model.conf"

_graxpert_mutex = threading.Lock()
_graxpert_version = None

def get_executable(siril):
    if siril is None or not siril.connected:
        return None
    executable = siril.get_siril_config('core', 'graxpert_path')
    print(executable)
    return None if 'not set' in executable else executable

def check_graxpert_version(executable):
    """
    Check the version of the GraXpert executable.
    Returns the version string if successful, None otherwise.
    """
    version_key = "version: "

    # Check if executable is valid
    if not executable or not executable.strip():
        return None

    # Check if the file exists and is executable
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        print("Error: cannot access or execute the GraXpert path", file=sys.stderr)
        return None

    with _graxpert_mutex:
        try:
            # Prepare command arguments
            cmd = [executable, "-v"]

            # Run the process with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Wait for process with timeout (200ms)
            try:
                print("Checking GraXpert version...")
                stdout, stderr = process.communicate(timeout=30)
                exit_status = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                print("GraXpert version check timed out")
                return None

            # Check for errors
            if exit_status != 0:
                print(f"Spawning GraXpert failed during version check: {stderr}")
                return None

            # Process output to extract version
            output = stderr + stdout
            if output:
                # Find the version string
                version_start = output.find(version_key)
                if version_start >= 0:
                    # Move past the key
                    version_start += len(version_key)
                    # Find the end of the version string
                    version_text = output[version_start:]
                    version_end = version_text.find(" ")

                    if version_end >= 0:
                        # Extract just the version number
                        version_string = version_text[:version_end].strip()
                    else:
                        # If no space after version, take the rest of the line
                        version_string = version_text.strip()

                    global _graxpert_version
                    _graxpert_version = version_string

            return version_string

        except Exception as e:
            print(f"Error checking GraXpert version: {str(e)}")
            return None

def get_available_local_operations():
    operations = {
            'bge': 'Background Extraction',
            'denoise': 'Denoising'
        }
    # Get the GraXpert directory
    deconvolution_stars_dir = os.path.join(user_data_dir(appname="GraXpert"), 'deconvolution-stars-ai-models')
    deconvolution_obj_dir = os.path.join(user_data_dir(appname="GraXpert"), 'deconvolution-object-ai-models')
    if get_available_local_models(deconvolution_stars_dir):
        operations.update({
            'deconvolution-stars': 'Deconvolution (Stellar)'})
    if get_available_local_models(deconvolution_obj_dir):
        operations.update({
            'deconvolution-object': 'Deconvolution (Objects)'})
    return operations

def get_available_operations():
    if _graxpert_version is None:
        return None
    version = Version(_graxpert_version)
    # If version check failed or version is less than 3.0.0, abort initialization
    operations = {
            'bge': 'Background Extraction',
            'denoise': 'Denoising'
        }
    if (version.release[0] == 3 and version.release[1] == 1 and
            version.release[2] == 0 and version.is_prerelease):
        operations.update({
            'deconvolution-stars': 'Deconvolution (Stellar)',
            'deconvolution-object': 'Deconvolution (Objects)'
        })
    return operations

def get_available_local_models(subdir : str) -> dict:
    """
    Get a dictionary of available models from the GraXpert directory.
    Returns a dict with model names as keys and paths as values.
    """

    # Get the GraXpert directory
    models_dir = os.path.join(user_data_dir(appname="GraXpert"), subdir)
    model_paths = {}

    # Check if directory exists
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        # Search for model.onnx files in subdirectories
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    # Use subdirectory name as the display name
                    model_paths[subdir] = model_path

    return model_paths

def list_available_models(models_dir):
    """List all available models and exit. For use with CLI interfaces. """
    # Check if directory exists
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        print(f"Models directory not found: {models_dir}")
        sys.exit(1)

    # Find all available models
    available_models = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            model_path = os.path.join(subdir_path, "model.onnx")
            if os.path.exists(model_path) and os.path.isfile(model_path):
                available_models.append(subdir)

    if not available_models:
        print("No models found")
        sys.exit(1)

    # Sort models and print them
    available_models.sort()
    print("Available models:")
    for model in available_models:
        print(f"  {model}")

    # Print highest available model
    print(f"\nLatest available model: {available_models[-1]} (default if no model specified)")
    sys.exit(0)

def get_image_data_from_file(siril, path):
    """
    Load image data from a file. If the data is not in a bit depth Siril handles it
    will be converted to np.float32

    Args:
        path: Path to the image file

    Returns:
        Tuple of (data, header) where data is a numpy array and header is a FITS header
    """
    if path.lower().endswith((".fit", ".fits")):
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data.dtype not in (np.float32, np.uint16):
                data = data.astype(np.float32)
            header = hdul[0].header.copy()  # Copy the header
            return data, header
    else:
        try:
            header = None
            self.siril.cmd(f"load {path}")
            header = siril.get_image_fits_header()
        except SirilError as e:
            self.siril.log(f"Error reading file {path}: {e}", s.LogColor.RED)
            return None, None
        return siril.get_image().data, header

def save_fits(data, path, original_header=None, history_text=""):
    """
    Save data to a FITS file.

    Args:
        data: Numpy array to save
        path: Path to save to
        original_header: Optional FITS header to use
        history_text: Text to add to the HISTORY keyword
    """
    if data.dtype not in (np.float32, np.uint16):
        data = data.astype(np.float32)
    # Create a new header if none is provided
    if original_header is None:
        header = fits.Header()
    else:
        try:
            with SuppressedStderr():
                header = fits.Header.fromstring(original_header, sep='\n')
        except Exception:
            header = fits.Header()
    # Add the HISTORY line
    header['HISTORY'] = history_text
    fits.writeto(path, data, header, overwrite=True)

def clear_layout(layout):
    """
    Safely removes all widgets and layouts from a layout.
    """
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    clear_layout(sub_layout)

class GUIInterface(QMainWindow):
    """Class providing the GUI interface for GraXpert AI Operations."""

    def __init__(self, siril):
        super().__init__()

        if not siril:
            raise ValueError("No SirilInterface provided to GUIInterface()")

        self.siril = siril
        self.model_manager = None

        # Get available operations
        self.operations = get_available_local_operations()
        self.selected_operation = 'bge'  # Default operation

        # Initialize processor reference (will be set based on operation)
        self.processor = None

        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()
        if not (image_loaded or seq_loaded):
            QMessageBox.critical(self, "Error", "No image or sequence loaded")
            self.close()
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except Exception:  # Replace s.CommandError with generic Exception for now
            self.close()
            return

        # Initialize variables for UI
        self.model_path = ""
        self.strength_value = 0.5
        self.smoothing_value = 0.5
        self.psf_size_value = 5.0
        self.batch_size_value = 4
        self.keep_bg_value = False
        self.gpu_acceleration_value = True
        self.correction_type_value = "subtraction"
        self.model_path_mapping = {}

        self.siril.log("This script is under ongoing development. Please report any bugs to "
            "https://gitlab.com/free-astro/siril-scripts. We are also especially keen "
            "for confirmation of success / failure from Linux users with AMD Radeon "
            "or Intel ARC GPUs as we do not have these hardware / OS combinations among "
            "the development team", color=s.LogColor.BLUE)

        # Create widgets
        self.operation_widget_map = {}  # New mapping for QStackedWidget
        self.create_widgets()

        # Set default operation to bge
        if 'bge' in self.operations:
            self.selected_operation = 'bge'
            self._on_operation_selected()  # Initialize the correct processor

        # Set progress label
        if image_loaded:
            self._update_progress("Single image loaded: will process this image only")
        else:
            self._update_progress("Sequence loaded: will process selected frames of the sequence")

    def create_widgets(self):
        """Create Qt widgets to provide the script GUI"""
        self.setWindowTitle(f"GraXpert AI - Siril interface v{VERSION}")
        # Remove fixed size - let the window size itself to content
        self.setMinimumWidth(500)
        self.setMaximumWidth(700)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Title and version
        title_label = QLabel("GraXpert AI")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        version_label = QLabel(f"Script version: {VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(version_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)

        # Operation selection
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Operation:"))

        self.op_dropdown = QComboBox()
        self.op_dropdown.addItems(list(self.operations.values()))
        self.op_dropdown.currentTextChanged.connect(self._on_operation_dropdown_changed)
        op_layout.addWidget(self.op_dropdown)
        main_layout.addLayout(op_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))

        self.model_dropdown = QComboBox()
        self.model_dropdown.currentTextChanged.connect(self._on_model_selected)
        model_layout.addWidget(self.model_dropdown)
        main_layout.addLayout(model_layout)

        # Parameters Frame - using a QStackedWidget with compact size policy
        self.params_stack = QStackedWidget()
        # Set size policy to fit content and not expand unnecessarily
        self.params_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(self.params_stack)

        # Create frames for each operation's parameters and add to the stack
        self.operation_widget_map = {}
        self._create_operation_parameters()

        # Advanced parameters (common for all operations)
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QVBoxLayout(advanced_group)

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        batch_layout.addStretch()
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        self.batch_size_spin.valueChanged.connect(self._update_batch_size)
        batch_layout.addWidget(self.batch_size_spin)
        advanced_layout.addLayout(batch_layout)

        # GPU acceleration checkbox
        self.gpu_checkbox = QCheckBox("Use GPU acceleration (if available)")
        self.gpu_checkbox.setChecked(True)
        self.gpu_checkbox.toggled.connect(self._update_gpu_acceleration)
        advanced_layout.addWidget(self.gpu_checkbox)

        main_layout.addWidget(advanced_group)

        # Action buttons
        buttons_layout = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        buttons_layout.addWidget(apply_btn)

        model_btn = QPushButton("GraXpert Model Manager")
        model_btn.clicked.connect(self.load_model_manager)
        buttons_layout.addWidget(model_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons_layout.addWidget(close_btn)

        main_layout.addLayout(buttons_layout)

        # Progress message label
        self.progress_label = QLabel("")
        # Set word wrap in case of long messages
        self.progress_label.setWordWrap(True)
        main_layout.addWidget(self.progress_label)

        # Remove the addStretch() that was causing the blank space
        # The layout will now size itself to fit the content

        # Initialize with default operation
        self._populate_model_dropdown()

        # Resize window to fit contents after everything is set up
        self.adjustSize()

    def _create_operation_parameters(self):
        """Create parameter widgets for each operation and add them to the stacked widget"""
        # Denoise operation parameters
        denoise_widget = QWidget()
        denoise_layout = QVBoxLayout(denoise_widget)
        denoise_layout.setSpacing(8)
        denoise_layout.setContentsMargins(10, 10, 10, 10)

        # Strength slider and spinbox
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)  # 0-1.0 mapped to 0-100
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self._update_strength_from_slider)
        strength_layout.addWidget(self.strength_slider)

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setSingleStep(0.01)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setValue(0.5)
        self.strength_spin.valueChanged.connect(self._update_strength_from_spin)
        strength_layout.addWidget(self.strength_spin)

        denoise_layout.addLayout(strength_layout)
        # Remove addStretch() that was creating unnecessary space

        self.params_stack.addWidget(denoise_widget)
        self.operation_widget_map['denoise'] = self.params_stack.indexOf(denoise_widget)

        # BGE operation parameters
        bge_widget = QWidget()
        bge_layout = QVBoxLayout(bge_widget)
        bge_layout.setSpacing(8)
        bge_layout.setContentsMargins(10, 10, 10, 10)

        # Smoothing slider and spinbox
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing:"))

        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(50)
        self.smoothing_slider.valueChanged.connect(self._update_smoothing_from_slider)
        smoothing_layout.addWidget(self.smoothing_slider)

        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 1.0)
        self.smoothing_spin.setSingleStep(0.01)
        self.smoothing_spin.setDecimals(2)
        self.smoothing_spin.setValue(0.5)
        self.smoothing_spin.valueChanged.connect(self._update_smoothing_from_spin)
        smoothing_layout.addWidget(self.smoothing_spin)

        bge_layout.addLayout(smoothing_layout)

        # Correction type
        correction_layout = QHBoxLayout()
        correction_layout.addWidget(QLabel("Correction Type:"))
        correction_layout.addStretch()

        self.correction_combo = QComboBox()
        self.correction_combo.addItems(["subtraction", "division"])
        self.correction_combo.currentTextChanged.connect(self._update_correction_type)
        correction_layout.addWidget(self.correction_combo)
        bge_layout.addLayout(correction_layout)

        # Keep background checkbox
        self.keep_bg_checkbox = QCheckBox("Keep background")
        self.keep_bg_checkbox.toggled.connect(self._update_keep_bg)
        bge_layout.addWidget(self.keep_bg_checkbox)

        # Remove addStretch() that was creating unnecessary space

        self.params_stack.addWidget(bge_widget)
        self.operation_widget_map['bge'] = self.params_stack.indexOf(bge_widget)

        # Deconvolution operations parameters
        deconv_stars_widget = QWidget()
        deconv_stars_layout = QVBoxLayout(deconv_stars_widget)
        deconv_stars_layout.setSpacing(8)
        deconv_stars_layout.setContentsMargins(10, 10, 10, 10)

        # Strength slider and spinbox
        strength_layout_s = QHBoxLayout()
        strength_layout_s.addWidget(QLabel("Strength:"))

        self.deconv_stars_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.deconv_stars_strength_slider.setRange(0, 100)
        self.deconv_stars_strength_slider.setValue(50)
        self.deconv_stars_strength_slider.valueChanged.connect(self._update_strength_from_slider)
        strength_layout_s.addWidget(self.deconv_stars_strength_slider)

        self.deconv_stars_strength_spin = QDoubleSpinBox()
        self.deconv_stars_strength_spin.setRange(0.0, 1.0)
        self.deconv_stars_strength_spin.setSingleStep(0.01)
        self.deconv_stars_strength_spin.setDecimals(2)
        self.deconv_stars_strength_spin.setValue(0.5)
        self.deconv_stars_strength_spin.valueChanged.connect(self._update_strength_from_spin)
        strength_layout_s.addWidget(self.deconv_stars_strength_spin)

        deconv_stars_layout.addLayout(strength_layout_s)

        # PSF Size slider and spinbox
        psf_layout_s = QHBoxLayout()
        psf_layout_s.addWidget(QLabel("PSF Size:"))

        self.deconv_stars_psf_slider = QSlider(Qt.Orientation.Horizontal)
        self.deconv_stars_psf_slider.setRange(1, 100)
        self.deconv_stars_psf_slider.setValue(50)
        self.deconv_stars_psf_slider.valueChanged.connect(self._update_psf_from_slider)
        psf_layout_s.addWidget(self.deconv_stars_psf_slider)

        self.deconv_stars_psf_spin = QDoubleSpinBox()
        self.deconv_stars_psf_spin.setRange(0.1, 10.0)
        self.deconv_stars_psf_spin.setSingleStep(0.1)
        self.deconv_stars_psf_spin.setDecimals(1)
        self.deconv_stars_psf_spin.setValue(5.0)
        self.deconv_stars_psf_spin.valueChanged.connect(self._update_psf_from_spin)
        psf_layout_s.addWidget(self.deconv_stars_psf_spin)

        deconv_stars_layout.addLayout(psf_layout_s)
        # Remove addStretch() that was creating unnecessary space

        self.params_stack.addWidget(deconv_stars_widget)
        self.operation_widget_map['deconvolution-stars'] = self.params_stack.indexOf(deconv_stars_widget)

        deconv_object_widget = QWidget()
        deconv_object_layout = QVBoxLayout(deconv_object_widget)
        deconv_object_layout.setSpacing(8)
        deconv_object_layout.setContentsMargins(10, 10, 10, 10)

        # Strength slider and spinbox
        strength_layout_o = QHBoxLayout()
        strength_layout_o.addWidget(QLabel("Strength:"))

        self.deconv_object_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.deconv_object_strength_slider.setRange(0, 100)
        self.deconv_object_strength_slider.setValue(50)
        self.deconv_object_strength_slider.valueChanged.connect(self._update_strength_from_slider)
        strength_layout_o.addWidget(self.deconv_object_strength_slider)

        self.deconv_object_strength_spin = QDoubleSpinBox()
        self.deconv_object_strength_spin.setRange(0.0, 1.0)
        self.deconv_object_strength_spin.setSingleStep(0.01)
        self.deconv_object_strength_spin.setDecimals(2)
        self.deconv_object_strength_spin.setValue(0.5)
        self.deconv_object_strength_spin.valueChanged.connect(self._update_strength_from_spin)
        strength_layout_o.addWidget(self.deconv_object_strength_spin)

        deconv_object_layout.addLayout(strength_layout_o)

        # PSF Size slider and spinbox
        psf_layout_o = QHBoxLayout()
        psf_layout_o.addWidget(QLabel("PSF Size:"))

        self.deconv_object_psf_slider = QSlider(Qt.Orientation.Horizontal)
        self.deconv_object_psf_slider.setRange(1, 100)
        self.deconv_object_psf_slider.setValue(50)
        self.deconv_object_psf_slider.valueChanged.connect(self._update_psf_from_slider)
        psf_layout_o.addWidget(self.deconv_object_psf_slider)

        self.deconv_object_psf_spin = QDoubleSpinBox()
        self.deconv_object_psf_spin.setRange(0.1, 10.0)
        self.deconv_object_psf_spin.setSingleStep(0.1)
        self.deconv_object_psf_spin.setDecimals(1)
        self.deconv_object_psf_spin.setValue(5.0)
        self.deconv_object_psf_spin.valueChanged.connect(self._update_psf_from_spin)
        psf_layout_o.addWidget(self.deconv_object_psf_spin)

        deconv_object_layout.addLayout(psf_layout_o)
        # Remove addStretch() that was creating unnecessary space

        self.params_stack.addWidget(deconv_object_widget)
        self.operation_widget_map['deconvolution-object'] = self.params_stack.indexOf(deconv_object_widget)

    def _update_strength_from_slider(self):
        """Update strength spinbox when slider changes"""
        sender = self.sender()
        value = sender.value() / 100.0
        self.strength_value = value

        # Update the corresponding spinbox
        if hasattr(self, 'strength_spin'):
            self.strength_spin.blockSignals(True)
            self.strength_spin.setValue(value)
            self.strength_spin.blockSignals(False)

    def _update_strength_from_spin(self):
        """Update strength slider when spinbox changes"""
        sender = self.sender()
        value = sender.value()
        self.strength_value = value

        # Update the corresponding slider
        if hasattr(self, 'strength_slider'):
            self.strength_slider.blockSignals(True)
            self.strength_slider.setValue(int(value * 100))
            self.strength_slider.blockSignals(False)

    def _update_smoothing_from_slider(self):
        """Update smoothing spinbox when slider changes"""
        value = self.smoothing_slider.value() / 100.0
        self.smoothing_value = value
        self.smoothing_spin.blockSignals(True)
        self.smoothing_spin.setValue(value)
        self.smoothing_spin.blockSignals(False)

    def _update_smoothing_from_spin(self):
        """Update smoothing slider when spinbox changes"""
        value = self.smoothing_spin.value()
        self.smoothing_value = value
        self.smoothing_slider.blockSignals(True)
        self.smoothing_slider.setValue(int(value * 100))
        self.smoothing_slider.blockSignals(False)

    def _update_psf_from_slider(self):
        """Update PSF spinbox when slider changes"""
        sender = self.sender()
        value = sender.value() / 10.0  # Map 1-100 to 0.1-10.0
        self.psf_size_value = value

        # Find and update the corresponding spinbox
        operation = self.selected_operation
        if operation == 'deconvolution-stars':
            self.deconv_stars_psf_spin.blockSignals(True)
            self.deconv_stars_psf_spin.setValue(value)
            self.deconv_stars_psf_spin.blockSignals(False)
        elif operation == 'deconvolution-object':
            self.deconv_object_psf_spin.blockSignals(True)
            self.deconv_object_psf_spin.setValue(value)
            self.deconv_object_psf_spin.blockSignals(False)

    def _update_psf_from_spin(self):
        """Update PSF slider when spinbox changes"""
        sender = self.sender()
        value = sender.value()
        self.psf_size_value = value

        # Find and update the corresponding slider
        operation = self.selected_operation
        if operation == 'deconvolution-stars':
            self.deconv_stars_psf_slider.blockSignals(True)
            self.deconv_stars_psf_slider.setValue(int(value * 10))
            self.deconv_stars_psf_slider.blockSignals(False)
        elif operation == 'deconvolution-object':
            self.deconv_object_psf_slider.blockSignals(True)
            self.deconv_object_psf_slider.setValue(int(value * 10))
            self.deconv_object_psf_slider.blockSignals(False)

    def _update_correction_type(self, value):
        """Update correction type value"""
        self.correction_type_value = value

    def _update_keep_bg(self, checked):
        """Update keep background value"""
        self.keep_bg_value = checked

    def _update_batch_size(self, value):
        """Update batch size value"""
        self.batch_size_value = value

    def _update_gpu_acceleration(self, checked):
        """Update GPU acceleration value"""
        self.gpu_acceleration_value = checked

    def _on_operation_dropdown_changed(self, display_name):
        """Handle operation selection change from dropdown"""
        # Map display name back to operation key
        operation_keys = list(self.operations.keys())
        operation_names = list(self.operations.values())
        try:
            index = operation_names.index(display_name)
            operation = operation_keys[index]
            self.selected_operation = operation
        except ValueError:
            operation = self.selected_operation or 'bge'

        self._on_operation_selected()

    def _on_operation_selected(self):
        """Handle operation selection change"""
        operation = self.selected_operation

        # Update the processor based on the selected operation
        if operation == 'denoise':
            self.processor = DenoiserProcessing(self.siril)
            model_path = self.processor.check_config_file()
            self.model_path = model_path or ""
        elif operation == 'bge':
            self.processor = BGEProcessing(self.siril)
            model_path = self.processor.check_config_file()
            self.model_path = model_path or ""
        elif operation in ['deconvolution-stars', 'deconvolution-object']:
            self.processor = DeconvolutionProcessing(self.siril)
            model_path = self.processor.check_config_file(operation)
            self.model_path = model_path or ""

        # Update model dropdown based on operation
        self._populate_model_dropdown()

        # Show the appropriate parameter widget using QStackedWidget
        if operation in self.operation_widget_map:
            index_to_show = self.operation_widget_map[operation]
            self.params_stack.setCurrentIndex(index_to_show)

        # Update the window title
        op_display_name = self.operations.get(operation, "Operation")
        self.setWindowTitle(f"GraXpert AI {op_display_name} - Siril interface v{VERSION}")

        # Resize window to fit new content
        self.adjustSize()

    def load_model_manager(self):
        """Load a model manager dialog"""
        model_manager = GraXpertModelManager(self, self.siril, self.update_dropdowns)
        model_manager.show_dialog()

    def update_dropdowns(self):
        """Update dropdowns after model manager operations"""
        self._populate_model_dropdown()
        self._populate_operations_dropdown()

    def _populate_operations_dropdown(self):
        """
        Rescans available operations and updates the operations dropdown.
        Should be called when new operations are downloaded or become available.
        """
        # Get the most up-to-date operations
        self.operations = get_available_local_operations()

        # Convert operations dict to display names for dropdown
        operation_names = list(self.operations.values())
        operation_keys = list(self.operations.keys())

        # Update the dropdown with new values
        self.op_dropdown.clear()
        self.op_dropdown.addItems(operation_names)

        # Get the current selected operation key (if any)
        current_key = self.selected_operation

        if current_key and current_key in self.operations:
            # If current selection is still available, keep it selected
            display_name = self.operations[current_key]
            index = operation_names.index(display_name)
            self.op_dropdown.setCurrentIndex(index)
        elif operation_keys:
            # Otherwise select the first available operation
            first_key = operation_keys[0]
            self.selected_operation = first_key
            self.op_dropdown.setCurrentIndex(0)
            # Since selection changed, update processor and UI
            self._on_operation_selected()
        else:
            # If no operations are available
            self.op_dropdown.addItem("No operations available")
            self.selected_operation = ""

        # Update window title
        if self.selected_operation and self.selected_operation in self.operations:
            op_display_name = self.operations[self.selected_operation]
            self.setWindowTitle(f"GraXpert AI {op_display_name} - Siril interface v{VERSION}")
        else:
            self.setWindowTitle(f"GraXpert AI - Siril interface v{VERSION}")

    def _populate_model_dropdown(self):
        """Populate the model dropdown with available models for the current operation"""
        operation = self.selected_operation

        # Get model directory name based on operation
        model_dir = None
        if operation == 'denoise':
            model_dir = "denoise-ai-models"
        elif operation == 'bge':
            model_dir = "bge-ai-models"
        elif operation == 'deconvolution-stars':
            model_dir = "deconvolution-stars-ai-models"
        elif operation == 'deconvolution-object':
            model_dir = "deconvolution-object-ai-models"
        else:
            # Default to denoise models if operation not recognized
            model_dir = "denoise-ai-models"

        # Dictionary to store model name -> full path mapping
        model_paths = get_available_local_models(model_dir)

        # Update the dropdown values
        self.model_dropdown.clear()
        if model_paths:
            # Sort model names alphabetically
            model_names = sorted(model_paths.keys())
            self.model_dropdown.addItems(model_names)

            # Store the full path mapping for when selection changes
            self.model_path_mapping = model_paths

            # Get the previously selected model path
            current_path = self.model_path

            # Find the model name associated with the current path
            selected_model = None
            for name, path in model_paths.items():
                if path == current_path:
                    selected_model = name
                    break

            if selected_model and selected_model in model_names:
                # Set the dropdown to the previously selected model
                index = model_names.index(selected_model)
                self.model_dropdown.setCurrentIndex(index)
            else:
                # If no match found or no previous selection, select the first model
                self.model_dropdown.setCurrentIndex(0)
                if model_names:
                    selected_model = model_names[0]
                    self.model_path = model_paths[selected_model]
        else:
            self.model_dropdown.addItem("No models found")
            self.model_path = ""

    def _on_model_selected(self):
        """Update the model_path when a model is selected from dropdown"""
        selected_model = self.model_dropdown.currentText()
        if selected_model in self.model_path_mapping:
            # Set the full path
            self.model_path = self.model_path_mapping[selected_model]

            # Save config with updated model path
            if self.processor:
                self.processor.save_config_file(self.model_path)
                # Reset the cached image to force recalculation
                self.processor.reset_cache()

    def _on_apply(self):
        """
        Handle the 'Apply' button click event.
        This method starts the appropriate processing thread based on the current state.
        """
        model_path = self.model_path
        operation = self.selected_operation

        # Get processing parameters from UI
        batch_size = self.batch_size_value
        gpu_acceleration = self.gpu_acceleration_value

        # Validate model path
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.critical(self, "Error", "Please select a valid ONNX model file.")
            return

        if self.siril.is_image_loaded():
            if operation == "bge":
                correction_type = self.correction_type_value
                smoothing = self.smoothing_value

                # Cache the original image
                self.processor.cached_original_image = self.siril.get_image_pixeldata()

                # Reshape mono images to 3D with a channels size of 1
                if self.processor.cached_original_image.ndim == 2:
                    self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                filename = None
                header = None
                keep_bg = self.keep_bg_value
                if keep_bg:
                    filename = self.siril.get_image_filename()
                    header = self.siril.get_image_fits_header()

                # Start image processing thread
                threading.Thread(
                    target=self.processor.process_image,
                    args=(
                        model_path,
                        correction_type,
                        smoothing,
                        keep_bg,
                        filename,
                        header,
                        gpu_acceleration,
                        self._update_progress
                    ),
                    daemon=True
                ).start()

            elif operation == "denoise":
                strength = self.strength_value
                if self.processor.cached_processed_image is None:
                    # Cache the original image if this is first-time processing
                    self.processor.cached_original_image = self.siril.get_image_pixeldata()

                    # Reshape mono images to 3D with a channels size of 1
                    if self.processor.cached_original_image.ndim == 2:
                        self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                    # Start image processing thread
                    threading.Thread(
                        target=self.processor.process_image,
                        args=(model_path, strength, batch_size,
                            gpu_acceleration, self._update_progress),
                        daemon=True
                    ).start()
                else:
                    # Apply operation-specific blend
                    threading.Thread(
                        target=lambda: self.processor.apply_blend(strength),
                        daemon=True
                    ).start()

            elif operation in ["deconvolution-stars", "deconvolution-object"]:
                strength = self.strength_value
                psf_size = self.psf_size_value

                if self.processor.cached_processed_image is None:
                    # Cache the original image if this is first-time processing
                    self.processor.cached_original_image = self.siril.get_image_pixeldata()

                # Reshape mono images to 3D with a channels size of 1
                if self.processor.cached_original_image.ndim == 2:
                    self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

                deconv_type = "stars" if operation == "deconvolution-stars" else "object"
                self._update_progress(f"Processing image with {deconv_type} deconvolution (PSF size: {psf_size:.2f}, strength: {strength:.2f})")

                threading.Thread(
                    target=self.processor.process_image,
                    args=(model_path, strength, psf_size, batch_size,
                          gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()

        elif self.siril.is_sequence_loaded():
            sequence_name = self.siril.get_seq().seqname

            if operation == "bge":
                correction_type = self.correction_type_value
                smoothing = self.smoothing_value
                keep_bg = self.keep_bg_value

                # Start sequence processing thread
                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, correction_type, smoothing,
                        keep_bg, gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()

            elif operation == "denoise":
                strength = self.strength_value

                # Start sequence processing thread
                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, strength, batch_size,
                        gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()

            elif operation in ["deconvolution-stars", "deconvolution-object"]:
                strength = self.strength_value
                psf_size = self.psf_size_value

                deconv_type = "stars" if operation == "deconvolution-stars" else "object"
                self._update_progress(f"Processing {sequence_name} with {deconv_type} deconvolution (PSF size: {psf_size:.2f}, strength: {strength:.2f})")

                threading.Thread(
                    target=self.processor.process_sequence,
                    args=(sequence_name, model_path, strength, psf_size, batch_size,
                        gpu_acceleration, self._update_progress),
                    daemon=True
                ).start()
        else:
            QMessageBox.critical(self, "Error", "No sequence or image is loaded.")

    def _update_progress(self, message, progress=0):
        """Update progress message"""
        self.progress_label.setText(message)
        self.siril.update_progress(message, progress)

    def closeEvent(self, event):
        """Handle window close event"""
        self.siril.disconnect()
        event.accept()

class GraXpertModelManager(QDialog):
    def __init__(self, parent, siril, callback=None):
        """
        Initialize the GraXpert Model Manager dialog.

        Args:
            parent: The parent window/widget
        """
        super().__init__(parent)
        self.parent = parent
        self.siril = siril
        self.callback = callback
        self.models_by_operation = {}
        self.initialized = False

        # Check GraXpert version and set up operations accordingly
        graxpert_executable = get_executable(siril)
        if graxpert_executable is None:
            QMessageBox.critical(self, "GraXpert not found",
                               "Please set the location of the GraXpert executable in Siril Preferences -> Miscellaneous")
            return
        check_graxpert_version(graxpert_executable)
        if _graxpert_version is None:
            QMessageBox.critical(self, "Error checking GraXpert version",
                               "Please check the location of the GraXpert executable in Siril Preferences -> Miscellaneous")
            return
        self.operations = get_available_operations()

        self.operation_cmd_map = {
            'bge': 'background-extraction',
            'denoise': 'denoising',
            'deconvolution-stars': 'deconv-stellar',
            'deconvolution-object': 'deconv-obj'
        }
        self.initialized = True

    def show_dialog(self):
        """Show the model manager dialog"""
        self.setWindowTitle("GraXpert Model Manager")
        self.resize(600, 600)
        self.setMinimumSize(500, 400)
        self.setModal(True)

        self.create_widgets()

        # Start refreshing the model list
        self.refresh_models()

        # Position the dialog relative to the parent window
        self.center_dialog()
        self.show()

    def center_dialog(self):
        """Center the dialog on the parent window"""
        if self.parent:
            parent_geometry = self.parent.geometry()
            dialog_geometry = self.geometry()

            x = parent_geometry.x() + (parent_geometry.width() - dialog_geometry.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - dialog_geometry.height()) // 2

            self.move(x, y)

    def create_widgets(self):
        """Create the widgets for the dialog"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Operation selection
        op_group = QGroupBox("Operation")
        op_layout = QGridLayout(op_group)

        self.operation_group = QButtonGroup()
        self.operation_buttons = {}

        max_columns = 2
        for i, (op_key, op_name) in enumerate(self.operations.items()):
            row = i // max_columns
            column = i % max_columns

            radio_btn = QRadioButton(op_name)
            radio_btn.setObjectName(op_key)  # Store the key as object name
            self.operation_group.addButton(radio_btn, i)
            self.operation_buttons[op_key] = radio_btn

            op_layout.addWidget(radio_btn, row, column)

        # Set default selection
        if 'bge' in self.operation_buttons:
            self.operation_buttons['bge'].setChecked(True)

        self.operation_group.buttonClicked.connect(self.on_operation_changed)
        layout.addWidget(op_group)

        # Model list frame
        model_group = QGroupBox("Models Available Remotely")
        model_layout = QVBoxLayout(model_group)

        self.model_listbox = QListWidget()
        self.model_listbox.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        model_layout.addWidget(self.model_listbox)

        layout.addWidget(model_group)

        # Status section
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_models)
        buttons_layout.addWidget(self.refresh_btn)

        buttons_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_btn)

        self.download_btn = QPushButton("Download Selected Model")
        self.download_btn.clicked.connect(self.download_selected_model)
        self.download_btn.setEnabled(False)
        buttons_layout.addWidget(self.download_btn)

        layout.addLayout(buttons_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Initially hidden
        layout.addWidget(self.progress_bar)

    def on_operation_changed(self):
        """Handle operation change event"""
        checked_button = self.operation_group.checkedButton()
        if checked_button:
            operation = checked_button.objectName()
            self.update_model_list(operation)

    def update_model_list(self, operation):
        """Update the model list based on the selected operation"""
        self.model_listbox.clear()
        self.download_btn.setEnabled(False)

        if operation in self.models_by_operation:
            models = self.models_by_operation[operation]
            if models:
                for model in models:
                    self.model_listbox.addItem(model)
                if self.model_listbox.count() > 0:
                    self.model_listbox.setCurrentRow(0)
                    self.download_btn.setEnabled(True)
            else:
                self.model_listbox.addItem("No models available")
        else:
            self.model_listbox.addItem("Click Refresh to check available models")

    def refresh_models(self):
        """Refresh the available models list"""
        checked_button = self.operation_group.checkedButton()
        if not checked_button:
            return

        operation = checked_button.objectName()
        operation_name = self.operations[operation]

        self.status_label.setText(f"Checking available models for {operation_name}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.refresh_btn.setEnabled(False)
        self.download_btn.setEnabled(False)

        # Start a thread to avoid blocking the UI
        self.fetch_thread = ModelFetchThread(self, operation)
        self.fetch_thread.finished.connect(self._update_after_fetch)
        self.fetch_thread.start()

    def _update_after_fetch(self, operation, models):
        """Update UI after fetching models"""
        if self.callback:
            self.callback()

        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)

        if models:
            self.models_by_operation[operation] = models
            operation_name = self.operations[operation]
            self.status_label.setText(f"Found {len(models)} models for {operation_name}")
            self.update_model_list(operation)
        else:
            self.models_by_operation[operation] = []
            self.status_label.setText("Failed to retrieve models. Check GraXpert installation.")
            self.model_listbox.clear()
            self.model_listbox.addItem("Error retrieving models")

    def download_selected_model(self):
        """Download the selected model"""
        checked_button = self.operation_group.checkedButton()
        if not checked_button:
            return

        operation = checked_button.objectName()
        current_item = self.model_listbox.currentItem()

        if not current_item:
            QMessageBox.critical(self, "Selection Error", "Please select a model to download")
            return

        model_version = current_item.text()

        self.status_label.setText(f"Downloading {model_version}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.download_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)

        # Start a thread to download the model
        self.download_thread = ModelDownloadThread(self, operation, model_version)
        self.download_thread.finished.connect(self._update_after_download)
        self.download_thread.start()

    def _update_after_download(self, success, version):
        """Update UI after downloading model"""
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.download_btn.setEnabled(True)

        if success:
            self.status_label.setText(f"Successfully downloaded model {version}")
            QMessageBox.information(self, "Download Complete",
                                  f"Model {version} has been downloaded successfully.")
        else:
            self.status_label.setText("Failed to download model")
            QMessageBox.critical(self, "Download Failed",
                               "Failed to download the selected model. Check the logs for details.")

    def check_ai_versions(self, operation):
        """
        Check available AI model versions for the specified GraXpert operation.

        Args:
            operation (str): One of 'denoise', 'bge', 'deconv_star', or 'deconv_obj'

        Returns:
            list: List of available AI model versions, or None if check fails
        """
        # Map operation names to GraXpert command arguments
        operation_map = self.operation_cmd_map

        if operation not in operation_map:
            print(f"Invalid operation: {operation}")
            return None

        try:
            executable = get_executable(self.siril)
            with _graxpert_mutex:
                if not executable:
                    return None

                # Prepare command arguments
                cmd_args = [
                    executable,
                    "-cmd",
                    operation_map[operation],
                    "--help"
                ]

                try:
                    # Execute the command with a timeout
                    process = subprocess.Popen(
                        cmd_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        universal_newlines=True
                    )

                    # Wait for the process with timeout (500ms in the C code)
                    try:
                        stdout, stderr = process.communicate(timeout=10)
                        output = stdout + stderr  # Combine both stdout and stderr as the help text might be in either
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                        output = stdout + stderr
                        print("GraXpert process timed out")
                        return None
                    # Check if version information was found
                    return self.parse_ai_versions(output)

                except Exception as e:
                    print(f"Error executing GraXpert: {str(e)}")
                    return None
        except Exception as e:
            print(f"Error in check_ai_versions: {str(e)}")
            return None

    def parse_ai_versions(self, output):
        """
        Parse the output of GraXpert to extract available AI model versions.

        Args:
            output (str): The output from GraXpert command

        Returns:
            list: List of available AI model versions, or None if parsing fails
        """
        # Preprocess the output to simplify the parsing
        # 1. Split into lines
        lines = output.split('\n')

        # 2. For each line, strip everything up to "root INFO" and any leading/trailing whitespace
        cleaned_lines = []
        for line in lines:
            root_info_pos = line.find("root INFO")
            if root_info_pos != -1:
                # Extract content after "root INFO" and strip whitespace
                cleaned_line = line[root_info_pos + len("root INFO"):].strip()
                if cleaned_line:  # Only add non-empty lines
                    cleaned_lines.append(cleaned_line)

        # 3. Join the cleaned lines with spaces
        cleaned_output = ' '.join(cleaned_lines)

        # 4. Replace multiple spaces with a single space
        cleaned_output = re.sub(r'\s+', ' ', cleaned_output)

        # Now we can use a simple regex to find and extract the version list
        match = re.search(r'available remotely:\s*\[(.*?)\]', cleaned_output)
        if not match:
            print("No 'available remotely:' list found")
            return None

        # Extract the list content
        list_content = match.group(1).strip()

        # Split by commas and extract versions
        versions = []
        for part in list_content.split(','):
            part = part.strip()
            # Skip empty parts
            if not part:
                continue
            # Extract version patterns (like 1.0.0, 2.3.4, etc.)
            version_match = re.search(r'\d+\.\d+\.\d+', part)
            if version_match:
                versions.append(version_match.group(0))

        if not versions:
            print("No valid versions found in list")
            return None
        return versions
    
    def download_model(self, operation, version):
        """
        Download a specific AI model version.

        Args:
            operation (str): Operation type ('denoise', 'bge', 'deconv_star', 'deconv_obj')
            version (str): Model version to download

        Returns:
            bool: True if download was successful, False otherwise
        """
        executable = get_executable(self.siril)
        if not executable:
            return False

        command = self.operation_cmd_map[operation]

        # Create temporary FITS file
        with _graxpert_mutex:
            try:
                with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp_file:
                    temp_fits_path = tmp_file.name

                # Create a minimal FITS file
                data = np.zeros((256, 256), dtype=np.float32)
                hdu = fits.PrimaryHDU(data)
                hdu.writeto(temp_fits_path, overwrite=True)

                # Prepare download command
                cmd_args = [
                    executable,
                    "-cli",
                    "-cmd",
                    command,
                    "-ai_version",
                    version,
                    "-output",
                    temp_fits_path.rstrip('.fits'),
                    temp_fits_path
                ]

                print(f"Running download command: {' '.join(cmd_args)}")

                # Execute the command
                process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True
                )

                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    print(f"Error downloading model: {stderr}")
                    return False

                # Check if download was successful by looking for success indicators in output
                success = ("download successful" in stdout or
                        process.returncode == 0)

                if not success:
                    # Handle case where success isn't explicitly stated but download was successful
                    # For example, some tools may not output success messages
                    success = "error" not in stdout.lower() and "failed" not in stdout.lower()

                if success:
                    print("Download succeeded")
                    return True
                else:
                    print("Download failed")
                    return False

            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                return False
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_fits_path):
                        os.unlink(temp_fits_path)
                except:
                    pass

class ModelFetchThread(QThread):
    """Thread for fetching models without blocking UI"""
    finished = pyqtSignal(str, list)  # operation, models

    def __init__(self, manager, operation):
        super().__init__()
        self.manager = manager
        self.operation = operation

    def run(self):
        models = self.manager.check_ai_versions(self.operation)
        self.finished.emit(self.operation, models or [])

class ModelDownloadThread(QThread):
    """Thread for downloading models without blocking UI"""
    finished = pyqtSignal(bool, str)  # success, version

    def __init__(self, manager, operation, version):
        super().__init__()
        self.manager = manager
        self.operation = operation
        self.version = version

    def run(self):
        success = self.manager.download_model(self.operation, self.version)
        self.finished.emit(success, self.version)

class DenoiserProcessing:
    """Class encapsulating the core image processing functionality for GraXpert AI Denoise."""

    def __init__(self, siril):
        """
        Initialize the processing class.

        Args:
            siril: SirilInterface instance, which must already be connected.
        """
        if not siril:
            raise ValueError("No SirilInterface provided to DenoiserProcessing()")

        self.siril = siril

        # Cache for processed images
        self.cached_processed_image = None
        self.cached_original_image = None

        self.config_dir = self.siril.get_siril_configdir() if self.siril else None

    def reset_cache(self):
        """Reset the cached denoised image."""
        self.cached_processed_image = None
        self.cached_original_image = None

    def check_config_file(self):
        """
        Check for a saved model path in the configuration file.
        Returns model_path or None if not found.
        """
        if not self.config_dir:
            return None

        config_file_path = os.path.join(self.config_dir, DENOISE_CONFIG_FILENAME)
        model_path = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
        return model_path

    def save_config_file(self, model_path):
        """
        Save the selected model path to the configuration file.
        """
        if not self.config_dir:
            return

        config_file_path = os.path.join(self.config_dir, DENOISE_CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(model_path + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def apply_blend(self, strength):
        """
        Apply blending with cached denoised image.

        Args:
            strength: Blending strength (0-1)

        Returns:
            Blended image as numpy array
        """
        try:
            if self.cached_processed_image is None or self.cached_original_image is None:
                print("No cached images for blending")
                return None

            if strength != 1.0:
                original_dtype = self.cached_processed_image.dtype
                blended = self.cached_processed_image * strength + \
                        self.cached_original_image * (1 - strength)
                if blended.dtype != original_dtype:
                    blended = blended.astype(original_dtype)
            else:
                blended = self.cached_processed_image

            if self.siril.is_image_loaded():
                try:
                    with self.siril.image_lock():
                        if not self.siril.is_cli():
                            self.siril.undo_save_state(f"GraXpert AI denoise: strength {strength:.2f}")
                        self.siril.set_image_pixeldata(blended)
                except s.ImageDialogOpenError:
                    messagebox.showerror("Image dialog open", "An image dialog is open: please close it and try again.")
                    self.siril.reset_progress()
                except s.ProcessingThreadBusyError:
                    messagebox.showerror("Thread busy", "The processing thread is busy. Please wait for it to finish "
                        "and try again.")
                    self.siril.reset_progress()

            self.siril.reset_progress()
            return blended

        except Exception as e:
            print(f"Error in blending: {str(e)}")
            return None

    def denoise(self, image, ai_path, batch_size=4, progress_callback=None,
                ai_gpu_acceleration=True, window_size=256, stride=128):
        """
        Apply AI-based denoising to an image.

        Args:
            image: Input image as numpy array
            ai_path: Path to ONNX model
            batch_size: Number of patches to process at once
            window_size: Size of patch window
            stride: Stride between patches
            progress_callback: Function to call with progress updates
            ai_gpu_acceleration: Whether to use GPU acceleration

        Returns:
            Denoised image as numpy array
        """
        print("Starting denoising")

        # Handle planar format (c, h, w) -> (h, w, c)
        if image.shape[0] < image.shape[1] and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
            planar_format = True
        else:
            planar_format = False

        # Sanitize batch size
        if batch_size < 1:
            print(f"mapping batch_size of {batch_size} to 1")
            batch_size = 1
        elif batch_size > 32:
            print(f"mapping batch_size of {batch_size} to 32")
            batch_size = 32
        elif batch_size & batch_size - 1 != 0:  # check if batch_size is power of two
            print(f"mapping batch_size of {batch_size} to {2 ** (batch_size).bit_length() // 2}")
            batch_size = 2 ** (batch_size).bit_length() // 2  # map batch_size to power of two

        # Calculate median and median absolute deviation (MAD)
        median = np.median(image[::4, ::4, :], axis=[0, 1])
        mad = np.median(np.abs(image[::4, ::4, :] - median), axis=[0, 1])

        # Set model threshold based on version
        if "1.0.0" in ai_path or "1.1.0" in ai_path:
            model_threshold = 1.0
        else:
            model_threshold = 10.0

        # Handle grayscale images
        num_colors = image.shape[-1]
        if num_colors == 1:
            image = np.array([image[:, :, 0], image[:, :, 0], image[:, :, 0]])
            image = np.moveaxis(image, 0, -1)

        H, W, _ = image.shape
        offset = int((window_size - stride) / 2)

        # Calculate number of patches
        h, w, _ = image.shape
        ith = int(h / stride) + 1
        itw = int(w / stride) + 1

        # Pad image
        dh = ith * stride - h
        dw = itw * stride - w
        image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
        image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)

        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset):, :, :]), axis=0)
        image = np.concatenate((image[:offset, :, :], image), axis=0)
        image = np.concatenate((image, image[:, (w - offset):, :]), axis=1)
        image = np.concatenate((image[:, :offset, :], image), axis=1)

        output = copy.deepcopy(image)

        # Initialize ONNX runtime session
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)
            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")
                # Retry with CPU only
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                except ONNXRuntimeError as err:
                    messagebox.showerror("Error", "Cannot build an inference model on this device")
                    return

        print(f"Using inference providers: {session.get_providers()}")

        # Process image in batches
        cancel_flag = False
        last_progress = 0

        for b in range(0, ith * itw + batch_size, batch_size):
            if cancel_flag:
                print("Denoising cancelled")
                return None

            input_tiles = []
            input_tile_copies = []
            for t_idx in range(0, batch_size):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j

                tile = image[x:x + window_size, y:y + window_size, :]
                tile = (tile - median) / mad * 0.04
                input_tile_copies.append(np.copy(tile))
                tile = np.clip(tile, -model_threshold, model_threshold)

                input_tiles.append(tile)

            if not input_tiles:
                continue

            input_tiles = np.array(input_tiles)

            # Run inference
            output_tiles = []

            session_result, session = onnx_helper.run(session, ai_path, None, \
                        {"gen_input_image": input_tiles}, return_first_output=True)

            for e in session_result:
                output_tiles.append(e)

            output_tiles = np.array(output_tiles)

            # Place denoised tiles back into output image
            for t_idx, tile in enumerate(output_tiles):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j
                tile = np.where(input_tile_copies[t_idx] < model_threshold, tile, \
                                input_tile_copies[t_idx])
                tile = tile / 0.04 * mad + median
                tile = tile[offset:offset + stride, offset:offset + stride, :]
                output[x + offset:stride * (i + 1) + offset, y + offset:stride * \
                                           (j + 1) + offset, :] = tile

            # Update progress
            p = int(b / (ith * itw + batch_size) * 100)
            if p > last_progress:
                if progress_callback is not None:
                    progress_callback(f"Processing: {p}%", p/100)
                else:
                    print(f"Progress: {p}%")
                last_progress = p

        # Crop output back to original size
        output = output[offset:H + offset, offset:W + offset, :]

        # Handle grayscale output
        if num_colors == 1:
            output = np.array([output[:, :, 0]])
            output = np.moveaxis(output, 0, -1)

        # Convert back to planar format if needed
        if planar_format:
            output = np.transpose(output, (2, 0, 1))

        # Cache denoised image for future blending
        self.cached_processed_image = output

        print("Finished denoising")
        return output

    def process_image(self, model_path, strength=1.0, batch_size=4,
                    gpu_acceleration=True, progress_callback=None):
        """
        Process an image with denoising and blending.

        Args:
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Processed image as numpy array
        """
        try:
            if progress_callback:
                progress_callback("Fetching image data...")

            # Get original shape and data format
            original_shape = self.cached_original_image.shape
            is_planar = len(original_shape) == 3 and original_shape[0] <= 4

            # Normalize if pixel values exceed [0,1]
            pixel_data = self.cached_original_image
            original_dtype = pixel_data.dtype
            if original_dtype == np.uint16:
                pixel_data = pixel_data.astype(np.float32) / 65535.0

            # Process the image
            if progress_callback:
                progress_callback("Starting denoising process...")

            denoised = self.denoise(
                pixel_data,
                model_path,
                batch_size,
                progress_callback,
                gpu_acceleration
            )

            if denoised is None:
                if progress_callback:
                    progress_callback("Processing cancelled.")
                return None

            # Scale back if normalized
            if original_dtype == np.uint16:
                denoised = denoised * 65535.0
                denoised = denoised.astype(np.uint16)

            # Ensure the output has the same shape as input
            if denoised.shape != original_shape:
                if progress_callback:
                    progress_callback("Reshaping output to match input dimensions...")
                if len(original_shape) == 2:
                    # Handle special case for grayscale
                    if len(denoised.shape) == 3:
                        denoised = denoised[:, :, 0]
                elif is_planar and len(denoised.shape) == 3 and denoised.shape[2] <= 4:
                    # If input was planar (c,h,w) and output is (h,w,c)
                    denoised = np.transpose(denoised, (2, 0, 1))

            # Cache the denoised image for future blending
            self.cached_processed_image = denoised

            # Apply blending
            denoised = self.apply_blend(strength)

            if progress_callback:
                progress_callback("Processing complete.", 1.0)
            return denoised

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_sequence(self, sequence_name, model_path, strength=1.0, batch_size=4,
                       gpu_acceleration=True,
                       progress_callback=None):
        """
        Process a sequence with denoising and blending.

        Args:
            sequence_name: Name of the sequence to process
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Processing sequence {sequence_name}...")

            if not self.siril.is_sequence_loaded():
                # Try to load the sequence
                try:
                    self.siril.cmd("load_seq", f"\"{sequence_name}\"")
                except SirilError:
                    if progress_callback:
                        progress_callback(f"Failed to load sequence {sequence_name}")
                    return False

            sequence = self.siril.get_seq()
            input_seqname = sequence.seqname
            output_seqname = 'denoised_' + input_seqname

            # Get files to process
            files = [self.siril.get_seq_frame_filename(i) for i in range(sequence.number) \
                    if sequence.imgparam[i].incl]

            total_files = len(files)
            if total_files == 0:
                if progress_callback:
                    progress_callback("No files to process in sequence")
                return False

            for i, f in enumerate(files):
                # Reset the cached image
                self.reset_cache()

                file_progress = i / total_files
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}", file_progress)

                # Get the pixel data and FITS header
                self.cached_original_image, header = get_image_data_from_file(self.siril, f)
                if self.cached_original_image is None:
                    print(f"Error loading file {f}, skipping this file...")
                    continue

                # Reshape mono images to 3D with a channels size of 1
                if self.cached_original_image.ndim == 2:
                    self.cached_original_image = self.cached_original_image[np.newaxis, ...]

                # Define a callback to maintain overall progress
                # pylint: disable=cell-var-from-loop
                def file_progress_callback(msg, p = None):
                    if progress_callback:
                        if p:
                            overall_progress = file_progress + p / total_files
                            progress_callback(f"File {i+1}/{total_files}: {msg}", overall_progress)
                        else:
                            progress_callback(msg)

                # Process the image
                denoised = self.process_image(
                    model_path,
                    strength,
                    batch_size,
                    gpu_acceleration,
                    file_progress_callback
                )

                if denoised is None:
                    continue

                # Save the processed image
                output_path = os.path.join(self.siril.get_siril_wd(),
                                           f"{output_seqname}{(i+1):05d}.fit")
                print(f"Saving frame as {output_path}")
                save_fits(denoised, output_path, original_header=header,
                            history_text=f"GraXpert denoise (strength {strength:.2f})")

            # Create the new sequence
            self.siril.create_new_seq(output_seqname)

            # On completion, load the new sequence
            self.siril.cmd("load_seq", f"\"{output_seqname}\"")

            if progress_callback:
                progress_callback(f"Sequence processing complete: {output_seqname}", 1.0)
                sleep(1)
                self.siril.reset_progress()

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return False

class DeconvolutionProcessing:
    """
    Class encapsulating the core image processing functionality
    for GraXpert AI Deconvolution.
    """

    def __init__(self, siril):
        """
        Initialize the processing class.

        Args:
            siril: SirilInterface instance, which must already be connected.
        """
        if not siril:
            raise ValueError("No SirilInterface provided to DeconvolutionProcessing()")

        self.siril = siril

        # Cache for processed images
        self.cached_processed_image = None
        self.cached_original_image = None

        self.config_dir = self.siril.get_siril_configdir() if self.siril else None

    def reset_cache(self):
        """Reset the cached deconvolved image."""
        self.cached_processed_image = None
        self.cached_original_image = None

    def check_config_file(self, operation):
        """
        Check for a saved model path in the configuration file.
        Returns model_path or None if not found.
        """
        if not self.config_dir:
            return None

        config_file_path = None
        if operation == "deconvolution-stars":
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_STARS_CONFIG_FILENAME)
        else:
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_OBJECTS_CONFIG_FILENAME)

        model_path = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
        return model_path

    def save_config_file(self, model_path):
        """
        Save the selected model path to the configuration file.
        """
        if not self.config_dir:
            return

        if 'stars' in model_path:
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_STARS_CONFIG_FILENAME)
        else:
            config_file_path = os.path.join(self.config_dir, DECONVOLVE_OBJECTS_CONFIG_FILENAME)

        try:
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(model_path + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def deconvolve(self, image, ai_path, strength, psfsize, batch_size=4,
                  window_size=512, stride=448, progress_callback=None, ai_gpu_acceleration=True):
        """
        Apply AI-based deconvolution to an image.

        Args:
            image: Input image as numpy array
            ai_path: Path to ONNX model
            strength: Deconvolution strength (0-1)
            psfsize: Size of the PSF (Point Spread Function)
            batch_size: Number of patches to process at once
            window_size: Size of patch window
            stride: Stride between patches
            progress_callback: Function to call with progress updates
            ai_gpu_acceleration: Whether to use GPU acceleration

        Returns:
            Deconvolved image as numpy array
        """
        print("Starting deconvolution")

        # Handle planar format (c, h, w) -> (h, w, c)
        if image.shape[0] < image.shape[1] and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
            planar_format = True
        else:
            planar_format = False

        # Determine deconvolution type based on model path
        if "stars" in ai_path:
            deconv_type = "Stellar"
        elif "obj" in ai_path:
            deconv_type = "Obj"
        else:
            deconv_type = "Unknown"
            print(f"Unknown model type: {ai_path}, assuming Object type")

        # Adjust strength (as per original function)
        strength = 0.95 * strength  # TODO: strength of exactly 1.0 brings no results, to fix

        # Normalize PSF size according to the model type
        if deconv_type == "Stellar":
            psfsize = np.clip((psfsize / 2.355 - 1.5) / 3.0, 0.05, 0.95)  # Stellar
        else:
            if "1.0.0" in ai_path:
                psfsize = np.clip((psfsize / 2.355 - 1.0) / 5.0, 0.05, 0.95)  # Object v1.0.0
            else:
                psfsize = np.clip((psfsize / 2.355 - 0.5) / 5.5, 0.05, 0.95)  # Object v1.0.1

        print(f"Calculated normalized PSFsize value: {psfsize}")

        # Sanitize batch size
        if batch_size < 1:
            print(f"mapping batch_size of {batch_size} to 1")
            batch_size = 1
        elif batch_size > 32:
            print(f"mapping batch_size of {batch_size} to 32")
            batch_size = 32
        elif not (batch_size & (batch_size - 1) == 0):  # check if batch_size is power of two
            print(f"mapping batch_size of {batch_size} to {2 ** (batch_size).bit_length() // 2}")
            batch_size = 2 ** (batch_size).bit_length() // 2  # map batch_size to power of two

        # Adjust batch size for RGB images
        if batch_size >= 4 and image.shape[-1] == 3:
            batch_size = batch_size // 4

        num_colors = image.shape[-1]

        H, W, _ = image.shape
        offset = int((window_size - stride) / 2)

        # Calculate padding and pad the image
        h, w, _ = image.shape
        ith = int(h / stride) + 1
        itw = int(w / stride) + 1

        dh = ith * stride - h
        dw = itw * stride - w

        image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
        image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)

        h, w, _ = image.shape
        image = np.concatenate((image, image[(h - offset):, :, :]), axis=0)
        image = np.concatenate((image[:offset, :, :], image), axis=0)
        image = np.concatenate((image, image[:, (w - offset):, :]), axis=1)
        image = np.concatenate((image[:, :offset, :], image), axis=1)

        output = copy.deepcopy(image)

        # Initialize ONNX runtime session
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)
            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")
                # Retry with CPU only
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                except ONNXRuntimeError as err:
                    messagebox.showerror("Error", "Cannot build an inference model on this device")
                    return

        print(f"Using inference providers: {session.get_providers()}")

        # Process image in batches
        cancel_flag = False
        last_progress = 0

        for b in range(0, ith * itw + batch_size, batch_size):
            if cancel_flag:
                print("Deconvolution cancelled")
                return None

            input_tiles = []
            input_tile_copies = []
            params = []
            for t_idx in range(0, batch_size):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j

                tile = image[x:x + window_size, y:y + window_size, :]

                # Logarithmic preprocessing
                _min = np.min(tile, axis=(0, 1))
                tile = tile - _min + 1e-5
                tile = np.log(tile)

                _mean = tile.mean()
                _std = tile.std()
                _mean, _std = _mean.astype(np.float32), _std.astype(np.float32)
                tile = (tile - _mean) / _std * 0.1
                params.append([_mean, _std, _min])

                input_tile_copies.append(np.copy(tile))
                input_tiles.append(tile)

            if not input_tiles:
                continue

            input_tiles = np.array(input_tiles)
            input_tiles = np.moveaxis(input_tiles, -1, 1)
            input_tiles = np.reshape(input_tiles, [input_tiles.shape[0] * num_colors, 1, window_size, window_size])

            # Run inference
            output_tiles = []
            sigma = np.full(shape=(input_tiles.shape[0], 1), fill_value=psfsize, dtype=np.float32)
            strenght_p = np.full(shape=(input_tiles.shape[0], 1), fill_value=strength, dtype=np.float32)
            conds = np.concatenate([sigma, strenght_p], axis=-1)

            if deconv_type == "Obj" and "1.0.0" in ai_path:
                session_result, session = onnx_helper.run(session, ai_path, None, \
                            {"gen_input_image": input_tiles, "sigma": sigma, "strenght": strenght_p}, return_first_output=True)

            else:
                session_result, session = onnx_helper.run(session, ai_path, None, \
                            {"gen_input_image": input_tiles, "params": conds}, return_first_output=True)

            for e in session_result:
                output_tiles.append(e)

            output_tiles = np.array(output_tiles)
            output_tiles = input_tiles - output_tiles
            output_tiles = np.reshape(output_tiles, [output_tiles.shape[0] // num_colors, num_colors, window_size, window_size])
            output_tiles = np.moveaxis(output_tiles, 1, -1)

            # Post-process tiles
            for idx in range(len(params)):
                output_tiles[idx] = output_tiles[idx] * params[idx][1] / 0.1 + params[idx][0]
                output_tiles[idx] = np.exp(output_tiles[idx])
                output_tiles[idx] = output_tiles[idx] + params[idx][2] - 1e-5

            # Place deconvolved tiles back into output image
            for t_idx, tile in enumerate(output_tiles):
                index = b + t_idx
                i = index % ith
                j = index // ith

                if i >= ith or j >= itw:
                    break

                x = stride * i
                y = stride * j
                tile = tile[offset:offset + stride, offset:offset + stride, :]
                output[x + offset:stride * (i + 1) + offset, y + offset:stride * (j + 1) + offset, :] = tile

            # Update progress
            p = int(b / (ith * itw + batch_size) * 100)
            if p > last_progress:
                if progress_callback is not None:
                    progress_callback(f"Processing: {p}%", p/100)
                else:
                    print(f"Progress: {p}%")
                last_progress = p

        # Crop output back to original size and clip values
        output = output[offset:H + offset, offset:W + offset, :]
        output = np.clip(output, 0.0, 1.0)

        # Convert back to planar format if needed
        if planar_format:
            output = np.transpose(output, (2, 0, 1))

        # Cache deconvolved image for future blending
        self.cached_processed_image = output

        print("Finished deconvolution")
        return output

    def process_image(self, model_path, strength=1.0, psfsize=2.5, batch_size=4,
                     gpu_acceleration=True, progress_callback=None):
        """
        Process an image with deconvolution and blending.

        Args:
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            psfsize: Size of the PSF (Point Spread Function)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Processed image as numpy array
        """
        try:
            if progress_callback:
                progress_callback("Fetching image data...")

            # Get original shape and data format
            original_shape = self.cached_original_image.shape
            is_planar = len(original_shape) == 3 and original_shape[0] <= 4

            # Normalize if pixel values exceed [0,1]
            pixel_data = self.cached_original_image
            original_dtype = pixel_data.dtype
            if original_dtype == np.uint16:
                pixel_data = pixel_data.astype(np.float32) / 65535.0

            # Process the image
            if progress_callback:
                progress_callback("Starting deconvolution process...")

            deconvolved = self.deconvolve(
                pixel_data,
                model_path,
                strength,
                psfsize,
                batch_size,
                progress_callback=progress_callback,
                ai_gpu_acceleration=gpu_acceleration
            )

            if deconvolved is None:
                if progress_callback:
                    progress_callback("Processing cancelled.")
                return None

            # Scale back if normalized
            if original_dtype == np.uint16:
                deconvolved = deconvolved * 65535.0
                deconvolved = deconvolved.astype(np.uint16)

            # Ensure the output has the same shape as input
            if deconvolved.shape != original_shape:
                if progress_callback:
                    progress_callback("Reshaping output to match input dimensions...")
                if len(original_shape) == 2:
                    # Handle special case for grayscale
                    if len(deconvolved.shape) == 3:
                        deconvolved = deconvolved[:, :, 0]
                elif is_planar and len(deconvolved.shape) == 3 and deconvolved.shape[2] <= 4:
                    # If input was planar (c,h,w) and output is (h,w,c)
                    deconvolved = np.transpose(deconvolved, (2, 0, 1))

            # Cache the deconvolved image for future blending
            self.cached_processed_image = deconvolved

            # Update image
            if self.siril.is_image_loaded():
                try:
                    with self.siril.image_lock():
                        if not self.siril.is_cli():
                            self.siril.undo_save_state(f"GraXpert AI deconvolve: strength {strength:.2f}")
                        self.siril.set_image_pixeldata(deconvolved)
                except s.ImageDialogOpenError:
                    messagebox.showerror("Image dialog open", "An image dialog is open: please close it and try again.")
                    self.siril.reset_progress()
                except s.ProcessingThreadBusyError:
                    messagebox.showerror("Thread busy", "The processing thread is busy. Please wait for it to finish "
                        "and try again.")
                    self.siril.reset_progress()

            if progress_callback:
                progress_callback("Processing complete.", 1.0)
            return deconvolved

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_sequence(self, sequence_name, model_path, strength=1.0, psfsize=2.5, batch_size=4,
                       gpu_acceleration=True,
                       progress_callback=None):
        """
        Process a sequence with deconvolution and blending.

        Args:
            sequence_name: Name of the sequence to process
            model_path: Path to the ONNX model
            strength: Blending strength (0-1)
            psfsize: Size of the PSF (Point Spread Function)
            batch_size: Number of patches to process at once
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Processing sequence {sequence_name}...")

            if not self.siril.is_sequence_loaded():
                # Try to load the sequence
                try:
                    self.siril.cmd("load_seq", f"\"{sequence_name}\"")
                except SirilError:
                    if progress_callback:
                        progress_callback(f"Failed to load sequence {sequence_name}")
                    return False

            sequence = self.siril.get_seq()
            input_seqname = sequence.seqname
            print(model_path)
            output_seqname = 'deconv_obj_' + input_seqname \
                if 'deconvolution-object' in model_path else \
                'deconv_stellar_' + input_seqname

            # Get files to process
            files = [self.siril.get_seq_frame_filename(i) for i in range(sequence.number) \
                    if sequence.imgparam[i].incl]

            total_files = len(files)
            if total_files == 0:
                if progress_callback:
                    progress_callback("No files to process in sequence")
                return False

            for i, f in enumerate(files):
                # Reset the cached image
                self.reset_cache()

                file_progress = i / total_files
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}", file_progress)

                # Get the pixel data and FITS header
                self.cached_original_image, header = get_image_data_from_file(self.siril, f)  # Assuming this function exists
                if self.cached_original_image is None:
                    print(f"Error loading file {f}, skipping this file...")
                    continue

                # Reshape mono images to 3D with a channels size of 1
                if self.cached_original_image.ndim == 2:
                    self.cached_original_image = self.cached_original_image[np.newaxis, ...]

                # Define a callback to maintain overall progress
                def file_progress_callback(msg, p=None):
                    if progress_callback:
                        if p:
                            overall_progress = file_progress + p / total_files
                            progress_callback(f"File {i+1}/{total_files}: {msg}", overall_progress)
                        else:
                            progress_callback(msg)

                # Process the image
                deconvolved = self.process_image(
                    model_path,
                    strength,
                    psfsize,
                    batch_size,
                    gpu_acceleration,
                    file_progress_callback
                )

                if deconvolved is None:
                    continue

                # Save the processed image
                output_path = os.path.join(self.siril.get_siril_wd(),
                                           f"{output_seqname}{(i+1):05d}.fit")
                print(f"Saving frame as {output_path}")
                save_fits(deconvolved, output_path, original_header=header,
                            history_text=f"GraXpert deconvolve (strength {strength:.2f}, psfsize {psfsize:.2f})")

            # Create the new sequence
            self.siril.create_new_seq(output_seqname)

            # On completion, load the new sequence
            self.siril.cmd("load_seq", f"\"{output_seqname}\"")

            if progress_callback:
                progress_callback(f"Sequence processing complete: {output_seqname}", 1.0)
                sleep(1)
                self.siril.reset_progress()

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return False

class BGEProcessing:
    """Class encapsulating the core image processing functionality for GraXpert Background Extraction."""

    def __init__(self, siril):
        """
        Initialize the processing class.

        Args:
            siril: SirilInterface instance, which must already be connected.
        """
        if not siril:
            raise ValueError("No SirilInterface provided to BGEProcessing()")

        self.siril = siril

        # Cache for processed images
        self.cached_background_image = None
        self.cached_original_image = None
        self.cached_processed_image = None

        self.config_dir = self.siril.get_siril_configdir() if self.siril else None

    def reset_cache(self):
        """Reset the cached images."""
        self.cached_background_image = None
        self.cached_original_image = None
        self.cached_processed_image = None

    def check_config_file(self):
        """
        Check for a saved model path in the configuration file.
        Returns model_path or None if not found.
        """
        if not self.config_dir:
            return None

        config_file_path = os.path.join(self.config_dir, BGE_CONFIG_FILENAME)
        model_path = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    model_path = lines[0].strip()
                    if not os.path.isfile(model_path):
                        model_path = None
        return model_path

    def save_config_file(self, model_path):
        """
        Save the selected model path to the configuration file.
        """
        if not self.config_dir:
            return

        config_file_path = os.path.join(self.config_dir, BGE_CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(model_path + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

    def gaussian_kernel(self, sigma):
        """Calculate appropriate kernel size for Gaussian blur based on sigma"""
        size = int(8.0 * sigma + 1.0)
        if size % 2 == 0:
            size += 1
        return (size, size)

    def apply_correction(self, image, background, correction_type):
        """
        Apply correction using the cached background image.

        Args:
            correction_type: Type of correction ('subtraction' or 'division')

        Returns:
            Corrected image as numpy array
        """
        try:
            if image is None or image is None:
                print("No image for correction")
                return None

            # Create a copy of the original image to work with
            corrected = copy.deepcopy(image)

            # Apply the correction based on the selected type
            if correction_type == "subtraction":
                mean = np.mean(background)
                corrected = corrected - background + mean
            elif correction_type == "division":
                # Handle each channel separately for division
                num_colors = 3 if len(corrected.shape) > 2 else 1
                if num_colors == 1 and len(corrected.shape) == 2:
                    # Handle grayscale as a special case
                    mean = np.mean(corrected)
                    corrected = corrected / background * mean
                else:
                    for c in range(num_colors):
                        mean = np.mean(corrected[c, :, :])
                        corrected[c, :, :] = corrected[c, :, :] / background[c, :, :] * mean

            # Clip the result to valid range
            corrected = np.clip(corrected, 0.0, 1.0)

            # Cache the corrected image
            self.cached_processed_image = corrected
            self.siril.reset_progress()
            return corrected

        except Exception as e:
            print(f"Error in correction: {str(e)}")
            return None

    def extract_background_ai(self, image, ai_path, smoothing=0,
                              ai_gpu_acceleration=True, progress_callback=None):
        """
        Apply AI-based background extraction to an image.

        Args:
            image: Input image as numpy array
            ai_path: Path to ONNX model
            smoothing: Amount of smoothing to apply (0-1)
            ai_gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Background image as numpy array
        """
        print("Starting background extraction")

        # Handle different image formats
        was_mono = False
        if len(image.shape) == 2:
            # Handle grayscale image
            was_mono = True
            image = np.expand_dims(image, -1)
        # Convert to hwc format if needed:
        was_planar = False
        if image.shape[0] < 4 and len(image.shape) == 3 and image.shape[0] < image.shape[1] \
                              and image.shape[0] < image.shape[2]:
            was_planar = True
            image = np.transpose(image, (1, 2, 0))

        # Store original shape for later reshaping
        original_shape = image.shape
        num_colors = image.shape[-1]
        if num_colors == 1:
            was_mono = True
        # Shrink and pad to avoid artifacts on borders
        padding = 8
        if progress_callback:
            progress_callback("Preparing image...", 0.05)
        # Resize to a standard size for the AI model
        imarray_shrink = cv2.resize(image, dsize=(256 - 2*padding, 256 - 2*padding),
                                    interpolation=cv2.INTER_LINEAR)

        if len(imarray_shrink.shape) == 2:
            imarray_shrink = np.expand_dims(imarray_shrink, -1)
        # Pad the image to avoid edge artifacts
        imarray_shrink = np.pad(imarray_shrink, ((padding, padding), (padding, padding), (0, 0)),
                               mode="edge")
        if progress_callback:
            progress_callback("Computing image statistics...", 0.1)

        # Calculate median and median absolute deviation for each channel
        median = []
        mad = []
        for c in range(num_colors):
            median.append(np.median(imarray_shrink[:, :, c]))
            mad.append(np.median(np.abs(imarray_shrink[:, :, c] - median[c])))
        if progress_callback:
            progress_callback("Normalizing image...", 0.15)

        # Normalize the image for the AI model
        imarray_shrink = (imarray_shrink - median) / mad * 0.04
        imarray_shrink = np.clip(imarray_shrink, -1.0, 1.0)
        # For grayscale, convert to RGB for the AI model
        if num_colors == 1:
            imarray_shrink = np.array([imarray_shrink[:, :, 0],
                                       imarray_shrink[:, :, 0],
                                       imarray_shrink[:, :, 0]])
            imarray_shrink = np.moveaxis(imarray_shrink, 0, -1)

        if progress_callback:
            progress_callback("Initializing ONNX runtime...", 0.25)

        # Initialize ONNX runtime session
        with s.SuppressedStderr():
            providers = onnx_helper.get_execution_providers_ordered(ai_gpu_acceleration)

            try:
                session = onnxruntime.InferenceSession(ai_path, providers=providers)
            except Exception as err:
                error_message = str(err)
                print("Warning: falling back to CPU.")
                if "cudaErrorNoKernelImageForDevice" in error_message \
                    or "Error compiling model" in error_message:
                    print("ONNX cannot build an inferencing kernel for this GPU.")
                # Retry with CPU only
                providers = ['CPUExecutionProvider']
                try:
                    session = onnxruntime.InferenceSession(ai_path, providers=providers)
                except ONNXRuntimeError as err:
                    messagebox.showerror("Error", "Cannot build an inference model on this device")
                    return

        print(f"Using inference providers: {session.get_providers()}")

        if progress_callback:
            progress_callback("Running inference...", 0.4)

        # Run inference
        background, session = onnx_helper.run(session, ai_path, None, \
                    {"gen_input_image": np.expand_dims(imarray_shrink, axis=0)})
        background = background[0][0]

        if progress_callback:
            progress_callback("Post-processing...", 0.6)

        # Denormalize the background
        background = background / 0.04 * mad + median

        # Apply smoothing if requested
        if smoothing != 0:
            sigma = smoothing * 20
            kernel = self.gaussian_kernel(sigma)
            background = cv2.GaussianBlur(background, ksize=kernel, sigmaX=sigma, sigmaY=sigma)

        if progress_callback:
            progress_callback("Finalizing background...", 0.8)

        # Remove padding
        if padding != 0:
            background = background[padding:-padding, padding:-padding, :]

        # Apply additional smoothing for better results
        sigma = 3.0
        kernel = self.gaussian_kernel(sigma)
        background = cv2.GaussianBlur(background, ksize=kernel, sigmaX=sigma, sigmaY=sigma)

        # Resize back to original dimensions
        background = cv2.resize(background, dsize=(original_shape[1], original_shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        if was_planar:
            background = np.transpose(background, (2, 0, 1))

        # Ensure output has the same shape as input
        if len(background.shape) == 2 and len(original_shape) == 3:
            background = np.expand_dims(background, -1)
        elif was_mono and len(background.shape) == 3:
            background = background[0, :, :]

        # Cache the extracted background
        self.cached_background_image = background

        if progress_callback:
            progress_callback("Background extraction completed", 1.0)

        print("Finished background extraction")
        return background

    def process_image(self, model_path, correction_type="subtraction", smoothing=0,
                    keep_bg=False, filename=None, header=None, gpu_acceleration=True,
                    progress_callback=None):
        """
        Process an image with background extraction and correction.

        Args:
            model_path: Path to the ONNX model
            correction_type: Type of correction ('subtraction' or 'division')
            smoothing: Amount of smoothing to apply (0-1)
            keep_bg: whether to save the extracted background
            filename: filename (for use with keep_bg)
            header: original header (for use with keep_bg)
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            Processed image as numpy array
        """
        try:
            if progress_callback:
                progress_callback("Fetching image data...")

            # Ensure we have an original image
            if self.cached_original_image is None:
                if progress_callback:
                    progress_callback("No image loaded.")
                return None

            # Get original shape and data format
            original_shape = self.cached_original_image.shape
            original_dtype = self.cached_original_image.dtype

            # Normalize if needed
            pixel_data = self.cached_original_image
            if original_dtype == np.uint16:
                pixel_data = pixel_data.astype(np.float32) / 65535.0

            # Process the image
            if progress_callback:
                progress_callback("Starting background extraction...")

            # Extract the background
            background = self.extract_background_ai(
                pixel_data,
                model_path,
                smoothing,
                gpu_acceleration,
                progress_callback
            )

            if background is None:
                if progress_callback:
                    progress_callback("Processing cancelled.")
                return None

            # Apply correction
            if progress_callback:
                progress_callback("Applying correction...")

            corrected = self.apply_correction(pixel_data, background, correction_type)

            if corrected is None:
                if progress_callback:
                    progress_callback("Correction failed.")
                return None

            # Scale back if needed
            if original_dtype == np.uint16:
                corrected = corrected * 65535.0
                corrected = corrected.astype(np.uint16)
                if self.cached_background_image is not None:
                    self.cached_background_image = self.cached_background_image * 65535.0
                    self.cached_background_image = self.cached_background_image.astype(np.uint16)

            if keep_bg:
                extension = self.siril.get_siril_config("core", "extension")
                output_path = os.path.splitext(filename)[0] + "_bg" + extension
                print(f"Saving background as {output_path}")
                save_fits(self.cached_background_image, path=output_path, original_header=header,
                          history_text="Extracted background")

            # Update the display if an image is loaded
            if self.siril.is_image_loaded():
                try:
                    with self.siril.image_lock():
                        if not self.siril.is_cli():
                            self.siril.undo_save_state(f"GraXpert AI BGE: {correction_type}")
                        self.siril.set_image_pixeldata(corrected)
                except s.ImageDialogOpenError:
                    messagebox.showerror("Image dialog open", "An image dialog is open: please close it and try again.")
                    self.siril.reset_progress()
                except s.ProcessingThreadBusyError:
                    messagebox.showerror("Thread busy", "The processing thread is busy. Please wait for it to finish "
                        "and try again.")
                    self.siril.reset_progress()

            if progress_callback:
                progress_callback("Processing complete.", 1.0)
            return corrected

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_sequence(self, sequence_name, model_path, correction_type="subtraction",
                       smoothing=0.5, keep_bg=False, gpu_acceleration=True,
                       progress_callback=None):
        """
        Process a sequence with background extraction and correction.

        Args:
            sequence_name: Name of the sequence to process
            model_path: Path to the ONNX model
            correction_type: Type of correction ('subtraction' or 'division')
            smoothing: Amount of smoothing to apply (0-1)
            gpu_acceleration: Whether to use GPU acceleration
            progress_callback: Function to call with progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Processing sequence {sequence_name}...")

            if not self.siril.is_sequence_loaded():
                # Try to load the sequence
                try:
                    self.siril.cmd("load_seq", f"\"{sequence_name}\"")
                except SirilError:
                    if progress_callback:
                        progress_callback(f"Failed to load sequence {sequence_name}")
                    return False

            sequence = self.siril.get_seq()
            input_seqname = sequence.seqname
            output_seqname = 'bge_' + input_seqname

            # Get files to process
            files = [self.siril.get_seq_frame_filename(i) for i in range(sequence.number)
                    if sequence.imgparam[i].incl]

            total_files = len(files)
            if total_files == 0:
                if progress_callback:
                    progress_callback("No files to process in sequence")
                return False

            for i, f in enumerate(files):
                # Reset the cached image
                self.reset_cache()

                file_progress = i / total_files
                if progress_callback:
                    progress_callback(f"Processing file {i+1} of {total_files}", file_progress)

                # Get the pixel data and FITS header
                self.cached_original_image, header = get_image_data_from_file(self.siril, f)
                if self.cached_original_image is None:
                    print(f"Error loading file {f}, skipping this file...")
                    continue

                # Define a callback to maintain overall progress
                def file_progress_callback(msg, p=None):
                    if progress_callback:
                        if p is not None:
                            overall_progress = file_progress + p / total_files
                            progress_callback(f"File {i+1}/{total_files}: {msg}", overall_progress)
                        else:
                            progress_callback(msg)

                output_path = os.path.join(self.siril.get_siril_wd(),
                                          f"{output_seqname}{(i+1):05d}.fit")
                header = self.siril.get_seq_frame_header(i)

                # Process the image
                corrected = self.process_image(
                    model_path,
                    correction_type,
                    smoothing,
                    keep_bg,
                    output_path,
                    header,
                    gpu_acceleration,
                    file_progress_callback
                )

                if corrected is None:
                    continue

                # Save the processed image
                print(f"Saving frame as {output_path}")
                save_fits(corrected, output_path, original_header=header,
                             history_text=f"GraXpert BGE ({correction_type})")

            # Create the new sequence
            self.siril.create_new_seq(output_seqname)

            # On completion, load the new sequence
            self.siril.cmd("load_seq", f"\"{output_seqname}\"")

            if progress_callback:
                progress_callback(f"Sequence processing complete: {output_seqname}", 1.0)
                self.siril.reset_progress()

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            print(f"Error: {str(e)}")
            return False

class DenoiserCLI:
    """CLI interface for GraXpert AI Denoise."""

    def __init__(self, siril, args=None):
        """ init method """
        # Parse command line arguments
        if not siril:
            raise ValueError("No SirilInterface provided to DenoiserCLI()")

        self.siril = siril
        if args is None:
            args = sys.argv[1:]

        self.args = self.parse_arguments(args)
        if self.args.listmodels:
            models_dir = os.path.join(user_data_dir(appname="GraXpert"), "denoise-ai-models")
            list_available_models(models_dir)
            return

        # Initialize the processing class
        self.processor = DenoiserProcessing(self.siril)

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if not (image_loaded or seq_loaded):
            self.error("No image or sequence loaded")
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            self.error("Error: this script requires Siril 1.4.0-beta2 or higher")
            return

        # Find and set the model path based on arguments
        self.model_path = self.get_model_path()
        if not self.model_path:
            self.error("No valid model found")
            return

        # Start processing
        if image_loaded:
            self.process_image()
        else:
            self.process_sequence()

    def parse_arguments(self, args):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"GraXpert AI Denoise - Siril CLI v{VERSION}")
        parser.add_argument("-strength", type=float, default=0.5, help="Denoising strength (0.0-1.0)")
        parser.add_argument("-batch", type=int, default=4, help="Batch size for processing")
        parser.add_argument("-model", type=str, help="Model name to use (directory name in GraXpert models folder)")

        # Boolean flag for GPU usage - store_true/store_false approach
        parser.add_argument("-gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
        parser.add_argument("-nogpu", action="store_true", default=False, help="Disable GPU acceleration")
        # List models flag
        parser.add_argument("-listmodels", action="store_true", help="List available models and exit")

        return parser.parse_args(args)

    def get_model_path(self):
        """Get the model path based on user choice or highest available version."""
        # Get the GraXpert directory
        models_dir = os.path.join(user_data_dir(appname="GraXpert"), "denoise-ai-models")

        # Check if directory exists
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            self.error(f"Models directory not found: {models_dir}")
            return None

        # Find all available models
        available_models = {}
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    available_models[subdir] = model_path

        if not available_models:
            self.error("No models found")
            return None

        # If model specified, use it if available
        if self.args.model:
            if self.args.model in available_models:
                model_path = available_models[self.args.model]
                print(f"Using specified model: {self.args.model}")
                return model_path
            self.error(f"Specified model '{self.args.model}' not found. Available models: {', '.join(available_models.keys())}")
            return None

        # Otherwise use the highest available version
        model_names = sorted(available_models.keys())
        highest_model = model_names[-1]
        print(f"Using highest available model: {highest_model}")

        # Save the selected model to config
        self.processor.save_config_file(available_models[highest_model])

        return available_models[highest_model]

    def process_image(self):
        """Process a single image."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        print(f"Processing image with strength={self.args.strength}, batch={self.args.batch}, gpu={use_gpu}")

        # Cache the original image
        self.processor.cached_original_image = self.siril.get_image_pixeldata()

        # Reshape mono images to 3D with a channels size of 1
        if self.processor.cached_original_image.ndim == 2:
            self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

        # Start image processing thread
        thread = threading.Thread(
            target=self.processor.process_image,
            args=(
                self.model_path,
                self.args.strength,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Image processing complete")

    def process_sequence(self):
        """Process a sequence."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        sequence_name = self.siril.get_seq().seqname
        print(f"Processing sequence {sequence_name} with strength={self.args.strength}, batch={self.args.batch}, gpu={use_gpu}")

        # Start sequence processing thread
        thread = threading.Thread(
            target=self.processor.process_sequence,
            args=(
                sequence_name,
                self.model_path,
                self.args.strength,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Sequence processing complete")

    def update_progress(self, message, progress=0):
        """Update progress information."""
        self.siril.update_progress(message, progress)

    def error(self, message):
        """Print error message and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        if hasattr(self, 'siril'):
            self.siril.disconnect()
        sys.exit(1)

class DeconvolutionCLI:
    """CLI interface for GraXpert AI Deconvolution."""

    def __init__(self, siril, deconv_type, args=None):
        """ init method """
        # Parse command line arguments
        if not siril:
            raise ValueError("No SirilInterface provided to DeconvolutionCLI()")

        self.siril = siril
        if args is None:
            args = sys.argv[1:]
        self.deconv_obj = True if deconv_type == "deconv_obj" else False
        self.deconv_stellar = True if deconv_type == "deconv_stellar" else False

        self.args = self.parse_arguments(args)
        self.folder = None
        if self.deconv_obj:
            self.folder = "deconvolution-object-ai-models"
        elif self.deconv_stellar:
            self.folder = "deconvolution-stars-ai-models"
        if self.folder is None:
            self.error("Incorrect argument")
            return

        if self.args.listmodels:
            models_dir = os.path.join(user_data_dir(appname="GraXpert"), self.folder)
            list_available_models(models_dir)
            return

        # Initialize the processing class
        self.processor = DeconvolutionProcessing(self.siril)

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if not (image_loaded or seq_loaded):
            self.error("No image or sequence loaded")
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            self.error("Error: this script requires Siril 1.4.0-beta2 or higher")
            return

        # Find and set the model path based on arguments
        self.model_path = self.get_model_path()
        if not self.model_path:
            self.error("No valid model found")
            return

        # Start processing
        if image_loaded:
            self.process_image()
        else:
            self.process_sequence()

    def parse_arguments(self, args):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"GraXpert AI Deconvolution - Siril CLI v{VERSION}")
        parser.add_argument("-strength", type=float, default=0.5, help="Deconvolution strength (0.0-1.0)")
        parser.add_argument("-psfsize", type=float, default=5.0, help="Point Spread Function size")
        parser.add_argument("-batch", type=int, default=4, help="Batch size for processing")
        parser.add_argument("-model", type=str, help="Model name to use (directory name in GraXpert models folder)")

        # Boolean flag for GPU usage - store_true/store_false approach
        parser.add_argument("-gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
        parser.add_argument("-nogpu", action="store_true", default=False, help="Disable GPU acceleration")
        # List models flag
        parser.add_argument("-listmodels", action="store_true", help="List available models and exit")

        return parser.parse_args(args)

    def get_model_path(self):
        """Get the model path based on user choice or highest available version."""
        # Get the GraXpert directory
        models_dir = os.path.join(user_data_dir(appname="GraXpert"), self.folder)

        # Check if directory exists
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            self.error(f"Models directory not found: {models_dir}")
            return None

        # Find all available models
        available_models = {}
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    available_models[subdir] = model_path

        if not available_models:
            self.error("No models found")
            return None

        # If model specified, use it if available
        if self.args.model:
            if self.args.model in available_models:
                model_path = available_models[self.args.model]
                print(f"Using specified model: {self.args.model}")
                return model_path
            self.error(f"Specified model '{self.args.model}' not found. Available models: {', '.join(available_models.keys())}")
            return None

        # Otherwise use the highest available version
        model_names = sorted(available_models.keys())
        highest_model = model_names[-1]
        print(f"Using highest available model: {highest_model}")

        # Save the selected model to config
        self.processor.save_config_file(available_models[highest_model])

        return available_models[highest_model]

    def process_image(self):
        """Process a single image."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        print(f"Processing image with strength={self.args.strength}, psfsize={self.args.psfsize}, batch={self.args.batch}, gpu={use_gpu}")

        # Cache the original image
        self.processor.cached_original_image = self.siril.get_image_pixeldata()

        # Reshape mono images to 3D with a channels size of 1
        if self.processor.cached_original_image.ndim == 2:
            self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

        # Start image processing thread
        thread = threading.Thread(
            target=self.processor.process_image,
            args=(
                self.model_path,
                self.args.strength,
                self.args.psfsize,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Image processing complete")

    def process_sequence(self):
        """Process a sequence."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        sequence_name = self.siril.get_seq().seqname
        print(f"Processing sequence {sequence_name} with strength={self.args.strength}, psfsize={self.args.psfsize}, batch={self.args.batch}, gpu={use_gpu}")

        # Start sequence processing thread
        thread = threading.Thread(
            target=self.processor.process_sequence,
            args=(
                sequence_name,
                self.model_path,
                self.args.strength,
                self.args.psfsize,
                self.args.batch,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Sequence processing complete")

    def update_progress(self, message, progress=0):
        """Update progress information."""
        self.siril.update_progress(message, progress)

    def error(self, message):
        """Print error message and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        if hasattr(self, 'siril'):
            self.siril.disconnect()
        sys.exit(1)

class BackgroundExtractionCLI:
    """CLI interface for GraXpert AI Denoise."""

    def __init__(self, siril, args=None):
        """ init method """
        # Parse command line arguments
        if not siril:
            raise ValueError("No SirilInterface provided to BackgroundExtractionCLI()")

        self.siril = siril
        if args is None:
            args = sys.argv[1:]

        self.args = self.parse_arguments(args)
        if self.args.listmodels:
            models_dir = os.path.join(user_data_dir(appname="GraXpert"), "bge-ai-models")
            list_available_models(models_dir)
            return

        # Initialize the processing class
        self.processor = BGEProcessing(self.siril)

        # Check if an image or sequence is loaded
        image_loaded = self.siril.is_image_loaded()
        seq_loaded = self.siril.is_sequence_loaded()

        if not (image_loaded or seq_loaded):
            self.error("No image or sequence loaded")
            return

        # Find and set the model path based on arguments
        self.model_path = self.get_model_path()
        if not self.model_path:
            self.error("No valid model found")
            return

        # Start processing
        if image_loaded:
            self.process_image()
        else:
            self.process_sequence()

    def parse_arguments(self, args):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"GraXpert AI BG Extraction - Siril CLI v{VERSION}")
        parser.add_argument("-correction", type=str, default="subtraction", help="Correction type ('subtraction' (default) or 'division')")
        parser.add_argument("-smoothing", type=float, default=0.5, help="Smoothing (0.0-1.0)")
        parser.add_argument("-model", type=str, help="Model name to use (directory name in GraXpert models folder)")
        parser.add_argument("-keep_bg", action="store_true", help="Keep the extracted background")

        # Boolean flag for GPU usage - store_true/store_false approach
        parser.add_argument("-gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
        parser.add_argument("-nogpu", action="store_true", default=False, help="Disable GPU acceleration")
        # List models flag
        parser.add_argument("-listmodels", action="store_true", help="List available models and exit")

        return parser.parse_args(args)

    def get_model_path(self):
        """Get the model path based on user choice or highest available version."""
        # Get the GraXpert directory
        models_dir = os.path.join(user_data_dir(appname="GraXpert"), "bge-ai-models")

        # Check if directory exists
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            self.error(f"Models directory not found: {models_dir}")
            return None

        # Find all available models
        available_models = {}
        for subdir in os.listdir(models_dir):
            subdir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(subdir_path):
                model_path = os.path.join(subdir_path, "model.onnx")
                if os.path.exists(model_path) and os.path.isfile(model_path):
                    available_models[subdir] = model_path

        if not available_models:
            self.error("No models found")
            return None

        # If model specified, use it if available
        if self.args.model:
            if self.args.model in available_models:
                model_path = available_models[self.args.model]
                print(f"Using specified model: {self.args.model}")
                return model_path
            self.error(f"Specified model '{self.args.model}' not found. Available models: {', '.join(available_models.keys())}")
            return None

        # Otherwise use the highest available version
        model_names = sorted(available_models.keys())
        highest_model = model_names[-1]
        print(f"Using latest available model: {highest_model}")

        # Save the selected model to config
        self.processor.save_config_file(available_models[highest_model])

        return available_models[highest_model]

    def process_image(self):
        """Process a single image."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        print(f"Processing image with correction={self.args.correction}, smoothing={self.args.smoothing}, keep_bg={self.args.keep_bg}, gpu={use_gpu}")

        # Cache the original image
        self.processor.cached_original_image = self.siril.get_image_pixeldata()

        # Reshape mono images to 3D with a channels size of 1
        if self.processor.cached_original_image.ndim == 2:
            self.processor.cached_original_image = self.processor.cached_original_image[np.newaxis, ...]

        filename = self.siril.get_image_filename()
        header = self.siril.get_image_fits_header()

        # Start image processing thread
        thread = threading.Thread(
            target=self.processor.process_image,
            args=(
                self.model_path,
                self.args.correction,
                self.args.smoothing,
                self.args.keep_bg,
                filename,
                header,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Image processing complete")

    def process_sequence(self):
        """Process a sequence."""
        # Set GPU usage based on arguments
        use_gpu = self.args.gpu and not self.args.nogpu

        sequence_name = self.siril.get_seq().seqname
        print(f"Processing sequence {sequence_name} with strength={self.args.strength}, batch={self.args.batch}, gpu={use_gpu}")

        # Start sequence processing thread
        thread = threading.Thread(
            target=self.processor.process_sequence,
            args=(
                sequence_name,
                self.model_path,
                self.args.correction,
                self.args.smoothing,
                self.args.keep_bg,
                use_gpu,
                self.update_progress
            ),
            daemon=True
        )
        thread.start()
        thread.join()

        print("Sequence processing complete")

    def update_progress(self, message, progress=0):
        """Update progress information."""
        self.siril.update_progress(message, progress)

    def error(self, message):
        """Print error message and exit."""
        print(f"ERROR: {message}", file=sys.stderr)
        if hasattr(self, 'siril'):
            self.siril.disconnect()
        sys.exit(1)

def main():
    """ main entry point into the script """
    siril = s.SirilInterface()
    try:
        siril.connect()

        if siril.is_cli():
            # Top level argument parser
            parser = argparse.ArgumentParser(
                description=f"GraXpert AI Tools - Siril CLI v{VERSION}",
                add_help=False  # Disable the default help so we can handle it ourselves
            )

            # Add help argument manually
            parser.add_argument('-h', '--help', action='store_true',
                               help='Show this help message and exit')

            # Tool selection arguments
            group = parser.add_mutually_exclusive_group()
            group.add_argument("-denoise", action="store_true", help="Use the AI Denoising tool")
            group.add_argument("-deconv_obj", action="store_true", help="Use the AI Object Deconvolution tool")
            group.add_argument("-deconv_stellar", action="store_true", help="Use the AI Stellar Deconvolution tool")
            group.add_argument("-bge", action="store_true", help="Use the AI Background Extraction tool")

            # First parse, just to get the tool type (parse_known_args ignores unknown args)
            args, remaining_args = parser.parse_known_args()

            # If general help is requested, print it and exit
            if not (args.denoise or args.bge or args.deconv_obj or args.deconv_stellar):
                parser.print_help()
                sys.exit(0)

            # Determine which tool to use and handle tool-specific help
            if args.denoise:
                # If help is requested for the denoiser
                if args.help:
                    # Create parser just to show help
                    denoise_parser = argparse.ArgumentParser(
                        description=f"GraXpert AI Denoise - Siril CLI v{VERSION}")
                    # Add denoiser-specific arguments
                    denoise_parser.add_argument("-strength", type=float, default=1.0,
                                               help="Denoising strength (0.0-1.0)")
                    denoise_parser.add_argument("-batch", type=int, default=4,
                                               help="Batch size for processing")
                    denoise_parser.add_argument("-model", type=str,
                                               help="Model name to use (directory name in GraXpert models folder)")
                    denoise_parser.add_argument("-gpu", action="store_true", default=True,
                                               help="Enable GPU acceleration (default)")
                    denoise_parser.add_argument("-nogpu", action="store_true", default=False,
                                               help="Disable GPU acceleration")
                    denoise_parser.add_argument("-listmodels", action="store_true",
                                               help="List available models and exit")
                    denoise_parser.print_help()
                    sys.exit(0)

                # Otherwise proceed with denoiser
                DenoiserCLI(siril, remaining_args)
            elif args.deconv_obj or args.deconv_stellar:
                # If help is requested for the denoiser
                if args.help:
                    # Create parser just to show help
                    denoise_parser = argparse.ArgumentParser(
                        description=f"GraXpert AI Deconvolution - Siril CLI v{VERSION}")
                    # Add denoiser-specific arguments
                    denoise_parser.add_argument("-strength", type=float, default=1.0,
                                               help="Denoising strength (0.0-1.0)")
                    denoise_parser.add_argument("-psfsize", type=float, default=0.3,
                                               help="PSF size: adjust to suit your seeing")
                    denoise_parser.add_argument("-batch", type=int, default=4,
                                               help="Batch size for processing")
                    denoise_parser.add_argument("-model", type=str,
                                               help="Model name to use (directory name in GraXpert models folder)")
                    denoise_parser.add_argument("-gpu", action="store_true", default=True,
                                               help="Enable GPU acceleration (default)")
                    denoise_parser.add_argument("-nogpu", action="store_true", default=False,
                                               help="Disable GPU acceleration")
                    denoise_parser.add_argument("-listmodels", action="store_true",
                                               help="List available models and exit")
                    denoise_parser.print_help()
                    sys.exit(0)

                deconv_type = "deconv_obj" if args.deconv_obj else "deconv_stellar"
                DeconvolutionCLI(siril, deconv_type, remaining_args)
            elif args.bge:
                # If help is requested for background extraction
                if args.help:
                    # Create parser just to show help
                    bge_parser = argparse.ArgumentParser(
                        description=f"GraXpert AI BG Extraction - Siril CLI v{VERSION}")
                    # Add bge-specific arguments
                    bge_parser.add_argument("-correction", type=str, default="subtraction",
                                           help="Correction type ('subtraction' (default) or 'division')")
                    bge_parser.add_argument("-smoothing", type=float, default=1.0,
                                           help="Smoothing (0.0-1.0)")
                    bge_parser.add_argument("-model", type=str,
                                           help="Model name to use (directory name in GraXpert models folder)")
                    bge_parser.add_argument("-gpu", action="store_true", default=True,
                                           help="Enable GPU acceleration (default)")
                    bge_parser.add_argument("-nogpu", action="store_true", default=False,
                                           help="Disable GPU acceleration")
                    bge_parser.add_argument("-listmodels", action="store_true",
                                           help="List available models and exit")
                    bge_parser.print_help()
                    sys.exit(0)

                # Otherwise proceed with background extraction
                BackgroundExtractionCLI(siril, remaining_args)
            else:
                # No tool specified - show available tools and default to denoiser
                print("No tool specified. Use -denoise, -bge, -deconv-stellar or -deconv-obj to select a tool.")
        else:
            # GUI mode - use PyQt6
            app = QApplication(sys.argv)
            window = GUIInterface(siril)
            window.show()
            sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
