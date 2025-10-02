"""
Image blinker / browser / filter / sorter utility script for Siril with 
adaptive caching.
Supports both loaded sequences and raw FITS images in the working directory.
Enhanced with image filtering capabilities based on analysis data.
"""
# (c) 2025 Adrian Knagg-Baugh
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 2.0.0

import os
import sys
import math
import glob
import shutil
import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import sirilpy as s
s.ensure_installed("PyQt6", "psutil", "matplotlib", "psutil", "pytz")

import pytz
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import psutil
import matplotlib.pyplot as plt

from sirilpy import ImageAnalysis, ImageType, SirilError

VERSION = "2.0.0"
CACHE_FRACTION = 0.75  # default fraction of available memory

def floor_value(value: float, decimals: int = 2) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

def numpy_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """
    Convert a numpy array into a QImage.
    Supports grayscale (H, W) or color in either (3, H, W) or (H, W, 3).
    Returns QImage (copied to own buffer).
    """
    if arr is None:
        raise ValueError("None array passed to numpy_to_qimage")

    if arr.ndim == 2:
        img = np.require(arr, dtype=np.uint8, requirements=["C_CONTIGUOUS"])
        h, w = img.shape
        qimg = QtGui.QImage(
            img.data, w, h, img.strides[0], QtGui.QImage.Format.Format_Grayscale8
        )
        return qimg.copy()

    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[1] != 3:
            img = np.transpose(arr, (1, 2, 0))
        else:
            img = arr
        img = np.require(img, dtype=np.uint8, requirements=["C_CONTIGUOUS"])
        h, w, c = img.shape
        if c != 3:
            raise ValueError(f"Unsupported channel count: {c}")
        qimg = QtGui.QImage(
            img.data, w, h, img.strides[0], QtGui.QImage.Format.Format_RGB888
        )
        return qimg.copy()

    raise ValueError(f"Unsupported array shape: {arr.shape}")


# --------------------- Frame sources ---------------------

class FrameSource:
    """Abstract base class for frame sources with caching."""

    def __init__(self, mem_fraction: float = CACHE_FRACTION):
        self.cache: Dict[int, QtGui.QPixmap] = {}
        self.mem_fraction = mem_fraction
        self.max_cache_bytes = int(psutil.virtual_memory().available * self.mem_fraction)
        self.bytes_per_frame = 1  # estimated later
        self.cache_radius = 0

    def __len__(self) -> int:
        raise NotImplementedError

    def _estimate_frame_size(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap.isNull():
            return
        bpl = pixmap.toImage().bytesPerLine()
        self.bytes_per_frame = bpl * pixmap.height()
        # estimate radius = max frames we can hold
        if self.bytes_per_frame > 0:
            max_frames = max(1, self.max_cache_bytes // self.bytes_per_frame)
            # keep ~half in memory around current index
            self.cache_radius = max(1, max_frames // 2)

    def _evict_outside_window(self, center: int) -> None:
        low = max(0, center - self.cache_radius)
        high = min(len(self) - 1, center + self.cache_radius)
        to_delete = [k for k in self.cache if k < low or k > high]
        for k in to_delete:
            del self.cache[k]

    def get_pixmap(self, i: int) -> QtGui.QPixmap:
        if i in self.cache:
            return self.cache[i]
        pixmap = self._load_frame(i)
        if self.bytes_per_frame == 1 and not pixmap.isNull():
            self._estimate_frame_size(pixmap)
        self.cache[i] = pixmap
        self._evict_outside_window(i)
        return pixmap

    def _load_frame(self, i: int) -> QtGui.QPixmap:
        raise NotImplementedError


class SequenceFrameSource(FrameSource):
    """Frames from Siril sequence (preview)."""

    def __init__(self, siril: s.SirilInterface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siril = siril
        self.sequence = siril.get_seq()
        self.frame_indices = [
            i for i in range(self.sequence.number) if self.sequence.imgparam[i].incl
        ]

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _load_frame(self, i: int) -> QtGui.QPixmap:
        idx = self.frame_indices[i]
        arr = self.siril.get_seq_frame_pixeldata(idx, preview=True)
        qimg = numpy_to_qimage(arr)
        return QtGui.QPixmap.fromImage(qimg)

from pathlib import Path
import shutil

class DirectoryFrameSource(FrameSource):
    """Frames from FITS and RAW files in working directory.
       Non-matching files are moved into a 'rejected' subfolder."""

    def __init__(self, siril: s.SirilInterface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siril = siril
        self.wd = Path(siril.get_siril_wd())

        fits_exts = {".fit", ".fits", ".fts"}
        raw_exts = {
            ".cr2", ".cr3", ".crw",   # Canon
            ".nef", ".nrw",           # Nikon
            ".arw", ".srf", ".sr2",   # Sony
            ".raf",                   # Fujifilm
            ".rw2",                   # Panasonic/Leica
            ".orf",                   # Olympus/OM System
            ".pef",                   # Pentax
            ".dng",                   # Adobe DNG / ProRAW
            ".3fr", ".fff",           # Hasselblad
            ".iiq",                   # Phase One
            ".x3f",                   # Sigma
            ".raw",                   # Generic
        }
        exts = fits_exts | raw_exts

        files = []
        rejected_files = []  # Collect files to reject

        for f in self.wd.iterdir():
            if f.is_file():
                if f.suffix.lower() in exts:
                    files.append(str(f))  # store as string
                else:
                    rejected_files.append(f)

        # Only create rejected folder and move files if there are files to reject
        if rejected_files:
            rejected_dir = self.wd / "rejected"
            rejected_dir.mkdir(exist_ok=True)

            for f in rejected_files:
                target = rejected_dir / f.name
                try:
                    shutil.move(str(f), str(target))
                    print(f"Unexpected filetype {f} found: moved to 'rejected' subdirectory")
                except Exception as e:
                    print(f"Failed to move {f} to {target}: {e}", file=sys.stderr)

        self.files = sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def _load_frame(self, i: int) -> QtGui.QPixmap:
        fname = str(self.files[i])  # convert Path to string
        try:
            arr = self.siril.load_image_from_file(fname, preview=True).data
            qimg = numpy_to_qimage(arr)
            return QtGui.QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Error loading frame {fname}: {e}", file=sys.stderr)
            return QtGui.QPixmap()

# --------------------- Instructions Dialog ---------------------

class InstructionsDialog(QtWidgets.QDialog):
    """Popup dialog to show usage instructions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instructions - Blink / Browse / Filter / Sort")
        self.setModal(True)
        self.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QTextEdit(self)
        instructions.setReadOnly(True)
        instructions.setPlainText(
            "This script can be used as a blink comparator or an image filter. It can operate on "
            "either the currently loaded sequence (in sequence mode) or the current directory "
            "(in directory mode). If a sequence is loaded the script will start in sequence mode but "
            "it is possible to switch between the two, and to change the directory in directory mode.\n\n"
            "1. A sequence or working directory must contain FITS, XISF or camera raw images. "
            "When operating on a directory, any other files will be moved to a \"rejected\" subfolder.\n\n"
            "2. If a sequence is loaded, selected frames are blinked. Analysis and filtering is not "
            "possible in sequence mode, as that functionality already exists in Siril - sequence mode "
            "is just intended to provide blink comparator functionality.\n\n"
            "3. If no sequence is loaded the script will start in directory mode and all FITS, XISF and "
            "camera RAW files in the working directory are browsed.\n\n"
            "4. Frames are adaptively cached depending on available memory.\n\n"
            "5. Adjust blink speed and press Go, or press Next or Prev to browse manually.\n\n"
            "6. Images can be manually included or excluded with the \"Toggle Include\" button. "
            "Excluded images are marked with a cross. If a sequence is loaded, the frame "
            "inclusion will be updated. If a folder is being browsed, the included images "
            "can be converted into a sequence using the \"Create Sequence\" button. The "
            "sequence name can be set using the appropriate text entry widget.\n\n"
            "7. Use the selection buttons to quickly select all, unselect all, or invert "
            "the selection of all images at once.\n\n"
            "8. Mouse wheel zooms in/out on the image, and you can pan by dragging with "
            "the mouse.\n\n"
            "9. For directory browsing, use \"Analyse files\" to gather image quality metrics, "
            "then use the filtering options to automatically select/deselect images based on "
            "background noise, FWHM, star count, and roundness criteria.\n\n"
            "10. If you have a directory with mixed lights and calibration frames, once analysis "
            "is run you can use the \"Sort\" button to sort them into separate subdirectories "
            "(lights, darks, flats and biases) and automatically create calibration frame sequences. "
            "This will leave you in the Lights folder ready to filter the lights and create a "
            "sequence of the files you want included.\n\n"
            "11. If images are taken on multiple nights they will be split into different sessions "
            "(taking noon local time as the break between sessions): directories containing lights "
            "from different sessions may be selected using a drop-down for filtering. For lights and "
            "flats taken with different filters, they will also be sorted into different subdirectories "
            "for each filter used on a given night. If some images have filter names, any images "
            "without filters will be allocated to a 'No_filter' subdirectory.\n\n"
            "12. Note that while this script can be used for pre-filtering images before even "
            "starting to preprocess them in Siril, measurements such as background noise will not "
            "be as accurate when measured on raw uncalibrated lights as when measured later in "
            "a Siril workflow. However the functionality still offers a means of filtering out "
            "the worst images from a session before beginning your main Siril workflow"
        )
        layout.addWidget(instructions)

        # OK button
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

# --------------------- GUI ---------------------

class GraphicsView(QtWidgets.QGraphicsView):
    """QGraphicsView with high-performance pan and zoom."""

    zoom_factor_base = 1.25

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.SmartViewportUpdate
        )
        self.setCacheMode(QtWidgets.QGraphicsView.CacheModeFlag.CacheBackground)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        if event.angleDelta().y() == 0:
            return
        factor = self.zoom_factor_base if event.angleDelta().y() > 0 else 1 / self.zoom_factor_base
        self.scale(factor, factor)


class BlinkInterface(QtWidgets.QWidget):
    """Main widget for the blink comparator."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Blink / Browse / Filter / Sort - v{VERSION}")
        self.resize(1100, 700)
        self.current_index = 0
        self.included = {}
        self.analysis = {}
        self.allow_analysis = True
        self.analysis_complete = False
        self.sessions_data = {}  # Will store all session data
        self.current_session = None
        self.create_cal_sequences = True  # Default to enabled
        self.mode_switching_enabled = False  # Will be set based on sequence availablity

        # Siril connection
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            QtWidgets.QMessageBox.critical(self, "Connection Error", f"Failed to connect to Siril: {e}")
            QtCore.QTimer.singleShot(0, self.close)
            return

        try:
            self.siril.cmd("requires", "1.4.0-beta3")
        except s.CommandError:
            QtWidgets.QMessageBox.critical(self, "Error", "Siril 1.4.0-beta3 or higher is required.")
            QtCore.QTimer.singleShot(0, self.close)
            return

        # State
        self.from_files = False if self.siril.is_sequence_loaded() else True
        self.source: Optional[FrameSource] = None
        self.current_index = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # UI
        self._build_ui()

    def _build_ui(self) -> None:
            root = QtWidgets.QHBoxLayout(self)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)

            # Create scroll area for the left panel
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setMinimumWidth(350)
            scroll_area.setMaximumWidth(400)

            # Create the scrollable widget
            scroll_widget = QtWidgets.QWidget()
            left = QtWidgets.QVBoxLayout(scroll_widget)
            left.setContentsMargins(8, 8, 8, 8)

            root.addWidget(scroll_area, 0)
            scroll_area.setWidget(scroll_widget)

            title = QtWidgets.QLabel("Siril Blink / Browse / Filter / Sort")
            font = title.font()
            font.setPointSize(font.pointSize() + 4)
            font.setBold(True)
            title.setFont(font)
            left.addWidget(title)

            # Instructions button
            self.instructions_btn = QtWidgets.QPushButton("Instructions", self)
            self.instructions_btn.clicked.connect(self.show_instructions)
            left.addWidget(self.instructions_btn)

            # Mode switching (only show if sequence is loaded)
            self.mode_group = QtWidgets.QGroupBox("Mode", self)
            mode_layout = QtWidgets.QHBoxLayout(self.mode_group)

            self.seq_mode_btn = QtWidgets.QRadioButton("Sequence Mode", self)
            self.dir_mode_btn = QtWidgets.QRadioButton("Directory Mode", self)
            self.seq_mode_btn.toggled.connect(self._mode_changed)
            self.dir_mode_btn.toggled.connect(self._mode_changed)

            mode_layout.addWidget(self.seq_mode_btn)
            mode_layout.addWidget(self.dir_mode_btn)

            # Set initial mode and visibility
            if self.siril.is_sequence_loaded():
                self.seq_mode_btn.setChecked(True)
                self.mode_group.setVisible(True)
                self.mode_switching_enabled = True
            else:
                self.dir_mode_btn.setChecked(True)
                self.mode_group.setVisible(False)
                self.mode_switching_enabled = False

            left.addWidget(self.mode_group)

            # Directory selection - should only be visible in directory mode
            self.dir_group = QtWidgets.QGroupBox("Working Directory", self)
            self.dir_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
            dir_layout = QtWidgets.QVBoxLayout(self.dir_group)

            self.current_dir_label = QtWidgets.QLabel(self.siril.get_siril_wd(), self)
            self.current_dir_label.setWordWrap(True)
            self.current_dir_label.setStyleSheet("font-family: monospace; padding: 4px; border: 1px solid #ccc;")

            self.change_dir_btn = QtWidgets.QPushButton("Change Directory", self)
            self.change_dir_btn.clicked.connect(self._change_directory)

            dir_layout.addWidget(QtWidgets.QLabel("Current directory:"))
            dir_layout.addWidget(self.current_dir_label)
            dir_layout.addWidget(self.change_dir_btn)

            # Set initial visibility based on mode
            self.dir_group.setVisible(self.from_files)

            left.addWidget(self.dir_group)

            # Blink and Select section
            blink_select_group = QtWidgets.QGroupBox("Blink and Select", self)
            blink_select_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
            blink_select_v = QtWidgets.QVBoxLayout(blink_select_group)

            actions = QtWidgets.QGridLayout()
            self.go_btn = QtWidgets.QPushButton("Go!", self)
            self.go_btn.clicked.connect(self.start_blink)
            self.go_btn.setToolTip("Blink the images")
            self.pause_btn = QtWidgets.QPushButton("Pause", self)
            self.pause_btn.clicked.connect(self.pause_blink)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setToolTip("Pause / resume blinking the images")
            self.stop_btn = QtWidgets.QPushButton("Stop", self)
            self.stop_btn.clicked.connect(self.stop_blink)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setToolTip("Stop blinking the images")
            self.prev_btn = QtWidgets.QPushButton("Prev", self)
            self.prev_btn.clicked.connect(self.prev_frame)
            self.prev_btn.setToolTip("Manually step to the previous image")
            self.next_btn = QtWidgets.QPushButton("Next", self)
            self.next_btn.clicked.connect(self.next_frame)
            self.next_btn.setToolTip("Manually step to the next image")
            self.toggle_btn = QtWidgets.QPushButton("Toggle Include", self)
            self.toggle_btn.clicked.connect(self.toggle_inclusion)
            self.toggle_btn.setToolTip("Toggle whether the current image is included or not")

            # Selection buttons
            self.select_all_btn = QtWidgets.QPushButton("Select All", self)
            self.select_all_btn.clicked.connect(self.select_all)
            self.select_all_btn.setToolTip("Select all images")
            self.unselect_all_btn = QtWidgets.QPushButton("Unselect All", self)
            self.unselect_all_btn.clicked.connect(self.unselect_all)
            self.unselect_all_btn.setToolTip("Unselect all images")
            self.invert_selection_btn = QtWidgets.QPushButton("Invert Selection", self)
            self.invert_selection_btn.clicked.connect(self.invert_selection)
            self.invert_selection_btn.setToolTip("Invert the current image selections")

            # Add widgets to grid: addWidget(widget, row, column)
            actions.addWidget(self.go_btn, 0, 0)      # Row 0, Column 0
            actions.addWidget(self.pause_btn, 0, 1)   # Row 0, Column 1
            actions.addWidget(self.stop_btn, 0, 2)    # Row 0, Column 2
            actions.addWidget(self.prev_btn, 1, 0)    # Row 1, Column 0
            actions.addWidget(self.next_btn, 1, 1)    # Row 1, Column 1
            actions.addWidget(self.toggle_btn, 1, 2)  # Row 1, Column 2
            # Selection buttons in row 2
            actions.addWidget(self.select_all_btn, 2, 0)    # Row 2, Column 0
            actions.addWidget(self.unselect_all_btn, 2, 1)  # Row 2, Column 1
            actions.addWidget(self.invert_selection_btn, 2, 2)  # Row 2, Column 2

            blink_select_v.addLayout(actions)

            # Blink duration slider (moved here after selection buttons)
            speed_row = QtWidgets.QHBoxLayout()
            speed_label = QtWidgets.QLabel("Blink duration / s:")
            self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
            self.speed_slider.setMinimum(1)
            self.speed_slider.setMaximum(20)
            self.speed_slider.setValue(3)
            self.speed_slider.valueChanged.connect(self._update_speed_label)
            self.speed_value = QtWidgets.QLabel("0.30")
            self.speed_value.setMinimumWidth(50)
            self.speed_value.setToolTip("Blink interval / s")
            speed_row.addWidget(speed_label)
            speed_row.addWidget(self.speed_slider, 1)
            speed_row.addWidget(self.speed_value)
            blink_select_v.addLayout(speed_row)

            left.addWidget(blink_select_group)

            # Analysis and Filtering section
            analysis_group = QtWidgets.QGroupBox("Analysis and Filtering", self)
            analysis_group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
            analysis_v = QtWidgets.QVBoxLayout(analysis_group)

            analysis_actions = QtWidgets.QGridLayout()

            self.create_seq_btn = QtWidgets.QPushButton("Create sequence", self)
            self.create_seq_btn.clicked.connect(self.create_seq)
            self.create_seq_btn.setVisible(self.from_files == True)
            self.create_seq_btn.setToolTip("Create a sequence of all the included files")
            self.seqname_label = QtWidgets.QLabel("Sequence name:", self)
            self.seqname_label.setVisible(self.from_files == True)
            self.seqname_label.setToolTip("Set the name of the sequence that will be created using the 'Create Sequence' button")
            self.seqname_entry = QtWidgets.QLineEdit(self)
            self.seqname_entry.setText("sequence")
            self.seqname_entry.setVisible(self.from_files == True)
            self.analyse_btn = QtWidgets.QPushButton("Analyse files", self)
            self.analyse_btn.clicked.connect(self.analyse)
            self.analyse_btn.setVisible(self.from_files == True)
            self.analyse_btn.setToolTip("Analyse files for quality parameters and image type. This must be run before files can be sorted.")
            self.stop_analysis_btn = QtWidgets.QPushButton("Stop analysis", self)
            self.stop_analysis_btn.clicked.connect(self.stop_analysis)
            self.stop_analysis_btn.setVisible(self.from_files == True)
            self.stop_analysis_btn.setToolTip("Stop analysis")
            self.sort_btn = QtWidgets.QPushButton("Sort images", self)
            self.sort_btn.clicked.connect(self.sort_image_files)
            self.sort_btn.setVisible(self.from_files == True)
            self.sort_btn.setEnabled(False)
            self.sort_btn.setToolTip("Sort image files by type (Dark, Flat, Bias, Light) into separate subdirectories. You must analyse the files before this is available")
            self.session_label = QtWidgets.QLabel("Session:", self)
            self.session_label.setVisible(False)
            self.session_combo = QtWidgets.QComboBox(self)
            self.session_combo.currentTextChanged.connect(self.switch_session)
            self.session_combo.setVisible(False)
            self.cal_seq_toggle = QtWidgets.QCheckBox("Sort creates calibration frame sequences", self)
            self.cal_seq_toggle.setChecked(True)  # Default enabled
            self.cal_seq_toggle.setVisible(self.from_files == True)
            self.cal_seq_toggle.setToolTip("Automatically create sequences for calibration frames (darks, flats, bias) when sorting files. "
                "You may wish to disable this if using the script to prepare correct directories for one of the legacy .SSF preprocessing scripts, "
                "which will create its own sequences")
            self.cal_seq_toggle.toggled.connect(self._update_cal_seq_setting)

            # Analysis buttons in row 0
            analysis_actions.addWidget(self.analyse_btn, 0, 0)  # Row 0, Column 0
            analysis_actions.addWidget(self.stop_analysis_btn, 0, 1)  # Row 0, Column 1
            analysis_actions.addWidget(self.sort_btn, 0, 2)  # Row 0, Column 2
            # Auto-create calibration sequences in row 1
            analysis_actions.addWidget(self.cal_seq_toggle, 1, 0, 1, 3)  # Row 1, spans all columns
            # Sequence creation in row 2
            analysis_actions.addWidget(self.create_seq_btn, 2, 0)  # Row 2, Column 0
            analysis_actions.addWidget(self.seqname_label, 2, 1)   # Row 2, Column 1
            analysis_actions.addWidget(self.seqname_entry, 2, 2)  # Row 2, Column 2
            # Session widgets in row 3
            analysis_actions.addWidget(self.session_label, 3, 0)    # Row 3, Column 0
            analysis_actions.addWidget(self.session_combo, 3, 1, 1, 2)  # Row 3, Columns 1-2

            analysis_v.addLayout(analysis_actions)
            left.addWidget(analysis_group)

            # Add filtering controls in an expander
            if self.from_files:
                self._build_filter_expander(left)

            self.status = QtWidgets.QLabel("")
            self.status.setMaximumWidth(400)
            left.addWidget(self.status)
            left.addStretch(1)

            right_v = QtWidgets.QVBoxLayout()
            root.addLayout(right_v, 1)

            zoom_row = QtWidgets.QHBoxLayout()
            self.zoom_in_btn = QtWidgets.QPushButton("Zoom In", self)
            self.zoom_out_btn = QtWidgets.QPushButton("Zoom Out", self)
            self.fit_btn = QtWidgets.QPushButton("Fit to Preview", self)
            self.zoom_in_btn.clicked.connect(lambda: self.view.scale(1.25, 1.25))
            self.zoom_out_btn.clicked.connect(lambda: self.view.scale(1/1.25, 1/1.25))
            self.fit_btn.clicked.connect(self.fit_to_view)
            zoom_row.addWidget(self.zoom_in_btn)
            zoom_row.addWidget(self.zoom_out_btn)
            zoom_row.addWidget(self.fit_btn)
            right_v.addLayout(zoom_row)

            self.scene = QtWidgets.QGraphicsScene(self)
            self.view = GraphicsView(self.scene, self)
            self.view.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.black))
            right_v.addWidget(self.view, 1)

            self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
            self.pixmap_item.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)
            self.scene.addItem(self.pixmap_item)

            pen = QtGui.QPen(QtCore.Qt.GlobalColor.white, 4)
            self.cross_lines = [
                self.scene.addLine(0, 0, 0, 0, pen),
                self.scene.addLine(0, 0, 0, 0, pen)
            ]
            for line in self.cross_lines:
                line.setZValue(1)       # above pixmap
                line.setVisible(False)  # hidden initially

            self._update_speed_label()
            if self.siril.is_sequence_loaded():
                self.status.setText("Operating in sequence mode")
            else:
                self.status.setText("Operating in directory mode")

    def _update_cal_seq_setting(self, checked: bool) -> None:
        """Update calibration sequence creation setting."""
        self.create_cal_sequences = checked

    def _build_filter_expander(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the filtering UI controls in an expandable widget."""
        # Create expander widget
        expander_widget = QtWidgets.QWidget()
        expander_layout = QtWidgets.QVBoxLayout(expander_widget)
        expander_layout.setContentsMargins(0, 0, 0, 0)

        # Create toggle button
        self.filter_toggle_btn = QtWidgets.QPushButton("▶ Image Filtering", self)
        self.filter_toggle_btn.setFlat(True)
        self.filter_toggle_btn.setStyleSheet("text-align: left; font-weight: bold; padding: 5px;")
        self.filter_toggle_btn.clicked.connect(self._toggle_filter_expander)
        expander_layout.addWidget(self.filter_toggle_btn)

        # Create collapsible content widget
        self.filter_content = QtWidgets.QWidget()
        self.filter_content.setVisible(False)  # Start collapsed
        filter_layout = QtWidgets.QVBoxLayout(self.filter_content)

        # Plot button
        self.plot_btn = QtWidgets.QPushButton("Plot Analysis Results", self)
        self.plot_btn.clicked.connect(self.plot_analysis_results)
        self.plot_btn.setEnabled(False)  # Enabled only once analysis is complete
        self.plot_btn.setToolTip("Plot image analyses. If no criteria are selected, all criteria will be plotted, otherwise only the selected criteria will be plotted")
        filter_layout.addWidget(self.plot_btn)

        # Filter button
        self.filter_btn = QtWidgets.QPushButton("Filter Images", self)
        self.filter_btn.clicked.connect(self.apply_filters)
        self.filter_btn.setEnabled(False)  # Disabled until analysis complete
        self.filter_btn.setToolTip("Filter the images based on the selected criteria")
        filter_layout.addWidget(self.filter_btn)

        # Define filter configurations
        filter_configs = {
            'bgnoise': {'label': 'Background Noise', 'smaller_better': True, 'abs_max': 1.0},
            'fwhm': {'label': 'FWHM', 'smaller_better': True, 'abs_max': 50.0},
            'wfwhm': {'label': 'WFWHM', 'smaller_better': True, 'abs_max': 50.0},
            'nbstars': {'label': 'Number of Stars', 'smaller_better': False, 'abs_max': 99999.0},
            'roundness': {'label': 'Roundness', 'smaller_better': False, 'abs_max': 1.0}
        }

        # Build filter rows
        for key, config in filter_configs.items():
            self._build_filter_row(filter_layout, config['label'], key,
                                config['smaller_better'], config['abs_max'])

        expander_layout.addWidget(self.filter_content)
        layout.addWidget(expander_widget)

    def _toggle_filter_expander(self) -> None:
        """Toggle the visibility of the filter content."""
        is_expanded = self.filter_content.isVisible()
        self.filter_content.setVisible(not is_expanded)
        # Update button text with arrow direction
        arrow = "▼" if not is_expanded else "▶"
        self.filter_toggle_btn.setText(f"{arrow} Image Filtering")

    def _build_filter_row(self, layout: QtWidgets.QVBoxLayout, label: str, key: str,
                        smaller_better: bool, abs_max: float) -> None:
        """Build a single filter row with checkbox, threshold input, and percentage/absolute toggle."""
        # Main container
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QVBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)

        # Checkbox and label with direction indicator
        header_layout = QtWidgets.QHBoxLayout()
        direction_text = "lower" if smaller_better else "higher"
        checkbox = QtWidgets.QCheckBox(f"Filter by {label} ({direction_text} is better)")
        checkbox.setToolTip("Select / deselect this criterion")
        header_layout.addWidget(checkbox)
        header_layout.addStretch()
        row_layout.addLayout(header_layout)

        # Controls container
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls_widget)
        controls_layout.setContentsMargins(20, 0, 0, 0)  # Indent controls

        # Threshold value
        threshold_label = "Keep images with values" + (" below:" if smaller_better else " above:")
        controls_layout.addWidget(QtWidgets.QLabel(threshold_label), 0, 0)

        threshold_spin = QtWidgets.QDoubleSpinBox()
        threshold_spin.setDecimals(3)
        controls_layout.addWidget(threshold_spin, 0, 1)

        # Type selection (absolute/percentage)
        type_layout = QtWidgets.QHBoxLayout()
        abs_radio = QtWidgets.QRadioButton("Absolute")
        pct_radio = QtWidgets.QRadioButton("Percentage")
        abs_radio.setChecked(True)  # Default to absolute
        type_layout.addWidget(abs_radio)
        type_layout.addWidget(pct_radio)
        type_layout.addStretch()
        controls_layout.addLayout(type_layout, 1, 0, 1, 2)

        # Function to update range based on type selection
        def update_range():
            if abs_radio.isChecked():
                # Absolute mode
                threshold_spin.setRange(0.001, abs_max)
                threshold_spin.setValue(abs_max / 2)  # Set to middle of range
            else:
                # Percentage mode (0-100%)
                threshold_spin.setRange(0.1, 100.0)
                threshold_spin.setValue(50.0)  # Default to 50%

        # Connect radio button changes to range update
        abs_radio.toggled.connect(update_range)
        pct_radio.toggled.connect(update_range)

        # Set initial range
        update_range()

        # Enable/disable controls based on checkbox
        def toggle_controls(enabled):
            threshold_spin.setEnabled(enabled)
            abs_radio.setEnabled(enabled)
            pct_radio.setEnabled(enabled)

        checkbox.toggled.connect(toggle_controls)
        toggle_controls(False)  # Initially disabled

        row_layout.addWidget(controls_widget)
        layout.addWidget(row_widget)

        # Store references for later access
        setattr(self, f"{key}_enabled", checkbox)
        setattr(self, f"{key}_threshold", threshold_spin)
        setattr(self, f"{key}_absolute", abs_radio)
        setattr(self, f"{key}_percentage", pct_radio)
        setattr(self, f"{key}_smaller_better", smaller_better)

    def show_instructions(self) -> None:
        """Show the instructions dialog."""
        dialog = InstructionsDialog(self)
        dialog.exec()

    def select_all(self) -> None:
        """Select all frames."""
        if not self.source:
            return

        for i in range(len(self.source)):
            self.included[i] = True
            if isinstance(self.source, SequenceFrameSource):
                seq_idx = self.source.frame_indices[i]
                self.siril.set_seq_frame_incl(seq_idx, True)

        self.show_frame(self.current_index)  # Refresh display
        self.update_status()

    def unselect_all(self) -> None:
        """Unselect all frames."""
        if not self.source:
            return

        for i in range(len(self.source)):
            self.included[i] = False
            if isinstance(self.source, SequenceFrameSource):
                seq_idx = self.source.frame_indices[i]
                self.siril.set_seq_frame_incl(seq_idx, False)

        self.show_frame(self.current_index)  # Refresh display
        self.update_status()

    def invert_selection(self) -> None:
        """Invert the selection of all frames."""
        if not self.source:
            return

        for i in range(len(self.source)):
            current_state = self.included.get(i, True)
            new_state = not current_state
            self.included[i] = new_state
            if isinstance(self.source, SequenceFrameSource):
                seq_idx = self.source.frame_indices[i]
                self.siril.set_seq_frame_incl(seq_idx, new_state)

        selected_count = sum(1 for state in self.included.values() if state)
        self.show_frame(self.current_index)  # Refresh display
        self.update_status()

    def _blink_interval_ms(self) -> int:
        val = self.speed_slider.value() * 0.1
        val = max(0.1, min(2.0, val))
        return int(floor_value(val, 2) * 1000)

    def _update_speed_label(self) -> None:
        seconds = self.speed_slider.value() * 0.1
        seconds = floor_value(seconds, 2)
        self.speed_value.setText(f"{seconds:.2f}")
        if self.timer.isActive():
            self.timer.setInterval(self._blink_interval_ms())

    def fit_to_view(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        rect = self.pixmap_item.boundingRect()
        self.view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def pause_blink(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.status.setText("Blink paused.")
            self.pause_btn.setText("Resume")
        else:
            self.timer.start(self._blink_interval_ms())
            self.status.setText(f"Resumed blinking: frame {self.current_index+1}")
            self.pause_btn.setText("Pause")

    def update_status(self) -> None:
        incl_state = "included" if self.included.get(self.current_index, True) else "excluded"
        incl = sum(self.included.values())
        tot = len(self.source)

        status_text = f"Frame {self.current_index+1}/{tot} ({incl_state})\n{incl} / {tot} included"

        # Add analysis data if available
        if self.current_index in self.analysis:
            analysis = self.analysis[self.current_index]
            status_text += f"\nBG Noise: {analysis.bgnoise:.3e}"
            status_text += f"\nFWHM: {analysis.fwhm:.2f}"
            status_text += f"\nWFWHM: {analysis.wfwhm:.2f}"
            status_text += f"\nStars: {analysis.nbstars}"
            status_text += f"\nRoundness: {analysis.roundness:.3f}"

        self.status.setText(status_text)

        # Position/update cross overlay
        rect = self.pixmap_item.boundingRect()
        if incl_state == "excluded":
            self.cross_lines[0].setLine(rect.left(), rect.top(), rect.right(), rect.bottom())
            self.cross_lines[1].setLine(rect.right(), rect.top(), rect.left(), rect.bottom())
            for line in self.cross_lines:
                line.setVisible(True)
        else:
            for line in self.cross_lines:
                line.setVisible(False)

    def show_frame(self, index: int) -> None:
        self.current_index = index % len(self.source)
        pm = self.source.get_pixmap(self.current_index)
        self.pixmap_item.setPixmap(pm)
        self.update_status()

    def configure_blink(self) -> None:
        self.status.setText("Preparing frames... Please wait.")
        QtWidgets.QApplication.processEvents()

        # choose source
        if not self.source:
            if self.siril.is_sequence_loaded():
                self.source = SequenceFrameSource(self.siril)
            else:
                self.source = DirectoryFrameSource(self.siril)

            if len(self.source) == 0:
                self.status.setText("No frames found.")
                return

        # Initialize inclusion states
        if len(self.included) == 0:
            if isinstance(self.source, SequenceFrameSource):
                # obtain inclusion from Siril sequence flags
                for i in range(len(self.source)):
                    seq_idx = self.source.frame_indices[i]
                    try:
                        state = bool(self.siril.get_seq_frame_incl(seq_idx))
                    except Exception:
                        # fallback: assume included
                        state = True
                    self.included[i] = state
            else:
                # directory: assume all included initially
                self.included = {i: True for i in range(len(self.source))}

        # show first frame and start timer
        self.show_frame(self.current_index)
        self.view.resetTransform()
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fit_to_view()

    def prev_frame(self) -> None:
        if not self.source or len(self.source) == 0:
            self.configure_blink()
            self.current_index = 1
        self.show_frame(self.current_index - 1)

    def toggle_inclusion(self) -> None:
        if not self.source:
            return
        current = self.included.get(self.current_index, True)
        new_state = not current
        self.included[self.current_index] = new_state
        if isinstance(self.source, SequenceFrameSource):
            seq_idx = self.source.frame_indices[self.current_index]
            self.siril.set_seq_frame_incl(seq_idx, new_state)
        self.show_frame(self.current_index)

    def start_blink(self) -> None:
        # stop any existing blink
        if self.timer.isActive():
            self.timer.stop()
        self.configure_blink()
        self.timer.start(self._blink_interval_ms())
        self.pause_btn.setEnabled(True)
        self.go_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_blink(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.current_index = 0
        self.show_frame(self.current_index)
        self.status.setText("Blinking stopped.")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.go_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def next_frame(self) -> None:
        if not self.source or len(self.source) == 0:
            self.configure_blink()
            self.current_index = len(self.source) - 1
        self.show_frame(self.current_index + 1)

    def compute_wfwhm_weights(self) -> None:
        """
        Compute weighted FWHM values for all analyzed images.
        This should be called after analysis is complete.
        """
        # self.debug_weights = True
        if not self.analysis_complete or not self.analysis:
            print("Analysis not complete or no analysis data available")
            return

        # Extract valid FWHM values (> 0) and their indices
        valid_fwhm_data = []
        for idx, analysis in self.analysis.items():
            if analysis.fwhm > 0:
                valid_fwhm_data.append((idx, analysis.fwhm))

        if len(valid_fwhm_data) < 2:
            print("Need at least 2 valid FWHM measurements to compute weights")
            return

        # Find min and max FWHM values
        fwhm_values = [fwhm for _, fwhm in valid_fwhm_data]
        fwhm_min = min(fwhm_values)
        fwhm_max = max(fwhm_values)

        if fwhm_min == fwhm_max:
            # All FWHM values are the same, assign equal weights
            for idx in self.analysis:
                self.analysis[idx].wfwhm = 1.0
            return

        # Compute normalization factors (following the C code logic)
        inv_denom = 1.0 / (1.0 / (fwhm_min * fwhm_min) - 1.0 / (fwhm_max * fwhm_max))
        inv_fwhm_max2 = 1.0 / (fwhm_max * fwhm_max)

        # Calculate weights for each image
        weights = {}
        total_weight = 0.0
        valid_count = 0

        for idx, analysis in self.analysis.items():
            if analysis.fwhm > 0:
                # Apply the weighting formula from the C code
                weight = (1.0 / (analysis.fwhm * analysis.fwhm) - inv_fwhm_max2) * inv_denom
                weights[idx] = weight
                total_weight += weight
                valid_count += 1
            else:
                weights[idx] = 0.0

        # Normalize weights to average to 1.0 (like the C code does)
        if total_weight > 0 and valid_count > 0:
            norm_factor = valid_count / total_weight

            for idx in weights:
                weights[idx] *= norm_factor
                # Store the weighted FWHM in the analysis structure
                self.analysis[idx].wfwhm = weights[idx]
        else:
            # Fallback: set all weights to 0
            for idx in self.analysis:
                self.analysis[idx].wfwhm = 0.0

        print(f"Computed wFWHM weights for {valid_count} images")
        if hasattr(self, 'debug_weights') and self.debug_weights:
            for idx in sorted(self.analysis.keys()):
                analysis = self.analysis[idx]
                print(f"Image #{idx} - FWHM: {analysis.fwhm:.2f} - wFWHM weight: {analysis.wfwhm:.2f}")

    def analyse(self) -> None:
        """
        Analyse a directory for use in culling images
        """
        if not self.source:
            self.configure_blink()

        if not isinstance(self.source, DirectoryFrameSource):
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "Directory analysis is only available when working with files from directory."
            )
            return

        # Initialize analysis state
        self.current_analysis_index = 0
        self.analysis_success_count = 0
        self.allow_analysis = True
        self.analysis_complete = False
        self.analysis = {}  # Clear previous analysis
        self.analysis_width = -1
        self.analysis_height = -1
        self.different_sizes = False

        # Disable filter button during analysis
        if hasattr(self, 'filter_btn'):
            self.filter_btn.setEnabled(False)

        # Create and start timer
        self.analysis_timer = QtCore.QTimer()
        self.analysis_timer.timeout.connect(self.analyse_single_file)
        self.analysis_timer.start(10)  # Process every 10ms, to maintain interactivity

    def sort_image_files(self):
        """
        Sort image files by session, filter (lights/flats only), and type into subdirectories with symlinks
        """
        # Store original working directory
        original_dir = self.siril.get_siril_wd()
        os.chdir(original_dir)

        # Group all files by session, filter, and type
        sessions = self.group_files_by_session_filter_and_type(
            {i: self.source.files[i] for i in range(len(self.source.files))},
            self.included,
            self.analysis
        )

        if not sessions:
            QtWidgets.QMessageBox.warning(self, "Warning", "No analyzed files to sort.")
            return

        # Check if sorting is actually needed
        if not self._check_if_sorting_needed(sessions):
            self.status.setText("All files are in the same session and image type. No sorting needed.")
            self.sort_btn.setEnabled(False)
            return

        # Store sessions data for later switching
        self.sessions_data = sessions

        # Process each session
        earliest_session = None
        earliest_lights_path = None
        session_lights_paths = []

        for session_name in sorted(sessions.keys()):
            session_data = sessions[session_name]
            session_path = Path(original_dir) / session_name
            session_path.mkdir(exist_ok=True)

            # Process each group within this session
            for group_name in sorted(session_data.keys()):
                group_data = session_data[group_name]

                # Determine if this group represents a filter or common area
                is_filter_group = group_name != "common"

                if is_filter_group:
                    # Create filter subdirectory
                    group_path = session_path / group_name
                    group_path.mkdir(exist_ok=True)
                else:
                    # Use session directory directly for common files
                    group_path = session_path

                # Process each image type for this session/group combination
                for image_type, type_data in group_data.items():
                    if not type_data or image_type == ImageType.UNKNOWN:
                        continue

                    type_name = self._get_type_directory_name(image_type)
                    type_path = group_path / type_name

                    if type_data:  # Only create directory if we have files
                        type_path.mkdir(exist_ok=True)
                        self._process_session_image_type(type_data, str(type_path), type_name.lower())

                        # Track lights folders for dropdown
                        if image_type == ImageType.LIGHT:
                            if is_filter_group:
                                combo_name = f"{session_name} - {group_name}"
                            else:
                                combo_name = session_name
                            session_lights_paths.append((combo_name, str(type_path), session_name, group_name))
                            if earliest_session is None or session_name < earliest_session:
                                earliest_session = session_name
                                earliest_lights_path = str(type_path)

        # Set up session dropdown if multiple light sessions exist
        if len(session_lights_paths) > 1:
            self.session_combo.clear()
            for combo_name, path, session_name, group_name in session_lights_paths:
                lights_count = len(sessions[session_name][group_name][ImageType.LIGHT])
                display_name = f"{combo_name} ({lights_count} lights)"
                self.session_combo.addItem(display_name, (session_name, group_name, path))
            self.session_label.setVisible(True)
            self.session_combo.setVisible(True)

            # Set to earliest session
            for i in range(self.session_combo.count()):
                session_name, group_name, path = self.session_combo.itemData(i)
                if session_name == earliest_session:
                    self.session_combo.setCurrentIndex(i)
                    break
        else:
            self.session_label.setVisible(False)
            self.session_combo.setVisible(False)

        # Switch to earliest session's lights directory
        if earliest_lights_path and os.path.exists(earliest_lights_path):
            # Find the earliest group for the earliest session
            earliest_group = None
            if earliest_session in sessions:
                # Find which group corresponds to the earliest_lights_path
                for group_name, group_data in sessions[earliest_session].items():
                    if ImageType.LIGHT in group_data and group_data[ImageType.LIGHT]:
                        # Check if this matches our earliest path
                        if group_name == "common":
                            expected_path = str(Path(original_dir) / earliest_session / "lights")
                        else:
                            expected_path = str(Path(original_dir) / earliest_session / group_name / "lights")

                        if expected_path == earliest_lights_path:
                            earliest_group = group_name
                            break

                if earliest_group is None:
                    # Fallback to first group alphabetically
                    earliest_group = sorted(sessions[earliest_session].keys())[0]

            self._switch_to_lights_directory(earliest_session, earliest_group, earliest_lights_path)
        else:
            os.chdir(original_dir)
            self.siril.cmd("cd", f'"{original_dir}"')
            self.status.setText("No light frames found in any session.")

        self.sort_btn.setEnabled(False)

    def _get_type_directory_name(self, image_type):
        """Convert ImageType enum to directory name."""
        type_names = {
            ImageType.LIGHT: "lights",
            ImageType.DARK: "darks",
            ImageType.FLAT: "flats",
            ImageType.BIAS: "bias"
        }
        return type_names.get(image_type, "Unknown")

    def _process_session_image_type(self, type_data, type_path, prefix):
        """Process a specific image type for a session."""
        if not type_data:
            return

        os.chdir(type_path)
        self.siril.cmd("cd", f'"{type_path}"')

        moved_files = {}
        for key, file_info in type_data.items():
            source_file = Path(file_info['file'])
            destination = Path(source_file.name)

            # Move file to subdirectory
            try:
                if source_file.exists():
                    shutil.move(str(source_file), str(destination))
                    moved_files[key] = destination.name
            except Exception as e:
                print(f"Failed to move {source_file}: {e}")
                continue

        # For lights, don't create symlinks, just leave files as-is
        if "lights" in prefix.lower():
            return

        # Only create symlinks and sequences if the toggle is enabled
        if not self.create_cal_sequences:
            print(f"Skipped creating symlinks and sequence for {prefix} (disabled by user)")
            return

        # For other image types, create sequentially named symlinks
        for i, (key, filename) in enumerate(moved_files.items(), 1):
            source_path = Path(filename)
            symlink_name = f"{prefix}_{i:05d}.fit"

            # Remove existing symlink if it exists
            if Path(symlink_name).exists() or Path(symlink_name).is_symlink():
                Path(symlink_name).unlink()

            # Create new symlink
            try:
                Path(symlink_name).symlink_to(source_path)
            except Exception:
                # If we can't use symlinks, move the file
                try:
                    shutil.move(str(source_path), str(symlink_name))
                except Exception as e:
                    print(f"Error: unable to symlink or move {source_path}: {e}")

        # Create sequences for calibration frames (check Windows file limit)
        if sys.platform.startswith('win') and len(moved_files) > 2048:
            print(f"More than 2048 {prefix} images: will not create {prefix} sequence")
        else:
            try:
                self.siril.create_new_seq(f"{prefix}_")
                print(f"Created {prefix}_ sequence")
            except Exception as e:
                print(f"Failed to create {prefix}_ sequence: {e}")

    def get_local_timezone(self):
        """Get the local timezone, with fallback to UTC if detection fails."""
        try:
            import time
            if hasattr(time, 'tzname') and time.tzname[0]:
                # Try to get system timezone
                return pytz.timezone(time.tzname[0])
        except:
            pass

        # Fallback: try to detect from system
        try:
            return pytz.timezone('UTC')  # You may want to make this configurable
        except:
            return pytz.UTC

    def get_session_from_timestamp(self, timestamp: int) -> str:
        """Convert unix timestamp to session date string (YYYY-MM-DD).
        Sessions split at noon local time."""
        try:
            local_tz = self.get_local_timezone()
            dt = datetime.datetime.fromtimestamp(timestamp, tz=local_tz)

            # If before noon, belongs to previous day's session
            if dt.hour < 12:
                session_date = (dt.date() - datetime.timedelta(days=1))
            else:
                session_date = dt.date()

            return session_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")
            return "unknown-session"

    def group_files_by_session_filter_and_type(self, files_dict, included_dict, analysis_dict):
        """Group files by session date, filter (for lights/flats only), and image type."""
        sessions = {}

        # First pass: collect all filters for lights and flats to determine if filter subdirs are needed
        lights_filters = set()
        flats_filters = set()

        for key in files_dict:
            if key not in analysis_dict:
                continue
            analysis = analysis_dict[key]
            filter_name = analysis.filter or "No_filter"

            if analysis.imagetype == ImageType.LIGHT:
                lights_filters.add(filter_name)
            elif analysis.imagetype == ImageType.FLAT:
                flats_filters.add(filter_name)

        # Determine if filter subdirectories are needed
        needs_light_filter_dirs = len(lights_filters) > 1
        needs_flat_filter_dirs = len(flats_filters) > 1

        # Second pass: group files
        for key in files_dict:
            if key not in analysis_dict:
                continue
            analysis = analysis_dict[key]
            session = self.get_session_from_timestamp(analysis.timestamp)
            image_type = analysis.imagetype

            # Determine grouping strategy based on image type
            if image_type == ImageType.LIGHT and needs_light_filter_dirs:
                # Group lights by filter
                filter_name = analysis.filter or "No_filter"
                group_key = filter_name
            elif image_type == ImageType.FLAT and needs_flat_filter_dirs:
                # Group flats by filter
                filter_name = analysis.filter or "No_filter"
                group_key = filter_name
            else:
                # For darks, bias, or single-filter lights/flats, use a common group
                group_key = "common"

            if session not in sessions:
                sessions[session] = {}
            if group_key not in sessions[session]:
                sessions[session][group_key] = {
                    ImageType.LIGHT: {},
                    ImageType.DARK: {},
                    ImageType.FLAT: {},
                    ImageType.BIAS: {},
                    ImageType.UNKNOWN: {}
                }

            sessions[session][group_key][image_type][key] = {
                'file': files_dict[key],
                'included': included_dict[key],
                'analysis': analysis
            }

        return sessions

    def _check_if_sorting_needed(self, sessions):
        """Check if sorting is needed based on number of sessions, filters, and image types."""
        if len(sessions) > 1:
            return True  # Multiple date sessions

        # Single session - check if we have multiple groups or image types
        session_data = list(sessions.values())[0]

        # If we have multiple groups (filters), sorting is needed
        if len(session_data) > 1:
            return True

        # Single group - check image types
        group_data = list(session_data.values())[0]
        populated_types = [img_type for img_type, type_data in group_data.items()
                        if type_data and img_type != ImageType.UNKNOWN]

        return len(populated_types) > 1  # Multiple image types

    def switch_session(self, session_display_text):
        """Switch to a different session's lights folder."""
        if not session_display_text:
            return

        current_data = self.session_combo.currentData()
        if current_data:
            session_name, group_name, lights_path = current_data
            self._switch_to_lights_directory(session_name, group_name, lights_path)

    def _switch_to_lights_directory(self, session_name, group_name, lights_path):
        """Switch to a specific session's lights directory and update data."""
        if not os.path.exists(lights_path):
            display_name = f"{session_name} - {group_name}" if group_name != "common" else session_name
            self.status.setText(f"Lights directory for {display_name} not found.")
            return

        os.chdir(lights_path)
        self.siril.cmd("cd", f'"{lights_path}"')
        self.current_session = session_name

        # Create new DirectoryFrameSource for this lights directory
        self.source = DirectoryFrameSource(self.siril)

        # Rebuild included and analysis dicts for this session's lights
        if session_name in self.sessions_data and group_name in self.sessions_data[session_name]:
            session_group_lights = self.sessions_data[session_name][group_name][ImageType.LIGHT]

            new_included = {}
            new_analysis = {}

            # Map to new sequential indices based on files in this directory
            for new_idx, source_file in enumerate(self.source.files):
                source_basename = os.path.basename(source_file)

                # Find matching entry in session data
                for old_key, file_info in session_group_lights.items():
                    if os.path.basename(file_info['file']) == source_basename:
                        new_included[new_idx] = file_info['included']
                        new_analysis[new_idx] = file_info['analysis']
                        break
                else:
                    # Default if not found
                    new_included[new_idx] = True

            self.included = new_included
            self.analysis = new_analysis
        else:
            # Fallback
            self.included = {i: True for i in range(len(self.source))}
            self.analysis = {}

        # Reset to first frame
        self.current_index = 0
        if len(self.source) > 0:
            self.show_frame(0)

        lights_count = len(self.source)
        if group_name != "common":
            self.status.setText(f"Session {session_name} - {group_name}: viewing {lights_count} light frames.")
            self.seqname_entry.setText(f"lights_{session_name}_{group_name}")
        else:
            self.status.setText(f"Session {session_name}: viewing {lights_count} light frames.")
            self.seqname_entry.setText(f"lights_{session_name}")

    def _process_image_type(self, files_dict, directory_name, prefix):
        """
        Helper method to process a specific image type
        Move files to subdirectory and create sequentially named symlinks (except for lights)
        """
        if not files_dict:
            return

        # Create directory if it doesn't exist
        dir_path = Path(directory_name)
        dir_path.mkdir(exist_ok=True)

        # First, move all files to the subdirectory
        moved_files = {}
        for key, filepath in files_dict.items():
            source_file = Path(filepath)
            destination = dir_path / source_file.name

            # Move file to subdirectory
            try:
                shutil.move(str(source_file), str(destination))
                moved_files[key] = destination.name
            except Exception:
                continue

        # For lights, just move files without creating symlinks
        if "lights" in prefix.lower():
            return

        # Only create symlinks and sequences if the toggle is enabled
        if not self.create_cal_sequences:
            print(f"Skipped creating symlinks and sequence for {prefix} (disabled by user)")
            # Change back to parent directory
            os.chdir("..")
            self.siril.cmd("cd", "..")
            return

        # Change to the subdirectory
        os.chdir(dir_path)
        self.siril.cmd("cd", f"\"{dir_path}\"")

        # For other image types, create sequentially named symlinks
        for i, (key, filename) in enumerate(moved_files.items(), 1):
            source_path = Path(filename)  # File is now in current directory
            symlink_name = f"{prefix}_{i:05d}.fit"

            # Remove existing symlink if it exists
            if Path(symlink_name).exists() or Path(symlink_name).is_symlink():
                Path(symlink_name).unlink()

            # Create new symlink
            try:
                Path(symlink_name).symlink_to(source_path)
                print(f"Created symlink: {symlink_name} -> {source_path}")
            except Exception:
                # If we can't use symlinks, move the file
                try:
                    shutil.move(str(source_path), str(symlink_name))
                    moved_files[key] = symlink_name
                except Exception as e:
                    print(f"Error: unable to symlink or move {source_path}: {e}")

        # Create sequences for the simple image types that won't need filtering
        # Check Windows file limit
        if sys.platform.startswith('win') and len(moved_files) > 2048:
            print(f"More than 2048 {prefix} images: will not create {prefix} sequence")
        else:
            self.siril.create_new_seq(f"{prefix}_")
            print(f"Created {prefix}_ sequence")

        # Change back to parent directory
        os.chdir("..")
        self.siril.cmd("cd", "..")

    def analyse_single_file(self) -> None:
        """Process a single file in the analysis"""
        if not self.allow_analysis:
            # Analysis stopped
            self.analysis_timer.stop()
            self.status.setText(f"Analysis interrupted: {self.analysis_success_count} successful out of {len(self.source)}")
            return

        if self.current_analysis_index >= len(self.source):
            # Analysis complete
            self.analysis_completed()
            return

        try:
            analysis_result = self.siril.analyse_image_from_file(
                self.source.files[self.current_analysis_index]
            )

            if not self.different_sizes:
                if self.analysis_width == -1:
                    self.analysis_width = analysis_result.width
                    self.analysis_height = analysis_result.height
                elif (self.analysis_width != analysis_result.width or
                    self.analysis_height != analysis_result.height):
                    self.different_sizes = True

            # Check if the image has 3 channels (color image)
            if analysis_result.channels == 3:
                # Move file to rejected directory and mark for removal
                self._move_file_to_rejected(self.current_analysis_index)
                # Mark this index as needing removal (will be cleaned up after analysis)
                if not hasattr(self, '_rejected_indices'):
                    self._rejected_indices = []
                self._rejected_indices.append(self.current_analysis_index)
            else:
                self.analysis[self.current_analysis_index] = analysis_result
                self.analysis_success_count += 1

        except SirilError as e:
            pass  # Skip failed analyses

        self.current_analysis_index += 1
        self.status.setText(f"Analysed {self.current_analysis_index} files ({self.analysis_success_count} successful)")

    def _move_file_to_rejected(self, file_index: int) -> None:
        """Move a file to the rejected directory."""
        if not isinstance(self.source, DirectoryFrameSource):
            return

        try:
            # Get the file path
            file_path = self.source.files[file_index]
            filename = os.path.basename(file_path)

            # Create rejected directory if it doesn't exist
            rejected_dir = Path(self.siril.get_siril_wd()) / "rejected"
            rejected_dir.mkdir(exist_ok=True)

            # Handle filename conflicts
            target_path = rejected_dir / filename
            counter = 1
            while target_path.exists():
                name, ext = os.path.splitext(filename)
                target_path = rejected_dir / f"{name}_{counter}{ext}"
                counter += 1

            # Move the file
            shutil.move(file_path, str(target_path))
            print(f"3-channel image {filename} moved to rejected directory")

            # Track rejected count for reporting
            if not hasattr(self, '_temp_rejected_count'):
                self._temp_rejected_count = []
            self._temp_rejected_count.append(file_index)

        except Exception as e:
            print(f"Failed to move file {file_path} to rejected: {e}", file=sys.stderr)

    def analysis_completed(self) -> None:
        """ Callback to run once analysis run is complete """
        self.analysis_timer.stop()

        # Clean up rejected files from tracking
        rejected_indices = []
        if hasattr(self, '_rejected_indices'):
            rejected_indices = self._rejected_indices.copy()

            # Remove rejected indices from included dict
            for idx in rejected_indices:
                if idx in self.included:
                    del self.included[idx]

            # Clean up the rejected indices list
            del self._rejected_indices

            # Rebuild the source to reflect current directory state
            if isinstance(self.source, DirectoryFrameSource):
                old_current_index = self.current_index
                self.source = DirectoryFrameSource(self.siril)

                # Remap analysis and included dicts based on remaining files
                old_analysis = self.analysis.copy()
                old_included = self.included.copy()

                # Create new dicts with sequential indices
                new_analysis = {}
                new_included = {}
                new_index = 0

                # Map old indices to new based on files that still exist
                for old_idx in sorted(old_analysis.keys()):
                    if old_idx not in rejected_indices:
                        new_analysis[new_index] = old_analysis[old_idx]
                        new_included[new_index] = old_included.get(old_idx, True)
                        new_index += 1

                self.analysis = new_analysis
                self.included = new_included

                # Adjust current index if necessary
                if old_current_index >= len(self.source):
                    self.current_index = max(0, len(self.source) - 1)

        self.analysis_complete = True

        # Compute weighted FWHM after analysis is complete
        self.compute_wfwhm_weights()

        remaining_files = len(self.source) if self.source else 0
        rejected_count = len(rejected_indices)
        status_text = f"Analysis complete: {self.analysis_success_count} successful out of {remaining_files} remaining files"
        if rejected_count > 0:
            status_text += f"\n{rejected_count} 3-channel images moved to rejected"
        if self.different_sizes:
            status_text += "\nWARNING: Multiple image sizes detected. Manual review required - cannot automatically determine which size group to keep"
        self.status.setText(status_text)

        if hasattr(self, '_temp_rejected_count'):
            del self._temp_rejected_count

        # Enable filter button
        if hasattr(self, 'filter_btn'):
            self.filter_btn.setEnabled(True)

        # Enable plot button
        if hasattr(self, 'plot_btn'):
            self.plot_btn.setEnabled(True)

        # Enable sort button
        if hasattr(self, 'sort_btn'):
            self.sort_btn.setEnabled(True)

    def stop_analysis(self) -> None:
        self.allow_analysis = False

    def plot_analysis_results(self) -> None:
        """Plot analysis metrics using matplotlib, respecting selected criteria."""
        if not self.analysis_complete or not self.analysis:
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "Analysis must be completed before plotting."
            )
            return

        indices = sorted(self.analysis.keys())

        # Collect available metrics
        metrics = {
            "bgnoise": [self.analysis[i].bgnoise for i in indices],
            "fwhm": [self.analysis[i].fwhm for i in indices],
            "wfwhm": [self.analysis[i].wfwhm for i in indices],
            "nbstars": [self.analysis[i].nbstars for i in indices],
            "roundness": [self.analysis[i].roundness for i in indices],
        }
        labels = {
            "bgnoise": "Background Noise",
            "fwhm": "FWHM",
            "wfwhm": "Weighted FWHM",
            "nbstars": "Number of Stars",
            "roundness": "Roundness",
        }

        # Determine which criteria are selected
        selected_keys = [
            key for key in metrics
            if getattr(self, f"{key}_enabled", None) and getattr(self, f"{key}_enabled").isChecked()
        ]

        # If none selected, plot all
        if not selected_keys:
            selected_keys = list(metrics.keys())

        # Create subplots dynamically
        fig, axes = plt.subplots(len(selected_keys), 1, figsize=(10, 3 * len(selected_keys)), sharex=True)

        if len(selected_keys) == 1:
            axes = [axes]  # Ensure iterable if only one subplot

        for ax, key in zip(axes, selected_keys):
            ax.plot(indices, metrics[key], "o-")
            ax.set_ylabel(labels[key])
            ax.grid(True, linestyle="--", alpha=0.5)

        axes[-1].set_xlabel("Image index")

        fig.tight_layout()
        plt.show()

    def apply_filters(self) -> None:
        """Apply filtering criteria to select/deselect images based on analysis data."""
        if not self.analysis_complete or not self.analysis:
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "Analysis must be completed before filtering can be applied."
            )
            return

        if not isinstance(self.source, DirectoryFrameSource):
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "Filtering is only available when working with files from directory."
            )
            return

        # Collect all analysis values for percentage calculations
        analysis_values = {
            'bgnoise': [],
            'fwhm': [],
            'wfwhm': [],
            'nbstars': [],
            'roundness': []
        }

        # Get values for images that have analysis data
        valid_indices = []
        for i in self.analysis:
            analysis = self.analysis[i]
            analysis_values['bgnoise'].append(analysis.bgnoise)
            analysis_values['fwhm'].append(analysis.fwhm)
            analysis_values['wfwhm'].append(analysis.wfwhm)
            analysis_values['nbstars'].append(analysis.nbstars)
            analysis_values['roundness'].append(analysis.roundness)
            valid_indices.append(i)

        filtered_count = 0
        total_count = len(self.source)

        # Apply filters to each image
        for i in range(total_count):
            if i not in self.analysis:
                # No analysis data, exclude by default when filtering
                self.included[i] = False
                continue

            analysis = self.analysis[i]
            passes_all_filters = True

            # Check each filter criterion
            for criterion in ['bgnoise', 'fwhm', 'wfwhm', 'nbstars', 'roundness']:
                enabled_attr = getattr(self, f"{criterion}_enabled")
                if not enabled_attr.isChecked():
                    continue  # Skip disabled filters

                threshold_spin = getattr(self, f"{criterion}_threshold")
                abs_radio = getattr(self, f"{criterion}_absolute")
                smaller_better = getattr(self, f"{criterion}_smaller_better")

                threshold_value = threshold_spin.value()
                actual_value = getattr(analysis, criterion)

                if abs_radio.isChecked():
                    # Absolute mode - use threshold directly
                    if smaller_better:
                        passes_filter = actual_value <= threshold_value
                    else:
                        passes_filter = actual_value >= threshold_value
                else:
                    # Percentage mode - find the percentile threshold
                    values = analysis_values[criterion]
                    if not values:
                        passes_filter = False
                    else:
                        if smaller_better:
                            # For "smaller is better", threshold% means keep the best X%
                            # (i.e., the X% with smallest values)
                            percentile_threshold = np.percentile(values, threshold_value)
                            passes_filter = actual_value <= percentile_threshold
                        else:
                            # For "larger is better", threshold% means keep the best X%
                            # (i.e., the X% with largest values)
                            percentile_threshold = np.percentile(values, 100 - threshold_value)
                            passes_filter = actual_value >= percentile_threshold

                if not passes_filter:
                    passes_all_filters = False
                    break

            # Set inclusion based on whether all filters pass
            self.included[i] = passes_all_filters
            if passes_all_filters:
                filtered_count += 1

        # Update display
        self.show_frame(self.current_index)

        # Show results
        QtWidgets.QMessageBox.information(
            self, "Filtering Complete",
            f"Filtering applied successfully.\n"
            f"Images passing all criteria: {filtered_count} out of {total_count}\n"
            f"Images excluded: {total_count - filtered_count}"
        )

    def create_seq(self) -> None:
            """
            Create a sequence from included files in the DirectoryFrameSource.
            Creates symlinks (or copies if symlinks fail) with sequential naming.
            Moves non-included files to a "rejected" subdirectory.
            """
            if not isinstance(self.source, DirectoryFrameSource):
                QtWidgets.QMessageBox.warning(
                    self, "Warning",
                    "Create sequence is only available when working with files from directory."
                )
                return

            # Get the prefix from the QLineEdit
            prefix = self.seqname_entry.text().strip()
            if not prefix:
                QtWidgets.QMessageBox.warning(
                    self, "Warning",
                    "Please enter a sequence name."
                )
                return

            # Get included files and handle rejected files
            included_files = []
            rejected_files = []

            for i in range(len(self.source)):
                if self.included.get(i, True):  # Default to True if not in dict
                    included_files.append(self.source.files[i])
                else:
                    rejected_files.append(self.source.files[i])

            if not included_files:
                QtWidgets.QMessageBox.warning(
                    self, "Warning",
                    "No files are included for sequence creation."
                )
                return

            # Check Windows file limit
            if sys.platform.startswith('win') and len(included_files) > 2048:
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    f"Windows limitation: Cannot create sequence with more than 2048 files.\n"
                    f"You have {len(included_files)} included files."
                )
                return

            try:
                # Get file extension from Siril config
                ext = self.siril.get_siril_config("core", "extension")

                # Get working directory
                wd = self.siril.get_siril_wd()

                # Handle rejected files
                rejected_moved = 0
                rejected_failed = []
                if rejected_files:
                    rejected_dir = os.path.join(wd, "rejected")
                    try:
                        # Create rejected directory if it doesn't exist
                        if not os.path.exists(rejected_dir):
                            os.makedirs(rejected_dir)

                        # Move rejected files
                        for rejected_file in rejected_files:
                            try:
                                filename = os.path.basename(rejected_file)
                                target_path = os.path.join(rejected_dir, filename)

                                # Handle filename conflicts by adding a counter
                                counter = 1
                                original_target = target_path
                                while os.path.exists(target_path):
                                    name, ext_part = os.path.splitext(filename)
                                    target_path = os.path.join(rejected_dir, f"{name}_{counter}{ext_part}")
                                    counter += 1

                                shutil.move(rejected_file, target_path)
                                rejected_moved += 1
                            except Exception as e:
                                rejected_failed.append((rejected_file, str(e)))

                    except Exception as e:
                        QtWidgets.QMessageBox.warning(
                            self, "Warning",
                            f"Could not create rejected directory or move files: {e}"
                        )

                created_files = []
                failed_files = []
                fallback = False
                # Create symlinks/copies for each included file
                for i, source_file in enumerate(included_files):
                    # Generate target filename: prefix_00001.ext
                    target_name = f"{prefix}_{i+1:05d}{ext}"
                    target_path = os.path.join(wd, target_name)

                    try:
                        # Try to create symlink first
                        if hasattr(os, 'symlink'):
                            # Remove existing file if it exists
                            if os.path.exists(target_path):
                                os.remove(target_path)
                            os.symlink(source_file, target_path)
                            created_files.append((target_name, "symlink"))
                        else:
                            raise OSError("Symlinks not supported")

                    except (OSError, NotImplementedError):
                        # Fallback to copying if symlink fails
                        try:
                            if os.path.exists(target_path):
                                os.remove(target_path)
                            shutil.copy2(source_file, target_path)
                            created_files.append((target_name, "copy"))
                            fallback = True
                        except Exception as e:
                            failed_files.append((source_file, str(e)))

                # Report results
                error_message = ""
                if fallback:
                    error_message = "Failed to create symlinks on this platform: " \
                        "attempted to make copies instead.\n"

                if failed_files:
                    error_message += f"Failed to process {len(failed_files)} files:\n"
                    for fname, error in failed_files[:5]:  # Show first 5 errors
                        error_message += f"• {os.path.basename(fname)}: {error}\n"
                    if len(failed_files) > 5:
                        error_message += f"... and {len(failed_files) - 5} more"

                if rejected_failed:
                    error_message += f"Failed to move {len(rejected_failed)} rejected files:\n"
                    for fname, error in rejected_failed[:3]:  # Show first 3 errors
                        error_message += f"• {os.path.basename(fname)}: {error}\n"
                    if len(rejected_failed) > 3:
                        error_message += f"... and {len(rejected_failed) - 3} more"

                if error_message:
                    QtWidgets.QMessageBox.warning(self, "Partial Success", error_message)

                if created_files:
                    # Create the sequence in Siril
                    try:
                        self.siril.create_new_seq(f"{prefix}_")

                        # Count symlinks vs copies for user info
                        symlink_count = sum(1 for _, method in created_files if method == "symlink")
                        copy_count = len(created_files) - symlink_count

                        success_msg = f"Sequence '{prefix}' created successfully!\n"
                        success_msg += f"Files processed: {len(created_files)}\n"
                        if symlink_count > 0:
                            success_msg += f"Symlinks: {symlink_count}\n"
                        if copy_count > 0:
                            success_msg += f"Copies: {copy_count}\n"
                        if rejected_moved > 0:
                            success_msg += f"Files moved to rejected: {rejected_moved}\n"

                        QtWidgets.QMessageBox.information(self, "Success", success_msg)
                        self.siril.cmd("load_seq", f"{prefix}_")

                    except Exception as e:
                        QtWidgets.QMessageBox.critical(
                            self, "Error",
                            f"Files created but failed to create Siril sequence: {e}"
                        )
                else:
                    QtWidgets.QMessageBox.critical(
                        self, "Error",
                        "No files could be processed for sequence creation."
                    )

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    f"Failed to create sequence: {e}"
                )

    def _mode_changed(self):
        """Handle mode switching between sequence and directory."""
        if not self.mode_switching_enabled:
            return

        if self.seq_mode_btn.isChecked():
            self._switch_to_sequence_mode()
        else:
            self._switch_to_directory_mode()

    def _switch_to_sequence_mode(self):
        """Switch to sequence mode."""
        if not self.siril.is_sequence_loaded():
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "No sequence is currently loaded in Siril."
            )
            self.dir_mode_btn.setChecked(True)
            return

        # Reset state
        self._reset_state()
        self.from_files = False
        self.source = None

        # Update UI visibility
        self._update_mode_ui_visibility()

        self.status.setText("Switched to sequence mode")

    def _switch_to_directory_mode(self):
        """Switch to directory mode."""
        # Reset state
        self._reset_state()
        self.from_files = True
        self.source = None

        # Update UI visibility
        self._update_mode_ui_visibility()

        self.status.setText("Switched to directory mode")

    def _update_mode_ui_visibility(self):
        """Update UI element visibility based on current mode."""
        # Directory-only controls
        directory_widgets = [
            'create_seq_btn', 'seqname_label', 'seqname_entry',
            'analyse_btn', 'stop_analysis_btn', 'sort_btn'
        ]

        for widget_name in directory_widgets:
            if hasattr(self, widget_name):
                getattr(self, widget_name).setVisible(self.from_files)

        # Cal sequence toggle
        if hasattr(self, 'cal_seq_toggle'):
            self.cal_seq_toggle.setVisible(self.from_files)

        # Working directory group - only show in directory mode
        if hasattr(self, 'dir_group'):
            self.dir_group.setVisible(self.from_files)

        # Session selector (only visible if we have session data and in directory mode)
        if hasattr(self, 'session_label') and hasattr(self, 'session_combo'):
            should_show_session = (self.from_files and
                                hasattr(self, 'sessions_data') and
                                len(getattr(self, 'sessions_data', {})) > 0)
            self.session_label.setVisible(should_show_session)
            self.session_combo.setVisible(should_show_session)

        # Filter expander (directory mode only)
        if hasattr(self, 'filter_toggle_btn'):
            self.filter_toggle_btn.setVisible(self.from_files)
            if hasattr(self, 'filter_content'):
                if not self.from_files:
                    self.filter_content.setVisible(False)
                    self.filter_toggle_btn.setText("▶ Image Filtering")

    def _change_directory(self):
        """Open directory chooser and change working directory."""
        current_dir = self.siril.get_siril_wd()

        new_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            current_dir,
            QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontResolveSymlinks
        )

        if new_dir and new_dir != current_dir:
            try:
                # Change Siril working directory
                self.siril.cmd("cd", f'"{new_dir}"')

                # Update UI
                self.current_dir_label.setText(new_dir)

                # Reset everything
                self._reset_state()

                # Check if sequence is available in new directory
                if self.siril.is_sequence_loaded():
                    if not self.mode_switching_enabled:
                        # Enable mode switching if we now have a sequence
                        self.mode_switching_enabled = True
                        self.mode_group.setVisible(True)
                        # Default to sequence mode if we have one
                        self.seq_mode_btn.setChecked(True)
                        self.from_files = False
                else:
                    if self.mode_switching_enabled:
                        # Disable mode switching if no sequence available
                        self.mode_switching_enabled = False
                        self.mode_group.setVisible(False)
                    # Force directory mode
                    self.dir_mode_btn.setChecked(True)
                    self.from_files = True

                self._update_mode_ui_visibility()

                if self.from_files:
                    self.status.setText(f"Changed to directory mode")
                else:
                    self.status.setText(f"Changed to sequence mode in:\n{new_dir}")

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    f"Failed to change directory: {e}"
                )

    def _reset_state(self):
        """Reset all state when switching modes or directories."""
        # Stop any running operations
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        if hasattr(self, 'analysis_timer') and hasattr(self.analysis_timer, 'isActive') and self.analysis_timer.isActive():
            self.analysis_timer.stop()

        # Reset data
        self.source = None
        self.current_index = 0
        self.included = {}
        self.analysis = {}
        self.analysis_complete = False

        # Reset session data
        if hasattr(self, 'sessions_data'):
            self.sessions_data = {}
        if hasattr(self, 'current_session'):
            self.current_session = None

        # Clear session dropdown
        if hasattr(self, 'session_combo'):
            self.session_combo.clear()

        # Reset UI state
        self.go_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("Pause")

        if hasattr(self, 'filter_btn'):
            self.filter_btn.setEnabled(False)
        if hasattr(self, 'plot_btn'):
            self.plot_btn.setEnabled(False)
        if hasattr(self, 'sort_btn'):
            self.sort_btn.setEnabled(False)

        # Clear display
        self.pixmap_item.setPixmap(QtGui.QPixmap())
        for line in self.cross_lines:
            line.setVisible(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            if self.siril:
                self.siril.disconnect()
        except Exception as e:
            print(f"Error disconnecting from Siril: {e}", file=sys.stderr)
        super().closeEvent(event)

def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Blink / Browse / Filter / Sort")
    app.setStyle("Fusion")


    w = BlinkInterface()
    w.show()
    try:
        return app.exec()
    except Exception as e:
        print(f"Error initializing application: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
