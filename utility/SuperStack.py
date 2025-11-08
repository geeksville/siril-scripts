"""
(c) Cecile Melis 2025
SPDX-License-Identifier: GPL-3.0-or-later

Performs superstacking of images of a sequence, i.e moving average of images

usage: SuperStack.py [-h] [-n] [-d] [nbframes] [step]

positional arguments:
  nbframes      Number of frames in each superstack
  step          Step between each superstack

options:
  -h, --help    show this help message and exit
  -n, --norm    Normalize the sequence before stacking
  -d, --dryrun  Check the command summary without executing it

GUI version accessible through Siril Scripts Menu

Version history:
1.0.0: Initial version
1.1.0: Add a PyQt6 GUI (AKB)
1.1.1: Add dry run checkbox to GUI (AKB)
1.1.2: Refactor code to share logic between CLI and GUI (CME)
"""

import sys
from pathlib import Path
import sirilpy as s
from sirilpy import SirilInterface
from typing import Callable
import os
import shutil
import re
import numpy as np
s.ensure_installed("PyQt6")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QCheckBox, QPushButton, QMessageBox, QTextEdit,
    QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

VERSION = "1.1.2"
SIRILPY_MIN_VERSION = '1.0.0'
NORM_PREFIX = "norm_"
SUPERSTACK_PREFIX = "superstack_"
STACK_METHOD = 'mean 3 3'

def seq_has_registration(seq: s.Sequence) -> bool:
    """Check if the registration data is present in the sequence"""
    for sr in seq.regparam:
        if not all(x is None for x in sr):
            for r, p in zip(sr, seq.imgparam):
                if p is None or not p.incl:
                    continue
                if r is not None:
                    H = np.array([
                        [r.H.h00, r.H.h01, r.H.h02],
                        [r.H.h10, r.H.h11, r.H.h12],
                        [r.H.h20, r.H.h21, r.H.h22]
                    ])
                    if not (np.allclose(H, np.eye(3)) or np.allclose(H, np.zeros([3, 3]))):
                        return True
    return False


class SuperStackWorker(QThread):
    """Worker thread to run the superstacking process"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)  # message, color
    progress = pyqtSignal(str, float)
    
    def __init__(self, siril, nbframes, step, norm, dryrun):
        super().__init__()
        self.siril = siril
        self.nbframes = nbframes
        self.step = step
        self.norm = norm
        self.dryrun = dryrun
        
    def run(self):
        try:
            # Create SuperStack with GUI logger
            superstack = SuperStack(
                self.siril, 
                self.nbframes, 
                self.step, 
                self.dryrun, 
                self.norm, 
                logger=self._gui_logger
            )
            superstack.run()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def _get_color_map(self, color: s.LogColor = s.LogColor.DEFAULT) -> str:
        if color == s.LogColor.RED:
            return "red"
        elif color == s.LogColor.GREEN:
            return "green"
        elif color == s.LogColor.BLUE:
            return "blue"
        else:
            return "white"
    
    def _gui_logger(self, message: str, color: s.LogColor = s.LogColor.DEFAULT):
        self.log_message.emit(message, self._get_color_map(color))


class SuperStackGUI(QMainWindow):
    def __init__(self, siril: SirilInterface):
        super().__init__()
        self.siril = siril
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"SuperStack v{VERSION}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("SuperStack - Moving Average of Images")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(10)
        
        # Number of frames
        nbframes_layout = QHBoxLayout()
        nbframes_label = QLabel("Number of frames per superstack:")
        nbframes_label.setMinimumWidth(250)
        self.nbframes_spinbox = QSpinBox()
        self.nbframes_spinbox.setMinimum(2)
        self.nbframes_spinbox.setMaximum(1000)
        self.nbframes_spinbox.setValue(3)
        self.nbframes_spinbox.setToolTip("Number of frames to average in each superstack")
        nbframes_layout.addWidget(nbframes_label)
        nbframes_layout.addWidget(self.nbframes_spinbox)
        nbframes_layout.addStretch()
        params_layout.addLayout(nbframes_layout)
        
        # Step
        step_layout = QHBoxLayout()
        step_label = QLabel("Step between each superstack:")
        step_label.setMinimumWidth(250)
        self.step_spinbox = QSpinBox()
        self.step_spinbox.setMinimum(1)
        self.step_spinbox.setMaximum(1000)
        self.step_spinbox.setValue(1)
        self.step_spinbox.setToolTip("Number of frames to advance between consecutive superstacks")
        step_layout.addWidget(step_label)
        step_layout.addWidget(self.step_spinbox)
        step_layout.addStretch()
        params_layout.addLayout(step_layout)
        
        # Normalize checkbox
        self.norm_checkbox = QCheckBox("Normalize sequence before stacking")
        self.norm_checkbox.setToolTip("Apply normalization to the sequence before performing superstacking")
        params_layout.addWidget(self.norm_checkbox)
        
        # Dry run checkbox
        self.dryrun_checkbox = QCheckBox("Dry run (preview only, don't create files)")
        self.dryrun_checkbox.setToolTip("Check the command summary without executing the superstacking process")
        params_layout.addWidget(self.dryrun_checkbox)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Log output
        log_group = QGroupBox("Output")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.run_button = QPushButton("Run")
        self.run_button.setMinimumWidth(100)
        self.run_button.setMinimumHeight(35)
        self.run_button.setToolTip("Execute the superstacking process")
        self.run_button.clicked.connect(self.run_superstack)
        button_layout.addWidget(self.run_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.setMinimumWidth(100)
        self.close_button.setMinimumHeight(35)
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout)

    def log(self, message : str, color: s.LogColor=s.LogColor.DEFAULT):
        """Add a message to the log with color"""
        color_map = {
            s.LogColor.RED: "#dc3545",
            s.LogColor.GREEN: "#28a745",
            s.LogColor.BLUE: "#007bff",
            s.LogColor.DEFAULT: "#ffffff"
        }
        html_color = color_map.get(color, color)
        self.log_text.append(f'<span style="color: {html_color};">{message}</span>')
    
    def run_superstack(self):
        """Run the superstacking process"""

        # Get parameters
        nbframes = self.nbframes_spinbox.value()
        step = self.step_spinbox.value()
        norm = self.norm_checkbox.isChecked()
        dryrun = self.dryrun_checkbox.isChecked()
        
        # Disable run button during execution
        self.run_button.setEnabled(False)
        if dryrun:
            self.run_button.setText("Checking...")
        else:
            self.run_button.setText("Running...")
        
        # Clear log
        self.log_text.clear()
        if dryrun:
            self.log("Starting dry run (no files will be created)...", s.LogColor.BLUE)
        else:
            self.log("Starting SuperStack process...", s.LogColor.BLUE)
        
        # Create and start worker thread
        self.worker = SuperStackWorker(self.siril, nbframes, step, norm, dryrun)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log_message.connect(self.log)
        self.worker.start()
    
    def on_finished(self):
        """Handle successful completion"""
        self.run_button.setEnabled(True)
        self.run_button.setText("Run")
        if not self.dryrun_checkbox.isChecked():
            QMessageBox.information(self, "Success", "Superstacking completed successfully!")
    
    def on_error(self, error_message):
        """Handle errors"""
        self.run_button.setEnabled(True)
        self.run_button.setText("Run")
        self.log(f"Error: {error_message}", s.LogColor.RED)
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{error_message}")


class SuperStack:
    """Algorithm for SuperStack"""

    def __init__(
        self, 
        siril: SirilInterface, 
        nbframes: int = 3, 
        step: int = 1, 
        dryrun: bool = False, 
        norm: bool = False, 
        logger: Callable[[str, s.LogColor], None] | None = None
    ):
        self.nbframes = nbframes
        self.step = step
        self.dryrun = dryrun
        self.norm = norm
        self.siril = siril
        self.logger = logger if logger is not None else self.siril.log

    def log(self, message: str, color: s.LogColor = s.LogColor.DEFAULT):
        """Log a message using the provided logger"""
        self.logger(message, color)

    def run(self):
        seq = self.siril.get_seq()
        if seq is None:
            raise ValueError("Failed to retrieve image sequence")
        
        seqname = seq.seqname
        
        if seq.selnum - self.nbframes < 0:
            raise ValueError("Not enough frames in the sequence to create a superstack")
        
        nbstacks = (seq.selnum - self.nbframes) // self.step + 1
        
        if nbstacks <= 1:
            raise ValueError("Not enough frames in the sequence to create a superstack")

        self.log(f'Number of frames per superstack: {self.nbframes}')
        self.log(f'Step between each stack: {self.step}')
        self.log(f'Number of selected frames in the sequence: {seq.selnum}')
        self.log(f'Number of superstacks: {nbstacks}')

        if self.dryrun:
            self.log("Dry run complete - no files created", s.LogColor.GREEN)
            return
        
        cwd = Path(self.siril.get_siril_wd())
        filenums = [seq.imgparam[i].filenum for i in range(seq.number) if seq.imgparam[i].incl]
        
        # Normalization
        if self.norm:
            norm_seqname = f'{NORM_PREFIX}{seqname}'
            if os.path.isfile(cwd / f'{norm_seqname}.seq'):
                self.log(f'\nSequence {norm_seqname} already exists - not creating again')
                seqname = norm_seqname
            else:
                self.log('Normalizing sequence', s.LogColor.GREEN)
                self.log('Computing sequence statistics')
                self.siril.cmd('seqstat', seqname, 'stats.csv', 'main')
                seq = self.siril.get_seq() # refresh the sequence to get the statistics
                if seq is None:
                    raise ValueError('Cannot reload sequence to get statistics')
                offset = np.array([[seq.stats[l][i].median for l in range(seq.nb_layers)]for i in range(seq.selnum)])
                scale = np.array([[1.5 * seq.stats[l][i].sqrtbwmv for l in range(seq.nb_layers)] for i in range(seq.selnum)])
                ref = seq.reference_image
                offset0 = offset[ref, :]
                scale0 = scale[ref, :]
                
                for i in range(seq.selnum):
                    for l in range(seq.nb_layers):
                        scale[i][l] = scale0[l] / scale[i][l] if scale[i][l] != 0. else 1.0
                        offset[i][l] = scale[i][l] * offset[i][l] - offset0[l]
                
                files = [self.siril.get_seq_frame_filename(i) for i in range(seq.number) if seq.imgparam[i].incl]
                self.siril.reset_progress()
                for i,f in enumerate(files):
                    if f is None:
                        continue
                    fit = self.siril.load_image_from_file(f, with_pixels=True)
                    if fit is None:
                        self.log(f'Cannot load image {f}', s.LogColor.RED)
                        continue
                    assert fit.data is not None, f'Image {f} has no pixel data'
                    base = os.path.basename(f)
                    self.siril.update_progress(f'Normalizing {base}', float(i) / seq.selnum)
                    self.log(f'Normalizing {base}')
                    for l in range(seq.nb_layers):
                        fit.data[l] = fit.data[l] * scale[i][l] - offset[i][l]
                    savename = cwd / f'{NORM_PREFIX}{base}'
                    self.siril.save_image_file(fit, filename=str(savename))
                
                self.siril.reset_progress()
                seqname = norm_seqname
                self.log('Normalization done', s.LogColor.GREEN)
        
        supstackseqname = f'{SUPERSTACK_PREFIX}{self.nbframes}_{self.step}_{seqname}'
        
        # Preparing tmp folder
        tmpfolder = cwd / 'tmp'
        if tmpfolder.exists():
            shutil.rmtree(tmpfolder)
        Path(tmpfolder).mkdir()
        
        # Removing all existing files
        all_files = os.listdir(cwd)
        ext = self.siril.get_siril_config('core', 'extension')
        pattern = fr'^{supstackseqname}\d{{5}}{ext}$'
        regex = re.compile(pattern)
        exact_matches = [os.path.join(cwd, f) for f in all_files if regex.match(f)]
        for f in exact_matches:
            os.remove(f)
        if os.path.isfile(f'{cwd}/{supstackseqname}.seq'):
            os.remove(f'{cwd}/{supstackseqname}.seq')
        
        # Super stacking
        self.siril.cmd('close')
        self.siril.cmd('cd', f'"{str(tmpfolder)}"')
        
        for i in range(nbstacks):
            start = i * self.step
            indexes = [filenums[j] for j in range(start, start + self.nbframes)]
            self.log(f'Processing superstack {i + 1}/{nbstacks}: {indexes[0]}-{indexes[-1]}', s.LogColor.GREEN)
            
            for j in indexes:
                src = f'{cwd}/{seqname}{j:05d}{ext}'
                dst = f'{cwd}/tmp/{seqname}{j:05d}{ext}'
                os.symlink(src, dst)
            
            supstack = f'{supstackseqname}{i + 1:05d}{ext}'
            self.siril.cmd('stack', f'"{seqname}"', STACK_METHOD, '-nonorm', f'"-out=../{supstack}"')
            
            for j in indexes:
                dst = f'./tmp/{seqname}{j:05d}{ext}'
                os.remove(dst)
            os.remove(f'{cwd}/tmp/{seqname}.seq')
        
        self.siril.cmd('cd', '..')
        self.log(f'Superstack sequence {SUPERSTACK_PREFIX}{self.nbframes}_{self.step}_{seqname} finished', s.LogColor.GREEN)
        
        cmd = self.siril.create_new_seq(supstackseqname)
        if cmd:
            self.siril.cmd('load_seq', f'"{supstackseqname}"')
            self.log('Superstacking completed', s.LogColor.GREEN)
        else:
            raise Exception(f"Failed to create new sequence {supstackseqname}")


def main():

    # Check sirilpy version once at startup
    if not s.check_module_version(f'>={SIRILPY_MIN_VERSION}'):
        print(f"Please install sirilpy version {SIRILPY_MIN_VERSION} or higher")
        sys.exit(1)

    # Connecting to Siril and running sanity checks
    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception:
        print("Failed to connect to Siril")
        sys.exit(1)

    if not siril.is_sequence_loaded():
        siril.log("No sequence is loaded", s.LogColor.RED)
        sys.exit(1)

    seq = siril.get_seq()
    assert seq is not None, "Failed to retrieve image sequence"
    if seq_has_registration(seq):
        siril.log('Registration data found in the sequence, apply it before superstacking', s.LogColor.RED)
        sys.exit(1)
    
    if seq.type != s.SequenceType.SEQ_REGULAR:
        siril.log('Superstacking only works on regular sequences, not FITSEQ nor SER', s.LogColor.RED)
        sys.exit(1)

    if siril.is_cli(): #CLI mode
        import argparse
        parser = argparse.ArgumentParser(description="Performs superstacking of images of a sequence, i.e moving average of images.")
        parser.add_argument("nbframes", type=int, nargs='?', default=3,
                            help="Number of frames in each superstack")
        parser.add_argument("step", type=int, nargs='?', default=1,
                            help="Step between each superstack")
        parser.add_argument("-n", "--norm", action="store_true",
                            help="Normalize the sequence before stacking")
        parser.add_argument("-d", "--dryrun", action="store_true",
                            help="Check the command summary without executing it")
        
        args = parser.parse_args()
        try:
            superstack = SuperStack(siril, **vars(args))
            superstack.run()
        except Exception as e:
            siril.log(f"Error: {str(e)}", s.LogColor.RED)
            sys.exit(1)
    else:  # GUI mode
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = SuperStackGUI(siril)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
