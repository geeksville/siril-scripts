# (c) Cecile Melis 2025
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Plots the currently loaded image in 3D using matplotlib (only the selection if one is drawn)

Version history:
0.0.1: Initial version
"""

import sirilpy as s
import numpy as np
import os, sys
import warnings

VERSION = "0.0.1"

# Check sirilpy version once at startup
if not s.check_module_version(f'>=1.0.0'):
    print(f"Please install sirilpy version 1.0.0 or higher")
    sys.exit(1)

s.ensure_installed('matplotlib')
s.ensure_installed('PyQt6')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QCheckBox, QLabel, QGroupBox, QPushButton)
from PyQt6.QtCore import Qt

siril = s.SirilInterface()

try:
    siril.connect()
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    quit()


class Plot3DWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Siril 3D Plot v{VERSION}')
        self.setGeometry(100, 100, 1400, 800)

        # Set always on top by default
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self.data = None
        self.nlayers = 0
        self.img = None
        self.img_width = 0
        self.img_height = 0
        self.sel = None
        self.current_channel = 1
        self.show_contour = False
        self.use_log_scale = False
        self.ax1 = None
        self.ax2 = None
        self.syncing = False  # Flag to prevent recursive updates
        
        self.init_ui()
        self.load_image_data()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Control panel
        control_layout = QVBoxLayout()
        
        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(['Red', 'Green', 'Blue'])
        self.channel_combo.setCurrentIndex(1)
        self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
        self.channel_combo.setEnabled(False)
        channel_layout.addWidget(QLabel("Select Channel:"))
        channel_layout.addWidget(self.channel_combo)
        channel_group.setLayout(channel_layout)
        control_layout.addWidget(channel_group)

        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        self.contour_check = QCheckBox("Show Contour Plot")
        self.contour_check.stateChanged.connect(self.on_contour_toggled)
        display_layout.addWidget(self.contour_check)
        
        self.log_scale_check = QCheckBox("Log Scale Z")
        self.log_scale_check.stateChanged.connect(self.on_log_scale_toggled)
        display_layout.addWidget(self.log_scale_check)

        self.ontop_check = QCheckBox("Always on Top")
        self.ontop_check.setChecked(True)
        self.ontop_check.stateChanged.connect(self.on_ontop_toggled)
        display_layout.addWidget(self.ontop_check)

        display_group.setLayout(display_layout)
        control_layout.addWidget(display_group)

        # Refresh button
        refresh_group = QGroupBox("Data")
        refresh_layout = QVBoxLayout()
        self.refresh_button = QPushButton("Refresh Data")
        self.refresh_button.clicked.connect(self.on_refresh_clicked)
        refresh_layout.addWidget(self.refresh_button)
        # Save button
        self.save_button = QPushButton("Quick Save to PNG")
        self.save_button.clicked.connect(self.on_save_clicked)
        refresh_layout.addWidget(self.save_button)
        
        refresh_group.setLayout(refresh_layout)
        control_layout.addWidget(refresh_group)
        
        control_layout.addStretch()

        # Plot canvas with toolbar
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create a layout for canvas and toolbar
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(canvas_layout, 3)
    
    def load_image_data(self):
        try:
            if not siril.is_image_loaded() and not siril.is_sequence_loaded():
                siril.log("No image or sequence loaded, aborting")
                return
            
            sel = siril.get_siril_selection()
            if sel is not None:
                sel = list(sel)
            self.sel = sel
            if siril.is_image_loaded():
                self.data = siril.get_image_pixeldata(self.sel)
                self.img = siril.get_image_filename()
            elif siril.is_sequence_loaded():
                seq = siril.get_seq()
                assert seq is not None, "Failed to retrieve image sequence"
                self.data = siril.get_seq_frame_pixeldata(seq.current, self.sel)
                self.img = siril.get_seq_frame_filename(seq.current)
            
            if self.data is None:
                print("No image loaded, aborting")
                return
            
            assert self.data.shape is not None
            if len(self.data.shape) == 3:
                self.nlayers = self.data.shape[0]
                self.img_width = self.data.shape[1]
                self.img_height = self.data.shape[2]
                self.channel_combo.setEnabled(True)
            elif len(self.data.shape) == 2:
                self.nlayers = 1
                self.img_width = self.data.shape[0]
                self.img_height = self.data.shape[1]
                self.channel_combo.setEnabled(False)
            else:
                raise ValueError("Unsupported data shape")
            self.plot_3d()
        
        except Exception as e:
            siril.log(f"Error loading image: {e}")
    
    def get_plot_data(self):
        assert self.data is not None
        if self.nlayers == 3:
            return self.data[self.current_channel, :, :]
        else:
            return self.data

    def plot_3d(self):
        self.figure.clear()
        if self.data is None:
            return
        title = f"{os.path.basename(self.img) if self.img else 'Image'}"
        if self.nlayers == 3:
            channel_names = ['R', 'G', 'B']
            title += f'[{channel_names[self.current_channel]}]'
        if self.sel is not None:
            title += f"\nSelection [{self.sel[0]},{self.sel[1]},{self.sel[2]},{self.sel[3]}]"
        self.figure.suptitle(title, fontsize=12)
        
        plot_data = self.get_plot_data()
        x = np.arange(self.img_height)
        y = np.arange(self.img_width)
        X, Y = np.meshgrid(x, y)
        
        if self.show_contour:
            self.ax1 = self.figure.add_subplot(121, projection='3d')
            self.ax2 = self.figure.add_subplot(122, projection='3d')
        else:
            self.ax1 = self.figure.add_subplot(111, projection='3d')
            self.ax2 = None
        
        # Main surface plot
        plot_data_display = plot_data.copy()
        if self.use_log_scale:
            plot_data_display = np.log10(np.maximum(plot_data_display, 1e-10))
        self.ax1.plot_surface(X, Y, plot_data_display, cmap='viridis')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z (log10)' if self.use_log_scale else 'Z')
        self.ax1.set_title('Surface Plot')
        x_range = self.ax1.get_xlim()[1] - self.ax1.get_xlim()[0]
        y_range = self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0]
        
        # Set correct aspect ratio for ax1
        self.ax1.set_xlim([0, self.img_height])
        self.ax1.set_ylim([0, self.img_width])
        z_max = np.nanmax(plot_data_display)
        z_min = np.nanmin(plot_data_display)
        self.ax1.set_zlim([z_min, z_max])
        self.ax1.set_box_aspect([x_range, y_range, max(x_range, y_range)])
        self.ax1.tick_params(axis='z', pad=10)

        if self.use_log_scale:
            self._set_log_scale_ticks(self.ax1)
        
        # Contour plot if enabled
        if self.show_contour and self.ax2:
            self.ax2.contour(X, Y, plot_data_display, zdir='z', levels=10, cmap='viridis')
            self.ax2.set_xlabel('X')
            self.ax2.set_ylabel('Y')
            self.ax2.set_zlabel('Z (log10)' if self.use_log_scale else 'Z')
            self.ax2.set_title('Contour Plot')
            
            # Set correct aspect ratio for ax2
            self.ax2.set_xlim([0, self.img_height])
            self.ax2.set_ylim([0, self.img_width])
            self.ax2.set_zlim([z_min, z_max])
            self.ax2.set_box_aspect([x_range, y_range, max(x_range, y_range)])
            if self.use_log_scale:
                self._set_log_scale_ticks(self.ax2)
            self.ax2.tick_params(axis='z', pad=10)
            
            # Connect mouse motion events for view synchronization
            self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Tight layout not applied')
            self.figure.tight_layout()
        self.canvas.draw()
    
    def on_motion(self, event):
        """Synchronize view between ax1 and ax2"""
        if self.syncing or not self.ax2:
            return
        
        self.syncing = True
        try:
            if event.inaxes == self.ax1:
                self.ax2.view_init(elev=self.ax1.elev, azim=self.ax1.azim, roll=self.ax1.roll)
            elif event.inaxes == self.ax2:
                self.ax1.view_init(elev=self.ax2.elev, azim=self.ax2.azim, roll=self.ax2.roll)
            self.canvas.draw_idle()
        finally:
            self.syncing = False
    
    def on_channel_changed(self, label):
        channel_map = {'Red': 0, 'Green': 1, 'Blue': 2}
        self.current_channel = channel_map[label]
        self.plot_3d()
    
    def on_contour_toggled(self, state):
        self.show_contour = state == Qt.CheckState.Checked.value
        self.plot_3d()

    def on_refresh_clicked(self):
        """Reload image data and refresh the plot"""
        self.load_image_data()
    
    def on_log_scale_toggled(self, state):
        self.use_log_scale = state == Qt.CheckState.Checked.value
        self.plot_3d()

    def _set_log_scale_ticks(self, ax):
        """Set ticks at round numbers in original scale, positioned in log space"""
        z_min, z_max = ax.get_zlim()
        
        # Convert log positions back to linear values
        linear_min = 10 ** z_min
        linear_max = 10 ** z_max
        
        # Generate nice round ticks
        linear_ticks = []
        
        # Determine the order of magnitude range
        min_order = int(np.floor(np.log10(linear_min)))
        max_order = int(np.floor(np.log10(linear_max)))

        for order in range(min_order, max_order + 1):
            base = 10 ** order
            for multiplier in range(1, 6):
                tick_value = multiplier * base
                if linear_min <= tick_value <= linear_max:
                    linear_ticks.append(tick_value)
        
        # Convert to log positions for display
        log_ticks = np.log10(linear_ticks)
        
        # Set ticks at log positions with labels showing multiplier × 10^order
        labels = []
        for val in linear_ticks:
            order = int(np.floor(np.log10(val)))
            multiplier = val / (10 ** order)
            labels.append(f'{multiplier:.0f}×10$^{{{order}}}$')
        
        ax.set_zticks(log_ticks)
        ax.set_zticklabels(labels)
        ax.tick_params(axis='z', pad=10)

    def on_save_clicked(self):
        """Save the figure to a PNG file"""
        assert self.img is not None, "No image loaded to save figure for"
        try:
            folder, savename = os.path.split(self.img)
            img_base, _ = os.path.splitext(savename)
            if self.nlayers == 3:
                channel_names = ['R', 'G', 'B']
                img_base += f"_{channel_names[self.current_channel]}"
            if self.sel is not None:
                img_base += f"_x{self.sel[0]}y{self.sel[1]}w{self.sel[2]}h{self.sel[3]}"
            filename = os.path.join(folder, f"{img_base}.png")
            self.figure.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {filename}")
        except Exception as e:
            siril.log(f"Error saving figure: {e}")
            print(f"Error saving figure: {e}")

    def on_ontop_toggled(self, state):
        if state == Qt.CheckState.Checked.value:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

def siril_plot3D():
    app = QApplication.instance() or QApplication(sys.argv)
    window = Plot3DWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    siril_plot3D()