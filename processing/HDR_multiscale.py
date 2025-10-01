"""
HDR Multiscale for Siril
from Franklin Marek SAS code (2025)
Adapted for Siril by Cyril Richard

(c) Cyril Richard 2025
SPDX-License-Identifier: GPL-3.0-or-later

HDR compression using à trous wavelet decomposition
with luminance masking and Lab color space processing
"""

import sirilpy as s
s.ensure_installed('PyQt6')
s.ensure_installed('scipy')

import sys
import numpy as np
from scipy.ndimage import convolve as nd_convolve

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QSlider, QPushButton,
                            QGroupBox, QMessageBox, QSpinBox,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                            QSplitter, QProgressBar)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QImage, QPixmap

VERSION = "1.0.0"

if not s.check_module_version('>=0.6.42'):
    print("Error: requires sirilpy module >= 0.6.42")
    sys.exit(1)

# ============================================================================
# Core Math Functions
# ============================================================================

_B3 = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

def _conv_sep_reflect(image2d, k1d, axis):
    """Separable 1D convolution with reflect padding"""
    if axis == 1:
        return nd_convolve(image2d, k1d.reshape(1, -1), mode='reflect')
    else:
        return nd_convolve(image2d, k1d.reshape(-1, 1), mode='reflect')

def _build_spaced_kernel(kernel, scale_idx):
    """Build dilated kernel for à trous algorithm"""
    if scale_idx == 0:
        return kernel.astype(np.float32, copy=False)
    step = 2 ** scale_idx
    spaced_len = len(kernel) + (len(kernel) - 1) * (step - 1)
    spaced = np.zeros(spaced_len, dtype=np.float32)
    spaced[0::step] = kernel
    return spaced

def _atrous_decompose(img2d, n_scales, base_k):
    """À trous wavelet decomposition"""
    current = img2d.astype(np.float32, copy=True)
    scales = []
    for s in range(n_scales):
        k = _build_spaced_kernel(base_k, s)
        tmp = _conv_sep_reflect(current, k, axis=1)
        smooth = _conv_sep_reflect(tmp, k, axis=0)
        scales.append(current - smooth)
        current = smooth
    scales.append(current)
    return scales

def _atrous_reconstruct(scales):
    """Reconstruct from wavelet scales"""
    out = scales[-1].astype(np.float32, copy=True)
    for w in scales[:-1]:
        out += w
    return out

def _rgb_to_lab(rgb):
    """RGB to Lab color space conversion"""
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    xyz = rgb.reshape(-1, 3) @ M.T
    xyz = xyz.reshape(rgb.shape)
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883
    
    delta = 6/29
    def f(t):
        return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + (4/29))
    
    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1)

def _lab_to_rgb(lab):
    """Lab to RGB color space conversion"""
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    delta = 6/29
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    
    def finv(t):
        return np.where(t > delta, t**3, 3*delta**2*(t - 4/29))
    
    X = 0.95047 * finv(fx)
    Y = finv(fy)
    Z = 1.08883 * finv(fz)
    xyz = np.stack([X, Y, Z], axis=-1)
    rgb = xyz.reshape(-1, 3) @ M_inv.T
    rgb = rgb.reshape(xyz.shape)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

def _mask_from_L(L, gamma):
    """Create luminance mask from L channel"""
    m = np.clip(L / 100.0, 0.0, 1.0).astype(np.float32)
    if gamma != 1.0:
        m = np.power(m, gamma, dtype=np.float32)
    return m

def _apply_dim_curve(rgb, gamma):
    """Apply dimming curve to tame highlights"""
    return np.power(np.clip(rgb, 0.0, 1.0), gamma, dtype=np.float32)

def compute_wavescale_hdr(rgb_image, n_scales=5, compression_factor=1.5,
                         mask_gamma=1.0, base_kernel=_B3, decay_rate=0.5,
                         dim_gamma=None):
    """
    Main HDR processing function
    Returns (transformed_rgb, luminance_mask)
    """
    lab = _rgb_to_lab(rgb_image)
    L0 = lab[..., 0].astype(np.float32, copy=True)
    scales = _atrous_decompose(L0, n_scales, base_kernel)
    
    mask = _mask_from_L(L0, mask_gamma)
    planes, residual = scales[:-1], scales[-1]
    
    # Process detail layers with decay
    for i, wp in enumerate(planes):
        decay = decay_rate ** i
        scale = (1.0 + (compression_factor - 1.0) * mask * decay) * 2.0
        planes[i] = wp * scale
    
    Lr = _atrous_reconstruct(planes + [residual])
    
    # Midtones alignment
    med0 = float(np.median(L0))
    med1 = float(np.median(Lr)) or 1.0
    Lr = np.clip(Lr * (med0 / med1), 0.0, 100.0)
    
    lab[..., 0] = Lr
    rgb = _lab_to_rgb(lab)
    
    # Dimming curve
    g = (1.0 + n_scales * 0.2) if dim_gamma is None else float(dim_gamma)
    rgb = _apply_dim_curve(rgb, gamma=g)
    
    return rgb, mask

# ============================================================================
# Worker Thread
# ============================================================================

class HDRWorker(QObject):
    progress_update = pyqtSignal(str, int)
    finished = pyqtSignal(object, object)
    
    def __init__(self, rgb_image, n_scales, compression_factor, mask_gamma, base_kernel):
        super().__init__()
        self.rgb_image = rgb_image
        self.n_scales = n_scales
        self.compression_factor = compression_factor
        self.mask_gamma = mask_gamma
        self.base_kernel = base_kernel
    
    def run(self):
        try:
            self.progress_update.emit("Converting to Lab color space...", 10)
            self.progress_update.emit("Decomposing with wavelets...", 30)
            
            transformed, mask = compute_wavescale_hdr(
                self.rgb_image, self.n_scales, self.compression_factor,
                self.mask_gamma, self.base_kernel, decay_rate=0.5
            )
            
            self.progress_update.emit("Finalizing...", 95)
            self.finished.emit(transformed, mask)
        except Exception as e:
            print(f"HDR Multiscale error: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit(None, None)

# ============================================================================
# Zoomable Graphics View
# ============================================================================

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.setScene(scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._zoom = 1.0
        self._zoom_min = 0.05
        self._zoom_max = 10.0
        self._zoom_step = 1.25
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()
    
    def zoom_in(self):
        new_zoom = min(self._zoom * self._zoom_step, self._zoom_max)
        self._apply_zoom(new_zoom)
    
    def zoom_out(self):
        new_zoom = max(self._zoom / self._zoom_step, self._zoom_min)
        self._apply_zoom(new_zoom)
    
    def fit_item(self, item):
        if item and not item.pixmap().isNull():
            self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0
    
    def _apply_zoom(self, new_zoom):
        factor = new_zoom / self._zoom
        self.scale(factor, factor)
        self._zoom = new_zoom

# ============================================================================
# Main Dialog
# ============================================================================

class WaveScaleHDRDialog(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.setWindowTitle(f"HDR Multiscale v{VERSION}")
        self.resize(1000, 700)
        
        self.siril = siril
        
        if not self.siril.is_image_loaded():
            QMessageBox.critical(self, "Error", "No image loaded in Siril")
            raise SystemExit(1)
        
        try:
            self.siril.cmd("requires", "1.3.6")
        except Exception:
            pass
        
        # Get image data and ensure it's float32 [0,1]
        fit = self.siril.get_image()
        fit.ensure_data_type(np.float32)
        img = fit.data
        
        # Handle mono/color
        if img.ndim == 2:
            img_rgb = np.repeat(img[:, :, None], 3, axis=2)
            self._was_mono = True
        elif img.ndim == 3:
            if img.shape[0] == 1:
                img_rgb = np.repeat(img[0, :, :, None], 3, axis=2)
                self._was_mono = True
            elif img.shape[0] == 3:
                img_rgb = np.transpose(img, (1, 2, 0))
                self._was_mono = False
            else:
                img_rgb = img
                self._was_mono = False
        
        img_rgb = np.clip(img_rgb, 0.0, 1.0).astype(np.float32, copy=False)
        
        self.original_rgb = img_rgb
        self.preview_rgb = img_rgb.copy()
        self.base_kernel = _B3
        
        self.init_ui()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([350, 650])
        
        self._set_pix(self.preview_rgb)
    
    def create_left_panel(self):
        left_widget = QWidget()
        left_widget.setFixedWidth(350)
        layout = QVBoxLayout(left_widget)
        layout.setSpacing(10)
        
        title = QLabel("HDR Multiscale")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        controls_group = QGroupBox("Parameters")
        controls_layout = QVBoxLayout(controls_group)
        
        # Number of scales
        scales_layout = QHBoxLayout()
        scales_layout.addWidget(QLabel("Number of Scales:"))
        self.s_scales = QSpinBox()
        self.s_scales.setRange(2, 10)
        self.s_scales.setValue(5)
        self.s_scales.setToolTip("Number of wavelet decomposition scales")
        scales_layout.addWidget(self.s_scales)
        scales_layout.addStretch()
        controls_layout.addLayout(scales_layout)
        
        # Compression factor
        comp_layout = QHBoxLayout()
        comp_layout.addWidget(QLabel("Compression:"))
        self.s_comp = QSlider(Qt.Orientation.Horizontal)
        self.s_comp.setRange(10, 500)
        self.s_comp.setValue(150)
        self.s_comp.valueChanged.connect(self.update_comp_display)
        self.s_comp.setToolTip("HDR compression strength")
        comp_layout.addWidget(self.s_comp)
        self.comp_label = QLabel("1.50")
        self.comp_label.setFixedWidth(40)
        comp_layout.addWidget(self.comp_label)
        controls_layout.addLayout(comp_layout)
        
        # Mask gamma
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Mask Gamma:"))
        self.s_gamma = QSlider(Qt.Orientation.Horizontal)
        self.s_gamma.setRange(10, 1000)
        self.s_gamma.setValue(500)
        self.s_gamma.valueChanged.connect(self.update_gamma_display)
        self.s_gamma.setToolTip("Controls where the HDR effect is applied based on luminance")
        gamma_layout.addWidget(self.s_gamma)
        self.gamma_label = QLabel("5.0")
        self.gamma_label.setFixedWidth(40)
        gamma_layout.addWidget(self.gamma_label)
        controls_layout.addLayout(gamma_layout)
        
        layout.addWidget(controls_group)
        
        # Preview buttons
        preview_layout = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._start_preview)
        preview_layout.addWidget(self.btn_preview)
        
        self.btn_toggle = QPushButton("Show Original")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self._toggle)
        preview_layout.addWidget(self.btn_toggle)
        layout.addLayout(preview_layout)
        
        # Progress
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.lbl_step = QLabel("Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.lbl_step)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Siril")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_to_siril)
        bottom_layout.addWidget(self.btn_apply)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset)
        bottom_layout.addWidget(self.btn_reset)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        bottom_layout.addWidget(self.btn_close)
        layout.addLayout(bottom_layout)
        
        footer_label = QLabel("Written by Franklin Marek\nSiril port by Cyril Richard\nwww.setiastro.com")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer_label)
        
        layout.addStretch()
        
        return left_widget
    
    def create_right_panel(self):
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        zoom_layout = QHBoxLayout()
        btn_zoom_in = QPushButton("Zoom In")
        btn_zoom_in.clicked.connect(lambda: self.view.zoom_in())
        zoom_layout.addWidget(btn_zoom_in)
        
        btn_zoom_out = QPushButton("Zoom Out")
        btn_zoom_out.clicked.connect(lambda: self.view.zoom_out())
        zoom_layout.addWidget(btn_zoom_out)
        
        btn_fit = QPushButton("Fit to Preview")
        btn_fit.clicked.connect(lambda: self.view.fit_item(self.pix))
        zoom_layout.addWidget(btn_fit)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene, self)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        layout.addWidget(self.view)
        
        return right_widget
    
    def update_comp_display(self):
        value = self.s_comp.value() / 100.0
        self.comp_label.setText(f"{value:.2f}")
    
    def update_gamma_display(self):
        value = self.s_gamma.value() / 100.0
        self.gamma_label.setText(f"{value:.1f}")
    
    def _set_pix(self, rgb):
        display_rgb = np.flipud(rgb)
        arr = (np.clip(display_rgb, 0, 1) * 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())
    
    def _toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText("Show Preview")
            self._set_pix(self.original_rgb)
        else:
            self.btn_toggle.setText("Show Original")
            self._set_pix(self.preview_rgb)
    
    def _reset(self):
        self.s_scales.setValue(5)
        self.s_comp.setValue(150)
        self.s_gamma.setValue(500)
        self.preview_rgb = self.original_rgb.copy()
        self._set_pix(self.preview_rgb)
        self.lbl_step.setText("Idle")
        self.progress_bar.setValue(0)
        self.btn_apply.setEnabled(False)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Show Original")
    
    def _start_preview(self):
        self.btn_preview.setEnabled(False)
        self.btn_apply.setEnabled(False)
        
        n_scales = int(self.s_scales.value())
        comp = float(self.s_comp.value()) / 100.0
        mgamma = float(self.s_gamma.value()) / 100.0
        
        self.thread = QThread(self)
        self.worker = HDRWorker(self.original_rgb, n_scales, comp, mgamma, self.base_kernel)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    
    def _on_progress(self, step, pct):
        self.lbl_step.setText(step)
        self.progress_bar.setValue(pct)
    
    def _on_finished(self, transformed_rgb, mask):
        self.btn_preview.setEnabled(True)
        if transformed_rgb is None:
            QMessageBox.critical(self, "HDR Multiscale", "Processing failed")
            return
        
        # Blend using luminance mask (like original code)
        # This protects low-luminance areas (sky background) from being affected
        m3 = np.repeat(mask[..., None], 3, axis=2)
        self.preview_rgb = self.original_rgb * (1.0 - m3) + transformed_rgb * m3
        
        self._set_pix(self.preview_rgb)
        self.btn_apply.setEnabled(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Show Original")
        self.lbl_step.setText("Preview ready")
        self.progress_bar.setValue(100)
    
    def _apply_to_siril(self):
        out = self.preview_rgb
        
        if self._was_mono:
            mono = np.mean(out, axis=2, dtype=np.float32)
            out = mono
        
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        
        if out.ndim == 3:
            out = np.transpose(out, (2, 0, 1))
        
        try:
            with self.siril.image_lock():
                self.siril.undo_save_state(f"HDR Multiscale: {self.s_scales.value()} scales")
                self.siril.set_image_pixeldata(out)
            
            self.siril.log("HDR Multiscale processing completed", s.LogColor.GREEN)
            QMessageBox.information(self, "Success", "Image applied to Siril successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply to Siril:\n{e}")

def main():
    try:
        siril = s.SirilInterface()
        try:
            siril.connect()
        except s.SirilConnectionError:
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Error", "Failed to connect to Siril")
            sys.exit(1)
        
        if siril.is_cli():
            print("This script requires GUI mode")
            sys.exit(1)
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setApplicationName("HDR Multiscale")
        window = WaveScaleHDRDialog(siril)
        window.show()
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
