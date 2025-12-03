##############################################
# VeraLux — HyperMetric Stretch
# Photometric Hyperbolic Stretch Engine
# Author: Riccardo Paterniti (2025)
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — HyperMetric Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.1
#
# Credits / Origin
# ----------------
#   • Inspired by: The "True Color" methodology of Dr. Roger N. Clark
#   • Math basis: Generalized Hyperbolic Stretch (GHS) & Vector Color Preservation
#   • Sensor Science: Hardware-specific Quantum Efficiency weighting
#

"""
Overview
--------
A precision linear-to-nonlinear stretching engine designed to maximize sensor 
fidelity while managing the transition to the visible domain.

HyperMetric Stretch (HMS) operates on a fundamental axiom: standard histogram 
transformations often destroy the photometric relationships between color channels 
(hue shifts) and clip high-dynamic range data. HMS solves this by decoupling 
Luminance geometry from Chromatic vectors.

The tool introduces a "Dual Philosophy" architecture: it serves both the 
photometric purist (Scientific Mode) and the aesthetic imager (Ready-to-Use Mode) 
without compromising the mathematical integrity of either workflow.

Design Goals
------------
• Preserve original vector color ratios during extreme stretching (True Color)
• Optimize Luminance extraction based on specific hardware (Sensor Profiles)
• Provide a mathematically "Safe" expansion for high-dynamic targets
• Eliminate the "flat look" of logarithmic stretches via Adaptive Scaling
• Ensure strictly linear input handling (requires SPCC calibration)

Core Features
-------------
• Dual Processing Philosophy:
  - Scientific Mode: 100% lossless, hard-clip at physical saturation (1.0), 
    no post-processing. Ideal for photometry and pure data analysis.
  - Ready-to-Use Mode: Applies "Star-Safe" expansion, Linked MTF optimization, 
    and highlight soft-clipping. Delivers an aesthetic, export-ready image.
• Hardware-Aware Color Science:
  - Custom weighting profiles for IMX571, IMX662, and generic standards 
    (Rec.709, ProPhoto) to match the stretching math to the sensor's QE.
• Star-Safe Expansion Engine:
  - Intelligent dynamic range analysis that distinguishes between diffuse nebulae 
    and stellar cores, preventing "bloating" on bright targets.
• Physics-Based Math:
  - Log-GHS Engine: Generalized Hyperbolic Stretch focused on midtone contrast.
  - Color Convergence: Controls the roll-off to white point for star cores.

Usage
-----
1. Pre-requisite: Image MUST be Linear and Color Calibrated (SPCC).
2. Setup: Select your Sensor Profile (or Rec.709) and Processing Mode.
   (Default is "Ready-to-Use" for immediate results).
3. Calibrate: Click Calculate Optimal Log D (Auto-Solver) to analyze the 
   linear data and find the mathematical sweet spot for the stretch.
4. Refine: Adjust Stretch Power and Highlight Protection (b) if needed.
5. Process: Click PROCESS.
   - If Scientific: The result is a raw container for further tone mapping.
   - If Ready-to-Use: The result is ready for export or fine-tuning.

Inputs & Outputs
----------------
Input:
• Linear FITS images (RGB or Mono), pre-processed and color calibrated (SPCC).

Output:
• Non-linear (Stretched) 32-bit Float FITS.

Compatibility
-------------
• Siril 1.3+
• Python 3.10+ (via sirilpy)
• Dependencies:
  - sirilpy
  - PyQt6
  - numpy

License
-------
Released under GPL-3.0-or-later.

This script is part of the VeraLux family of tools —
focused on maximizing data fidelity through physics-based processing.
"""

import sys
import os
import traceback

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    # Fallback for CLI testing or missing modules
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

# Ensure dependencies are present
s.ensure_installed("PyQt6", "numpy")

import numpy as np
import math

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QDoubleSpinBox, QSlider,
                            QPushButton, QGroupBox, QMessageBox, QProgressBar,
                            QComboBox, QRadioButton, QButtonGroup, QCheckBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ---------------------
#  THEME & STYLING
# ---------------------
DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }

/* Windows Fix: Explicitly style indicators to ensure visibility on custom dark backgrounds */
QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 7px; }
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked { background-color: #285299; border: 1px solid #88aaff; image: none; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; image: none; }
QRadioButton::indicator:checked { background: qradialgradient(cx:0.5, cy:0.5, radius: 0.4, fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #285299); }
QCheckBox::indicator:checked { background: #285299; }

QDoubleSpinBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox:hover { border-color: #777777; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #aaaaaa; margin-right: 6px; }
QComboBox QAbstractItemView { background-color: #3c3c3c; color: #ffffff; selection-background-color: #285299; border: 1px solid #555555; }

/* SLIDER FIX: Added min-height to prevent handle clipping */
QSlider { min-height: 22px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #aaaaaa; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #ffffff; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#AutoButton { background-color: #8c6a00; border: 1px solid #a37c00; }
QPushButton#AutoButton:hover { background-color: #bfa100; color: #000000;}
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }
QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }
"""

VERSION = "1.0.1"

# =============================================================================
#  WORKING SPACE PROFILES
# =============================================================================

SENSOR_PROFILES = {
    "Rec.709 (Recommended)": {
        'weights': (0.2126, 0.7152, 0.0722),
        'description': "ITU-R BT.709 standard for sRGB/HDTV",
        'info': "Default choice. Best for general use and consumer cameras.",
        'category': 'standard'
    },
    "ProPhoto RGB": {
        'weights': (0.2880, 0.7118, 0.0002),
        'description': "Wide gamut color space (90% Pointer)",
        'info': "Use for highly saturated emission nebulae. Low blue weight may increase noise.",
        'category': 'wide-gamut'
    },
    "Sony IMX571": {
        'weights': (0.2450, 0.6850, 0.0700),
        'description': "Optimized for Sony IMX571 sensor (A6700, ASI533MC, QHY268C)",
        'info': "Compensates for IMX571 QE curve and IR-cut filter. +9% luminance on Hα regions.",
        'category': 'sensor-specific'
    },
    "Sony IMX662 (Starvis 2)": {
        'weights': (0.2950, 0.6200, 0.0850),
        'description': "Optimized for Sony IMX662 Starvis 2 (NIR-extended)",
        'info': "For security/astro cameras with NIR response. Excellent for SII narrowband.",
        'category': 'sensor-specific'
    },
    "Narrowband HOO": {
        'weights': (0.5000, 0.2500, 0.2500),
        'description': "Bicolor palette: Hα=Red, OIII=Green+Blue",
        'info': "Balanced weighting for HOO synthetic palette processing.",
        'category': 'narrowband'
    },
    "Narrowband SHO": {
        'weights': (0.3333, 0.3400, 0.3267),
        'description': "Hubble palette: SII=Red, Hα=Green, OIII=Blue",
        'info': "Nearly uniform weighting for SHO tricolor narrowband.",
        'category': 'narrowband'
    }
}

DEFAULT_PROFILE = "Rec.709 (Recommended)"

# =============================================================================
#  MATH ENGINE
# =============================================================================

def ghs_stretch(data, D, b, SP=0.0):
    """Generalized Hyperbolic Stretch."""
    D = max(D, 0.1)
    b = max(b, 0.1)
    term1 = np.arcsinh(D * (data - SP) + b)
    term2 = np.arcsinh(b)
    norm_factor = np.arcsinh(D * (1.0 - SP) + b) - term2
    if norm_factor == 0: norm_factor = 1e-6
    stretched = (term1 - term2) / norm_factor
    return stretched

def solve_stretch_factor(data_sample, target_median, b_val, luma_weights=(0.2126, 0.7152, 0.0722)):
    median_in = np.median(data_sample)
    if median_in < 1e-8: return 2.0 
    low_log = 0.0; high_log = 7.0; best_log_D = 2.0
    for _ in range(40):
        mid_log = (low_log + high_log) / 2.0
        mid_D = 10.0 ** mid_log
        test_val = ghs_stretch(median_in, mid_D, b_val)
        if abs(test_val - target_median) < 0.002:
            best_log_D = mid_log; break
        if test_val < target_median: low_log = mid_log
        else: high_log = mid_log
    return best_log_D

def apply_mtf(data, m):
    term1 = (m - 1.0) * data
    term2 = (2.0 * m - 1.0) * data - m
    with np.errstate(divide='ignore', invalid='ignore'):
        res = term1 / term2
    return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

def adaptive_output_scaling(img_data, working_space="Rec.709 (Standard)", 
                            target_bg=0.20, progress_callback=None):
    """Adaptive Star-Safe Scaling"""
    if progress_callback: progress_callback("Adaptive Scaling: Analyzing Dynamic Range (Anti-Bloat)...")
    luma_r, luma_g, luma_b = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img_data.ndim == 3 and img_data.shape[0] == 3)
    
    if is_rgb:
        R, G, B = img_data[0].copy(), img_data[1].copy(), img_data[2].copy()
        L_raw = luma_r * R + luma_g * G + luma_b * B
    else:
        L_raw = img_data[0] if img_data.ndim == 3 else img_data
    
    median_L = float(np.median(L_raw))
    std_L    = float(np.std(L_raw))
    min_L    = float(np.min(L_raw))
    global_floor = max(min_L, median_L - 2.7 * std_L)
    
    PEDESTAL = 0.001
    TARGET_SOFT = 0.98; TARGET_HARD = 1.0
    
    if is_rgb:
        soft_r = np.percentile(R, 99.0); soft_g = np.percentile(G, 99.0); soft_b = np.percentile(B, 99.0)
        soft_ceil = max(soft_r, soft_g, soft_b)
        hard_r = np.percentile(R, 99.99); hard_g = np.percentile(G, 99.99); hard_b = np.percentile(B, 99.99)
        hard_ceil = max(hard_r, hard_g, hard_b)
    else:
        soft_ceil = np.percentile(L_raw, 99.0)
        hard_ceil = np.percentile(L_raw, 99.99)
        
    if soft_ceil <= global_floor: soft_ceil = global_floor + 1e-6
    if hard_ceil <= soft_ceil: hard_ceil = soft_ceil + 1e-6
    
    scale_contrast = (TARGET_SOFT - PEDESTAL) / (soft_ceil - global_floor + 1e-9)
    scale_safety = (TARGET_HARD - PEDESTAL) / (hard_ceil - global_floor + 1e-9)
    final_scale = min(scale_contrast, scale_safety)
    
    if progress_callback:
        mode = "PROTECTION" if scale_safety < scale_contrast else "CONTRAST"
        progress_callback(f"Expansion Mode: {mode} (Scale: {final_scale:.2f})")
    
    def expand_channel(c):
        return np.clip((c - global_floor) * final_scale + PEDESTAL, 0.0, 1.0)
        
    if is_rgb:
        R = expand_channel(R); G = expand_channel(G); B = expand_channel(B)
        L_norm = luma_r * R + luma_g * G + luma_b * B
    else:
        L_norm = expand_channel(L_raw)
        
    if is_rgb: L = luma_r * R + luma_g * G + luma_b * B
    else: L = L_norm
    
    current_bg = float(np.median(L))
    if current_bg > 0.0 and current_bg < 1.0 and abs(current_bg - target_bg) > 1e-3:
        if progress_callback: progress_callback(f"Applying MTF (Bg: {current_bg:.3f} -> {target_bg})")
        x = current_bg; y = target_bg
        m = (x * (y - 1.0)) / (x * (2.0 * y - 1.0) - y)
        if is_rgb:
            img_data[0] = apply_mtf(R, m); img_data[1] = apply_mtf(G, m); img_data[2] = apply_mtf(B, m)
        else:
            L_out = apply_mtf(L_norm, m)
            if img_data.ndim == 3: img_data[0] = L_out
            else: img_data[:] = L_out
    else:
        if is_rgb: img_data[0], img_data[1], img_data[2] = R, G, B
        else:
            if img_data.ndim == 3: img_data[0] = L_norm
            else: img_data[:] = L_norm
    return img_data

def apply_ready_to_use_soft_clip(img_data, threshold=0.98, rolloff=2.0, progress_callback=None):
    if progress_callback: progress_callback(f"Final Polish: Soft-clip > {threshold:.2f}")
    def soft_clip_channel(c, thresh, roll):
        mask = c > thresh
        result = c.copy()
        if np.any(mask):
            t = (c[mask] - thresh) / (1.0 - thresh + 1e-9)
            t = np.clip(t, 0.0, 1.0)
            f = 1.0 - np.power(1.0 - t, roll)
            result[mask] = thresh + (1.0 - thresh) * f
        return np.clip(result, 0.0, 1.0)
    
    is_rgb = (img_data.ndim == 3 and img_data.shape[0] == 3)
    if is_rgb:
        img_data[0] = soft_clip_channel(img_data[0], threshold, rolloff)
        img_data[1] = soft_clip_channel(img_data[1], threshold, rolloff)
        img_data[2] = soft_clip_channel(img_data[2], threshold, rolloff)
    else:
        if img_data.ndim == 3: img_data[0] = soft_clip_channel(img_data[0], threshold, rolloff)
        else: img_data = soft_clip_channel(img_data, threshold, rolloff)
    return img_data

def process_veralux_v6(img_data, log_D, protect_b, convergence_power, 
                       working_space="Rec.709 (Standard)", 
                       processing_mode="ready_to_use",
                       target_bg=None,
                       progress_callback=None):
    if progress_callback: progress_callback("Analyzing Data Structure...")
    luma_r, luma_g, luma_b = SENSOR_PROFILES[working_space]['weights']
    if img_data.ndim == 3: img = img_data.transpose(1, 2, 0).astype(np.float64)
    else: img = img_data.astype(np.float64)[:, :, np.newaxis]
    is_rgb = (img.shape[2] == 3)
    
    if np.median(img) > 1.0: img = np.clip(img, 0.0, 65535.0) / 65535.0
    else: img = np.clip(img, 0.0, 1.0)

    if progress_callback: progress_callback("Calculating Soft-Landing Anchor...")
    if is_rgb:
        stride = max(1, img.size // 500000)
        floors = [np.percentile(img[:,:,c].flatten()[::stride], 0.5) for c in range(3)]
        anchor = max(0.0, min(floors) - 0.00025)
    else:
        stride = max(1, img.size // 200000)
        anchor = max(0.0, np.percentile(img[:,:,0].flatten()[::stride], 0.5) - 0.00025)
        
    img_anchored = np.maximum(img - anchor, 0.0)
    
    if progress_callback: progress_callback(f"Extracting Luminance ({working_space})...")
    if is_rgb: L_anchored = (luma_r * img_anchored[:, :, 0] + luma_g * img_anchored[:, :, 1] + luma_b * img_anchored[:, :, 2])
    else: L_anchored = img_anchored[:, :, 0]
        
    epsilon = 1e-9; L_safe = L_anchored + epsilon
    if is_rgb:
        r_ratio = (img_anchored[:,:,0]) / L_safe; g_ratio = (img_anchored[:,:,1]) / L_safe; b_ratio = (img_anchored[:,:,2]) / L_safe

    if progress_callback: progress_callback(f"Stretching (Log D={log_D:.2f})...")
    L_str = ghs_stretch(L_anchored, 10.0 ** log_D, protect_b)
    
    if progress_callback: progress_callback("Dynamic Color Convergence...")
    final = np.zeros_like(img); L_str = np.clip(L_str, 0.0, 1.0)
    
    if is_rgb:
        k = np.power(L_str, convergence_power)
        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k
        final[:,:,0] = L_str * r_final; final[:,:,1] = L_str * g_final; final[:,:,2] = L_str * b_final
    else: final[:,:,0] = L_str

    final = final * (1.0 - 0.005) + 0.005; final = np.clip(final, 0.0, 1.0)
    final = final.astype(np.float32)
    if final.ndim == 3: final = final.transpose(2, 0, 1)
    
    if processing_mode == "ready_to_use":
        if progress_callback: progress_callback("Ready-to-Use: Applying Star-Safe Expansion...")
        effective_bg = 0.20 if target_bg is None else float(target_bg)
        final = adaptive_output_scaling(final, working_space, effective_bg, progress_callback)
        if progress_callback: progress_callback("Ready-to-Use: Soft-clipping highlights...")
        final = apply_ready_to_use_soft_clip(final, 0.98, 2.0, progress_callback)
        if progress_callback: progress_callback("Output ready for export!")
    else:
        if progress_callback: progress_callback("Scientific mode: Preserving raw stretched data")
    
    return final

# =============================================================================
#  THREADING
# =============================================================================

class AutoSolverThread(QThread):
    result_ready = pyqtSignal(float)
    def __init__(self, data, target, b_val, luma_weights):
        super().__init__()
        self.data = data; self.target = target; self.b_val = b_val; self.luma_weights = luma_weights
    def run(self):
        try:
            if self.data.ndim == 3:
                floors = []
                if self.data.shape[0] == 3:
                    for c in range(3): floors.append(np.percentile(self.data[c, ::100, ::100], 0.5))
                    anchor = max(0.0, min(floors) - (0.00025 * 65535.0 if np.median(self.data) > 1.0 else 0.00025))
                    indices = np.random.choice(self.data.shape[1] * self.data.shape[2], 100000, replace=False)
                    c0 = np.maximum(self.data[0].flatten()[indices] - anchor, 0)
                    c1 = np.maximum(self.data[1].flatten()[indices] - anchor, 0)
                    c2 = np.maximum(self.data[2].flatten()[indices] - anchor, 0)
                    r, g, b = self.luma_weights
                    L_sample = r * c0 + g * c1 + b * c2
                else:
                    anchor = np.percentile(self.data[::100, ::100], 0.5)
                    L_sample = np.maximum(self.data.flatten()[::100] - anchor, 0)
            else: L_sample = self.data.flatten()[::100]
            
            if np.median(L_sample) > 1.0: L_sample = L_sample / 65535.0
            valid = L_sample[L_sample > 1e-7]
            if len(valid) == 0:
                best_log_d = 2.0
            else:
                best_log_d = solve_stretch_factor(valid, self.target, self.b_val, self.luma_weights)
            self.result_ready.emit(best_log_d)
        except Exception as e:
            print(f"Solver Error: {e}")
            self.result_ready.emit(2.0)

class ProcessingThread(QThread):
    finished = pyqtSignal(object); progress = pyqtSignal(str)
    def __init__(self, img, D, b, conv, working_space, processing_mode, target_bg):
        super().__init__()
        self.img = img; self.D = D; self.b = b; self.conv = conv
        self.working_space = working_space; self.processing_mode = processing_mode; self.target_bg = target_bg
    def run(self):
        try:
            res = process_veralux_v6(self.img, self.D, self.b, self.conv, self.working_space, self.processing_mode, self.target_bg, self.progress.emit)
            self.finished.emit(res)
        except Exception as e: self.progress.emit(f"Error: {str(e)}")

# =============================================================================
#  GUI
# =============================================================================

class VeraLuxInterface:
    def __init__(self, siril_app, qt_app):
        self.siril = siril_app
        self.app = qt_app
        
        # --- HEADER LOG ---
        header_msg = (
            "\n##############################################\n"
            "# VeraLux — HyperMetric Stretch\n"
            "# Photometric Hyperbolic Stretch Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "##############################################"
        )
        try:
            self.siril.log(header_msg)
        except:
            print(header_msg)
        # ------------------

        self.linear_cache = None
        self.window = QMainWindow()
        self.window.setWindowTitle(f"VeraLux v{VERSION}")
        
        self.app.setStyle("Fusion") 
        self.window.setStyleSheet(DARK_STYLESHEET)
        
        self.window.setMinimumWidth(620) 
        self.window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8) 
        
        # Header
        head = QLabel(f"VERALUX v{VERSION}\nPhotometric GHS Engine")
        head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
        layout.addWidget(head)
        
        subhead = QLabel("Requirement: Linear Data • Color Calibration (SPCC) Applied")
        subhead.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subhead.setStyleSheet("font-size: 9pt; color: #999999; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(subhead)
        
        # --- TOP ROW: 0. Mode & 1. Sensor ---
        top_row = QHBoxLayout()
        
        # 0. PROCESSING MODE
        grp_mode = QGroupBox("0. Processing Mode")
        l_mode = QVBoxLayout(grp_mode)
        
        # Define Ready-to-Use
        self.radio_ready = QRadioButton("Ready-to-Use (Aesthetic)")
        self.radio_ready.setToolTip(
            "<b>Ready-to-Use Mode:</b><br>"
            "Produces an aesthetic, export-ready image.<br>"
            "• Applies adaptive 'Star-Safe' expansion.<br>"
            "• Applies Linked MTF to set background.<br>"
            "• Soft-clips highlights to reduce star blooming."
        )
        
        # Define Scientific
        self.radio_scientific = QRadioButton("Scientific (Preserve)")
        self.radio_scientific.setToolTip(
            "<b>Scientific Mode:</b><br>"
            "Produces a 100% mathematically consistent output.<br>"
            "• Clips only at physical saturation (1.0).<br>"
            "• Ideal for photometry or manual tone mapping (Curves/GHS)."
        )
        
        # Default Checked
        self.radio_ready.setChecked(True) 
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_ready, 0)
        self.mode_group.addButton(self.radio_scientific, 1)
        
        # Add to Layout: Ready FIRST
        l_mode.addWidget(self.radio_ready)
        l_mode.addWidget(self.radio_scientific)
        
        self.label_mode_info = QLabel()
        self.label_mode_info.setWordWrap(True)
        self.label_mode_info.setStyleSheet("color: #999999; font-size: 9pt; margin-top: 2px;")
        self.update_mode_info()
        l_mode.addWidget(self.label_mode_info)
        self.radio_ready.toggled.connect(self.update_mode_info)
        top_row.addWidget(grp_mode)
        
        # 1. SENSOR CALIBRATION
        grp_space = QGroupBox("1. Sensor Calibration")
        l_space = QVBoxLayout(grp_space)
        l_combo = QHBoxLayout()
        l_combo.addWidget(QLabel("Sensor Profile:")) 
        self.combo_profile = QComboBox()
        self.combo_profile.setToolTip(
            "<b>Sensor Profile:</b><br>"
            "Defines the Luminance coefficients (Weights) used for the stretch.<br>"
            "Choose <b>Rec.709</b> for general use or a specific sensor profile if known."
        )
        for profile_name in SENSOR_PROFILES.keys(): self.combo_profile.addItem(profile_name)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.combo_profile.currentTextChanged.connect(self.update_profile_info)
        l_combo.addWidget(self.combo_profile)
        l_space.addLayout(l_combo)
        self.label_profile_info = QLabel("")
        self.label_profile_info.setWordWrap(True)
        self.label_profile_info.setStyleSheet("color: #999999; font-size: 9pt; padding: 2px;")
        l_space.addWidget(self.label_profile_info)
        top_row.addWidget(grp_space)
        
        layout.addLayout(top_row)
        
        # --- 2. STRETCH ENGINE & CALIBRATION ---
        grp_combined = QGroupBox("2. Stretch Engine & Calibration")
        l_combined = QVBoxLayout(grp_combined)
        
        # A. CALIBRATION SUB-SECTION
        l_calib = QHBoxLayout()
        l_calib.addWidget(QLabel("Target Background:"))
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setToolTip(
            "<b>Target Background Level:</b><br>"
            "The desired median value for the background sky.<br>"
            "• <b>0.20</b> (Default): Standard brightness.<br>"
            "• <b>0.12</b>: High-contrast, darker background."
        )
        self.spin_target.setRange(0.05, 0.50); self.spin_target.setValue(0.20); self.spin_target.setSingleStep(0.01)
        l_calib.addWidget(self.spin_target)
        
        self.slide_target = QSlider(Qt.Orientation.Horizontal)
        self.slide_target.setToolTip("Adjust target background level")
        self.slide_target.setRange(5, 50); self.slide_target.setValue(20)
        self.slide_target.valueChanged.connect(lambda v: self.spin_target.setValue(v/100.0))
        self.spin_target.valueChanged.connect(lambda v: self.slide_target.setValue(int(v*100)))
        l_calib.addWidget(self.slide_target)
        
        self.btn_auto = QPushButton("⚡ Auto-Calculate Log D")
        self.btn_auto.setToolTip(
            "<b>Auto-Solver:</b><br>"
            "Analyzes the image data to find the <b>Stretch Factor (Log D)</b><br>"
            "that places the current background median at the Target Level."
        )
        self.btn_auto.setObjectName("AutoButton")
        self.btn_auto.clicked.connect(self.run_solver)
        l_calib.addWidget(self.btn_auto)
        
        l_combined.addLayout(l_calib)
        
        # Separator
        l_combined.addSpacing(5)
        
        # B. MANUAL ENGINE SUB-SECTION
        l_manual = QHBoxLayout()
        
        # Log D
        l_manual.addWidget(QLabel("Log D:"))
        self.spin_d = QDoubleSpinBox()
        self.spin_d.setToolTip(
            "<b>GHS Intensity (Log D):</b><br>"
            "Controls the strength of the stretch."
        )
        self.spin_d.setRange(0.0, 7.0); self.spin_d.setValue(2.0); self.spin_d.setDecimals(2); self.spin_d.setSingleStep(0.1)
        
        self.slide_d = QSlider(Qt.Orientation.Horizontal)
        self.slide_d.setRange(0, 700); self.slide_d.setValue(200)
        self.slide_d.valueChanged.connect(lambda v: self.spin_d.setValue(v/100.0))
        self.spin_d.valueChanged.connect(lambda v: self.slide_d.setValue(int(v*100)))
        
        l_manual.addWidget(self.spin_d)
        l_manual.addWidget(self.slide_d)
        
        # Spacer
        l_manual.addSpacing(15)
        
        # Protection b
        l_manual.addWidget(QLabel("Protect b:"))
        self.spin_b = QDoubleSpinBox()
        self.spin_b.setToolTip(
            "<b>Highlight Protection (b):</b><br>"
            "Controls the 'knee' of the GHS curve."
        )
        self.spin_b.setRange(0.1, 15.0); self.spin_b.setValue(6.0); self.spin_b.setSingleStep(0.1)
        l_manual.addWidget(self.spin_b)
        
        l_combined.addLayout(l_manual)
        
        layout.addWidget(grp_combined)
        
        # 3. PHYSICS
        grp_phys = QGroupBox("3. Physics & Convergence")
        l_phys = QVBoxLayout(grp_phys)
        l_conv = QHBoxLayout()
        l_conv.addWidget(QLabel("Star Core Recovery (White Point):"))
        self.spin_conv = QDoubleSpinBox()
        self.spin_conv.setToolTip(
            "<b>Color Convergence:</b><br>"
            "Controls how quickly saturated colors transition to white.<br>"
            "• Mimics the physical response of sensors/film.<br>"
            "• Higher values = Faster transition to white core (avoids color artifacts)."
        )
        self.spin_conv.setRange(1.0, 10.0); self.spin_conv.setValue(3.5)
        l_conv.addWidget(self.spin_conv)
        l_phys.addLayout(l_conv)
        layout.addWidget(grp_phys)
        
        # Footer
        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready. Please cache input first.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        
        btns = QHBoxLayout()
        
        # Always on top Toggle
        self.chk_ontop = QCheckBox("Always on top")
        self.chk_ontop.setToolTip("Keep this window above Siril")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        btns.addWidget(self.chk_ontop)
        
        # SPLIT RESET/RELOAD
        b_reset = QPushButton("Default Settings")
        b_reset.setToolTip("Reset all sliders and dropdowns to default values.")
        b_reset.clicked.connect(self.set_defaults)
        btns.addWidget(b_reset)

        b_reload = QPushButton("Reload Input")
        b_reload.setToolTip("Re-cache the image from Siril (Undo changes).")
        b_reload.clicked.connect(self.cache_input)
        btns.addWidget(b_reload)
        
        b_proc = QPushButton("PROCESS")
        b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Apply the stretch to the image.")
        b_proc.clicked.connect(self.run_process)
        btns.addWidget(b_proc)
        
        b_close = QPushButton("Close")
        b_close.setObjectName("CloseButton")
        b_close.clicked.connect(self.window.close)
        btns.addWidget(b_close)
        
        layout.addLayout(btns)
        self.update_profile_info(DEFAULT_PROFILE)
        self.window.show()
        self.center_window()
        self.cache_input() # Initial cache

    def center_window(self):
        screen = self.app.primaryScreen()
        if screen:
            self.window.move(self.window.frameGeometry().topLeft())
            frame_geo = self.window.frameGeometry()
            frame_geo.moveCenter(screen.availableGeometry().center())
            self.window.move(frame_geo.topLeft())

    def toggle_ontop(self, checked):
        pos = self.window.pos()
        if checked: self.window.setWindowFlags(self.window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else: self.window.setWindowFlags(self.window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.window.show(); self.window.move(pos)

    def update_mode_info(self):
        if self.radio_ready.isChecked():
            text = ("✓ Star-Safe Expansion\n"
                   "✓ Linked MTF Stretch\n"
                   "✓ Soft-clip highlights\n"
                   "✓ Ready for export")
        else:
            text = ("✓ Pure GHS stretch (1.0)\n"
                   "✓ Manual tone mapping\n"
                   "✓ Lossless data\n"
                   "✓ Accurate for scientific")
        self.label_mode_info.setText(text)

    def update_profile_info(self, profile_name):
        if profile_name in SENSOR_PROFILES:
            profile = SENSOR_PROFILES[profile_name]
            r, g, b = profile['weights']
            text = f"{profile['description']}\nWeights: R={r:.4f}, G={g:.4f}, B={b:.4f}" # Info removed to save space
            self.label_profile_info.setText(text)

    def set_defaults(self):
        """Resets all GUI elements to default values."""
        self.spin_d.setValue(2.0)
        self.spin_b.setValue(6.0)
        self.spin_target.setValue(0.20)
        self.spin_conv.setValue(3.5)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.radio_ready.setChecked(True) # Default
        self.chk_ontop.setChecked(True)
        self.status.setText("Settings reset to defaults.")

    def cache_input(self):
        try:
            if not self.siril.connected: self.siril.connect()
            self.status.setText("Caching Linear Data...")
            self.app.processEvents()
            with self.siril.image_lock(): self.linear_cache = self.siril.get_image_pixeldata()
            if self.linear_cache is None: self.status.setText("Error: No image open.")
            else:
                self.status.setText("Input Cached.")
                self.siril.log("VeraLux: Input Cached.", color=LogColor.GREEN)
        except Exception as e: self.status.setText("Connection Error."); print(e)

    def run_solver(self):
        if self.linear_cache is None: return
        self.status.setText("Solving..."); self.btn_auto.setEnabled(False); self.progress.setRange(0, 0)
        tgt = self.spin_target.value(); b = self.spin_b.value(); ws = self.combo_profile.currentText()
        luma = SENSOR_PROFILES[ws]['weights']
        self.solver = AutoSolverThread(self.linear_cache, tgt, b, luma)
        self.solver.result_ready.connect(self.apply_solver_result)
        self.solver.start()
        
    def apply_solver_result(self, log_d):
        self.spin_d.setValue(log_d); self.progress.setRange(0, 100); self.progress.setValue(100)
        self.btn_auto.setEnabled(True); ws = self.combo_profile.currentText()
        self.status.setText(f"Solved: Log D = {log_d:.2f}")
        self.siril.log(f"VeraLux Solver: Optimal Log D={log_d:.2f} [{ws}]", color=LogColor.GREEN)

    def run_process(self):
        if self.linear_cache is None: return
        try: self.siril.undo_save_state("VeraLux v6.0 Stretch")
        except: pass
        D = self.spin_d.value(); b = self.spin_b.value(); conv = self.spin_conv.value()
        ws = self.combo_profile.currentText(); t_bg = self.spin_target.value()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        self.status.setText("Processing..."); self.progress.setRange(0, 0)
        img_copy = self.linear_cache.copy()
        self.worker = ProcessingThread(img_copy, D, b, conv, ws, mode, t_bg)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.finish_process)
        self.worker.start()
        
    def finish_process(self, result_img):
        self.progress.setRange(0, 100); self.progress.setValue(100); self.status.setText("Complete.")
        mode = "Ready-to-Use" if self.radio_ready.isChecked() else "Scientific"
        ws = self.combo_profile.currentText()
        if result_img is not None:
            with self.siril.image_lock(): self.siril.set_image_pixeldata(result_img)
            self.siril.cmd("stat"); self.siril.log(f"VeraLux v6.0: {mode} mode applied [{ws}]", color=LogColor.GREEN)

def main():
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        siril = s.SirilInterface()
        gui = VeraLuxInterface(siril, app)
        app.exec()
    except Exception as e:
        print(f"Error starting VeraLux: {e}")
        # In case of CLI or other contexts, print trace
        traceback.print_exc()

if __name__ == "__main__":
    main()