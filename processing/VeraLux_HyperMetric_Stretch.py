##############################################
# VeraLux — HyperMetric Stretch
# Photometric Hyperbolic Stretch Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — HyperMetric Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.2.0
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
• Bridge the gap between numerical processing and visual feedback (Live Preview)
• Allow controlled hybrid tone-mapping for highlight management (Color Grip)

Core Features
-------------
• Live Preview Engine (New in v1.2.0):
  - Interactive floating window offering real-time feedback on parameter changes.
  - Features Smart Proxy technology for fluid response even on massive files.
  - Includes professional navigation controls (Zoom, Pan, Fit-to-Screen).
• Hybrid Color Engine (Color Grip):
  - Allows blending between Vector Preservation (Vivid) and Scalar Stretching (Soft).
  - Gives users control over star core saturation without losing data fidelity.
• Unified Math Core:
  - Implements a "Single Source of Truth" architecture. The Auto-Solver, Live 
    Preview, and Main Processor share the exact same logic.
• Robust Input Normalization:
  - Automatically handles 8, 16, 32-bit Integer or 32-bit Float inputs, 
    preventing clipping errors on high-dynamic range files.

Usage
-----
1. Pre-requisite: Image MUST be Linear and Color Calibrated (SPCC).
2. Setup: Select your Sensor Profile (or Rec.709) and Processing Mode.
   (Default is "Ready-to-Use" for immediate results).
3. Calibrate: Click Calculate Optimal Log D (Auto-Solver) to analyze the 
   linear data and find the mathematical sweet spot.
4. Refine (Interactive): Click [Live Preview] to open the visualizer.
   Adjust Sliders (Log D, Protect b, Color Grip) and observe changes in real-time.
5. Process: Click PROCESS.
   - The script automatically resets the display visualization to linear 
     to ensure the result is immediately visible and correctly scaled.

Inputs & Outputs
----------------
Input:
• Linear FITS/TIFF images (RGB or Mono).
• Supports 16-bit/32-bit Integer and 32-bit Float formats automatically.

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
                            QComboBox, QRadioButton, QButtonGroup, QCheckBox, QFrame,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QWheelEvent, QMouseEvent

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
QPushButton#PreviewButton { background-color: #2a5a2a; border: 1px solid #408040; }
QPushButton#PreviewButton:hover { background-color: #3a7a3a; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* Preview Toolbar Buttons */
QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

/* GHOST HELP BUTTON */
QPushButton#HelpButton { 
    background-color: transparent; 
    color: #555555; 
    border: none; 
    font-weight: bold; 
    min-width: 20px;
}
QPushButton#HelpButton:hover { 
    color: #aaaaaa; 
}

QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }
"""

VERSION = "1.2.0"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.2.0: Major Upgrade. Added Live Preview Engine with Smart Proxy technology.
#        Introduced "Color Grip" Hybrid Stretch for star control.
# 1.1.0: Architecture Upgrade. Introduced VeraLuxCore (Single Source of Truth).
#        Fixed 32-bit/Mono input handling & visual refresh issues (visu reset).
#        Added robust input normalization & improved Solver precision.
# 1.0.3: Added help button (?) that prints Operational Guide to Siril Console.
#        Added contact e-mail. Texts consistency minor fixes.
# 1.0.2: Sensor Database Update (v2.0). Added real QE weights for 15+ sensors.
# 1.0.1: Fix Windows GUI artifacts (invisible checkboxes) and UI polish.
# ------------------------------------------------------------------------------

# =============================================================================
#  WORKING SPACE PROFILES (Database v2.1 - Siril SPCC Derived)
# =============================================================================

SENSOR_PROFILES = {
    # --- STANDARD ---
    "Rec.709 (Recommended)": {
        'weights': (0.2126, 0.7152, 0.0722),
        'description': "ITU-R BT.709 standard for sRGB/HDTV",
        'info': "Default choice. Best for general use, DSLR and unknown sensors.",
        'category': 'standard'
    },
    
    # --- SONY MODERN BSI (Consumer) ---
    "Sony IMX571 (ASI2600/QHY268)": {
        'weights': (0.2944, 0.5021, 0.2035),
        'description': "Sony IMX571 26MP APS-C BSI (STARVIS)",
        'info': "Gold standard APS-C. Excellent balance for broadband.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX533 (ASI533)": {
        'weights': (0.2910, 0.5072, 0.2018),
        'description': "Sony IMX533 9MP 1\" Square BSI (STARVIS)",
        'info': "Popular square format. Very low noise.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX455 (ASI6200/QHY600)": {
        'weights': (0.2987, 0.5001, 0.2013),
        'description': "Sony IMX455 61MP Full Frame BSI (STARVIS)",
        'info': "Full frame reference sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX294 (ASI294)": {
        'weights': (0.3068, 0.5008, 0.1925),
        'description': "Sony IMX294 11.7MP 4/3\" BSI",
        'info': "High sensitivity 4/3 format.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX183 (ASI183)": {
        'weights': (0.2967, 0.4983, 0.2050),
        'description': "Sony IMX183 20MP 1\" BSI",
        'info': "High resolution 1-inch sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX178 (ASI178)": {
        'weights': (0.2346, 0.5206, 0.2448),
        'description': "Sony IMX178 6.4MP 1/1.8\" BSI",
        'info': "High resolution entry-level sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX224 (ASI224)": {
        'weights': (0.3402, 0.4765, 0.1833),
        'description': "Sony IMX224 1.27MP 1/3\" BSI",
        'info': "Classic planetary sensor. High Red response.",
        'category': 'sensor-specific'
    },
    
    # --- SONY STARVIS 2 (NIR Optimized) ---
    "Sony IMX585 (ASI585) - STARVIS 2": {
        'weights': (0.3431, 0.4822, 0.1747),
        'description': "Sony IMX585 8.3MP 1/1.2\" BSI (STARVIS 2)",
        'info': "NIR optimized. Excellent for H-Alpha/Narrowband.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX662 (ASI662) - STARVIS 2": {
        'weights': (0.3430, 0.4821, 0.1749),
        'description': "Sony IMX662 2.1MP 1/2.8\" BSI (STARVIS 2)",
        'info': "Planetary/Guiding. High Red/NIR sensitivity.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX678/715 - STARVIS 2": {
        'weights': (0.3426, 0.4825, 0.1750),
        'description': "Sony IMX678/715 BSI (STARVIS 2)",
        'info': "High resolution planetary/security sensors.",
        'category': 'sensor-specific'
    },
    
    # --- PANASONIC / OTHERS ---
    "Panasonic MN34230 (ASI1600/QHY163)": {
        'weights': (0.2650, 0.5250, 0.2100),
        'description': "Panasonic MN34230 4/3\" CMOS",
        'info': "Classic Mono/OSC sensor. Optimized weights.",
        'category': 'sensor-specific'
    },
    
    # --- CANON DSLR (Averaged Profiles) ---
    "Canon EOS (Modern - 60D/6D/R)": {
        'weights': (0.2550, 0.5250, 0.2200),
        'description': "Canon CMOS Profile (Modern)",
        'info': "Balanced profile for most Canon EOS cameras (60D, 6D, 5D, R-series).",
        'category': 'sensor-specific'
    },
    
    "Canon EOS (Legacy - 300D/40D)": {
        'weights': (0.2400, 0.5400, 0.2200),
        'description': "Canon CMOS Profile (Legacy)",
        'info': "For older Canon models (Digic 2/3 era).",
        'category': 'sensor-specific'
    },
    
    # --- NIKON DSLR (Averaged Profiles) ---
    "Nikon DSLR (Modern - D5300/D850)": {
        'weights': (0.2600, 0.5100, 0.2300),
        'description': "Nikon CMOS Profile (Modern)",
        'info': "Balanced profile for Nikon Expeed 4+ cameras.",
        'category': 'sensor-specific'
    },
    
    # --- SMART TELESCOPES ---
    "ZWO Seestar S50": {
        'weights': (0.3333, 0.4866, 0.1801),
        'description': "ZWO Seestar S50 (IMX462)",
        'info': "Specific profile for Seestar S50 smart telescope.",
        'category': 'sensor-specific'
    },
    
    "ZWO Seestar S30": {
        'weights': (0.2928, 0.5053, 0.2019),
        'description': "ZWO Seestar S30",
        'info': "Specific profile for Seestar S30 smart telescope.",
        'category': 'sensor-specific'
    },
    
    # --- NARROWBAND ---
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
#  CORE ENGINE (Single Source of Truth) - V1.2.0 Implementation
# =============================================================================

class VeraLuxCore:
    """
    Centralized math library for VeraLux.
    Ensures consistency between AutoSolver (Preview) and Main Processor.
    """
    
    @staticmethod
    def normalize_input(img_data):
        """
        Robustly normalizes input data to float32 range [0.0, 1.0].
        Handles uint8, uint16, uint32 and various float scenarios.
        """
        input_dtype = img_data.dtype
        
        # Helper to convert to float32 before division
        img_float = img_data.astype(np.float32)
        
        # Case 1: Integer Types
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8:
                return img_float / 255.0
            elif input_dtype == np.uint16:
                return img_float / 65535.0
            elif input_dtype == np.uint32:
                return img_float / 4294967295.0
            else:
                info = np.iinfo(input_dtype)
                return img_float / float(info.max)
                
        # Case 2: Float Types (Container analysis)
        elif np.issubdtype(input_dtype, np.floating):
            # Check maximum only if necessary for optimization
            # Use np.max() on flattened array or sample if huge, 
            # but for safety, check global max.
            current_max = np.max(img_data)
            
            # Scenario A: Already normalized [0-1]
            if current_max <= 1.0 + 1e-5:
                return img_float
            
            # Scenario B: Scaled to 8-bit [0-255]
            if current_max <= 255.0:
                return img_float / 255.0
                
            # Scenario C: Scaled to 16-bit [0-65535]
            if current_max <= 65535.0:
                return img_float / 65535.0
                
            # Scenario D: Scaled to 32-bit [0-4.29B] (IL TUO CASO)
            # If > 65535, assume 32-bit scaling
            return img_float / 4294967295.0
            
        return img_float

    @staticmethod
    def calculate_anchor(data_norm):
        """
        Unified Black Point (Anchor) calculation.
        input: data_norm (normalized 0-1), shape (Channels, H, W) or (H, W)
        """
        if data_norm.ndim == 3:
            # Multi-channel: Find floor per channel
            floors = []
            stride = max(1, data_norm.size // 500000) # Adaptive subsampling for speed
            for c in range(data_norm.shape[0]):
                channel_floor = np.percentile(data_norm[c].flatten()[::stride], 0.5)
                floors.append(channel_floor)
            # Safe floor is min of channels minus pedestal
            anchor = max(0.0, min(floors) - 0.00025)
        else:
            # Mono
            stride = max(1, data_norm.size // 200000)
            floor = np.percentile(data_norm.flatten()[::stride], 0.5)
            anchor = max(0.0, floor - 0.00025)
        return anchor

    @staticmethod
    def extract_luminance(data_norm, anchor, weights):
        """
        Extracts anchored Luminance based on sensor weights.
        Returns: L_anchored (0-based) and L_safe (for division)
        """
        r_w, g_w, b_w = weights
        img_anchored = np.maximum(data_norm - anchor, 0.0)
        
        # Robust Mono/RGB handling
        # True RGB: (3, H, W)
        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            L_anchored = (r_w * img_anchored[0] + 
                          g_w * img_anchored[1] + 
                          b_w * img_anchored[2])
        # Mono masked as 3D: (1, H, W)
        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            L_anchored = img_anchored[0]
            img_anchored = img_anchored[0] # Flatten for consistency
        # Pure Mono: (H, W)
        else:
            L_anchored = img_anchored
            
        return L_anchored, img_anchored

    @staticmethod
    def ghs_stretch(data, D, b, SP=0.0):
        """Generalized Hyperbolic Stretch Formula."""
        D = max(D, 0.1)
        b = max(b, 0.1)
        term1 = np.arcsinh(D * (data - SP) + b)
        term2 = np.arcsinh(b)
        norm_factor = np.arcsinh(D * (1.0 - SP) + b) - term2
        if norm_factor == 0: norm_factor = 1e-6
        stretched = (term1 - term2) / norm_factor
        return stretched

    @staticmethod
    def solve_log_d(luma_sample, target_median, b_val):
        """Iterative solver for Log D."""
        median_in = np.median(luma_sample)
        if median_in < 1e-9: return 2.0 
        
        low_log = 0.0; high_log = 7.0; best_log_D = 2.0
        
        for _ in range(40):
            mid_log = (low_log + high_log) / 2.0
            mid_D = 10.0 ** mid_log
            test_val = VeraLuxCore.ghs_stretch(median_in, mid_D, b_val)
            
            if abs(test_val - target_median) < 0.0001:
                best_log_D = mid_log; break
            
            if test_val < target_median: low_log = mid_log
            else: high_log = mid_log
            
        return best_log_D

    @staticmethod
    def apply_mtf(data, m):
        term1 = (m - 1.0) * data
        term2 = (2.0 * m - 1.0) * data - m
        with np.errstate(divide='ignore', invalid='ignore'):
            res = term1 / term2
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

# =============================================================================
#  HELPER FUNCTIONS (Ready-to-Use Logic)
# =============================================================================

def adaptive_output_scaling(img_data, working_space="Rec.709 (Standard)", 
                            target_bg=0.20, progress_callback=None):
    """Adaptive Star-Safe Scaling - operates on normalized (CHW) data"""
    if progress_callback: progress_callback("Adaptive Scaling: Analyzing Dynamic Range (Anti-Bloat)...")
    luma_r, luma_g, luma_b = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img_data.ndim == 3 and img_data.shape[0] == 3)
    
    if is_rgb:
        R, G, B = img_data[0], img_data[1], img_data[2]
        L_raw = luma_r * R + luma_g * G + luma_b * B
    else:
        L_raw = img_data
    
    median_L = float(np.median(L_raw))
    std_L    = float(np.std(L_raw))
    min_L    = float(np.min(L_raw))
    global_floor = max(min_L, median_L - 2.7 * std_L)
    
    PEDESTAL = 0.001
    TARGET_SOFT = 0.98; TARGET_HARD = 1.0
    
    if is_rgb:
        # Use simple percentiles for speed
        stride = max(1, R.size // 500000)
        soft_r = np.percentile(R.flatten()[::stride], 99.0)
        soft_g = np.percentile(G.flatten()[::stride], 99.0)
        soft_b = np.percentile(B.flatten()[::stride], 99.0)
        soft_ceil = max(soft_r, soft_g, soft_b)
        
        hard_r = np.percentile(R.flatten()[::stride], 99.99)
        hard_g = np.percentile(G.flatten()[::stride], 99.99)
        hard_b = np.percentile(B.flatten()[::stride], 99.99)
        hard_ceil = max(hard_r, hard_g, hard_b)
    else:
        stride = max(1, L_raw.size // 200000)
        soft_ceil = np.percentile(L_raw.flatten()[::stride], 99.0)
        hard_ceil = np.percentile(L_raw.flatten()[::stride], 99.99)
        
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
        
    # Apply expansion
    if is_rgb:
        img_data[0] = expand_channel(R)
        img_data[1] = expand_channel(G)
        img_data[2] = expand_channel(B)
        L = luma_r * img_data[0] + luma_g * img_data[1] + luma_b * img_data[2]
    else:
        img_data = expand_channel(L_raw)
        L = img_data
    
    current_bg = float(np.median(L))
    if current_bg > 0.0 and current_bg < 1.0 and abs(current_bg - target_bg) > 1e-3:
        if progress_callback: progress_callback(f"Applying MTF (Bg: {current_bg:.3f} -> {target_bg})")
        x = current_bg; y = target_bg
        m = (x * (y - 1.0)) / (x * (2.0 * y - 1.0) - y)
        if is_rgb:
            img_data[0] = VeraLuxCore.apply_mtf(img_data[0], m)
            img_data[1] = VeraLuxCore.apply_mtf(img_data[1], m)
            img_data[2] = VeraLuxCore.apply_mtf(img_data[2], m)
        else:
            img_data = VeraLuxCore.apply_mtf(img_data, m)
            
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
    
    if img_data.ndim == 3:
        for i in range(img_data.shape[0]):
            img_data[i] = soft_clip_channel(img_data[i], threshold, rolloff)
    else:
        img_data = soft_clip_channel(img_data, threshold, rolloff)
    return img_data

def process_veralux_v6(img_data, log_D, protect_b, convergence_power, 
                       working_space="Rec.709 (Standard)", 
                       processing_mode="ready_to_use",
                       target_bg=None,
                       color_grip=1.0, # Added in v1.2.0
                       progress_callback=None):
    
    if progress_callback: progress_callback("Normalization & Analysis...")
    
    # 1. Normalize Input using Core
    img = VeraLuxCore.normalize_input(img_data)
    
    # Ensure format is (Channels, H, W)
    if img.ndim == 2:
        pass # Mono
    elif img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3:
        img = img.transpose(2, 0, 1)

    luma_weights = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img.ndim == 3)

    # 2. Calculate Anchor using Core
    if progress_callback: progress_callback("Calculating Anchor...")
    anchor = VeraLuxCore.calculate_anchor(img)
    
    # 3. Extract Luminance using Core
    if progress_callback: progress_callback(f"Extracting Luminance ({working_space})...")
    L_anchored, img_anchored = VeraLuxCore.extract_luminance(img, anchor, luma_weights)
    
    # Prepare Ratios for Vector Preservation
    epsilon = 1e-9
    L_safe = L_anchored + epsilon
    
    if is_rgb:
        r_ratio = img_anchored[0] / L_safe
        g_ratio = img_anchored[1] / L_safe
        b_ratio = img_anchored[2] / L_safe

    # 4. Stretch using Core GHS
    if progress_callback: progress_callback(f"Stretching (Log D={log_D:.2f})...")
    L_str = VeraLuxCore.ghs_stretch(L_anchored, 10.0 ** log_D, protect_b)
    L_str = np.clip(L_str, 0.0, 1.0)
    
    # 5. Dynamic Color Convergence
    if progress_callback: progress_callback("Dynamic Color Convergence...")
    final = np.zeros_like(img)
    
    if is_rgb:
        k = np.power(L_str, convergence_power)
        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k
        
        final[0] = L_str * r_final
        final[1] = L_str * g_final
        final[2] = L_str * b_final
        
        # --- Hybrid Stretch Logic (Color Grip) ---
        if color_grip < 1.0:
            if progress_callback: progress_callback("Mixing Hybrid Scalar Stretch...")
            # Calculate Scalar (Independent) stretch for blending
            D_val = 10.0 ** log_D
            scalar = np.zeros_like(final)
            scalar[0] = VeraLuxCore.ghs_stretch(img_anchored[0], D_val, protect_b)
            scalar[1] = VeraLuxCore.ghs_stretch(img_anchored[1], D_val, protect_b)
            scalar[2] = VeraLuxCore.ghs_stretch(img_anchored[2], D_val, protect_b)
            scalar = np.clip(scalar, 0.0, 1.0)
            
            # Blend: Final = Vector * Grip + Scalar * (1-Grip)
            final = (final * color_grip) + (scalar * (1.0 - color_grip))
            
    else:
        final = L_str

    # Restore pedestal for safety
    final = final * (1.0 - 0.005) + 0.005
    final = np.clip(final, 0.0, 1.0)
    final = final.astype(np.float32)
    
    # 6. Output Formatting (Siril expects (C, H, W) usually, but let's check input)
    # The input was (C, H, W). We return (C, H, W).
    
    if processing_mode == "ready_to_use":
        if progress_callback: progress_callback("Ready-to-Use: Star-Safe Expansion...")
        effective_bg = 0.20 if target_bg is None else float(target_bg)
        final = adaptive_output_scaling(final, working_space, effective_bg, progress_callback)
        
        if progress_callback: progress_callback("Ready-to-Use: Polish...")
        final = apply_ready_to_use_soft_clip(final, 0.98, 2.0, progress_callback)
        
        if progress_callback: progress_callback("Output ready!")
    else:
        if progress_callback: progress_callback("Scientific mode: Raw output")
    
    return final

# =============================================================================
#  LIVE PREVIEW SYSTEM
# =============================================================================

class VeraLuxPreviewWindow(QWidget):
    """
    Independent Preview Window with Zoom Buttons and Double-Click Reset.
    Stays on top.
    """
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("VeraLux Live Preview")
        self.resize(800, 600)
        # FORCE ON TOP
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(DARK_STYLESHEET) 

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # --- TOOLBAR (Top) ---
        toolbar = QWidget()
        toolbar.setStyleSheet("background-color: #333333; border-bottom: 1px solid #555555;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(5, 5, 5, 5)
        tb_layout.setSpacing(10)
        
        # Zoom Buttons
        btn_in = QPushButton("+")
        btn_in.setObjectName("ZoomBtn")
        btn_in.setToolTip("Zoom In")
        btn_in.clicked.connect(self.zoom_in)
        
        btn_out = QPushButton("-")
        btn_out.setObjectName("ZoomBtn")
        btn_out.setToolTip("Zoom Out")
        btn_out.clicked.connect(self.zoom_out)
        
        btn_fit = QPushButton("Fit")
        btn_fit.setObjectName("ZoomBtn")
        btn_fit.setToolTip("Fit to Window (Double-Click Image)")
        btn_fit.clicked.connect(self.fit_to_view)
        
        tb_layout.addWidget(btn_out)
        tb_layout.addWidget(btn_fit)
        tb_layout.addWidget(btn_in)
        tb_layout.addStretch() # Push buttons to left
        
        layout.addWidget(toolbar)
        
        # --- GRAPHICS VIEW ---
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background-color: #1e1e1e; border: none;")
        
        layout.addWidget(self.view)
        
        # Data container
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.processed_pixmap = None
        
        # Status Label overlay
        self.lbl_info = QLabel("Loading...", self.view)
        self.lbl_info.setStyleSheet("background-color: rgba(0,0,0,150); color: white; padding: 5px; border-radius: 3px;")
        self.lbl_info.move(10, 10)
        self.lbl_info.adjustSize()

    def set_image(self, qimg):
        """Updates the displayed image."""
        pixmap = QPixmap.fromImage(qimg)
        self.processed_pixmap = pixmap
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.lbl_info.setText("Preview Updated")
        self.lbl_info.adjustSize()

    def fit_to_view(self):
        """Fits image to window."""
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1/1.2, 1/1.2)

    def wheelEvent(self, event: QWheelEvent):
        """Keep wheel zoom for mouse users."""
        if event.angleDelta().y() > 0: self.zoom_in()
        else: self.zoom_out()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Reset view on double click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_view()

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
            img_norm = VeraLuxCore.normalize_input(self.data) 
            if img_norm.ndim == 3 and img_norm.shape[0] != 3 and img_norm.shape[2] == 3:
                img_norm = img_norm.transpose(2, 0, 1)
            
            if img_norm.ndim == 3:
                h, w = img_norm.shape[1], img_norm.shape[2]
                num_pixels = h * w
                indices = np.random.choice(num_pixels, min(num_pixels, 100000), replace=False)
                c0 = img_norm[0].flatten()[indices]
                c1 = img_norm[1].flatten()[indices]
                c2 = img_norm[2].flatten()[indices]
                sub_data = np.vstack((c0, c1, c2))
            else:
                h, w = img_norm.shape
                num_pixels = h * w
                indices = np.random.choice(num_pixels, min(num_pixels, 100000), replace=False)
                sub_data = img_norm.flatten()[indices]

            anchor = VeraLuxCore.calculate_anchor(sub_data)
            L_anchored, _ = VeraLuxCore.extract_luminance(sub_data, anchor, self.luma_weights)
            valid = L_anchored[L_anchored > 1e-7]
            if len(valid) == 0: best_log_d = 2.0
            else: best_log_d = VeraLuxCore.solve_log_d(valid, self.target, self.b_val)
            self.result_ready.emit(best_log_d)
        except Exception as e:
            print(f"Solver Error: {e}")
            self.result_ready.emit(2.0)

class ProcessingThread(QThread):
    finished = pyqtSignal(object); progress = pyqtSignal(str)
    def __init__(self, img, D, b, conv, working_space, processing_mode, target_bg, color_grip):
        super().__init__()
        self.img = img; self.D = D; self.b = b; self.conv = conv
        self.working_space = working_space; self.processing_mode = processing_mode; self.target_bg = target_bg
        self.color_grip = color_grip
    def run(self):
        try:
            # Pass Raw data, logic handles normalization
            res = process_veralux_v6(self.img, self.D, self.b, self.conv, self.working_space, self.processing_mode, self.target_bg, self.color_grip, self.progress.emit)
            self.finished.emit(res)
        except Exception as e: 
            traceback.print_exc()
            self.progress.emit(f"Error: {str(e)}")

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
            "# Contact: info@veralux.space\n"
            "##############################################"
        )
        try:
            self.siril.log(header_msg)
        except:
            print(header_msg)

        self.linear_cache = None
        self.preview_proxy = None # Low-res copy for preview
        self.preview_window = None
        
        self.window = QMainWindow()
        # Clean Exit handler
        self.window.closeEvent = self.handle_close_event
        
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
        head_title = QLabel(f"VeraLux HyperMetric Stretch v{VERSION}")
        head_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
        layout.addWidget(head_title)
        
        subhead = QLabel("Requirement: Linear Data • Color Calibration (SPCC) Applied")
        subhead.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subhead.setStyleSheet("font-size: 9pt; color: #999999; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(subhead)
        
        # --- GUI BLOCKS ---
        # 0. Mode
        grp_mode = QGroupBox("0. Processing Mode")
        l_mode = QVBoxLayout(grp_mode)
        self.radio_ready = QRadioButton("Ready-to-Use (Aesthetic)")
        self.radio_ready.setToolTip(
            "<b>Ready-to-Use Mode:</b><br>"
            "Produces an aesthetic, export-ready image.<br>"
            "• Applies adaptive 'Star-Safe' expansion.<br>"
            "• Applies Linked MTF to set background.<br>"
            "• Soft-clips highlights to reduce star blooming."
        )
        self.radio_scientific = QRadioButton("Scientific (Preserve)")
        self.radio_scientific.setToolTip(
            "<b>Scientific Mode:</b><br>"
            "Produces a 100% mathematically consistent output.<br>"
            "• Clips only at physical saturation (1.0).<br>"
            "• Ideal for photometry or manual tone mapping (Curves/GHS)."
        )
        self.radio_ready.setChecked(True) 
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_ready, 0)
        self.mode_group.addButton(self.radio_scientific, 1)
        l_mode.addWidget(self.radio_ready)
        l_mode.addWidget(self.radio_scientific)
        self.label_mode_info = QLabel("✓ Ready-to-Use selected")
        self.label_mode_info.setStyleSheet("color: #999999; font-size: 9pt;")
        l_mode.addWidget(self.label_mode_info)
        
        # 1. Sensor
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
        l_combo.addWidget(self.combo_profile)
        l_space.addLayout(l_combo)
        self.label_profile_info = QLabel("Rec.709 Standard")
        self.label_profile_info.setStyleSheet("color: #999999; font-size: 9pt;")
        l_space.addWidget(self.label_profile_info)
        
        top_row = QHBoxLayout()
        top_row.addWidget(grp_mode); top_row.addWidget(grp_space)
        layout.addLayout(top_row)
        
        # 2. Stretch & Calibration
        grp_combined = QGroupBox("2. Stretch Engine & Calibration")
        l_combined = QVBoxLayout(grp_combined)
        
        # Target + Auto Button + Preview Button
        l_calib = QHBoxLayout()
        l_calib.addWidget(QLabel("Target Bg:"))
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setToolTip(
            "<b>Target Background (Median):</b><br>"
            "The desired median value for the background sky.<br>"
            "• <b>0.20</b> is standard for good visibility (Statistical Stretch style).<br>"
            "• <b>0.12</b> for high-contrast dark skies."
        )
        self.spin_target.setRange(0.05, 0.50); self.spin_target.setValue(0.20); self.spin_target.setSingleStep(0.01)
        l_calib.addWidget(self.spin_target)
        
        self.btn_auto = QPushButton("⚡ Auto-Calc Log D")
        self.btn_auto.setToolTip(
            "<b>Auto-Solver:</b><br>"
            "Analyzes the image data to find the <b>Stretch Factor (Log D)</b><br>"
            "that places the current background median at the Target Level."
        )
        self.btn_auto.setObjectName("AutoButton")
        self.btn_auto.clicked.connect(self.run_solver)
        l_calib.addWidget(self.btn_auto)
        
        self.btn_preview = QPushButton("👁️ Live Preview")
        self.btn_preview.setObjectName("PreviewButton")
        self.btn_preview.setToolTip("Toggle Real-Time Interactive Preview Window")
        self.btn_preview.clicked.connect(self.toggle_preview)
        l_calib.addWidget(self.btn_preview)
        
        l_combined.addLayout(l_calib)
        l_combined.addSpacing(5)
        
        # Manual Sliders
        l_manual = QHBoxLayout()
        l_manual.addWidget(QLabel("Log D:"))
        self.spin_d = QDoubleSpinBox()
        self.spin_d.setToolTip(
            "<b>GHS Intensity (Log D):</b><br>"
            "Controls the strength of the stretch."
        )
        self.spin_d.setRange(0.0, 7.0); self.spin_d.setValue(2.0); self.spin_d.setDecimals(2); self.spin_d.setSingleStep(0.1)
        self.slide_d = QSlider(Qt.Orientation.Horizontal)
        self.slide_d.setRange(0, 700); self.slide_d.setValue(200)
        l_manual.addWidget(self.spin_d); l_manual.addWidget(self.slide_d)
        
        l_manual.addSpacing(15)
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
        
        # 3. Physics
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
        
        # Color Grip Slider
        l_grip = QHBoxLayout()
        l_grip.addWidget(QLabel("Chromatic Preservation (Color Grip):"))
        self.spin_grip = QDoubleSpinBox()
        self.spin_grip.setToolTip(
            "<b>Color Grip:</b> Controls the rigor of Color Vector preservation.<br>"
            "• <b>1.00 (Default):</b> Pure VeraLux. 100% Vector lock. Maximum vividness.<br>"
            "• <b>< 1.00:</b> Blends with standard Scalar stretch. Softens star cores and relaxes saturation in highlights."
        )
        self.spin_grip.setRange(0.0, 1.0); self.spin_grip.setValue(1.0); self.spin_grip.setSingleStep(0.05)
        self.slide_grip = QSlider(Qt.Orientation.Horizontal)
        self.slide_grip.setRange(0, 100); self.slide_grip.setValue(100)
        
        # Sync Grip
        self.slide_grip.valueChanged.connect(lambda v: self.spin_grip.setValue(v/100.0))
        self.spin_grip.valueChanged.connect(lambda v: self.slide_grip.setValue(int(v*100)))
        
        l_grip.addWidget(self.spin_grip); l_grip.addWidget(self.slide_grip)
        l_phys.addLayout(l_grip)
        
        layout.addWidget(grp_phys)
        
        # Footer
        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready. Please cache input first.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        
        # Buttons
        btns = QHBoxLayout()
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20)
        self.btn_help.setToolTip("Print Operational Guide to Siril Console")
        self.chk_ontop = QCheckBox("Always on top"); self.chk_ontop.setChecked(True)
        self.chk_ontop.setToolTip("Keep this window above Siril")
        b_reset = QPushButton("Defaults")
        b_reset.setToolTip("Reset all sliders and dropdowns to default values.")
        b_reload = QPushButton("Reload Input")
        b_reload.setToolTip("Reload linear image from Siril memory. For Undo must use Siril back button.")
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Apply the stretch to the image.")
        b_close = QPushButton("Close"); b_close.setObjectName("CloseButton")
        
        btns.addWidget(self.btn_help); btns.addWidget(self.chk_ontop)
        btns.addWidget(b_reset); btns.addWidget(b_reload); btns.addWidget(b_proc); btns.addWidget(b_close)
        layout.addLayout(btns)
        
        # CONNECT SIGNALS
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.btn_help.clicked.connect(self.print_help_to_console)
        b_reset.clicked.connect(self.set_defaults)
        b_reload.clicked.connect(self.cache_input)
        b_proc.clicked.connect(self.run_process)
        b_close.clicked.connect(self.window.close)
        
        # Sync Sliders
        self.slide_d.valueChanged.connect(lambda v: self.spin_d.setValue(v/100.0))
        self.spin_d.valueChanged.connect(lambda v: self.slide_d.setValue(int(v*100)))
        
        self.radio_ready.toggled.connect(self.update_mode_info)
        self.combo_profile.currentTextChanged.connect(self.update_profile_info)
        
        # LIVE PREVIEW CONNECTIONS (Debounced)
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(150) # 150ms delay
        self.debounce_timer.timeout.connect(self.update_preview_image)
        
        # Trigger preview update on changes
        for widget in [self.spin_d, self.spin_b, self.spin_conv, self.spin_target, self.spin_grip]:
            widget.valueChanged.connect(self.trigger_preview_update)
        self.combo_profile.currentTextChanged.connect(self.trigger_preview_update)
        self.radio_ready.toggled.connect(self.trigger_preview_update)
        self.slide_d.valueChanged.connect(self.trigger_preview_update)
        self.slide_grip.valueChanged.connect(self.trigger_preview_update)

        self.update_profile_info(DEFAULT_PROFILE)
        self.window.show()
        self.center_window()
        self.cache_input() # Initial cache

    # --- LIVE PREVIEW LOGIC ---
    
    def toggle_preview(self):
        if not self.preview_window:
            self.preview_window = VeraLuxPreviewWindow()
        
        if self.preview_window.isVisible():
            self.preview_window.hide()
        else:
            if self.preview_proxy is None:
                self.prepare_preview_proxy()
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()
            self.update_preview_image()
            self.preview_window.fit_to_view()

    def prepare_preview_proxy(self):
        """Creates a high-quality downsampled version of the image for fast preview."""
        if self.linear_cache is None: return
        
        # Smart Downsample (Max 1600px long edge)
        # We need to maintain (C, H, W) format
        img = VeraLuxCore.normalize_input(self.linear_cache)
        if img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1) # Ensure (C, H, W)
            
        h = img.shape[1] if img.ndim == 3 else img.shape[0]
        w = img.shape[2] if img.ndim == 3 else img.shape[1]
        
        scale = 1600 / max(h, w)
        if scale >= 1.0:
            self.preview_proxy = img # Use original if small
        else:
            # Simple slicing for speed (block reduce is better but requires scipy/skimage)
            step = int(1 / scale)
            if img.ndim == 3:
                self.preview_proxy = img[:, ::step, ::step]
            else:
                self.preview_proxy = img[::step, ::step]

    def trigger_preview_update(self):
        """Starts timer to update preview (Debouncing)."""
        if self.preview_window and self.preview_window.isVisible():
            self.debounce_timer.start()

    def update_preview_image(self):
        """Runs the math on the proxy and updates the window."""
        if self.preview_proxy is None: return
        
        # Gather params
        D = self.spin_d.value()
        b = self.spin_b.value()
        conv = self.spin_conv.value()
        grip = self.spin_grip.value()
        ws = self.combo_profile.currentText()
        target_bg = self.spin_target.value()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        
        # Run Core Math on Proxy
        # We assume VeraLuxCore logic handles the shape correctly
        res = process_veralux_v6(self.preview_proxy.copy(), D, b, conv, ws, mode, target_bg, grip, None)
        
        # Convert to Display
        qimg = self.numpy_to_qimage(res)
        self.preview_window.set_image(qimg)

    def numpy_to_qimage(self, img_data):
        """Converts float32 (C,H,W) to QImage for display."""
        # Convert to (H,W,C) for QImage
        if img_data.ndim == 3:
            disp = img_data.transpose(1, 2, 0)
        else:
            disp = img_data
            
        # Clip and Scale to 8-bit (Processed data is linear 0-1)
        disp = np.clip(disp * 255.0, 0, 255).astype(np.uint8)     
        disp = np.flipud(disp)

        # Force contiguous memory
        disp = np.ascontiguousarray(disp)
        
        h, w = disp.shape[0], disp.shape[1]
        bytes_per_line = disp.strides[0]
        data_bytes = disp.data.tobytes()
        
        if disp.ndim == 2: # Mono
            qimg = QImage(data_bytes, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else: # RGB
            if disp.shape[2] == 3:
                qimg = QImage(data_bytes, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                return QImage()
                
        return qimg.copy()

    # --- STANDARD METHODS ---

    def handle_close_event(self, event):
        """Ensures the preview window closes when the main window closes."""
        if self.preview_window:
            self.preview_window.close()
        event.accept()

    def print_help_to_console(self):
        guide_lines = [
            "\n-------------------------------------------------------------\n"
            "VeraLux HyperMetric Stretch — OPERATIONAL GUIDE\n"
            "Physics-Based Photometric Hyperbolic Stretch Engine\n"
            "-------------------------------------------------------------\n\n"
            "",
            "[1] PREREQUISITES (CRITICAL)\n"
            "             • Input must be LINEAR (not yet stretched).\n"
            "             • Background extraction must have already been done.\n"
            "             • RGB input must be Color Calibrated (SPCC) within Siril first.\n"
            "               (HMS locks color vectors: wrong input color = wrong output).\n\n"
            "",
            "[2] WORKFLOW & PARAMETERS\n\n"
            "             A. SETUP PHASE\n"
            "                • Processing Mode:\n"
            "                  - \"Ready-to-Use\" (Default): Applies Star-Safe expansion, Linked MTF\n"
            "                    at the end of the pipeline and Highlight Soft-Clip. Export-ready results.\n"
            "                  - \"Scientific\": Raw GHS stretch. Hard-clip at 1.0. No cosmetic fixes.\n"
            "                    Need for subsequent tone mapping (e.g. curves) in most cases.\n"
            "                    If you use curves, do not touch the black and white points\n"
            "                    so as not to lose precious data!\n"
            "                • Sensor Profile: Defines Luminance weights based on Quantum Efficiency.\n"
            "                  (Use \"Rec.709\" if sensor is unknown or for mono images).\n\n"
            "",
            "   B. CALIBRATION (THE MAGIC BUTTON)\n"
            "                • Target Background: Desired sky brightness (0.20 is standard).\n"
            "                • [⚡ Auto-Calculate]: Analyzes image stats and automatically sets\n"
            "                  the optimal \"Log D\" to match your Target Background.\n\n"
            "",
            "   C. FINE TUNING (PHYSICS)\n"
            "                • Stretch (Log D): Intensity. Higher = brighter midtones.\n"
            "                • Protect b: Highlight Protection. Controls the curve \"knee\".\n"
            "                  - Higher (>6.0): Protects stars from bloating (sharper profiles).\n"
            "                  - Lower (<2.0): Brighter highlights but higher risk of bloating.\n"
            "                • Star Recovery (White Point): Controls Color Convergence.\n"
            "                  - Determines how fast a saturated core fades to pure white.\n"
            "                  - Increase this if you see \"donut\" artifacts or holes in stars.\n"
            "                • Color Grip: Controls vector strictness in highlights.\n"
            "                  - 1.0: Pure Vector (Vivid). < 1.0: Hybrid (Softer Stars).\n\n"
            "",
            "   D. EXECUTION & RESET\n"
            "                • [Default Settings]: Resets only sliders/menus to factory defaults.\n"
            "                • [Reload Input]: Re-caches the linear image from Siril.\n"
            "                • For full \"Undo\" use the back button in Siril.\n\n"
            "",
            "Support & Info: info@veralux.space\n"
            "-------------------------------------------------------------"
        ]
        try:
            for line in guide_lines: self.siril.log(line)
            self.status.setText("Guide printed to Console.")
        except:
            print("\n".join(guide_lines))

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
            self.label_mode_info.setText("✓ Ready-to-Use: Star-Safe expansion + MTF + Soft Clip")
        else:
            self.label_mode_info.setText("✓ Scientific: Pure GHS (1.0), preserved vectors, no clip")

    def update_profile_info(self, profile_name):
        if profile_name in SENSOR_PROFILES:
            profile = SENSOR_PROFILES[profile_name]
            r, g, b = profile['weights']
            self.label_profile_info.setText(f"{profile['description']} (R:{r:.2f} G:{g:.2f} B:{b:.2f})")

    def set_defaults(self):
        self.spin_d.setValue(2.0); self.spin_b.setValue(6.0); self.spin_target.setValue(0.20)
        self.spin_conv.setValue(3.5); self.spin_grip.setValue(1.0); self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.radio_ready.setChecked(True)

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
                self.preview_proxy = None # Invalidate preview cache
                if self.preview_window and self.preview_window.isVisible():
                    self.prepare_preview_proxy()
                    self.update_preview_image()
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
        grip = self.spin_grip.value()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        self.status.setText("Processing..."); self.progress.setRange(0, 0)
        img_copy = self.linear_cache.copy()
        self.worker = ProcessingThread(img_copy, D, b, conv, ws, mode, t_bg, grip)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.finish_process)
        self.worker.start()
        
    def finish_process(self, result_img):
        self.progress.setRange(0, 100); self.progress.setValue(100); self.status.setText("Complete.")
        mode = "Ready-to-Use" if self.radio_ready.isChecked() else "Scientific"
        ws = self.combo_profile.currentText()
        if result_img is not None:
            with self.siril.image_lock(): self.siril.set_image_pixeldata(result_img)
            self.siril.cmd("stat")
            try: self.siril.cmd("visu 0 65535") 
            except: pass
            self.siril.log(f"VeraLux v{VERSION}: {mode} mode applied [{ws}]", color=LogColor.GREEN)

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
        traceback.print_exc()

if __name__ == "__main__":
    main()