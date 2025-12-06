##############################################
# VeraLux — HyperMetric Stretch
# Photometric Hyperbolic Stretch Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — HyperMetric Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.1.0 (Architecture Upgrade)
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

Version 1.1.0 Architecture Notes:
- Unified Math Core: Solver and Processor now use identical logic via VeraLuxCore class.
- Robust Data Normalization: Explicit handling of uint8/uint16/float inputs.
- Consistent Anchor Calculation: Single algorithm for black point detection.

... [Previous documentation preserved] ...
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

VERSION = "1.1.0"
# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
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
#  CORE ENGINE (Single Source of Truth) - V1.1.0 Implementation
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
                
            # Scenario D: Scaled to 32-bit [0-4.29B]
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
                       progress_callback=None):
    
    if progress_callback: progress_callback("Normalization & Analysis...")
    
    # 1. Normalize Input using Core
    img = VeraLuxCore.normalize_input(img_data)
    
    # Ensure format is (Channels, H, W) for internal processing
    # Siril usually gives (Channels, H, W). We stick to it.
    if img.ndim == 2:
        pass # Mono
    elif img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3:
        # If by chance input is (H, W, 3), transpose to (3, H, W)
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
    else:
        final = L_str

    # Restore pedestal for safety
    final = final * (1.0 - 0.005) + 0.005
    
    # Strict Clipping: Ensure no float exceeds 1.0, even by epsilon
    final = np.clip(final, 0.0, 1.0)
    
    # Final explicit cast to float32 for Siril compatibility
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
#  THREADING
# =============================================================================

class AutoSolverThread(QThread):
    result_ready = pyqtSignal(float)
    def __init__(self, data, target, b_val, luma_weights):
        super().__init__()
        self.data = data; self.target = target; self.b_val = b_val; self.luma_weights = luma_weights
        
    def run(self):
        try:
            # 1. Normalize (Core)
            # Use a copy to avoid modifying original cache if shared
            img_norm = VeraLuxCore.normalize_input(self.data) 
            
            # Ensure shape (C, H, W)
            if img_norm.ndim == 3 and img_norm.shape[0] != 3 and img_norm.shape[2] == 3:
                img_norm = img_norm.transpose(2, 0, 1)
            
            # 2. Subsample for Speed (Solver only needs stats)
            # We pick a random subset of pixels to estimate the median background
            if img_norm.ndim == 3:
                # Subsample 100k pixels
                h, w = img_norm.shape[1], img_norm.shape[2]
                num_pixels = h * w
                indices = np.random.choice(num_pixels, min(num_pixels, 100000), replace=False)
                
                # Flatten channels
                c0 = img_norm[0].flatten()[indices]
                c1 = img_norm[1].flatten()[indices]
                c2 = img_norm[2].flatten()[indices]
                
                # Reconstruct subsampled array (3, N)
                sub_data = np.vstack((c0, c1, c2))
                
            else:
                # Mono subsample
                h, w = img_norm.shape
                num_pixels = h * w
                indices = np.random.choice(num_pixels, min(num_pixels, 100000), replace=False)
                sub_data = img_norm.flatten()[indices]

            # 3. Calculate Anchor (Core)
            # Core handles subsampling internally, but here we pass already subsampled data
            # passing subsampled data to calculate_anchor works because it calculates percentile
            anchor = VeraLuxCore.calculate_anchor(sub_data)
            
            # 4. Extract Luminance (Core)
            L_anchored, _ = VeraLuxCore.extract_luminance(sub_data, anchor, self.luma_weights)
            
            # Filter valid pixels (remove 0s)
            valid = L_anchored[L_anchored > 1e-7]
            
            # 5. Solve (Core)
            if len(valid) == 0: best_log_d = 2.0
            else: best_log_d = VeraLuxCore.solve_log_d(valid, self.target, self.b_val)
                
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
            # Pass Raw data, logic handles normalization
            res = process_veralux_v6(self.img, self.D, self.b, self.conv, self.working_space, self.processing_mode, self.target_bg, self.progress.emit)
            self.finished.emit(res)
        except Exception as e: 
            traceback.print_exc()
            self.progress.emit(f"Error: {str(e)}")

# =============================================================================
#  GUI (Unchanged from v1.0.3 except version ref)
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
        self.window = QMainWindow()
        self.window.setWindowTitle(f"VeraLux v{VERSION}")
        
        # --- FIX: Set Fusion Style for Windows Geometry + Apply Dark Theme ---
        self.app.setStyle("Fusion") 
        self.window.setStyleSheet(DARK_STYLESHEET)
        
        self.window.setMinimumWidth(620) 
        self.window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8) 
        
        # Header Title
        head_title = QLabel(f"VeraLux HyperMetric Stretch v{VERSION}")
        head_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
        layout.addWidget(head_title)

        # Subtitle
        head_sub = QLabel("Photometric Hyperbolic Stretch Engine")
        head_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head_sub.setStyleSheet("font-size: 12pt; font-weight: normal; color: #88aaff;")
        layout.addWidget(head_sub)
        
        subhead = QLabel("Requirement: Linear Data • Color Calibration (SPCC) Applied")
        subhead.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subhead.setStyleSheet("font-size: 9pt; color: #999999; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(subhead)
        
        # --- TOP ROW: 0. Mode & 1. Sensor ---
        top_row = QHBoxLayout()
        
        # 0. PROCESSING MODE
        grp_mode = QGroupBox("0. Processing Mode")
        l_mode = QVBoxLayout(grp_mode)
        
        # Define Ready-to-Use FIRST
        self.radio_ready = QRadioButton("Ready-to-Use (Aesthetic)")
        self.radio_ready.setToolTip(
            "<b>Ready-to-Use Mode:</b><br>"
            "Produces an aesthetic, export-ready image.<br>"
            "• Applies adaptive 'Star-Safe' expansion.<br>"
            "• Applies Linked MTF to set background.<br>"
            "• Soft-clips highlights to reduce star blooming."
        )
        
        # Define Scientific SECOND
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
        
        # 1. SENSOR CALIBRATION (Renamed)
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
        
        # --- 2. STRETCH ENGINE & CALIBRATION (Merged) ---
        grp_combined = QGroupBox("2. Stretch Engine & Calibration")
        l_combined = QVBoxLayout(grp_combined)
        
        # A. CALIBRATION SUB-SECTION
        l_calib = QHBoxLayout()
        l_calib.addWidget(QLabel("Target Background:"))
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setToolTip(
            "<b>Target Background (Median):</b><br>"
            "The desired median value for the background sky.<br>"
            "• <b>0.20</b> is standard for good visibility (Statistical Stretch style).<br>"
            "• <b>0.12</b> for high-contrast dark skies."
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
        
        # Help Button
        self.btn_help = QPushButton("?")
        self.btn_help.setObjectName("HelpButton")
        self.btn_help.setToolTip("Print Operational Guide to Siril Console")
        self.btn_help.setFixedWidth(20)
        self.btn_help.clicked.connect(self.print_help_to_console)
        btns.addWidget(self.btn_help)
        
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
        b_reload.setToolTip("Reload linear image from Siril memory. For Undo must use Siril back button.")
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

    def print_help_to_console(self):
        """Prints the operational guide to the Siril console line by line to avoid truncation."""
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
            "                  - Increase this if you see \"donut\" artifacts or holes in stars.\n\n"
            "",
            "   D. EXECUTION & RESET\n"
            "                • [Default Settings]: Resets only sliders/menus to factory defaults.\n"
            "                • [Reload Input]: Re-caches the linear image from Siril.\n"
            "                  Acts as a full \"Undo\" to restart processing from scratch.\n\n"
            "",
            "Support & Info: info@veralux.space\n"
            "-------------------------------------------------------------"
        ]
        
        try:
            # Iterate and print line by line to avoid buffer limits
            for line in guide_lines:
                self.siril.log(line)
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
            text = f"{profile['description']}\n"
            text += f"Weights: R={r:.4f}, G={g:.4f}, B={b:.4f}\n"
            text += f"💡 {profile['info']}"
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
            # 1. Send data
            with self.siril.image_lock(): 
                self.siril.set_image_pixeldata(result_img)
            
            # 2. Update Statistics
            self.siril.cmd("stat")
            
            # 3. BLACK SCREEN FIX: Reset visualization
            # Use explicit numeric limits (0 65535), universal for Siril.
            # This resets the Screen Transfer Function.
            try:
                self.siril.cmd("visu 0 65535") 
            except:
                pass
            
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