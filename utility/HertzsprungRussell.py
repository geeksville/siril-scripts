"""
**Hertzsprung-Russell Diagram for Siril**

This script creates a Hertzsprung-Russell diagram from an astronomical image
with astrometric solution by querying Gaia DR3 directly and using Siril's
PSF photometry.

The script detects stars in the image using Siril's star detection, queries 
Gaia DR3 for photometric data, matches the stars, and plots a color-magnitude 
diagram showing stellar classification with reference curves.

Inspired by Mike Cranfield's script

(c) Cyril Richard 2025
SPDX-License-Identifier: GPL-3.0-or-later
"""

# Version History
# 1.0.0  Initial script release

import sirilpy as s
s.ensure_installed('PyQt6')
s.ensure_installed('astropy', 'astroquery', 'matplotlib', 'numpy')

import sys
import os
import base64
import numpy as np
import argparse

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QSlider, QPushButton,
                            QGroupBox, QMessageBox, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QSplitter, QProgressBar, QTableWidget,
                            QTableWidgetItem, QHeaderView, QComboBox, QScrollArea)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPixmap, QIcon

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astroquery.gaia import Gaia

VERSION = "1.0.0"

if not s.check_module_version('>=0.6.42'):
    print("Error: requires sirilpy module >= 0.6.42")
    sys.exit(1)

# ============================================================================
# Photometric Calibration Functions
# ============================================================================

def flux_to_mag(flux):
    """Convert flux to instrumental magnitude"""
    if flux <= 0:
        return 99.0
    return -2.5 * np.log10(flux)

def calculate_color_index(star):
    """
    Calculate B-R color index from RGB fluxes
    Returns color index or None if data missing
    """
    if 'img_mag_b' in star and 'img_mag_r' in star:
        return star['img_mag_b'] - star['img_mag_r']
    return None

def calculate_zero_points(star_data):
    """
    Calculate magnitude and color calibration using Gaia as reference.
    Uses linear regression for color transformation.
    
    Returns (mag_zero_point, color_slope, color_intercept, stats)
    
    The color transformation is: BP-RP = slope * (B-R) + intercept
    """
    # Filter stars with valid data
    valid_stars = []
    for star in star_data:
        img_color = calculate_color_index(star)
        if img_color is None:
            continue
        
        # Need reference magnitude (use green channel as V-band proxy)
        if 'img_mag_g' not in star:
            continue
            
        # Need Gaia color
        gaia_color = star['color']  # BP-RP from Gaia
        
        # Need valid parallax for good calibration stars
        if star['parallax'] <= 0:
            continue
        
        # Filter out extreme values that might be problematic
        if abs(img_color) > 10 or abs(gaia_color) > 6:
            continue
            
        valid_stars.append({
            'img_mag': star['img_mag_g'],
            'img_color': img_color,
            'gaia_mag': star['g_mag'],
            'gaia_color': gaia_color
        })
    
    if len(valid_stars) < 10:
        print(f"⚠️ Warning: Only {len(valid_stars)} stars available for calibration")
        return 0.0, 1.0, 0.0, {}
    
    # Calculate magnitude zero point using median (robust to outliers)
    mag_diffs = [s['gaia_mag'] - s['img_mag'] for s in valid_stars]
    mag_zp = np.median(mag_diffs)
    mag_std = np.std(mag_diffs)
    
    # linear regression for color transformation
    # We want: gaia_color = slope * img_color + intercept
    img_colors = np.array([s['img_color'] for s in valid_stars])
    gaia_colors = np.array([s['gaia_color'] for s in valid_stars])
    
    # Use numpy's polyfit for robust linear regression
    # Degree 1 = linear fit: y = slope*x + intercept
    coeffs = np.polyfit(img_colors, gaia_colors, 1)
    color_slope = coeffs[0]
    color_intercept = coeffs[1]
    
    # Calculate R² and residuals for quality assessment
    predicted = color_slope * img_colors + color_intercept
    residuals = gaia_colors - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((gaia_colors - np.mean(gaia_colors))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    color_std = np.std(residuals)
    
    stats = {
        'n_stars': len(valid_stars),
        'mag_zp': mag_zp,
        'mag_std': mag_std,
        'color_slope': color_slope,
        'color_intercept': color_intercept,
        'color_std': color_std,
        'r_squared': r_squared
    }
    
    print(f"\n=== PHOTOMETRIC CALIBRATION ===")
    print(f"Calibration stars: {len(valid_stars)}")
    print(f"Magnitude zero point: {mag_zp:.3f} ± {mag_std:.3f}")
    print(f"\nColor transformation (linear regression):")
    print(f"  BP-RP = {color_slope:.4f} × (B-R) + {color_intercept:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Residual σ = {color_std:.4f}")
    
    if abs(color_slope - 1.0) > 0.5:
        print(f"  ⚠️ Note: Slope differs significantly from 1.0")
        print(f"     This is expected for B-R vs BP-RP")
    
    return mag_zp, color_slope, color_intercept, stats

# ============================================================================
# HR Diagram Reference Curves
# ============================================================================

def get_main_sequence_curve():
    """
    Returns main sequence reference curve
    Data from empirical fits to nearby stars
    """
    # BP-RP color index values
    color = np.array([
        -0.4, -0.3, -0.2, -0.1, 0.0,
        0.2, 0.4, 0.6, 0.8,
        1.0, 1.2, 1.4,
        1.6, 1.8, 2.0, 2.2,
        2.5, 3.0, 3.5, 4.0
    ])
    
    # Absolute magnitude M_G
    abs_mag = np.array([
        -4.0, -3.0, -2.0, -1.0, 0.0,
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0,
        8.0, 9.5, 11.0, 12.5,
        14.0, 15.5, 17.0, 18.5
    ])
    
    return color, abs_mag

def get_giant_branch_curve():
    """Returns red giant branch reference curve"""
    color = np.array([
        0.8, 1.0, 1.2, 1.4, 1.6,
        1.8, 2.0, 2.2, 2.5, 2.8,
        3.2, 3.6, 4.0
    ])
    
    abs_mag = np.array([
        1.0, 0.5, 0.0, -0.5, -1.0,
        -1.5, -2.0, -2.5, -3.0, -3.5,
        -4.0, -4.5, -5.0
    ])
    
    return color, abs_mag

def get_supergiant_curve():
    """Returns supergiant reference curve"""
    color = np.array([
        -0.2, 0.0, 0.5, 1.0, 1.5,
        2.0, 2.5, 3.0
    ])
    
    abs_mag = np.array([
        -6.0, -6.5, -7.0, -7.5, -8.0,
        -8.5, -9.0, -9.0
    ])
    
    return color, abs_mag

def get_white_dwarf_curve():
    """Returns white dwarf cooling sequence"""
    color = np.array([
        -0.4, -0.2, 0.0, 0.2, 0.4,
        0.6, 0.8, 1.0
    ])
    
    abs_mag = np.array([
        10.0, 11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0
    ])
    
    return color, abs_mag

def transform_reference_curve_to_image_system(bp_rp_color, slope, intercept):
    """
    Transform BP-RP colors to image color system using INVERSE transformation.
    
    If: BP-RP = slope * img_color + intercept
    Then: img_color = (BP-RP - intercept) / slope
    """
    if abs(slope) < 0.001:
        return bp_rp_color  # Avoid division by zero
    return (bp_rp_color - intercept) / slope

# ============================================================================
# Processing Worker Thread
# ============================================================================

class GaiaQueryWorker(QObject):
    """Worker thread for Gaia query and star matching"""
    progress_update = pyqtSignal(str, int)
    finished = pyqtSignal(object, object, object)
    
    def __init__(self, siril, min_mag, max_mag, max_stars, max_img_mag):
        super().__init__()
        self.siril = siril
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.max_stars = max_stars
        self.max_img_mag = max_img_mag
    
    def run(self):
        try:
            self.progress_update.emit("Detecting stars in image...", 10)
            
            image_stars = self.detect_image_stars()
            
            if image_stars is None or len(image_stars) == 0:
                self.finished.emit(None, "No stars detected in image. Try adjusting star detection settings in Siril.", None)
                return
            
            print(f"Detected {len(image_stars)} stars in image")
            
            self.progress_update.emit("Querying Gaia DR3...", 30)
            
            gaia_results = self.query_gaia_field()
            
            if gaia_results is None or len(gaia_results) == 0:
                self.finished.emit(None, "No Gaia stars found in field", None)
                return
            
            self.progress_update.emit("Matching stars...", 60)
            
            star_data = self.match_gaia_to_detected_stars(gaia_results, image_stars)
            
            if len(star_data) == 0:
                self.finished.emit(None, "No stars matched between Gaia and image", None)
                return
            
            self.progress_update.emit("Calibrating photometry...", 90)
            
            # Calculate photometric calibration with linear regression
            mag_zp, color_slope, color_intercept, calib_stats = calculate_zero_points(star_data)
            
            # Apply calibration to all stars
            for star in star_data:
                img_color = calculate_color_index(star)
                if img_color is not None:
                    # Transform to BP-RP equivalent using linear regression
                    star['img_color'] = color_slope * img_color + color_intercept
                    star['img_color_raw'] = img_color  # Keep original for reference
                else:
                    star['img_color'] = None
                    star['img_color_raw'] = None
                
                if 'img_mag_g' in star:
                    star['img_mag_calib'] = star['img_mag_g'] + mag_zp
                else:
                    star['img_mag_calib'] = None
            
            self.progress_update.emit("Complete", 100)
            self.finished.emit(star_data, None, calib_stats)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, f"Error: {str(e)}", None)
    
    def detect_image_stars(self):
        """Detect stars in the image using Siril's star detection"""
        try:
            print("\n=== STAR DETECTION ===")
            
            # Check if image is color
            fit = self.siril.get_image()
            if fit.data.ndim != 3:
                print("✗ Error: Image must be a color (RGB) image")
                return None
            
            hud = self.siril.get_image_keywords()
            pixel_size = float(hud.pixel_size_x)
            focal_length = float(hud.focal_length)
            
            print("Configuring star detection with photometry...")
            self.siril.cmd("setfindstar", "-moffat", 
                          f"-focal={focal_length}", 
                          f"-pixelsize={pixel_size}")
            
            all_stars = []
            
            print("Detecting on red channel (layer 0)...")
            self.siril.cmd("findstar", "-layer=0")
            stars_r = self.siril.get_image_stars()
            if stars_r:
                all_stars.extend(stars_r)
                print(f"  → {len(stars_r)} stars on red channel")
            
            print("Detecting on green channel (layer 1)...")
            self.siril.cmd("findstar", "-layer=1")
            stars_g = self.siril.get_image_stars()
            if stars_g:
                all_stars.extend(stars_g)
                print(f"  → {len(stars_g)} stars on green channel")
            
            print("Detecting on blue channel (layer 2)...")
            self.siril.cmd("findstar", "-layer=2")
            stars_b = self.siril.get_image_stars()
            if stars_b:
                all_stars.extend(stars_b)
                print(f"  → {len(stars_b)} stars on blue channel")
            
            if all_stars and len(all_stars) > 0:
                print(f"✓ Detected {len(all_stars)} total PSF measurements across all channels")
            else:
                print("✗ No stars detected")
                print("\nPlease check:")
                print("1. Image metadata: focal length and pixel size")
                print("2. Star detection settings in Siril")
                print("3. Image quality and contrast")
                        
            self.siril.cmd("clearstar")
            
            return all_stars
            
        except Exception as e:
            print(f"Error detecting stars: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def query_gaia_field(self):
        """Query Gaia DR3 for stars in the field"""
        try:
            channels, height, width = self.siril.get_image_shape()
            
            ra_center, dec_center = self.siril.pix2radec(width / 2, height / 2)
            ra_tl, dec_tl = self.siril.pix2radec(0, 0)
            ra_br, dec_br = self.siril.pix2radec(width, height)
            
            delta_ra = max(abs(ra_center - ra_tl), abs(ra_center - ra_br))
            delta_dec = max(abs(dec_center - dec_tl), abs(dec_center - dec_br))
            radius_deg = np.sqrt(delta_ra**2 + delta_dec**2)
            
            print(f"Querying Gaia DR3 around RA={ra_center:.4f}°, Dec={dec_center:.4f}°")
            print(f"Search radius: {radius_deg:.4f}° ({radius_deg*60:.2f}′)")
            
            query = f"""
            SELECT TOP {self.max_stars}
                source_id, ra, dec, parallax,
                pmra, pmdec,
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                DISTANCE(
                    POINT('ICRS', ra, dec),
                    POINT('ICRS', {ra_center}, {dec_center})
                ) AS dist
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
            )
            AND phot_g_mean_mag IS NOT NULL
            AND phot_bp_mean_mag IS NOT NULL
            AND phot_rp_mean_mag IS NOT NULL
            AND phot_g_mean_mag BETWEEN {self.min_mag} AND {self.max_mag}
            ORDER BY phot_g_mean_mag ASC
            """
            
            print("Executing Gaia query (this may take a moment)...")
            
            with s.SuppressedStdout():
                job = Gaia.launch_job_async(query)
                results = job.get_results()
            
            print(f"Found {len(results)} stars in Gaia DR3")
            
            return results
            
        except Exception as e:
            print(f"Error querying Gaia: {e}")
            return None
    
    def match_gaia_to_detected_stars(self, gaia_results, image_stars):
        """Match Gaia stars to detected image stars using PSF photometry"""
        matched_stars = []
        
        fit = self.siril.get_image()
        img_data = fit.data
        
        n_channels, height, width = img_data.shape
        
        print(f"\n=== MATCHING DEBUG ===")
        print(f"Image dimensions: {width}x{height}, channels: {n_channels}")
        print(f"Gaia stars to match: {len(gaia_results)}")
        print(f"Image stars detected: {len(image_stars)}")
        
        star_dict = {}
        
        for star in image_stars:
            key = (int(round(star.xpos)), int(round(star.ypos)))
            if key not in star_dict:
                star_dict[key] = {}
            star_dict[key][star.layer] = star
        
        print(f"Unique star positions: {len(star_dict)}")
        
        star_positions = np.array(list(star_dict.keys()))
        
        match_threshold_px = 10.0
        
        matched_count = 0
        rejected_out_of_bounds = 0
        rejected_too_far = 0
        rejected_bad_photometry = 0
        rejected_too_faint = 0
        
        for i, gaia_star in enumerate(gaia_results):
            if i % 50 == 0:
                progress = int(60 + (i / len(gaia_results)) * 35)
                self.progress_update.emit(f"Matching stars ({matched_count} matched)", progress)
            
            try:
                x, y = self.siril.radec2pix(gaia_star['ra'], gaia_star['dec'])
                
                if x < 0 or x >= width or y < 0 or y >= height:
                    rejected_out_of_bounds += 1
                    continue
                
                if len(star_positions) == 0:
                    continue
                    
                distances = np.sqrt((star_positions[:, 0] - x)**2 + 
                                  (star_positions[:, 1] - y)**2)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                                
                if min_dist > match_threshold_px:
                    rejected_too_far += 1
                    continue
                
                matched_pos = tuple(star_positions[min_idx])
                star_layers = star_dict[matched_pos]
                
                # Use green channel as reference
                reference_layer = 1 if 1 in star_layers else 0
                if reference_layer not in star_layers:
                    reference_layer = list(star_layers.keys())[0]
                
                reference_star = star_layers[reference_layer]
                
                if self.max_img_mag > 0 and reference_star.mag > self.max_img_mag:
                    rejected_too_faint += 1
                    continue
                
                color = float(gaia_star['phot_bp_mean_mag'] - gaia_star['phot_rp_mean_mag'])
                
                parallax = gaia_star['parallax']
                if parallax is not None and not np.ma.is_masked(parallax) and parallax > 0:
                    distance_pc = 1000.0 / parallax
                    distance_ly = distance_pc * 3.262
                    abs_mag = gaia_star['phot_g_mean_mag'] - 5 * np.log10(distance_pc / 10.0)
                else:
                    distance_ly = 0
                    abs_mag = None
                
                pmra = gaia_star['pmra']
                pmra_val = float(pmra) if (pmra is not None and not np.ma.is_masked(pmra)) else 0.0
                
                pmdec = gaia_star['pmdec']
                pmdec_val = float(pmdec) if (pmdec is not None and not np.ma.is_masked(pmdec)) else 0.0
                
                matched_star = {
                    'source_id': int(gaia_star['source_id']),
                    'x': float(reference_star.xpos),
                    'y': float(reference_star.ypos),
                    'ra': float(gaia_star['ra']),
                    'dec': float(gaia_star['dec']),
                    'parallax': float(parallax) if (parallax is not None and not np.ma.is_masked(parallax)) else 0.0,
                    'pmra': pmra_val,
                    'pmdec': pmdec_val,
                    'g_mag': float(gaia_star['phot_g_mean_mag']),
                    'bp_mag': float(gaia_star['phot_bp_mean_mag']),
                    'rp_mag': float(gaia_star['phot_rp_mean_mag']),
                    'color': color,
                    'abs_mag': float(abs_mag) if abs_mag is not None else None,
                    'distance': float(distance_ly),
                    'fwhm': float((reference_star.fwhmx_arcsec + reference_star.fwhmy_arcsec) / 2.0),
                    'snr': float(reference_star.SNR)
                }
                
                # Store magnitudes from each channel
                for layer_num, psf_star in star_layers.items():
                    if layer_num == 0:
                        matched_star['img_mag_r'] = float(psf_star.mag)
                    elif layer_num == 1:
                        matched_star['img_mag_g'] = float(psf_star.mag)
                    elif layer_num == 2:
                        matched_star['img_mag_b'] = float(psf_star.mag)
                
                matched_stars.append(matched_star)
                matched_count += 1
                
            except Exception as e:
                if i < 10:
                    print(f"  Gaia {i+1}: Exception during matching: {e}")
                continue
        
        print(f"\n=== MATCHING RESULTS ===")
        print(f"Successfully matched: {len(matched_stars)} stars")
        print(f"Rejected out of bounds: {rejected_out_of_bounds}")
        print(f"Rejected too far (>{match_threshold_px}px): {rejected_too_far}")
        print(f"Rejected too faint (mag>{self.max_img_mag}): {rejected_too_faint}")
        print(f"Total Gaia stars processed: {len(gaia_results)}")
        
        if len(matched_stars) == 0:
            print("\n⚠️ WARNING: No matches found!")
            print("Possible issues:")
            print("1. Astrometric solution may be incorrect")
            print("2. Star detection settings may need adjustment")
            print("3. Image and Gaia catalog may not overlap")
            print("4. Try checking image orientation or re-solving")
        elif len(matched_stars) < 10:
            print(f"\n⚠️ WARNING: Very few matches ({len(matched_stars)})!")
            print("This might indicate an issue with the astrometric solution.")
        
        return matched_stars

# ============================================================================
# HR Diagram Canvas
# ============================================================================

class HRDiagramCanvas(FigureCanvas):
    """Matplotlib canvas for HR diagram with PyQt6"""
    point_clicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        plt.style.use('dark_background')
        
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.star_data = None
        self.use_absolute_mag = True
        self.show_reference = True
        self.scatter = None
        self.color_slope = 1.0
        self.color_intercept = 0.0
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('axes_enter_event', lambda event: self.setCursor(Qt.CursorShape.CrossCursor))
        self.fig.canvas.mpl_connect('axes_leave_event', lambda event: self.setCursor(Qt.CursorShape.ArrowCursor))
        
        self.setup_plot()
    
    def setup_plot(self):
        """Setup plot appearance for dark theme"""
        self.ax.set_facecolor('#1e1e1e')
        self.ax.grid(True, alpha=0.3, color='#666666')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        self.fig.tight_layout()
    
    def plot_data(self, star_data, use_absolute_mag=True, show_reference=True, 
                  color_slope=1.0, color_intercept=0.0):
        """Plot HR diagram with calibrated image photometry"""
        self.star_data = star_data
        self.use_absolute_mag = use_absolute_mag
        self.show_reference = show_reference
        self.color_slope = color_slope
        self.color_intercept = color_intercept
        
        self.ax.clear()
        self.setup_plot()
        
        if not star_data or len(star_data) == 0:
            self.ax.text(0.5, 0.5, 'No data to display', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        color='white', fontsize=14)
            self.draw()
            return
        
        # Plot reference curves if enabled and using absolute magnitude
        if show_reference and use_absolute_mag:
            self.plot_reference_curves()
        
        colors = []
        mags = []
        point_colors = []
        point_sizes = []
        
        for star in star_data:
            if star['img_color'] is None or star['img_mag_calib'] is None:
                continue
            
            # img_color is already transformed to BP-RP equivalent
            color = star['img_color']
            
            if use_absolute_mag and star['abs_mag'] is not None:
                parallax = star['parallax']
                if parallax > 0:
                    distance_pc = 1000.0 / parallax
                    mag = star['img_mag_calib'] - 5 * np.log10(distance_pc / 10.0)
                else:
                    continue
            else:
                mag = star['img_mag_calib']
            
            colors.append(color)
            mags.append(mag)
            
            # Color-code by BP-RP temperature
            if color < 0.5:
                point_colors.append('#4da6ff')
            elif color < 1.0:
                point_colors.append('#ffffff')
            elif color < 1.5:
                point_colors.append('#ffff99')
            elif color < 2.5:
                point_colors.append('#ff9933')
            else:
                point_colors.append('#ff3333')
            
            size = 8
            point_sizes.append(size)
        
        if len(colors) == 0:
            self.ax.text(0.5, 0.5, 'No valid photometric data', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        color='white', fontsize=14)
            self.draw()
            return
        
        self.scatter = self.ax.scatter(colors, mags, c=point_colors, 
                                      s=point_sizes, alpha=0.7, 
                                      picker=True, zorder=3)
        
        # Labels
        self.ax.set_xlabel('Color Index', color='white', fontsize=12)
        ylabel = 'Absolute Magnitude' if use_absolute_mag else 'Apparent Magnitude'
        self.ax.set_ylabel(ylabel, color='white', fontsize=12)
        
        title = 'Hertzsprung-Russell Diagram'
        self.ax.set_title(title, color='white', fontsize=14, pad=20)
        
        self.ax.invert_yaxis()
        
        # Spectral class labels
        if use_absolute_mag and len(mags) > 0:
            y_range = max(mags) - min(mags)
            if y_range > 5:
                self.ax.text(0.1, -5, 'O', ha='center', va='center', 
                           color='#4da6ff', fontsize=10, weight='bold', alpha=0.5)
                self.ax.text(0.5, 0, 'A', ha='center', va='center', 
                           color='#ffffff', fontsize=10, weight='bold', alpha=0.5)
                self.ax.text(1.0, 2, 'G', ha='center', va='center', 
                           color='#ffff99', fontsize=10, weight='bold', alpha=0.5)
                self.ax.text(1.5, 6, 'K', ha='center', va='center', 
                           color='#ff9933', fontsize=10, weight='bold', alpha=0.5)
                self.ax.text(2.5, 10, 'M', ha='center', va='center', 
                           color='#ff3333', fontsize=10, weight='bold', alpha=0.5)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_reference_curves(self):
        """Plot theoretical reference curves on the HR diagram"""
        # Main Sequence
        ms_color, ms_mag = get_main_sequence_curve()
        self.ax.plot(ms_color, ms_mag, 'c-', linewidth=2, alpha=0.6, 
                    label='Main Sequence', zorder=1)
        
        # Giant Branch
        gb_color, gb_mag = get_giant_branch_curve()
        self.ax.plot(gb_color, gb_mag, 'orange', linewidth=2, alpha=0.6, 
                    label='Red Giants', linestyle='--', zorder=1)
        
        # Supergiants
        sg_color, sg_mag = get_supergiant_curve()
        self.ax.plot(sg_color, sg_mag, 'red', linewidth=2, alpha=0.6, 
                    label='Supergiants', linestyle='--', zorder=1)
        
        # White Dwarfs
        wd_color, wd_mag = get_white_dwarf_curve()
        self.ax.plot(wd_color, wd_mag, 'lightblue', linewidth=2, alpha=0.6, 
                    label='White Dwarfs', linestyle=':', zorder=1)
        
        self.ax.legend(loc='upper left', framealpha=0.7, fontsize=9)
    
    def on_click(self, event):
        """Handle click on data point"""
        if event.inaxes != self.ax or self.scatter is None or not self.star_data:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        colors = []
        mags = []
        indices = []
        
        for i, star in enumerate(self.star_data):
            if star['img_color'] is None or star['img_mag_calib'] is None:
                continue
            
            color = star['img_color']
            
            if self.use_absolute_mag and star['abs_mag'] is not None:
                parallax = star['parallax']
                if parallax > 0:
                    distance_pc = 1000.0 / parallax
                    mag = star['img_mag_calib'] - 5 * np.log10(distance_pc / 10.0)
                else:
                    continue
            else:
                mag = star['img_mag_calib']
            
            colors.append(color)
            mags.append(mag)
            indices.append(i)
        
        if len(colors) == 0:
            return
        
        min_dist = float('inf')
        closest_idx = -1
        
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        
        for i, (c, m) in enumerate(zip(colors, mags)):
            dx = (c - event.xdata) / x_range
            dy = (m - event.ydata) / y_range
            dist = dx*dx + dy*dy
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx >= 0 and min_dist < 0.001:
            actual_idx = indices[closest_idx]
            self.point_clicked.emit(actual_idx)

# ============================================================================
# Main Dialog
# ============================================================================

class HRDiagramDialog(QMainWindow):
    """Main window for HR Diagram generation"""
    
    def __init__(self, siril):
        super().__init__()
        self.setWindowTitle(f"Hertzsprung-Russell Diagram v{VERSION}")
        self.resize(1400, 800)
        
        self.siril = siril
        self.star_data = None
        self.calib_stats = None
        
        if not self.siril.is_image_loaded():
            QMessageBox.critical(self, "Error", "No image loaded in Siril")
            raise SystemExit(1)
        
        # Check if image is color
        fit = self.siril.get_image()
        if fit.data.ndim != 3:
            QMessageBox.critical(self, "Error", 
                               "Image must be a color (RGB) image.\n"
                               "This script requires color photometry.")
            raise SystemExit(1)
        
        try:
            self.siril.pix2radec(0, 0)
        except:
            QMessageBox.critical(self, "Error", "Image has no astrometric solution.\n"
                               "Please plate solve the image first.")
            raise SystemExit(1)
        
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
        
        splitter.setSizes([350, 1050])
    
    def create_left_panel(self):
        left_widget = QWidget()
        left_widget.setFixedWidth(350)
        layout = QVBoxLayout(left_widget)
        layout.setSpacing(10)
        
        title = QLabel("Hertzsprung-Russell Diagram")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        version_label = QLabel(f"Version {VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        
        query_group = QGroupBox("Gaia Query Parameters")
        query_layout = QVBoxLayout(query_group)
        
        min_mag_layout = QHBoxLayout()
        min_mag_layout.addWidget(QLabel("Min Gaia Mag:"))
        self.min_mag_spin = QDoubleSpinBox()
        self.min_mag_spin.setRange(0.0, 20.0)
        self.min_mag_spin.setValue(6.0)
        self.min_mag_spin.setSingleStep(0.5)
        self.min_mag_spin.setToolTip("Minimum Gaia magnitude to query.\nBrighter stars have lower magnitudes.")
        min_mag_layout.addWidget(self.min_mag_spin)
        min_mag_layout.addStretch()
        query_layout.addLayout(min_mag_layout)
        
        max_mag_layout = QHBoxLayout()
        max_mag_layout.addWidget(QLabel("Max Gaia Mag:"))
        self.max_mag_spin = QDoubleSpinBox()
        self.max_mag_spin.setRange(0.0, 20.0)
        self.max_mag_spin.setValue(18.0)
        self.max_mag_spin.setSingleStep(0.5)
        self.max_mag_spin.setToolTip("Maximum Gaia magnitude to query.\nFainter stars have higher magnitudes.")
        max_mag_layout.addWidget(self.max_mag_spin)
        max_mag_layout.addStretch()
        query_layout.addLayout(max_mag_layout)
        
        max_stars_layout = QHBoxLayout()
        max_stars_layout.addWidget(QLabel("Max Stars:"))
        self.max_stars_spin = QSpinBox()
        self.max_stars_spin.setRange(100, 10000)
        self.max_stars_spin.setValue(5000)
        self.max_stars_spin.setSingleStep(100)
        self.max_stars_spin.setToolTip("Maximum number of stars to query from Gaia.\nHigher values take longer but provide more data.")
        max_stars_layout.addWidget(self.max_stars_spin)
        max_stars_layout.addStretch()
        query_layout.addLayout(max_stars_layout)
        
        max_img_mag_layout = QHBoxLayout()
        max_img_mag_layout.addWidget(QLabel("Max Image Mag:"))
        self.max_img_mag_spin = QDoubleSpinBox()
        self.max_img_mag_spin.setRange(0.0, 25.0)
        self.max_img_mag_spin.setValue(0.0)
        self.max_img_mag_spin.setSingleStep(0.5)
        self.max_img_mag_spin.setSpecialValueText("No limit")
        self.max_img_mag_spin.setToolTip("Maximum instrumental magnitude from image.\nSet to 0 for no limit. Use to filter out faint/noisy stars.")
        max_img_mag_layout.addWidget(self.max_img_mag_spin)
        max_img_mag_layout.addStretch()
        query_layout.addLayout(max_img_mag_layout)
        
        layout.addWidget(query_group)
        
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.use_abs_mag_check = QCheckBox("Use Absolute Magnitude")
        self.use_abs_mag_check.setChecked(True)
        self.use_abs_mag_check.setToolTip("Plot absolute magnitude (requires parallax data).\nUnchecked: plot apparent magnitude as seen from Earth.")
        self.use_abs_mag_check.stateChanged.connect(self.update_plot)
        display_layout.addWidget(self.use_abs_mag_check)
        
        self.show_reference_check = QCheckBox("Show Reference Curves")
        self.show_reference_check.setChecked(True)
        self.show_reference_check.setToolTip("Display theoretical stellar evolution curves:\nMain Sequence, Red Giants, Supergiants, and White Dwarfs.")
        self.show_reference_check.stateChanged.connect(self.update_plot)
        display_layout.addWidget(self.show_reference_check)
        
        layout.addWidget(display_group)
        
        btn_generate = QPushButton("Generate HR Diagram")
        btn_generate.setToolTip("Start star detection, Gaia query, and HR diagram generation.\nThis may take a few minutes depending on the number of stars.")
        btn_generate.clicked.connect(self.start_processing)
        layout.addWidget(btn_generate)
        
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)

        stats_title = QLabel("Statistics")
        stats_title_font = QFont()
        stats_title_font.setBold(True)
        stats_title.setFont(stats_title_font)
        layout.addWidget(stats_title)
        
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setFixedHeight(150)
        stats_scroll.setFrameShape(QScrollArea.Shape.StyledPanel)
        
        self.stats_label = QLabel("No data")
        self.stats_label.setWordWrap(True)
        self.stats_label.setMargin(5)
        
        stats_scroll.setWidget(self.stats_label)
        layout.addWidget(stats_scroll)
      
        layout.addStretch()
        
        btn_close = QPushButton("Close")
        btn_close.setToolTip("Close the HR Diagram window")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)
        
        info_label = QLabel(
            "<b>Instructions:</b><br>"
            "1. Ensure image has an astrometric solution<br>"
            "2. Ensure SPCC has been applied on the image<br>"
            "3. Adjust Gaia query parameters if needed<br>"
            "4. Click 'Generate HR Diagram'<br>"
            "5. Click points on the diagram to see details<br>"
            "<br>"
            "<b>Note:</b> Uses B-R color index (Blue minus Red)"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        return left_widget
    
    def create_right_panel(self):
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        self.hr_canvas = HRDiagramCanvas()
        self.hr_canvas.point_clicked.connect(self.show_star_details)
        
        toolbar = NavigationToolbar(self.hr_canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.hr_canvas)
        
        return right_widget
    
    def start_processing(self):
        """Start the Gaia query and star matching process"""
        # Check if SPCC has been applied
        try:
            history = self.siril.get_image_history()
            spcc_found = False
            if history:
                for entry in history:
                    if 'SPCC' in entry.upper():
                        spcc_found = True
                        break
            
            if not spcc_found:
                reply = QMessageBox.warning(
                    self, 
                    "Warning: SPCC Not Applied",
                    "It appears that SPCC (Spectrophotometric Color Calibration) "
                    "has not been applied to this image.\n\n"
                    "For accurate color calibration and HR diagram results, "
                    "it is strongly recommended to apply SPCC first.\n\n"
                    "Do you want to continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
        except Exception as e:
            # If we can't check history, just continue with a warning
            print(f"Warning: Could not verify SPCC application: {e}")
        
        self.status_label.setText("Starting...")
        self.progress_bar.setValue(0)
        
        min_mag = self.min_mag_spin.value()
        max_mag = self.max_mag_spin.value()
        max_stars = self.max_stars_spin.value()
        max_img_mag = self.max_img_mag_spin.value()
        
        self.thread = QThread(self)
        self.worker = GaiaQueryWorker(self.siril, min_mag, max_mag, max_stars, max_img_mag)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
    
    def on_progress(self, message, percent):
        self.status_label.setText(message)
        self.progress_bar.setValue(percent)
    
    def on_finished(self, star_data, error, calib_stats):
        if error:
            self.status_label.setText("Error")
            self.progress_bar.setValue(0)
            QMessageBox.warning(self, "Processing Error", error)
            return
        
        if star_data:
            self.star_data = star_data
            self.calib_stats = calib_stats
            
            n_with_parallax = sum(1 for s in star_data if s['parallax'] > 0)
            n_with_img_phot = sum(1 for s in star_data if s.get('img_color') is not None)
            
            valid_fwhm = [s['fwhm'] for s in star_data if s['fwhm'] > 0]
            avg_fwhm = np.mean(valid_fwhm) if len(valid_fwhm) > 0 else 0.0
            
            valid_snr = [s['snr'] for s in star_data if s['snr'] > 0]
            avg_snr = np.mean(valid_snr) if len(valid_snr) > 0 else 0.0
            
            stats_text = (f"Total stars: {len(star_data)}\n"
                         f"With parallax: {n_with_parallax}\n"
                         f"With image photometry: {n_with_img_phot}")
            
            if avg_fwhm > 0:
                stats_text += f"\nAvg FWHM: {avg_fwhm:.2f}\""
            if avg_snr > 0:
                stats_text += f"\nAvg SNR: {avg_snr:.1f}"
            
            if calib_stats and 'n_stars' in calib_stats:
                stats_text += f"\n\nCalibration (linear):"
                stats_text += f"\nStars used: {calib_stats['n_stars']}"
                stats_text += f"\nMag ZP: {calib_stats['mag_zp']:.3f}"
                stats_text += f"\nColor slope: {calib_stats['color_slope']:.4f}"
                stats_text += f"\nColor intercept: {calib_stats['color_intercept']:.4f}"
                stats_text += f"\nR²: {calib_stats['r_squared']:.4f}"
            
            self.stats_label.setText(stats_text)
            
            self.update_plot()
            
            self.status_label.setText(f"Complete - {len(star_data)} stars processed")
            self.progress_bar.setValue(100)
            
            if n_with_img_phot > 0:
                self.siril.log(f"HR Diagram: {n_with_img_phot} stars with calibrated image photometry", 
                              color=s.LogColor.GREEN)
            else:
                self.siril.log(f"HR Diagram: {len(star_data)} stars plotted", 
                              color=s.LogColor.GREEN)
        else:
            self.status_label.setText("No data")
            self.progress_bar.setValue(0)
    
    def update_plot(self):
        """Update the HR diagram plot"""
        if self.star_data:
            use_abs = self.use_abs_mag_check.isChecked()
            show_ref = self.show_reference_check.isChecked()
            
            color_slope = 1.0
            color_intercept = 0.0
            if self.calib_stats:
                color_slope = self.calib_stats.get('color_slope', 1.0)
                color_intercept = self.calib_stats.get('color_intercept', 0.0)
            
            self.hr_canvas.plot_data(self.star_data, use_abs, show_ref,
                                    color_slope, color_intercept)
    
    def show_star_details(self, index):
        """Show detailed information for a clicked star"""
        if not self.star_data or index >= len(self.star_data):
            return
        
        star = self.star_data[index]
        
        coord = SkyCoord(ra=star['ra']*u.deg, dec=star['dec']*u.deg, frame='icrs')
        ra_str = coord.ra.to_string(u.hour, sep=':', precision=2)
        dec_str = coord.dec.to_string(u.deg, sep=':', precision=2)
        
        details = f"<b>Gaia DR3 {star['source_id']}</b><br><br>"
        details += "<b>Position:</b><br>"
        details += f"RA: {ra_str}<br>"
        details += f"Dec: {dec_str}<br>"
        details += f"Parallax: {star['parallax']:.4f} mas<br>"
        
        if star['distance'] > 0:
            details += f"Distance: {star['distance']:.1f} ly<br>"
        
        details += "<br><b>Gaia Reference Data:</b><br>"
        details += f"G magnitude: {star['g_mag']:.3f}<br>"
        details += f"BP-RP color: {star['color']:.3f}<br>"
        
        if star['abs_mag'] is not None:
            details += f"Absolute magnitude: {star['abs_mag']:.3f}<br>"
        
        details += "<br><b>Image Photometry (PSF):</b><br>"
        
        if 'img_mag_r' in star:
            details += f"R: {star['img_mag_r']:.3f}<br>"
        if 'img_mag_g' in star:
            details += f"G: {star['img_mag_g']:.3f}<br>"
        if 'img_mag_b' in star:
            details += f"B: {star['img_mag_b']:.3f}<br>"
        
        if star.get('img_mag_calib') is not None:
            details += f"<br><b>Calibrated:</b><br>"
            details += f"Magnitude: {star['img_mag_calib']:.3f}<br>"
            if star.get('img_color') is not None:
                details += f"Color (BP-RP equiv): {star['img_color']:.3f}<br>"
            if star.get('img_color_raw') is not None:
                details += f"Color B-R (raw): {star['img_color_raw']:.3f}<br>"
        
        if star.get('fwhm', 0) > 0:
            details += f"<br>FWHM: {star['fwhm']:.2f}\"<br>"
        if star.get('snr', 0) > 0:
            details += f"SNR: {star['snr']:.1f}<br>"
        
        details += "<br><b>Proper Motion:</b><br>"
        details += f"μ_α: {star['pmra']:.3f} mas/yr<br>"
        details += f"μ_δ: {star['pmdec']:.3f} mas/yr"
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Star Details")
        msg.setText(details)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

# ============================================================================
# Main Entry Point
# ============================================================================

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
        app.setApplicationName("Hertzsprung-Russell Diagram")
        
        # Set application icon (PNG format - more reliable than JPEG)
        icon_data = base64.b64decode("""iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACNklEQVR4nO2bMW4CMRBFTZQ2Utowx0idG2xHjsABUtBGCNGmyAE4Quj2BtR7DItjJEXkZTJr72Iz9thZP8kSXgmY+Z75LDYslsvlt5oxd9IBSFMFkA5AmiqAdADSVAGkA5CmCiAdgDRXC9A0Tcw4xJh9BdyHPvH5Q/ePuw2wBCNBUAXg5G3zkvAWwJWshAh6+9SPUIJbQBJbwuYa7M5erzV7E/QWwGV43QbU6qD7EYupcvdth6AKoCKY5DExRXDS/Q7dXC9CsAdgEVzJrg5aHdcJPiK74SUjArTjnlCcB/ia3BTFCTDAsvqYqXZgEcBV5sc1KK11P7iA3ZmtEtgqgIpgksdwiqAUTzuw3ghhEVzJaq0VAJ8xQnseLfN/Z4LcFHkrTDGrjCthauUN0QQAGHrA5foWzXd873ll0pioLUB7nSavlBrMUxPdAwCgH65kJUWYvQlWAVK+mcvwOI3Ql+QVQJM1c31q+pESkfsAKgJNWp8aBS9tkliqB0gHIE0WAtByT1X+SmX0XcCWNN7g5N4JMmRRATbo7u4thx9jZCtAKqoA0gG4oD0fywOyMUEbtqRDNj3GyLYCbNC9P58TIBdFCRADlhbYo6Ox9xRHYYzcXAF7ci5I55zQnufwgCgm+Pl1EeHtlbciOJLGsHvAw+PfORYjR24WoLSep7C0ABYh9xWnsLcA7XluD+AmignmnjTGS4DSfi/cttMbK4v6t7mZUwWQDkCaKoB0ANJUAaQDkGb2AvwA1tqvDXMFjcoAAAAASUVORK5CYII=""")
        pixmap = QPixmap()
        pixmap.loadFromData(icon_data)
        app_icon = QIcon(pixmap)
        app.setWindowIcon(app_icon)
        
        window = HRDiagramDialog(siril)
        window.show()
        sys.exit(app.exec())
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
