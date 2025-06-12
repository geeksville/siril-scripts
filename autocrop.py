##############################################################################################
#
#  _______             ____        __  ________                        
# /_  __(_)___  __  __/ __ \____  / /_/ ____/ /_  ____ _________  _____
#  / / / / __ \/ / / / / / / __ \/ __/ /   / __ \/ __ `/ ___/ _ \/ ___/
# / / / / / / / /_/ / /_/ / /_/ / /_/ /___/ / / / /_/ (__  )  __/ /    
#/_/ /_/_/ /_/\__, /_____/\____/\__/\____/_/ /_/\__,_/____/\___/_/     
#            /____/                                                    
#
#                   ╔═╗┌─┐┌┬┐┬─┐┌─┐┌┐┌┌─┐┌┬┐┬ ┬         
#                   ╠═╣└─┐ │ ├┬┘│ │││││ ││││└┬┘         
#                   ╩ ╩└─┘ ┴ ┴└─└─┘┘└┘└─┘┴ ┴ ┴          
#       ╔═╗┌─┐┌┬┐┬─┐┌─┐┌─┐┬ ┬┌─┐┌┬┐┌─┐┌─┐┬─┐┌─┐┌─┐┬ ┬┬ ┬
#       ╠═╣└─┐ │ ├┬┘│ │├─┘├─┤│ │ │ │ ││ ┬├┬┘├─┤├─┘├─┤└┬┘
#       ╩ ╩└─┘ ┴ ┴└─└─┘┴  ┴ ┴└─┘ ┴ └─┘└─┘┴└─┴ ┴┴  ┴ ┴ ┴ 
#
#              by Gottfried Rotter (gofraro@gmail.com)
#                           June 2025
#
#		              Follow the light dots
#
##################### Description ############################################################
# autocrop.py
#
# Siril python script for autocropping images after stacking with framing=max
# 
# This script takes a already loaded fits-image, creates a mask and determines a contour of 
# the content. In a second step, a rectangular box is determined that best fits into the contour.
# The original image will than be cropped based on this rectangle.
# In an optional step, the image can be further cropped to eliminate edges with bad SNR.
# The result is saved in fits format in the defined Siril working folder with a filename modification 
# The full fits header is transformed and updated with an updated history
# Optionally, the result can be automatically loaded into Siril for further processing
# Siril log messages will be provided in various levels (see pysiril API reference)
# 
##################### INSTRUCTIONS ##########################################################
#
# usage: autocrop.py [-h] [--threshold THRESHOLD] [--debug] [--loadimage]
#                    [--refinecrop [REFINECROP]] [--sim] [--stat]
# Autocrop for astro images with optional threshold and debugging.
# options:
#   -h, --help            show this help message and exit
#   --threshold THRESHOLD
#                         Threshold (between 0.0 and 1.0), optional.
#   --debug               activate Debug-Mode with additional data and graphics
#                         in Siril log and /debug_output
#   --loadimage           Loads the saved crop image directly back into Siril
#   --refinecrop [REFINECROP]
#                         Enables optional fine cropping based on noise
#                         evaluation. Default = 2.0.
#   --sim                 Simulation mode: no image will be saved or reloaded. Instead
#                         the contours will be shown in the GUI
#   --stat                Show image statistics like dimensions and area
#                         changes.
# 
#
# In the command line of the Siril console enter pyscript autocrop.py [script_argv]
#
# In Siril scripts, use pyscript scriptname.py [script_argv] and ensure that a fits file is loaded
#
#############################################################################################
# MIT License
# Copyright (c) 2025 Gottfried Rotter (gofraro@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
#############################################################################################
#
# Change_log: 0.9.1 
#                   - replace tempfile and output handling with Siril API function
#                   - remove import tempfile, no longer needed
#                   - add new option --stat to printout statistics
#                   - add new option --sim to run simulations without changes 
#                     can be used in any combination of options. i.e. --sim --stat --debug
#                     The latter provides additional info in the log and /debug_output
#              0.9.3 
#                   - tried to replace matlib with Siril plot function, but
#                       siril plot pop-ups-only a window instead of saving file in parallel
#                       for printout, matlib is still needed. Issue reported. Will be fixed next.
#                   - --sim draws contours in display, but color settings in polygone is weird
#              0.9.6
#                   - stabilized version. Image orientation behaviour of Siril considered
#                   - contour drawing via polygon function implemented, color control adapted
#                   - handling of files with blanks or dots in filename enabled
#                   - output file normalisation (especially for Seestar generated stacking files)
#              0.9.8
#                   - allow mono image processing
#              0.9.9
#                   - more detailed stat output, incl. a bar-chart
#                   - warning message when already autocropped file is initially loaded
#                   - file extension handling changed
#                   - script end routine updated with close and load of original file
#              0.9.9.1
#                   - signifcant performance improvement implemnted with automated adaption
#                     reducing autocrop runtime by factor 10x
#              0.9.9.2
#                   - file handling correction to allow file loaded from outside the working dir
#
# What's in the pipeline:
#                   - replacement of matlib plot function
#                   - automatic epsilon setting for polygon point reduction
#                   - GUI version with tkinter
#                   - did I mention already testing, testing, testing
#                   - documentation and cleanup
#
#############################################################################################

import sirilpy as s
from sirilpy import plot as sp
from sirilpy import Polygon, FPoint
import numpy as np
import cv2
import sys
import argparse
import time
import os
s.ensure_installed("astropy")
from astropy.io import fits
from astropy.io.fits import Header
s.ensure_installed("matplotlib")
from matplotlib import pyplot as plt
s.ensure_installed("largestinteriorrectangle")
import largestinteriorrectangle as lir
from sirilpy.exceptions import SirilConnectionError, SirilError

__version__ = "0.9.9.2"
global_debug_dir= "debug_output"

# Siril-Verbindung
siril = s.SirilInterface()
try:
   siril.connect()
   print("Connected successfully!")
except SirilConnectionError as e:
   print(f"Connection failed: {e}")
   quit()
   
# clear the screen from previous overlays in sim-mode
siril.overlay_clear_polygons()

# Siril Logging #######################################################
from sirilpy.enums import LogColor
def log_info(msg):
    siril.log(msg, color=LogColor.DEFAULT)
def log_warn(msg):
    siril.log(f"[WARNING] {msg}", color=LogColor.SALMON)
def log_error(msg):
    siril.log(f"[ERROR] {msg}", color=LogColor.RED)
def log_success(msg):
    siril.log(f"[OK] {msg}", color=LogColor.GREEN)
def log_debug(msg):
    siril.log(f"[DEBUG] {msg}", color=LogColor.BLUE)
########################################################################
# Farbdefinition für Plot RGBA
def rgba_to_int(r, g, b, a=255):
    """
    Wandelt RGBA-Werte (0–255) in einen 32-bit Integer im Siril-kompatiblen Format: 0xRRGGBBAA
    """
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0, min(255, a))
    return (r << 24) + (g << 16) + (b << 8) + a
########################################################################

log_info(f"***** Executing AutoCrop v{__version__} *****")

def stretch_grayscale(gray, debug=False):
    p_low, p_high = np.percentile(gray, (1, 99))
    stretched = np.clip((gray - p_low) / (p_high - p_low), 0, 1)
    return stretched
    
# Performanceoptimization for contour and lir-box
# keeping the balance of faster detection and accuracy
def get_resize_factor_for_mask(width: int, height: int) -> float:
    min_dim = min(width, height)
    if min_dim >= 4000:
        return 0.1
    elif min_dim >= 2500:
        return 0.2
    elif min_dim >= 1500:
        return 0.5
    else:
        return 1.0

def auto_crop(image, threshold=None, debug=False, simulate=False, top_down=False):
    
    # flip image for mask etc. horizontally, because the display is flipped
    if not top_down:  # == BOTTOM-UP
        image = np.flipud(image)
        if debug:
            log_debug("Mask vertically flipped due to BOTTOM-UP orientation.") 
             
    siril.update_progress("create grayscale", 0.2)
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    stretched = stretch_grayscale(gray, debug=True)

    siril.update_progress("set threshold", 0.3)
    if threshold is None:
        threshold = np.percentile(stretched, 95)
        log_info(f"Original Threshold: {threshold:.4f}")
        threshold -= 0.05
        log_info(f"Lowered Threshold: {threshold:.4f}")

    siril.update_progress("create Mask", 0.4)
    mask = (stretched > threshold).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Performanceoptimierung: Masken-Downscale je nach Bildgröße
    resize_factor = get_resize_factor_for_mask(mask.shape[1], mask.shape[0])
    scale_back = 1.0
    if resize_factor < 1.0:
        mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        scale_back = 1.0 / resize_factor
        log_info(f"Optimization activated with factor: {resize_factor:.2f}")


    mask_height, mask_width = mask.shape
    mask_area = mask_height * mask_width

    debug_dir = global_debug_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if debug:
        siril.update_progress("Plot histogram", -2.0)
        os.makedirs(debug_dir, exist_ok=True)

        # Save stretched + mask images
        plt.imsave(os.path.join(debug_dir, f"debug_stretched_{timestamp}.png"), stretched, cmap='gray')
        plt.imsave(os.path.join(debug_dir, f"debug_mask_{timestamp}.png"), mask, cmap='gray')

        # Histogram data
        hist_vals, bins = np.histogram(stretched, bins=100, range=(0, 1))

        # GUI Plot with Siril interface
        plot = sp.PlotData(
           title="Histogram of the stretched gray image",
           xlabel="Pixel Value",
           ylabel="Count",
            savename=f"debug_histogram_{timestamp}"  # only used as title in GUI
        )
        plot.add_series(
            x_coords=bins[:-1].tolist(),
            y_coords=hist_vals.tolist(),
            label="Histogram",
            plot_type=sp.PlotType.LINES
        )
        plot.add_series(
            x_coords=[threshold, threshold],
            y_coords=[0, max(hist_vals)],
            label=f"Threshold = {threshold:.2f}",
        plot_type=sp.PlotType.LINES
        )

        # Show in Siril GUI
        siril.xy_plot(plot)

        # Also save as PNG via matplotlib
        plt.figure()
        plt.title("Histogram of the stretched gray image")
        plt.plot(bins[:-1], hist_vals, label="Histogram")
        plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
        plt.xlabel("Pixel Value")
        plt.ylabel("Count")
        plt.legend()
        plotfile = os.path.join(debug_dir, f"debug_histogram_{timestamp}.png")
        plt.savefig(plotfile)
        plt.close()

        log_debug(f"Threshold histogram saved: {plotfile}")

    # Determine contour, this can last a while, especially with larger and complex mask shapes
    siril.update_progress("determine contour", 0.5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log_warn("No contours detected – no cutting.")
        return image, None, None, mask_area

    min_area = int(0.01 * mask_area)
    max_area = int(0.95 * mask_area)
    usable_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    if not usable_contours:
        log_warn("No suitable contour found – using largest fallback.")
        if contours:
            fallback_cnt = max(contours, key=cv2.contourArea)
            usable_contours = [fallback_cnt]
            if debug:
                fallback_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(fallback_mask, [fallback_cnt], -1, 255, thickness=cv2.FILLED)
                fallback_path = os.path.join(debug_dir, f"debug_fallback_kontur_{timestamp}.png")
                cv2.imwrite(fallback_path, fallback_mask)
                log_debug(f"Fallback contour mask saved: {fallback_path}")
        else:
            log_error("No contour found at all – abort.")
            return image, None, None, mask_area

    best_cnt = max(usable_contours, key=cv2.contourArea)

    if debug:
        overlay_contour = image.copy()
        if overlay_contour.dtype != np.uint8:
            overlay_contour = cv2.normalize(overlay_contour, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if scale_back != 1.0:
            best_cnt_scale = (best_cnt * scale_back).astype(np.int32)
        else:
            best_cnt_scale = (best_cnt).astype(np.int32)    
        cv2.drawContours(overlay_contour, [best_cnt_scale], -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, f"debug_best_contour_{timestamp}.png"), overlay_contour)
          

    # Determine the best rectangular shape within the contour. The more complex the contour the longer it takes
    siril.update_progress("determine LIR-box", 0.6)
    try:
        # Konturfläche berechnen
        # Fläche aus verkleinerter Kontur berechnen
        contour_area = cv2.contourArea(best_cnt)
        img_area = image.shape[0] * image.shape[1]


        contour_array = np.array([best_cnt[:, 0, :]])
        x, y, w, h = lir.lir(contour_array)

         # Für LIR-Box: Kontur skalieren
        if scale_back != 1.0:
            best_cnt = (best_cnt * scale_back).astype(np.int32)
            contour_area = cv2.contourArea(best_cnt)
            x = (x * scale_back).astype(np.int32)
            y = (y * scale_back).astype(np.int32)
            w = (w * scale_back).astype(np.int32)
            h = (h * scale_back).astype(np.int32)

        # Wenn Simulation aktiv, dann Box in GUI einzeichnen
        
        if simulate:
            siril.overlay_clear_polygons()

            # Farben definieren
            turkis = rgba_to_int(64, 224, 208)  # Für LIR-Box
            magenta = rgba_to_int(255, 0, 255)  # Für Kontur

            # --- Kontur einzeichnen (zuerst) ---
            try:
                # Epsilon steuert die Glättung – je höher, desto stärker vereinfacht
                epsilon = 3.0  # Du kannst hier mit 1.0–3.0 experimentieren
                simplified_cnt = cv2.approxPolyDP(best_cnt, epsilon, closed=True)

                if debug:
                    log_debug(f"Original contour points: {len(best_cnt)}")
                    log_debug(f"Simplified contour points: {len(simplified_cnt)}")

                if len(simplified_cnt) < 3:
                   log_warn("Simplified contour has fewer than 3 points – skipping overlay.")
                else:
                    contour_points = [FPoint(float(px), float(py)) for [[px, py]] in simplified_cnt]

                    contour_poly = Polygon(
                        points=contour_points,
                        color=magenta,  # definiere z.B. vorher: magenta = rgba_to_int(255, 0, 255)
                        fill=False
                    )

                siril.overlay_add_polygon(contour_poly)
                if debug:
                    log_debug("Contour polygon successfully added to overlay.")

            except Exception as e:
                log_error(f"Error during contour polygon overlay: {e}")

            # --- LIR-Box einzeichnen ---
            lir_points = [
                FPoint(float(x), float(y)),
                FPoint(float(x + w), float(y)),
                FPoint(float(x + w), float(y + h)),
                FPoint(float(x), float(y + h))
            ]
            lir_poly = Polygon(
                points=lir_points,
                color=turkis,
                fill=False
            )
            siril.overlay_add_polygon(lir_poly)


        # Debug-Overlay speichern
        if debug:
            overlay = image.copy()
            overlay_8bit = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.drawContours(overlay_8bit, [best_cnt], -1, (0, 255, 0), 1)
            cv2.rectangle(overlay_8bit, (x, y), (x + w, y + h), (0, 0, 255), 2)
            debug_path = os.path.join(debug_dir, f"debug_overlay_box_{timestamp}.png")
            cv2.imwrite(debug_path, overlay_8bit)
            log_debug(f"Overlay with contour and box saved: {debug_path}")

        # Bild zuschneiden
        cropped = image[y:y+h, x:x+w, :]
        log_info(f"[CUTTING rough] x={x}:{x+w}, y={y}:{y+h}, width={w}, height={h}")

        return cropped, (x, y, w, h), contour_area, img_area

    except Exception as e:
        log_error(f"Exception during LIR box calculation: {e}")
        img_area = image.shape[0] * image.shape[1]
        return image, None, None, img_area


def refine_crop_by_snr(image: np.ndarray, lir_box=None, tolerance: float = 2.0, debug=False, timestamp="", simulate=False) -> np.ndarray:
    """
    Beschneidet das Bild zusätzlich, wenn die Randbereiche deutlich verrauschter sind als das Zentrum.
    
    :param image: Cropped RGB-Image als NumPy-Array (H, W, 3)
    :param tolerance: SNR-Schwelle als Faktor zur Referenz-SNR (z. B. 0.4)
    :param debug: Speichert Debug-Informationen
    :param timestamp: Zeitstempel für Debug-Dateien
    :return: Optional weiter beschnittenes Bild
    """
    debug_dir = global_debug_dir
    
    # Coordinates of the LIR Box
    if lir_box is None:
        log_error("LIR-Box coordinates missing")
        raise ValueError("LIR-Box coordinates missing")

    x_lir, y_lir, w_lir, h_lir = lir_box
    
    # Calculate SNR of the scanned region
    def compute_snr(region):
        mean = np.mean(region)
        std = np.std(region)
        return mean / std if std != 0 else 0

    h, w, _ = image.shape
    margin = int(0.1 * min(h, w))  # 10 % Randbreite
    center = image[h//3:2*h//3, w//3:2*w//3, :]
    snr_center = compute_snr(center)

    if debug:
        log_debug(f"[SNR] center: {snr_center:.2f}")

    new_crop = [0, h, 0, w]  # y_start, y_end, x_start, x_end

    sides = {
        "top": (slice(0, margin), slice(None)),
        "bottom": (slice(h - margin, h), slice(None)),
        "left": (slice(None), slice(0, margin)),
        "right": (slice(None), slice(w - margin, w)),
    }

    # Analyze Image
    # Progress 8
    siril.update_progress("analyze image", 0.8)
    for side, (yslice, xslice) in sides.items():
        region = image[yslice, xslice, :]
        snr = compute_snr(region)
        if debug:
            log_debug(f"[SNR] edge {side}: {snr:.2f}")

        if snr < snr_center * tolerance:
            if debug:
                log_debug(f"[SNR] → {side} is additionally trimmed.")
            if side == "top":
                new_crop[0] += margin
            elif side == "bottom":
                new_crop[1] -= margin
            elif side == "left":
                new_crop[2] += margin
            elif side == "right":
                new_crop[3] -= margin

    y1, y2, x1, x2 = new_crop
    refined = image[y1:y2, x1:x2, :]

    log_info(f"[CUTTING fine] x={x1}:{x2}, y={y1}:{y2}, width={x2 - x1}, height={y2 - y1}")

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"debug_refine_crop_{timestamp}.png")
        refined_8bit = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(debug_path, cv2.cvtColor(refined_8bit, cv2.COLOR_RGB2BGR))
        log_debug(f"Refined cropping saved under: {debug_path}")
        
        refoverlay = image.copy()

        # Falls 16-bit oder Float: auf 8-bit skalieren (nur fürs Debugbild!)
        refoverlay_8bit = cv2.normalize(refoverlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Rechteck (rot)
        cv2.rectangle(refoverlay_8bit, (x1, y1), (x2, y2), (0, 0, 255), 2)

        debug_path = os.path.join(debug_dir, f"debug_refoverlay_box_{timestamp}.png")
        cv2.imwrite(debug_path, refoverlay_8bit)
        log_debug(f"Refine overlay with cutout saved as: {debug_path}")


    # Wir zeichnen die Feinzuschnitt-Kontur ins aktuelle Display ein

    if simulate:

        green = rgba_to_int(0, 255, 0)

        # Originalbild-Größe
        h, w = image.shape[:2]

        # Größe des refined_crop (also der zugeschnittene Bildausschnitt)
        box_w = refined.shape[1]
        box_h = refined.shape[0]

        # LIR-Box-Position (x, y) wird vom vorherigen Schritt übernommen
        # Position relativ zur LIR-Box (zentriert innerhalb)
        x0 = x_lir + (w_lir - box_w) // 2
        y0 = y_lir + (h_lir - box_h) // 2

        refined_points = [
            FPoint(float(x0), float(y0)),
            FPoint(float(x0 + box_w), float(y0)),
            FPoint(float(x0 + box_w), float(y0 + box_h)),
            FPoint(float(x0), float(y0 + box_h))
        ]

        refined_poly = Polygon(
            points=refined_points,
            color=green,
            fill=False
        )

        siril.overlay_add_polygon(refined_poly)

    return refined

def main(threshold, debug, loadimage, refinecrop, simulate, stat):
    siril.update_progress("process input", 0.0)
    log_info(f"Threshold={threshold}, Debugging={debug}, LoadImage={loadimage}, Refinecrop={refinecrop}, Simulate={simulate}, Stat={stat}")

    # Warning, when autocrop-file loaded 
    try:
        img = siril.get_image()
        data = img.data
        log_info(f"Shape of loaded image: {data.shape}")

        if "autocrop.fit" in siril.get_image_filename():
            log_warn(f"The loaded image appears to have already been autocropped: '{siril.get_image_filename()}'")
    except SirilConnectionError:
        log_error("No image loaded in Siril. Please load one before running the script.")
        return
        
    # Achtung: Wenn ROWORDER not TOP-DOWN, müssen wir stretch/mask/kontur spiegeln
    try:
        header = siril.get_image_fits_header(return_as='dict')
        roworder = header.get("ROWORDER", "").strip().upper()
        is_top_down = (roworder == "TOP-DOWN")
        log_info(f"Image ROWORDER: {roworder if roworder else 'UNKNOWN (assuming BOTTOM-UP)'}")
    except Exception as e:
        log_warn(f"Could not determine ROWORDER, assuming BOTTOM-UP: {e}")
        is_top_down = False

    # Conversation for processing Mono images
    is_gray = False  # Flag für spätere Rückkonvertierung

    if data.ndim == 3 and data.shape[0] == 3:
        # Siril liefert (3, H, W) → transponieren zu (H, W, 3)
        data = np.transpose(data, (1, 2, 0))
    elif data.ndim == 2:
        # Monochromes Bild (H, W) → auf (H, W, 3) erweitern
        data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        is_gray = True
    elif data.ndim == 3 and data.shape[2] != 3:
        log_error("Invalid image format – expected RGB image (H, W, 3).")
        raise ValueError("Invalid image format – expected (H, W, 3).")

    data = data.astype(np.float32)

    # Start autocrop procedure
    siril.update_progress("start autocrop process", 0.1)
    start_autocrop = time.perf_counter()
    cropped, (x, y, w, h), contour_area, original_area = auto_crop(
    data, threshold=threshold, debug=debug, simulate=simulate, top_down=is_top_down
    )
    end_autocrop = time.perf_counter()
    if debug:
        print(f"Runtime autocrop: {end_autocrop - start_autocrop:.2f} seconds")
    
    # Falls refinecrop aktiviert ist
    if refinecrop is not None:
        start_refinecrop = time.perf_counter()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        siril.update_progress("start refinecrop", 0.7)
        cropped = refine_crop_by_snr(cropped, lir_box=(x, y, w, h), tolerance=refinecrop, debug=debug, timestamp=timestamp, simulate=simulate)
        end_refinecrop = time.perf_counter()
        if debug:
            print(f"Runtime refinecrop: {end_refinecrop - start_refinecrop:.2f} seconds")
    


    # Statistik optional ausgeben
    if stat and (x is not None):
        # Originalgröße
        original_h, original_w = data.shape[:2]
        original_area = original_w * original_h
        original_area_mp = original_area / 1_000_000

        # Contour-Fläche
        contour_area_mp = contour_area / 1_000_000
        contour_percent_orig = 100 * contour_area / original_area

        # AutoCrop-Fläche
        area_autocrop = int(w) * int(h)
        area_autocrop_mp = area_autocrop / 1_000_000
        percent_autocrop_orig = 100 * area_autocrop / original_area
        percent_autocrop_contour = 100 * area_autocrop / contour_area

        # Final Crop
        cut_h, cut_w = cropped.shape[:2]
        refined_area = cut_w * cut_h
        refined_area_mp = refined_area / 1_000_000
        percent_refined_orig = 100 * refined_area / original_area
        percent_refined_contour = 100 * refined_area / contour_area
        percent_refined_autocrop = 100 * refined_area / area_autocrop

        # Reduktionen
        width_reduction = original_w - cut_w
        height_reduction = original_h - cut_h
        percent_width_reduction = 100 * width_reduction / original_w
        percent_height_reduction = 100 * height_reduction / original_h

        # Ausgabe
        print("======================== [Image Stats] =======================")
        print(f"Original WxH:       {original_w} x {original_h} = {original_area_mp:.3f} MP")
        print(f"Contour Area:       {contour_area_mp:.3f} MP → {contour_percent_orig:.1f}% of original")
        print(f"AUTOCROP WxH:       {w} x {h} = {area_autocrop_mp:.3f} MP")
        if refinecrop is not None:
            print(f"FINECROP WxH:       {cut_w} x {cut_h} = {refined_area_mp:.3f} MP")
        print(f"Width reduction:    {width_reduction} px → {percent_width_reduction:.1f}%")
        print(f"Height reduction:   {height_reduction} px → {percent_height_reduction:.1f}%")

        def print_bar(label, percent, width=30):
            filled = int(width * percent / 100)
            bar = '█' * filled + '-' * (width - filled)
            print(f"{label:<12}: |{bar}| {percent:.1f}%")

        print("================== Image Utilization Analysis ================")
        print_bar("Contour of Original   ", contour_percent_orig)
        print_bar("AutoCrop of Original  ", percent_autocrop_orig)
        print_bar("AutoCrop of Contour   ", percent_autocrop_contour)
        if refinecrop is not None:
            print_bar("RefineCrop of Contour ", percent_refined_contour)
            print_bar("RefineCrop of AutoCrop", percent_refined_autocrop)
        print("==============================================================")
    
    # Simulation: Kein Speichern oder Laden
    if simulate:
          log_success("--sim enabled: Skipping save/load, exiting after processing.")
          return


    # Ausgabe der Ergebnisse
    siril.update_progress("prepare output", 0.9)
    
    # conversation back to original image format
    if is_gray:
        # Einen Kanal extrahieren (z. B. den ersten) und wieder transponieren für FITS
        cropped_fits_data = cropped[:, :, 0]  # zurück zu (H, W)
    else:
        # wie gehabt → transponieren zu (3, H, W)
        cropped_fits_data = np.transpose(cropped, (2, 0, 1))
    
    # Nur wenn das Bild außerhalb des erwarteten 0–1-Bereichs liegt, i.e. Seestar Images
    if cropped_fits_data.dtype == np.float32 and np.max(cropped_fits_data) > 1.0:
        log_info("Normalizing cropped_fits_data to range 0.0 - 1.0")
        cropped_fits_data = (cropped_fits_data - np.min(cropped_fits_data)) / (np.max(cropped_fits_data) - np.min(cropped_fits_data))

    if debug:
        sirilwd = siril.get_siril_wd()
        log_debug(f"WD: {sirilwd}")
        original_path = siril.get_image_filename()
        log_debug(f"Original Image: {original_path}")

    original_path = siril.get_image_filename()
    original_base = os.path.splitext(os.path.basename(original_path))[0]
    ext = siril.get_siril_config("core", "extension")
    output_name = f"{original_base}_autocrop{ext}"


    try:
        with siril.image_lock():
            siril.set_image_pixeldata(cropped_fits_data)
            if debug:
                log_debug("Image data successfully set via image_lock().")
    except siril.ProcessingThreadBusyError:
        log_error("Processing thread is currently busy.")
        return
    except siril.ImageDialogOpenError:
        log_error("Image processing dialog is open; cannot modify image.")
        return
    except siril.SirilError as e:
        log_error(f"SirilError occurred: {e}")
        return

    # fits header update and save 
    siril.cmd("update_key -comment", f'"AutoCrop v{__version__}"')
    siril.cmd("save", f'"{output_name}"')
    swd = siril.get_siril_wd()
    log_success(f"Cropped image saved as {output_name} in {swd}")
    
    if not loadimage:
        siril.cmd("load", f'"{original_path}"')

    if loadimage:
        if os.path.exists(output_name):
            siril.cmd("load", f'"{output_name}"')
            log_success(f"Result image loaded for further processing: {output_name}")
        else:
            log_error(f"Could not find {output_name} to load.")

    siril.update_progress("Finished", 1.0)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocrop for astro images with optional threshold and debugging.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold (between 0.0 and 1.0), optional.")
    parser.add_argument("--debug", action="store_true", help="activate Debug-Mode with additional data and graphics in Siril log and /debug_output")
    parser.add_argument("--loadimage", action="store_true", help="Loads the saved crop image directly back into Siril")
    parser.add_argument("--refinecrop", nargs="?", const=2.0, type=float, default=None, help="Enables optional fine cropping based on noise evaluation. Default = 2.0.")
    parser.add_argument("--sim", action="store_true", help="Simulation mode: no image will be saved or reloaded.")
    parser.add_argument("--stat", action="store_true", help="Show image statistics like dimensions and area changes.")
    args = parser.parse_args()

    main(
        threshold=args.threshold,
        debug=args.debug,
        loadimage=args.loadimage,
        refinecrop=args.refinecrop,
        simulate=args.sim,
        stat=args.stat
    )
