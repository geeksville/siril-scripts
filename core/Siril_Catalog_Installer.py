# (c) Adrian Knagg-Baugh 2025
# Siril Catalog Installer
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.0

VERSION = "1.0.0"

# Catalog retrieval details
ASTRO_RECORD = 14692304
ASTRO_INDEXLEVEL = 8

SPCC_RECORD = 14697693
SPCC_CHUNKLEVEL = 1
SPCC_INDEXLEVEL = 8

import sirilpy as s
from sirilpy import tksiril
import argparse
import bz2
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import urllib.request
import numpy as np

s.ensure_installed("astropy", "astropy_healpix", "matplotlib", "requests", "ttkthemes")
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy_healpix import HEALPix
import requests
from ttkthemes import ThemedTk

class SirilCatInstallerInterface:
    def __init__(self, root=None, cli_args=None):
        # If no CLI args, create a default namespace with defaults
        if cli_args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-lat", type=float, default=0.0)
            parser.add_argument("-min_elev", type=float, default=0.0)
            parser.add_argument("-type", type=str)
            cli_args = parser.parse_args([])

        self.cli_args = cli_args

        if root:
            self.root = root
            self.root.title(f"Siril Catalog Installer - v{VERSION}")
            self.root.resizable(False, False)
            self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        if not self.siril.connect():
            if root:
                self.siril.error_messagebox("Failed to connect to Siril")
            else:
                print("Failed to connect to Siril")
            return

        if not self.siril.cmd("requires", "1.3.6"):
            return

        if root:
            self.create_widgets()
            tksiril.match_theme_to_siril(self.root, self.siril)

        # Only apply changes if CLI arguments are non-default
        if cli_args and (cli_args.type):
            self.apply_changes(from_cli=True)

    def create_widgets(self):
        # Main frame with no padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Siril Catalog Installer",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Astrometry Catalog frame
        astrometry_frame = ttk.LabelFrame(main_frame, text="Astrometry Catalog", padding=10)
        astrometry_frame.pack(fill=tk.X, padx=5, pady=5)

        # Install button for Astrometry
        astrometry_install_btn = ttk.Button(
            astrometry_frame,
            text="Install",
            command=self.install_astrometry,
            style="TButton"
        )
        astrometry_install_btn.pack(pady=10)
        tksiril.create_tooltip(astrometry_install_btn, "Install or update the Astrometry catalog. This will "
                        "be installed to the Siril user data directory and set in Preferences -> Astrometry")

        # SPCC Catalog frame
        spcc_frame = ttk.LabelFrame(main_frame, text="SPCC Catalog", padding=10)
        spcc_frame.pack(fill=tk.X, padx=5, pady=10)

        # Observer Latitude entry
        latitude_frame = ttk.Frame(spcc_frame)
        latitude_frame.pack(fill=tk.X, pady=5)
        ttk.Label(latitude_frame, text="Observer Latitude:").pack(side=tk.LEFT)
        self.latitude_var = tk.DoubleVar()
        latitude_entry = ttk.Entry(
            latitude_frame,
            textvariable=self.latitude_var,
            width=10
        )
        latitude_entry.pack(side=tk.LEFT, padx=10)
        tksiril.create_tooltip(latitude_entry, "Enter your observatory latitude in degrees")

        # Minimum elevation entry
        elevation_frame = ttk.Frame(spcc_frame)
        elevation_frame.pack(fill=tk.X, pady=5)
        ttk.Label(elevation_frame, text="Minimum elevation:").pack(side=tk.LEFT)
        self.elevation_var = tk.DoubleVar()
        elevation_entry = ttk.Entry(
            elevation_frame,
            textvariable=self.elevation_var,
            width=10
        )
        elevation_entry.pack(side=tk.LEFT, padx=10)
        tksiril.create_tooltip(elevation_entry, "Enter minimum elevation in degrees")

        # Areas of Interest combobox
        area_frame = ttk.Frame(spcc_frame)
        area_frame.pack(fill=tk.X, pady=5)
        ttk.Label(area_frame, text="Areas of Interest:").pack(side=tk.LEFT)
        self.area_var = tk.StringVar()
        area_combo = ttk.Combobox(
            area_frame,
            textvariable=self.area_var,
            values=["Galaxy Season", "Magellanic Clouds", "Milky Way", "Orion to Taurus", "Summer Triangle"],
            state="readonly",
            width=20
        )
        self.area_var.set("Galaxy Season")
        area_combo.pack(side=tk.LEFT, padx=10)
        tksiril.create_tooltip(area_combo, "Select the area of interest for the SPCC catalog. This will install "
                               "only chunks covering the area of interest")

        # Selection Method combobox
        method_frame = ttk.Frame(spcc_frame)
        method_frame.pack(fill=tk.X, pady=5)
        ttk.Label(method_frame, text="Selection Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar()
        method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=["All", "Visible from Latitude", "Area of Interest"],
            state="readonly",
            width=20
        )
        self.method_var.set("All")
        method_combo.pack(side=tk.LEFT, padx=10)
        tksiril.create_tooltip(method_combo, "Select how to filter the SPCC catalog: 'All' will install "
                        "all chunks; 'Visible from Latiude' will install all chunks that are visible from the observer's "
                        "latitude above the given minimum elevation during the course of the year; 'Area "
                        "of Interest' will install chunks covering the specified area of interest")

         # Buttons
        spcc_button_frame = ttk.Frame(spcc_frame)
        spcc_button_frame.pack(fill=tk.X, pady=5)

        # Preview button for HEALpixel coverage
        healpix_btn = ttk.Button(
            spcc_button_frame,
            text="Preview coverage",
            command=self.preview_coverage,
            style="TButton"
        )
        healpix_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(healpix_btn, "Preview HEALpix coverage")

        # Install button for SPCC
        spcc_install_btn = ttk.Button(
            spcc_button_frame,
            text="Install",
            command=self.install_spcc,
            style="TButton"
        )
        spcc_install_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(spcc_install_btn, "Install or update the SPCC catalog with selected parameters")

    def get_pixels_from_ui(self):
        pixels = None
        method = self.method_var.get()
        if method == "Area of Interest":
            area = self.area_var.get()
            pixels = get_area_of_interest(area)
        elif method == "Visible from Latitude":
            lat = self.latitude_var.get()
            min_elev = self.elevation_var.get()
            pixels = get_visible_healpix(latitude=lat, min_elevation=min_elev)
        else: # method == "All":
            pixels = list(range(48))
        return pixels

    def download_with_progress(self, url, file_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = downloaded_size / total_size
                    self.siril.update_progress("Downloading...", progress)
        self.siril.reset_progress()

    def decompress_with_progress(self, bz2_path, decompressed_path):
        print(f"Decompressing {bz2_path} to {decompressed_path}...")

        # Get the total size of the compressed file for progress calculation
        total_size = os.path.getsize(bz2_path)
        processed_size = 0  # Tracks how much of the file has been read

        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(decompressed_path, 'wb') as f_out:
            while True:
                chunk = f_in.read(8192)  # Read in chunks
                if not chunk:  # Stop when no more data is available
                    break
                f_out.write(chunk)  # Write the chunk to the output file
                processed_size += 8192  # Update the processed size
                processed_size = min(processed_size, total_size)

                # Calculate progress percentage and display it
                progress = processed_size / total_size
                if progress > 0.99:
                    self.siril.update_progress("Decompressing... (nearly done!)", progress)
                else:
                    self.siril.update_progress("Decompressing...", progress)
        self.siril.reset_progress()

    def verify_sha256sum(self, bz2_path, sha256sum_path):
        # Read the expected SHA256 checksum from the .sha256sum file
        with open(sha256sum_path, 'r') as f:
            expected_checksum = f.read().split()[0]

        # Calculate the SHA256 checksum of the downloaded .bz2 file
        sha256_hash = hashlib.sha256()
        with open(bz2_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        actual_checksum = sha256_hash.hexdigest()

        # Verify the checksum
        if actual_checksum != expected_checksum:
            print(f"Checksum verification failed. Expected {expected_checksum}, got {actual_checksum}")
            return False
        else:
            print("Checksum verfication succeeded.")
            return True

    def install_astrometry(self):
        # URLs of the files to download
        catfile = f"siril_cat_healpix{ASTRO_INDEXLEVEL}_astro.dat.bz2"
        shasumfile = f"{catfile}.sha256sum"
        bz2_url = f"https://zenodo.org/records/{ASTRO_RECORD}/files/{catfile}"
        sha256sum_url = f"{bz2_url}.sha256sum"

        # Set target dir
        target_dir = self.siril.get_siril_userdatadir()

        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Download the .sha256sum file
        sha256sum_path = os.path.join(target_dir, shasumfile)
        print(f"Downloading {sha256sum_url} to {sha256sum_path}...")
        response = requests.get(sha256sum_url)
        with open(sha256sum_path, 'wb') as f:
            f.write(response.content)

        # Does the compressed archive already exist? If so, check the checksum
        # If it doesn't exist or the checksum is invalid, download again
        bz2_path = os.path.join(target_dir, catfile)
        if os.path.exists(bz2_path) and self.verify_sha256sum(bz2_path, sha256sum_path):
            print("Existing archive found with valid checksum...")
        else:
            # Download the .bz2 file with progress reporting
            print(f"Downloading {bz2_url} to {bz2_path}...")
            self.download_with_progress(bz2_url, bz2_path)
            if not self.verify_sha256sum(bz2_path, sha256sum_path):
                print("Checksum verification error, unable to proceed.")
                return

        # Determine the decompressed file path by removing the .bz2 extension
        decompressed_filename = os.path.basename(bz2_path).rsplit('.bz2', 1)[0]
        decompressed_path = os.path.join(target_dir, decompressed_filename)
        # Decompress the .bz2 file
        self.decompress_with_progress(bz2_path, decompressed_path)

        # Clean up: remove the compressed archive and checksum file
        print("Cleaning up...")
        os.remove(bz2_path)
        os.remove(sha256sum_path)

        print("Installation completed successfully.")

    def install_spcc(self):
        pixels = self.get_pixels_from_ui()
        print(f"Installing the following Level 1 HEALpixels: {pixels}")
        chunks = []
        error = 0
        for pixel in pixels:
            catfile = f"siril_cat{SPCC_CHUNKLEVEL}_healpix{SPCC_INDEXLEVEL}_xpsamp_{pixel}.dat.bz2"
            chunks.append(catfile)
            shasumfile = f"{catfile}.sha256sum"
            bz2_url = f"https://zenodo.org/records/{SPCC_RECORD}/files/{catfile}"
            sha256sum_url = f"{bz2_url}.sha256sum"

            # Set target dir
            target_dir = os.path.join(self.siril.get_siril_userdatadir(), f"siril_cat{SPCC_CHUNKLEVEL}_healpix{SPCC_INDEXLEVEL}_xpsamp")

            # Ensure the target directory exists
            os.makedirs(target_dir, exist_ok=True)

            # Download the .sha256sum file
            sha256sum_path = os.path.join(target_dir, shasumfile)
            print(f"Downloading {sha256sum_url} to {sha256sum_path}...")
            response = requests.get(sha256sum_url)
            with open(sha256sum_path, 'wb') as f:
                f.write(response.content)

            # Does the compressed archive already exist? If so, check the checksum
            # If it doesn't exist or the checksum is invalid, download again
            bz2_path = os.path.join(target_dir, catfile)
            if os.path.exists(bz2_path) and self.verify_sha256sum(bz2_path, sha256sum_path):
                print("Existing archive found with valid checksum...")
            else:
                # Download the .bz2 file with progress reporting
                print(f"Downloading {bz2_url} to {bz2_path}...")
                self.download_with_progress(bz2_url, bz2_path)
                if not self.verify_sha256sum(bz2_path, sha256sum_path):
                    print(f"Checksum verification error for {bz2_path}, skipping HEALpixel {pixel}.", file=sys.stderr)
                    error = 1
                    continue

            # Determine the decompressed file path by removing the .bz2 extension
            decompressed_filename = os.path.basename(bz2_path).rsplit('.bz2', 1)[0]
            decompressed_path = os.path.join(target_dir, decompressed_filename)
            # Decompress the .bz2 file
            self.decompress_with_progress(bz2_path, decompressed_path)

            # Clean up: remove the compressed archive and checksum file
            print("Cleaning up...")
            os.remove(bz2_path)
            os.remove(sha256sum_path)
            print(f"{decompressed_path} installed successfully.")

        if not error:
            print("Installation complete, all files installed successfully.")
        else:
            print("Installation complete but not all files installed successfully. Please review the error messages", file=sys.stderr)


        return

    def preview_coverage(self):
        pixels = self.get_pixels_from_ui()
        plot_visible_pixels(pixels)
        return

def calculate_colatitude(latitude_deg, elevation_deg):
    """
    Compute the most extreme celestial colatitude observable above the given minimum elevation
    from an observer's terrestrial latitude.

    The Earth's axial tilt is taken into account. For observers near the equator,
    this function correctly handles cases where regions near both poles may not meet
    the minimum elevation requirement.

    Parameters:
    -----------
    latitude_deg : float
        Observer's latitude in degrees
    elevation_deg : float
        Minimum elevation angle in degrees

    Returns:
    --------
    colatitude_rad : float
        The colatitude in radians from the primary pole
    exclusion_colatitude_rad_north : float or None
        The colatitude in radians from the north pole for regions that never
        reach minimum elevation. None if all northern regions can potentially
        reach minimum elevation.
    exclusion_colatitude_rad_south : float or None
        The colatitude in radians from the south pole for regions that never
        reach minimum elevation. None if all southern regions can potentially
        reach minimum elevation.
    """
    # Earth's axial tilt in radians
    epsilon = math.radians(23.44)

    # Convert latitude and elevation from degrees to radians
    latitude_rad = math.radians(latitude_deg)
    elevation_rad = math.radians(elevation_deg)

    # Calculate maximum declination visible for both hemispheres
    sin_delta_max_north = math.sin(latitude_rad) * math.sin(elevation_rad) + \
                         math.cos(latitude_rad) * math.cos(elevation_rad) * math.cos(epsilon)
    sin_delta_max_south = math.sin(-latitude_rad) * math.sin(elevation_rad) + \
                         math.cos(-latitude_rad) * math.cos(elevation_rad) * math.cos(epsilon)

    # Convert to declination angles (bounded to valid range)
    delta_max_north = math.asin(min(1, max(-1, sin_delta_max_north)))
    delta_max_south = math.asin(min(1, max(-1, sin_delta_max_south)))

    # For northern hemisphere observers
    if latitude_deg >= 0:
        # Colatitude is the angular distance from north pole to southernmost visible point
        colatitude_rad = math.pi/2 + abs(delta_max_south)
    # For southern hemisphere observers
    else:
        # Colatitude is the angular distance from south pole to northernmost visible point
        colatitude_rad = math.pi/2 + abs(delta_max_north)

    # Check both poles for observers near the equator or when either pole might be invisible
    exclusion_colatitude_rad_north = None
    exclusion_colatitude_rad_south = None

    # Check northern regions
    if latitude_deg < elevation_deg:
        exclusion_colatitude_rad_north = math.pi - delta_max_north

    # Check southern regions
    if -latitude_deg < elevation_deg:
        exclusion_colatitude_rad_south = math.pi - abs(delta_max_south)

    return colatitude_rad, exclusion_colatitude_rad_north, exclusion_colatitude_rad_south

def get_visible_healpix(latitude, min_elevation):
    """
    Compute HEALPix level 1 pixel numbers visible above minimum elevation,
    accounting for regions near either pole that may never reach the minimum elevation.

    Parameters:
    -----------
    latitude : float
        Observer's latitude in degrees
    min_elevation : float
        Minimum elevation angle in degrees

    Returns:
    --------
    pixels : list
        List of unique NEST HEALPix pixel numbers for level 1.
    """
    colatitude, excl_north, excl_south = calculate_colatitude(latitude, min_elevation)
    nside = 2
    healpix = HEALPix(nside=nside, order='nested', frame='icrs')

    # Convert colatitude to radius (in degrees)
    colatitude_deg = u.Quantity(colatitude, u.radian).to(u.deg)

    # Determine which pole to start from based on observer's hemisphere
    if latitude >= 0:
        primary_lon = 0 * u.deg
        primary_lat = 90 * u.deg
        primary_excl = excl_north
    else:
        primary_lon = 0 * u.deg
        primary_lat = -90 * u.deg
        primary_excl = excl_south

    # Get visible pixels from primary search
    visible_pixels = healpix.cone_search_lonlat(primary_lon, primary_lat, colatitude_deg)

    # Check northern exclusion if it exists
    if excl_north is not None:
        excl_north_deg = u.Quantity(excl_north, u.radian).to(u.deg)
        north_visible = healpix.cone_search_lonlat(0 * u.deg, 90 * u.deg, excl_north_deg)
        visible_pixels = np.intersect1d(visible_pixels, north_visible)

    # Check southern exclusion if it exists
    if excl_south is not None:
        excl_south_deg = u.Quantity(excl_south, u.radian).to(u.deg)
        south_visible = healpix.cone_search_lonlat(0 * u.deg, -90 * u.deg, excl_south_deg)
        visible_pixels = np.intersect1d(visible_pixels, south_visible)

    return visible_pixels.tolist()
def get_area_of_interest(area):
    if area == "Galaxy Season":
        return [5,8,10,24,25,26,27]
    elif area == "Magellanic Clouds":
        return [32,33,36,38]
    elif area == "Summer Triangle":
        return [9,12,13,14,15,29,31]
    elif area == "Milky Way":
        return [0,1,2,3,12,13,16,17,18,19,28,29,30,36,37,40,41,42,43,45,46,47]
    elif area == "Orion to Taurus":
        return [0,1,6,20,21,22,23]
    else:
        return []

def plot_visible_pixels(visible_pixels, nside=2):
    """
    Create a visualization of the visible pixels using a Mollweide projection
    by running the plotting logic in a subprocess (to avoid a clash with the
    matplotlib and TKinter main loops)

    Parameters:
    -----------
    visible_pixels : np.ndarray
        Array of visible HEALPix pixel indices
    nside : int, optional
        HEALPix nside parameter.
    """
    # Convert visible_pixels to a string representation
    visible_pixels_str = ', '.join(map(str, visible_pixels))
    print(visible_pixels_str)

    # Define the script to be executed in the subprocess
    script = f"""
import numpy as np
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def spherical_to_cartesian(lon, lat, radius = 1):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def get_constellation_data():
    return [
        (42.3187, 2.0649, 41.4019, 1.6134),
        (41.4019, 1.6134, 48.6155, 1.6333),
        (41.4019, 1.6134, 35.6151, 1.1623),
        (35.6151, 1.1623, 30.848, 0.6555),
        (30.848, 0.6555, 29.0833, 0.1394),
        (35.6151, 1.1623, 38.4856, 0.9454),
        (38.4856, 0.9454, 41.0639, 0.83),
        (24.2648, 0.7888, 29.3011, 0.6421),
        (29.3011, 0.6421, 30.848, 0.6555),
        (30.848, 0.6555, 38.6689, 0.285),
        (38.6689, 0.285, 43.264, 23.6357),
        (43.264, 23.6357, 44.3355, 23.6735),
        (44.3355, 23.6735, 46.4497, 23.6261),
        (43.264, 23.6357, 42.3187, 23.0318),
        (-37.1334, 10.945, -31.0658, 10.4523),
        (-31.0658, 10.4523, -27.7655, 9.7368),
        (-27.7655, 9.7368, -35.9474, 9.487),
        (-77.5155, 16.7178, -78.8848, 16.5573),
        (-78.8848, 16.5573, -78.6843, 16.3388),
        (-78.6843, 16.3388, -79.0338, 14.7976),
        (-21.1651, 23.157, -15.8194, 22.9107),
        (-15.8194, 22.9107, -13.5848, 22.8266),
        (-13.5848, 22.8266, -7.5688, 22.8767),
        (-7.5688, 22.8767, -6.0332, 23.2384),
        (-6.0332, 23.2384, -0.0172, 22.4806),
        (-0.0172, 22.4806, -1.3808, 22.3606),
        (-1.3808, 22.3606, -0.3151, 22.0959),
        (-0.3151, 22.0959, -5.5691, 21.526),
        (-5.5691, 21.526, -8.984, 20.8771),
        (-8.984, 20.8771, -9.4825, 20.7945),
        (-0.3151, 22.0959, -7.7808, 22.2804),
        (-7.7808, 22.2804, -13.8656, 22.1074),
        (15.0688, 18.9932, 13.8484, 19.0902),
        (13.8484, 19.0902, 10.5997, 19.7712),
        (10.5997, 19.7712, 8.8694, 19.8461),
        (8.8694, 19.8461, 6.3999, 19.9217),
        (6.3999, 19.9217, 1.0027, 19.8744),
        (1.0027, 19.8744, -1.2834, 19.6116),
        (-1.2834, 19.6116, -4.8644, 19.1039),
        (-4.8644, 19.1039, -5.7353, 19.0279),
        (-0.8193, 20.1884, 1.0027, 19.8744),
        (1.0027, 19.8744, 3.0997, 19.4252),
        (-60.682, 17.5184, -56.3676, 17.4229),
        (-56.3676, 17.4229, -55.5139, 17.4217),
        (-55.5139, 17.4217, -49.8645, 17.5306),
        (-49.8645, 17.5306, -53.1476, 16.9928),
        (-53.1476, 16.9928, -55.9837, 16.9767),
        (-55.9837, 16.9767, -59.0318, 16.8293),
        (-55.9837, 16.9767, -56.3676, 17.4229),
        (19.7155, 3.194, 21.3312, 2.9866),
        (21.3312, 2.9866, 27.2499, 2.8327),
        (27.2499, 2.8327, 23.4512, 2.1196),
        (23.4512, 2.1196, 20.7984, 1.9106),
        (20.7984, 1.9106, 19.2858, 1.8923),
        (54.282, 5.9924, 44.9314, 5.9924),
        (44.9314, 5.9924, 37.2021, 5.995),
        (37.2021, 5.995, 33.1513, 4.95),
        (33.1513, 4.95, 41.2358, 5.1085),
        (41.2358, 5.1085, 41.0639, 5.0413),
        (41.0639, 5.0413, 43.8141, 5.0329),
        (43.8141, 5.0329, 41.2358, 5.1085),
        (41.2358, 5.1085, 45.9856, 5.2777),
        (45.9856, 5.2777, 44.9314, 5.9924),
        (51.7839, 14.2246, 46.083, 14.2728),
        (46.083, 14.2728, 38.3022, 14.5344),
        (38.3022, 14.5344, 40.3821, 15.0321),
        (40.3821, 15.0321, 33.3003, 15.2582),
        (33.3003, 15.2582, 37.3683, 15.4076),
        (33.3003, 15.2582, 27.0665, 14.7495),
        (27.0665, 14.7495, 19.1654, 14.2613),
        (19.1654, 14.2613, 30.3668, 14.5306),
        (30.3668, 14.5306, 38.3022, 14.5344),
        (15.785, 13.8243, 17.4523, 13.7877),
        (17.4523, 13.7877, 18.3805, 13.911),
        (18.3805, 13.911, 19.1654, 14.2613),
        (19.1654, 14.2613, 16.4152, 14.6788),
        (16.4152, 14.6788, 13.7166, 14.6857),
        (-41.8488, 4.6761, -37.1334, 4.7005),
        (53.7492, 4.9546, 60.4356, 5.0565),
        (60.4356, 5.0565, 66.3313, 4.9007),
        (66.3313, 4.9007, 71.3161, 3.8396),
        (71.3161, 3.8396, 59.9314, 3.4843),
        (11.8488, 8.9744, 18.1513, 8.7445),
        (18.1513, 8.7445, 21.4687, 8.7212),
        (21.4687, 8.7212, 28.751, 8.7777),
        (18.1513, 8.7445, 9.1845, 8.275),
        (38.3022, 12.9332, 41.3504, 12.5623),
        (-12.0321, 6.9026, -15.636, 7.0623),
        (-15.636, 7.0623, -17.0512, 6.9355),
        (-17.0512, 6.9355, -16.7017, 6.7521),
        (-16.7017, 6.7521, -17.9508, 6.3782),
        (-16.7017, 6.7521, -23.835, 7.0501),
        (-23.835, 7.0501, -26.3847, 7.1394),
        (-26.3847, 7.1394, -27.9317, 7.0283),
        (-27.9317, 7.0283, -28.9687, 6.9771),
        (-28.9687, 6.9771, -30.0516, 6.3384),
        (-26.3847, 7.1394, -26.7686, 7.2468),
        (-26.7686, 7.2468, -24.9523, 7.3117),
        (-26.7686, 7.2468, -29.3011, 7.4019),
        (5.2139, 7.6551, 8.285, 7.4523),
        (-12.5019, 20.2938, -14.7651, 20.3499),
        (-14.7651, 20.3499, -25.2674, 20.7682),
        (-25.2674, 20.7682, -26.9176, 20.8633),
        (-26.9176, 20.8633, -24.9981, 21.1188),
        (-24.9981, 21.1188, -22.4026, 21.4443),
        (-22.4026, 21.4443, -19.4519, 21.6177),
        (-19.4519, 21.6177, -16.1173, 21.7839),
        (-16.1173, 21.7839, -16.6502, 21.6677),
        (-16.6502, 21.6677, -16.8335, 21.3706),
        (-16.8335, 21.3706, -17.2174, 21.099),
        (-17.2174, 21.099, -14.7651, 20.3499),
        (-52.6835, 6.3988, -52.9642, 6.5829),
        (-52.9642, 6.5829, -53.6174, 6.8304),
        (-53.6174, 6.8304, -52.9642, 7.9462),
        (-52.9642, 7.9462, -59.5017, 8.3751),
        (-59.5017, 8.3751, -59.2668, 9.2846),
        (-59.2668, 9.2846, -61.3179, 10.2846),
        (-61.3179, 10.2846, -61.6846, 10.5333),
        (-61.6846, 10.5333, -64.3833, 10.7155),
        (-64.3833, 10.7155, -70.0326, 10.2288),
        (-70.0326, 10.2288, -69.7175, 9.22),
        (-69.7175, 9.22, -65.0651, 9.785),
        (63.6671, 1.9068, 60.2351, 1.4301),
        (60.2351, 1.4301, 60.7163, 0.945),
        (60.7163, 0.945, 56.5337, 0.6746),
        (56.5337, 0.6746, 59.135, 0.1528),
        (60.7163, 0.945, 62.9165, 0.55),
        (62.9165, 0.55, 67.4028, 2.4843),
        (56.5337, 0.6746, 53.881, 0.6161),
        (53.881, 0.6161, 55.1357, 1.1849),
        (-36.6979, 13.3434, -39.4023, 13.5172),
        (-39.4023, 13.5172, -34.452, 13.8239),
        (-34.452, 13.8239, -36.3656, 14.1112),
        (-36.3656, 14.1112, -37.884, 14.3423),
        (-37.884, 14.3423, -37.7808, 14.6994),
        (-37.7808, 14.6994, -35.1681, 14.7273),
        (-37.884, 14.3423, -42.1525, 14.5917),
        (-42.1525, 14.5917, -42.1009, 14.9863),
        (-36.3656, 14.1112, -41.6827, 13.8251),
        (-41.6827, 13.8251, -42.4676, 13.8266),
        (-42.4676, 13.8266, -47.2805, 13.9255),
        (-47.2805, 13.9255, -53.4512, 13.6643),
        (-53.4512, 13.6643, -48.9478, 12.6918),
        (-48.9478, 12.6918, -40.1643, 12.8904),
        (-40.1643, 12.8904, -41.6827, 13.8251),
        (-48.9478, 12.6918, -50.214, 12.4672),
        (-50.214, 12.4672, -50.7182, 12.1394),
        (-50.7182, 12.1394, -52.3683, 12.1941),
        (-52.3683, 12.1941, -54.4826, 11.3499),
        (-54.4826, 11.3499, -61.169, 11.775),
        (-61.169, 11.775, -63.0139, 11.5963),
        (-60.3668, 14.0634, -53.4512, 13.6643),
        (-53.4512, 13.6643, -60.8309, 14.6601),
        (62.9853, 20.4928, 61.8336, 20.7544),
        (61.8336, 20.7544, 62.5842, 21.3094),
        (62.5842, 21.3094, 70.5483, 21.4779),
        (70.5483, 21.4779, 77.6186, 23.6555),
        (77.6186, 23.6555, 66.1995, 22.8278),
        (66.1995, 22.8278, 58.2011, 22.1807),
        (58.2011, 22.1807, 62.5842, 21.3094),
        (58.2011, 22.1807, 57.0322, 22.2506),
        (57.0322, 22.2506, 58.4016, 22.4863),
        (3.2315, 2.7215, 4.0852, 3.0378),
        (4.0852, 3.0378, 8.898, 2.995),
        (8.898, 2.995, 10.1012, 2.7491),
        (10.1012, 2.7491, 8.4511, 2.4694),
        (8.4511, 2.4694, 5.5806, 2.5978),
        (5.5806, 2.5978, 3.2315, 2.7215),
        (3.2315, 2.7215, 0.3151, 2.6578),
        (0.3151, 2.6578, -2.9679, 2.3224),
        (-2.9679, 2.3224, -10.3362, 1.8579),
        (-10.3362, 1.8579, -8.1818, 1.3999),
        (-8.1818, 1.3999, -10.1643, 1.1429),
        (-10.1643, 1.1429, -15.934, 1.7338),
        (-15.934, 1.7338, -10.3362, 1.8579),
        (-15.934, 1.7338, -21.0677, 2.0),
        (-21.0677, 2.0, -13.8484, 2.7349),
        (-10.1643, 1.1429, -17.9851, 0.7261),
        (-17.9851, 0.7261, -8.8178, 0.3239),
        (-76.9139, 8.309, -78.5984, 10.5909),
        (-78.5984, 10.5909, -79.2974, 12.3056),
        (-79.2974, 12.3056, -80.5349, 10.7628),
        (-80.5349, 10.7628, -77.4811, 8.3438),
        (-77.4811, 8.3438, -76.9139, 8.309),
        (-59.3183, 15.3893, -64.9677, 14.7082),
        (-64.9677, 14.7082, -58.8027, 15.2915),
        (-35.4661, 5.5199, -34.0681, 5.6604),
        (-34.0681, 5.6604, -35.764, 5.8495),
        (-35.764, 5.8495, -35.2827, 5.9588),
        (-35.2827, 5.9588, -35.1338, 6.2754),
        (-35.1338, 6.2754, -33.4321, 6.3682),
        (-35.764, 5.8495, -42.7999, 5.9855),
        (17.5153, 13.1662, 27.8687, 13.1979),
        (27.8687, 13.1979, 28.264, 12.4488),
        (-37.099, 18.9783, -37.0475, 19.1066),
        (-37.0475, 19.1066, -37.9012, 19.1578),
        (-37.9012, 19.1578, -39.3336, 19.1673),
        (-39.3336, 19.1673, -40.4852, 19.1387),
        (-40.4852, 19.1387, -42.0838, 19.0516),
        (-42.0838, 19.0516, -42.7025, 18.9378),
        (-42.7025, 18.9378, -43.4359, 18.8262),
        (31.3523, 15.5489, 29.1005, 15.4637),
        (29.1005, 15.4637, 26.6998, 15.578),
        (26.6998, 15.578, 26.2816, 15.7124),
        (26.2816, 15.7124, 26.0639, 15.8266),
        (26.0639, 15.8266, 26.866, 15.9595),
        (26.866, 15.9595, 29.8511, 16.0237),
        (-24.7174, 12.1398, -22.6146, 12.1685),
        (-22.6146, 12.1685, -17.5325, 12.2632),
        (-17.5325, 12.2632, -16.5012, 12.4977),
        (-16.5012, 12.4977, -23.3824, 12.5726),
        (-23.3824, 12.5726, -22.6146, 12.1685),
        (-17.1486, 11.9332, -18.3518, 11.746),
        (-18.3518, 11.746, -17.6815, 11.4145),
        (-17.6815, 11.4145, -14.7651, 11.3224),
        (-14.7651, 11.3224, -10.8518, 11.4099),
        (-10.8518, 11.4099, -9.7976, 11.6112),
        (-17.6815, 11.4145, -22.8152, 11.1941),
        (-22.8152, 11.1941, -18.2831, 10.9962),
        (-18.2831, 10.9962, -14.7651, 11.3224),
        (-63.0827, 12.4435, -57.101, 12.5195),
        (-59.685, 12.7949, -58.7339, 12.2521),
        (53.3653, 19.285, 51.7152, 19.4951),
        (51.7152, 19.4951, 50.214, 19.6074),
        (50.214, 19.6074, 45.1147, 19.7495),
        (45.1147, 19.7495, 40.2503, 20.3706),
        (40.2503, 20.3706, 33.9649, 20.7701),
        (33.9649, 20.7701, 30.2178, 21.2155),
        (30.2178, 21.2155, 28.7338, 21.7357),
        (30.2178, 21.2155, 34.8817, 21.2984),
        (34.8817, 21.2984, 38.0329, 21.246),
        (38.0329, 21.246, 39.3851, 21.29),
        (39.3851, 21.29, 45.5845, 21.5661),
        (45.5845, 21.5661, 49.2973, 21.78),
        (45.2694, 20.6907, 40.2503, 20.3706),
        (40.2503, 20.3706, 35.0822, 19.9382),
        (35.0822, 19.9382, 32.8992, 19.8427),
        (32.8992, 19.8427, 27.9489, 19.5115),
        (11.2987, 20.5535, 14.5818, 20.6257),
        (14.5818, 20.6257, 15.0688, 20.7243),
        (15.0688, 20.7243, 16.1173, 20.7774),
        (16.1173, 20.7774, 15.8996, 20.6605),
        (15.8996, 20.6605, 14.5818, 20.6257),
        (-51.486, 4.2674, -55.0326, 4.5669),
        (-55.0326, 4.5669, -57.4677, 5.0917),
        (-57.4677, 5.0917, -62.481, 5.5607),
        (-62.481, 5.5607, -65.7354, 5.746),
        (-62.481, 5.5607, -63.0827, 5.9018),
        (55.1816, 17.5359, 52.2996, 17.5073),
        (52.2996, 17.5073, 51.486, 17.9435),
        (51.486, 17.9435, 56.8661, 17.8923),
        (56.8661, 17.8923, 55.1816, 17.5359),
        (56.8661, 17.8923, 67.6491, 19.209),
        (67.6491, 19.209, 70.2675, 19.8029),
        (70.2675, 19.8029, 72.7141, 18.3507),
        (72.7141, 18.3507, 65.7011, 17.146),
        (65.7011, 17.146, 61.5013, 16.3996),
        (61.5013, 16.3996, 58.5506, 16.031),
        (58.5506, 16.031, 58.9516, 15.4156),
        (58.9516, 15.4156, 64.3661, 14.0726),
        (64.3661, 14.0726, 69.7805, 12.5577),
        (69.7805, 12.5577, 69.3164, 11.5233),
        (9.9981, 21.2411, 6.801, 21.3816),
        (6.801, 21.3816, 5.2311, 21.2632),
        (5.2311, 21.2632, 10.1184, 21.1723),
        (-8.7491, 5.152, -5.0821, 5.1306),
        (-5.0821, 5.1306, -5.4488, 4.8816),
        (-5.4488, 4.8816, -3.2487, 4.7582),
        (-3.2487, 4.7582, -3.3518, 4.6051),
        (-3.3518, 4.6051, -6.8354, 4.1979),
        (-6.8354, 4.1979, -13.4989, 3.9672),
        (-13.4989, 3.9672, -12.1009, 3.7689),
        (-12.1009, 3.7689, -9.7517, 3.7204),
        (-9.7517, 3.7204, -9.4481, 3.5489),
        (-9.4481, 3.5489, -8.8808, 2.9404),
        (-8.8808, 2.9404, -18.5638, 2.7517),
        (-18.5638, 2.7517, -20.9989, 2.8499),
        (-20.9989, 2.8499, -23.6173, 3.0401),
        (-23.6173, 3.0401, -21.7495, 3.3251),
        (-21.7495, 3.3251, -21.6177, 3.5627),
        (-21.6177, 3.5627, -23.2334, 3.7804),
        (-23.2334, 3.7804, -24.0012, 3.9985),
        (-24.0012, 3.9985, -29.7652, 4.5585),
        (-29.7652, 4.5585, -30.5501, 4.5921),
        (-30.5501, 4.5921, -34.0165, 4.4007),
        (-34.0165, 4.4007, -33.7816, 4.2983),
        (-33.7816, 4.2983, -36.1995, 3.8239),
        (-36.1995, 3.8239, -37.6147, 3.8102),
        (-37.6147, 3.8102, -43.0692, 3.3323),
        (-43.0692, 3.3323, -40.3019, 2.971),
        (-40.3019, 2.971, -39.8492, 2.6776),
        (-39.8492, 2.6776, -42.8859, 2.6635),
        (-42.8859, 2.6635, -47.6987, 2.4496),
        (-47.6987, 2.4496, -51.4974, 2.275),
        (-51.4974, 2.275, -51.6006, 1.9324),
        (-51.6006, 1.9324, -57.2328, 1.6283),
        (-28.9859, 3.2009, -32.4008, 2.8178),
        (-32.4008, 2.8178, -29.2839, 2.0745),
        (22.5001, 6.2479, 22.5001, 6.3827),
        (22.5001, 6.3827, 25.1185, 6.7323),
        (25.1185, 6.7323, 30.235, 7.1857),
        (30.235, 7.1857, 31.782, 7.4851),
        (31.782, 7.4851, 31.8851, 7.5768),
        (31.8851, 7.5768, 28.8828, 7.7216),
        (28.8828, 7.7216, 28.0176, 7.7552),
        (28.0176, 7.7552, 24.3851, 7.7407),
        (24.3851, 7.7407, 21.9672, 7.335),
        (21.9672, 7.335, 20.5692, 7.0684),
        (20.5692, 7.0684, 16.3809, 6.6284),
        (16.3809, 6.6284, 20.2025, 6.4828),
        (20.2025, 6.4828, 22.5001, 6.3827),
        (16.3809, 6.6284, 13.2181, 6.7326),
        (21.9672, 7.335, 16.5356, 7.3018),
        (-37.3511, 21.8988, -39.5341, 22.1017),
        (-39.5341, 22.1017, -41.3332, 22.2602),
        (-41.3332, 22.2602, -43.4818, 22.4878),
        (-43.4818, 22.4878, -46.8851, 22.7113),
        (-46.8851, 22.7113, -51.3141, 22.8091),
        (-51.3141, 22.8091, -52.7522, 23.0146),
        (-46.9482, 22.1372, -46.8851, 22.7113),
        (-46.8851, 22.7113, -45.235, 23.1727),
        (-45.235, 23.1727, -43.5161, 23.1143),
        (44.9314, 16.146, 42.4333, 16.5684),
        (42.4333, 16.5684, 38.9153, 16.7151),
        (38.9153, 16.7151, 31.5986, 16.688),
        (31.5986, 16.688, 21.4859, 16.5035),
        (21.4859, 16.5035, 19.1482, 16.3648),
        (21.4859, 16.5035, 14.3812, 17.2437),
        (46.0028, 17.6578, 37.248, 17.9374),
        (37.248, 17.9374, 37.1334, 17.3946),
        (37.1334, 17.3946, 36.8011, 17.2506),
        (36.8011, 17.2506, 38.9153, 16.7151),
        (36.8011, 17.2506, 30.9168, 17.0046),
        (30.9168, 17.0046, 31.5986, 16.688),
        (30.9168, 17.0046, 24.832, 17.2498),
        (24.832, 17.2498, 26.0982, 17.5115),
        (26.0982, 17.5115, 27.714, 17.7743),
        (27.714, 17.7743, 29.2323, 17.9626),
        (29.2323, 17.9626, 28.751, 18.1257),
        (28.751, 18.1257, 30.1834, 17.9748),
        (28.751, 18.1257, 21.7667, 18.395),
        (21.7667, 18.395, 20.7984, 18.146),
        (21.7667, 18.395, 20.5348, 18.7609),
        (20.5348, 18.7609, 18.1685, 18.7835),
        (-42.2843, 4.2334, -50.7984, 2.7089),
        (-50.7984, 2.7089, -52.5345, 2.6234),
        (-52.5345, 2.6234, -54.5513, 2.6776),
        (-54.5513, 2.6776, -59.7309, 3.06),
        (-59.7309, 3.06, -64.0681, 2.9801),
        (-26.6655, 14.1062, -23.1647, 13.3152),
        (-23.1647, 13.3152, -33.9019, 11.8816),
        (-33.9019, 11.8816, -31.8507, 11.5501),
        (-31.8507, 11.5501, -16.1861, 10.8266),
        (-16.1861, 10.8266, -16.8335, 10.4343),
        (-16.8335, 10.4343, -12.3472, 10.1761),
        (-12.3472, 10.1761, -13.052, 10.0848),
        (-13.052, 10.0848, -14.8339, 9.8579),
        (-14.8339, 9.8579, -8.6517, 9.4595),
        (-8.6517, 9.4595, -1.1345, 9.6639),
        (-1.1345, 9.6639, 2.2976, 9.2395),
        (2.2976, 9.2395, 5.9358, 8.9232),
        (5.9358, 8.9232, 3.3862, 8.72),
        (3.3862, 8.72, 3.3346, 8.6456),
        (3.3346, 8.6456, 5.7009, 8.6272),
        (5.7009, 8.6272, 6.4171, 8.7796),
        (6.4171, 8.7796, 5.9358, 8.9232),
        (-74.2324, 3.7873, -77.2519, 0.429),
        (-77.2519, 0.429, -68.2679, 2.6593),
        (-68.2679, 2.6593, -68.6518, 2.3621),
        (-68.6518, 2.3621, -67.6319, 1.9156),
        (-67.6319, 1.9156, -61.5643, 1.9794),
        (-47.2805, 20.6261, -53.434, 21.3312),
        (-53.434, 21.3312, -54.981, 21.9649),
        (-53.434, 21.3312, -58.4474, 20.9133),
        (37.735, 22.2663, 43.1151, 22.5077),
        (43.1151, 22.5077, 47.6987, 22.4924),
        (47.6987, 22.4924, 49.4692, 22.4084),
        (49.4692, 22.4084, 52.2194, 22.3927),
        (52.2194, 22.3927, 50.2656, 22.5211),
        (50.2656, 22.5211, 47.6987, 22.4924),
        (26.167, 9.4106, 22.9641, 9.5283),
        (22.9641, 9.5283, 23.7663, 9.764),
        (23.7663, 9.764, 26.0008, 9.8793),
        (26.0008, 9.8793, 23.4168, 10.2777),
        (23.4168, 10.2777, 19.8358, 10.3327),
        (19.8358, 10.3327, 16.7476, 10.1223),
        (16.7476, 10.1223, 11.9691, 10.1394),
        (11.9691, 10.1394, 9.8835, 9.6857),
        (11.9691, 10.1394, 15.4183, 11.2372),
        (15.4183, 11.2372, 14.5646, 11.8178),
        (14.5646, 11.8178, 20.5176, 11.2349),
        (20.5176, 11.2349, 19.8358, 10.3327),
        (15.4183, 11.2372, 10.5195, 11.3984),
        (10.5195, 11.3984, 6.0161, 11.3522),
        (34.1999, 10.8885, 36.6979, 10.4645),
        (36.6979, 10.4645, 35.2312, 10.1238),
        (-11.866, 5.2051, -12.9317, 5.2204),
        (-12.9317, 5.2204, -13.1666, 5.3262),
        (-13.1666, 5.3262, -16.1975, 5.2154),
        (-16.1975, 5.2154, -17.819, 5.5455),
        (-17.819, 5.5455, -14.8167, 5.7823),
        (-14.8167, 5.7823, -14.4844, 5.8266),
        (-14.4844, 5.8266, -14.1692, 5.94),
        (-17.819, 5.5455, -20.7525, 5.4706),
        (-20.8671, 5.8549, -22.4313, 5.741),
        (-22.4313, 5.741, -20.7525, 5.4706),
        (-20.7525, 5.4706, -21.2338, 5.3407),
        (-21.2338, 5.3407, -22.3683, 5.0909),
        (-14.2666, 15.9695, -16.7189, 15.8973),
        (-16.7189, 15.8973, -15.6647, 15.7346),
        (-15.6647, 15.7346, -14.7823, 15.5921),
        (-14.7823, 15.5921, -9.3679, 15.2835),
        (-9.3679, 15.2835, -16.0314, 14.8476),
        (-16.0314, 14.8476, -25.2674, 15.0676),
        (-25.2674, 15.0676, -28.1322, 15.6165),
        (-28.1322, 15.6165, -29.7652, 15.644),
        (-25.2674, 15.0676, -9.3679, 15.2835),
        (-36.8011, 16.11, -38.3824, 16.0016),
        (-38.3824, 16.0016, -41.167, 15.5856),
        (-41.167, 15.5856, -44.685, 15.3778),
        (-44.685, 15.3778, -48.7358, 15.199),
        (-48.7358, 15.199, -52.0819, 15.2044),
        (-52.0819, 15.2044, -47.3836, 14.699),
        (-47.3836, 14.699, -43.1323, 14.9756),
        (-43.1323, 14.9756, -40.6342, 15.356),
        (-40.6342, 15.356, -36.251, 15.3633),
        (-36.251, 15.3633, -33.6154, 15.8495),
        (-41.167, 15.5856, -40.6342, 15.356),
        (34.3832, 9.3507, 36.8011, 9.314),
        (36.8011, 9.314, 41.7686, 9.0107),
        (41.7686, 9.0107, 43.1838, 8.3805),
        (43.1838, 8.3805, 49.1999, 7.445),
        (49.1999, 7.445, 58.4188, 6.9546),
        (58.4188, 6.9546, 58.9975, 6.3266),
        (43.9344, 18.9221, 39.133, 19.2296),
        (39.133, 19.2296, 38.1361, 19.2728),
        (38.1361, 19.2728, 38.7835, 18.6154),
        (38.7835, 18.6154, 39.6659, 18.7388),
        (38.7835, 18.6154, 37.5975, 18.746),
        (37.5975, 18.746, 33.3519, 18.8343),
        (33.3519, 18.8343, 32.6815, 18.9821),
        (32.6815, 18.9821, 36.9672, 18.8954),
        (36.9672, 18.8954, 37.5975, 18.746),
        (38.7835, 18.6154, 36.0505, 18.3312),
        (-74.7481, 6.1704, -76.3352, 5.531),
        (-76.3352, 5.531, -74.9314, 4.9194),
        (-74.9314, 4.9194, -71.2989, 5.0451),
        (-40.8003, 21.3461, -32.1659, 21.2988),
        (-32.1659, 21.2988, -32.2518, 21.0211),
        (-32.2518, 21.0211, -33.7644, 20.8327),
        (-2.9851, 8.1429, -9.5512, 7.6872),
        (-9.5512, 7.6872, -0.4813, 7.1979),
        (-0.4813, 7.1979, -7.0187, 6.4802),
        (-7.0187, 6.4802, -6.2682, 6.2471),
        (-6.2682, 6.2471, 4.5837, 6.3961),
        (4.5837, 6.3961, 7.3167, 6.5481),
        (-71.5338, 13.0378, -69.1331, 12.6196),
        (-69.1331, 12.6196, -72.1182, 12.5413),
        (-68.1018, 12.7712, -69.1331, 12.6196),
        (-69.1331, 12.6196, -67.9528, 12.2926),
        (-67.9528, 12.2926, -66.8011, 11.8041),
        (-47.5498, 16.4527, -50.0651, 16.2835),
        (-50.0651, 16.2835, -50.151, 16.3304),
        (-50.151, 16.3304, -49.2171, 16.0535),
        (-83.669, 14.4485, -77.3837, 21.691),
        (-77.3837, 21.691, -80.4318, 22.3335),
        (-80.4318, 22.3335, -81.3657, 22.7671),
        (-81.3657, 22.7671, -83.669, 14.4485),
        (-24.981, 17.3667, -15.7162, 17.1727),
        (-15.7162, 17.1727, -10.5653, 16.6196),
        (-10.5653, 16.6196, -16.5986, 16.5188),
        (-10.5653, 16.6196, -4.6811, 16.3049),
        (-4.6811, 16.3049, -3.6841, 16.2388),
        (-3.6841, 16.2388, 1.9824, 16.5149),
        (1.9824, 16.5149, 9.3679, 16.9611),
        (9.3679, 16.9611, 10.1528, 16.8996),
        (9.3679, 16.9611, 12.5478, 17.5818),
        (12.5478, 17.5818, 9.5512, 18.1223),
        (9.5512, 18.1223, 2.4809, 18.0906),
        (2.4809, 18.0906, -9.7689, 17.984),
        (2.4809, 18.0906, 2.9164, 18.0107),
        (2.9164, 18.0107, 2.6986, 17.7976),
        (2.6986, 17.7976, 4.5665, 17.7243),
        (4.5665, 17.7243, 4.131, 17.4416),
        (20.1337, 6.0649, 14.1979, 6.199),
        (14.1979, 6.199, 9.6314, 6.0394),
        (9.6314, 6.0394, 7.4026, 5.9194),
        (7.4026, 5.9194, -1.9309, 5.6795),
        (-1.9309, 5.6795, -9.6658, 5.7957),
        (-9.6658, 5.7957, -8.199, 5.2422),
        (-8.199, 5.2422, -0.2807, 5.5332),
        (-0.2807, 5.5332, 6.3312, 5.419),
        (6.3312, 5.419, 9.9351, 5.5856),
        (9.9351, 5.5856, 7.4026, 5.9194),
        (6.3312, 5.419, 6.95, 4.8304),
        (6.95, 4.8304, 5.5978, 4.8533),
        (5.5978, 4.8533, 2.4351, 4.9038),
        (2.4351, 4.9038, 1.7017, 4.9756),
        (6.95, 4.8304, 8.898, 4.8434),
        (8.898, 4.8434, 10.1528, 4.9148),
        (-56.7343, 20.4271, -66.1995, 20.7487),
        (-66.1995, 20.7487, -66.7496, 20.6991),
        (-66.7496, 20.6991, -72.8974, 20.0096),
        (-72.8974, 20.0096, -71.4192, 18.7174),
        (-71.4192, 18.7174, -64.7156, 17.7617),
        (-64.7156, 17.7617, -63.6671, 18.1429),
        (-63.6671, 18.1429, -61.4841, 18.3874),
        (-61.4841, 18.3874, -62.1831, 18.8702),
        (-62.1831, 18.8702, -66.1652, 20.1448),
        (-66.1652, 20.1448, -66.1995, 20.7487),
        (-66.1995, 20.7487, -65.3516, 21.4405),
        (9.8663, 21.7361, 6.1822, 22.17),
        (6.1822, 22.17, 10.8174, 22.691),
        (10.8174, 22.691, 12.1639, 22.7785),
        (12.1639, 22.7785, 15.2006, 23.0795),
        (15.2006, 23.0795, 15.1834, 0.2204),
        (15.1834, 0.2204, 29.0833, 0.1394),
        (29.0833, 0.1394, 28.0692, 23.0627),
        (28.0692, 23.0627, 15.2006, 23.0795),
        (25.6341, 21.7437, 25.3362, 22.1166),
        (25.3362, 22.1166, 30.2178, 22.7166),
        (30.2178, 22.7166, 28.0692, 23.0627),
        (28.0692, 23.0627, 24.6028, 22.8335),
        (24.6028, 22.8335, 23.5486, 22.7755),
        (23.5486, 22.7755, 17.3492, 21.7418),
        (50.3515, 4.1096, 48.3977, 4.2479),
        (48.3977, 4.2479, 47.6987, 4.1444),
        (47.6987, 4.1444, 47.7847, 3.7151),
        (47.7847, 3.7151, 48.1858, 3.6077),
        (48.1858, 3.6077, 49.8473, 3.4049),
        (49.8473, 3.4049, 53.5028, 3.0798),
        (53.5028, 3.0798, 55.8806, 2.8449),
        (47.7847, 3.7151, 42.565, 3.7533),
        (42.565, 3.7533, 39.9982, 3.9637),
        (39.9982, 3.9637, 35.7812, 3.9828),
        (35.7812, 3.9828, 31.8851, 3.9022),
        (31.8851, 3.9022, 32.2862, 3.7384),
        (49.8473, 3.4049, 49.601, 3.1513),
        (49.601, 3.1513, 49.2171, 2.7361),
        (49.601, 3.1513, 44.8511, 3.1581),
        (44.8511, 3.1581, 40.9493, 3.136),
        (40.9493, 3.136, 38.8351, 3.086),
        (38.8351, 3.086, 39.6487, 2.9794),
        (-49.0681, 1.5206, -43.3156, 1.4729),
        (-43.3156, 1.4729, -45.5158, 1.2529),
        (-45.5158, 1.2529, -46.719, 1.1012),
        (-46.719, 1.1012, -42.3015, 0.4377),
        (-42.3015, 0.4377, -45.7335, 0.1566),
        (-45.7335, 0.1566, -42.5994, 23.5845),
        (-42.5994, 23.5845, -45.4814, 23.6307),
        (-45.4814, 23.6307, -52.735, 23.9821),
        (-46.719, 1.1012, -55.2331, 1.1394),
        (-55.2331, 1.1394, -57.4505, 0.7223),
        (-57.4505, 0.7223, -45.7335, 0.1566),
        (-61.931, 6.8029, -56.1671, 5.8304),
        (-56.1671, 5.8304, -51.0505, 5.7876),
        (30.086, 1.1944, 27.2499, 1.3243),
        (27.2499, 1.3243, 24.5856, 1.2288),
        (24.5856, 1.2288, 15.3324, 1.5244),
        (15.3324, 1.5244, 9.1501, 1.7567),
        (9.1501, 1.7567, 2.7502, 2.034),
        (2.7502, 2.034, 5.4832, 1.6906),
        (5.4832, 1.6906, 6.1306, 1.5027),
        (6.1306, 1.5027, 7.8839, 1.0489),
        (7.8839, 1.0489, 7.586, 0.8109),
        (7.586, 0.8109, 6.8526, 23.9882),
        (6.8526, 23.9882, 5.615, 23.6654),
        (5.615, 23.6654, 6.3656, 23.4661),
        (6.3656, 23.4661, 3.2659, 23.2862),
        (3.2659, 23.2862, 1.249, 23.4489),
        (1.249, 23.4489, 1.7647, 23.7006),
        (1.7647, 23.7006, 5.615, 23.6654),
        (-29.6162, 22.9607, -27.0321, 22.6773),
        (-27.0321, 22.6773, -27.7655, 22.2384),
        (-27.7655, 22.2384, -30.8824, 21.7957),
        (-30.8824, 21.7957, -33.0138, 21.7487),
        (-33.0138, 21.7487, -32.9852, 22.1395),
        (-32.9852, 22.1395, -32.332, 22.5249),
        (-32.332, 22.5249, -32.8649, 22.8755),
        (-32.8649, 22.8755, -32.5325, 22.9321),
        (-32.5325, 22.9321, -29.6162, 22.9607),
        (-19.2342, 8.1505, -22.8667, 7.9477),
        (-22.8667, 7.9477, -24.8492, 7.8216),
        (-24.8492, 7.8216, -26.7972, 7.6467),
        (-26.7972, 7.6467, -28.3671, 7.5894),
        (-28.3671, 7.5894, -30.9512, 7.5115),
        (-30.9512, 7.5115, -37.0818, 7.2857),
        (-37.0818, 7.2857, -43.1838, 6.6295),
        (-43.1838, 6.6295, -50.5979, 6.8323),
        (-50.5979, 6.8323, -44.6334, 7.2254),
        (-44.6334, 7.2254, -43.2984, 7.4874),
        (-43.2984, 7.4874, -39.9982, 8.0596),
        (-39.9982, 8.0596, -24.2991, 8.1257),
        (-24.2991, 8.1257, -22.8667, 7.9477),
        (-35.2999, 8.6685, -33.1857, 8.7265),
        (-33.1857, 8.7265, -27.7025, 8.8423),
        (-64.8015, 3.7361, -62.4639, 4.2407),
        (-62.4639, 4.2407, -59.3011, 4.2743),
        (-59.3011, 4.2743, -61.0658, 4.0218),
        (-61.0658, 4.0218, -61.3982, 3.979),
        (-61.3982, 3.979, -64.8015, 3.7361),
        (19.9848, 20.0856, 19.4806, 19.979),
        (19.4806, 19.979, 18.5352, 19.7896),
        (18.5352, 19.7896, 17.4695, 19.6838),
        (18.5352, 19.7896, 18.0023, 19.6685),
        (-44.7824, 19.3866, -41.866, 19.921),
        (-41.866, 19.921, -40.5998, 19.3977),
        (-41.866, 19.921, -35.2656, 19.9955),
        (-35.2656, 19.9955, -34.6811, 19.9974),
        (-34.6811, 19.9974, -27.7025, 20.0444),
        (-27.7025, 20.0444, -27.1639, 19.9489),
        (-27.1639, 19.9489, -24.8836, 19.6116),
        (-24.8836, 19.6116, -25.2503, 19.259),
        (-25.2503, 19.259, -27.6681, 19.1154),
        (-27.6681, 19.1154, -29.8683, 19.0432),
        (-29.8683, 19.0432, -26.9806, 18.7605),
        (-26.9806, 18.7605, -26.2816, 18.921),
        (-26.2816, 18.921, -27.6681, 19.1154),
        (-26.2816, 18.921, -21.7323, 19.078),
        (-21.7323, 19.078, -21.0161, 19.1628),
        (-21.0161, 19.1628, -18.9477, 19.2938),
        (-18.9477, 19.2938, -17.8362, 19.361),
        (-21.7323, 19.078, -21.102, 18.9622),
        (-26.9806, 18.7605, -25.4164, 18.466),
        (-25.4164, 18.466, -21.0505, 18.2296),
        (-25.4164, 18.466, -29.8167, 18.3499),
        (-29.8167, 18.3499, -34.3832, 18.4026),
        (-34.3832, 18.4026, -36.7495, 18.2938),
        (-29.8167, 18.3499, -30.4183, 18.0967),
        (-30.4183, 18.0967, -27.8171, 17.7926),
        (-19.8014, 16.0906, -20.6666, 16.1135),
        (-20.6666, 16.1135, -19.4347, 16.1994),
        (-19.4347, 16.1994, -11.3675, 16.0722),
        (-11.3675, 16.0722, -19.8014, 16.0906),
        (-19.8014, 16.0906, -22.6146, 16.0054),
        (-22.6146, 16.0054, -26.0982, 15.9806),
        (-26.0982, 15.9806, -29.1979, 15.9477),
        (-22.6146, 16.0054, -25.5826, 16.3526),
        (-25.5826, 16.3526, -26.4191, 16.4893),
        (-26.4191, 16.4893, -28.201, 16.5978),
        (-28.201, 16.5978, -34.2858, 16.8362),
        (-34.2858, 16.8362, -34.6983, 16.523),
        (-34.6983, 16.523, -35.2484, 16.6062),
        (-35.2484, 16.6062, -34.2858, 16.8362),
        (-34.2858, 16.8362, -38.0329, 16.8644),
        (-38.0329, 16.8644, -42.3473, 16.9095),
        (-42.3473, 16.9095, -43.2354, 17.2021),
        (-43.2354, 17.2021, -42.9833, 17.6215),
        (-42.9833, 17.6215, -40.1185, 17.7926),
        (-40.1185, 17.7926, -39.0184, 17.7078),
        (-39.0184, 17.7078, -37.099, 17.56),
        (-40.1185, 17.7926, -37.036, 17.8304),
        (-29.3526, 0.9767, -28.9687, 0.3583),
        (-28.9687, 0.3583, -28.115, 23.8156),
        (-28.115, 23.8156, -32.5154, 23.3133),
        (-32.5154, 23.3133, -37.8152, 23.5493),
        (-4.7326, 18.7861, -8.2334, 18.5868),
        (-8.2334, 18.5868, -14.5474, 18.4867),
        (-8.2334, 18.5868, -8.9324, 18.3946),
        (-3.4148, 15.8266, 2.183, 15.8385),
        (2.183, 15.8385, 4.4691, 15.8465),
        (4.4691, 15.8465, 6.4171, 15.7376),
        (6.4171, 15.7376, 10.531, 15.5799),
        (10.531, 15.5799, 15.4183, 15.7693),
        (15.4183, 15.7693, 15.6475, 15.9404),
        (15.6475, 15.9404, 18.1341, 15.8121),
        (18.1341, 15.8121, 15.4183, 15.7693),
        (-0.636, 10.5046, -0.3667, 10.1322),
        (-0.3667, 10.1322, -8.1016, 9.8751),
        (28.6021, 5.4385, 22.9527, 4.704),
        (22.9527, 4.704, 19.1654, 4.4767),
        (19.1654, 4.4767, 17.9164, 4.4244),
        (17.9164, 4.4244, 17.5325, 4.3824),
        (17.5325, 4.3824, 15.6188, 4.33),
        (15.6188, 4.33, 15.8652, 4.4779),
        (15.8652, 4.4779, 16.5012, 4.5982),
        (16.5012, 4.5982, 21.1307, 5.6272),
        (15.6188, 4.33, 12.4848, 4.0111),
        (12.4848, 4.0111, 12.9317, 3.5145),
        (12.9317, 3.5145, 9.7174, 3.4526),
        (9.7174, 3.4526, 9.0184, 3.4133),
        (12.4848, 4.0111, 8.8808, 4.2582),
        (8.8808, 4.2582, 5.9817, 4.0523),
        (-45.9512, 18.1872, -45.9684, 18.4496),
        (-45.9684, 18.4496, -49.0681, 18.4806),
        (29.5646, 1.8843, 33.2831, 2.0493),
        (33.2831, 2.0493, 34.9848, 2.1589),
        (34.9848, 2.1589, 33.8332, 2.2884),
        (33.8332, 2.2884, 29.5646, 1.8843),
        (-69.0185, 16.8106, -63.415, 15.9191),
        (-63.415, 15.9191, -66.3141, 15.6116),
        (-66.3141, 15.6116, -68.669, 15.3152),
        (-68.669, 15.3152, -69.0185, 16.8106),
        (-64.9505, 22.4557, -60.2522, 22.3083),
        (-60.2522, 22.3083, -58.2354, 23.2907),
        (-58.2354, 23.2907, -65.5693, 23.9981),
        (-65.5693, 23.9981, -64.8646, 0.3346),
        (-64.8646, 0.3346, -62.9509, 0.5256),
        (-62.9509, 0.5256, -58.2354, 23.2907),
        (49.2973, 13.7922, 54.918, 13.3988),
        (54.918, 13.3988, 55.9493, 12.9007),
        (55.9493, 12.9007, 57.015, 12.2571),
        (57.015, 12.2571, 53.6861, 11.8973),
        (53.6861, 11.8973, 56.3676, 11.0306),
        (56.3676, 11.0306, 61.7477, 11.0623),
        (53.6861, 11.8973, 47.7675, 11.7674),
        (47.7675, 11.7674, 31.5184, 11.3029),
        (47.7675, 11.7674, 44.4844, 11.1612),
        (44.4844, 11.1612, 42.8974, 10.285),
        (44.4844, 11.1612, 41.4821, 10.3721),
        (61.7477, 11.0623, 63.0483, 9.5256),
        (63.0483, 9.5256, 60.7163, 8.5046),
        (56.3676, 11.0306, 54.0528, 9.8682),
        (54.0528, 9.8682, 59.0318, 9.8495),
        (54.0528, 9.8682, 51.6693, 9.5474),
        (51.6693, 9.5474, 48.0311, 8.9867),
        (48.0311, 8.9867, 47.1487, 9.0604),
        (47.1487, 9.0604, 51.6693, 9.5474),
        (89.2496, 2.5302, 86.5854, 17.5367),
        (86.5854, 17.5367, 82.0361, 16.7663),
        (82.0361, 16.7663, 77.7848, 15.7346),
        (77.7848, 15.7346, 74.1522, 14.845),
        (74.1522, 14.845, 71.8317, 15.3457),
        (71.8317, 15.3457, 75.7508, 16.2915),
        (75.7508, 16.2915, 77.7848, 15.7346),
        (-49.4176, 10.7796, -48.2144, 10.6215),
        (-48.2144, 10.6215, -42.1181, 10.2456),
        (-42.1181, 10.2456, -40.468, 9.5115),
        (-40.468, 9.5115, -43.4187, 9.1333),
        (-43.4187, 9.1333, -42.6338, 8.7399),
        (-42.6338, 8.7399, -42.9833, 8.6272),
        (-42.9833, 8.6272, -47.332, 8.1589),
        (-47.332, 8.1589, -54.7003, 8.7445),
        (-54.7003, 8.7445, -54.9982, 9.3682),
        (-54.9982, 9.3682, -54.5685, 9.9477),
        (-54.5685, 9.9477, -49.4176, 10.7796),
        (8.7147, 12.0867, 6.5145, 11.7644),
        (6.5145, 11.7644, 1.7475, 11.8449),
        (1.7475, 11.8449, -0.6646, 12.3316),
        (-0.6646, 12.3316, -1.4324, 12.6945),
        (-1.4324, 12.6945, 3.3862, 12.9267),
        (3.3862, 12.9267, 10.9492, 13.0359),
        (3.3862, 12.9267, -0.5844, 13.5783),
        (-0.5844, 13.5783, -11.1498, 13.4198),
        (-11.1498, 13.4198, -5.5348, 13.1654),
        (-5.5348, 13.1654, -1.4324, 12.6945),
        (-0.5844, 13.5783, 1.5355, 14.0272),
        (1.5355, 14.0272, 1.885, 14.7705),
        (-11.1498, 13.4198, -10.2674, 14.2151),
        (-10.2674, 14.2151, -5.9989, 14.2666),
        (-5.9989, 14.2666, -5.6494, 14.7178),
        (-66.3829, 9.0405, -66.1308, 8.429),
        (-66.1308, 8.429, -68.6174, 8.1318),
        (-68.6174, 8.1318, -67.9528, 7.2804),
        (-67.9528, 7.2804, -70.4853, 7.1452),
        (-70.4853, 7.1452, -72.5995, 7.6967),
        (-72.5995, 7.6967, -68.6174, 8.1318),
        (24.0642, 19.8912, 24.6486, 19.4783),
        (24.6486, 19.4783, 21.3828, 19.2701),
        (4.1998, 18.9366, -2.882, 18.3549),
        (-2.882, 18.3549, -12.8686, 17.6899),
        (-12.8686, 17.6899, -15.3839, 17.6261),
        (-15.3839, 17.6261, -12.8343, 17.3473),
    ]

def preprocess_constellation_data_np(constellation_data):
    # Initialize lists for Cartesian coordinates
    cartesian_data = []

    for dec1, ra1, dec2, ra2 in constellation_data:
        # Convert RA and Dec to radians
        ra1_rad, dec1_rad = 2 * np.pi - np.radians(ra1 * 15), np.radians(dec1)
        ra2_rad, dec2_rad = 2 * np.pi - np.radians(ra2 * 15), np.radians(dec2)

        # Convert spherical to Cartesian
        x1, y1, z1 = spherical_to_cartesian(ra1_rad, dec1_rad)
        x2, y2, z2 = spherical_to_cartesian(ra2_rad, dec2_rad)

        cartesian_data.append([x1, y1, z1, x2, y2, z2])

    return np.array(cartesian_data)

def update_visible_constellations_np(ax, fig, preprocessed_data):
    def on_view_change(event=None):
        # Clear the existing constellation lines
        for line in ax.lines:
            line.remove()

        # Get the current camera direction
        camera_direction = get_viewing_direction(ax)

        # Split the array for easier processing
        start_points = preprocessed_data[:, :3]  # (x1, y1, z1)
        end_points = preprocessed_data[:, 3:]    # (x2, y2, z2)

        # Determine visibility using vectorized operations
        visible_mask_start = is_visible_from_camera_vectorized(camera_direction, start_points)
        visible_mask_end = is_visible_from_camera_vectorized(camera_direction, end_points)
        visible_mask = visible_mask_start | visible_mask_end

        # Filter the visible constellations
        visible_arcs = preprocessed_data[visible_mask]

        # Plot all visible arcs
        for arc in visible_arcs:
            x1, y1, z1, x2, y2, z2 = arc
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='black', linewidth=0.5, alpha=1.0, zorder=4)

        # Add celestial equator
        lon_eq = np.linspace(0, 2 * np.pi, 100)
        lat_eq = np.zeros_like(lon_eq)  # Latitude = 0 for equator
        x_eq, y_eq, z_eq = spherical_to_cartesian(lon_eq, lat_eq)
        ax.plot(x_eq, y_eq, z_eq, 'r', linewidth=1, label='Celestial Equator', zorder=4)

        # Refresh the figure
        fig.canvas.draw_idle()

    # Attach the callback to the figure's rotation
    fig.canvas.mpl_connect('motion_notify_event', on_view_change)

def is_visible_from_camera_vectorized(camera_direction, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cx, cy, cz = camera_direction
    return (x * cx + y * cy + z * cz) > 0  # Example dot product logic for visibility

def get_viewing_direction(ax):
    azim = np.radians(ax.azim)  # Azimuth angle in radians
    elev = np.radians(ax.elev)  # Elevation angle in radians

    # Spherical to Cartesian conversion for the camera direction
    x = np.cos(azim) * np.cos(elev)
    y = np.sin(azim) * np.cos(elev)
    z = np.sin(elev)
    return np.array([x, y, z])

# Close any existing figures
plt.close('all')

# Parameters
visible_pixels = np.array([{visible_pixels_str}])
nside_grid = 2
nside_fine = 8
nside_ratio = nside_fine // nside_grid
pixels_per_coarse = nside_ratio * nside_ratio
hp = HEALPix(nside=nside_fine, order = 'nested', frame = 'icrs')
boundaries = hp.boundaries_skycoord(range(hp.npix), step = 10)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

panel = []
col = []
for i, boundary in enumerate(boundaries):
    lon = boundary.ra.radian
    lat = boundary.dec.radian
    lon = np.append(lon, lon[0])
    lat = np.append(lat, lat[0])
    lon = 2 * np.pi - lon
    x, y, z = spherical_to_cartesian(lon, lat)
    panel.append(np.column_stack((x, y, z)))
    containing_grid_pixel = i // pixels_per_coarse
    col.append('gold' if (containing_grid_pixel in visible_pixels) else 'midnightblue')

ax.add_collection(Poly3DCollection(panel, facecolors=col, edgecolors='none', alpha=0.9, zorder=1))

# Add celestial equator
lon_eq = np.linspace(0, 2*np.pi, 100)
lat_eq = np.zeros_like(lon_eq)  # Latitude = 0 for equator
x_eq, y_eq, z_eq = spherical_to_cartesian(lon_eq, lat_eq)
ax.plot(x_eq, y_eq, z_eq, 'r', linewidth=1, label='Celestial Equator', zorder=4)

# Add North Celestial Pole (NCP)
# NCP is at latitude = 90° (pi/2 radians)
x_ncp, y_ncp, z_ncp = spherical_to_cartesian(0, np.pi/2, 1.1)
ax.scatter(x_ncp, y_ncp, z_ncp, color='red', s=100, marker='*', label='NCP', zorder=2)

constellation_data = get_constellation_data()
preprocessed_data = preprocess_constellation_data_np(constellation_data)
update_visible_constellations_np(ax, fig, preprocessed_data)

ax.view_init(elev=30, azim=45)
ax.set_axis_off()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect('equal')
plt.tight_layout()

plt.show()

"""

    # Execute the script in a subprocess
    process = subprocess.run(
        [sys.executable, "-c", script],  # Run the script as a string
        text=True,
        capture_output=True
    )

    # Check if the subprocess encountered an error
    if process.returncode != 0:
        print("Subprocess encountered an error:")
        print(process.stderr)

def fetch_and_store_chunk(pixel, url_base, target_path):
    # Create the file name and sha256sum file name using the provided pixel
    file_name = f"siril_cat2_healpix8_xpsamp_{pixel}.dat.bz2"
    sha256_file_name = f"{file_name}.sha256sum"

    # Path to the existing .dat.bz2 file
    target_file_path = os.path.join(target_path, file_name)

    # URL for downloading the sha256sum file
    url_sha256sum = f"{url_base}/{sha256_file_name}"

    # Fetch the sha256sum from the URL
    print(f"Fetching checksum file for pixel {pixel}...")
    sha256sum_temp = urllib.request.urlopen(url_sha256sum).read().decode().strip()

    # Check if the file already exists in the target path
    if os.path.exists(target_file_path):
        # Verify the sha256sum of the existing file
        print(f"Verifying existing file for pixel {pixel}...")
        actual_sha256sum = hashlib.sha256()
        with open(target_file_path, 'rb') as file:
            while chunk := file.read(8192):  # Read in chunks
                actual_sha256sum.update(chunk)

        if actual_sha256sum.hexdigest() == sha256sum_temp:
            # If the file exists and the checksum matches, no need to download
            print(f"File for pixel {pixel} already exists and checksum matches.")
            return
        else:
            # File exists but checksum does not match, remove the incorrect file
            print(f"Checksum mismatch for existing file for pixel {pixel}, removing file.")
            os.remove(target_file_path)

    # File not found or checksum mismatch, download the bz2 file
    print(f"Fetching .bz2 file for pixel {pixel}...")
    url_file = f"{url_base}/{file_name}"
    urllib.request.urlretrieve(url_file, target_file_path)

    # Verify the downloaded file's sha256sum
    print(f"Verifying checksum of downloaded file for pixel {pixel}...")
    downloaded_sha256sum = hashlib.sha256()
    with open(target_file_path, 'rb') as file:
        while chunk := file.read(8192):
            downloaded_sha256sum.update(chunk)

    # Check if downloaded file's checksum matches the expected checksum
    if downloaded_sha256sum.hexdigest() != sha256sum_temp:
        print(f"SHA256 mismatch for pixel {pixel}, aborting.")
        os.remove(target_file_path)  # Remove the corrupted download
        return

    # File checksum verified, decompress it
    print(f"Decompressing file for pixel {pixel}...")

    # Uncompress the .bz2 file and store it in the target directory
    decompressed_file_path = target_file_path[:-4]  # Remove .bz2 extension
    with bz2.BZ2File(target_file_path, 'rb') as bz2_file:
        with open(decompressed_file_path, 'wb') as decompressed_file:
            shutil.copyfileobj(bz2_file, decompressed_file)

    # Optionally, clean up the bz2 file after decompression
    os.remove(target_file_path)

    print(f"Decompression successful for pixel {pixel}.")

def process_pixels(pixels, url_base, target_path):
    for pixel in pixels:
        fetch_and_store_chunk(pixel, url_base, target_path)

def main():
    parser = argparse.ArgumentParser(description="Siril Catalog Installer")
    parser.add_argument("-lat", type=float, default=0.0,
                        help="Observer latitude in degrees (-90 to +90)")
    parser.add_argument("-min_elev", type=float, default=0.0,
                        help="Minimum elevation you image at (0 to +90)")
    parser.add_argument("-type", type=str,
                        help="Type of catalog to install ('astro' or 'xp_sampled')")

    args = parser.parse_args()
    try:
        if args.type:
            # CLI mode
            app = SirilCatInstallerInterface(cli_args=args)
        else:
            # GUI mode
            root = ThemedTk()
            app = SirilCatInstallerInterface(root)
            root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

    # Example usage

#    process_pixels(pixels, "https://some.url", "~/.local/share/siril/siril_cat2_healpix8_xpsamp_test")
