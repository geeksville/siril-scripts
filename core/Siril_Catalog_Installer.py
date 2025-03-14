# (c) Adrian Knagg-Baugh 2025
# Siril Catalog Installer
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.5
# Version history:
# 1.0.0 Initial release
# 1.0.1 Update SPCC DOI number to reflect fixed catalog
# 1.0.2 Cyril Richard: Fix paths with spaces in catalog installation directories
# 1.0.3 Adrian Knagg-Baugh: Fix paths with backslashes in catalog installation directories
# 1.0.4 Adrian Knagg-Baugh: Improve error handling, adding retries and resume
# 1.0.5 AKB: convert "requires" to use exception handling

VERSION = "1.0.5"

# Catalog retrieval details
ASTRO_RECORD = 14692304
ASTRO_INDEXLEVEL = 8

SPCC_RECORD = 14738271
SPCC_CHUNKLEVEL = 1
SPCC_INDEXLEVEL = 8

import sirilpy as s
from sirilpy import tksiril
import argparse
import bz2
import hashlib
import math
import os
import subprocess
import sys
import time
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
    def __init__(self, root=None):

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

        try:
            siril.cmd("requires", "1.3.6")
        except:
            return

        self.catalog_path = self.siril.get_siril_userdatadir()

        if root:
            self.create_widgets()
            tksiril.match_theme_to_siril(self.root, self.siril)

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
        area_combo.pack(side=tk.RIGHT)
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
        self.method_var.set("")
        method_combo.pack(side=tk.RIGHT)
        tksiril.create_tooltip(method_combo, "Select how to filter the SPCC catalog: 'All' will install "
                        "all chunks; 'Visible from Latiude' will install all chunks that are visible from the observer's "
                        "latitude above the given minimum elevation during the course of the year; 'Area "
                        "of Interest' will install chunks covering the specified area of interest")

         # Buttons
        spcc_button_frame = ttk.Frame(spcc_frame)
        spcc_button_frame.pack(fill=tk.X, pady=5)

        # Configure the frame's column weights to make buttons equal
        spcc_button_frame.columnconfigure(0, weight=1)
        spcc_button_frame.columnconfigure(1, weight=1)

        # Preview button for HEALpixel coverage
        healpix_btn = ttk.Button(
            spcc_button_frame,
            text="Preview coverage",
            command=self.preview_coverage,
            style="TButton"
        )
        healpix_btn.grid(row=0, column=0, pady=2, sticky='ew')
        tksiril.create_tooltip(healpix_btn, "Preview HEALpix coverage")

        # Install button for SPCC
        spcc_install_btn = ttk.Button(
            spcc_button_frame,
            text="Install",
            command=self.install_spcc,
            style="TButton"
        )
        spcc_install_btn.grid(row=0, column=1, pady=2, sticky='ew')
        tksiril.create_tooltip(spcc_install_btn, "Install or update the SPCC catalog with selected parameters")

        # Catalog Path Selection Frame
        catpath_frame = ttk.LabelFrame(main_frame, text="Catalog Path", padding=10)
        catpath_frame.pack(fill=tk.X, padx=5, pady=5)

        self.catalog_path_var = tk.StringVar(value=self.catalog_path or "")
        catpath_entry = ttk.Entry(
            catpath_frame,
            textvariable=self.catalog_path_var,
            width=40
        )
        catpath_entry.pack(side=tk.LEFT, padx=(0, 5), expand=True)

        ttk.Button(
            catpath_frame,
            text="Browse",
            command=self._browse_catalog,
            style="TButton"
        ).pack(side=tk.LEFT)
        tksiril.create_tooltip(catpath_entry, "Set the catalog installation directory")

        # Close button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT)
        tksiril.create_tooltip(close_btn, "Close the Catalog Installer dialog")

    def _browse_catalog(self):
        filename = filedialog.askdirectory(
            title="Select Catalog Installation Path",
            initialdir=self.catalog_path
        )
        if filename:
            self.catalog_path_var.set(filename)

    def close_dialog(self):
        self.siril.disconnect()
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()

    def get_pixels_from_ui(self):
        pixels = None
        method = self.method_var.get()
        if method == "":
            pixels = []
        elif method == "Area of Interest":
            area = self.area_var.get()
            pixels = get_area_of_interest(area)
        elif method == "Visible from Latitude":
            lat = self.latitude_var.get()
            min_elev = self.elevation_var.get()
            pixels = get_visible_healpix(latitude=lat, min_elevation=min_elev)
        else: # method == "All":
            pixels = list(range(48))
        return pixels

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
        # Confirmation dialog, as this is a large amount of data
        proceed = messagebox.askyesno(
            "Confirm Download",
            "This will download a large amount of data. Are you sure you want to proceed?",
            icon='warning'
        )

        if proceed:
            # URLs of the files to download
            catfile = f"siril_cat_healpix{ASTRO_INDEXLEVEL}_astro.dat.bz2"
            shasumfile = f"{catfile}.sha256sum"
            bz2_url = f"https://zenodo.org/records/{ASTRO_RECORD}/files/{catfile}"
            sha256sum_url = f"{bz2_url}.sha256sum"

            # Set target dir
            target_dir = self.catalog_path_var.get()

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
                s.download_with_progress(self.siril, bz2_url, bz2_path)
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

            # Set the catalog in preferences
            print("Setting the catalog location in Preferences->Astrometry")
            escaped_path = decompressed_path.replace('\\', '\\\\')
            self.siril.cmd("set", f"\"core.catalogue_gaia_astro={escaped_path}\"")

            print("Installation completed successfully.")

    def install_spcc(self):
        proceed = messagebox.askyesno(
            "Confirm Download",
            "This will download a large amount of data. Are you sure you want to proceed?",
            icon='warning'
        )

        if proceed:
            pixels = self.get_pixels_from_ui()
            print(f"Installing the following Level 1 HEALpixels: {pixels}")
            chunks = []
            error = 0
            # Set target dir
            target_dir = os.path.join(self.catalog_path_var.get(), f"siril_cat{SPCC_CHUNKLEVEL}_healpix{SPCC_INDEXLEVEL}_xpsamp")
            # Ensure the target directory exists
            os.makedirs(target_dir, exist_ok=True)

            for pixel in pixels:
                catfile = f"siril_cat{SPCC_CHUNKLEVEL}_healpix{SPCC_INDEXLEVEL}_xpsamp_{pixel}.dat.bz2"
                chunks.append(catfile)
                shasumfile = f"{catfile}.sha256sum"
                bz2_url = f"https://zenodo.org/records/{SPCC_RECORD}/files/{catfile}"
                sha256sum_url = f"{bz2_url}.sha256sum"

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
                    try:
                        download_successful = s.download_with_progress(self.siril, bz2_url, bz2_path)
                    except RuntimeError as e:
                        self.siril.log(f"Download error: {e}")
                        self.siril.reset_progress()
                        raise
                        
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

            print("Setting the catalog location in Preferences->Astrometry")
            escaped_dir = target_dir.replace('\\', '\\\\')
            self.siril.cmd("set", f"\"core.catalogue_gaia_photo={escaped_dir}\"")

            if not error:
                print("Installation complete, all files installed successfully.")
            else:
                print("Installation complete but not all files installed successfully. Please review the error messages", file=sys.stderr)
            return

    def preview_coverage(self):
        pixels = self.get_pixels_from_ui()
        if pixels == []:
            print("Warning: no catalog chunks selected. Set the selection method.")
        cat_path = os.path.join(self.siril.get_siril_systemdatadir(), "catalogue", "constellations.csv")
        plot_visible_pixels(pixels, cat_path)
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
        return [5,8,9,10,24,25,26,27]
    elif area == "Magellanic Clouds":
        return [32,33,36,38]
    elif area == "Summer Triangle":
        return [9,12,13,14,15,29,31]
    elif area == "Milky Way":
        return [2,3,12,13,14,15,28,29,30,31,36,37,38,39,40,41,42,46]
    elif area == "Orion to Taurus":
        return [0,1,6,20,21,22,23]
    else:
        return []

def plot_visible_pixels(visible_pixels, filename, nside=2):
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

    # Define the script to be executed in the subprocess
    script = f"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from astropy.coordinates import get_sun, SkyCoord
from astropy.time import Time

def sphtoc(lon, lat, r = 1):
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

def pp_arcs(arcs):
    cartesian_data = []
    for ra1, dec1, ra2, dec2 in arcs:
        # Convert RA and Dec to radians
        ra1_rad, dec1_rad = 2 * np.pi - np.radians(ra1), np.radians(dec1)
        ra2_rad, dec2_rad = 2 * np.pi - np.radians(ra2), np.radians(dec2)
        x1, y1, z1 = sphtoc(ra1_rad, dec1_rad)
        x2, y2, z2 = sphtoc(ra2_rad, dec2_rad)
        cartesian_data.append([x1, y1, z1, x2, y2, z2])
    return np.array(cartesian_data)

def upd_vis_cons(ax, fig, ppdata):
    def on_view_change(event=None):
        # Clear the existing constellation lines
        for line in ax.lines:
            line.remove()
        camera_direction = get_view_dir(ax)
        start_points = ppdata[:, :3]
        end_points = ppdata[:, 3:]
        visible_mask_start = vis_from_cam(camera_direction, start_points)
        visible_mask_end = vis_from_cam(camera_direction, end_points)
        visible_mask = visible_mask_start | visible_mask_end
        visible_arcs = ppdata[visible_mask]
        for arc in visible_arcs:
            x1, y1, z1, x2, y2, z2 = arc
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='black', linewidth=0.5, alpha=1.0, zorder=4)
        # Celestial equator visibility
        lon_eq = np.linspace(0, 2 * np.pi, 100)
        lat_eq = np.zeros_like(lon_eq)
        x_eq, y_eq, z_eq = sphtoc(lon_eq, lat_eq)
        visible_eq_mask = vis_from_cam(camera_direction, np.column_stack((x_eq, y_eq, z_eq)))
        visible_indices = np.where(visible_eq_mask)[0]
        if len(visible_indices) > 0:
            segments = np.split(visible_indices, np.where(np.diff(visible_indices) != 1)[0] + 1)
            for segment in segments:
                ax.plot(x_eq[segment], y_eq[segment], z_eq[segment], 'r', linewidth=0.5, label='Celestial Equator', zorder=4)
        x_ncp, y_ncp, z_ncp = sphtoc(0, np.pi/2, 1.1)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', on_view_change)
    on_view_change()

def vis_from_cam(camera_direction, points):
    return np.dot(points, camera_direction) > 0

def get_view_dir(ax):
    x, y, z = sphtoc(np.radians(ax.azim), np.radians(ax.elev))
    return np.array([x, y, z])

plt.close('all')
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
    x, y, z = sphtoc(lon, lat)
    panel.append(np.column_stack((x, y, z)))
    containing_grid_pixel = i // pixels_per_coarse
    col.append('gold' if (containing_grid_pixel in visible_pixels) else 'midnightblue')
ax.add_collection(Poly3DCollection(panel, facecolors=col, edgecolors='none', alpha=0.9, zorder=1))
arcs = []
with open(r'{filename}', 'r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for row in reader:
        arcs.append(tuple(map(float, row)))
ppdata = pp_arcs(arcs)
date = Time.now()
sun_coords = get_sun(date)
ra_as = (sun_coords.ra.deg + 180) % 360
dec_as = -sun_coords.dec.deg
def get_roll(elev, azim):
    if abs(elev) > 90:
        return 180
    return 0
roll = get_roll(ra_as, dec_as)
ax.view_init(elev=ra_as, azim=dec_as, roll=roll)
ax.set_axis_off()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_aspect('equal')
plt.tight_layout()
upd_vis_cons(ax, fig, ppdata)
plt.show()
"""

    # Execute the script in a subprocess
    process = subprocess.Popen([sys.executable, "-c", script])

    # Check if the subprocess encountered an error starting up
    if process.errors is not None:
        print("Subprocess encountered an error:")
        print(process.errors)

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
    try:
        # GUI mode
        root = ThemedTk()
        app = SirilCatInstallerInterface(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
