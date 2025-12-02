# (c) Adrian Knagg-Baugh 2025
# Siril Catalog Installer
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 2.2.2
# Version history:
# 1.0.0 Initial release
# 1.0.1 Update SPCC DOI number to reflect fixed catalog
# 1.0.2 Cyril Richard: Fix paths with spaces in catalog installation directories
# 1.0.3 Adrian Knagg-Baugh: Fix paths with backslashes in catalog installation directories
# 1.0.4 Adrian Knagg-Baugh: Improve error handling, adding retries and resume
# 1.0.5 AKB: convert "requires" to use exception handling
# 1.0.6 CME: remove unnecesary imports, add missing import for shutil, corrected errors enums
# 1.0.7 CME: use new sirilpy filedialog module for Linux
# 2.0.0 CR: Using PyQt6 instead of tkinter
# 2.1.0 Refactored with Qt OpenGL sphere visualization
# 2.2.0 Refactored with VisPy sphere visualization
# 2.2.1 Disable zooming (prevents visual glitches)
# 2.2.2 Force X11 backend: PyOpenGL has problems on pure Wayland desktops

VERSION = "2.2.2"

# Catalog retrieval details
ASTRO_RECORD = 14692304
ASTRO_INDEXLEVEL = 8

SPCC_RECORD = 14738271
SPCC_CHUNKLEVEL = 1
SPCC_INDEXLEVEL = 8

import sirilpy as s
import bz2
import hashlib
import math
import os
import sys
import ctypes
import urllib.request
import numpy as np
import shutil
import csv

s.ensure_installed("PyQt6", "astropy", "astropy_healpix", "requests", "vispy")

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import get_sun, SkyCoord
from astropy.time import Time
import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QMessageBox, QComboBox, QFrame,
    QFileDialog, QGridLayout, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QDoubleValidator

# VisPy imports
import vispy
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.color import Color
from vispy.util.quaternion import Quaternion
from vispy.geometry import create_sphere

# Use PyQt6 backend for VisPy
vispy.use(app='pyqt6')

class SkyVisualizationDialog(QDialog):
    """Dialog containing the 3D sky visualization"""

    def __init__(self, visible_pixels, constellation_file, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sky Coverage Preview")
        self.setModal(False)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        try:
            self.sky_widget = SkyVisualizationWidget(visible_pixels, constellation_file)
            # Get the native widget from VisPy canvas
            native_widget = self.sky_widget.native
            layout.addWidget(native_widget)

            # Add control buttons
            control_layout = QHBoxLayout()

            view_toggle_button = QPushButton("Switch to Internal View")
            view_toggle_button.clicked.connect(self.toggle_view_mode)
            control_layout.addWidget(view_toggle_button)
            self.view_toggle_button = view_toggle_button

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close)
            control_layout.addWidget(close_button)

            layout.addLayout(control_layout)

        except Exception as e:
            error_label = QLabel(f"VisPy visualization not available: {str(e)}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)

    def toggle_view_mode(self):
        """Toggle between external and internal view modes"""
        if hasattr(self, 'sky_widget'):
            is_internal = self.sky_widget.toggle_view_mode()
            if is_internal:
                self.view_toggle_button.setText("Switch to External View")
            else:
                self.view_toggle_button.setText("Switch to Internal View")


class SkyVisualizationWidget:
    """VisPy-based widget for rendering the celestial sphere with HEALPix coverage"""

    def __init__(self, visible_pixels, constellation_file, parent=None):
        self.visible_pixels = set(visible_pixels)
        self.constellation_file = constellation_file
        self.is_internal_view = False

        # Create VisPy canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor='black',
            size=(800, 600),
            show=False
        )
        # Disconnect scroll wheel handler (zoom)
        # (prevents visual glitches)
        self.canvas.events.mouse_wheel.disconnect()

        # Create view and camera
        self.view = self.canvas.central_widget.add_view()
        self.setup_external_camera()

        # Initialize all visualizations (both internal and external elements)
        self.setup_all_visualizations()

    @property
    def native(self):
        """Return the native Qt widget"""
        return self.canvas.native

    def setup_external_camera(self):
        """Setup camera for external view"""
        camera = scene.ArcballCamera(
            fov=45,
            distance=3.0,
            center=(0, 0, 0)
        )
        camera.flip_factors = (1, 1, 1)  # Normal controls
        self.view.camera = camera

    def setup_internal_camera(self):
        """Setup camera for internal view"""
        camera = scene.ArcballCamera(
            fov=75,  # Wider FOV for internal view
            distance=0.1,  # Very close to center
            center=(0, 0, 0)
        )
        camera.flip_factors = (1, 1, 1)  # Reversed pan controls
        self.view.camera = camera

    def toggle_view_mode(self):
        """Toggle between external and internal view modes"""
        self.is_internal_view = not self.is_internal_view

        if self.is_internal_view:
            self.setup_internal_camera()
        else:
            self.setup_external_camera()

        return self.is_internal_view

    def setup_all_visualizations(self):
        """Set up all visualization components - both internal and external versions"""
        # Create opaque black sphere at radius 1.0
        self.create_opaque_sphere()

        # Create external elements (outside sphere)
        self.create_external_healpix_mesh()
        self.create_external_grid_lines()
        self.create_external_constellation_lines()

        # Create internal elements (inside sphere)
        self.create_internal_healpix_mesh()
        self.create_internal_grid_lines()
        self.create_internal_constellation_lines()

    def create_opaque_sphere(self):
        """Create an opaque black sphere that blocks visibility between internal/external"""
        # Create sphere geometry
        phi, theta = np.mgrid[0:np.pi:36j, 0:2*np.pi:72j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Stack coordinates and reshape for mesh
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1).astype(np.float32)

        # Create faces for the sphere
        faces = []
        n_phi, n_theta = phi.shape
        for i in range(n_phi - 1):
            for j in range(n_theta - 1):
                # Current quad vertices
                v1 = i * n_theta + j
                v2 = i * n_theta + (j + 1) % n_theta
                v3 = (i + 1) * n_theta + (j + 1) % n_theta
                v4 = (i + 1) * n_theta + j

                # Two triangles per quad
                faces.extend([[v1, v2, v3], [v1, v3, v4]])

        faces = np.array(faces, dtype=np.uint32)

        self.sphere_visual = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color='black',
            shading='flat',
            parent=self.view.scene
        )
        # Opaque sphere with no face culling - blocks view in both directions
        self.sphere_visual.set_gl_state(depth_test=True, blend=False, cull_face=False)

    def create_healpix_mesh(self, radius, cull_faces=True):
        """Create HEALPix mesh at specified radius with optional face culling"""
        nside_fine = 8
        nside_coarse = 2
        pixels_per_coarse = (nside_fine // nside_coarse) ** 2
        hp = HEALPix(nside=nside_fine, order='nested', frame='icrs')
        all_vertices = []
        all_faces = []
        vertex_count = 0

        # Process every nth pixel to reduce complexity while maintaining coverage
        step = max(1, hp.npix // 1000)  # Limit to ~1000 polygons max

        for i in range(0, hp.npix, step):
            containing_grid_pixel = i // pixels_per_coarse
            is_visible = containing_grid_pixel in self.visible_pixels

            if is_visible:
                try:
                    # Get pixel boundary with more points for smoother edges
                    boundary = hp.boundaries_skycoord(i, step=1)

                    # Convert SkyCoord to Cartesian coordinates
                    polygon_points = []

                    # Extract coordinates as arrays first
                    lon_array = boundary.ra.radian
                    lat_array = boundary.dec.radian

                    # Flatten the arrays to 1D
                    lon_flat = lon_array.flatten()
                    lat_flat = lat_array.flatten()

                    # Convert to Cartesian coordinates
                    for j in range(len(lon_flat)):
                        lon_val = float(lon_flat[j])
                        lat_val = float(lat_flat[j])

                        x = math.cos(lat_val) * math.cos(lon_val)
                        y = math.cos(lat_val) * math.sin(lon_val)
                        z = math.sin(lat_val)
                        polygon_points.append([x, y, z])

                    # Ensure we have enough points for triangulation
                    if len(polygon_points) >= 3:
                        # Remove duplicate points
                        unique_points = []
                        tolerance = 1e-12

                        for point in polygon_points:
                            is_duplicate = False
                            for existing in unique_points:
                                if (abs(point[0] - existing[0]) < tolerance and
                                    abs(point[1] - existing[1]) < tolerance and
                                    abs(point[2] - existing[2]) < tolerance):
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                unique_points.append(point)

                        polygon_points = unique_points

                        if len(polygon_points) >= 3:
                            # Calculate centroid
                            centroid = [0.0, 0.0, 0.0]
                            for point in polygon_points:
                                centroid[0] += point[0]
                                centroid[1] += point[1]
                                centroid[2] += point[2]

                            # Normalize centroid to sphere surface
                            n_points = len(polygon_points)
                            centroid = [x / n_points for x in centroid]
                            norm = math.sqrt(sum(x*x for x in centroid))
                            if norm > 0:
                                centroid = [x / norm for x in centroid]

                            # Add centroid as first vertex
                            all_vertices.append(centroid)
                            centroid_idx = vertex_count
                            vertex_count += 1

                            # Add polygon vertices
                            all_vertices.extend(polygon_points)

                            # Create triangulation
                            n_poly_vertices = len(polygon_points)
                            for k in range(n_poly_vertices):
                                next_k = (k + 1) % n_poly_vertices

                                # Check triangle validity
                                v1 = polygon_points[k]
                                v2 = polygon_points[next_k]

                                # Calculate triangle area
                                d1 = [v2[j] - centroid[j] for j in range(3)]
                                d2 = [v1[j] - centroid[j] for j in range(3)]
                                cross = [
                                    d1[1]*d2[2] - d1[2]*d2[1],
                                    d1[2]*d2[0] - d1[0]*d2[2],
                                    d1[0]*d2[1] - d1[1]*d2[0]
                                ]
                                area = 0.5 * math.sqrt(sum(x*x for x in cross))

                                if area > 1e-12:
                                    face = [centroid_idx, vertex_count + k, vertex_count + next_k]
                                    all_faces.append(face)

                            vertex_count += n_poly_vertices

                except Exception as e:
                    print(f"Error processing pixel {i}: {e}")
                    continue

        if all_vertices and all_faces:
            vertices = np.array(all_vertices, dtype=np.float32)
            faces = np.array(all_faces, dtype=np.uint32)

            # Validate faces
            valid_faces = []
            for face in faces:
                if len(set(face)) == 3 and all(0 <= idx < len(vertices) for idx in face):
                    valid_faces.append(face)

            if valid_faces:
                faces = np.array(valid_faces, dtype=np.uint32)

                # Scale vertices to specified radius
                vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius

                mesh = scene.visuals.Mesh(
                    vertices=vertices,
                    faces=faces,
                    color=(1.0, 0.843, 0.0, 1.0),  # Gold
                    shading='smooth',
                    parent=self.view.scene
                )

                mesh.set_gl_state(
                    depth_test=True,
                    blend=False,
                    cull_face=cull_faces
                )

                return mesh

        return None

    def create_external_healpix_mesh(self):
        """Create HEALPix mesh for external viewing (outside sphere)"""
        self.external_healpix = self.create_healpix_mesh(radius=1.005, cull_faces=True)

    def create_internal_healpix_mesh(self):
        """Create HEALPix mesh for internal viewing (inside sphere)"""
        self.internal_healpix = self.create_healpix_mesh(radius=0.99, cull_faces=False)

    def create_grid_lines(self, radius):
        """Create coordinate grid lines at specified radius"""
        grid_visuals = []

        # Generate longitude lines
        lon_lines = []
        for lon_deg in range(0, 360, 15):
            lon_rad = math.radians(lon_deg)
            line_points = []
            for lat_deg in range(-90, 91, 5):
                lat_rad = math.radians(lat_deg)
                x = math.cos(lat_rad) * math.cos(lon_rad) * radius
                y = math.cos(lat_rad) * math.sin(lon_rad) * radius
                z = math.sin(lat_rad) * radius
                line_points.append([x, y, z])
            lon_lines.extend(line_points)

        # Generate latitude lines
        lat_lines = []
        equator_lines = []
        for lat_deg in range(-90, 91, 15):
            lat_rad = math.radians(lat_deg)
            line_points = []
            for lon_deg in np.linspace(0, 360, 73, endpoint=True):
                lon_rad = math.radians(lon_deg)
                x = math.cos(lat_rad) * math.cos(lon_rad) * radius
                y = math.cos(lat_rad) * math.sin(lon_rad) * radius
                z = math.sin(lat_rad) * radius
                line_points.append([x, y, z])

            if lat_deg == 0:  # Equator
                equator_lines.extend(line_points)
            else:
                lat_lines.extend(line_points)

        # Create line visuals
        if lon_lines:
            lon_visual = scene.visuals.Line(
                pos=np.array(lon_lines),
                color='gray',
                width=1,
                connect='segments',
                parent=self.view.scene
            )
            grid_visuals.append(lon_visual)

        if lat_lines:
            lat_visual = scene.visuals.Line(
                pos=np.array(lat_lines),
                color='gray',
                width=1,
                connect='segments',
                parent=self.view.scene
            )
            grid_visuals.append(lat_visual)

        if equator_lines:
            eq_visual = scene.visuals.Line(
                pos=np.array(equator_lines),
                color='red',
                width=2,
                connect='segments',
                parent=self.view.scene
            )
            grid_visuals.append(eq_visual)

        return grid_visuals

    def create_external_grid_lines(self):
        """Create grid lines for external viewing"""
        self.external_grid = self.create_grid_lines(radius=1.01)

    def create_internal_grid_lines(self):
        """Create grid lines for internal viewing"""
        self.internal_grid = self.create_grid_lines(radius=0.98)

    def create_constellation_lines(self, radius):
        """Create constellation lines at specified radius"""
        if not os.path.exists(self.constellation_file):
            print(f"Constellation file not found: {self.constellation_file}")
            return None

        constellation_lines = []

        try:
            with open(self.constellation_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header

                for row in reader:
                    if len(row) >= 4:
                        try:
                            ra1, dec1, ra2, dec2 = map(float, row[:4])

                            # Convert to radians and then to Cartesian
                            ra1_rad = math.radians(ra1)
                            dec1_rad = math.radians(dec1)
                            ra2_rad = math.radians(ra2)
                            dec2_rad = math.radians(dec2)

                            x1 = math.cos(dec1_rad) * math.cos(ra1_rad) * radius
                            y1 = math.cos(dec1_rad) * math.sin(ra1_rad) * radius
                            z1 = math.sin(dec1_rad) * radius

                            x2 = math.cos(dec2_rad) * math.cos(ra2_rad) * radius
                            y2 = math.cos(dec2_rad) * math.sin(ra2_rad) * radius
                            z2 = math.sin(dec2_rad) * radius

                            constellation_lines.extend([[x1, y1, z1], [x2, y2, z2]])

                        except (ValueError, IndexError):
                            continue  # Skip malformed rows

        except Exception as e:
            print(f"Error loading constellation data: {e}")

        # Create constellation line visual
        if constellation_lines:
            return scene.visuals.Line(
                pos=np.array(constellation_lines),
                color='white',
                width=1,
                connect='segments',
                parent=self.view.scene
            )

        return None

    def create_external_constellation_lines(self):
        """Create constellation lines for external viewing"""
        self.external_constellations = self.create_constellation_lines(radius=1.015)

    def create_internal_constellation_lines(self):
        """Create constellation lines for internal viewing"""
        self.internal_constellations = self.create_constellation_lines(radius=0.97)

class SirilCatInstallerInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Siril Catalog Installer - v{VERSION}")

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            QMessageBox.critical(self, "Connection Error", "Failed to connect to Siril")
            self.close()
            raise RuntimeError(f"Error connecting to Siril: {e}") from e

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError as e:
            self.close()
            raise

        self.catalog_path = self.siril.get_siril_userdatadir()
        self.create_widgets()

    def create_widgets(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Astrometry Catalog frame
        astrometry_group = QGroupBox("Astrometry Catalog")
        astrometry_layout = QVBoxLayout(astrometry_group)

        # Install button for Astrometry
        astrometry_install_btn = QPushButton("Install")
        astrometry_install_btn.clicked.connect(self.install_astrometry)
        astrometry_install_btn.setToolTip("Install or update the Astrometry catalog. This will "
                        "be installed to the Siril user data directory and set in Preferences -> Astrometry")
        astrometry_layout.addWidget(astrometry_install_btn)

        main_layout.addWidget(astrometry_group)

        # SPCC Catalog frame
        spcc_group = QGroupBox("SPCC Catalog")
        spcc_layout = QVBoxLayout(spcc_group)

        # Observer Latitude entry
        latitude_layout = QHBoxLayout()
        latitude_layout.addWidget(QLabel("Observer Latitude:"))
        self.latitude_entry = QLineEdit()
        self.latitude_entry.setValidator(QDoubleValidator())
        self.latitude_entry.setToolTip("Enter your observatory latitude in degrees")
        latitude_layout.addWidget(self.latitude_entry)
        spcc_layout.addLayout(latitude_layout)

        # Minimum elevation entry
        elevation_layout = QHBoxLayout()
        elevation_layout.addWidget(QLabel("Minimum elevation:"))
        self.elevation_entry = QLineEdit()
        self.elevation_entry.setValidator(QDoubleValidator())
        self.elevation_entry.setToolTip("Enter minimum elevation in degrees")
        elevation_layout.addWidget(self.elevation_entry)
        spcc_layout.addLayout(elevation_layout)

        # Areas of Interest combobox
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("Areas of Interest:"))
        self.area_combo = QComboBox()
        self.area_combo.addItems(["Galaxy Season", "Magellanic Clouds", "Milky Way", "Orion to Taurus", "Summer Triangle"])
        self.area_combo.setCurrentText("Galaxy Season")
        self.area_combo.setToolTip("Select the area of interest for the SPCC catalog. This will install "
                               "only chunks covering the area of interest")
        area_layout.addWidget(self.area_combo)
        spcc_layout.addLayout(area_layout)

        # Selection Method combobox
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Selection Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["", "All", "Visible from Latitude", "Area of Interest"])
        self.method_combo.setToolTip("Select how to filter the SPCC catalog: 'All' will install "
                        "all chunks; 'Visible from Latitude' will install all chunks that are visible from the observer's "
                        "latitude above the given minimum elevation during the course of the year; 'Area "
                        "of Interest' will install chunks covering the specified area of interest")
        method_layout.addWidget(self.method_combo)
        spcc_layout.addLayout(method_layout)

        # Buttons for SPCC
        spcc_button_layout = QHBoxLayout()

        # Preview button for HEALpixel coverage
        healpix_btn = QPushButton("Preview coverage")
        healpix_btn.clicked.connect(self.preview_coverage)
        healpix_btn.setToolTip("Preview HEALpix coverage")
        spcc_button_layout.addWidget(healpix_btn)

        # Install button for SPCC
        spcc_install_btn = QPushButton("Install")
        spcc_install_btn.clicked.connect(self.install_spcc)
        spcc_install_btn.setToolTip("Install or update the SPCC catalog with selected parameters")
        spcc_button_layout.addWidget(spcc_install_btn)

        spcc_layout.addLayout(spcc_button_layout)
        main_layout.addWidget(spcc_group)

        # Catalog Path Selection Frame
        catpath_group = QGroupBox("Catalog Path")
        catpath_layout = QHBoxLayout(catpath_group)

        self.catalog_path_entry = QLineEdit(self.catalog_path or "")
        self.catalog_path_entry.setToolTip("Set the catalog installation directory")
        catpath_layout.addWidget(self.catalog_path_entry)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_catalog)
        catpath_layout.addWidget(browse_btn)

        main_layout.addWidget(catpath_group)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setToolTip("Close the Catalog Installer dialog")
        main_layout.addWidget(close_btn)

    def browse_catalog(self):
        """Browse for catalog directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Catalog Installation Path",
            self.catalog_path or ""
        )
        if directory:
            self.catalog_path_entry.setText(directory)

    def get_pixels_from_ui(self):
        """Get pixel list based on UI selection"""
        pixels = None
        method = self.method_combo.currentText()

        if method == "":
            pixels = []
        elif method == "Area of Interest":
            area = self.area_combo.currentText()
            pixels = get_area_of_interest(area)
        elif method == "Visible from Latitude":
            try:
                lat = float(self.latitude_entry.text() or "0")
                min_elev = float(self.elevation_entry.text() or "0")
                pixels = get_visible_healpix(latitude=lat, min_elevation=min_elev)
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Please enter valid latitude and elevation values")
                return []
        else:  # method == "All":
            pixels = list(range(48))
        return pixels

    def decompress_with_progress(self, bz2_path, decompressed_path):
        """Decompress bz2 file with progress updates"""
        print(f"Decompressing {bz2_path} to {decompressed_path}...")

        total_size = os.path.getsize(bz2_path)
        processed_size = 0

        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(decompressed_path, 'wb') as f_out:
            while True:
                chunk = f_in.read(8192)
                if not chunk:
                    break
                f_out.write(chunk)
                processed_size += 8192
                processed_size = min(processed_size, total_size)

                progress = processed_size / total_size
                if progress > 0.99:
                    self.siril.update_progress("Decompressing... (nearly done!)", progress)
                else:
                    self.siril.update_progress("Decompressing...", progress)
        self.siril.reset_progress()

    def verify_sha256sum(self, bz2_path, sha256sum_path):
        """Verify SHA256 checksum"""
        with open(sha256sum_path, 'r') as f:
            expected_checksum = f.read().split()[0]

        sha256_hash = hashlib.sha256()
        with open(bz2_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        actual_checksum = sha256_hash.hexdigest()

        if actual_checksum != expected_checksum:
            print(f"Checksum verification failed. Expected {expected_checksum}, got {actual_checksum}")
            return False
        else:
            print("Checksum verification succeeded.")
            return True

    def install_astrometry(self):
        """Install astrometry catalog"""
        reply = QMessageBox.question(
            self,
            "Confirm Download",
            "This will download a large amount of data. Are you sure you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # URLs of the files to download
            catfile = f"siril_cat_healpix{ASTRO_INDEXLEVEL}_astro.dat.bz2"
            shasumfile = f"{catfile}.sha256sum"
            bz2_url = f"https://zenodo.org/records/{ASTRO_RECORD}/files/{catfile}"
            sha256sum_url = f"{bz2_url}.sha256sum"

            # Set target dir
            target_dir = self.catalog_path_entry.text()

            # Ensure the target directory exists
            os.makedirs(target_dir, exist_ok=True)

            try:
                # Download the .sha256sum file
                sha256sum_path = os.path.join(target_dir, shasumfile)
                print(f"Downloading {sha256sum_url} to {sha256sum_path}...")
                response = requests.get(sha256sum_url)
                with open(sha256sum_path, 'wb') as f:
                    f.write(response.content)

                # Check if compressed archive exists and verify checksum
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

                # Decompress the .bz2 file
                decompressed_filename = os.path.basename(bz2_path).rsplit('.bz2', 1)[0]
                decompressed_path = os.path.join(target_dir, decompressed_filename)
                self.decompress_with_progress(bz2_path, decompressed_path)

                # Clean up
                print("Cleaning up...")
                os.remove(bz2_path)
                os.remove(sha256sum_path)

                # Set the catalog in preferences
                print("Setting the catalog location in Preferences->Astrometry")
                escaped_path = decompressed_path.replace('\\', '\\\\')
                self.siril.cmd("set", f"\"core.catalogue_gaia_astro={escaped_path}\"")

                print("Installation completed successfully.")
                QMessageBox.information(self, "Success", "Astrometry catalog installed successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Installation failed: {str(e)}")

    def install_spcc(self):
        """Install SPCC catalog"""
        reply = QMessageBox.question(
            self,
            "Confirm Download",
            "This will download a large amount of data. Are you sure you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            pixels = self.get_pixels_from_ui()
            if not pixels:
                QMessageBox.warning(self, "Selection Error", "No pixels selected. Please check your selection method and parameters.")
                return

            print(f"Installing the following Level 1 HEALpixels: {pixels}")
            error = 0

            # Set target dir
            target_dir = os.path.join(self.catalog_path_entry.text(), f"siril_cat{SPCC_CHUNKLEVEL}_healpix{SPCC_INDEXLEVEL}_xpsamp")
            os.makedirs(target_dir, exist_ok=True)

            try:
                for pixel in pixels:
                    catfile = f"siril_cat{SPCC_CHUNKLEVEL}_healpix{SPCC_INDEXLEVEL}_xpsamp_{pixel}.dat.bz2"
                    shasumfile = f"{catfile}.sha256sum"
                    bz2_url = f"https://zenodo.org/records/{SPCC_RECORD}/files/{catfile}"
                    sha256sum_url = f"{bz2_url}.sha256sum"

                    # Download the .sha256sum file
                    sha256sum_path = os.path.join(target_dir, shasumfile)
                    print(f"Downloading {sha256sum_url} to {sha256sum_path}...")
                    response = requests.get(sha256sum_url)
                    with open(sha256sum_path, 'wb') as f:
                        f.write(response.content)

                    # Check if compressed archive exists and verify checksum
                    bz2_path = os.path.join(target_dir, catfile)
                    if os.path.exists(bz2_path) and self.verify_sha256sum(bz2_path, sha256sum_path):
                        print("Existing archive found with valid checksum...")
                    else:
                        # Download the .bz2 file with progress reporting
                        print(f"Downloading {bz2_url} to {bz2_path}...")
                        try:
                            s.download_with_progress(self.siril, bz2_url, bz2_path)
                        except RuntimeError as e:
                            self.siril.log(f"Download error: {e}")
                            self.siril.reset_progress()
                            raise

                        if not self.verify_sha256sum(bz2_path, sha256sum_path):
                            print(f"Checksum verification error for {bz2_path}, skipping HEALpixel {pixel}.")
                            error = 1
                            continue

                    # Decompress the .bz2 file
                    decompressed_filename = os.path.basename(bz2_path).rsplit('.bz2', 1)[0]
                    decompressed_path = os.path.join(target_dir, decompressed_filename)
                    self.decompress_with_progress(bz2_path, decompressed_path)

                    # Clean up
                    print("Cleaning up...")
                    os.remove(bz2_path)
                    os.remove(sha256sum_path)
                    print(f"{decompressed_path} installed successfully.")

                print("Setting the catalog location in Preferences->Astrometry")
                escaped_dir = target_dir.replace('\\', '\\\\')
                self.siril.cmd("set", f"\"core.catalogue_gaia_photo={escaped_dir}\"")

                if not error:
                    print("Installation complete, all files installed successfully.")
                    QMessageBox.information(self, "Success", "SPCC catalog installed successfully!")
                else:
                    print("Installation complete but not all files installed successfully.")
                    QMessageBox.warning(self, "Partial Success", "Installation complete but not all files installed successfully. Please review the error messages.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Installation failed: {str(e)}")

    def preview_coverage(self):
        """Preview HEALpix coverage using VisPy visualization"""
        pixels = self.get_pixels_from_ui()
        if not pixels:
            print("Warning: no catalog chunks selected. Set the selection method.")
            QMessageBox.warning(self, "Selection Warning", "No catalog chunks selected. Set the selection method.")
            return

        # Get constellation file path
        cat_path = os.path.join(self.siril.get_siril_systemdatadir(), "catalogue", "constellations.csv")

        # Create and show the visualization dialog
        dialog = SkyVisualizationDialog(pixels, cat_path, self)
        dialog.show()

    def closeEvent(self, event):
        """Handle window close event"""
        self.siril.disconnect()
        event.accept()

def calculate_colatitude(latitude_deg, elevation_deg):
    """
    Compute the most extreme celestial colatitude observable above the given minimum elevation
    from an observer's terrestrial latitude.
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
    Compute HEALPix level 1 pixel numbers visible above minimum elevation.
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
    """Get HEALpix pixels for predefined areas of interest"""
    area_map = {
        "Galaxy Season": [5,8,9,10,24,25,26,27],
        "Magellanic Clouds": [32,33,36,38],
        "Summer Triangle": [9,12,13,14,15,29,31],
        "Milky Way": [2,3,12,13,14,15,28,29,30,31,36,37,38,39,40,41,42,46],
        "Orion to Taurus": [0,1,6,20,21,22,23]
    }
    return area_map.get(area, [])

def fetch_and_store_chunk(pixel, url_base, target_path):
    """Fetch and store a single catalog chunk"""
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
    """Process multiple pixels"""
    for pixel in pixels:
        fetch_and_store_chunk(pixel, url_base, target_path)

def main():
    """Main entry point"""
    try:
        os.environ['QT_QPA_PLATFORM'] = 'xcb' # Force XWayland: Wayland doesn't seem to work with PyOpenGL
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setApplicationName("Siril Catalog Installer")
        
        window = SirilCatInstallerInterface()
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
