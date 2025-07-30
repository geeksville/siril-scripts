# (c) Carlo Mollicone - AstroBOH (2025)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Description:
# This Siril script provides an interface to remove satellite trails
# from images. Trails can be filled with black pixels or with pixels
# from a reference image.

# Features:
# - Spline tracing with multiple management.
# - Stretched preview from Siril.
# - Mask with adjustable blur.
# - Progressive blend (soft edges) with controllable strength.
# - Filling with:
#       Black
#       Background median
#       Reference image
# - Visual overlay for mask confirmation.
#
#
# Versions:
# 0.0.1 - Initial release
# 0.0.5 - Stop manually handling the offset and rely completely on tkinter's internal methods
#         for managing coordinates on the Canvas, which take both zooming and panning into account.
#       - Create "extra space" so that the drag area is larger than the image itself,
#         so you can move the image all the way to the edges of the preview pane and still have room to maneuver.
#       - Fixed fit_to_preview because it didn't explicitly recenter the view within this new spacing
#         and the image was not centered in the canvas.
#       - Added instructions and improved the UI.
# 0.0.6 - Integration of the DAOPHOT MMM (Mean, Median, Mode) algorithm,
#         a robust astronomical standard for estimating the sky background while avoiding outliers, via the library: photutils
#       - Change "Black fill" with full black no blend
# 0.1.0 - Many improvements, New Features, Bug Fixes & Improvements
# 0.1.1 - Minor additions
# 0.2.0 - Big improvements, new features, bug fixes
#       - add auto-stretch for detection
#       - add AI detection button
#       - add Canny Edge Detection
#       - add Hough Transform for trail detection
#       - add debug parameters for canny and hough
#       - add DBSCAN clustering for trail grouping
#
# 0.2.1 - Bug Fixes
# 0.2.2 - Converting instructions to labels for a smoother GUI.
#         Update the instructions. Clearer and more complete.
# 1.0.0 - Better filedialog for Linux
#
#

VERSION = "1.0.0"

# --- Core Imports ---
import sys
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
# This allows the script to be run externally for testing (with limited functionality).
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=0.6.37'):
        messagebox.showerror("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    SIRIL_ENV = True

    # Better filedialog for Linux
    if s.check_module_version(">=0.6.0") and sys.platform.startswith("linux"):
        import sirilpy.tkfilebrowser as filedialog
    else:
        from tkinter import filedialog

    # Import Siril GUI related components
    from sirilpy import tksiril, SirilError
    from sirilpy.models import FPoint

    s.ensure_installed("ttkthemes", "numpy", "astropy", "opencv-python", "photutils", "pillow", "scikit-learn")
    from ttkthemes import ThemedTk

    # --- Imports for Image Processing ---
    import cv2
    import numpy as np

    from astropy.io import fits
    from photutils.background import MMMBackground, StdBackgroundRMS
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    from sklearn.cluster import DBSCAN

except ImportError:
    SIRIL_ENV = False
    messagebox.showerror("Warning: sirilpy not found. The script is not running in the Siril environment.")

# --- Interpolation Function ---
def get_curve_points(points, tension=0.5, num_of_segments=16):
    """
    Calculates the points of a cardinal spline passing through the given points.
    """
    if not points or len(points) < 2:
        return []

    pts = [coord for p in points for coord in p]
    res = []

    _pts = pts[:]
    _pts.insert(0, pts[1])
    _pts.insert(0, pts[0])
    _pts.append(pts[-2])
    _pts.append(pts[-1])

    for i in range(2, len(_pts) - 4, 2):
        for t in range(num_of_segments + 1):
            st = t / num_of_segments

            t1x = (_pts[i+2] - _pts[i-2]) * tension
            t2x = (_pts[i+4] - _pts[i]) * tension
            t1y = (_pts[i+3] - _pts[i-1]) * tension
            t2y = (_pts[i+5] - _pts[i+1]) * tension

            c1 = 2 * (st**3) - 3 * (st**2) + 1
            c2 = -2 * (st**3) + 3 * (st**2)
            c3 = (st**3) - 2 * (st**2) + st
            c4 = (st**3) - (st**2)

            x = c1 * _pts[i] + c2 * _pts[i+2] + c3 * t1x + c4 * t2x
            y = c1 * _pts[i+1] + c2 * _pts[i+3] + c3 * t1y + c4 * t2y
            res.append((x, y))

    return res

def compute_mmm_background(image):
    """
    Computes the sky background value using the DAOPHOT MMM method.
    If the image has 3 channels (RGB), returns an array for each channel.
    """
    mmm = MMMBackground()   # use its default parameters for box_size, sigma_clip, etc., as per the documentation
    
    if image.ndim == 3 and image.shape[2] == 3:  # RGB
        return np.array([mmm(image[..., c]) for c in range(3)])
    else:  # Monochrome or single channel
        return mmm(image)

def pack_rgba_color(r, g, b, a):
    """
    Packs RGBA values into a single 32-bit integer.
    The byte order required by Siril is (A << 24) | (B << 16) | (G << 8) | R.
    """
    return (a << 24) | (b << 16) | (g << 8) | r

def create_thick_line_polygon(points, width):
    """
    Creates a list of FPoint objects that define a closed polygon
    representing a thick line that follows the given points.
    """
    if len(points) < 2:
        return []

    polygon_points = []
    half_width = width / 2.0

    # Calculate the outline points
    left_side = []
    right_side = []

    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])

        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue

        perpendicular = np.array([-direction[1], direction[0]])
        normal = perpendicular / norm

        if i == 0:
            left_side.append(p1 + normal * half_width)
            right_side.append(p1 - normal * half_width)

        left_side.append(p2 + normal * half_width)
        right_side.append(p2 - normal * half_width)

    # Combine the two sides and convert to FPoint objects
    full_outline = left_side + right_side[::-1]

    # Return a list of FPoint objects, as required by sirilpy
    return [FPoint(x=p[0], y=p[1]) for p in full_outline]

def find_trails_with_hough(stretched_image_8bit, hough_params, debug_mode=False):
    """
    Detects lines in an image using the Hough Transform with customizable parameters.
    Returns a list of coordinates (x1, y1, x2, y2) for each line found.
    """
    # Extract Canny thresholds
    canny_low = hough_params['canny_low']
    canny_high = hough_params['canny_high']

    # Extract Hough parameters from the sub-dictionary
    hough = hough_params['hough']
    hough_threshold = hough['threshold']
    min_line_length = hough['minLineLength']
    max_line_gap = hough['maxLineGap']

    # Apply a slight blur to reduce noise
    #blurred = cv2.GaussianBlur(stretched_image_8bit, (5, 5), 0)
    blurred = stretched_image_8bit

    if debug_mode:
        # for canny auto detection
        bkg_estimator = MMMBackground()
        bkg_rms = StdBackgroundRMS()
        median = bkg_estimator(blurred)
        sigma = bkg_rms(blurred)
        lower_thresh = int(max(0, (1.0 - sigma) * median))
        upper_thresh = int(min(255, (1.0 + sigma) * median))

        print("--- DEBUG: canny test for auto detection value ---")
        print(f"    lower_thresh = {lower_thresh} --   upper_thresh = {upper_thresh}")
        print("----------------------------")


    # cv2.Canny(image, low_threshold, high_threshold)
    # The two arguments, low_threshold and high_threshold, allow for the isolation of adjacent pixels that follow the most intense gradient.
    # - If the gradient is greater than the upper threshold, it is identified as an edge pixel.
    # - If it is lower than the lower threshold, it is rejected.
    # The gradient between the thresholds is accepted only if it is connected to a strong edge.
    # Standard Canny thresholds: a low-to-high threshold ratio of 1:3.
    edges = cv2.Canny(blurred, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho = 1,
        theta = np.pi / 180,
        threshold = hough_threshold,
        minLineLength = min_line_length,
        maxLineGap = max_line_gap
    )
    
    found_lines = [line[0] for line in lines] if lines is not None else []
    return found_lines, edges

def process_detected_lines(lines, img_shape, debug_step_1=False, debug_step_2=False):
    """
    Processes Hough lines with robust logic and edge extrapolation.
    Includes two flag-controlled debug steps.

    1. Filters out segments that are too short.
    2. Groups neighboring segments based on their position.
    3. For each group, creates a line containing all the segments.
    4. Extends the final line to the exact edges of the image.

    - debug_step_1=True: Shows only the original Hough segments after length filtering.
    - debug_step_2=True: Shows the merged stripes but BEFORE edge extrapolation.
    """
    img_h, img_w = img_shape[:2]
    img_diag = math.hypot(img_w, img_h)

    # --- PASSO 1: Filter out lines that are too short ---
    min_length_perc = 0.015
    min_length_threshold = img_diag * min_length_perc

    segments = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_length_threshold:
            segments.append({
                'id': i,
                'p1': np.array([x1, y1]),
                'p2': np.array([x2, y2]),
                'points': [np.array([x1, y1]), np.array([x2, y2])]
            })

    if not segments:
        return []

    # --- DEBUG CASE 1: Show only the original filtered segments ---
    # If this flag is set, the function stops here and returns the raw (but length-filtered) segments
    # for analysis.
    if debug_step_1:
        print("--- DEBUG STEP 1: Show original segments after length filter ---")
        final_trails = []
        for seg in segments:
            final_trails.append({
                'points': [(int(seg['p1'][0]), int(seg['p1'][1])), (int(seg['p2'][0]), int(seg['p2'][1]))],
                'width': 2  # Assign a fixed width for the display
            })
        return final_trails

    # --- STEP 2 & 3: Group and Merge Neighboring Lines ---
    merge_dist_perc = 0.025
    merge_dist_threshold = img_diag * merge_dist_perc

    final_trails = []
    visited_indices = set()

    for i in range(len(segments)):
        if i in visited_indices:
            continue

        current_group_indices = {i}
        queue = [i]
        visited_indices.add(i)

        while queue:
            current_idx = queue.pop(0)
            current_segment = segments[current_idx]

            for j in range(len(segments)):
                if j not in visited_indices:
                    other_segment = segments[j]
                    dist = min(
                        np.linalg.norm(current_segment['p1'] - other_segment['p1']),
                        np.linalg.norm(current_segment['p1'] - other_segment['p2']),
                        np.linalg.norm(current_segment['p2'] - other_segment['p1']),
                        np.linalg.norm(current_segment['p2'] - other_segment['p2'])
                    )
                    if dist < merge_dist_threshold:
                        visited_indices.add(j)
                        current_group_indices.add(j)
                        queue.append(j)

        if not current_group_indices:
            continue
            
        all_points_in_group = []
        for idx in current_group_indices:
            all_points_in_group.extend(segments[idx]['points'])

        if len(all_points_in_group) < 2:
            continue

        max_dist = 0
        p_final_1, p_final_2 = all_points_in_group[0], all_points_in_group[1]
        for p_i_idx in range(len(all_points_in_group)):
            for p_j_idx in range(p_i_idx + 1, len(all_points_in_group)):
                dist = np.linalg.norm(all_points_in_group[p_i_idx] - all_points_in_group[p_j_idx])
                if dist > max_dist:
                    max_dist = dist
                    p_final_1, p_final_2 = all_points_in_group[p_i_idx], all_points_in_group[p_j_idx]

        [vx_fit, vy_fit, x0_fit, y0_fit] = cv2.fitLine(np.array(all_points_in_group, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        distances = np.abs((np.array(all_points_in_group)[:, 0] - x0_fit[0]) * vy_fit[0] - (np.array(all_points_in_group)[:, 1] - y0_fit[0]) * vx_fit[0])
        width = int(np.percentile(distances, 95) * 2 + 15)

        # --- DEBUG CASE 2 vs PRODUCTION (Extrapolation) ---
        if debug_step_2:
            # If this flag is set, the function adds the merged strip
            # as is, without extending it to the edges.
            print("--- DEBUG STEP 2: Show merged strips WITHOUT extrapolation ---")
            final_trails.append({
                'points': [(int(p_final_1[0]), int(p_final_1[1])), (int(p_final_2[0]), int(p_final_2[1]))],
                'width': width
            })
        else:
            # PRODUCTION Logic: Performs extrapolation to exact edges
            x0, y0 = p_final_1[0], p_final_1[1]
            vx, vy = p_final_2[0] - x0, p_final_2[1] - y0

            intersections = []
            if vx != 0:
                intersections.append((0, y0 - x0 * vy / vx))
                intersections.append((img_w, y0 + (img_w - x0) * vy / vx))
            if vy != 0:
                intersections.append((x0 - y0 * vx / vy, 0))
                intersections.append((x0 + (img_h - y0) * vx / vy, img_h))

            valid_points = [p for p in intersections if 0 <= p[0] <= img_w and 0 <= p[1] <= img_h]

            if len(valid_points) >= 2:
                max_dist_ext = 0
                ext_p1, ext_p2 = valid_points[0], valid_points[-1]
                for idx_i in range(len(valid_points)):
                    for idx_j in range(idx_i + 1, len(valid_points)):
                        dist = math.hypot(valid_points[idx_i][0] - valid_points[idx_j][0], valid_points[idx_i][1] - valid_points[idx_j][1])
                        if dist > max_dist_ext:
                            max_dist_ext = dist
                            ext_p1, ext_p2 = valid_points[idx_i], valid_points[idx_j]
                
                final_trails.append({
                    'points': [(int(ext_p1[0]), int(ext_p1[1])), (int(ext_p2[0]), int(ext_p2[1]))],
                    'width': width
                })
            else:
                final_trails.append({
                    'points': [(int(p_final_1[0]), int(p_final_1[1])), (int(p_final_2[0]), int(p_final_2[1]))],
                    'width': width
                })

    return final_trails

# --- Trail Management Classes ---
class trail:
    def __init__(self, trail_id):
        self.id = trail_id
        self.points = []
        self.line_width = 30    # Default Trail width in pixels
        self.spline_points = []
        self.color = "#FF0000" # Red for the active trail

    def add_point(self, x, y):
        self.points.append((x, y))
        self.reorder_points()
        self.update_spline()

    def remove_point_at(self, index):
        if 0 <= index < len(self.points):
            del self.points[index]
            self.update_spline()

    def update_spline(self):
        self.spline_points = get_curve_points(self.points)

    def reorder_points(self):
        if len(self.points) < 3:
            return

        # Find the two furthest points
        max_dist = 0
        p1_idx, p2_idx = -1, -1
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                dist = math.hypot(self.points[i][0] - self.points[j][0], self.points[i][1] - self.points[j][1])
                if dist > max_dist:
                    max_dist = dist
                    p1_idx, p2_idx = i, j

        start_point = self.points[p1_idx]
        self.points.sort(key=lambda p: math.hypot(p[0] - start_point[0], p[1] - start_point[1]))

class TrailCollection:
    def __init__(self):
        self.trail = []
        self.active_trail_idx = -1
        self._next_id = 0

    def add_trail(self):
        new_trail = trail(self._next_id)
        self.trail.append(new_trail)
        self._next_id += 1
        self.set_active(len(self.trail) - 1)
        return new_trail

    def remove_trail(self, index):
        if 0 <= index < len(self.trail):
            del self.trail[index]
            if self.active_trail_idx >= index:
                self.active_trail_idx -= 1
            if self.active_trail_idx < 0 and self.trail:
                self.active_trail_idx = 0
            self.update_trail_colors()

    def set_active(self, index):
        """
        Sets the active trail by its index. If index is -1, no trail is active.
        """
        if -1 <= index < len(self.trail): # Allow -1 to mean no active trail
            self.active_trail_idx = index
            self.update_trail_colors()
        else: # Handle out-of-bounds or invalid indices for safety
            self.active_trail_idx = -1
            self.update_trail_colors() # Ensure colors are updated even if invalid index was passed

    def get_active_trail(self):
        if self.active_trail_idx != -1 and self.trail:
            return self.trail[self.active_trail_idx]
        return None

    def update_trail_colors(self):
        # Active Red, Inactive Yellow
        for i, trail in enumerate(self.trail):
            # If no trail is active (self.active_trail_idx is -1), all will become yellow
            trail.color = "#FF0000" if i == self.active_trail_idx else "#FFFF00"

# --- Main Application Class ---
class TrailRemovalAPP:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Satellite Trail Removal Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        # --- Siril Connection ---
        # Initialize Siril connection
        self.siril = None # Initialize to None
        if SIRIL_ENV:
            self.siril = s.SirilInterface()
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                messagebox.showerror("Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
                self.on_closing()
                return

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image is loaded")
            self.on_closing()
            return

        # --- State Variables ---
        self.trail_collection = TrailCollection()
        self.zoom_factor = 1.0
        self.display_photo_image = None
        self.reference_path = None
        self.drag_data = {"x": 0, "y": 0, "item": None, "trail": None, "point_idx": -1}
        self.blur_radius_var = tk.DoubleVar(value=1.0)
        self.blend_strength_var = tk.DoubleVar(value=1.0)
        self.preview_overlay_applied = False  # Flag to track if a preview of overlay has been sent
        self.real_correction_applied = False  # Flag if the user has performed a real correction
        self.image_stats = None  # Contains a dictionary e.g.: {'median': [r,g,b], 'sigma': [r,g,b]}

        # Internal AI_Sensitivity names/IDs
        self.ai_sensitivity_low = "L"
        self.ai_sensitivity_mid = "M"
        self.ai_sensitivity_max = "H"

        # Associate IDs with names to display to the user.
        self.ai_Sensitivity_display_names = {
            self.ai_sensitivity_low: "Low",
            self.ai_sensitivity_mid: "Mid",
            self.ai_sensitivity_max: "Max"
        }

        # Associate tooltips with IDs, not display text.
        self.ai_Sensitivity_tooltips = {
            self.ai_sensitivity_low: "Only strong trails are detected (very safe)",
            self.ai_sensitivity_mid: "Balanced detection with good noise rejection",
            self.ai_sensitivity_max: "Detects even very faint trails (may include noise)"
        }

        self.ui_debug = True
        self.ui_ai_tuning = True
        self.show_canny_debug_var = tk.BooleanVar(value=False)
        self.show_hough_debug_var = tk.BooleanVar(value=False)
        self.show_track_check_var = tk.BooleanVar(value=False)

        # dafault parameters for Canny and Hough is mid value of ai sensitivity
        self.canny_thresh1_debug_var = tk.IntVar(value=10)
        self.canny_thresh2_debug_var = tk.IntVar(value=30)
        self.hough_threshold_debug_var = tk.IntVar(value=90)
        self.hough_min_len_debug_var = tk.IntVar(value=90)
        self.hough_max_gap_debug_var = tk.IntVar(value=30)

        try:
            # Load full-depth data for processing
            self.full_image_data = self.siril.get_image_pixeldata(preview=False)
            if self.full_image_data is None:
                raise SirilError("No image loaded in Siril.")
            
            # Store the original dtype
            self.original_image_dtype = self.full_image_data.dtype
            
            # Convert full_image to HWC if necessary
            if len(self.full_image_data.shape) == 3 and self.full_image_data.shape[0] in [1, 3]:
                self.full_image_data = self.full_image_data.transpose(1, 2, 0)

            self._calculate_image_statistics()
            
            self.siril.log("Generating custom visual preview...", s.LogColor.BLUE)
            preview_data = self.autostretch(self.full_image_data, detection=False)
            
            self.preview_pil_image = Image.fromarray(preview_data)

            # Flip vertically to align with Siril's coordinate system
            self.preview_pil_image = self.preview_pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            # initially the canvas points to the preview
            self.current_canvas_image = self.preview_pil_image

        except (SirilError) as e:
            self.siril.log(f"Error - Cannot start script: {e}", s.LogColor.RED)
            self.root.destroy()
            return

        # --- GUI Setup ---
        self.style = tksiril.standard_style()
        tksiril.match_theme_to_siril(self.root, self.siril)
        self.create_widgets()

        #setting window size And Center the window
        width=1200
        height=800
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)

        # Defer the first fit until the Canvas has valid dimensions
        self.root.after(10, self.fit_to_preview)
        self.bind_events()

    def create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # --- Control Panel (Left) ---
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1) 

        scrollable_left_frame = tksiril.ScrollableFrame(left_frame)
        scrollable_left_frame.pack(fill=tk.BOTH, expand=True)
        scrollable_left_frame.add_mousewheel_binding()

        #ttk.Label(scrollable_left_frame.scrollable_frame, text="(c) Carlo Mollicone - AstroBOH", style="Header.TLabel").pack(pady=10)

        # Trails Frame
        trail_frame = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="Trail Management", padding=5)
        trail_frame.pack(fill=tk.X, padx=5, pady=5)

        # Container frame for Listbox + Scrollbar
        listbox_frame = ttk.Frame(trail_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        scrollbar_listbox_frame = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)

        # Listbox with link to the scrollbar
        self.trail_listbox = tk.Listbox(
            listbox_frame,
            height=5,
            exportselection=0,
            yscrollcommand=scrollbar_listbox_frame.set
        )
        self.trail_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar packed next to it
        scrollbar_listbox_frame.config(command=self.trail_listbox.yview)
        scrollbar_listbox_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.trail_listbox.bind('<<ListboxSelect>>', self.on_trail_select)
        self.trail_listbox.bind('<Button-1>', self.on_listbox_click)

        btn_frame = ttk.Frame(trail_frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add", command=self.add_trail).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Duplicate", command=self.duplicate_trail).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Remove", command=self.remove_trail).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Deselect", command=self.clear_trail_selection).pack(side=tk.LEFT, expand=True, fill=tk.X)

        width_frame = ttk.Frame(trail_frame)
        width_frame.pack(fill=tk.X, padx=2, pady=5)

        ttk.Label(width_frame, text="Line Width (px) :").pack(side=tk.LEFT, padx=5)
        self.line_width_var = tk.IntVar(value=30)  # Default line width
        self.line_width_spinbox = ttk.Spinbox(width_frame, from_=1, to=1000, increment=5, textvariable=self.line_width_var, width=5)
        self.line_width_spinbox.pack(side=tk.LEFT, padx=5, pady=5)

        update_width_btn = ttk.Button(width_frame, text="Update Width", command=self.update_selected_trail_width)
        update_width_btn.pack(side=tk.LEFT, padx=(0, 5))
        tksiril.create_tooltip(update_width_btn, "Apply the current width to the selected trail.")

        # AI Detection
        ai_find_frame = ttk.Frame(trail_frame)
        ai_find_frame.pack(fill=tk.X, padx=2, pady=5)

        find_button = ttk.Button(ai_find_frame, text="Find Trails (AI)", command=self.auto_detect_trails)
        find_button.pack(side=tk.LEFT, padx=(0, 5))
        tksiril.create_tooltip(find_button, "Automatically detect satellite trails using Hough Transform.")

        ai_Sensitivity_frame = ttk.LabelFrame(ai_find_frame, text="AI Sensitivity", padding=5)
        ai_Sensitivity_frame.pack(fill=tk.X, padx=5, pady=5)

        # The control variable will now contain the selected ID ("L", "M", "H")
        self.selected_ai_sensitivity_id = tk.StringVar(self.root)
        # Set the default value using the ID.
        self.selected_ai_sensitivity_id.set(self.ai_sensitivity_mid)

        # Iterate over the dictionary items to access both the ID and the name.
        for sensitivity_id, display_name in self.ai_Sensitivity_display_names.items():
            rb = ttk.Radiobutton(
                ai_Sensitivity_frame,
                text=display_name,                          # Text that sees the user
                variable=self.selected_ai_sensitivity_id,   # Common control variable
                value=sensitivity_id,                       # Value associated with this button (the ID!)
                command=self.update_ai_tuning_parameters
            )
            rb.pack(side="left", expand=True, fill=tk.X, padx=5, pady=5)

            # Retrieve the tooltip using the ID as the key.
            tooltip_text = self.ai_Sensitivity_tooltips.get(sensitivity_id, "Tooltip non trovato")
            tksiril.create_tooltip(rb, tooltip_text)

        # AI Tuning
        if self.ui_ai_tuning:
            tuning_frame = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="AI Detection Tuning", padding=5)
            tuning_frame.pack(fill=tk.X, padx=5, pady=5)
            
            def add_slider(parent, text, var, from_, to, tooltip_text=None):
                f = ttk.Frame(parent); f.pack(fill=tk.X, pady=1)
                ttk.Label(f, text=f"{text}:", width=14, anchor='w').pack(side=tk.LEFT)
                s_val = ttk.Label(f, textvariable=var, width=4); s_val.pack(side=tk.RIGHT, padx=2)
                s_widget = ttk.Scale(f, from_=from_, to=to, variable=var, orient='horizontal')
                s_widget.pack(fill=tk.X, expand=True)
                s_widget.bind("<B1-Motion>", lambda e, v=var, w=s_widget: v.set(int(w.get())))
                s_widget.bind("<ButtonRelease-1>", lambda e, v=var, w=s_widget: v.set(int(w.get())))
                if tooltip_text:
                    tksiril.create_tooltip(s_widget, tooltip_text)

            add_slider(tuning_frame, "Canny Low", self.canny_thresh1_debug_var, 1, 255, "Lower threshold for Canny edge detection. Lower = more sensitive.")
            add_slider(tuning_frame, "Canny High", self.canny_thresh2_debug_var, 1, 255, "Upper threshold for Canny edge detection. Higher = fewer edges.")
            add_slider(tuning_frame, "Hough Thresh", self.hough_threshold_debug_var, 10, 200, "Minimum number of edge points required to detect a line.")
            add_slider(tuning_frame, "Hough MinLen", self.hough_min_len_debug_var, 10, 200, "Minimum length (in pixels) of a detected line segment.")
            add_slider(tuning_frame, "Hough MaxGap", self.hough_max_gap_debug_var, 1, 100, "Maximum allowed gap (in pixels) between line segments to link them.")

            debug_canny_check = ttk.Checkbutton(tuning_frame, text="Show Canny Edges", variable=self.show_canny_debug_var)
            debug_canny_check.pack(side=tk.LEFT, pady=(10,0))
            debug_hough_check = ttk.Checkbutton(tuning_frame, text="Show Hough Lines", variable=self.show_hough_debug_var)
            debug_hough_check.pack(side=tk.LEFT, pady=(10,0))
            # debug_track_check = ttk.Checkbutton(tuning_frame, text="Show Track on image", variable=self.show_track_check_var)
            # debug_track_check.pack(side=tk.LEFT, pady=(10,0))

        self.update_ai_tuning_parameters()

        # Actions Frame
        action_frame = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="Correction Actions", padding=5)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        # Parameters Frame
        params_frame = ttk.Frame(action_frame)
        params_frame.pack(fill=tk.X, padx=2, pady=5)

        ttk.Label(params_frame, text="Parameters for fusion :").pack(side=tk.LEFT, padx=5)
        ttk.Label(params_frame, text="Blur :").pack(side=tk.LEFT, padx=(10, 2))
        blur_Spinbox = ttk.Spinbox(params_frame, from_=0.0, to=10.0, increment=0.1, textvariable=self.blur_radius_var, width=5)
        blur_Spinbox.pack(side=tk.LEFT, padx=5, pady=5)
        tksiril.create_tooltip(blur_Spinbox, "0 = Minimal blur, 10 = Maximum blur.")

        ttk.Label(params_frame, text="Blend :").pack(side=tk.LEFT, padx=(10, 2))
        blend_Spinbox = ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.blend_strength_var, width=5)
        blend_Spinbox.pack(side=tk.LEFT, padx=5, pady=5)
        tksiril.create_tooltip(blend_Spinbox, "0 = Full blend, 1 = No blend.")

        fill_btn_frame = ttk.Frame(action_frame)
        fill_btn_frame.pack(fill="x", padx=2, pady=5)

        btn_Background = ttk.Button(fill_btn_frame, text="Apply (Background)", command=lambda: self.apply_changes(mode="background"))
        btn_Background.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        tksiril.create_tooltip(btn_Background, "Using the DAOPHOT MMM (Mean, Median, Mode) algorithm, apply a synthetic background with the same noise intensity.")

        btn_black = ttk.Button(fill_btn_frame, text="Apply (Black)", command=lambda: self.apply_changes(mode="black"))
        btn_black.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tksiril.create_tooltip(btn_black, "Apply pure black to the image to exclude those pixels from stacking (you can still use blending with a strength lower than 1 to soften the black).")

        ref_button = ttk.Button(fill_btn_frame, text="Select Ref.", command=self.select_reference)
        ref_button.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        tksiril.create_tooltip(ref_button, "Select a 'previous or 'subsequent' frame that does not contain 'satellite' or 'airplane trails'. If the 'tracking' was good, there should be no 'shift' between adjacent frames, therefore the 'pixel transfer' will be perfect.")

        apply_ref_button = ttk.Button(fill_btn_frame, text="Apply (Ref.)", command=self.apply_reference)
        apply_ref_button.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        tksiril.create_tooltip(apply_ref_button, "The pixels from the reference image will be inserted within the region defined by the trace, respecting the Blur and Blend settings.")

        fill_btn_frame.columnconfigure(0, weight=1)
        fill_btn_frame.columnconfigure(1, weight=1)

        # Trail Preview with Overlay Frame
        action_Preview = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="Trail Preview", padding=5)
        action_Preview.pack(fill=tk.X, padx=5, pady=5)

        ref_btn_frame = ttk.Frame(action_Preview)
        ref_btn_frame.pack(fill=tk.X, pady=2)

        preview_trail_button = ttk.Button(ref_btn_frame, text="Preview Trail", command=self.send_preview_overaly_to_siril)
        preview_trail_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tksiril.create_tooltip(preview_trail_button, "Send the track drawing to siril so that we can evaluate its correct positioning.")
        
        clear_overlay_button = ttk.Button(ref_btn_frame, text="Clear Overlay", command=self.clear_preview_to_siril)
        clear_overlay_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tksiril.create_tooltip(clear_overlay_button, "Deletes the overlay sent to siril. Does not delete traces drawn on the canvas.")

        # Separator
        sep = ttk.Separator(scrollable_left_frame.scrollable_frame, orient='horizontal')
        sep.pack(fill=tk.X, pady=5)

        # Instructions frame
        Instructions_frame = ttk.LabelFrame(scrollable_left_frame.scrollable_frame, text="Instructions", padding=10)
        Instructions_frame.pack(fill=tk.X, padx=5, pady=5)
        Instructions_frame.columnconfigure(1, weight=1) # use all space in col 1

        # --- Instructions ---
        instructions_text = (
            "## Workflow and Usage\n\n"
            "Trail removal can be performed on CFA or RGB debayered images.\n"
            "RGB debayered images are very bright (in the presence of large star fields), making automatic detection difficult; it's highly recommended to use a low sensitivity.\n"
            "Another tip for automatic detection: if the stars are elongated, it will most likely fail.\n\n"

            "### Track Management\n"
            " 'Add': Creates a new manual track that becomes 'active' (in red).\n"
            " 'Duplicate': Select a track and click to create an identical copy, useful for working on variations.\n"
            " 'Remove': Select a track from the list to delete it.\n"
            " 'Deselect': Removes the selection from any active tracks.\n\n"
            " 'Line Width (px)': Sets the thickness of the stroke. When a stroke is created by the AI, this value is calculated automatically, but can be changed manually by selecting the stroke and clicking 'Update Width'.\n\n"

            "### Manual Tracking\n"
            "After adding or selecting a trace:\n"
            " 'Add Points': Click on the image to add control points.\n"
            " 'Move Points': Drag an existing point to reposition it.\n"
            " 'Remove Points': Hold down 'Ctrl' and click on a point to remove it.\n"
            "Note: The script uses a 'cardinal spline' to create a smooth curve between points.\n\n"

            "### Automatic Detection (AI)\n"
            " 'Find Trail (AI)': Starts the automatic detection algorithm. **Warning: All existing tracks will be deleted.** The script will analyze the image and automatically create tracks for each detected satellite or aircraft.\n"
            " 'AI Sensitivity': Allows you to choose between three presets (Low, Medium, High) that adjust the algorithm's parameters to accommodate more or less obvious traces.\n"
            " - Low:  Ideal for debayered RGB images.\n"
            " - Medium: A balanced setting for most situations.\n"
            " - High: Attempts to detect even the faintest trail, but may identify false positives. Do not use on RGB images.\n"
            " 'AI Detection Tuning': This section for advanced users allows you to manually adjust key parameters of the Canny (for edges) and Hough (for lines) algorithms, giving you complete control over the detection process. The sensitivity presets automatically adjust these values.\n\n"

            "### Parameters for fusion\n"
            " 'Blur' and 'Blend': Controls the blur and blend strength of the correction mask.\n"

            "### Correction Actions\n"
            "Once you are satisfied with the trail tracing, you can apply the correction:\n"
            " 'Apply (Background)': Fills the trail areas with the 'DAOPHOT MMM (Mean, Median, Mode) algorithm' of the image. Useful for blending the correction with the surrounding star field.\n"
            " 'Apply (Black)': Fills the trail areas with 'black pixels'.\n"
            " 'Select Ref.': Click this button to select a 'reference image' (a FITS file) to use for filling. The reference image will be brightness-balanced with the current image before application.\n"
            " 'Apply (Ref.)': Fills the trail areas using pixels from the 'selected reference image'. The dimensions of the reference image must match the current image in Siril.\n"
            "Important: After clicking one of the 'Apply' buttons, the change will be applied directly to the image in Siril. An 'undo state' will also be saved in Siril ('Trail Removal') to allow you to revert the change if needed.\n\n"

            "### Trail Preview\n"
            " 'Preview Trail': Click this button to display a 'red transparent overlay' of the created mask directly in the Siril interface. This helps you confirm that your trails have been traced correctly before applying the correction.\n"
            " 'Clear Overlay': Removes the preview overlay from Siril.\n\n"

            "### Reload Image from Siril\n"
            " 'Reload Image from Siril': Click this button to reload the image (and its preview) from Siril. This is useful if you've made changes to the image in Siril after opening the script and want the script to work with the latest version. All traced trails in the script will be cleared as the new image might be different.\n\n"

            "### Zoom and Display\n"
            " 'Pan': Click and drag an empty area to move the image.\n"
            " The 'Zoom Out' and 'Zoom In' buttons allow you to zoom in or out of the image.\n"
            " The 'Fit to view' button adjusts the image to fit the canvas window size.\n"
            " You can also use the 'mouse wheel' for zooming (on Windows/macOS) or 'Button-4 / Button-5' (on Linux).\n\n"

            "### Closing the Script\n"
            " Clicking the 'X' button on the window will close the script. If you sent a preview ('Preview Trail') but did not apply a real correction, the preview overlay will be automatically removed from Siril."
        )

        label = ttk.Label(
            Instructions_frame,
            text=instructions_text,
            wraplength=390,
            justify="left",
            anchor="nw",     # in alto a sinistra
            font=("TkDefaultFont", 8)
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 0))
        
        # --- Image Panel (Right) ---
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=7)

        zoom_frame = ttk.Frame(right_frame)
        zoom_frame.pack(fill=tk.X)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Fit to view", command=self.fit_to_preview).pack(side=tk.LEFT)

        # Reload Image from Siril
        ttk.Button(zoom_frame, text="Reload Image from Siril", command=self.load_new_image_from_siril).pack(side=tk.RIGHT, padx=(0, 20))

        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Control-Button-1>", self.on_ctrl_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)   # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)     # Linux up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)     # Linux down
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _calculate_image_statistics(self):
        """
        Calculates and stores statistics (median, sigma) for the loaded image.
        It handles both color and monochrome images.
        """
        self.siril.log("Calculating initial image statistics...", s.LogColor.BLUE)
        if self.full_image_data is None:
            self.image_stats = None
            return

        data_for_stats = self.full_image_data.astype(np.float32)

        bkg_estimator = MMMBackground()
        bkg_rms = StdBackgroundRMS()

        # Check if the image is in color (RGB)
        if data_for_stats.ndim == 3 and data_for_stats.shape[2] == 3:
            # Calculates median and sigma for each channel and stores them
            medians = [bkg_estimator(data_for_stats[..., c]) for c in range(3)]
            sigmas = [bkg_rms(data_for_stats[..., c]) for c in range(3)]
            self.image_stats = {'median': np.array(medians), 'sigma': np.array(sigmas)}
        else:
            # The image is monochrome
            median = bkg_estimator(data_for_stats)
            sigma = bkg_rms(data_for_stats)
            self.image_stats = {'median': median, 'sigma': sigma}
        
        stats = self.image_stats
        formatted_stats = "\n\nImage statistics calculated:\n"

        def format_array(arr: np.ndarray, decimals: int = 8):
            return '[' + ', '.join(f"{x:.{decimals}g}" for x in arr) + ']'

        if isinstance(stats['median'], np.ndarray):
            formatted_stats += f"median: array{format_array(stats['median'])}, dtype={stats['median'].dtype}\n"
            formatted_stats += f"sigma:  array{format_array(stats['sigma'])}, dtype={stats['sigma'].dtype}"
        else:
            formatted_stats += f"median: {stats['median']:.8g}\n"
            formatted_stats += f"sigma:  {stats['sigma']:.8g}"

        self.siril.log(formatted_stats + "\n", s.LogColor.GREEN)

    def autostretch(self, image_data, detection=False):
        """
        Performs a robust autostretch on the linear image to create
        a balanced visual preview. For color images, it performs
        an unlinked stretch to maintain proper color balance.

        Performs robust autostretch.
        - If detection=False, creates a balanced visual preview.
        - If detection=True, performs advanced preprocessing on the linear data to isolate satellite tracks before returning an 8-bit image.
        """
        # Work on float data for calculations
        data = image_data.copy().astype(np.float32)

        # --- Detection Section ---
        if detection:
            if data.ndim == 3 and data.shape[2] == 3:   # Color image (RGB)
                stretched_channels = []
                for i in range(3):  # Iterate over R, G, B channels
                    channel = data[..., i]

                    median = self.image_stats['median'][i]
                    sigma = self.image_stats['sigma'][i]

                    black_point = median + 2.0 * sigma
                    white_point = median + 2.5 * sigma

                    channel = np.clip(channel, black_point, white_point)

                    median_filtered = cv2.medianBlur(channel, 5)

                    stretched_channel = cv2.normalize(median_filtered, None, 0, 255, cv2.NORM_MINMAX)
                    stretched_channels.append(stretched_channel)

                # Recombines channels into a temporary color image
                color_image = cv2.merge(stretched_channels).astype(np.uint8)
                # Convert the final color image to black and white (grayscale)
                stretched_8bit = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
                
            else:   # Monochrome image
                # Apply stretch to monochrome image
                median = self.image_stats['median']
                sigma = self.image_stats['sigma']
                
                black_point = median + 0.5 * sigma
                white_point = median + 2.5 * sigma
                
                data = np.clip(data, black_point, white_point)
                
                median_filtered = cv2.medianBlur(data, 5)
                
                stretched_8bit = cv2.normalize(median_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            return stretched_8bit

        # --- Visual Preview Section ---
        else:
            if data.ndim == 3 and data.shape[2] == 3:   # Color image (RGB)
                stretched_channels = []
                for i in range(3):  # Iterate over R, G, B channels
                    channel = data[..., i]

                    # Compute robust sky background statistics using INDEPENDENT photutils for each channel
                    # Why It's an Improvement
                    # Robustness against Outliers: The current method, np.median(data), calculates the median over the entire image. If the image contains large nebulae or bright galaxies,
                    # these objects will heavily influence the median value, distorting the estimate of the "true" sky background. The photutils library is specifically designed
                    # for astrophotography, and its algorithms are much more robust, as they are able to ignore (or "clip") outliers such as stars and other bright objects during the calculation.

                    # More Accurate Noise Estimation: As a result, StdBackgroundRMS provides a much more accurate estimate of the standard deviation (sigma) than background noise alone.
                    # The manual method based on MAD (Median Absolute Deviation) is good, but less precise when applied globally to a complex astronomical image.

                    # Better Contrast for Detection: The goal of the autostretch function is to maximize the contrast of trails.
                    # Using a more precise median and sigma that are more representative of the sky background, the black and white points of the stretch will be set much more effectively.
                    # This will make the faint trails "emerge" better from the background, improving the effectiveness of the Canny and Hough algorithms that follow.

                    median = self.image_stats['median'][i]
                    sigma = self.image_stats['sigma'][i]

                    # Defines the SPECIFIC stretch points for this channel
                    # Black point: slightly below the median for a dark but unclipped sky background
                    black_point = median - 1.5 * sigma
                    # White point: High enough to reveal faint details,
                    # but not so harsh as to completely burn out the stars. A value between 5 and 10 sigma is a good compromise.
                    white_point = median + 3.0 * sigma

                    # Apply the stretch to a single channel
                    # "Clipping" the values to isolate the range of interest
                    channel = np.clip(channel, black_point, white_point)
                    # Normalizes the channel to use the entire 0-255 range
                    stretched_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
                    stretched_channels.append(stretched_channel)

                stretched_8bit = cv2.merge(stretched_channels).astype(np.uint8)
                
            else:   # Monochrome image
                # Apply stretch to monochrome image
                median = self.image_stats['median']
                sigma = self.image_stats['sigma']
                black_point = median - 1.5 * sigma
                white_point = median + 3.0 * sigma
                data = np.clip(data, black_point, white_point)
                stretched_8bit = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            return stretched_8bit

    def update_ai_tuning_parameters(self):
        """
        Updates the Canny threshold variables based on the selected AI sensitivity.
        This function is called whenever a sensitivity radio button is clicked.
        """
        selected_id = self.selected_ai_sensitivity_id.get()

        # NOTE: Set your desired Canny threshold values here for each level.
        # A higher threshold means the algorithm is LESS sensitive.
        if selected_id == self.ai_sensitivity_low:
            # "Low" sensitivity = Higher Canny thresholds (detects only strong edges)
            self.canny_thresh1_debug_var.set(50)
            self.canny_thresh2_debug_var.set(150)

            self.hough_threshold_debug_var.set(90)
            self.hough_min_len_debug_var.set(90)
            self.hough_max_gap_debug_var.set(10)

        elif selected_id == self.ai_sensitivity_mid:
            # "Mid" sensitivity = Balanced thresholds
            self.canny_thresh1_debug_var.set(20)
            self.canny_thresh2_debug_var.set(60)

            self.hough_threshold_debug_var.set(70)
            self.hough_min_len_debug_var.set(50)
            self.hough_max_gap_debug_var.set(5)

        elif selected_id == self.ai_sensitivity_max:
            # "Max" sensitivity = Lower Canny thresholds (detects even faint edges)
            self.canny_thresh1_debug_var.set(5)
            self.canny_thresh2_debug_var.set(15)

            self.hough_threshold_debug_var.set(70)
            self.hough_min_len_debug_var.set(50)
            self.hough_max_gap_debug_var.set(15)

    def load_new_image_from_siril(self):
        """
        Reloads the full-resolution and preview image data from Siril,
        updates the display, and clears existing trails.
        """
        if not SIRIL_ENV or not self.siril:
            messagebox.showerror("Error", "Siril is not connected or the environment is not Siril.")
            return

        try:
            self.siril.log("Reloading image from Siril...", s.LogColor.BLUE)

            # Update full-resolution image data
            new_full_image_data = self.siril.get_image_pixeldata(preview=False)
            if new_full_image_data is None:
                self.siril.log("No image loaded in Siril or unable to retrieve full-resolution data.", s.LogColor.RED)

            # Update the original dtype
            self.original_image_dtype = new_full_image_data.dtype
            
            # Ensure HWC format if necessary (channels last)
            if len(new_full_image_data.shape) == 3 and new_full_image_data.shape[0] in [1, 3]:
                self.full_image_data = new_full_image_data.transpose(1, 2, 0)
            else:
                self.full_image_data = new_full_image_data

            # Recalculate statistics for the NEW image before using them.
            self._calculate_image_statistics()

            # Generate the new preview using our custom autostretch logic
            self.siril.log("Generating new custom visual preview...", s.LogColor.BLUE)
            new_preview_data = self.autostretch(self.full_image_data, detection=False)
            
            self.preview_pil_image = Image.fromarray(new_preview_data)

            # Flip vertically to align with Siril's coordinate system
            self.preview_pil_image = self.preview_pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            self.update_canvas_image(self.preview_pil_image)

            # Clear existing trails (as the image might have changed significantly)
            # This is generally a good idea when reloading an image to avoid misaligned trails.
            self.trail_collection = TrailCollection()
            self.trail_listbox.delete(0, tk.END)
            self.siril.overlay_clear_polygons()

            # Refresh the canvas display and reset zoom
            self.fit_to_preview()
            self.siril.log("Image reloaded and display updated.", s.LogColor.GREEN)

        except (SirilError, Exception) as e:
            self.siril.log(f"Error while reloading image: {e}", s.LogColor.RED)
            messagebox.showerror("Image Reload Error", f"An error occurred while reloading the image: {e}")

    # --- Trail Management Methods ---
    def clear_trail_selection(self):
        """
        Clears the selection in the Listbox and deselects the active trail
        in the TrailCollection.
        """
        # Visually clear Listbox selection
        self.trail_listbox.selection_clear(0, tk.END)# Clear all Listbox selections
        self.trail_collection.set_active(-1)         # Set active trail to none in your collection
        self.redraw_canvas_overlays()                # Redraw canvas to update trail colors (all become inactive color)

    def add_trail(self):
        trail = self.trail_collection.add_trail()
        # La nuova traccia ottiene la larghezza corrente dallo spinbox
        trail.line_width = self.line_width_var.get()
        self.trail_listbox.insert(tk.END, f"Trail {trail.id}")
        self.trail_listbox.selection_clear(0, tk.END)
        self.trail_listbox.selection_set(tk.END)
        # Aggiorniamo la visualizzazione
        self.redraw_canvas_overlays()

    def duplicate_trail(self):
        """
        Duplicates the currently selected trail, creating a new, independent trail
        with the same points and width.
        """
        # Get the source trail (the currently active one)
        source_trail = self.trail_collection.get_active_trail()

        # Check if a trail is actually selected
        if not source_trail:
            self.siril.log("No trail selected to duplicate.", s.LogColor.RED)
            return

        # Use the existing method to create a new trail.
        # This already sets it as the active trail.
        new_trail = self.trail_collection.add_trail()

        # Copy the important properties from the source trail to the new one
        # It is FUNDAMENTAL to use .copy() to create a new independent list of points.
        new_trail.points = source_trail.points.copy()
        new_trail.line_width = source_trail.line_width

        # Update the spline for the new trail based on the newly copied points
        new_trail.update_spline()

        # Update the Listbox in the user interface
        self.trail_listbox.insert(tk.END, f"Trail {new_trail.id} (copy)")
        self.trail_listbox.selection_clear(0, tk.END)
        self.trail_listbox.selection_set(tk.END) # Select the new duplicated trail

        # Redraw the canvas to show the new trail
        self.redraw_canvas_overlays()

        self.siril.log(f"Duplicated Trail {source_trail.id} to new Trail {new_trail.id}.", s.LogColor.BLUE)

    def update_selected_trail_width(self):
        """
        Applies the value from the line width spinbox to the currently selected trail.
        """
        active_trail = self.trail_collection.get_active_trail()
        if not active_trail:
            self.siril.log("No trail selected to update.", s.LogColor.RED)
            return

        new_width = self.line_width_var.get()
        active_trail.line_width = new_width
        self.redraw_canvas_overlays()
        self.siril.log(f"Trail {active_trail.id} width updated to {new_width}px.", s.LogColor.BLUE)

    def remove_trail(self):
        selected_indices = self.trail_listbox.curselection()
        if not selected_indices:
            self.siril.log(f"No trail selected.", s.LogColor.RED)
            return

        active_trail = self.trail_collection.get_active_trail()
        try:
            idx = selected_indices[0]
            self.trail_collection.remove_trail(idx)
            self.trail_listbox.delete(idx)
            self.siril.log(f"Trail {active_trail.id} deleted", s.LogColor.RED)

            if self.trail_collection.trail:
                new_selection = min(idx, len(self.trail_collection.trail) - 1)
                self.trail_listbox.selection_set(new_selection)
                self.trail_collection.set_active(new_selection)

            self.redraw_canvas_overlays()
        except Exception as e:
            self.siril.log(f"Error: {e}", s.LogColor.RED)

    def on_listbox_click(self, event):
        """
        Handles clicks on the listbox: if an empty area is clicked, deselect everything.
        """
        index = self.trail_listbox.nearest(event.y)
        bbox = self.trail_listbox.bbox(index)
        if bbox:
            x, y, w, h = bbox
            if not (y <= event.y <= y + h):
                self.clear_trail_selection()
        else:
            self.clear_trail_selection()

    def on_trail_select(self, event):
        selected_indices = self.trail_listbox.curselection()
        if not selected_indices:
            # No selection in the Listbox. Call set_active with -1.
            self.trail_collection.set_active(-1)
        else:
            # An item is selected. Pass its index to set_active.
            idx = selected_indices[0]
            self.trail_collection.set_active(idx)
            
            # Always update UI and redraw after selection changes
            active_trail = self.trail_collection.get_active_trail()
            if active_trail:
                # Ensures spinbox reflects active trail's width or a default
                self.line_width_var.set(active_trail.line_width)

        # Update the trail colors on the canvas (red/yellow)
        self.redraw_canvas_overlays()

    def auto_detect_trails(self):
        """
        Performs automatic detection of trails and adds them to the list.
        """
        self.siril.log("Starting automatic trail detection...", s.LogColor.BLUE)
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        # Clear existing trails (as the image might have changed significantly)
        # This is generally a good idea when reloading an image to avoid misaligned trails.
        self.trail_collection = TrailCollection()
        self.trail_listbox.delete(0, tk.END) # Clear listbox display

        # We use the full-depth data and apply autostretch
        self.siril.log("Performing custom autostretch for detection...", s.LogColor.BLUE)
        # The autostretch function handles both mono and color, returning an 8-bit image
        stretched_image_for_detection = self.autostretch(self.full_image_data, detection=True)
        
        if self.ui_ai_tuning:
            # --- High-Reliability Detection in DEBUG MODE ---
            # If debug mode is active, we use the parameters from the UI controls
            self.siril.log("Pass 1: Reads parameters from the UI controls DEBUG MODE", s.LogColor.BLUE)
            hough_params = {
                'canny_low': self.canny_thresh1_debug_var.get(),
                'canny_high': self.canny_thresh2_debug_var.get(),
                'hough': {
                    'threshold': self.hough_threshold_debug_var.get(),
                    'minLineLength': self.hough_min_len_debug_var.get(),
                    'maxLineGap': self.hough_max_gap_debug_var.get()
                }
            }
        else:
            # --- High-Reliability Detection ---
            self.siril.log("Pass 1: Detecting high-confidence trails...", s.LogColor.BLUE)
            hough_params = {
                'canny_low': 10,
                'canny_high': 30,
                'hough': {
                    'threshold': 90,
                    'minLineLength': 90,
                    'maxLineGap': 20
                }
            }

        self.siril.log(f"Using detection parameters: {hough_params}", s.LogColor.BLUE)

        # Execute detection
        hough_lines, edges_image  = find_trails_with_hough(stretched_image_for_detection, hough_params)

        # If debug mode is active, display the Canny edge map and stop
        if self.show_canny_debug_var.get():
            self.canvas.delete("overlay")

            self.siril.log("Debug mode: Displaying Canny edge map.", s.LogColor.BLUE)
            # The preview image is flipped, so the debug image must also be flipped for correct display
            debug_canny_image = Image.fromarray(edges_image).transpose(Image.FLIP_TOP_BOTTOM)
            self.update_canvas_image(debug_canny_image)
            
            if self.show_hough_debug_var.get() == False and self.show_track_check_var.get() == False:
                self.root.config(cursor="")
                return
            
        if self.show_hough_debug_var.get():
            self.canvas.delete("overlay")
            self.siril.log(f"Debug mode: Drawing {len(hough_lines)} raw Hough lines for debug.", s.LogColor.BLUE)

            debug_hough_image = self.current_canvas_image.copy().convert("RGB")
            draw = ImageDraw.Draw(debug_hough_image)
            for x1, y1, x2, y2 in hough_lines:
                # --- CORRECTION: Invert the Y-axis to align with the flipped image ---
                y1_flipped = self.preview_pil_image.height - y1
                y2_flipped = self.preview_pil_image.height - y2
                draw.line([(x1, y1_flipped), (x2, y2_flipped)], fill="lime", width=2)
        
            self.update_canvas_image(debug_hough_image)
            
            if self.show_track_check_var.get() == False:
                self.root.config(cursor="")
                return

        # sempre visualizzazione normale
        self.update_canvas_image(self.preview_pil_image)
            
        if not hough_lines:
            self.siril.log("No high-confidence trails found. Detection stopped.", s.LogColor.RED)
            self.root.config(cursor="")
            return

        self.siril.log(f"Found {len(hough_lines)} high-confidence segments.", s.LogColor.BLUE)

        # --- Final Fusion ---
        # Merge the reliable lines with the validated weak fragments
        final_line_candidates = hough_lines
        self.siril.log(f"Processing {len(final_line_candidates)} total segments...", s.LogColor.BLUE)
        
        processed_trails = process_detected_lines(final_line_candidates, stretched_image_for_detection.shape)

        if not processed_trails:
            self.siril.log("No consolidated trails found after processing.", s.LogColor.RED)
            self.root.config(cursor="")
            return

        self.siril.log(f"Reconstructed {len(processed_trails)} full trails.", s.LogColor.GREEN)
        
        for trail_data in processed_trails:
            new_trail = self.trail_collection.add_trail()
            self.trail_listbox.insert(tk.END, f"Trail {new_trail.id} (AI)")
            
            p1, p2 = trail_data['points']
            
            # Invert Y with respect to the image height
            p1_inverted_y = self.preview_pil_image.height - p1[1]
            p2_inverted_y = self.preview_pil_image.height - p2[1]

            new_trail.add_point(p1[0], p1_inverted_y)
            new_trail.add_point(p2[0], p2_inverted_y)
            
            # Set the dynamically calculated width on the trail
            new_trail.line_width = trail_data['width']
        
        # Update UI
        self.trail_listbox.selection_clear(0, tk.END)
        self.trail_listbox.selection_set(tk.END)
        #self.trail_collection.set_active(len(self.trail_collection.trail) - 1)
        self.redraw_canvas_overlays()
        self.clear_trail_selection()
        
        self.root.config(cursor="")
        self.siril.log(f"Detection Complete: Reconstructed {len(processed_trails)} trails.", s.LogColor.BLUE)

    # --- Canvas Interaction Methods ---
    def on_press(self, event):
        # Use canvasx/canvasy to get the correct coordinates INSIDE the canvas,
        # which take panning into account.
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Prepare for dragging a point
        active_trail = self.trail_collection.get_active_trail()
        if active_trail:
            for i, (px, py) in enumerate(active_trail.points):
                point_canvas_x = px * self.zoom_factor
                point_canvas_y = py * self.zoom_factor
                
                # The tolerance is applied in the canvas space
                if math.hypot(point_canvas_x - canvas_x, point_canvas_y - canvas_y) < 10:
                    self.drag_data = {'item': "point", 'trail': active_trail, 'point_idx': i}
                    return

        # Store the initial click coordinates to distinguish a click from a drag
        self.drag_data = {'item': "canvas", 'start_x': event.x, 'start_y': event.y}
        self.canvas.scan_mark(event.x, event.y)

    def on_drag(self, event):
        if self.drag_data.get('item') == "point":
            # Convert the drag coordinates into correct image coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            img_x = canvas_x / self.zoom_factor
            img_y = canvas_y / self.zoom_factor

            trail = self.drag_data['trail']
            trail.points[self.drag_data['point_idx']] = (img_x, img_y)
            trail.update_spline()
            self.redraw_canvas_overlays()
        elif self.drag_data.get('item') == 'canvas':
            # Perform the pan
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_release(self, event):
        if self.drag_data.get('item') == 'point':
            trail = self.drag_data['trail']
            if trail:
                trail.reorder_points()
                trail.update_spline()
                self.redraw_canvas_overlays()
                
        elif self.drag_data.get('item') == 'canvas':
            # Calcola la distanza percorsa dal mouse
            start_x = self.drag_data['start_x']
            start_y = self.drag_data['start_y']
            end_x = event.x
            end_y = event.y
            distance = math.hypot(end_x - start_x, end_y - start_y)

            # Se la distanza è minima, consideralo un CLICK, altrimenti era un DRAG (pan)
            if distance < 5:  # Soglia in pixel per considerare l'azione un click
                # Only add a point if no panning occurred.
                active_trail = self.trail_collection.get_active_trail()
                if active_trail:
                    # Use canvasx/canvasy to find the correct point to add the anchor
                    canvas_x = self.canvas.canvasx(event.x)
                    canvas_y = self.canvas.canvasy(event.y)
                    img_x = canvas_x / self.zoom_factor
                    img_y = canvas_y / self.zoom_factor

                    # Check that the click is within the image
                    # if 0 <= img_x <= self.preview_pil_image.width and 0 <= img_y <= self.preview_pil_image.height:
                    #     active_trail.add_point(img_x, img_y)
                    #     self.redraw_canvas_overlays()
                    
                    active_trail.add_point(img_x, img_y)
                    self.redraw_canvas_overlays()

        # Resetta i dati di trascinamento
        self.drag_data = {}

    def on_ctrl_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        active_trail = self.trail_collection.get_active_trail()
        if active_trail:
            for i, (px, py) in reversed(list(enumerate(active_trail.points))):
                point_canvas_x = px * self.zoom_factor
                point_canvas_y = py * self.zoom_factor
                if math.hypot(point_canvas_x - canvas_x, point_canvas_y - canvas_y) < 10:
                    active_trail.remove_point_at(i)
                    self.redraw_canvas_overlays()
                    return

    # --- Display and Zoom Methods ---
    def update_canvas_image(self, image_to_display=None):
        # If no specific image is passed, use the default one
        if image_to_display is None:
            image_to_display = self.current_canvas_image
        else:
            self.current_canvas_image = image_to_display

        w = int(image_to_display.width * self.zoom_factor)
        h = int(image_to_display.height * self.zoom_factor)
        if w <= 0 or h <= 0:
            return

        zoomed_image = image_to_display.resize((w, h), Image.LANCZOS)
        self.display_photo_image = ImageTk.PhotoImage(zoomed_image)

        # Draw the image at the origin (0,0) of the canvas
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_photo_image, tags="image")
        
        # Add a large margin (padding) to the scroll region to allow panning beyond the edges.
        # We provide a margin equal to the size of the image on each side.
        padding_x = w 
        padding_y = h
        self.canvas.config(scrollregion=(-padding_x, -padding_y, w + padding_x, h + padding_y))
        
        if self.show_canny_debug_var.get() == False:
            # If not in debug mode, redraw the overlays
            self.redraw_canvas_overlays()

    def redraw_canvas_overlays(self):
        self.canvas.delete("overlay")
        for trail in self.trail_collection.trail:
            # Draw the spline directly in the zoomed canvas coordinates
            if len(trail.spline_points) > 1:
                scaled_spline = [(p[0] * self.zoom_factor, p[1] * self.zoom_factor) for p in trail.spline_points]
                visible_width = max(1, int(trail.line_width * self.zoom_factor))

                dash_a = min(visible_width * 10, 255)
                dash_b = min(visible_width * 2, 255)

                self.canvas.create_line(scaled_spline, fill=trail.color, width=visible_width, tags="overlay", capstyle=tk.BUTT, dash=(dash_a, dash_b))

            # Draw the points
            for px, py in trail.points:
                cx = px * self.zoom_factor
                cy = py * self.zoom_factor
                x0, y0 = cx - 5, cy - 5
                x1, y1 = cx + 5, cy + 5
                self.canvas.create_oval(x0, y0, x1, y1, fill=trail.color, outline='white', tags="overlay")

    def _perform_zoom(self, factor, anchor_x, anchor_y):
        """
        Central function that performs the zoom anchored to a specific point,
        correctly managing the scrollregion with padding.
        """
        old_zoom = self.zoom_factor
        new_zoom = old_zoom * factor
        
        if not (0.05 < new_zoom < 50.0):
            return

        # Coordinates of the point on the virtual canvas BEFORE zooming
        canvas_x_before = self.canvas.canvasx(anchor_x)
        canvas_y_before = self.canvas.canvasy(anchor_y)

        # Coordinates of the pixel on the original (non-zoomed) image
        img_x = canvas_x_before / old_zoom
        img_y = canvas_y_before / old_zoom

        # Apply the new zoom factor and redraw the canvas and scrollregion
        self.zoom_factor = new_zoom
        self.update_canvas_image()

        # Calculate the new position of our anchor point on the virtual canvas
        new_canvas_x = img_x * self.zoom_factor
        new_canvas_y = img_y * self.zoom_factor

        # Calculate the new position of the top-left corner of the view to keep the anchor point under the cursor.
        target_view_x = new_canvas_x - anchor_x
        target_view_y = new_canvas_y - anchor_y

        # Convert the absolute coordinates into fractions for xview_moveto/yview_moveto, taking padding into account.
        w_zoomed = int(self.preview_pil_image.width * self.zoom_factor)
        h_zoomed = int(self.preview_pil_image.height * self.zoom_factor)
        padding_x = w_zoomed
        padding_y = h_zoomed
        
        total_scroll_width = w_zoomed + 2 * padding_x
        total_scroll_height = h_zoomed + 2 * padding_y

        if total_scroll_width > 0:
            fraction_x = (target_view_x + padding_x) / total_scroll_width
            self.canvas.xview_moveto(fraction_x)
        
        if total_scroll_height > 0:
            fraction_y = (target_view_y + padding_y) / total_scroll_height
            self.canvas.yview_moveto(fraction_y)

    def on_mouse_wheel(self, event):
        zoom_factor_step = 1.2 # Passo di zoom più graduale
        
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = zoom_factor_step
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1 / zoom_factor_step
        else:
            return
            
        self._perform_zoom(factor, event.x, event.y)

    def zoom_in(self):
        anchor_x = self.canvas.winfo_width() / 2
        anchor_y = self.canvas.winfo_height() / 2
        self._perform_zoom(1.3, anchor_x, anchor_y)     # A little bigger for the buttons

    def zoom_out(self):
        anchor_x = self.canvas.winfo_width() / 2
        anchor_y = self.canvas.winfo_height() / 2
        self._perform_zoom(1 / 1.3, anchor_x, anchor_y) # A little bigger for the buttons

    def fit_to_preview(self):
        self.root.update_idletasks()
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_w, img_h = self.preview_pil_image.size
        
        if img_w > 0 and img_h > 0:
            # Calculate the zoom factor to fit the image within the canvas
            self.zoom_factor = min(canvas_w / img_w, canvas_h / img_h)
            
            # Update the image on the canvas with the new zoom factor
            self.update_canvas_image()

            # Calculate the total padded scrollregion dimensions
            padded_img_w = int(self.preview_pil_image.width * self.zoom_factor)
            padded_img_h = int(self.preview_pil_image.height * self.zoom_factor)
            
            # The scrollregion starts at (-padding_x, -padding_y) where padding_x = padded_img_w
            # So, the actual image starts at (padded_img_w, padded_img_h) within the scrollregion's coordinates (0,0) based
            
            # Calculate the offset needed to center the *image* within the *canvas*
            # The image width on canvas is padded_img_w, image height on canvas is padded_img_h
            # The canvas width is canvas_w, canvas height is canvas_h
            
            # Calculate the top-left corner of where the image should be placed
            # within the canvas to be centered.
            center_offset_x = (canvas_w - padded_img_w) / 2
            center_offset_y = (canvas_h - padded_img_h) / 2

            # Adjust the canvas view. The xview_moveto/yview_moveto arguments are fractions of the total scrollregion.
            # To center the image, we need to move the scrollbar such that the top-left of the image
            # (which is at (padded_img_w, padded_img_h) in scrollregion coords relative to its start)
            # is shifted by 'center_offset_x' and 'center_offset_y'.
            
            # The total width of the scrollregion is w + 2*padding_x = 3*padded_img_w
            # The total height of the scrollregion is h + 2*padding_y = 3*padded_img_h

            # We want the image's top-left corner (at padded_img_w, padded_img_h in scrollregion coords)
            # to be visible at center_offset_x, center_offset_y on the canvas.
            
            # The target scroll_x is the position in the scrollregion that should align with the canvas's (0,0).
            # So, we want (padded_img_w - center_offset_x) to be at the canvas's (0,0).
            # And (padded_img_h - center_offset_y) for y.

            target_scroll_x = (padded_img_w - center_offset_x) 
            target_scroll_y = (padded_img_h - center_offset_y) 

            # Normalize to a fraction of the total scrollregion width/height.
            # The total width of scrollregion is w + 2*padding_x (which is 3*w_zoomed)
            # The total height of scrollregion is h + 2*padding_y (which is 3*h_zoomed)
            total_scroll_width = padded_img_w + 2 * padded_img_w # This is 3 * padded_img_w
            total_scroll_height = padded_img_h + 2 * padded_img_h # This is 3 * padded_img_h

            if total_scroll_width > 0:
                self.canvas.xview_moveto(target_scroll_x / total_scroll_width)
            if total_scroll_height > 0:
                self.canvas.yview_moveto(target_scroll_y / total_scroll_height)

    def clear_preview_to_siril(self):
        # Remove previous overlays for a clean preview
        self.siril.overlay_clear_polygons()

    def send_preview_overaly_to_siril(self):
        valid_trails = [t for t in self.trail_collection.trail if len(t.points) >= 2]
        if not valid_trails:
            self.siril.log("Warning - No valid trails (with at least 2 points) for preview.", s.LogColor.RED)
            return

        # Remove previous overlays for a clean preview
        self.siril.overlay_clear_polygons()

        # Calculate scale factors and get total image height
        img_h, img_w = self.full_image_data.shape[:2]
        prev_w, prev_h = self.preview_pil_image.size

        scale_x = img_w / prev_w
        scale_y = img_h / prev_h

        for trail in valid_trails:
            if not trail.spline_points:
                continue

            # Apply scaling to spline points to map them to the real image
            scaled_spline_points = [(p[0] * scale_x, p[1] * scale_y) for p in trail.spline_points]

            # Calculate line width in real image pixels
            scaled_width = max(1, int(trail.line_width * (scale_x + scale_y) / 2))

            # Create a list of FPoint objects for the polygon outline
            poly_points_list = create_thick_line_polygon(scaled_spline_points, scaled_width)

            # Invert the Y coordinate of each point to align with Siril's system.
            #flipped_poly_points = [FPoint(x=p.x, y=img_h - p.y) for p in poly_points_list]

            if len(poly_points_list) >= 3:  # if you wanted to use inverted coordinates, replace with: flipped_poly_points
                # Pack the RGBA color into a single integer
                packed_color = pack_rgba_color(255, 0, 0, 128) # Semi-transparent red

                # Create the sirilpy Polygon object using points with the correct Y
                polygon_overlay = s.models.Polygon(
                    points=poly_points_list, # if you wanted to use inverted coordinates, replace with: flipped_poly_points
                    color=packed_color,
                    fill=False
                )
                self.siril.overlay_add_polygon(polygon_overlay)

        self.preview_overlay_applied = True
        self.siril.log("Trail preview sent to Siril overlay.", s.LogColor.GREEN)

    def select_reference(self):
        path = filedialog.askopenfilename(title="Select Reference Image", filetypes=[("FITS Images", "*.fit *.fits")])
        if path:
            self.reference_path = path
            self.siril.log(f"Reference Selected - Selected: {os.path.basename(path)}", s.LogColor.BLUE)

    def apply_reference(self):
        if not self.reference_path:
            self.siril.log("Warning - No reference image selected.", s.LogColor.RED)
            return
        self.apply_changes(mode="reference")

    def apply_changes(self, mode):
        valid_trails = [t for t in self.trail_collection.trail if len(t.points) >= 2]
        if not valid_trails:
            self.siril.log("Warning - No valid trails (with at least 2 points) to apply.", s.LogColor.RED)
            return

        if self.siril.is_image_loaded():
            try:
                self.siril.update_progress("Process started.", 0.0)
                self.siril.overlay_clear_polygons()

                # Create mask
                img_h, img_w = self.full_image_data.shape[:2]
                prev_w, prev_h = self.preview_pil_image.size

                # Scale trail coordinates from preview to real size before creating the mask
                scale_x = img_w / prev_w
                scale_y = img_h / prev_h

                # Create empty mask
                mask = Image.new('L', (img_w, img_h), 0)
                draw = ImageDraw.Draw(mask)
                
                for trail in valid_trails:
                    # Apply scaling and a vertical flip to the points, bringing the Y back to the correct system for the full-resolution image in Siril
                    scaled_points = [(x * scale_x, img_h - (y * scale_y)) for (x, y) in trail.spline_points]
                    scaled_width = max(1, int(trail.line_width * (scale_x + scale_y) / 2))
                    
                    # Dynamically extending mask endpoints
                    border_threshold = 5.0  # Pixel tolerance for considering a point "on edge"
                    extension_pixels = 100.0 # How many pixels to extend the line beyond the edge?

                    # Create an editable copy of the point list
                    extended_scaled_points = list(scaled_points)

                    if len(extended_scaled_points) >= 2:
                        # --- Check the starting point of the track ---
                        p0 = extended_scaled_points[0]
                        is_on_border_start = (p0[0] < border_threshold or 
                                            p0[0] > img_w - border_threshold or 
                                            p0[1] < border_threshold or 
                                            p0[1] > img_h - border_threshold)

                        if is_on_border_start:
                            p1 = extended_scaled_points[1]
                            # Calculate the direction vector (from point 0 to point 1)
                            dx = p1[0] - p0[0]
                            dy = p1[1] - p0[1]
                            length = math.hypot(dx, dy)
                            if length > 0:
                                # Move point p0 backwards along the vector
                                new_x = p0[0] - (dx / length) * extension_pixels
                                new_y = p0[1] - (dy / length) * extension_pixels
                                extended_scaled_points[0] = (new_x, new_y)

                        # --- Check the end point of the track ---
                        pN = extended_scaled_points[-1]
                        is_on_border_end = (pN[0] < border_threshold or 
                                            pN[0] > img_w - border_threshold or 
                                            pN[1] < border_threshold or 
                                            pN[1] > img_h - border_threshold)
                        
                        if is_on_border_end:
                            p_prev = extended_scaled_points[-2]
                            # Calculate the direction vector (from the penultimate to the last point)
                            dx = pN[0] - p_prev[0]
                            dy = pN[1] - p_prev[1]
                            length = math.hypot(dx, dy)
                            if length > 0:
                                # Move the point pN forward along the vector
                                new_x = pN[0] + (dx / length) * extension_pixels
                                new_y = pN[1] + (dy / length) * extension_pixels
                                extended_scaled_points[-1] = (new_x, new_y)

                    # Use potentially extended points to draw the line
                    draw.line(extended_scaled_points, fill=255, width=scaled_width, joint='curve')

                #mask_np = np.array(mask) > 0 # Boolean mask

                self.siril.update_progress("Step: Blurring mask...", 0.1)

                # Get user parameters
                blur_radius = self.blur_radius_var.get()
                blend_strength = self.blend_strength_var.get()
                noise_factor = 0.8

                # Apply blur and normalize mask between 0.0 and 1.0
                blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                mask_np = np.array(blurred_mask).astype(np.float32) / 255.0
                mask_np *= blend_strength

                self.siril.update_progress("Step: Applying correction...", 0.2)

                # Apply changes
                with self.siril.image_lock():
                    # Save undo state.
                    self.siril.undo_save_state("Trail Removal")
                    
                    img_np = self.full_image_data.copy().astype(np.float32) # Lavora con float per precisione

                    # ... logica di fusione per "black", "background", "reference" ...
                    # Tutte queste operazioni promuoveranno img_np a un tipo float (float32 o float64)

                    if mode == "black":
                        self.siril.update_progress("Step: Apply black...", 0.3)
                        # Blend with black (0) (no blend)
                        # binary_mask = mask_np > 0

                        # if img_np.ndim == 3:    # RGB o multi-canale
                        #     binary_mask = binary_mask[..., None]  # per broadcasting
                        # img_np[binary_mask] = 0

                        # old with blend
                        if len(img_np.shape) == 3:  # RGB or 3-channels
                            mask_np = mask_np[..., None]  # expand for broadcasting
                            black = np.zeros_like(img_np)
                            img_np = img_np * (1.0 - mask_np) + black * mask_np
                        else:                       # mono image
                            img_np = img_np * (1.0 - mask_np)
                        
                        self.siril.update_progress("Step: Apply black...", 0.6)

                    elif mode == "background":
                        self.siril.update_progress("Step: Apply background...", 0.3)
                        # --- IMPROVED BACKGROUND LOGIC FOR CFA AND MONO/RGB ---
                        is_color = img_np.ndim == 3 and img_np.shape[2] == 3
                        is_mono = img_np.ndim == 2

                        if is_color:
                            self.siril.log("RGB image detected. Processing standard background.", s.LogColor.BLUE)
                            fill_value = compute_mmm_background(img_np)
                            fill_image = np.ones_like(img_np) * fill_value
                            mask_np_expanded = mask_np[..., None]
                            
                            try:
                                bkg_rms_estimator = StdBackgroundRMS()
                                noise_std_dev = np.array([bkg_rms_estimator(self.full_image_data[..., c]) for c in range(3)])
                                self.siril.log(f"Estimated Noise (RMS per channel): {noise_std_dev}", s.LogColor.BLUE)
                                noise = np.random.normal(loc=0.0, scale=noise_std_dev * noise_factor, size=fill_image.shape)
                                fill_image += noise
                            except Exception as e:
                                self.siril.log(f"Warning: Could not add background noise. Error: {e}", s.LogColor.RED)
                            
                            self.siril.update_progress("Step: Apply background...", 0.4)
                            img_np = img_np * (1.0 - mask_np_expanded) + fill_image * mask_np_expanded
                        
                        elif is_mono:
                            cfa_pattern = None
                            try:
                                # The only way to get the path of the FITS file in Siril
                                fits_path = self.siril.get_image_filename()
                                header = fits.getheader(fits_path)
                                cfa_pattern = header.get('BAYERPAT') or header.get('CFAIMAG')
                                self.siril.log(f"Reading header from {os.path.basename(fits_path)}", s.LogColor.BLUE)

                                bitpix = header.get('BITPIX')
                                if bitpix:
                                    bit_depth_map = {
                                        8: "8-bit Integer",
                                        16: "16-bit Integer",
                                        32: "32-bit Integer",
                                        -32: "32-bit Float",
                                        -64: "64-bit Float"
                                    }
                                    bit_depth_str = bit_depth_map.get(bitpix, f"Unknown ({bitpix})")
                                    self.siril.log(f"Image Bit Depth (BITPIX): {bit_depth_str}", s.LogColor.BLUE)
                                
                                self.siril.update_progress("Step: Apply background...", 0.4)

                            except Exception:
                                self.siril.log("Could not read FITS header. Assuming true monochrome.", s.LogColor.RED)

                            if cfa_pattern:
                                # --- Logic for CFA images ---
                                self.siril.log(f"CFA image detected (Pattern={cfa_pattern}). Processing per-channel background.", s.LogColor.BLUE)
                                
                                h, w = img_np.shape
                                r_mask, g_mask, b_mask = (np.zeros_like(img_np, dtype=bool) for _ in range(3))
                                p = cfa_pattern.upper()
                                
                                if p == "RGGB":
                                    r_mask[0::2, 0::2] = True
                                    g_mask[0::2, 1::2] = True
                                    g_mask[1::2, 0::2] = True
                                    b_mask[1::2, 1::2] = True
                                elif p == "GBRG":
                                    g_mask[0::2, 0::2] = True
                                    b_mask[0::2, 1::2] = True
                                    r_mask[1::2, 0::2] = True
                                    g_mask[1::2, 1::2] = True
                                elif p == "GRBG":
                                    g_mask[0::2, 0::2] = True
                                    r_mask[0::2, 1::2] = True
                                    b_mask[1::2, 0::2] = True
                                    g_mask[1::2, 1::2] = True
                                elif p == "BGGR":
                                    b_mask[0::2, 0::2] = True
                                    g_mask[0::2, 1::2] = True
                                    g_mask[1::2, 0::2] = True
                                    r_mask[1::2, 1::2] = True
                                else:
                                    self.siril.log(f"CFA pattern '{p}' not supported. Treating as Mono.", s.LogColor.RED)
                                    cfa_pattern = None # Return to mono logic
                                
                                self.siril.update_progress("Step: Apply background...", 0.5)

                            if not cfa_pattern: # If it is true MONO or an unsupported pattern
                                self.siril.log("True Monochrome image detected. Processing standard background.", s.LogColor.BLUE)
                                fill_value = compute_mmm_background(img_np)
                                fill_image = np.ones_like(img_np) * fill_value
                                try:
                                    noise_std_dev = StdBackgroundRMS()(self.full_image_data)
                                    noise = np.random.normal(loc=0.0, scale=noise_std_dev * noise_factor, size=fill_image.shape)
                                    fill_image += noise
                                except Exception as e:
                                    self.siril.log(f"Warning: Could not add background noise. Error: {e}", s.LogColor.ORANGE)
                                
                                self.siril.update_progress("Step: Apply background...", 0.5)
                                img_np = img_np * (1.0 - mask_np) + fill_image * mask_np

                            else: # If it's a CFA image
                                # Calculate the background and noise for each channel
                                fill_r = compute_mmm_background(img_np[r_mask])
                                fill_g = compute_mmm_background(img_np[g_mask])
                                fill_b = compute_mmm_background(img_np[b_mask])
                                
                                noise_r = StdBackgroundRMS()(img_np[r_mask])
                                noise_g = StdBackgroundRMS()(img_np[g_mask])
                                noise_b = StdBackgroundRMS()(img_np[b_mask])

                                # Create the synthetic fill image with the CFA pattern
                                fill_image = np.zeros_like(img_np, dtype=np.float32)
                                fill_image[r_mask] = fill_r
                                fill_image[g_mask] = fill_g
                                fill_image[b_mask] = fill_b

                                # Add per-channel noise
                                fill_image[r_mask] += np.random.normal(0.0, noise_r * noise_factor, np.count_nonzero(r_mask))
                                fill_image[g_mask] += np.random.normal(0.0, noise_g * noise_factor, np.count_nonzero(g_mask))
                                fill_image[b_mask] += np.random.normal(0.0, noise_b * noise_factor, np.count_nonzero(b_mask))
                                
                                self.siril.update_progress("Step: Apply background...", 0.5)
                                img_np = img_np * (1.0 - mask_np) + fill_image * mask_np

                    elif mode == "reference":
                        try:
                            self.siril.update_progress("Step: Apply reference...", 0.3)

                            # Instantiate the robust background estimator from photutils
                            bkg_estimator = MMMBackground()

                            with fits.open(self.reference_path) as hdul:
                                ref_data = hdul[0].data.astype(np.float32)  # Read how to float for calculations

                            # If the reference image is CHW, convert it to HWC
                            if len(ref_data.shape) == 3 and ref_data.shape[0] in [1,3]:
                                ref_data = ref_data.transpose(1, 2, 0)

                            if ref_data.shape[:2] != img_np.shape[:2]:
                                self.siril.log(f"Error - Reference image dimensions do not match", s.LogColor.RED)
                                return
                            
                            self.siril.update_progress("Step: Balancing reference brightness...", 0.4)

                            # Calculate background using photutils for a more accurate brightness scaling
                            if len(img_np.shape) == 3: # Color Image
                                self.siril.log("Calculating background for RGB images...", s.LogColor.BLUE)
                                # Calculate background per channel using photutils
                                img_background = np.array([bkg_estimator(img_np[..., c]) for c in range(3)])
                                ref_background = np.array([bkg_estimator(ref_data[..., c]) for c in range(3)])
                                self.siril.log(f"Current img background (RGB): {np.round(img_background, 2)}", s.LogColor.BLUE)
                                self.siril.log(f"Reference img background (RGB): {np.round(ref_background, 2)}", s.LogColor.BLUE)

                                # Calculate scale factor and avoid division by zero
                                scale = np.divide(img_background, ref_background, out=np.ones_like(img_background), where=ref_background != 0)
                                self.siril.log(f"Calculated scale factors (RGB): {np.round(scale, 3)}", s.LogColor.GREEN)
                                ref_data *= scale

                            else: # Monochrome Image
                                self.siril.log("Calculating background for mono image...", s.LogColor.BLUE)
                                # Calculate background using photutils for both images
                                img_background = bkg_estimator(img_np)
                                ref_background = bkg_estimator(ref_data)
                                self.siril.log(f"Current img background: {img_background:.3f}, Reference img background: {ref_background:.3f}", s.LogColor.BLUE)

                                scale = img_background / ref_background if ref_background != 0 else 1.0
                                self.siril.log(f"Calculated scale factor: {scale:.3f}", s.LogColor.GREEN)
                                ref_data *= scale

                            self.siril.update_progress("Step: Applying reference...", 0.5)

                            # --- CORRECTED CLIPPING BLOCK ---
                            # Check the ORIGINAL data type, not the float copy's
                            if np.issubdtype(self.original_image_dtype, np.integer):
                                # Use the ORIGINAL data type information
                                info = np.iinfo(self.original_image_dtype)
                                max_val = info.max
                                ref_data = np.clip(ref_data, 0, max_val)
                            else: # For float data
                                # For floats, clipping isn't strictly necessary if the data is already normalized,
                                # but we keep it for safety (e.g., from 0 to 1.0).
                                ref_data = np.clip(ref_data, 0, 1.0)

                            self.siril.update_progress("Step: Blending images...", 0.6)

                            # Ensure data types match for the final blend
                            # La fusione avverrà tra due array float32
                            mask_np_expanded = mask_np[..., None] if len(img_np.shape) == 3 else mask_np
                            
                            # Use blending to combine images
                            img_np = img_np * (1.0 - mask_np_expanded) + ref_data * mask_np_expanded

                        except Exception as e:
                            self.siril.log(f"Error - Could not use reference image: {e}", s.LogColor.RED)
                            return

                    self.siril.update_progress("Step: Finalizing image...", 0.8)
                    # Final reconversion to the original data type
                    # clipping values if converting to an integer type
                    if np.issubdtype(self.original_image_dtype, np.integer):
                        info = np.iinfo(self.original_image_dtype)
                        final_image_data = np.clip(img_np, info.min, info.max).astype(self.original_image_dtype)
                    else:
                        final_image_data = img_np.astype(self.original_image_dtype)

                    # Transpose axes if it's a color image (channels last)
                    if final_image_data.ndim == 3:
                        final_image_data = final_image_data.transpose(2, 0, 1)
                    
                    # Update the image in Siril
                    self.siril.set_image_pixeldata(final_image_data)
                    self.real_correction_applied = True
                    self.siril.log("Trail removal performed, image updated.", s.LogColor.GREEN)
            finally:
                self.siril.update_progress("Process complete.", 1.0)
                self.siril.reset_progress()
            
        elif self.siril.is_sequence_loaded():
            self.siril.log("Warning: There is a sequence loaded, open the image using the OPEN command", s.LogColor.RED)

    def on_closing(self):
        """
        Handle dialog close - Called when the window is closed via the 'X' button.
        Close the dialog and disconnect from Siril
        """
        try:
            if SIRIL_ENV and self.siril:
                #if self.preview_applied and not self.real_correction_applied:
                    
                # Remove previous overlays for a clean preview
                self.siril.overlay_clear_polygons()
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.BLUE)
            self.siril.disconnect()
            self.root.quit()
        except Exception:
            pass
        self.root.destroy()

def main():
    try:
        if SIRIL_ENV:
            # Create the main GUI window
            #root = ThemedTk(theme="adapta") # Try a modern theme if ttkthemes is available
            root = ThemedTk()
        else:
            root = tk.Tk()

        # Create an instance of our application
        app = TrailRemovalAPP(root)
        # Start the GUI event loop, which keeps it running
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
if __name__ == '__main__':
    main()