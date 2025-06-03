# (c) Steffen Schreiber, Patrick Wagner 2025
# SPDX-License-Identifier: GPL-3.0-or-later
#
# For bug reports or feature requests, please open an issue at
# https://gitlab.com/schreiberste/siril-scripts/-/issues

"""
Creates galaxy annotations from a Simbad query with several 
catalogs. Combines the original image with annotation overlays
and a thumbnail table of the found galaxies.
"""

# Version History
# 1.0.0 Initial release

# Core module imports
import os
import sys
import math
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import sirilpy as s

# Check the module version is enough to provide get_image_fits_header(return_as = 'dict')
if not s.check_module_version('>=0.6.37'):
    print("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
    sys.exit(1)

from sirilpy import tksiril, SirilError
s.ensure_installed("ttkthemes")
s.ensure_installed("astropy", "astroquery", "matplotlib", "numpy", "pandas", "Pillow", "scikit-image", "subprocess" )

from ttkthemes import ThemedTk

# Add any additional imports here
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from skimage.transform import resize
from PIL import Image
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import coordinates as coord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from astroquery.simbad import Simbad
import pandas as pd

VERSION = "1.0.0"
CONFIG_FILENAME = "Galaxy_Annotations.conf"

def annotate_fit(siril, fit, catalogs, output, title, logo_path, overlay_alpha):
    """
    The main processing function for creating the annotation images.
    Returns the number of found and annotated objects.
    """
    print(f"Title: {title}")
    print(f"Logo: {logo_path}")
    
    main_object = title
    output_fname = get_combined_filename(output)
    output_overlay_fname = get_overlay_filename(output)
    output_table_fname = get_table_filename(output)

    if fit.data.ndim == 2:
        # Convert mono image to color image
        img = np.expand_dims(fit.data, -1)
        img = np.tile(img, (1,1,3))
    else:
        # fit.data is channels-first: (C, H, W)
        # get the input image in channels-last format
        img = np.transpose(fit.data, (1, 2, 0))

    H, W, C = img.shape
    print(f"Input dimensions: {W} x {H}")

    # minimum size of galaxies to annotate in pixels
    minsize_pixels = 5  
    # minimum size of annotation patches in pixels
    min_patch_size = int(round(max(W,H) / 100)) 

    # normalize to float, range [0,1]
    if img.dtype == np.uint16:
        print("Input image is 16 bit")
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        print("Input image is 8 bit")
        img = img.astype(np.float32) / 255.0
    else:
        # should be float now, but make sure...
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        # at least some 8 bit images were represented 
        # as float in range [0.. 255/65535] -> normalize them
        maxValue = np.max(img)
        if maxValue <= (255.0 / 65535.0) or maxValue > 1.0:
            print(f"Normalizing value range to [0..1]")
            img = img / maxValue

    # get center coordinates from fit
    (center_ra, center_dec) = siril.pix2radec(W / 2, H / 2)
    print(f"Center: {center_ra, center_dec}")
    
    # get world coordinates system
    header = siril.get_image_fits_header(return_as = 'dict')
    wcs = WCS(header,naxis=[1,2])

    # Query Simbad
    simbad = Simbad()
    simbad.TIMEOUT = 120
    simbad.add_votable_fields("otype", "galdim_majaxis")
    
    target_coord = SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg), frame='icrs')
    (TL_ra, TL_dec) = siril.pix2radec(0, 0)
    radius_deg = max([abs(center_ra - TL_ra), abs(center_dec - TL_dec)])
    # minimum size of galaxies we want to annotate
    (pixsize_ra, pixsize_dec) = siril.pix2radec(1, 1)
    pixsize_arcmin = 60 * max([abs(pixsize_ra - TL_ra), abs(pixsize_dec - TL_dec)])
    minsize_arcmin = minsize_pixels * pixsize_arcmin 
    radius = f"{radius_deg}d"
    # query for galaxies only
    # limit the query to galaxies with at least minsize (or unknown size)
    criteria_opt = f"otype='Galaxy..' AND (galdim_majaxis>{minsize_arcmin} OR (galdim_majaxis IS NULL))"
    print(f"Query radius: {radius}")
    print(f"      minimum size: {minsize_pixels} pixels ~ {minsize_arcmin}′")
    print(f"      criteria: {criteria_opt}")
    result_table = simbad.query_region(target_coord, radius, criteria=criteria_opt)
    
    result_table.sort("galdim_majaxis", reverse=True)
    df = result_table.to_pandas()
    print(f"Simbad query results: {df.shape[0]} entries")

    # filter by position, remove anything outside the image or too close to border
    df['Pixel_Position'] = df.apply(lambda row: siril.radec2pix(row['ra'], row['dec']), axis=1)
    df['px'] = df['Pixel_Position'].apply(lambda x: int(round(x[0])))
    df['py'] = df['Pixel_Position'].apply(lambda x: int(round(x[1])))
    df = df[(df.px > min_patch_size) & (df.py > min_patch_size) 
          & (df.px < W-min_patch_size) & (df.py < H-min_patch_size)]
    print(f"Filtered query result by image coordinates: {df.shape[0]} entries")

    # filter by catalog
    # get catalog/TYPE of object by simple string manipulation
    df['TYPE'] = df['main_id'].apply(lambda x: x.split(' ')[0].split('+')[0])
    filter_types = []
    for key, value in catalogs.items():
        if value.get_selected():
            filter_types.append(key)

    print(f"Filtering by catalogs: {filter_types}")
    filtered_result = Table.from_pandas(df[df['TYPE'].isin(filter_types)])
    dfi = filtered_result.to_pandas()
    print(f"Filtered by catalog: {dfi.shape[0]} entries")

    # remove duplicates
    dfi = dfi.drop_duplicates(subset=['px', 'py'])
    print(f"Filtered after removing duplicates: {dfi.shape[0]} entries")

    if dfi.shape[0] == 0:
        print("No objects found in image boundary.")
        return 0

    sub = 1
    dpi = 200
    # sorting order from catalogs list position
    rep_dic = {}
    for i, key in enumerate(catalogs.keys()):
        rep_dic[key] = f"{i:02d}"
        
    dfi['sorting'] = dfi.TYPE.replace(rep_dic)
    dfi.main_id = dfi.main_id.apply(lambda x: x.replace(' ', ''))
    dfi = dfi.sort_values(['sorting', 'main_id'], ascending=True).reset_index()
    # dfi = dfi.sort_values(['galdim_majaxis', 'main_id'], ascending=False).reset_index()
    # print(dfi)

    # set up the plot
    plt.style.use('dark_background')
    # figure size in inch for the upper image with overlays
    # Try to get roughly the size of the input image...
    extra_axis_label_size_inches = 1.15 # estimated extra size for axis labels
    fig = plt.figure(
        figsize=(W / dpi + extra_axis_label_size_inches, H / dpi + extra_axis_label_size_inches))
    ax1 = plt.subplot(projection=wcs, label='overlays')
    ax1.imshow(img[::sub, ::sub])
    ax1.coords.grid(True, color='white', ls=':', alpha=overlay_alpha)
    ax1.coords[0].set_axislabel('Right Ascension (J2000)')
    ax1.coords[1].set_axislabel('Declination (J2000)', minpad=-1)
    ax1.set_title(title, fontsize=24)

    all_patches = []
    filter_idxs = []
    
    for i, row in dfi.iterrows():
        siril.update_progress(f"Creating patches", i / (10 * dfi.shape[0]))
        # default style
        fontsize = 12
        try:
            color = catalogs[row.TYPE].color
        except KeyError:
            color = '#ff0000'
        
        # try to derive the patch size from galaxy angular size
        # row.galdim_majaxis is in arcmin
        angular_size = row.galdim_majaxis
        size_factor = 2
        if math.isnan(angular_size):
            print(f"No angular size information for {row.main_id}")
            angular_size = 0
            if row.TYPE == 'M':
                # Simbad is missing angular size for: Messier 8, 40, 43, 78, 82
                # Only m82 is a galaxy, but handle the others just in case...
                if row.main_id == "M8":
                    angular_size = 90
                elif row.main_id == "M40":
                    angular_size = 0.86
                elif row.main_id == "M43":
                    angular_size = 20
                elif row.main_id == "M78":
                    angular_size = 8
                elif row.main_id == "M82":
                    angular_size = 11.2

        if angular_size == 0:
            patch_size = min_patch_size
        else:
            # angular size is the major axis diameter in arcmin
            patch_diameter_deg = angular_size / 60.0
            # convert patch size to pixels, depends on position in the grid
            tmp = siril.radec2pix(row.ra, row.dec + patch_diameter_deg)
            dx = tmp[0] - row.Pixel_Position[0]
            dy = tmp[1] - row.Pixel_Position[1]
            patch_diameter_pix = math.sqrt(dx * dx + dy * dy)
            patch_size = int(round(patch_diameter_pix * size_factor))
            if patch_size == 0:
                siril.log(f"{i+1}. {row.main_id}: angular size = {angular_size} arcmin -> patch size = {patch_size} pixels", color=s.LogColor.RED)
                        
            patch_size = max(min_patch_size, patch_size)
                    
        # type dependent style
        if row.main_id == main_object:
            fontsize=20
            color = 'white'
        elif row.TYPE == 'M':
            fontsize = 18
        elif row.TYPE == 'NGC':
            fontsize = 18
        elif row.TYPE == 'SAI':
            fontsize = 16
        elif row.TYPE == 'UGC':
            fontsize = 16
        elif row.TYPE == 'MCG':
            fontsize = 16
        elif row.TYPE == 'IC':
            fontsize = 16
        elif row.TYPE == 'LEDA':
            # LEDA apparently often has large errors on galdim_majaxis
            # resulting in much too large patches. Usually, LEDA galaxies
            # are small...
            if not math.isnan(row.galdim_majaxis) and row.galdim_majaxis > 1.8:
                patch_size = min_patch_size
                print(f"Unlikely large LEDA galaxy {row.main_id}: {row.galdim_majaxis} -> patch size {patch_size}")

        annotation_text = str(i + 1)
        if patch_size > 200:
            annotation_text = f"{annotation_text}: {row.main_id}"
        
        # clip the patch size on image borders
        clipped = min(patch_size, (W - row.px) * 2)
        clipped = min(clipped, (H - row.py) * 2)
        clipped = min(clipped, row.px * 2)
        clipped = min(clipped, row.py * 2)
        
        x1 = row.px - clipped // 2
        x2 = row.px + clipped // 2
        y1 = H-row.py - clipped // 2
        y2 = H-row.py + clipped // 2

        annotatation_rectangles = False
        if annotatation_rectangles:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                alpha=overlay_alpha, linewidth=1, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            text_y = y1 - 6
            v_align = 'top'
            if text_y < 0:
                text_y = min(y2 + 6, H - (3*fontsize))
                v_align = 'bottom'
        else:
            # annotation circle
            annot_radius = max(min_patch_size, 1.2 * patch_diameter_pix / 2.0) 
            circ = Circle((row.px, H-row.py), radius=annot_radius,
                alpha=overlay_alpha, linewidth=1, edgecolor=color, facecolor='none')
            ax1.add_patch(circ)
            text_y = H-row.py - 6 - annot_radius
            v_align = 'top'
            if text_y < 0:
                text_y = min(H-row.py + 6  + annot_radius, H - (3*fontsize))
                v_align = 'bottom'
                
        ax1.text(row.px, text_y, annotation_text, 
            ha='center', va=v_align, color=color, alpha=overlay_alpha, fontsize=fontsize)
            
        
        patch = img[y1:y2, x1:x2]
        all_patches.append(patch)
        filter_idxs.append(i)
        

    plt.tight_layout()
    siril.update_progress("Saving overlay image...", 0.2)
    plt.savefig(output_overlay_fname, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    siril.update_progress("Finished overlay image.", 0.3)

    overlay_image = plt.imread(output_overlay_fname)

    # Resize and process patches
    new_patch_size = 512  # Adjust the patch size to your desired resolution

    siril.update_progress("Resizing patch images...", 0.4)
    all_patches_resized = [resize(patch, (new_patch_size, new_patch_size)) for patch in all_patches]
    all_patches = np.array(all_patches_resized)
    siril.update_progress("Patch images resized.", 0.5)

    # Define the scale for the subplots
    scale = 3  # inches per patch

    # Calculate numbers of rows and columns
    n = len(all_patches)
    mincols = 6 if (logo_path != "") else 5 
    ncols = max(mincols, int(np.floor(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))
    print(f"Grid size: nrows={nrows}, ncols={ncols}")

    # Create the subplots with the adjusted dimensions
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * scale))

    # Create a filtered DataFrame with the desired indices
    dft = dfi.iloc[filter_idxs].reset_index()

    for i, row in dft.iterrows():
        if nrows > 1:
            ax = axarr[i // ncols, i % ncols]
        else:
            ax = axarr[i]
        try:
            color = catalogs[row.TYPE].color
        except KeyError:
            color = '#ff0000'
        ax.imshow(all_patches[i][::-1])
        ax.set_title(row.main_id, fontsize=12, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        ax.text(2, 2, str(i + 1), ha='left', va='top', color='white', fontsize=18)

    for i in range(n, nrows * ncols):
        if nrows > 1:
            ax = axarr[i // ncols, i % ncols]
        else:
            ax = axarr[i]
        ax.axis('off')

    # If logo is requested, put it in the last grid cell
    # (as long as the cell is free)
    if (logo_path != "") and (nrows * ncols > n):
        logo_img = plt.imread(logo_path)
        if nrows > 1:
            axarr[nrows - 1, ncols - 1].imshow(logo_img)
        else:
            axarr[ncols - 1].imshow(logo_img)

    siril.update_progress("Creating thumbnail table...", 0.6)
    plt.tight_layout()
    plt.savefig(output_table_fname, bbox_inches='tight', pad_inches=.1, dpi=dpi)
    siril.update_progress("Saved thumbnail table image.", 0.7)

    table_image = plt.imread(output_table_fname)

    # Resize table_image to match overlay_image dimensions
    siril.update_progress("Creating combined output image...", 0.8)
    output_shape = (int(table_image.shape[0] * (overlay_image.shape[1] / table_image.shape[1])), overlay_image.shape[1])
    table_image_scaled = (resize(table_image, output_shape) * 255).astype(np.uint8)
    im = Image.fromarray(np.vstack([(overlay_image * 255).astype(np.uint8), table_image_scaled])[:, :, :3])

    siril.update_progress("Saving combined output image...", 0.9)
    im.save(output_fname)
    print("output image files:")
    print("  overlay:  ", output_overlay_fname)
    print("  table:    ", output_table_fname)
    print("  combined: ", output_fname)
    
    siril.update_progress("Finished.", 1)
    
    return dfi.shape[0]
    
def get_output_filename(output_basename, suffix=''):
    filename, extension = os.path.splitext(output_basename)
    if extension == '':
        extension = '.png'
    return f"{filename}{suffix}{extension}"
    
def get_overlay_filename(output_basename):
    return get_output_filename(output_basename, '_overlay')
    
def get_table_filename(output_basename):
    return get_output_filename(output_basename, '_table')
    
def get_combined_filename(output_basename):
    return get_output_filename(output_basename, '')


class CatalogEntry:
    """ This class provides properties of a catalog entry """
    def __init__(self, description, color='#ffffff', selection_default=True):
        self.description = description
        self.color = color
        self.selection_default = selection_default
        self.checkbox_var = None
        
    def get_selected(self):
        if self.checkbox_var is None:
            return self.selection_default
        else:
            return self.checkbox_var.get()

class AnnotationsScriptInterface:
    """ This class provides the GUI and related callbacks """
    def __init__(self, root=None, cli_args=None):
        # If no CLI args, create a default namespace with defaults
        if cli_args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-output", type=str, default=None)
            parser.add_argument("-title", type=str, default=None)
            parser.add_argument("-logo_path", type=str, default="")
            parser.add_argument("-overlay_alpha", type=float, default=0.6)
            cli_args = parser.parse_args([])

        self.cli_args = cli_args
        self.root = root
        
        # Catalog definitions
        self.catalogs = {
            'M': CatalogEntry('Messier Catalog', '#3e6ebe', True), 
            'IC': CatalogEntry('Index Catalogue', '#b2c5eb', True),
            'NGC': CatalogEntry('New General Catalogue', '#9ae483', True),
            'MGC': CatalogEntry('Millennium Galaxy Catalogue', '#30a500', True),
            'UGC': CatalogEntry('Uppsala General Catalogue', '#3abed1', True),
            'MCG': CatalogEntry('Morphological Catalogue of Galaxies', '#955ec2', True),
            'Mrk': CatalogEntry('Markarian galaxies', '#fbbd70', True),
            'LEDA': CatalogEntry('Lyon-Meudon Extragalactic Database', '#c29d94', True),
            'Z': CatalogEntry('Zwicky Catalogue of galaxies and of clusters of galaxies', '#fb9795', True),
            'Gaia': CatalogEntry('Gaia catalogues', '#c6aed8', True),
            '2MASX': CatalogEntry('Two Micron All Sky Survey, Extended source catalogue', '#895447', True),
            'SDSS': CatalogEntry('Sloan Digital Sky Survey', '#b2c5eb', False),
            'SDSSCGB': CatalogEntry('SDSS DR6 Compact Group Catalogue B', '#b2c5eb', False),
            'UGCA': CatalogEntry('Uppsala Selected non-UGC Galaxies', '#f5b3d3', True),
            'MASS': CatalogEntry(None, '#c8c8c8', True),
            'MFGC': CatalogEntry(None, '#b9c200', True),
            '2MFGC': CatalogEntry('2MASS Flat Galaxy Catalog', '#d9df85', True),
            'FIRST': CatalogEntry('FIRST Survey Catalogs', '#a3dae7', False),
            '2MASS': CatalogEntry('Two Micron All Sky Survey', '#895447', False)
        }

        if root:
            self.root.title(f"Galaxy Annotations Script - v{VERSION}")
            self.root.resizable(False, False)
            self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        try:
            self.siril.connect()
        except s.SirilConnectionError:
            if root:
                self.siril.error_messagebox("Failed to connect to Siril")
            else:
                print("Failed to connect to Siril")
            return

        # Initial checks: example - check if an image is loaded
        if not self.siril.is_image_loaded():
            if root:
                self.siril.error_messagebox("No image is loaded")
            else:
                print("No image is loaded")
            return
            
        # Check if the version of Siril is high enough
        try:
            self.siril.cmd("requires", "1.4.0-beta2")
        except s.CommandError:
            return

        if self.cli_args.output is None:
            # get the default output file name from input image file name
            basename = os.path.basename(self.siril.get_image_filename())
            filename, extension = os.path.splitext(basename)
            self.cli_args.output = "annotated_" + filename 

        if self.cli_args.title is None:
            # get the default title from the file name
            basename = os.path.basename(self.siril.get_image_filename())
            filename, extension = os.path.splitext(basename)
            self.cli_args.title = filename 

        # Initialization to do in GUI mode...
        if root:
            # load options from config file
            logo_path, overlay_alpha, selected_catalogs = self.load_config_file()
            self.cli_args.logo_path = logo_path
            self.cli_args.overlay_alpha = overlay_alpha
            # Create the UI and match its theme to Siril
            self.create_widgets()
            tksiril.match_theme_to_siril(self.root, self.siril)


        # Only apply changes if in CLI mode
        if self.siril.is_cli():
            print("Apply changes from CLI")
            self.apply_changes(from_cli=True)

    def _browse_logo_file(self):
        """
        Use a TK filedialog to browse for the logo file.
        """
        filename = filedialog.askopenfilename(
            title="Select a Logo Image File",
            initialdir=os.path.expanduser("~"),
            filetypes=[("Image file",".png .jpg .jpeg .ico .bmp .gif")]
        )
        if filename:
            self.logo_path.set(filename)
            # update config file
            self.save_config_file(filename, self.overlay_alpha_var.get(), None)


    def create_widgets(self):
        """Create the GUI's widgets, connect signals etc. """
        # Main frame with no padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Galaxy Annotations Script",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 5))
        version_label = ttk.Label(main_frame, text=f"Script version: {VERSION}")
        version_label.pack(pady=(0, 0))

        # Separator
        sep = ttk.Separator(main_frame, orient='horizontal')
        sep.pack(fill=tk.X, pady=5)

        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Output", padding=10)
        params_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        params_frame.columnconfigure(1, weight=1)  # use all space in col 1

        # image title
        row = 0
        titlelbl = ttk.Label(params_frame, text="Title: ")
        titlelbl.grid(column=0, row=row, sticky="WENS")
        self.title = tk.StringVar(self.root, value=self.cli_args.title)
        title_entry = ttk.Entry(params_frame, textvariable=self.title)
        title_entry.grid(column=1, row=row, columnspan=3, sticky="WENS", padx=2, pady=2)

        # Logo file selection
        row = row + 1
        logolbl = ttk.Label(params_frame,text="Logo: ")
        logolbl.grid(column=0, row=row, sticky="WENS")
        self.logo_path = tk.StringVar(self.root, value=self.cli_args.logo_path)
        logo_file_entry = ttk.Entry(params_frame, textvariable=self.logo_path)
        logo_file_entry.grid(column=1, row=row, columnspan=3, sticky="WENS", padx=2, pady=2)
        browsebtn = ttk.Button(params_frame, text="Browse", command=self._browse_logo_file, style="TButton")
        browsebtn.grid(column=4, row=row, sticky="W")

        # Output file name
        row = row + 1
        outputlbl = ttk.Label(params_frame, text="Output file: ")
        outputlbl.grid(column=0, row=row, sticky="WENS")
        self.output = tk.StringVar(self.root, value=self.cli_args.output)
        output_file_entry = ttk.Entry(params_frame, textvariable=self.output)
        output_file_entry.grid(column=1, row=row, columnspan=3, sticky="WENS", padx=2, pady=2)

        # Overlay transparency
        overlay_alpha_label = ttk.Label(params_frame, text="Overlay: ")
        overlay_alpha_label.grid(column=0, row=row, sticky="WENS")
        self.overlay_alpha_var = tk.DoubleVar(value=self.cli_args.overlay_alpha)
        overlay_alpha_slider = ttk.Scale(
            params_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.overlay_alpha_var
        )
        overlay_alpha_slider.grid(column=1, row=row, columnspan=3, sticky="WENS", padx=2, pady=2)
        self.overlay_alpha_value = ttk.Label(params_frame, text=f"{self.cli_args.overlay_alpha}")
        self.overlay_alpha_value.grid(column=4, row=row, sticky="WENS")
        tksiril.create_tooltip(overlay_alpha_slider, "Adjust the visibility of "
                "the annotation overlays. Smaller values result in more "
                "transparent overlays.")
        self.overlay_alpha_var.trace_add("write", self._update_alpha_label)

        # Load in Siril options
        row = row + 1
        loadlbl = ttk.Label(params_frame, text="Load in Siril: ")
        loadlbl.grid(column=0, row=row, sticky="WENS")
        self.load_in_siril = tk.StringVar(None, 'C')
        load_frame = ttk.Frame(params_frame)
        load_frame.grid(column=1, row=row, columnspan=4, sticky="WENS")
        rbtn = ttk.Radiobutton(load_frame, text='Combined', value='C', variable=self.load_in_siril)
        rbtn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(rbtn, "Load the combined result image in Siril")
        rbtn = ttk.Radiobutton(load_frame, text='Overlay', value='O', variable=self.load_in_siril)
        rbtn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(rbtn, "Load the result image with annotation overlays in Siril")
        rbtn = ttk.Radiobutton(load_frame, text='Table', value='T', variable=self.load_in_siril)
        rbtn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(rbtn, "Load the thumbnail table image in Siril")
        rbtn = ttk.Radiobutton(load_frame, text='None', value='', variable=self.load_in_siril)
        rbtn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(rbtn, "Just create output files without loading anything in Siril")

        # Catalog selection
        catalogs_frame = ttk.LabelFrame(main_frame, text="Catalogs", padding=10)
        catalogs_frame.pack(fill=tk.X, padx=5, pady=0)

        i = 0
        for key, value in self.catalogs.items():
            value.checkbox_var = tk.BooleanVar(self.root)
            value.checkbox_var.set(value.selection_default)
            checkbox = ttk.Checkbutton(catalogs_frame,
                text=key, variable=value.checkbox_var, style="TCheckbutton"
            )
            checkbox.grid(row=i, column=0, sticky="WENS")
            description = value.description
            if description is None:
                description = key
            label = ttk.Label(catalogs_frame, text=description)
            label.grid(row=i, column=1, sticky="WENS")
            i = i + 1
            
        # buttons for mass- (de-) selection
        select_frame = ttk.Frame(catalogs_frame)
        select_frame.grid(column=0, row=i, columnspan=2, sticky="WENS", pady=2)
        select_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(select_frame, text="Select: ").grid(row=0,column=0)
        select_all_btn = ttk.Button(select_frame, text="All", command=self.select_all)
        select_all_btn.grid(row=0, column=1, sticky="WENS")
        select_none_btn = ttk.Button(select_frame, text="None", command=self.select_none)
        select_none_btn.grid(row=0, column=2, sticky="WENS")
        select_default_btn = ttk.Button(select_frame, text="Defaults", command=self.select_default)
        select_default_btn.grid(row=0, column=3, sticky="WENS")

        # Separator
        sep2 = ttk.Separator(main_frame, orient='horizontal')
        sep2.pack(fill=tk.X, pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(close_btn, "Close, no changes will be made to the current image.")

        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self.apply_changes,
            style="TButton"
        )
        apply_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(apply_btn, "Create the annotated output image")

    def apply_changes(self, from_cli=False):
        """
        Get the necessary variables from CLI args or the GUI and call the algorithm
        """
        try:
            # Get the thread
            with self.siril.image_lock():
                # Determine parameters: prefer CLI args if provided,
                # else use GUI values
                if from_cli and self.cli_args:
                    output = self.cli_args.output
                    title = self.cli_args.title
                    logo_path = self.cli_args.logo_path
                    overlay_alpha = self.cli_args.overlay_alpha
                else:
                    output = self.output.get()
                    title = self.title.get()
                    logo_path = self.logo_path.get()
                    overlay_alpha = float(self.overlay_alpha_var.get())
                    # update config file
                    self.save_config_file(logo_path, overlay_alpha, None)

                # Get current image
                fit = self.siril.get_image()
                fit.ensure_data_type(np.float32)

                # ensure is plate solved
                try:
                    self.siril.pix2radec(0, 0)
                except ValueError:
                    self.siril.log("The image is not plate solved", color=s.LogColor.RED)
                    if not from_cli:
                        self.siril.error_messagebox("The image is not plate solved")
                    return

                # Create the annotated image
                found = annotate_fit(self.siril, fit, self.catalogs, output, title, logo_path, overlay_alpha)
                
                if found > 0:
                    self.siril.log("Annotations image created successfully.", color=s.LogColor.GREEN)
                    # Optionally load the annotated image in Siril
                    if self.load_in_siril.get() == 'C':
                       self.siril.cmd("load", "\"" + get_combined_filename(output) + "\"")
                    elif self.load_in_siril.get() == 'O':
                       self.siril.cmd("load", "\"" + get_overlay_filename(output) + "\"")
                    elif self.load_in_siril.get() == 'T':
                       self.siril.cmd("load", "\"" + get_table_filename(output) + "\"")


        except SirilError as e:
            if from_cli:
                print(f"Error: {str(e)}")
            else:
                messagebox.showerror("Error", str(e))

    def close_dialog(self):
        """ Close dialog """
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()
            
    def select_all(self):
        """ Select all catalogs """
        for key, value in self.catalogs.items():
            value.checkbox_var.set(True)
    
    def select_none(self):
        """ Select no catalogs """
        for key, value in self.catalogs.items():
            value.checkbox_var.set(False)

    def select_default(self):
        """ Select default set of catalogs """
        for key, value in self.catalogs.items():
            value.checkbox_var.set(value.selection_default)

    def _update_alpha_label(self, *args):
        """Update the overlay alpha value label when slider changes"""
        self.overlay_alpha_value.config(text=f"{self.overlay_alpha_var.get():.2f}")

    def load_config_file(self):
        """
        Check for a saved options in the configuration file.
        Returns (logo_path, overlay_alpha, selected_catalogs) or default values if not found.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        logo_path = None
        overlay_alpha = 0.6
        selected_catalogs = None
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    logo_path = lines[0].strip()
                    if not os.path.isfile(logo_path):
                        logo_path = None
                if len(lines) > 1:
                    overlay_alpha = float(lines[1].strip())
                if len(lines) > 2:
                    selected_catalogs = lines[2].strip()
        return logo_path, overlay_alpha, selected_catalogs

    def save_config_file(self, logo_path, overlay_alpha, selected_catalogs=None):
        """
        Save the options to the configuration file.
        """
        config_dir = self.siril.get_siril_configdir()
        config_file_path = os.path.join(config_dir, CONFIG_FILENAME)
        try:
            with open(config_file_path, 'w') as file:
                file.write(logo_path + "\n")
                file.write(f"{overlay_alpha:.2f}" + "\n")
                if selected_catalogs is not None:
                    file.write(str(selected_catalogs) + "\n")
        except Exception as e:
            print(f"Error saving config file: {str(e)}")

def main():
    """ Main entry point """
    parser = argparse.ArgumentParser(description="Annotations script")
    parser.add_argument("-output", type=str, default=None,
                        help="Output file name")
    parser.add_argument("-title", type=str, default="",
                        help="Optional image title")
    parser.add_argument("-logo_path", type=str, default="",
                        help="Optional logo image path")
    parser.add_argument("-overlay_alpha", type=float, default=0.6,
                        help="Optional overlay alpha value")

    args = parser.parse_args()

    try:
        if args.output != None:
            # CLI mode
            AnnotationsScriptInterface(cli_args=args)
        else:
            # GUI mode
            root = ThemedTk()
            AnnotationsScriptInterface(root)
            root.mainloop()
    except SirilError as e:
        print(f"Error initializing script: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
