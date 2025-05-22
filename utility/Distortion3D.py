# (c) Cecile Melis 2025
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Plots the 3D distortion map of a FITS image sing its SIP coefficients.
Distorsions are computed and reported in arcsec ["] and plotted positive when 
their direction is outwards (pincushion) and negative when inwards (barrel).
"""
import os
import sys
import sirilpy as s
# Ensure dependencies are installed
s.ensure_installed("matplotlib")
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

VERSION = "1.0.1"
REQUIRES_SIRILPY = "0.6.37"
N = 11 # defines the 3D plot grid resolution
EXAG = 0.3
COLORMAP = cm.jet

# Check if the required version of sirilpy is installed
if not s.check_module_version(f'>={REQUIRES_SIRILPY}'):
    print(f"Please install sirilpy version {REQUIRES_SIRILPY} or higher")
    sys.exit(1)


def extract_sip_from_header(header_data : dict) -> Tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Extracts SIP data from header dictionary
    
    Parameters:
    header_data (dict): Dictionary containing FITS header keywords and values
    
    Returns:
    tuple: Tuple containing the order of the SIP polynomial and the SIP coefficients matrices
    """

    # Set up SIP coefficients
    a_order = int(header_data['A_ORDER'])
    b_order = int(header_data['B_ORDER'])
    ap_order = int(header_data['AP_ORDER'])
    bp_order = int(header_data['BP_ORDER'])
    order = max(a_order, b_order, ap_order, bp_order)

    a = np.zeros([order + 1, order + 1])
    b = np.zeros([order + 1, order + 1])
    ap = np.zeros([order + 1, order + 1])
    bp = np.zeros([order + 1, order + 1])

        
    # Fill in the sip coefficients
    for key, value in header_data.items():
            # Forward coefficients
            if key.startswith('A_') and key != 'A_ORDER':
                i, j = map(int, key.split('_')[1:])
                a[i, j] = value
            elif key.startswith('B_') and key != 'B_ORDER':
                i, j = map(int, key.split('_')[1:])
                b[i, j] = value
            # Inverse coefficients
            elif key.startswith('AP_') and key != 'AP_ORDER':
                i, j = map(int, key.split('_')[1:])
                ap[i, j] = value
            elif key.startswith('BP_') and key != 'BP_ORDER':
                i, j = map(int, key.split('_')[1:])
                bp[i, j] = value
    
    return order, a, b, ap, bp

def do_plot(hdr : dict, title: str = None) -> None:
    """
    Parses the header to a dictionary, extracts SIP coefficients, and
    shows the plot
    
    Parameters:
    hdr (dict): Content of the FITS header as dictionary
    title (str): Title of the plot
    
    Returns:
    None
    """
    if not 'CTYPE1' in hdr.keys() or not 'SIP' in hdr['CTYPE1']:
        print('No distorsion info stored in this header, aborting')
        sys.exit(0)

    if 'CD1_1' in hdr:
        det = hdr['CD1_1'] * hdr['CD2_2'] - hdr['CD1_2'] * hdr['CD2_1']
        sampling = np.sqrt(abs(det)) * 3600.
    else:
        sampling = 0.5 * (abs(hdr['CDELT1']) + abs(hdr['CDELT1'])) * 3600.

    w = hdr['NAXIS1']
    h = hdr['NAXIS2']
    x0 = hdr['CRPIX1'] - 0.5
    y0 = hdr['CRPIX2'] - 0.5

    if (w > h) :
        n1 = (int)(w / h * N)
        n2 = N
    else:
        n1 = N
        n2 = (int)(h / w * N)

    order, A, B, _, _ = extract_sip_from_header(hdr)
    x = np.linspace(0., w, endpoint = True, num = n1) - x0
    y = np.linspace(0., h, endpoint = True, num = n2) - y0
    X, Y = np.meshgrid(x, y)
    DX = np.zeros_like(X)
    DY = np.zeros_like(X)
    for i in range(0, order+1):
        for j in range(0, order+1):
            DX += A[i,j] * X**i * Y**j
            DY += B[i,j] * X**i * Y**j
    DX *= sampling
    DY *= sampling
    DR = np.sqrt(DX**2 + DY**2)
    DR *= -np.sign(X * DX + Y * DY) # plot outwards distortions as positive
    X += x0
    Y += y0
    norm = Normalize(DR.min(), DR.max())

    # Plot the distorsion on a 3D map with contours below
    fig = plt.figure(figsize = (12, 5), num = f'distortion3D v{VERSION}')
    if title is not None:
        plt.suptitle(title)
    ax = fig.add_subplot(121, projection = '3d')
    surf = ax.plot_surface(X, Y, DR,
                           norm = norm,
                           cmap = COLORMAP,
                           alpha = 0.8,
                           linewidth = 0)
    # Get corner coordinates, values and colors
    corners_x = [X[0,0], X[0,-1], X[-1,0], X[-1,-1]]
    corners_y = [Y[0,0], Y[0,-1], Y[-1,0], Y[-1,-1]]
    corners_z = [DR[0,0], DR[0,-1], DR[-1,0], DR[-1,-1]]
    corners_c = COLORMAP(norm(corners_z))

    ax.scatter(corners_x, corners_y, corners_z, 
              color = corners_c, s = 50, marker = 'o')
    for cx, cy, cz in zip(corners_x, corners_y, corners_z):
        ax.text(cx, cy, cz + 0.1 * (DR.max() - DR.min()), 
               f'{cz:.1f}"', 
               horizontalalignment='center',
               verticalalignment='bottom')
    ax.contour(X, Y, DR, zdir = 'z', 
               offset = DR.min() - 50,
               levels = 10,
               norm = norm,
               cmap = COLORMAP)
    xrange = np.ptp(X)
    yrange = np.ptp(Y)
    zrange = EXAG * min(xrange, yrange)
    ax.set_box_aspect([xrange, yrange, zrange])
    ax.view_init(elev = 30, azim = 225)
    ax.grid(False)

    # Plot the distorsion as a coarser 2D grid
    pace = min(x[1] - x[0], y[1] - y[0])
    exag = EXAG * pace * sampling / np.max(np.abs(DR))

    ax = fig.add_subplot(122, projection = '3d')
    c_faces = np.lib.stride_tricks.sliding_window_view(DR, (2,2))
    c_faces = np.mean(c_faces, axis=(2, 3))
    colors = COLORMAP(norm(c_faces))
    ax.plot_wireframe(X, Y, np.zeros_like(X), 
                      linewidth = 0.5,
                      linestyle = 'dashed',
                      color = 'grey')
    surf.set_facecolor((0, 0, 0, 0))
    surf = ax.plot_surface((X - DX * exag), 
                           (Y - DY * exag),
                           np.zeros_like(X),
                           facecolors = colors, shade = False)
    surf.set_facecolor((0, 0, 0, 0))
    ax.zaxis.line.set_lw(0.)  # Make z-axis line invisible
    ax.set_zticks([])  # Remove ticks
    ax.view_init(elev = 90, azim = -90)  # Set viewing angle
    ax.mouse_init(rotate_btn = None) # Lock viewing angle
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('X [pix]')
    ax.set_ylabel('Y [pix]')
    fig.colorbar(cm.ScalarMappable(norm = norm, cmap = COLORMAP),
                 ax = ax,
                 label = 'R ["]',
                 shrink = 0.8)

    plt.tight_layout()
    plt.show()

def siril_distorsion3D() -> None:
    siril = s.SirilInterface()
    try:
        siril.connect()
        print("Connected successfully!")
    except s.SirilConnectionError as e:
        print(f"Connection failed: {e}")
        quit()

    try:
        if siril.is_image_loaded():
            header_dict = siril.get_image_fits_header(return_as = 'dict')
            filename = siril.get_image_filename()
        elif siril.is_sequence_loaded():
            seq = siril.get_seq()
            header_dict = siril.get_seq_frame_header(seq.current, return_as = 'dict')
            filename = siril.get_seq_frame_filename(seq.current)
        else:
            siril.log('No image or sequence loaded, aborting', s.LogColor.RED)
            sys.exit(1)
        filename = os.path.split(filename)[1]
        title = f'Distortion 3D map of {filename}'
        if not 'CTYPE1' in header_dict.keys() or not 'SIP' in header_dict['CTYPE1']:
            siril.log('No distortion info stored in this header, aborting', s.LogColor.SALMON)
            sys.exit(0)
        do_plot(header_dict, title)

    except Exception as e:
      siril.log(f"Error: {e}")
      sys.exit(1)

# Run the function
if __name__ == "__main__":
   siril_distorsion3D()





