# (c) Adrian Knagg-Baugh 2024
# SPDX-License-Identifier: GPL-3.0-or-later

# This script retrieves the Gaia DR3 continuous spectrum for any selected
# Gaia DR3 source in the image that has xp_continuous spectral data, and
# plots the externally-calibrated sampled representation of the spectrum
# complete with error bands. (Note that it will not work with very bright
# sources that are too bright for Gaia's processing pipeline and are
# therefore excluded from the catalogue.)

import sirilpy as s
s.ensure_installed("astropy", "astroquery", "GaiaXPy")
import sys
import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astroquery.gaia import Gaia
import gaiaxpy

def plot_spectrum(siril, from_cli, fmt):
    # Retrieve the selected star's position
    star = siril.get_selection_star()

    # Convert the star's coordinates to SkyCoord object
    coords = SkyCoord(ra=star.ra * u.deg, dec=star.dec * u.deg, frame='icrs')
    radius = 10  # search area in arcsec. This is large enough to account for any
                # plate solving inaccuracy or distortion, but small enough to
                # keep the query very quick

    # Query Gaia DR3 for the star using its coordinates
    query = f"""
    SELECT TOP 1 source_id, ra, dec,
        DISTANCE(
            POINT('ICRS', ra, dec),
            POINT('ICRS', {coords.ra.degree}, {coords.dec.degree})
        ) AS dist
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {coords.ra.degree}, {coords.dec.degree}, {radius/3600.0})
    )
    AND has_xp_continuous = 'True'
    ORDER BY dist ASC
    """

    try:
        with s.SuppressedStdout():
            job = Gaia.launch_job_async(query)
        results = job.get_results()

        # Check if any results were found
        if len(results) == 0:
            e = "No sources found near the specified coordinates, or the source is too bright and is excluded from the Gaia DR3 catalogue."
            if from_cli:
                print(f"{e}")
            else:
                siril.error_messagebox(f"{e}")

        else:
            # Select the closest source
            source_id = results[0]['source_id']

            # Plot the source location in the overlay
            cat_ra = results[0]['ra']
            cat_dec = results[0]['dec']
            siril.cmd("show", f"{cat_ra}", f"{cat_dec}", f"\"Gaia DR3 {source_id}\"")
            # Retrieve the xp_sampled representation of the source's spectrum
            try:
                with s.SuppressedStdout():
                    with s.SuppressedStderr():
                        output_data, output_sampling = gaiaxpy.calibrate([source_id], \
                                sampling=np.arange(336, 1021, 2), save_file=True, \
                                output_format=f'{fmt}', \
                                output_file=f'gaia_xpsampled_{source_id}')
                    gaiaxpy.plot_spectra(output_data, output_sampling, show_plot=True)
            except Exception as e:
                print(f"Error retrieving XP sampled spectrum: {e}")

    except Exception as e:
        print(f"Error querying Gaia DR3: {e}")

def main():
    parser = argparse.ArgumentParser(description="Siril Script to Plot the Spectrum of a Selected Star")
    parser.add_argument("-x", type=int,default=-1,
                        help="x position of center of search box")
    parser.add_argument("-y", type=int, default=-1,
                        help="y position of center of search box")
    parser.add_argument("-size", type=int,default=16,
                       help="size of search box")
    parser.add_argument("-fmt", type=str, default="xml",
                        help="output format (default = 'xml', options are 'avro', 'csv', 'ecsv', 'fits' or 'xml')")
    args = parser.parse_args()

    # Initialize Siril interface and ensure necessary packages are installed
    siril = s.SirilInterface()
    if not siril.connect():
        print("Failed to connect to Siril", file=sys.stderr)
        return

    if not siril.is_image_loaded():
        siril.error_messagebox("No image is loaded")
        return

    channels, height, width = siril.get_image_shape()

    halfsize = int(args.size / 2)

    from_cli = False
    selx = None
    sely = None
    selw = None
    selh = None
    if (args.x > halfsize and args.y > halfsize and args.x < (width - halfsize) and \
                args.y < (height - halfsize)):
        selx = args.x = halfsize
        sely = args.y - halfsize
        selw = args.size
        selh = args.size
        siril.set_siril_selection(selx, sely, selw, selh)
        from_cli = True
    else:
        selx, sely, selw, selh = siril.get_siril_selection()

    if (not selw and not selh):
        e = "Error: no selection set"
        if from_cli:
            print(f"{e}", file=sys.stderr)
        else:
            siril.error_messagebox(f"{e}")
        return

    try:
        plot_spectrum(siril, from_cli, args.fmt)

    except Exception as e:
        if from_cli:
            print(f"Error: {str(e)}", file=sys.stderr)
        else:
            siril.error_messagebox(f"{str(e)}")

    finally:
        siril.disconnect()
        return

if __name__ == "__main__":
    main()
