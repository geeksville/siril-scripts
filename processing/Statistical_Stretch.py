# (c) Cyril Richard 2024
# Code From Seti Astro Statistical Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Version 1.0.0

import sirilpy as s
s.ensure_installed("ttkthemes")

import os
import sys
import argparse
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import numpy as np

VERSION = "1.0.0"

class StatisticalStretchInterface:
    def __init__(self, root=None, cli_args=None):
        # If no CLI args, create a default namespace with defaults
        if cli_args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-median", type=float, default=0.2)
            parser.add_argument("-boost", type=float, default=0.0)
            parser.add_argument("-linked", action="store_true", default=False)
            parser.add_argument("-normalize", action="store_true", default=False)
            cli_args = parser.parse_args([])

        self.cli_args = cli_args

        if root:
            self.root = root
            self.root.title(f"Statistical Stretch Interface - v{VERSION}")
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

        if not self.siril.is_image_loaded():
            if root:
                self.siril.error_messagebox("No image is loaded")
            else:
                print("No image is loaded")
            return

        if not self.siril.cmd("requires", "1.3.6"):
            return

        if root:
            self.create_widgets()
            tksiril.match_theme_to_siril(self.root, self.siril)

        # Only apply changes if CLI arguments are non-default
        if cli_args and (cli_args.median != 0.2 or cli_args.boost != 0.0 or
                         cli_args.linked or cli_args.normalize):
            self.apply_changes(from_cli=True)

    def create_widgets(self):
        # Main frame with no padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Statistical Stretch Settings",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Target Median
        median_frame = ttk.Frame(params_frame)
        median_frame.pack(fill=tk.X, pady=5)

        ttk.Label(median_frame, text="Target median:").pack(side=tk.LEFT)
        self.target_median_var = tk.DoubleVar(value=self.cli_args.median)
        target_median_scale = ttk.Scale(
            median_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.target_median_var,
            length=200
        )
        target_median_scale.pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Label(
            median_frame,
            textvariable=self.target_median_var,
            width=5,
            style="Value.TLabel"
        ).pack(side=tk.LEFT)
        tksiril.create_tooltip(target_median_scale, f"Adjusts the target median value for image stretching. A lower value will darken the image, while a higher value will brighten it.")

        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=10)

        # Linked Stretch checkbox
        # Linked Stretch checkbox
        self.linked_stretch_var = tk.BooleanVar(value=self.cli_args.linked)
        linked_stretch_checkbox = ttk.Checkbutton(
            options_frame,
            text="Linked Stretch",
            variable=self.linked_stretch_var,
            style="TCheckbutton"
        )
        linked_stretch_checkbox.pack(anchor=tk.W, pady=2)
        tksiril.create_tooltip(linked_stretch_checkbox, "When enabled, applies the same stretching parameters to all color channels simultaneously for color images. When disabled, stretches each color channel independently.")

        # Normalize checkbox
        self.normalize_var = tk.BooleanVar(value=self.cli_args.normalize)
        normalize_stretch_checkbox = ttk.Checkbutton(
            options_frame,
            text="Normalize",
            variable=self.normalize_var,
            style="TCheckbutton"
        )
        normalize_stretch_checkbox.pack(anchor=tk.W, pady=2)
        tksiril.create_tooltip(normalize_stretch_checkbox, "Scales the image data to use the full dynamic range from 0 to 1, ensuring maximum contrast and detail preservation after stretching.")

        # Apply Curve checkbox
        self.apply_curve_var = tk.BooleanVar(value=self.cli_args.boost > 0)
        apply_curve_checkbox = ttk.Checkbutton(
            options_frame,
            text="Apply Curves Adjustment",
            variable=self.apply_curve_var,
            command=self.toggle_curves_boost,
            style="TCheckbutton"
        )
        apply_curve_checkbox.pack(anchor=tk.W, pady=2)
        tksiril.create_tooltip(apply_curve_checkbox, "Enables non-linear curve boosting to enhance image contrast and bring out finer details by applying a power law transformation.")

        # Curves Boost frame
        self.curves_boost_frame = ttk.Frame(options_frame)
        self.curves_boost_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.curves_boost_frame, text="Curves Boost:").pack(side=tk.LEFT)

        self.curves_boost_var = tk.DoubleVar(value=self.cli_args.boost)
        self.curves_boost_scale = ttk.Scale(
            self.curves_boost_frame,
            from_=0.0,
            to=0.5,
            orient=tk.HORIZONTAL,
            variable=self.curves_boost_var,
            length=200
        )
        self.curves_boost_scale.pack(side=tk.LEFT, padx=10, expand=True)
        ttk.Label(
            self.curves_boost_frame,
            textvariable=self.curves_boost_var,
            width=5,
            style="Value.TLabel"
        ).pack(side=tk.LEFT)
        tksiril.create_tooltip(self.curves_boost_scale, f"Controls the intensity of the non-linear curve adjustment. Higher values increase contrast and emphasize faint details, but can also introduce more noise or artifacts.")

        # Initially disable curves boost if no boost
        if self.cli_args.boost == 0:
            self.curves_boost_scale.state(['disabled'])

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(close_btn, "Close the Statistical Stretch interface and disconnect from Siril. No changes will be made to the current image.")

        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self.apply_changes,
            style="TButton"
        )
        apply_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(apply_btn, "Apply the selected statistical stretch parameters to the current image. Changes can be undone using Siril's undo function.")

    def toggle_curves_boost(self):
        if self.apply_curve_var.get():
            self.curves_boost_scale.state(['!disabled'])
        else:
            self.curves_boost_scale.state(['disabled'])

    def stretch_mono_image(self, fit, target_median, normalize=False, apply_curves=False, curves_boost=0.0):
        # Calculate black point
        black_point = max(np.min(fit.data), np.median(fit.data) - 2.7 * np.std(fit.data))

        # Rescale image
        rescaled_image = (fit.data - black_point) / (1 - black_point)
        median_image = np.median(rescaled_image)

        # Stretch image
        stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)

        # Optional normalization
        if normalize:
            stretched_image = stretched_image / np.max(stretched_image)

        # Optional curve boost
        if apply_curves:
            stretched_image = np.clip(stretched_image, 0, None) # Make sure no negative pixels are used
            stretched_image = np.power(stretched_image, 1.0 + curves_boost)

        return stretched_image

    def stretch_color_image(self, fit, target_median, linked=True, normalize=False, apply_curves=False, curves_boost=0.0):
        channels, height, width = fit.data.shape

        if linked:
            combined_median = np.median(fit.data)
            combined_std = np.std(fit.data)
            black_point = max(np.min(fit.data), combined_median - 2.7 * combined_std)

            rescaled_image = (fit.data - black_point) / (1 - black_point)
            median_image = np.median(rescaled_image)

            stretched_image = ((median_image - 1) * target_median * rescaled_image) / (median_image * (target_median + rescaled_image - 1) - target_median * rescaled_image)
        else:
            stretched_image = np.zeros_like(fit.data)

            for channel in range(channels):
                channel_data = fit.get_channel(channel)

                black_point = max(
                    np.min(channel_data),
                    np.median(channel_data) - 2.7 * np.std(channel_data)
                )

                rescaled_channel = (channel_data - black_point) / (1 - black_point)

                median_channel = np.median(rescaled_channel)

                stretched_channel = ((median_channel - 1) * target_median * rescaled_channel) / (
                    median_channel * (target_median + rescaled_channel - 1) - target_median * rescaled_channel
                )

                stretched_image[channel, ...] = stretched_channel

            if normalize:
                stretched_image = stretched_image / np.max(stretched_image)

            if apply_curves:
                stretched_image = np.clip(stretched_image, 0, None) # Make sure no negative pixels are used
                stretched_image = np.power(stretched_image, 1.0 + curves_boost)

        return stretched_image

    def apply_changes(self, from_cli=False):
        try:
            # Get the thread
            if self.siril.claim_thread():
                # Determine parameters: prefer CLI args if provided, else use GUI values
                if from_cli and self.cli_args:
                    target_median = self.cli_args.median
                    linked_stretch = self.cli_args.linked
                    normalize = self.cli_args.normalize
                    apply_curves = self.cli_args.boost is not None and self.cli_args.boost > 0
                    curves_boost = self.cli_args.boost or 0.0
                else:
                    target_median = self.target_median_var.get()
                    linked_stretch = self.linked_stretch_var.get()
                    normalize = self.normalize_var.get()
                    apply_curves = self.apply_curve_var.get()
                    curves_boost = self.curves_boost_var.get() if apply_curves else 0.0

                # Get current image
                fit = self.siril.get_image()
                fit.ensure_data_type(s.DataType.FLOAT_IMG)

                # Save original image for undo
                self.siril.undo_save_state(f"StatStretch: m={target_median:.2f} l={linked_stretch} n={normalize} c={apply_curves} b={curves_boost:.2f}")

                # Apply stretch based on image type
                if fit.data.ndim == 3:
                    stretched_image = self.stretch_color_image(
                        fit, target_median, linked_stretch, normalize, apply_curves, curves_boost
                    )
                else:
                    stretched_image = self.stretch_mono_image(
                        fit, target_median, normalize, apply_curves, curves_boost
                    )

                # Clip and update image data
                fit.data[:] = np.clip(stretched_image, 0, 1)
                self.siril.set_image_pixeldata(fit.data)

                if from_cli:
                    print("Statistical stretch applied successfully.")

        except Exception as e:
            if from_cli:
                print(f"Error: {str(e)}")
            else:
                messagebox.showerror("Error", str(e))

        finally:
            # Release the thread in the finally: block so that it is guaranteed to be released
            self.siril.release_thread()

    def close_dialog(self):
        self.siril.disconnect()
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Statistical Stretch for Astronomical Images")
    parser.add_argument("-median", type=float, default=0.2,
                        help="Target median value for stretch (0.0 to 1.0)")
    parser.add_argument("-boost", type=float, default=0.0,
                        help="Curves boost value (0.0 to 0.5)")
    parser.add_argument("-linked", action="store_true",
                        help="Use linked stretch for color images")
    parser.add_argument("-normalize", action="store_true",
                        help="Normalize image after stretch")

    args = parser.parse_args()

    try:
        if any([args.median != 0.2, args.boost != 0.0, args.linked, args.normalize]):
            # CLI mode
            app = StatisticalStretchInterface(cli_args=args)
        else:
            # GUI mode
            root = ThemedTk()
            app = StatisticalStretchInterface(root)
            root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
