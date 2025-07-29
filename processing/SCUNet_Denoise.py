# -*- coding: UTF-8 -*-
# ==============================================================================

# ------------------------------------------------------------------------------
# Project: Python siril script to run SCUNet denoiser via spandrel
# using model from https://github.com/cszn/SCUNet
# and https://ubersmooth.com/
#
# ------------------------------------------------------------------------------
#    Author:  Nicolas CASTEL <nic.castel (at) gmail.com>
#
# This program is provided without any guarantee.
#
# The license is  LGPL-v3
# For details, see GNU General Public License, version 3 or later.
#                        "https://www.gnu.org/licenses/gpl.html"
# ------------------------------------------------------------------------------

import sys
import os
import asyncio
import sirilpy as s
import numpy as np
import urllib.request
import ssl
import tempfile
import math
import zipfile

# Check the module version is enough to provide TorchHelper
if not s.check_module_version('>=0.7.46'):
    print("Error: requires sirilpy module >= 0.7.46")
    sys.exit(1)

print("Warning: a significant size of packages are about to be downloaded and installed and it will take some time")

s.ensure_installed("ttkthemes")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril

# Determine the correct torch package based on OS and hardware,
# and ensure it is installed
s.TorchHelper().ensure_torch()
import torch

s.ensure_installed("spandrel")
from spandrel import ImageModelDescriptor, ModelLoader

VERSION = "1.0.3"

# list of models
models = [["SCUNet Color Real PSNR","https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth","best all around model but can be too aggressive on stars"],
    ["SCUNet Color Real GAN","https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth","less agreesive denoise"],
    ["SCUNet Color 15","https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_15.pth","gaussian noise level 15"],
    ["SCUNet Color 25","https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_25.pth","gaussian noise level 25"],
    ["SCUNet Color 50","https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_50.pth","gaussian noise level 50"],
    ["UberSmooth dso stars 0.1","https://ubersmooth.com/uberSmooth-dso-stars-v0.1.zip","pretty good on stars but too aggressive on Hii regions within galaxies"],
    ["UberSmooth dso stars 0.2","https://ubersmooth.com/uberSmooth-dso-stars-v0.2.zip","not as aggressive as UberSmooth 0.1, but also not great"],
    ["UberSmooth planetary 0.1","https://ubersmooth.com/uberSmooth-planetary-v0.1.zip","only denoise/deblur no extra star treatment"]]

# suppported for SCUNet : Nvidia GPU / Apple MPS / DirectML on Windows / CPU
def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("cuda acceleration used")
        return torch.device("cuda")
    if torch.backends.mps.is_available() :
        print("mps acceleration used")
        return torch.device("mps")
    else:
        print("cpu used")
        return torch.device("cpu")

def image_to_tensor(device: torch.device, img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img)
    return tensor.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return (np.rollaxis(tensor.cpu().detach().numpy(), 1, 4).squeeze(0)).astype(np.float32)

def image_inference_tensor(
    model: ImageModelDescriptor, tensor: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        return model(tensor)

def tile_process(device: torch.device, model: ImageModelDescriptor, data: np.ndarray, scale, tile_size, yield_extra_details=False):
        """
        Process data [height, width, channel] into tiles of size [tile_size, tile_size, channel],
        feed them one by one into the model, then yield the resulting output tiles.
        """

        tile_pad=144

        # [height, width, channel] -> [1, channel, height, width]
        data = np.rollaxis(data, 2, 0)
        data = np.expand_dims(data, axis=0)

        batch, channel, height, width = data.shape

        tiles_x = width // tile_size
        if tiles_x*tile_size < width:
            tiles_x+=1
        tiles_y = height // tile_size
        if tiles_y*tile_size < height:
            tiles_y+=1

        for i in range(tiles_x * tiles_y):
            x = math.floor(i/tiles_y)
            y = i % tiles_y

            if x<tiles_x-1:
                input_start_x = x * tile_size
            else:
                input_start_x = width - tile_size
            if y<tiles_y-1:
                input_start_y = y * tile_size
            else:
                input_start_y = height - tile_size

            if input_start_x < 0 : input_start_x = 0
            if input_start_y < 0 : input_start_y = 0

            input_end_x = min(input_start_x + tile_size, width)
            input_end_y = min(input_start_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32)

            output_tile = image_inference_tensor(model,image_to_tensor(device, input_tile))
            progress = (i+1) / (tiles_y * tiles_x)

            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + (input_tile_width * scale)
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + (input_tile_height * scale)

            output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

            output_tile = tensor_to_image(output_tile)

            if yield_extra_details:
                yield (output_tile, input_start_y, input_start_x, input_tile_width, input_tile_height, progress)
            else:
                yield output_tile

        yield None

class SirilScunet:
    def __init__(self, root):
        self.root = root
        self.root.title(f"SCUNet Denoise - v{VERSION}")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = s.SirilInterface()

        if not self.siril.connect():
            self.siril.error_messagebox("Failed to connect to Siril")
            self.close_dialog()
            return

        if not self.siril.is_image_loaded():
            self.siril.error_messagebox("No image loaded")
            self.close_dialog()
            return

        tksiril.match_theme_to_siril(self.root, self.siril)

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="SCUNet Denoise Settings",
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # Model Selection Frame
        model_frame = ttk.LabelFrame(main_frame, text="Model selection", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.model_var = tk.StringVar(value="https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth")
        for model in models:
            rbutton = ttk.Radiobutton(
                model_frame,
                text=model[0],
                variable=self.model_var,
                value=model[1]
            )
            rbutton.pack(anchor=tk.W, pady=2)
            tksiril.create_tooltip(rbutton, model[2])

        self.strength_var = tk.DoubleVar(value=0.5)
        # Strength slider for denoise
        strength_frame = ttk.Frame(main_frame)
        strength_frame.pack(fill=tk.X, padx=5, pady=5)
        strength_label = ttk.Label(strength_frame, text="Strength:")
        strength_label.pack(side=tk.LEFT, padx=5)
        strength_slider = ttk.Scale(
            strength_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.strength_var,
            length=200
        )
        strength_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.strength_value = ttk.Label(strength_frame, text="0.5")
        self.strength_value.pack(side=tk.RIGHT, padx=5)
        tksiril.create_tooltip(strength_slider, "Adjust the denoising strength. The "
                "result is a linear blend of the denoised image and the original, "
                "with the mixture of the two images controlled by the strength "
                "parameter.")
        self.strength_var.trace_add("write", self._update_strength_label)

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.close_dialog,
            style="TButton"
        )
        close_btn.pack(side=tk.LEFT, padx=5)

        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self._on_apply,
            style="TButton"
        )
        apply_btn.pack(side=tk.LEFT, padx=5)

    def _update_strength_label(self, *args):
        """Update the strength value label when slider changes"""
        self.strength_value.config(text=f"{self.strength_var.get():.2f}")

    def _on_apply(self):
        # Wrap the async method to run in the event loop
        self.root.after(0, self._run_async_task)

    def _run_async_task(self):
        asyncio.run(self._apply_changes())

    def close_dialog(self):
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()

    async def _apply_changes(self):
        try:
            print("SCUNetDenoise:begin")

             # Read user input values
            model = self.model_var.get()
            strength = self.strength_var.get()

            self.siril.reset_progress()

            modelpath = os.path.join(self.siril.get_siril_userdatadir(),os.path.basename(model))

            if os.path.isfile(modelpath) :
                print("model found : "+modelpath)
            else :
                ssl._create_default_https_context = ssl._create_stdlib_context
                urllib.request.urlretrieve(model, modelpath)
                print("model downloaded : "+modelpath)

            if(zipfile.is_zipfile(modelpath)) :
                with zipfile.ZipFile(modelpath, 'r') as zip_ref:
                    zip_ref.extractall()
                    modelpath = modelpath.replace(".zip",".pth")
                    print("model extracted : "+modelpath)

            device = get_device()

            # load a model from disk
            model = ModelLoader().load_from_file(r""+modelpath).eval().to(device)
            # make sure it's an image to image model
            assert isinstance(model, ImageModelDescriptor)

            self.siril.update_progress("model initialised",0.05)
            image  = self.siril.get_image()

            # convert pixel data to 32 bit if 16bit mode is used
            original_data = image.data
            original_dtype = original_data.dtype
            if original_dtype == np.uint16:
                pixel_data = original_data.astype(np.float32) / 65535.0
            else:
                pixel_data = original_data

            # read image out send it to the GPU
            # Handle planar format (c, h, w) -> (h, w, c)
            pixel_data = np.transpose(pixel_data, (1, 2, 0))

            tile_size = 512
            scale = 1

            # Allocate an image to save the tiles
            imgresult = pixel_data.copy()

            for i, tile in enumerate(tile_process(device, model, pixel_data, scale, tile_size, yield_extra_details=True)):

                if tile is None:
                    break

                tile_data, x, y, w, h, p = tile
                if w != 0 and h != 0 :
                    imgresult[x*scale:x*scale+tile_size*scale,y*scale:y*scale+tile_size*scale] = tile_data

                self.siril.update_progress("Image denoising ongoing",p)

            # Convert back to planar format
            output_image = np.transpose(imgresult, (2, 0, 1))

            # Scale back if needed
            if original_dtype == np.uint16:
                output_image = output_image * 65535.0
                output_image = output_image.astype(np.uint16)

            # blend
            if strength != 1.0:
                original_dtype = output_image.dtype
                blended = output_image * strength + \
                        original_data * (1 - strength)
                if blended.dtype != original_dtype:
                    blended = blended.astype(original_dtype)
            else:
                blended = output_image

            self.siril.undo_save_state("SCUnet denoise")
            with self.siril.image_lock(): self.siril.set_image_pixeldata(blended)

            self.siril.update_progress("Image denoised",1.0)

        except Exception as e:
            print(f"Error in apply_changes: {str(e)}")
            self.siril.update_progress(f"Error: {str(e)}", 0)
        finally:
            print("SCUNetDenoise:end")

def main():
    try:
        root = ThemedTk()
        app = SirilScunet(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
