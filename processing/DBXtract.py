# DBXtract v1.0.0 (May 2025)
# (c) Raúl Hussein - Astrocitas
# SPDX-License-Identifier: GPL-3.0-or-later
# This script allows you to extract the Sulfur II, Hydrogen Alpha and Oxygen III signal from dual band filters to compose a SHO images in color cameras
#
# YouTube https://www.youtube.com/@astrocitas
# Instagram https://www.instagram.com/rahusga/
# Astrobin https://app.astrobin.com/u/rahusga
 
# === BLOCK 1: IMPORTS AND CONFIGURATION ===
import sirilpy as s
s.ensure_installed("ttkthemes", "tiffile", "astropy")

import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from sirilpy import tksiril
import os
import configparser
from astropy.io import fits

filetypes = []
if sys.platform.startswith("linux") and s.check_module_version(">=0.6.0"):
    import sirilpy.tkfilebrowser as filedialog
    filetypes = [("FITS/XISF files", "*.fit|*.fits|*.xisf|*.fts|*.tif")]        
else:
    from tkinter import filedialog
    filetypes = [("FITS/XISF files", "*.fit *.fits *.xisf *.fts *.tif")]    

VERSION = "1.0.0"

SENSORS = {
    "IMX 571": {"r1": 0.02, "r2": 0.82, "r3": 0.75, "g1": 0.85, "g2": 0.08, "g3": 0.08, "b1": 0.50, "b2": 0.02, "b3": 0.03},
    "IMX 294": {"r1": 0.03, "r2": 0.65, "r3": 0.63, "g1": 0.92, "g2": 0.16, "g3": 0.18, "b1": 0.50, "b2": 0.05, "b3": 0.08},
    "IMX 533": {"r1": 0.03, "r2": 0.80, "r3": 0.73, "g1": 0.92, "g2": 0.16, "g3": 0.18, "b1": 0.50, "b2": 0.05, "b3": 0.08},
    "IMX 585": {"r1": 0.07, "r2": 1.00, "r3": 0.95, "g1": 0.80, "g2": 0.20, "g3": 0.24, "b1": 0.40, "b2": 0.05, "b3": 0.08},
    "IMX 183": {"r1": 0.05, "r2": 0.77, "r3": 0.68, "g1": 0.92, "g2": 0.15, "g3": 0.18, "b1": 0.45, "b2": 0.05, "b3": 0.08},
    "IMX 071": {"r1": 0.05, "r2": 0.75, "r3": 0.68, "g1": 0.70, "g2": 0.10, "g3": 0.13, "b1": 0.45, "b2": 0.03, "b3": 0.05},
    "IMX 410": {"r1": 0.08, "r2": 0.80, "r3": 0.75, "g1": 0.93, "g2": 0.15, "g3": 0.18, "b1": 0.45, "b2": 0.10, "b3": 0.12},
    "IMX 178": {"r1": 0.05, "r2": 0.78, "r3": 0.68, "g1": 0.93, "g2": 0.16, "g3": 0.18, "b1": 0.50, "b2": 0.05, "b3": 0.08},
    "IMX 455": {"r1": 0.03, "r2": 0.65, "r3": 0.58, "g1": 0.68, "g2": 0.06, "g3": 0.08, "b1": 0.40, "b2": 0.02, "b3": 0.03},
    "IMX 094": {"r1": 0.05, "r2": 0.80, "r3": 0.68, "g1": 0.68, "g2": 0.09, "g3": 0.11, "b1": 0.45, "b2": 0.02, "b3": 0.03},
    "IMX 462": {"r1": 0.05, "r2": 0.81, "r3": 0.79, "g1": 0.78, "g2": 0.25, "g3": 0.30, "b1": 0.40, "b2": 0.11, "b3": 0.15},
    "IMX 662": {"r1": 0.05, "r2": 0.88, "r3": 0.82, "g1": 0.92, "g2": 0.36, "g3": 0.35, "b1": 0.40, "b2": 0.05, "b3": 0.07}
}

def print_info(self, msg, process=0):
    print(f"[DBXtract] {msg}")
    self.update_progress(msg, process) 

def save_fits(data, path, original_header=None, history_text=None):
    if data.dtype not in (np.float32, np.uint16):
        data = data.astype(np.float32)

    header = fits.Header()

    if original_header is not None:
        try:
            for key, value in original_header.items():
                # Avoid keys that define dimensions that might not match
                if key.upper().startswith("NAXIS") or key in ("SIMPLE", "BITPIX", ""):
                    continue
                try:
                    header[key] = value
                except Exception:
                    continue # Ignore wrong keys
        except Exception as e:
            print(f"[DBXtract] Warning: header fallback due to: {e}")
            header = fits.Header()
    
    header["CREATOR"] = f"Siril DBXtract v{VERSION}"
    if history_text:
        header["HISTORY"] = history_text

    fits.writeto(path, data, header, overwrite=True)

def extract_background(channel):
    return np.median(channel)

# === BLOCK 2: HO, SO processing and OIII combination ===
def extract_HA(data, coef):
    r, g, b = data
    bg_r = extract_background(r)
    bg_g = extract_background(g)
    bg_b = extract_background(b)
    r, g, b = r - bg_r, g - bg_g, b - bg_b

    cota = min(coef['g2']/coef['r2'], 0.12)
    OIII_G = (g - cota * r) / (coef['g1'] - coef['g2'] * coef['r1'] / coef['r2'])
    OIII_B = (b - coef['b2'] * r / coef['r2']) / (coef['b1'] - coef['b2'] * coef['r1'] / coef['r2'])
    OIII = ((2 * coef['g1'] * OIII_G) + (coef['b1'] * OIII_B)) / (2 * coef['g1'] + coef['b1']) + max(bg_b, bg_g)
    HA = (r - coef['r1'] * (OIII - max(bg_b, bg_g))) / coef['r2'] + (bg_r + max(bg_b, bg_g))
    return OIII, HA

def extract_SII(data, coef):    
    r, g, b = data
    bg_r = extract_background(r)
    bg_g = extract_background(g)
    bg_b = extract_background(b)
    r, g, b = r - bg_r, g - bg_g, b - bg_b

    cota = min(coef['g3']/coef['r3'], 0.12)
    OIII_G = (g - cota * r) / (coef['g1'] - coef['g3'] * coef['r1'] / coef['r3'])
    OIII_B = (b - coef['b3'] * r / coef['r3']) / (coef['b1'] - coef['b3'] * coef['r1'] / coef['r3'])
    OIII = ((2 * coef['g1'] * OIII_G) + (coef['b1'] * OIII_B)) / (2 * coef['g1'] + coef['b1']) + max(bg_b, bg_g)
    SII = (r - coef['r1'] * (OIII - max(bg_b, bg_g))) / coef['r3'] + (bg_r + max(bg_b, bg_g))
    return OIII, SII

def combine_OIII_adaptive(OIII_ho, OIII_so):    
    mean1, std1 = np.mean(OIII_ho), np.std(OIII_ho)
    mean2, std2 = np.mean(OIII_so), np.std(OIII_so)
    norm1 = (OIII_ho - mean1) / std1
    norm2 = (OIII_so - mean2) / std2
    combined = ((norm1 + norm2) / 2) * ((std1 + std2)/2) + ((mean1 + mean2)/2)
    return combined
    
# === BLOCK 3: DBXtractInterface main class (start) ===
class DBXtractInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"DBXtract v{VERSION}")
        #self.root.geometry("250x640") 
        self.root.resizable(False, False)
        self.style = tksiril.standard_style()
        self.siril = s.SirilInterface()                              

        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            self.siril.error_messagebox("Failed to connect to Siril")
            self.close_dialog()
            return
            
        try:
            self.siril.cmd("requires", "1.4.0-beta1")
        except s.CommandError:
            messagebox.showerror("Error", "Siril version requirement not met: 1.4.0-beta1 or higher is required")
            self.close_dialog()
            return
        
        self.sensor = tk.StringVar()
        self.load_last_sensor()

        self.use_loaded_ho = tk.BooleanVar(value=True)
        self.use_loaded_so = tk.BooleanVar(value=False)
        self.ho_file = None
        self.so_file = None        

        self.output_dir = os.path.join(os.getcwd(), "dbxtract_output")
        os.makedirs(self.output_dir, exist_ok=True)

        tksiril.match_theme_to_siril(self.root, self.siril)
        self.build_gui()

    def load_last_sensor(self):
        config = configparser.ConfigParser()
        if os.path.exists("dbxtract.ini"):
            config.read("dbxtract.ini")
            self.sensor.set(config.get("DEFAULT", "last_sensor", fallback="IMX 571"))
        else:
            self.sensor.set("IMX 571")

    def save_last_sensor(self):
        config = configparser.ConfigParser()
        config["DEFAULT"] = {"last_sensor": self.sensor.get()}
        with open("dbxtract.ini", "w") as f:
            config.write(f)


# === BLOCK 4: GUI AND MAIN CLASS ===
    def build_gui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Sensor selector (fila horizontal)
        sensor_row = ttk.Frame(frame)
        sensor_row.pack(fill=tk.X, pady=5)
        ttk.Label(sensor_row, text="Sensor:").pack(side=tk.LEFT)
        ttk.Combobox(sensor_row, textvariable=self.sensor, values=list(SENSORS.keys()), state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # HO group
        ho_frame = ttk.LabelFrame(frame, text="Ha + OIII (HO)", padding=10)
        ho_frame.pack(fill=tk.X, pady=10)
        ttk.Checkbutton(ho_frame, text="Use Siril Image preview", variable=self.use_loaded_ho).pack(anchor="w")
        ttk.Button(ho_frame, text="Select file...", command=self.select_ho_file).pack(anchor="w", pady=5)
        self.ho_label = ttk.Label(ho_frame, text="No file selected")
        self.ho_label.pack(anchor="w", padx=4)

        # SO group
        so_frame = ttk.LabelFrame(frame, text="SII + OIII (SO)", padding=10)
        so_frame.pack(fill=tk.X, pady=10)
        ttk.Checkbutton(so_frame, text="Use Siril Image preview", variable=self.use_loaded_so).pack(anchor="w")
        ttk.Button(so_frame, text="Select file...", command=self.select_so_file).pack(anchor="w", pady=5)
        self.so_label = ttk.Label(so_frame, text="No file selected")
        self.so_label.pack(anchor="w", padx=4)

        # Bottom buttons
        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=20)
        ttk.Button(button_row, text="Close", command=self.close_dialog).pack(side=tk.LEFT, padx=(0, 10))   
        ttk.Button(button_row, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=(0, 10))        
        ttk.Button(button_row, text="Extract", command=self.process).pack(side=tk.RIGHT)    

        # Progress message label
        self.progress_var = tk.StringVar(value="")
        progress_label = ttk.Label(frame, textvariable=self.progress_var)
        progress_label.pack(pady=5)
        

    def close_dialog(self):
        """ Close dialog """
        self.root.quit()
        self.root.destroy()
    
    def update_progress(self, message, progress=0):
        self.progress_var.set(message)
        self.root.update_idletasks()
        self.siril.update_progress(message, progress)

    def show_help(self):
        self.siril.info_messagebox("DBXtract adds on top of Siril’s built-in functionality a automatic extraction of narrowband emission lines from OSC images,"
        "specifically designed for dual-band filters (e.g. Ha+OIII or SII+OIII)."
        "\n\nIs sensor-aware extraction. Uses predefined quantum efficiency (QE) parameters per sensor (IMX571, IMX533, etc.)."
        "These QEs are used to weight RGB channels for physically accurate extraction. It also takes into account the pedestal value,"
        "ensuring greater accuracy when calculating each channel's contributions to the broadcast band."
        "\n\nFinally, combination of OIII from HO and SO sources. Implements adaptive normalization and scaling to combine OIII from two sources."
        "\n\nIn summary, the result of this is surprising and makes it much easier for the user to implement it through complex formulas in pixelmath.", True)
        tksiril.elevate(self.root)        

    def select_ho_file(self):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.ho_file = path
            self.use_loaded_ho.set(False)
            self.ho_label.config(text=os.path.basename(path))

    def select_so_file(self):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.so_file = path
            self.use_loaded_so.set(False)
            self.so_label.config(text=os.path.basename(path))

    def get_image_data_from_file(self, path):
        path = path.lower()

        # FITS o FIT
        if path.endswith((".fit", ".fits")):
            with fits.open(path) as hdul:
                data = hdul[0].data
                
                if data.ndim == 2:
                    raise Exception("DBXtract cannot work with non-debayered images.")
                elif data.ndim != 3:
                    raise Exception("The image is not RGB.")
                
                header = hdul[0].header.copy()
                return data, header

        # XISF
        elif path.endswith(".xisf"):
            self.siril.cmd(f"load {path}")
            data = self.siril.get_image().data
            
            if data.ndim == 2:
                raise Exception("DBXtract cannot work with non-debayered images.")
            elif data.ndim != 3:
                raise Exception("The image is not RGB.")
                    
            try:
                header = self.siril.get_image_fits_header()
            except:
                header = None # No header available
            return data, header

        # TIFF, PNG, JPEG, etc.
        else:
            try:
                self.siril.cmd(f"load {path}")
                data = self.siril.get_image().data
                
                if data.ndim == 2:
                    raise Exception("DBXtract cannot work with non-debayered images.")
                elif data.ndim != 3:
                    raise Exception("The image is not RGB.")
                    
                try:
                    header = self.siril.get_image_fits_header()
                except:
                    header = None  # No header available
                return data, header
            except Exception as e:
                self.siril.log(f"Error reading file {path}: {e}", s.LogColor.RED)
                return None, None
            
# === BLOCK 5: Processing and main ===
    def process(self):
        try:                                    
            print_info(self, "Extraction started.")            
            coef = SENSORS[self.sensor.get()]
            self.save_last_sensor()

            if self.use_loaded_ho.get():
                data_HO = self.siril.get_image().data
                
                if data_HO.ndim == 2:
                    raise Exception("DBXtract cannot work with non-debayered images.")
                elif data_HO.ndim != 3:
                    raise Exception("The image is not RGB.")
                    
                header_HO = None
            elif self.ho_file:
                data_HO, header_HO = self.get_image_data_from_file(self.ho_file)
            else:
                data_HO = header_HO = None

            if self.use_loaded_so.get():
                data_SO = self.siril.get_image().data
                
                if data_SO.ndim == 2:
                    raise Exception("DBXtract cannot work with non-debayered images.")
                elif data_SO.ndim != 3:
                    raise Exception("The image is not RGB.")
                    
                header_SO = None
            elif self.so_file:
                data_SO, header_SO = self.get_image_data_from_file(self.so_file)
            else:
                data_SO = header_SO = None

            oiii_list = []

            if data_HO is not None:
                print_info(self, "HA and OIII calculating.", 0.3)                       
                oiii_ho, ha = extract_HA(data_HO, coef)
                save_fits(oiii_ho, os.path.join(self.output_dir, "dbxtract_OIII_HO.fit"), header_HO)
                save_fits(ha, os.path.join(self.output_dir, "dbxtract_HA.fit"), header_HO)
                oiii_list.append(oiii_ho)

            if data_SO is not None:
                print_info(self, "SII and OIII calculating.", 0.6)                
                oiii_so, sii = extract_SII(data_SO, coef)
                save_fits(oiii_so, os.path.join(self.output_dir, "dbxtract_OIII_SO.fit"), header_SO)
                save_fits(sii, os.path.join(self.output_dir, "dbxtract_SII.fit"), header_SO)
                oiii_list.append(oiii_so)

            if not oiii_list:
                raise Exception("No valid HO or SO image loaded.")

            if len(oiii_list) == 2:
                print_info(self, "Combine two OIII images.", 0.9)
                oiii_final = combine_OIII_adaptive(*oiii_list)
            else:
                oiii_final = oiii_list[0]
            
            final_header = header_HO if header_HO else header_SO
            save_fits(oiii_final, os.path.join(self.output_dir, "dbxtract_OIII.fit"), final_header)
            
            print_info(self, "Extraction completed successfully", 1)            
            self.siril.cmd(f"load {os.path.join(self.output_dir, 'dbxtract_OIII.fit')}")            
            self.siril.reset_progress()         
            
        except Exception as e:
            print_info(self, f"ERROR: {str(e)}")            

def main():
    root = ThemedTk(theme="arc")
    app = DBXtractInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()


