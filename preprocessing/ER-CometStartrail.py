# Jun. 16, 2025
# (c) Eduardo Ramírez - @edramigon
# SPDX-License-Identifier: GPL-3.0-or-later
# Script for startrails with comet effect.
# 
# Instagram https://www.instagram.com/edramigon/
# TikTok https://www.tiktok.com/@edramigon

import sirilpy as s
s.ensure_installed("ttkthemes")
s.ensure_installed("astropy")
s.ensure_installed("scipy")

import os
import shutil
import time
import sys
import threading
import queue
import gc
import re
import tkinter as tk
import numpy as np
from scipy.ndimage import grey_dilation, gaussian_filter
from astropy.io import fits
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from sirilpy import LogColor, tksiril
from datetime import datetime

if s.check_module_version(">=0.6.0") and sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog

VERSION = "1.0.4"

# 1.0.0 Original release by Eduardo Ramírez
# 1.0.1 Improves cross-platform display consistency and adds support for sequences.
# 1.0.2 Now includes the ability to retain the starless frame.
# 1.0.3 Add support for monochromatic sequences.
# 1.0.4 Fixes a bug that caused the entire interface to remain disabled, and prevents an error when selecting a sequence with registration data

def safe_run(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.siril.log(f"Error in {func.__name__}: {str(e)}", color=LogColor.RED)
            self.enable_widget(self.work_frame)
            self.enable_widget(self.starmask_frame, False)
            self.enable_widget(self.final_frame, False)
            self.enable_widget(self.initial_frame, False)
            self.enable_widget(self.submit_button,False)
    return wrapper

class CometStarTrailsInterface:
    def __init__(self, root):
        self.root = root
        self.fits_checkbox_state = tk.IntVar(value=0)
        self.starless_checkbox_state = tk.IntVar(value=0)
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        self.root.resizable(False, False)
        self.root.title(f"ER-Comet Startrails - v{VERSION}")

        # Connect to Siril & match theme
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except s.SirilConnectionError as e:
            self.siril.log("Connection failed: {e}", color=LogColor.RED)
            sys.exit()

        tksiril.match_theme_to_siril(self.root, self.siril)
        self.siril.log("Connected successfully!", color=LogColor.GREEN)

        # Initial checks
        if not self.initial_checks():
            self.root.withdraw()
            sys.exit()
        
        # Initial vars
        self.date = 0
        self.time = 0
        self.bitpix = np.uint16
        self.convert_in_progress = False
        self.effect_queue = queue.Queue()
        self.effect_thread = None
        self.mask_gain = 0
        
        # Create GUI
        self.create_widgets()
        self.enable_widget(self.starmask_frame, False)
        self.enable_widget(self.initial_frame, False)
        self.enable_widget(self.final_frame, False)
        self.enable_widget(self.submit_button,False)
        self.work_path_var.trace_add('write', self.check_sequences)
        self.work_path_var.set(self.siril.get_siril_wd())
        self.stop_event=threading.Event()
        self.root.protocol("WM_DELETE_WINDOW", self.run_close)

    def initial_checks(self):
    
        # Check Siril version
        require_version = "1.3.6"
        self.default_ext = self.siril.get_siril_config("core", "extension")  # Default FITS extension
        try:
            self.siril.cmd("requires", require_version)
        except:
            self.siril.error_messagebox(f"This script requires Siril version {require_version} or later!")
            return False
        
        return True

    def create_widgets(self):
     
        # Create Frame
        frame = ttk.Frame(self.root)
        frame.pack(expand=True, fill="both")

        # Title
        title_label = ttk.Label(frame, text="Comet Startrails by @edramigon")
        title_label.pack(pady=(10, 10))

        # Create Frame for Work Path
        self.work_frame = ttk.LabelFrame(frame, text="Work Path", padding=10)
        self.work_frame.pack(fill="x", padx=10, pady=(0,5))
        
        # Create Frame for Folder Bowser
        path_frame = ttk.Frame(self.work_frame, padding=10)
        path_frame.pack(fill="x", pady=(0,5))        
        
        self.work_path_var = tk.StringVar(value="")
        work_path_entry = ttk.Entry(
            path_frame,
            textvariable=self.work_path_var,
            width=40
        )
        work_path_entry.pack(side=tk.LEFT, padx=(0, 5), expand=True)

        ttk.Button(
            path_frame,
            text="Browse",
            command=self._browse_folder,
            style="TButton"
        ).pack(side=tk.LEFT)
        tksiril.create_tooltip(work_path_entry, "Set the work directory")
        
        # Create Frame for Sequence
        self.sequence_frame = ttk.Frame(self.work_frame, padding=10)
        self.sequence_frame.pack(fill="x", pady=(0,5))
        
        #  Sequence combobox
        self.sequence_combobox = ttk.Combobox(
            self.sequence_frame,
            values=[],
            state="readonly",
        )
        self.sequence_combobox.pack(side=tk.LEFT, padx=(0,5),expand=True, fill="x")
        
        # Select Button
        ttk.Button(
            self.sequence_frame,
            text="Select",
            command=self.run_prepare_sequence_thread,
            style="TButton"
        ).pack(side=tk.LEFT, padx=(0,5))
        
        # or Label        
        self.or_label = ttk.Label(self.sequence_frame, text="or")
        self.or_label.pack(side="left", padx=(0,5))
        
        # Raw Button
        ttk.Button(
            self.sequence_frame,
            text="RAW Folder",
            command=self.run_convert_thread,
            style="TButton"
        ).pack(side=tk.RIGHT)
        
        # Create Frame for starmask
        self.starmask_frame = ttk.LabelFrame(frame, text="Starmask", padding=10)
        self.starmask_frame.pack(fill="x", padx=10, pady=(0,5))
        
        # Slider Text Label
        self.starmask_slider_text_label = ttk.Label(self.starmask_frame, text="Adjust the mask: ")
        self.starmask_slider_text_label.pack(side="left", padx=(0, 5))

        # Slider setup
        self.starmask_slider_value = tk.IntVar(value=120)  

        # Create Slider
        self.starmask_slider = ttk.Scale(self.starmask_frame, from_=100, 
                    to=200, orient="horizontal", variable=self.starmask_slider_value)
        self.starmask_slider.pack(side="left", expand=True, fill="x")

        # Slider Value Label
        self.starmask_slider_value_label = ttk.Label(self.starmask_frame, text="120")
        self.starmask_slider_value_label.pack(side="left", padx=(5, 0))

        # Set Slider value & update
        self.starmask_slider.set(120)
        self.starmask_slider_value.trace_add("write", self.update_starmask_slider_value)

        # Create Frame for initial effect
        self.initial_frame = ttk.LabelFrame(frame, text="Initial Effect", padding=10)
        self.initial_frame.pack(fill="x", padx=10, pady=(0,5))
        
         # Create Frame for initial mode
        self.initial_mode_frame = ttk.Frame(self.initial_frame)
        self.initial_mode_frame.pack(fill="x", pady=(0, 5))
        
        self.initial_modes_var = tk.StringVar(value="None")
        initial_modes = ["Soften", "None", "Enhance"]
        for mode in initial_modes:
            ttk.Radiobutton(
                self.initial_mode_frame,
                text=mode,
                variable=self.initial_modes_var,
                value=mode
            ).pack(side="left", expand=True, fill="x")
        
        
        # Create Frame for initial frames
        self.initial_images_frame = ttk.Frame(self.initial_frame)
        self.initial_images_frame.pack(fill="x", pady=(0, 5))

        # Slider Text Label
        self.initial_slider_text_label = ttk.Label(self.initial_images_frame, text="Images to apply: ")
        self.initial_slider_text_label.pack(side="left", padx=(0, 5))

        # Slider setup
        self.initial_slider_value = tk.IntVar(value=1)  

        # Create Slider
        self.initial_slider = ttk.Scale(self.initial_images_frame, from_=1, 
                    to=2, orient="horizontal", variable=self.initial_slider_value)
        self.initial_slider.pack(side="left", expand=True, fill="x")

        # Slider Value Label
        self.initial_slider_value_label = ttk.Label(self.initial_images_frame, text="1")
        self.initial_slider_value_label.pack(side="left", padx=(5, 0))

        # Set Slider value & update
        self.initial_slider.set(1)
        self.initial_slider_value.trace_add("write", self.update_initial_slider_value)
        
        # Create Frame for initial effect
        self.final_frame = ttk.LabelFrame(frame, text="Final Effect", padding=10)
        self.final_frame.pack(fill="x", padx=10, pady=(0,5))
        
         # Create Frame for final mode
        self.final_mode_frame = ttk.Frame(self.final_frame)
        self.final_mode_frame.pack(fill="x", pady=(0, 5))
        
        self.final_modes_var = tk.StringVar(value="None")
        self.final_modes = ["Soften", "None", "Enhance"]
        for mode in self.final_modes:
            ttk.Radiobutton(
                self.final_mode_frame,
                text=mode,
                variable=self.final_modes_var,
                value=mode
            ).pack(side="left", expand=True, fill="x")
        
        
        # Create Frame for final frames
        self.final_images_frame = ttk.Frame(self.final_frame)
        self.final_images_frame.pack(fill="x", pady=(0, 5))

        # Slider Text Label
        self.final_slider_text_label = ttk.Label(self.final_images_frame, text="Images to apply: ")
        self.final_slider_text_label.pack(side="left", padx=(0, 5))

        # Slider setup
        self.final_slider_value = tk.IntVar(value=1)  

        # Create Slider
        self.final_slider = ttk.Scale(self.final_images_frame, from_=1, 
                    to=2, orient="horizontal", variable=self.final_slider_value)
        self.final_slider.pack(side="left", expand=True, fill="x")

        # Slider Value Label
        self.final_slider_value_label = ttk.Label(self.final_images_frame, text="1")
        self.final_slider_value_label.pack(side="left", padx=(5, 0))

        # Set Slider value & update
        self.final_slider.set(1)
        self.final_slider_value.trace_add("write", self.update_final_slider_value)
                
        
        # Create Frame for Options
        Options_checkbox_frame = ttk.LabelFrame(frame, text="Options", padding=10)
        Options_checkbox_frame.pack(fill="x", padx=10, pady=(0, 5), anchor="w")

        # Sequence Checkbox
        fits_checkbox = ttk.Checkbutton(Options_checkbox_frame, text="Keep Sequence", variable=self.fits_checkbox_state)
        fits_checkbox.pack(side="left")
        
        # Starless Checkbox
        starless_checkbox = ttk.Checkbutton(Options_checkbox_frame, text="Keep Starless frame", variable=self.starless_checkbox_state)
        starless_checkbox.pack(side="left")
        
        # Buttons
        self.submit_button = ttk.Button(frame, text="Apply", width=7, command=self.run_startrail_thread)
        self.submit_button.pack(padx=5, side="right")

        close_button = ttk.Button(frame, text="Close", width=7, command=self.run_close)
        close_button.pack(padx=5, side="right")
        
        help_button = ttk.Button(frame, text="Help", width=7, command=self.show_help)
        help_button.pack(padx=10, side="left")
        
    def enable_children(self, parent, enabled=True):
        for child in parent.winfo_children():
            wtype = child.winfo_class()
            if wtype not in ('Frame', 'Labelframe', 'TFrame', 'TLabelframe'):
                child.configure(state=tk.NORMAL if enabled else tk.DISABLED)
            else:
                self.enable_children(child, enabled)

    def enable_widget(self, widget, enabled=True):
        wtype = widget.winfo_class()
        if wtype not in ('Frame', 'Labelframe', 'TFrame', 'TLabelframe'):
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        else:
            self.enable_children(widget, enabled)
    
    def update_initial_slider_value(self, *args):
        self.initial_slider_value_label.config(text=self.initial_slider_value.get())
        self.final_slider.config(to=self.seq.number-self.initial_slider_value.get())
        
    def update_final_slider_value(self, *args):
        self.final_slider_value_label.config(text=self.final_slider_value.get())
        self.initial_slider.config(to=self.seq.number-self.final_slider_value.get())
        
    def update_starmask_slider_value(self, *args):
        self.starmask_slider_value_label.config(text=self.starmask_slider_value.get())
        
    def update_initial_widgets(self, *args):
        if self.initial_modes_var.get()=="None":
            self.enable_widget(self.initial_images_frame, False)
            self.final_slider.config(to=self.seq.number-1)
        else:
            self.enable_widget(self.initial_images_frame)
            self.final_slider.config(to=self.seq.number-self.initial_slider_value.get())

    def update_final_widgets(self, *args):
        if self.final_modes_var.get()=="None":
            self.enable_widget(self.final_images_frame, False)
            self.initial_slider.config(to=self.seq.number-1)
        else:
            self.enable_widget(self.final_images_frame)
            self.initial_slider.config(to=self.seq.number-self.final_slider_value.get())

    def _browse_folder(self):
        filename = filedialog.askdirectory(
            title="Select sequence Path",
            initialdir=self.work_path_var.get()
        )
        if filename:
            self.work_path_var.set(filename)
        
    @safe_run
    def _convert(self):
        self.enable_widget(self.work_frame, False)
        filename = filedialog.askdirectory(
            title="Select RAW Folder",
            initialdir=self.work_path_var.get()
        )
        if filename:
            path=filename
        else:
            self.siril.log("Selected directory not found. Aborting execution.", color=LogColor.RED)
            self.enable_widget(self.work_frame)
            return 0
        self.enable_widget(self.work_frame, False)
        if os.path.exists(path+"/ER_TEMP"):
        # Confirmation dialog, delete ER_TEMP Folder
            proceed = messagebox.askyesno(
                "Confirm conversion",
                "Remnants from a previous execution have been detected. If you continue, the 'ER_TEMP' subfolder and all of its contents will be deleted. Are you sure you want to proceed?",
                icon='warning'
            )
            if proceed:
                if os.path.exists(path+"/ER_TEMP"):
                    shutil.rmtree(path+"/ER_TEMP")
                self.wait_until_folder_exist(path+"/ER_TEMP")
            else:
                self.enable_widget(self.work_frame)
                return 0
        self.convert_in_progress = True
        os.mkdir(path+"/ER_TEMP")
        self.siril.cmd("cd", f"'{path}'")
        self.siril.cmd("convertraw", "startrails", f"'-out={path}/ER_TEMP'", "-debayer")
        self.n_files=self.countfiles(path+"/ER_TEMP")-1
        if self.n_files > 0:
            self.siril.cmd("cd", f"'{path}/ER_TEMP'")
            self.siril.create_new_seq("startrails_")
            self.siril.cmd("load_seq","startrails_")
            self.seq=self.siril.get_seq()
        else:
            return 0
        self.convert_in_progress = False
        self.run_mask_thread()
        
    def run_convert_thread(self):
        if self.convert_in_progress:
            return
        self.thread_convert = threading.Thread(target=self._convert, daemon=False)
        self.thread_convert.start()
    
    @safe_run
    def _applymask(self, *args):
        fit=self.siril.get_seq_frame(self.seq.current)
        if self.seq.bitpix==20:
            self.bitpix=np.uint16
        else:
            self.bitpix=np.float32
        self.initial_slider.config(to=self.seq.number-1)
        self.final_slider.config(to=self.seq.number-1)
        self.initial_modes_var.trace_add('write', self.update_initial_widgets)
        self.final_modes_var.trace_add('write', self.update_final_widgets)
        self.enable_widget(self.starmask_frame)
        self.enable_widget(self.initial_mode_frame)
        self.enable_widget(self.final_mode_frame)
        self.enable_widget(self.submit_button)
        messagebox.showinfo(
        "Star Mask Application",
            (
                "🔹 Adjust the red star mask overlay by moving the slider to control the luminance threshold.\n\n"
                "🔹 You can switch between frames using *Sequence → Frame List* in Siril.\n"
                "🔹 You may also include or exclude frames (note: this may create gaps in the star trails).\n"
            )
        )
        lastframe=self.seq.current
        lastslidervalue=None
        aux=fit.data.copy()
        while not self.stop_event.is_set():
            if self.siril.is_sequence_loaded():
                self.seq=self.siril.get_seq()
                if lastframe!=self.seq.current:
                    with self.siril.image_lock():
                        self.siril.set_seq_frame_pixeldata(lastframe, aux)
                        fit=self.siril.get_seq_frame(self.seq.current)
                        aux=fit.data.copy()
                        self.siril.set_seq_frame_pixeldata(self.seq.current,aux)
                        lastframe=self.seq.current
            if lastslidervalue!=self.starmask_slider_value.get():
                lastslidervalue=self.starmask_slider_value.get()
                R=aux[0].copy()
                G=aux[1].copy()
                B=aux[2].copy()
                lum=(0.299*R+0.587*G+0.114*B).copy()
                umbral=np.median(lum)*lastslidervalue/100
                mask=lum.copy()
                mask[lum<umbral]=0
                mask=gaussian_filter(mask, sigma=0.5)
                mask=mask/(np.max(mask)+1e-8)
                R=R*(1-mask)+65535*mask
                G=G*(1-mask)
                B=B*(1-mask)
                fit.data[0, ...]=R.copy()
                fit.data[1, ...]=G.copy()
                fit.data[2, ...]=B.copy()
                if self.siril.is_sequence_loaded():
                    with self.siril.image_lock():
                        self.siril.set_seq_frame_pixeldata(lastframe,fit.data)
            time.sleep(0.5)
            R=G=B=lum=mask=umbral=None
            del R, G, B, lum, mask
            gc.collect()
        if self.siril.is_sequence_loaded():
            with self.siril.image_lock():
                self.siril.set_seq_frame_pixeldata(lastframe, aux)
        path=fit=aux=None
        del path, fit, aux
        gc.collect()

    def run_mask_thread(self):
        self.stop_event.clear()
        self.threading_applymask = threading.Thread(target=self._applymask, daemon=False)
        self.threading_applymask.start()
    
    @safe_run
    def _prepare_sequence(self, path, sequence):
        self.enable_widget(self.work_frame, False)
        if os.path.exists(path+"/ER_TEMP"):
        # Confirmation dialog, delete ER_TEMP Folder
            proceed = messagebox.askyesno(
                "Confirm conversion",
                "Remnants from a previous execution have been detected. If you continue, the 'ER_TEMP' subfolder and all of its contents will be deleted. Are you sure you want to proceed?",
                icon='warning'
            )
            if proceed:
                if os.path.exists(path+"/ER_TEMP"):
                    shutil.rmtree(path+"/ER_TEMP")
                self.wait_until_folder_exist(path+"/ER_TEMP")
            else:
                self.enable_widget(self.work_frame)
                return 0
        self.siril.cmd("cd", f"{path}")
        self.siril.cmd("load_seq",f"{sequence}")
        self.seq=self.siril.get_seq()
        if self.seq.type!=0:
            self.siril.log("Invalid sequence type. Only work with sequences of FITS files - no SER, FITSEQ or AVI files")
            self.enable_widget(self.work_frame)
            return 0
        self.siril.log(f"Copying the sequence, please wait...", color=LogColor.BLUE)
        self.copy_seq(self.seq, path, f"{path}/ER_TEMP", "startrails_")
        self.siril.cmd("cd","./ER_TEMP")
        self.siril.cmd("seqclean",f"startrails_{sequence}","-reg")
        self.siril.cmd("load_seq",f"startrails_{sequence}")
        self.seq=self.siril.get_seq()
        self.run_mask_thread()
    
    def run_prepare_sequence_thread(self):
        path=self.work_path_var.get()
        sequence=self.sequence_combobox.get()
        self.threading_prepare_sequence = threading.Thread(target=lambda: self._prepare_sequence(path, sequence), daemon=False)
        self.threading_prepare_sequence.start()
    
    def check_sequences(self, *args):
        path=self.work_path_var.get()
        self.sequence_combobox.set("")
        self.sequence_combobox["values"]=[]
        sequences = set()
        files = os.listdir(path)
        for file in files:
            if file.lower().endswith('.seq'):
                sequences.add(os.path.splitext(file)[0])
        sequences = sorted(sequences)
        self.sequence_combobox["values"]=sequences
        if sequences:
            self.sequence_combobox.set(sequences[0])

    def countfiles(self, path):
        try:
            files = os.listdir(path)
            n_files = len([f for f in files if os.path.isfile(os.path.join(path, f))])
            del files
            gc.collect()
            return n_files
        except FileNotFoundError:
            print(f"Error: The directory '{path}' does exist.")
            return -1
        except Exception as e:
            print(f"Error: {e}")
            return -1
    
    def show_help(self):
        self.siril.info_messagebox("Stack a startrail from a sequence or a Folder with RAW files."
        " You can also apply effects at beginning and end of the trails and select"
        "the number of images where the effect is applied."
        "\n\n-Check 'Keep Sequence' to save the final sequence and the fits images in addition to the the final result ", True)
        tksiril.elevate(self.root)
    
    @safe_run    
    def run_startrail(self):
        try:
            self.enable_widget(self.starmask_frame,False)
            self.enable_widget(self.initial_frame,False)
            self.enable_widget(self.final_frame,False)
            self.enable_widget(self.submit_button,False)
            self.stop_event.set()
            while self.threading_applymask.is_alive():
                self.root.update()
                time.sleep(0.1)
            self.siril.cmd("close")
            self.siril.log(f"Starting to apply the effect...", color=LogColor.BLUE)
            self.siril.cmd("stack", f"{self.seq.seqname}", "med")
            name=self.seq.seqname
            if not name.endswith("_"):
                name += "_"
            os.rename(f"{self.work_path_var.get()}/ER_TEMP/{name}stacked{self.default_ext}", f"{self.work_path_var.get()}/ER_TEMP/starless{self.default_ext}")
            with fits.open(self.work_path_var.get()+"/ER_TEMP/starless"+self.default_ext) as hdul:
                self.starless_data = hdul[0].data.astype(np.float32).copy()
                if self.starless_data.max() >= 1.5:
                    self.starless_data /= 65535.0
            self.siril.cmd("load_seq", f"{name}")
            self.effect_thread = threading.Thread(target=self.apply_effect_worker)
            self.effect_thread.start()
            for i in range(self.seq.number):
                gc.collect()
                index=i+1
                if self.initial_modes_var.get()!="None":
                    if index <= self.initial_slider_value.get():
                        if self.initial_modes_var.get()=="Soften":
                            value=1-(self.initial_slider_value.get()-index)/self.initial_slider_value.get()
                        else:
                            value=1+((self.initial_slider_value.get()-index+1)/self.initial_slider_value.get())*2
                        self.effect_queue.put((value, index))
                if self.final_modes_var.get()!="None":
                    if index > (self.seq.number-self.final_slider_value.get()):
                        if self.final_modes_var.get()=="Soften":
                            value=(self.seq.number-index)/self.final_slider_value.get()
                        else:
                            value=1+(1-(self.seq.number-index)/self.final_slider_value.get())*2
                        self.effect_queue.put((value, index))
            index=value=None
            del index, value
            gc.collect()
            self.effect_queue.join()
            self.effect_queue.put(None)
            self.effect_thread.join()
            self.siril.log("Starting to stack...")
            time.sleep(1)
            self.siril.cmd("stack", f"{name}", "max", "-filter-incl")
            gc.collect()
            now=datetime.now()
            self.date=now.date().strftime("%Y-%m-%d")
            self.time=now.time().strftime("%H-%M-%S")
            if self.starless_checkbox_state.get() != 0:
                os.rename(f"{self.work_path_var.get()}/ER_TEMP/starless{self.default_ext}",(f"{self.work_path_var.get()}/starless_ER_startrails_{self.date}_{self.time}{self.default_ext}"))
            else:
                os.remove(f"{self.work_path_var.get()}/ER_TEMP/starless{self.default_ext}")
            os.rename(f"{self.work_path_var.get()}/ER_TEMP/{name}stacked{self.default_ext}",f"{self.work_path_var.get()}/ER_startrails_{self.date}_{self.time}{self.default_ext}")
            self.siril.cmd("cd", f"{self.work_path_var.get()}")
            self.siril.cmd("load", f"'{self.work_path_var.get()}/ER_startrails_{self.date}_{self.time}{self.default_ext}'")
            if self.fits_checkbox_state.get() != 0:
                if self.date!=0 and self.time!=0:
                    self.siril.log("Saving the final sequence and the fits images.", color=LogColor.GREEN)
                    os.rename(f"{self.work_path_var.get()}/ER_TEMP",f"{self.work_path_var.get()}/ER_Startrails_{self.date}_{self.time}")
            self.siril.log("Startrail is complete!", color=LogColor.GREEN)
            self.run_close()
            return True
        except Exception as e:
            self.siril.log(f"Error in run_startrail: {str(e)}", color=LogColor.RED)
            return False

    def run_startrail_thread(self):
        self.thread_startrail = threading.Thread(target=self.run_startrail, daemon=False)
        self.thread_startrail.start()

    @safe_run
    def apply_effect_worker(self):
        while True:
            task = self.effect_queue.get()
            if task is None:
                break  
            value, index = task  
            self.siril.log(f"Applying effect to frame nº{index}", color=LogColor.BLUE)
            print(self.siril.get_seq_frame_filename(index-1))
            with fits.open(self.siril.get_seq_frame_filename(index-1), mode='update') as hdul:
                data=hdul[0].data
                data_f=data.astype(np.float32).copy()
                if data_f.max()>=1.5:
                    data_f/=65535.0
                starless_data=self.starless_data.astype(np.float32).copy()
                aux=1-((1-((data_f-starless_data)*value))*(1-starless_data))
                aux=np.maximum(aux, starless_data)
                aux=np.clip(aux, 0.0, 1.0)
                if self.bitpix == np.uint16:
                    aux = np.clip(aux * 65535.0, 0, 65535).astype(np.uint16)
                    data_final = np.clip(data_f * 65535.0, 0, 65535).astype(np.uint16)
                else:
                    aux = aux.astype(np.float32)
                    data_final = data_f.astype(np.float32)
                if data_f.ndim < 3:
                    lum=data_f.copy()
                else:
                    lum=(0.299*data_f[0]+0.587*data_f[1]+0.114*data_f[2])
                umbral=np.median(lum)*self.starmask_slider_value.get()/100
                mask=lum.copy()
                mask[lum<umbral]=0
                mask=grey_dilation(mask, size=(3,3))
                mask=gaussian_filter(mask, sigma=1)
                mask=mask-np.min(mask)
                mask=mask/(np.max(mask)+1e-8)
                if self.mask_gain == 0:
                    keyarea=(lum>umbral).astype(np.float32)
                    keyvalues=mask[keyarea>0]
                    self.mask_gain=1.0/(np.min(keyvalues)*0.7+1e-8)
                mask=np.clip(mask*self.mask_gain, 0, 1)
                mask = np.broadcast_to(mask, data.shape)
                result=(data_final*(1-mask))+aux*mask
                if self.bitpix == np.uint16:
                    result = np.clip(result, 0, 65535).astype(np.uint16)
                else:
                    result = result.astype(np.float32)
                data[...]=result.astype(self.bitpix).copy()
                hdul.flush()
            time.sleep(0.5)
            data_f=data=starless_data=result=aux=lum=mask=umbral=scale=None
            del data_f, data, starless_data, result, aux, lum, mask, umbral, scale
            gc.collect()
            time.sleep(0.5)
            self.effect_queue.task_done()

    @safe_run
    def copy_seq(self, sequence, source, dest, prefix):
        os.makedirs(dest, exist_ok=True)
        files=os.listdir(source)
        fileseq=sequence.seqname + ".seq"
        if sequence.type==0:
            pattern=re.compile(rf"^{re.escape(sequence.seqname)}\d+\.(fit|fits)$", re.IGNORECASE)
            for file in files:
                if pattern.match(file):
                    shutil.copy2(os.path.join(source, file), os.path.join(dest, prefix+file))
                elif file == fileseq:
                    with open(os.path.join(source, fileseq), "r", encoding="utf-8") as f_in:
                        cont = f_in.read()
                    cont_mod = cont.replace(sequence.seqname, prefix + sequence.seqname)
                    with open(os.path.join(dest, prefix+fileseq), "w", encoding="utf-8") as f_out:
                        f_out.write(cont_mod)
        elif sequence.type==2:
            for ext in [".fit",".fits"]:
                filefits=sequence.seqname+ext
                if filefits in files:
                    shutil.copy2(os.path.join(source, filefits), os.path.join(dest, prefix+filefits))
                elif file == fileseq:
                    with open(os.path.join(source, fileseq), "r", encoding="utf-8") as f_in:
                        cont = f_in.read()
                    cont_mod = cont.replace(sequence.seqname, prefix + sequence.seqname)
                    with open(os.path.join(dest, prefix+fileseq), "w", encoding="utf-8") as f_out:
                        f_out.write(cont_mod)
        else:
            fileser=sequence.seqname+".ser"
            if fileser in files:
                shutil.copy2(os.path.join(source, fileser), os.path.join(dest, prefix+fileser))
            elif file == fileseq:
                with open(os.path.join(source, fileseq), "r", encoding="utf-8") as f_in:
                    cont = f_in.read()
                cont_mod = cont.replace(sequence.seqname, prefix + sequence.seqname)
                with open(os.path.join(dest, prefix+fileseq), "w", encoding="utf-8") as f_out:
                    f_out.write(cont_mod)

    @safe_run
    def wait_until_folder_exist(self, folder, timeout=300):
        start = time.time()
        while os.path.exists(folder):
            if time.time() - start > timeout:
                self.siril.log("Error: The temporary folder was not deleted within the expected time.", color=LogColor.RED)
                return False
            time.sleep(0.1)
        return True

    @safe_run
    def run_close(self):        
        self.stop_event.set()
        path=self.work_path_var.get()
        if os.path.exists(path+"/ER_TEMP"):
            self.siril.log("Cleaning temporary files", color=LogColor.GREEN)
            self.siril.cmd("cd",path)
            shutil.rmtree(path+"/ER_TEMP")
        self.wait_until_folder_exist(path+"/ER_TEMP")
        self.root.destroy()


def main():
    root = ThemedTk()
    app = CometStarTrailsInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()