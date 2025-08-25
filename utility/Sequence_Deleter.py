#
# ***********************************************
#
# Copyright (C) 2025 - Carlo Mollicone - AstroBOH
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The author of this script is Carlo Mollicone (CarCarlo147) and can be reached at:
# https://www.astroboh.it
# https://www.facebook.com/carlo.mollicone.9
#
# ***********************************************
#
# --------------------------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------------------------
#
# Description:
# This script for Siril creates a simple graphical user interface (GUI)
# to delete image sequences and their related files.
#
# Features:
# 1. Scans the current Siril working directory for .seq files.
# 2. Displays the found sequences in a dropdown menu.
# 3. Allows the user to select a sequence and permanently delete
#    all associated FITS files, the .seq file itself, and the _conversion.txt file if it exists.
#
# Version History
# 1.0.0 Initial release
# 1.0.1 Minor fix: adjusted GUI layout
# 1.0.2 Minor fix: Center window on open
# 1.0.3 Added support for variable FITS sequences (SEQ_VARIABLE = FITSEQ, AVI or SER files) and regular FITS sequences (SEQ_REGULAR)
# 1.0.4 Added contact information
# 1.0.5 Improved detection of working directory
#
#

VERSION = "1.0.5"

# Core module imports
import os
import glob
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
# This allows the script to be run externally for testing (with limited functionality).
try:
    import sirilpy as s

    # Check the module version
    if not s.check_module_version('>=0.6.37'):
        messagebox.showerror("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    SIRIL_ENV = True

    # Import Siril GUI related components
    from sirilpy import tksiril, SirilError
    
    # Ensure ttkthemes is installed for better looking GUI
    s.ensure_installed("ttkthemes")

    from ttkthemes import ThemedTk
except ImportError:
    SIRIL_ENV = False
    messagebox.showerror("Warning: sirilpy not found. The script is not running in the Siril environment.")

# Funzione ausiliaria per eliminare i file, aggiornata per gestire i nomi base
def delete_file_if_exists(path, log_func=None):
    """
    Deletes a file if it exists, handling cases where the path might not include an extension.
    It will attempt to find and delete all files matching the base name, regardless of extension.
    """
    # Check if the path likely contains an extension already
    # A simple heuristic: if there's a dot and it's not the first character
    # and the part after the dot is not empty, assume it's an extension.
    has_explicit_extension = '.' in os.path.basename(path) and os.path.basename(path).split('.')[-1] != ''

    if has_explicit_extension:
        # If the path already includes an extension, try to delete that specific file.
        files_to_check = [path]
    else:
        # If no explicit extension, use glob to find all files matching the base name.
        # This covers cases like "HA" matching "HA.fit", "HA.fts", etc.
        files_to_check = glob.glob(f"{path}.*")
        
    if not files_to_check:
        # If glob found nothing or the specific path didn't exist
        if log_func:
            log_func(f"File(s) not found for removal: {path}", s.LogColor.RED)
        return

    for file_to_delete in files_to_check:
        try:
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                if log_func:
                    log_func(f"File removed: {os.path.basename(file_to_delete)}", s.LogColor.GREEN)
            else:
                # This case might be hit if glob found a path but it disappeared between glob and os.path.exists
                if log_func:
                    log_func(f"File disappeared before removal: {os.path.basename(file_to_delete)}", s.LogColor.RED)
        except Exception as e:
            if log_func:
                log_func(f"Error removing {os.path.basename(file_to_delete)}: {e}", s.LogColor.RED)

class SequenceDeleterApp:
    """
    Main class that handles the GUI and Siril script execution.
    """
    def __init__(self, root):
        self.root = root
        self.root.title(f"Sequences Deleter Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

        #setting window size And Center the window
        width=450
        height=150
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)

        self.root.resizable(False, False)
        self.style = tksiril.standard_style()

        # Initialize Siril connection
        self.siril = None # Initialize to None
        if SIRIL_ENV:
            self.siril = s.SirilInterface()
            try:
                self.siril.connect()
            except s.SirilConnectionError:
                messagebox.showerror("Connection Error", "Connection to Siril failed. Make sure Siril is open and ready.")
                self.close_dialog()
                return
        
        tksiril.match_theme_to_siril(self.root, self.siril)

        # Main frame to hold the widgets
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill="both")

        # Interface widgets
        label = ttk.Label(main_frame, text="Select the sequence to delete:")
        label.pack(pady=5, padx=5, anchor="w")

        self.sequence_var = tk.StringVar()
        self.sequence_dropdown = ttk.Combobox(main_frame, textvariable=self.sequence_var, state="readonly")
        self.sequence_dropdown.pack(pady=5, padx=5, fill="x")

        # Frame for the buttons, to place them side by side
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10, padx=5, fill="x")

        self.delete_button = ttk.Button(button_frame, text="Delete Selected", command=self.delete_selected_sequence)
        self.delete_button.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.refresh_button = ttk.Button(button_frame, text="Refresh List", command=self.populate_sequences)
        self.refresh_button.pack(side="left", expand=True, fill="x", padx=(5, 0))
        
        # Handle window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Populate the dropdown menu on startup
        self.populate_sequences()

    def on_closing(self):
        """
        Handle dialog close - Called when the window is closed via the 'X' button.
        Close the dialog and disconnect from Siril
        """
        try:
            if SIRIL_ENV and self.siril:
                self.siril.log("Window closed. Script cancelled by user.", s.LogColor.GREEN)
            self.siril.disconnect()
            self.root.quit()
        except Exception:
            pass
        self.root.destroy()

    def get_working_directory(self):
        """Gets the current working directory."""
        # If the script is in Siril, the CWD is already set correctly.
        # Otherwise, for testing, use the script's folder.
        if SIRIL_ENV:
            # If the script is running within the Siril environment (SIRIL_ENV is true),
            # then the working directory of the Python process (os.getcwd()) is the same as that of Siril.
            # While this is true in most cases, it is not guaranteed.
            # There may be situations where the two directories are out of sync, leading to difficult-to-diagnose errors.
            # return os.getcwd()
            return self.siril.get_siril_wd()
        else:
            return os.path.dirname(os.path.abspath(__file__))

    def find_sequences(self):
        """Finds all .seq files in the working directory."""
        cwd = self.get_working_directory()
        try:
            files = os.listdir(cwd)
            # Filter to find only files ending with .seq
            sequences = [f for f in files if f.endswith('.seq')]
            return sorted(sequences)
        except FileNotFoundError:
            self.siril.log(f"The working directory '{cwd}' was not found.", s.LogColor.RED)
            return []

    def populate_sequences(self):
        """Populates the dropdown menu with the found sequences."""
        sequences = self.find_sequences()
        if sequences:
            self.sequence_dropdown['values'] = sequences
            self.sequence_var.set(sequences[0]) # Select the first item by default
        else:
            self.sequence_dropdown['values'] = []
            self.sequence_var.set("No sequence found in the current folder")

    def delete_selected_sequence(self):
        """
        Main function that runs when the 'Delete' button is clicked.
        """
        selected_seq_file = self.sequence_var.get()

        # Check if a valid sequence has been selected
        if not selected_seq_file or not selected_seq_file.endswith('.seq'):
            self.siril.log(f"Please select a valid sequence from the list.", s.LogColor.RED)
            return

        # Asks the user for confirmation before proceeding
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you absolutely sure you want to permanently delete the sequence\n\n '{selected_seq_file}'\n\nand all its files?\n\nThis action cannot be undone.",
            icon='warning'
        )

        if not confirm:
            return

        # Proceeds with the deletion
        try:
            self.siril.log(f"Attempting to delete sequence: {selected_seq_file}", s.LogColor.BLUE)
            working_dir = self.get_working_directory()
            
            # Full path to the .seq file
            seq_file_full_path = os.path.join(working_dir, selected_seq_file)
            seq_base_name = os.path.splitext(selected_seq_file)[0] # e.g., "my_sequence" from "my_sequence.seq"

            deleted_count = 0
            container_deleted = None
            conversion_deleted = False
            
            # Flags for sequence type
            is_ser_sequence = False
            is_avi_sequence = False
            is_fits_sequence_regular = False # S line with var_size=0 (e.g., light_00001.fit)
            is_fits_sequence_variable = False # S line with var_size=1, F lines used

            # Sequence details from 'S' line
            seq_base_name_from_s = seq_base_name
            start_index_s = 0
            end_index_s = 0
            fixed_len_s = 0

            # --- Read the .seq file to determine sequence type and get details ---
            if os.path.exists(seq_file_full_path):
                with open(seq_file_full_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('T'):
                            type_char = line[1]
                            if type_char == 'S': # SER file
                                is_ser_sequence = True
                                break # Found container type, no need to read further for FITS files
                            elif type_char == 'A': # AVI file
                                is_avi_sequence = True
                                break # Found container type, no need to read further for FITS files
                            # For 'TF' or other 'T' types, it implies individual FITS files, continue to check 'S' line
                        elif line.startswith('S'):
                            # Parse the 'S' line: S 'name' begin end nbselected fixed_len ref_img_index version var_size fz_flag
                            try:
                                parts = line.split("'")
                                if len(parts) >= 3:
                                    seq_base_name_from_s = parts[1] # The name is in quotes
                                    numeric_parts = parts[2].strip().split()
                                    if len(numeric_parts) >= 7:
                                        start_index_s = int(numeric_parts[0])
                                        end_index_s = int(numeric_parts[1])
                                        fixed_len_s = int(numeric_parts[3])
                                        var_size_flag = int(numeric_parts[6]) # 6th numeric part is var_size

                                        if var_size_flag == 0: # SEQ_REGULAR
                                            is_fits_sequence_regular = True
                                        else: # SEQ_VARIABLE (F lines will be used)
                                            is_fits_sequence_variable = True
                            except ValueError as e:
                                self.siril.log(f"Warning: Could not parse 'S' line in {selected_seq_file}: {e}", s.LogColor.RED)
                                pass # Continue with default assumptions or further parsing

            # --- 1. Delete the sequence container file if it's SER or AVI ---
            if is_ser_sequence:
                container_file_to_delete = f"{seq_base_name}.ser"
            elif is_avi_sequence:
                container_file_to_delete = f"{seq_base_name}.avi"
            else:
                container_file_to_delete = None # No special container for FITSEQ or individual FITS

            if container_file_to_delete:
                full_container_path = os.path.join(working_dir, container_file_to_delete)
                if os.path.exists(full_container_path):
                    delete_file_if_exists(full_container_path, self.siril.log)
                    container_deleted = container_file_to_delete
                    #self.siril.log(f"Deleted sequence container file: {full_container_path}", s.LogColor.GREEN)
                else:
                    self.siril.log(f"Warning: Container file not found for deletion: {full_container_path}", s.LogColor.RED)

            # --- 2. Delete individual FITS files ---
            # This step only applies if it's NOT a SER or AVI sequence (where frames are inside the container)
            if not is_ser_sequence and not is_avi_sequence:
                if is_fits_sequence_regular:
                    # Handle SEQ_REGULAR: generate filenames from S line parameters
                    self.siril.log(f"Deleting individual FITS files for regular sequence '{seq_base_name_from_s}'...", s.LogColor.BLUE)
                    for i in range(start_index_s, end_index_s + 1):
                        # Format filename like Siril does: "name" + padded_index
                        # Example: "TestSer_" + "00001" = "TestSer_00001"
                        padded_index = f"{i:0{fixed_len_s}d}"
                        base_fits_file_name = f"{seq_base_name_from_s}{padded_index}"
                        
                        full_file_path_base = os.path.join(working_dir, base_fits_file_name)
                        
                        # Use glob to find the actual file with its extension (e.g., .fit, .fts)
                        matching_files = glob.glob(f"{full_file_path_base}.*")
                        if matching_files:
                            for mf in matching_files:
                                delete_file_if_exists(mf, self.siril.log)
                                deleted_count += 1
                        else:
                            self.siril.log(f"Warning: Individual FITS file not found: {base_fits_file_name}.*", s.LogColor.RED)
                
                elif is_fits_sequence_variable:
                    # Handle SEQ_VARIABLE: parse 'F' lines to find specific file paths
                    self.siril.log(f"Deleting individual FITS files for variable sequence from 'F' lines...", s.LogColor.BLUE)
                    # Re-read the .seq file to find individual FITS files (F lines)
                    if os.path.exists(seq_file_full_path):
                        with open(seq_file_full_path, 'r') as f:
                            for line in f:
                                if line.startswith('F'): # Line indicating an individual FITS file within the sequence
                                    parts = line.strip().split(' ', 2)
                                    if len(parts) > 2:
                                        file_name_from_seq = parts[2]
                                        full_file_path = os.path.join(working_dir, file_name_from_seq)
                                        
                                        # Use glob to find the actual file with its extension
                                        matching_files = glob.glob(f"{full_file_path}*") # Match any extension
                                        if matching_files:
                                            for mf in matching_files:
                                                delete_file_if_exists(mf, self.siril.log)
                                                deleted_count += 1
                                        else:
                                            self.siril.log(f"Warning: Individual file not found for deletion: {file_name_from_seq}", s.LogColor.RED)
            
            # --- 3. Delete the conversion.txt file ---
            conversion_txt = os.path.join(working_dir, f"{seq_base_name}conversion.txt")
            if os.path.exists(conversion_txt):
                delete_file_if_exists(conversion_txt, self.siril.log)
                conversion_deleted = True
                #self.siril.log(f"Deleted conversion file: {conversion_txt}", s.LogColor.GREEN)

            # --- 4. Delete the .seq file itself ---
            if os.path.exists(seq_file_full_path):
                delete_file_if_exists(seq_file_full_path, self.siril.log)
                #self.siril.log(f"Deleted sequence definition file: {seq_file_full_path}", s.LogColor.GREEN)
            
            # --- Final Summary ---
            message = (
                f"Deletion complete.\n\n"
                f"- {deleted_count} individual FITS files removed (if applicable).\n"
                f"- Sequence definition file removed."
            )
            if container_deleted:
                message += f"\n- Sequence container file removed."
            if conversion_deleted:
                message += f"\n- Conversion file removed."

            # Show final message
            self.siril.log(f"\nOperation Summary\n{message}", s.LogColor.GREEN)
            
            # Refresh the list in the dropdown menu
            self.populate_sequences()

        except Exception as e:
            self.siril.log(f"An error occurred during deletion:\n{e}", s.LogColor.RED)
    
def main():
    try:
        if SIRIL_ENV:
            # Create the main GUI window
            #root = ThemedTk(theme="adapta") # Try a modern theme if ttkthemes is available
            root = ThemedTk()
        else:
            root = tk.Tk()

        # Create an instance of our application
        app = SequenceDeleterApp(root)
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