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
# 1.0.0 - Initial release
# 1.0.1 - Minor fix: adjusted GUI layout
# 1.0.2 - Minor fix: Center window on open
# 1.0.3 - Added support for variable FITS sequences (SEQ_VARIABLE = FITSEQ, AVI or SER files) and regular FITS sequences (SEQ_REGULAR)
# 1.0.4 - Added contact information
# 1.0.5 - Improved detection of working directory
# 2.0.0 - Complete porting to PyQt6
# 2.0.1 - CMD requires
# 2.0.2 - Added Icon App
# 2.0.3 - Added check for currently loaded sequence. If it matches the selection, it's closed before deletion.
# 2.0.4 - Changed deletion to use send2trash for safer file removal
# 2.0.5 - CLI mode added
#

VERSION = "2.0.5"

# Core module imports
import os
import glob
import sys
import base64
import traceback
import argparse 

# Attempt to import sirilpy. If not running inside Siril, the import will fail.
try:
    # --- Imports for Siril and GUI ---
    import sirilpy as s
    
    # Check the module version
    if not s.check_module_version('>=0.6.37'):
        print("Error: requires sirilpy module >= 0.6.37 (Siril 1.4.0 Beta 2)")
        sys.exit(1)

    # Import Siril GUI related components
    from sirilpy import SirilError

    s.ensure_installed("PyQt6", "send2trash")

    # --- PyQt6 Imports ---
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QPushButton, QMessageBox, QStyle
    )
    from PyQt6.QtGui import QIcon, QPixmap
    from PyQt6.QtCore import Qt, QTimer

    # --- send2trash for safer deletions ---
    import send2trash

except ImportError:
    print("Warning: sirilpy not found. The script is not running in the Siril environment.")

# Helper function for deleting files, updated to handle base names
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
                # os.remove(file_to_delete)
                send2trash.send2trash(file_to_delete)
                if log_func:
                    log_func(f"File moved to trash: {os.path.basename(file_to_delete)}", s.LogColor.GREEN)
            else:
                # This case might be hit if glob found a path but it disappeared between glob and os.path.exists
                if log_func:
                    log_func(f"File disappeared before removal: {os.path.basename(file_to_delete)}", s.LogColor.RED)
        except Exception as e:
            if log_func:
                log_func(f"Error moving {os.path.basename(file_to_delete)} to trash: {e}", s.LogColor.RED)

class SequenceDeleterApp(QWidget):
    """
    Main class that handles the GUI and Siril script execution.
    """
    def __init__(self, cli_mode=False, siril_instance=None):
        super().__init__()
        
        # Assign the Siril instance to the object.
        self.siril = siril_instance

        # If we are not in CLI mode, we build the interface
        if not cli_mode:
            self.setWindowTitle(f"Sequences Deleter Tool v{VERSION}")
            
            try:
                self.siril.cmd("requires", "1.4.0-beta2")
            except s.CommandError:
                QTimer.singleShot(0, self.close)
                return

            # --- Main Layout --- 
            main_layout = QVBoxLayout(self) 
            # External margins (padding) and spacing between widgets 
            main_layout.setContentsMargins(10, 10, 10, 10) 
            main_layout.setSpacing(8) # Vertical space between widgets 

            header_layout = QVBoxLayout() 
            header_layout.setContentsMargins(0, 0, 0, 15) # Left, Top, Right, Bottom 

            copyright_label = QLabel("(c) 2025 - Carlo Mollicone - AstroBOH") 
            copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 

            header_layout.addWidget(copyright_label) 
            main_layout.addLayout(header_layout) 

            label = QLabel("Select the sequence to delete :")
            main_layout.addWidget(label)

            self.sequence_dropdown = QComboBox()
            main_layout.addWidget(self.sequence_dropdown)

            # Button frame
            button_layout = QHBoxLayout()
            button_layout.setSpacing(10)
            # Add a TOP margin to this group to separate it from the combobox
            button_layout.setContentsMargins(0, 10, 0, 0)

            self.delete_button = QPushButton("    Delete Selected    ")
            self.delete_button.setProperty("class", "DeleteBtn")
            self.delete_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogDiscardButton))
            self.delete_button.clicked.connect(self.delete_selected_sequence)

            self.refresh_button = QPushButton("    Refresh List    ")
            self.refresh_button.setProperty("class", "RefreshBtn")
            self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
            self.refresh_button.clicked.connect(self.populate_sequences)

            button_layout.addWidget(self.delete_button)
            button_layout.addWidget(self.refresh_button)
            main_layout.addLayout(button_layout)

            # Pushes contents to the top,
            # preventing them from expanding disproportionately vertically.
            main_layout.addStretch(1)

            # Populate initial list
            self.populate_sequences()

            # Window size and centering
            self.setFixedSize(450, 170)
            self.center_window()

    def center_window(self):
        """ Window centering. """
        screen_geometry = self.screen().availableGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def closeEvent(self, event):
        """
        Handle dialog close - Called when the window is closed via the 'X' button.
        Close the dialog and disconnect from Siril
        """
        try:
            self.siril.log("Window closed. Script cancelled by user.", s.LogColor.GREEN)
            self.siril.disconnect()
        except Exception:
            pass
        event.accept()

    def get_working_directory(self):
        # os.path.dirname(os.path.abspath(__file__))
        return self.siril.get_siril_wd()

    def find_sequences(self):
        """Finds all .seq files in the working directory."""
        cwd = self.get_working_directory()
        try:
            files = os.listdir(cwd)
            # Filter to find only files ending with .seq
            sequences = [f for f in files if f.endswith('.seq')]
            return sorted(sequences, key=str.lower)
        except FileNotFoundError:
            self.siril.log(f"The working directory '{cwd}' was not found.", s.LogColor.RED)
            return []

    def populate_sequences(self):
        """Populates the dropdown menu with the found sequences."""
        sequences = self.find_sequences()
        self.sequence_dropdown.clear()
        if sequences:
            self.sequence_dropdown.addItems(sequences)
        else:
            self.sequence_dropdown.addItem("No sequence found in the current folder")

    def delete_selected_sequence(self):
        """
        Main function that runs when the 'Delete' button is clicked.
        Handles user confirmation and calls the deletion logic.
        """
        selected_seq_file = self.sequence_dropdown.currentText()

        # Check if a valid sequence has been selected
        if not selected_seq_file or not selected_seq_file.endswith('.seq'):
            self.siril.log(f"Please select a valid sequence from the list.", s.LogColor.RED)
            return

        # Asks the user for confirmation before proceeding
        confirm = QMessageBox.question(
            self, "Confirm Move to Trash",
            f"Are you sure you want to move the sequence\n\n '{selected_seq_file}'\n\n"
            "and all its files to the trash?\n\nYou can recover the files from your system's trash.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return
        
        # Call the new method to perform the deletion
        self._perform_deletion(selected_seq_file)

    def _perform_deletion(self, selected_seq_file):
        """ Helper method to perform the deletion of the selected sequence. """
        # Proceeds with the deletion
        try:
            current_loaded_seq = None
            
            # Check if a sequence is loaded
            seq_loaded = self.siril.is_sequence_loaded()
            if seq_loaded:
                self.current_sequence = self.siril.get_seq()
                current_loaded_seq = self.current_sequence.seqname
                selected_seq_base_name = os.path.splitext(selected_seq_file)[0]

            # If the selected sequence is currently loaded, close it first
            if current_loaded_seq and current_loaded_seq == selected_seq_base_name:
                self.siril.log(f"Sequence '{current_loaded_seq}' is currently loaded. Closing it before deletion.", s.LogColor.GREEN)
                try:
                    self.siril.cmd("close")
                except s.CommandError as e:
                    self.siril.log(f"Failed to close the current sequence: {e}. Aborting deletion.", s.LogColor.RED)
                    QMessageBox.critical(self, "Error", f"Could not close the active sequence '{current_loaded_seq}'. Deletion aborted.")
                    return

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
                        if line.startswith('T') and len(line) > 1:
                            # S → SER
                            # A → AVI
                            # F → FITSEQ
                            # So the line will always be 2 characters long: "TS", "TA", "TF". We check the second character.
                            
                            type_char = line[1]
                            if type_char == 'S':    # SER file
                                is_ser_sequence = True
                                break # Found container type, no need to read further for FITS files
                            elif type_char == 'A':  # AVI file
                                is_avi_sequence = True
                                break # Found container type, no need to read further for FITS files
                            elif type_char == 'F':  # FITSEQ
                                # For 'TF' or other 'T' types, it implies individual FITS files, continue to check 'S' line
                                pass
                            else:
                                self.siril.log(f"Unknown sequence container type: {line}", s.LogColor.RED)

                        elif line.startswith('S'):
                            # Parse the 'S' line: S 'name' begin end nbselected fixed_len ref_img_index version var_size fz_flag
                            try:
                                parts = line.split("'")
                                if len(parts) >= 3:
                                    # #Siril sequence file. Contains list of images, selection, registration data and statistics
                                    # #S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag
                                    # S 'Dark-Sime-600_' 1 40 40 5 0 5 0 0
                                    # L 1
                                    # I 1 1
                                    # I 2 1
                                    # ... etc.

                                    # Explanation of S line fields:
                                    # start_index = 1
                                    # nb_images = 40
                                    # nb_selected = 40
                                    # fixed_len = 5 → File numbers have 5 digits (00001, 00002 … 00040)
                                    # reference_image = 0 (no reference image)
                                    # version = 5
                                    # variable_size = 0 (fixed-size sequence, not variable)
                                    # fz_flag = 0

                                    seq_base_name_from_s = parts[1] # The name is in quotes
                                    numeric_parts = parts[2].strip().split()
                                    if len(numeric_parts) >= 7:
                                        start_index_s = int(numeric_parts[0])

                                        nb_images = int(numeric_parts[1])
                                        # Safe if start_index_s is not 1 ex: start=5, nb=10 → files 5 to 14
                                        end_index_s = start_index_s + nb_images - 1

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

            # Refresh the list in the dropdown menu only if it exists (GUI mode)
            if hasattr(self, 'sequence_dropdown'):
                self.populate_sequences()
            
            # If we got here without errors, we return True
            return True

        except Exception as e:
            self.siril.log(f"An error occurred during deletion:\n{e}", s.LogColor.RED)
            # If an error occurs, we return False
            return False

# --- Main Execution Block ---
# Entry point for the Python script in Siril
# This code is executed when the script is run.
def main():
    # First we initialize Siril to be able to control the execution mode
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        if not siril.is_cli():
            QMessageBox.critical(None, "Error", "Failed to connect to Siril")
        else:
            print("Error: Connection to Siril failed. Make sure Siril is open and ready.")
        return
    
    # Let's create a QApplication instance for CLI and GUI.
    # Required to initialize QWidget.
    qapp = QApplication(sys.argv)

    # Let's check if the script was launched in CLI mode
    if siril.is_cli():
        # CLI mode
        siril.log(f"Sequence Deleter Tool v{VERSION} - CLI Mode - (c) Carlo Mollicone AstroBOH", s.LogColor.BLUE)
        
        parser = argparse.ArgumentParser(description=f"Sequence Deleter Tool v{VERSION} - CLI mode - (c) Carlo Mollicone AstroBOH")
        parser.add_argument("sequence_file", help="The name of the .seq file to delete (e.g. 'sequence_name.seq').")
        # We could add a flag to skip confirmation, but we'll omit it for now for simplicity
        
        args = parser.parse_args()
        
        # We create an instance of the class but without starting the GUI
        # We'll pass siril to avoid reconnecting
        app_logic = SequenceDeleterApp(siril_instance=siril, cli_mode=True)
        
        # Let's perform the deletion
        siril.log(f"CLI request to delete sequence: {args.sequence_file}", s.LogColor.BLUE)
        
        # Perform the deletion and save the result (True or False)
        success = app_logic._perform_deletion(args.sequence_file)
        
        siril.disconnect()

        # Use sys.exit() to terminate the script with the correct exit code
        if success:
            print("Operation completed successfully.")
            sys.exit(0) # Success
        else:
            print("Operation failed. Check the Siril log for details.")
            sys.exit(1) # Failure
        
    else:
        # GUI mode
        try:
            qapp.setApplicationName(f"Sequences Deleter Tool v{VERSION} - (c) Carlo Mollicone AstroBOH")

            icon_data = base64.b64decode("""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""")
            pixmap = QPixmap()
            pixmap.loadFromData(icon_data)
            app_icon = QIcon(pixmap)
            qapp.setWindowIcon(app_icon)

            qapp.setStyle("Fusion")

            # Define a Qt Style Sheet (QSS)
            stylesheet = """
                QPushButton[class="DeleteBtn"] {
                    background-color: #3574F0;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton[class="DeleteBtn"]:hover {
                    background-color: #4E8AFC;
                }

                QPushButton[class="RefreshBtn"] {
                    background-color: #f0f0f0;
                    color: #005A9C;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton[class="RefreshBtn"]:hover {
                    background-color: #e0e0e0;
                }
            """
            # Apply the stylesheet to the entire application
            qapp.setStyleSheet(stylesheet)

            # Now that the application context exists, create the main widget.
            # Pass the already connected siril instance
            app = SequenceDeleterApp(siril_instance=siril)
            app.show()

            sys.exit(qapp.exec())
        except Exception as e:
            print(f"Error initializing application: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()