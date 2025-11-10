"""
Workflow Summarizer Script
(c) Adrian Knagg-Baugh 2025
based on an idea by Adrian Nowik
SPDX-License-Identifier: GPL-3.0-or-later
Version: 1.2.0

This script uses Google Gemini to process the Siril log file and generate a
summary of the workflow. The use case is to provide a complete and easily-
readable documentation of a workflow you have just completed so that it can 
be replicated on other images. Although individual operations are saved in 
HISTORY FITS header cards, this script can provide a higher-level overview of
workflows involving combination of multiple images, e.g. star separation or 
separate processing and combination of multiple narrowband filters. Note that 
you will need a Google Gemini API key and usage will be subject to your free or 
paid tier token limits. For very long logs, the script will upload the log as 
a file attachment for more efficient processing, however extremely long logs
may still exceed Google Gemini's input token limit.
If you have set Siril's language preference, the output will be requested in 
that language; if the preference is not set, output will default to English.
"""

import sys
import os
import tempfile
import time
import sirilpy as s
s.ensure_installed("PyQt6", "google-generativeai", "configparser")
version_ok = s.check_module_version(">=0.8.7")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QFileDialog, QMessageBox, QProgressBar,
                             QComboBox, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
import configparser
import google.generativeai as genai

# Token estimation: roughly 4 characters per token
CHARS_PER_TOKEN = 4
# Use file upload for logs longer than 50k tokens (~200k characters)
FILE_UPLOAD_THRESHOLD = 200000

def build_prompt(level, language=None, text=None, use_file_upload=False, for_preview=False, custom_prompt=None):
    """
    Build the prompt for Gemini API.
    
    Args:
        level: Either "highlevel", "detailed", or "custom"
        language: Target language for the summary (e.g., "English")
        text: The actual log text to include (ignored if use_file_upload=True)
        use_file_upload: Whether the log will be uploaded as a file
        for_preview: If True, show placeholder variables; if False, populate with actual values
        custom_prompt: Custom prompt text (only used when level="custom")
    
    Returns:
        The complete prompt string
    """
    # Handle custom prompt
    if level == "custom":
        if custom_prompt:
            return custom_prompt
        else:
            # Return a default custom prompt template for preview
            return "Enter your custom prompt here. The log will be automatically appended or uploaded as a file."
    
    # Build components with conditional variable substitution
    if for_preview:
        lang_placeholder = "{language}"
        text_placeholder = "{text}"
    else:
        lang_placeholder = language if language else "English"
        text_placeholder = text if text else ""
    
    upload_text = "I have uploaded a Siril log from my astrophotography processing workflow.\n\n" if use_file_upload else ""
    
    undo_instructions = (
        f"IMPORTANT: Before summarizing, you must first identify and remove any operations that were undone. "
        f"When you see an 'Undo' command, it cancels the operation that came immediately before it. "
        f"Both the undone operation AND the Undo command itself should be excluded from your summary.\n\n"
        f"After filtering out undone operations, please provide in {lang_placeholder} language:\n"
    )
    
    if "highlevel" in level:
        summary_type = (
            f"- A chronological high-level summary of each remaining processing step\n"
            f"- Do not include the parameters used or their values\n"
        )
    else:
        summary_type = (
            f"- A chronological detailed summary of each remaining processing step\n"
            f"- Include the parameters used and their values where applicable\n"
        )
    
    format_instructions = (
        f"- Format as a structured markdown document\n"
        f"- Do not include citation references\n\n"
    )
    
    # Build final prompt
    prompt = upload_text + undo_instructions + summary_type + format_instructions
    
    # Add text content if not using file upload and not for preview
    if not use_file_upload and not for_preview:
        prompt += text_placeholder
    elif for_preview and not use_file_upload:
        prompt += text_placeholder
    
    return prompt

class WorkflowWorker(QThread):
    """Worker thread to process the Siril workflow without freezing the GUI"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, api_key, siril, level, custom_prompt=None):
        super().__init__()
        self.api_key = api_key
        self.siril = siril
        self.level = level
        self.custom_prompt = custom_prompt
        self.uploaded_file = None
    
    def run(self):
        try:
            # Configure API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Get log and language from the passed Siril interface
            self.progress.emit("Retrieving Siril log...")
            text = self.siril.get_siril_log()
            language = self.siril.get_siril_config("core", "lang")
            if "not set" in language:
                language = "English"

            # Check log length
            log_length = len(text)
            estimated_tokens = log_length // CHARS_PER_TOKEN
            self.progress.emit(f"Log size: {log_length:,} characters (~{estimated_tokens:,} tokens)")
            
            # Determine whether to use file upload
            use_file_upload = log_length > FILE_UPLOAD_THRESHOLD

            # Build prompt using the unified function
            prompt = build_prompt(
                level=self.level,
                language=language,
                text=text,
                use_file_upload=use_file_upload,
                for_preview=False,
                custom_prompt=self.custom_prompt
            )
            
            # For custom prompts, we need to append the text if not using file upload
            if self.level == "custom" and not use_file_upload:
                prompt = prompt + "\n\n" + text
            
            # Submit the prompt
            if use_file_upload:
                self.progress.emit("Log is large, uploading as file...")
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                                 delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(text)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Upload the file
                    self.progress.emit("Uploading to Gemini...")
                    self.uploaded_file = genai.upload_file(tmp_file_path, 
                                                           display_name="siril_log.txt")
                    
                    # Wait for file to be processed
                    self.progress.emit("Waiting for file processing...")
                    while self.uploaded_file.state.name == "PROCESSING":
                        time.sleep(1)
                        self.uploaded_file = genai.get_file(self.uploaded_file.name)
                    
                    if self.uploaded_file.state.name == "FAILED":
                        raise Exception("File processing failed")

                    self.progress.emit("Generating summary...")
                    # Generate response with file
                    response = model.generate_content([self.uploaded_file, prompt])
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
            else:
                # Use direct text in prompt for shorter logs
                self.progress.emit("Generating summary...")
                
                response = model.generate_content(prompt)
            
            result = response.text
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Clean up uploaded file if it exists
            if self.uploaded_file:
                try:
                    genai.delete_file(self.uploaded_file.name)
                except:
                    pass

class SirilSummaryGUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.markdown_content = ""
        self.worker = None
        self.config_file = None
        self.init_config()
        self.initUI()
    
    def init_config(self):
        """Initialize configuration file path"""
        try:
            config_dir = self.siril.get_siril_configdir()
            self.config_file = os.path.join(config_dir, "ai-log.conf")
        except Exception as e:
            print(f"Warning: Could not get Siril config directory: {e}")
            self.config_file = None
    
    def load_api_key(self):
        """Load API key from configuration file"""
        if not self.config_file or not os.path.exists(self.config_file):
            return ""
        
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            return config.get('API', 'gemini_key', fallback='')
        except Exception as e:
            print(f"Warning: Could not load API key: {e}")
            return ""
    
    def load_custom_prompt(self):
        """Load custom prompt from configuration file"""
        if not self.config_file or not os.path.exists(self.config_file):
            return ""
        
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file)
            return config.get('PROMPT', 'custom_prompt', fallback='')
        except Exception as e:
            print(f"Warning: Could not load custom prompt: {e}")
            return ""
    
    def save_custom_prompt(self, custom_prompt):
        """Save custom prompt to configuration file"""
        if not self.config_file:
            return False
        
        try:
            config = configparser.ConfigParser()
            
            # Load existing config if it exists
            if os.path.exists(self.config_file):
                config.read(self.config_file)
            
            # Ensure [PROMPT] section exists
            if not config.has_section('PROMPT'):
                config.add_section('PROMPT')
            
            # Set the custom prompt
            config.set('PROMPT', 'custom_prompt', custom_prompt)
            
            # Write to file
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save custom prompt: {e}")
            return False
    
    def load_and_set_api_key(self):
        """Load API key from config file at startup and set it in the input field"""
        saved_key = self.load_api_key()
        if saved_key:
            self.api_key_input.setText(saved_key)
            self.statusBar().showMessage("API key loaded from config file", 3000)
        else:
            # Fall back to environment variable if no saved key
            env_key = os.environ.get("GEMINI_API_KEY", "")
            if env_key:
                self.api_key_input.setText(env_key)
                self.statusBar().showMessage("API key loaded from environment variable", 3000)
    
    def save_api_key(self, api_key):
        """Save API key to configuration file"""
        if not self.config_file:
            return False
        
        try:
            config = configparser.ConfigParser()
            
            # Load existing config if it exists
            if os.path.exists(self.config_file):
                config.read(self.config_file)
            
            # Ensure [API] section exists
            if not config.has_section('API'):
                config.add_section('API')
            
            # Set the API key
            config.set('API', 'gemini_key', api_key)
            
            # Write to file
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                config.write(f)
            
            return True
        except Exception as e:
            print(f"Warning: Could not save API key: {e}")
            return False
    
    def save_api_key_manually(self):
        """Manually save the API key from the input field"""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "No API Key", 
                              "Please enter an API key before saving.")
            return
        
        if self.save_api_key(api_key):
            self.statusBar().showMessage("API key saved successfully", 3000)
            QMessageBox.information(self, "Success", 
                                  "API key has been saved to the config file.")
        else:
            QMessageBox.warning(self, "Save Failed", 
                              "Failed to save API key to config file.")
    
    def toggle_api_key_visibility(self):
        """Toggle between showing and hiding the API key"""
        if self.api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.toggle_visibility_btn.setText("🔒")
            self.toggle_visibility_btn.setToolTip("Hide API Key")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.toggle_visibility_btn.setText("👁")
            self.toggle_visibility_btn.setToolTip("Show API Key")
    
    def update_prompt_preview(self):
        """Update the prompt preview based on current settings"""
        level = self.level_combo.currentData()
        
        # Handle custom prompt mode
        if level == "custom":
            # Make the preview editable
            self.prompt_preview.setReadOnly(False)
            self.prompt_preview.setStyleSheet("background-color: #2a2a2a; font-family: monospace; font-size: 9pt;")
            
            # Load saved custom prompt or use default detailed prompt as starting point
            if not self.prompt_preview.toPlainText() or self.prompt_preview.toPlainText().startswith("I have uploaded"):
                # Only load if the text area is empty or contains a default prompt
                saved_custom = self.load_custom_prompt()
                if saved_custom:
                    self.prompt_preview.setPlainText(saved_custom)
                else:
                    # First time using custom prompt - show detailed prompt as template
                    default_prompt = build_prompt(
                        level="detailed",
                        use_file_upload=True,
                        for_preview=True
                    )
                    self.prompt_preview.setPlainText(default_prompt)
                    
                    # Show informational message
                    QMessageBox.information(
                        self,
                        "Custom Prompt",
                        "You can now edit this prompt to customize it for your needs.\n\n"
                        "The Siril log will be automatically appended (for short logs) "
                        "or uploaded as a file (for long logs) when you generate the summary.\n\n"
                        "Your custom prompt will be saved automatically when you click 'Generate Summary'."
                    )
        else:
            # Make the preview read-only for default prompts
            self.prompt_preview.setReadOnly(True)
            self.prompt_preview.setStyleSheet("background-color: #202020; font-family: monospace; font-size: 9pt;")
            
            # Use the unified build_prompt function with for_preview=True
            preview_prompt = build_prompt(
                level=level,
                use_file_upload=True,  # Show upload variant for preview
                for_preview=True
            )
            
            self.prompt_preview.setPlainText(preview_prompt)
    
    def initUI(self):
        self.setWindowTitle("Siril Workflow Summary Generator")
        self.setMinimumSize(800, 700)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # API Key section
        api_layout = QHBoxLayout()
        api_label = QLabel("Google Gemini API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter your API key here")
        
        # Toggle visibility button
        self.toggle_visibility_btn = QPushButton("👁")
        self.toggle_visibility_btn.setMaximumWidth(40)
        self.toggle_visibility_btn.setToolTip("Show/Hide API Key")
        self.toggle_visibility_btn.clicked.connect(self.toggle_api_key_visibility)
        
        # Save API key button
        self.save_key_btn = QPushButton("Save Key")
        self.save_key_btn.setMaximumWidth(100)
        self.save_key_btn.setToolTip("Save API Key to config file")
        self.save_key_btn.clicked.connect(self.save_api_key_manually)
        
        api_layout.addWidget(api_label)
        api_layout.addWidget(self.api_key_input, stretch=1)
        api_layout.addWidget(self.toggle_visibility_btn)
        api_layout.addWidget(self.save_key_btn)
        layout.addLayout(api_layout)
        
        # Load API key from config file at startup
        self.load_and_set_api_key()
        
        # API Key link
        link_layout = QHBoxLayout()
        link_label = QLabel("Don't have an API key?")
        get_key_btn = QPushButton("Get API Key")
        get_key_btn.clicked.connect(self.open_api_key_url)
        get_key_btn.setMaximumWidth(120)
        link_layout.addWidget(link_label)
        link_layout.addWidget(get_key_btn)
        link_layout.addStretch()
        layout.addLayout(link_layout)
        
        # Summary level selection
        level_layout = QHBoxLayout()
        level_label = QLabel("Summary Level:")
        self.level_combo = QComboBox()
        self.level_combo.addItem("Detailed with Parameters", "detailed")
        self.level_combo.addItem("High-Level Overview", "highlevel")
        self.level_combo.addItem("Custom Prompt", "custom")
        self.level_combo.currentIndexChanged.connect(self.update_prompt_preview)
        self.level_combo.setToolTip("Choose between preset summaries or create a custom prompt")
        level_layout.addWidget(level_label)
        level_layout.addWidget(self.level_combo, stretch=1)
        layout.addLayout(level_layout)
        
        # Prompt preview section
        prompt_group = QGroupBox("Prompt Preview")
        prompt_layout = QVBoxLayout()
        
        self.prompt_preview = QTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setMaximumHeight(150)
        self.prompt_preview.setStyleSheet("background-color: #202020; font-family: monospace; font-size: 9pt;")
        
        prompt_layout.addWidget(self.prompt_preview)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Initialize prompt preview
        self.update_prompt_preview()
        
        # Generate button
        self.generate_btn = QPushButton("Generate Summary")
        self.generate_btn.clicked.connect(self.generate_summary)
        self.generate_btn.setMinimumHeight(40)
        layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        # Result section
        result_label = QLabel("Workflow Summary:")
        result_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(result_label)
        
        # Rich text display
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Generated summary will appear here...")
        layout.addWidget(self.result_text)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Markdown")
        self.save_btn.clicked.connect(self.save_markdown)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(35)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setMinimumHeight(35)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def open_api_key_url(self):
        """Open the Google AI Studio page to get an API key"""
        url = QUrl("https://aistudio.google.com/app/apikey")
        QDesktopServices.openUrl(url)
    
    def generate_summary(self):
        """Start the workflow summary generation"""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "API Key Required", 
                              "Please enter your Google Gemini API key.")
            return
        
        # Save the API key to config file
        if self.save_api_key(api_key):
            self.statusBar().showMessage("API key saved", 2000)
        
        # Get selected level
        level = self.level_combo.currentData()
        
        # Handle custom prompt
        custom_prompt = None
        if level == "custom":
            custom_prompt = self.prompt_preview.toPlainText().strip()
            if not custom_prompt:
                QMessageBox.warning(self, "Empty Custom Prompt", 
                                  "Please enter a custom prompt before generating.")
                return
            # Save the custom prompt
            if self.save_custom_prompt(custom_prompt):
                self.statusBar().showMessage("Custom prompt saved", 2000)
        
        # Disable controls during processing
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.level_combo.setEnabled(False)
        self.prompt_preview.setReadOnly(True)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Starting...")
        
        # Start worker thread
        self.worker = WorkflowWorker(api_key, self.siril, level, custom_prompt)
        self.worker.finished.connect(self.on_generation_complete)
        self.worker.error.connect(self.on_generation_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.start()
    
    def on_progress_update(self, message):
        """Update status bar with progress messages"""
        self.statusBar().showMessage(message)
    
    def on_generation_complete(self, result):
        """Handle successful generation"""
        self.markdown_content = result
        self.result_text.setMarkdown(result)
        
        # Re-enable controls
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.level_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update prompt preview readonly state based on current level
        level = self.level_combo.currentData()
        if level == "custom":
            self.prompt_preview.setReadOnly(False)
        
        self.statusBar().showMessage("Workflow summary generated successfully!")
    
    def on_generation_error(self, error_msg):
        """Handle generation errors"""
        self.generate_btn.setEnabled(True)
        self.level_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update prompt preview readonly state based on current level
        level = self.level_combo.currentData()
        if level == "custom":
            self.prompt_preview.setReadOnly(False)
        
        self.statusBar().showMessage("Error occurred")
        
        QMessageBox.critical(self, "Error", 
                           f"An error occurred while generating the summary:\n\n{error_msg}")
    
    def save_markdown(self):
        """Save the markdown content to a file"""
        if not self.markdown_content:
            QMessageBox.warning(self, "No Content", 
                              "There is no content to save. Please generate a summary first.")
            return
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Markdown File",
            "siril_workflow_summary.md",
            "Markdown Files (*.md);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.markdown_content)
                
                self.statusBar().showMessage(f"Saved to {file_path}")
                QMessageBox.information(self, "Success", 
                                      f"Markdown file saved successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", 
                                   f"Failed to save file:\n\n{str(e)}")
    
    def clear_results(self):
        """Clear the results display"""
        self.result_text.clear()
        self.markdown_content = ""
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("Cleared")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    if not version_ok:
        print("Error: sirilpy version requirement not met, aborting...")
        exit()
    # Initialize Siril interface once
    try:
        siril = s.SirilInterface()
        siril.connect()
    except Exception as e:
        QMessageBox.critical(None, "Siril Connection Error",
                           f"Failed to connect to Siril:\n\n{str(e)}\n\n"
                           "Please ensure Siril is running and try again.")
        sys.exit(1)
    
    window = SirilSummaryGUI(siril)
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
