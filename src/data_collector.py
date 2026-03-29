import sys
import serial
import time
import csv
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QComboBox, QProgressBar, QMessageBox, QLineEdit)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QIntValidator
import pyqtgraph as pg
from config import Config

class CollectorWorker(QThread):
    """
    Background thread handles Serial reading and Data Capture logic.
    """
    raw_data_signal = pyqtSignal(list)       # For live plotting
    sample_captured_signal = pyqtSignal(list) # When a valid hit is recorded
    status_signal = pyqtSignal(str)          # For status bar updates

    def __init__(self):
        super().__init__()
        self.running = True
        self.recording = False
        self.ser = None
        self.last_capture_time = 0

    def run(self):
        print(f"--- Connecting to {Config.SERIAL_PORT} ---")
        try:
            self.ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=0.005)
            self.ser.reset_input_buffer()
            self.status_signal.emit(f"Connected to {Config.SERIAL_PORT}")
        except Exception as e:
            self.status_signal.emit(f"Error: {e}")
            return

        while self.running:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line: continue
                    
                    vals = list(map(int, line.split(',')))
                    if len(vals) != 4: continue

                    # 1. Emit raw data for Plotting (Always)
                    self.raw_data_signal.emit(vals)

                    # 2. Recording Logic (Only if enabled)
                    if self.recording:
                        # Trigger Threshold
                        if max(vals) > Config.TRIGGER_THRESHOLD and \
                           (time.time() - self.last_capture_time > 0.2): # 200ms cooldown for recording
                            
                            self._capture_sample(vals)

                except ValueError:
                    pass
    
    def _capture_sample(self, first_vals):
        """Capture 5ms window -> Find Peak"""
        batch = [first_vals]
        start_time = time.time()
        
        while (time.time() - start_time) < Config.CAPTURE_WINDOW:
            if self.ser.in_waiting:
                l = self.ser.readline().decode(errors='ignore').strip()
                if l:
                    try:
                        batch.append(list(map(int, l.split(','))))
                    except: pass
        
        # Find Peak
        peak_vals = np.max(np.array(batch), axis=0)
        
        # Emit the captured sample to GUI to be saved
        self.sample_captured_signal.emit(list(peak_vals))
        
        # Reset cooldown
        self.last_capture_time = time.time()

    def start_recording(self):
        self.recording = True
        self.ser.reset_input_buffer()

    def stop_recording(self):
        self.recording = False

    def stop(self):
        self.running = False
        if self.ser: self.ser.close()
        self.wait()

class DataCollectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taiko Data Collector (GUI)")
        self.resize(1000, 700)

        # Data State
        self.target_samples = 50 # Default
        self.current_count = 0
        self.current_label = Config.CLASSES[0]
        self.csv_path = '../data/taiko_data.csv'

        # Ensure Directory Exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Setup UI
        self.init_ui()

        # Start Serial Thread
        self.worker = CollectorWorker()
        self.worker.raw_data_signal.connect(self.update_plot)
        self.worker.sample_captured_signal.connect(self.save_sample)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === 1. Control Panel ===
        controls = QHBoxLayout()
        
        # Label Selector
        controls.addWidget(QLabel("Label:"))
        self.combo_label = QComboBox()
        self.combo_label.addItems(Config.CLASSES)
        self.combo_label.currentTextChanged.connect(self.change_label)
        controls.addWidget(self.combo_label)

        # Sample Count Input (NEW)
        controls.addWidget(QLabel("Count:"))
        self.input_count = QLineEdit()
        self.input_count.setValidator(QIntValidator(1, 9999)) # Only allow numbers
        self.input_count.setText(str(self.target_samples))
        self.input_count.setFixedWidth(60)
        controls.addWidget(self.input_count)

        # Start/Stop Recording Button (Appends by default)
        self.btn_record = QPushButton("Start Recording (Append)")
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        controls.addWidget(self.btn_record)

        # Clear Data Button
        self.btn_clear = QPushButton("Clear/Reset CSV")
        self.btn_clear.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px;")
        self.btn_clear.clicked.connect(self.clear_csv_data)
        controls.addWidget(self.btn_clear)

        layout.addLayout(controls)

        # === 2. Progress Bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.target_samples)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Collected: %v / {self.target_samples}")
        layout.addWidget(self.progress_bar)

        # === 3. Live Graph ===
        self.graph = pg.PlotWidget(title="Real-time Sensor Data")
        self.graph.setBackground('w')
        self.graph.setYRange(0, 1024)
        self.graph.showGrid(x=True, y=True, alpha=0.3)
        self.graph.addLegend()
        layout.addWidget(self.graph)

        # === 4. Plot Lines Setup (UPDATED COLORS & STYLES) ===
        self.buf_size = 200
        self.x_data = list(range(self.buf_size))
        self.y_data = [ [0]*self.buf_size for _ in range(4) ]
        
        # Color Definition: Red, Red, Blue, Blue
        colors = [(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
        
        # Style Definition: Solid, Dash, Solid, Dash
        styles = [Qt.SolidLine, Qt.DashLine, Qt.SolidLine, Qt.DashLine]
        
        names = Config.SENSOR_LABELS

        self.lines = []
        for i in range(4):
            pen = pg.mkPen(color=colors[i], width=2, style=styles[i])
            self.lines.append(self.graph.plot(self.x_data, self.y_data[i], name=names[i], pen=pen))

        # === 5. Last Captured Info ===
        self.lbl_info = QLabel("Ready to record...")
        self.lbl_info.setStyleSheet("font-size: 14px; color: blue;")
        layout.addWidget(self.lbl_info)

    def change_label(self, label):
        self.current_label = label
        self.reset_recording_state()

    def clear_csv_data(self):
        """Clears the existing CSV file to start fresh"""
        reply = QMessageBox.question(self, 'Clear Data', 
                                     "Are you sure you want to DELETE all existing training data?\nThis cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(self.csv_path):
                    os.remove(self.csv_path)
                
                # Re-create empty file with header
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(Config.SENSOR_LABELS + ['Label'])
                
                self.lbl_info.setText("CSV File Cleared! Starting fresh.")
                QMessageBox.information(self, "Success", "Data cleared successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def toggle_recording(self):
        if not self.worker.recording:
            # === START RECORDING ===
            # Read sample count from text input
            try:
                self.target_samples = int(self.input_count.text())
            except ValueError:
                self.target_samples = 50 # Fallback
                self.input_count.setText("50")

            # Reset Progress Bar Range
            self.progress_bar.setRange(0, self.target_samples)
            self.progress_bar.setFormat(f"Collected: %v / {self.target_samples}")

            self.reset_recording_state()
            self.worker.start_recording()
            
            # Update UI
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 6px;")
            self.combo_label.setEnabled(False) # Lock selection
            self.input_count.setEnabled(False) # Lock count
            self.btn_clear.setEnabled(False)   # Lock clear button
        else:
            # === STOP RECORDING ===
            self.worker.stop_recording()
            self.btn_record.setText("Start Recording (Append)")
            self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
            self.combo_label.setEnabled(True)
            self.input_count.setEnabled(True)
            self.btn_clear.setEnabled(True)

    def reset_recording_state(self):
        self.current_count = 0
        self.progress_bar.setValue(0)

    def update_plot(self, vals):
        for i in range(4):
            self.y_data[i] = self.y_data[i][1:] + [vals[i]]
            self.lines[i].setData(self.x_data, self.y_data[i])

    def update_status(self, msg):
        self.lbl_info.setText(msg)

    def save_sample(self, peak_vals):
        """Called when thread captures a valid sample"""
        if self.current_count >= self.target_samples:
            self.toggle_recording() # Auto stop
            QMessageBox.information(self, "Done", f"Finished collecting {self.target_samples} samples for {self.current_label}!")
            return

        # Write to CSV (Append Mode 'a')
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists or os.stat(self.csv_path).st_size == 0:
                writer.writerow(Config.SENSOR_LABELS + ['Label'])
            
            writer.writerow(peak_vals + [self.current_label])

        self.current_count += 1
        self.progress_bar.setValue(self.current_count)
        self.lbl_info.setText(f"Last Captured: {peak_vals} -> {self.current_label}")
        print(f"Recorded: {peak_vals} for {self.current_label}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataCollectorGUI()
    window.show()
    sys.exit(app.exec_())