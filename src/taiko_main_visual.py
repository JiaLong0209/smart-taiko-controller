import sys
import serial
import time
import numpy as np
import pyautogui
import joblib
import os
import torch
import torch.nn as nn

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QComboBox, QMessageBox, QFrame)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont

import pyqtgraph as pg
from config import Config

# === Global Settings ===
SERIAL_PORT = '/dev/ttyACM0'  # CHECK YOUR PORT!
BAUD_RATE = 115200

# === Logic Parameters ===
CROSSTALK_RATIO = Config.CROSSTALK_RATIO
THRESHOLD_MAP = [70] # [DL, DR, KL, KR]
TIME_THRESHOLD = Config.TIME_THRESHOLD
CAPTURE_WINDOW = Config.CAPTURE_WINDOW

# Optimize PyAutoGUI
pyautogui.PAUSE = 0

# === PyTorch Model Definition (Must match training) ===
class TaikoNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TaikoNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.output(out)
        return out

# === Custom Drum Visualization Widget ===
class DrumWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.active_sectors = {
            'Don_Left': 0, 'Don_Right': 0,
            'Ka_Left': 0,  'Ka_Right': 0
        } # 0 = Normal, 20 = Brightest Flash
        
        # Timer to fade out lights
        self.timer = QTimer()
        self.timer.timeout.connect(self.fade_animation)
        self.timer.start(30) # 30ms update rate (Slower fade for visibility)

    def flash(self, label):
        """Trigger a flash for a specific part"""
        if label in self.active_sectors:
            self.active_sectors[label] = 20 # Reset to max brightness (20)
            self.update() # Trigger repaint

    def fade_animation(self):
        """Decrease brightness step by step"""
        changed = False
        for k in self.active_sectors:
            if self.active_sectors[k] > 0:
                self.active_sectors[k] -= 1
                changed = True
        if changed:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Geometry
        w = self.width()
        h = self.height()
        center_x, center_y = w / 2, h / 2
        radius = min(w, h) / 2 - 10
        inner_radius = radius * 0.65 # Don size

        rect_outer = QRectF(center_x - radius, center_y - radius, radius*2, radius*2)
        rect_inner = QRectF(center_x - inner_radius, center_y - inner_radius, inner_radius*2, inner_radius*2)

        # Helper to get color based on flash intensity
        def get_color(base_hex, intensity):
            # Base color
            c = QColor(base_hex)
            
            # If flashing, mix with white
            if intensity > 0:
                # Intensity 0-20. 
                # At 20, we want almost pure white. At 0, base color.
                factor = (intensity / 20.0) * 0.8 # Max 80% whiteness
                
                r = c.red() + (255 - c.red()) * factor
                g = c.green() + (255 - c.green()) * factor
                b = c.blue() + (255 - c.blue()) * factor
                c = QColor(int(r), int(g), int(b))
                
            return c

        # === Draw Ka (Outer Ring) ===
        # Ka Left (Blue)
        painter.setBrush(QBrush(get_color("#0099FF", self.active_sectors['Ka_Left'])))
        painter.setPen(QPen(Qt.black, 2))
        painter.drawPie(rect_outer, 90 * 16, 180 * 16)

        # Ka Right (Blue)
        painter.setBrush(QBrush(get_color("#0099FF", self.active_sectors['Ka_Right'])))
        painter.drawPie(rect_outer, -90 * 16, 180 * 16)

        # === Draw Don (Inner Circle) ===
        # Don Left (Red)
        painter.setBrush(QBrush(get_color("#FF3333", self.active_sectors['Don_Left'])))
        painter.drawPie(rect_inner, 90 * 16, 180 * 16)
        
        # Don Right (Red)
        painter.setBrush(QBrush(get_color("#FF3333", self.active_sectors['Don_Right'])))
        painter.drawPie(rect_inner, -90 * 16, 180 * 16)

        # === Draw Labels (FORCE WHITE) ===
        painter.setPen(QPen(Qt.white, 2)) # White text
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Position labels roughly in center of sectors
        painter.drawText(int(center_x - radius*0.8), int(center_y), "Ka L")
        painter.drawText(int(center_x + radius*0.55), int(center_y), "Ka R")
        painter.drawText(int(center_x - inner_radius*0.7), int(center_y), "Don L")
        painter.drawText(int(center_x + inner_radius*0.2), int(center_y), "Don R")

# === Serial Worker Thread ===
class SerialWorker(QThread):
    data_received = pyqtSignal(list)
    hit_detected = pyqtSignal(str) # Signal for Drum Widget
    status_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.ser = None
        self.last_hit_time = 0
        self.mode = 'threshold'
        self.model = None
        self.encoder = None
        self.scaler = None

    def set_mode(self, new_mode):
        self.mode = new_mode
        if new_mode == 'threshold':
            self.model = None
            self.status_message.emit("✅ Mode: Rule-Based")
        else:
            self.load_ai_model(new_mode)

    def load_ai_model(self, model_type):
        try:
            enc_path = '../models/label_encoder.pkl'
            scaler_path = '../models/scaler.pkl'
            
            if not os.path.exists(enc_path):
                raise FileNotFoundError("Label Encoder not found")
            
            self.encoder = joblib.load(enc_path)
            
            # Load Model
            if model_type == 'taikonet':
                model_path = f'../models/taiko_{model_type}_model.pth'
                if not os.path.exists(model_path): raise FileNotFoundError(model_path)
                
                self.scaler = joblib.load(scaler_path)
                
                # Rebuild Structure
                self.model = TaikoNet(4, len(Config.CLASSES))
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                
            elif model_type == 'dnn': 
                pass 
            
            else: # Sklearn (RF, XGB, SVM)
                model_path = f'../models/taiko_{model_type}_model.pkl'
                if not os.path.exists(model_path): raise FileNotFoundError(model_path)
                
                if model_type == 'svm': # SVM might use scaler too
                    if os.path.exists(scaler_path): self.scaler = joblib.load(scaler_path)
                
                self.model = joblib.load(model_path)

            self.status_message.emit(f"✅ AI Loaded: {model_type.upper()}")

        except Exception as e:
            self.status_message.emit(f"❌ Load Error: {e}")
            self.mode = 'threshold'

    def run(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.005)
            self.ser.reset_input_buffer()
        except Exception as e:
            self.status_message.emit(f"Serial Error: {e}")
            return

        while self.running:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line: continue
                    vals = list(map(int, line.split(',')))
                    if len(vals) != 4: continue

                    self.data_received.emit(vals)

                    # Trigger Logic
                    is_triggered = False
                    if self.mode == 'threshold':
                        if max(vals) > Config.TRIGGER_THRESHOLD: is_triggered = True
                    else:
                        if max(vals) > Config.TRIGGER_THRESHOLD: is_triggered = True

                    if is_triggered and (time.time() - self.last_hit_time > TIME_THRESHOLD):
                        self._process_hit(vals)
                except ValueError: pass

    def _process_hit(self, first_vals):
        pred_label = None
        values_to_print = first_vals # Default to raw
        
        # === AI Mode ===
        if self.mode != 'threshold' and self.model:
            # Capture 5ms
            batch = [first_vals]
            start = time.time()
            while (time.time() - start) < CAPTURE_WINDOW:
                if self.ser.in_waiting:
                    l = self.ser.readline().decode(errors='ignore').strip()
                    if l: 
                        try: batch.append(list(map(int, l.split(','))))
                        except: pass
            
            peak = np.max(np.array(batch), axis=0)
            values_to_print = peak # Update to Peak values
            
            try:
                if self.mode == 'taikonet':
                    # Scale & Tensor
                    x_scaled = self.scaler.transform([peak])
                    x_tensor = torch.FloatTensor(x_scaled)
                    with torch.no_grad():
                        out = self.model(x_tensor)
                        _, idx = torch.max(out, 1)
                        pred_label = self.encoder.inverse_transform([idx.item()])[0]
                else:
                    # Sklearn
                    idx = self.model.predict([peak])[0]
                    pred_label = self.encoder.inverse_transform([idx])[0]

            except Exception as e:
                print(e)
                return

        # === Threshold Mode ===
        else:
            dL, dR, kL, kR = first_vals
            max_don = max(dL, dR)
            max_ka = max(kL, kR)
            
            if max_don > max_ka * CROSSTALK_RATIO:
                pred_label = 'Don_Left' if dL > dR else 'Don_Right'
            elif max_ka > max_don * CROSSTALK_RATIO:
                pred_label = 'Ka_Left' if kL > kR else 'Ka_Right'
        
        # Execute
        if pred_label and pred_label != 'Noise':
            key = Config.KEY_MAP.get(pred_label)
            if key:
                pyautogui.press(key)
                self.last_hit_time = time.time()
                
                # Update Visuals
                self.hit_detected.emit(pred_label) 
                
                # Print Values instead of just Label
                print(f"Hit: {pred_label:<10} | Values: {values_to_print}")

    def stop(self):
        self.running = False
        if self.ser: self.ser.close()
        self.wait()

# === Main Window ===
class TaikoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taiko AI Controller & Visualizer")
        self.resize(1100, 600)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) # Horizontal: Left(Graph) | Right(Drum)

        # === LEFT PANEL (Controls + Graph) ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 1. Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(Config.SUPPORTED_MODELS)
        
        # Set initial from config
        idx = self.mode_combo.findText(Config.MODEL_NAME)
        if idx >= 0: self.mode_combo.setCurrentIndex(idx)
        
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        controls.addWidget(self.mode_combo)
        controls.addStretch()
        left_layout.addLayout(controls)

        # 2. Status (Changed color from blue to dark gray)
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #333; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(self.status_label)
        
        self.val_label = QLabel("Values: 0, 0, 0, 0")
        left_layout.addWidget(self.val_label)

        # 3. Graph
        self.graph = pg.PlotWidget()
        self.graph.setBackground('w')
        self.graph.setYRange(0, 1024)
        self.graph.addLegend()
        left_layout.addWidget(self.graph)
        
        # Graph Lines
        self.buf = 200
        self.x = list(range(self.buf))
        self.ys = [[0]*self.buf for _ in range(4)]
        colors = ['r', 'r', 'b', 'b']
        styles = [Qt.SolidLine, Qt.DashLine, Qt.SolidLine, Qt.DashLine]
        names = ['Don L', 'Don R', 'Ka L', 'Ka R']
        self.lines = []
        for i in range(4):
            pen = pg.mkPen(color=colors[i], width=2, style=styles[i])
            self.lines.append(self.graph.plot(self.x, self.ys[i], name=names[i], pen=pen))

        main_layout.addWidget(left_panel, stretch=2) # Left side takes 2/3 space

        # === RIGHT PANEL (Drum Visual) ===
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addStretch()
        self.drum_widget = DrumWidget()
        right_layout.addWidget(self.drum_widget, alignment=Qt.AlignCenter)
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel, stretch=1) # Right side takes 1/3 space

        # === Worker ===
        self.worker = SerialWorker()
        self.worker.data_received.connect(self.update_plot)
        self.worker.status_message.connect(self.update_status)
        self.worker.hit_detected.connect(self.drum_widget.flash) # Connect hit to flash
        self.worker.start()
        
        # Trigger init
        self.change_mode(self.mode_combo.currentText())

    def change_mode(self, mode):
        self.worker.set_mode(mode)

    def update_status(self, msg):
        self.status_label.setText(msg)

    def update_plot(self, vals):
        self.val_label.setText(f"Raw: {vals}")
        for i in range(4):
            self.ys[i] = self.ys[i][1:] + [vals[i]]
            self.lines[i].setData(self.x, self.ys[i])

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TaikoWindow()
    window.show()
    sys.exit(app.exec_())