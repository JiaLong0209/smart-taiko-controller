import serial
import time
import joblib
import numpy as np
import pyautogui
import os
from config import Config

# Optimization: Disable pyautogui pause fail-safe for speed
pyautogui.PAUSE = 0 

class TaikoController:
    def __init__(self):
        self.ser = None
        self.model = None
        self.last_hit_time = 0
        
    def load_resources(self):
        model_path = '../models/taiko_rf_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found! Train it first.")
        
        print(f"Loading AI Model from {model_path}...")
        self.model = joblib.load(model_path)
        
        print(f"Opening Serial Port {Config.SERIAL_PORT}...")
        self.ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=0.01)
        # Flush initial garbage data
        self.ser.reset_input_buffer()

    def run(self):
        print(f"\n=== AI Taiko Controller Running ===")
        print(f"Mode: {Config.CAPTURE_WINDOW*1000}ms Low-Latency Window")
        print("Press Ctrl+C to stop.\n")
        
        try:
            while True:
                if self.ser.in_waiting:
                    # Read the first available line
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line: continue
                    
                    try:
                        vals = list(map(int, line.split(',')))
                        
                        # === 1. Threshold Trigger ===
                        # Check if signal exceeds threshold AND cooldown has passed
                        if max(vals) > Config.TRIGGER_THRESHOLD and \
                           (time.time() - self.last_hit_time > Config.COOLDOWN_TIME):
                            
                            self._process_hit(vals)
                            
                    except ValueError:
                        continue
                        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if self.ser: self.ser.close()

    def _process_hit(self, first_val):
        """
        Captures the 5ms window, finds the peak, predicts, and presses key.
        """
        batch = [first_val]
        start_time = time.time()
        
        # === 2. Capture Window (The Race) ===
        # Collect data for exactly 5ms
        while (time.time() - start_time) < Config.CAPTURE_WINDOW:
            if self.ser.in_waiting:
                l = self.ser.readline().decode(errors='ignore').strip()
                if l:
                    try:
                        batch.append(list(map(int, l.split(','))))
                    except: pass
        
        # === 3. Feature Extraction ===
        # Find the max peak for each sensor in this short window
        peak_vals = np.max(np.array(batch), axis=0)
        
        # === 4. AI Prediction ===
        # Reshape to 2D array: [[dL, dR, kL, kR]]
        prediction = self.model.predict([peak_vals])[0]
        
        # === 5. Actuation ===
        key = Config.KEY_MAP.get(prediction)
        
        if key:
            pyautogui.press(key)
            latency = (time.time() - start_time) * 1000
            print(f"Hit: {prediction:<10} | Key: {key} | Latency: {latency:.1f}ms | Peak: {peak_vals}")
            
            # Reset cooldown timer
            self.last_hit_time = time.time()

if __name__ == "__main__":
    app = TaikoController()
    app.load_resources()
    app.run()