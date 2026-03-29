import serial
import time
import sys

# === SETTINGS ===
PORT = '/dev/ttyACM0'  # Change this! (Linux: /dev/ttyUSB0)
BAUD = 115200
# ================

def test_serial():
    print(f"--- Attempting to connect to {PORT} at {BAUD} ---")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2) # Wait for Arduino to reset
        print("✅ Connection Successful!")
        print("Please hit your sensors. You should see raw numbers below.")
        print("Format: Don_L, Don_R, Ka_L, Ka_R")
        print("(Press Ctrl+C to stop)\n")

        while True:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"Received: {line}")
                except UnicodeDecodeError:
                    print("⚠️ Decode Error (Noise)")
                    
    except serial.SerialException as e:
        print(f"❌ Error: Could not open port {PORT}.")
        print("Hint: Is Arduino IDE Serial Monitor open? Close it.")
        print(f"Details: {e}")
    except KeyboardInterrupt:
        print("\nTest finished.")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    test_serial()

# sudo fuser -k /dev/ttyACM0