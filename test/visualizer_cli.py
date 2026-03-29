import serial
import time
import shutil

# === SETTINGS ===
PORT = '/dev/ttyACM0'  # Change this!
BAUD = 115200
# ================

def print_bars(values):
    # Get terminal width
    columns, _ = shutil.get_terminal_size()
    max_val = 1023
    bar_width = int(columns / 2) - 10
    
    labels = ["DL", "DR", "KL", "KR"]
    
    # Move cursor up 4 lines to overwrite previous frame
    print("\033[4A", end="") 
    
    for label, val in zip(labels, values):
        # Calculate bar length
        length = int((val / max_val) * bar_width)
        bar = "█" * length
        # Pad with spaces to clear old characters
        padding = " " * (bar_width - length)
        print(f"{label} [{val:4d}]: |{bar}{padding}|")

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print("Connected! Hit the drum to see signal strength.")
        print("Values range from 0 to 1023.")
        print("\n\n\n\n") # Make space for the bars
        
        while True:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    try:
                        vals = list(map(int, line.split(',')))
                        if len(vals) == 4:
                            print_bars(vals)
                    except ValueError:
                        pass
            time.sleep(0.01) # Refresh rate limit
            
    except Exception as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()