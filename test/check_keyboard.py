import pyautogui
import time

def test_keyboard():
    print("--- Keyboard Simulation Test ---")
    print("This script will type 'f', 'j', 'd', 'k' in 3 seconds.")
    print("👉 Please click on a Text Editor or Notepad NOW.")
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("\nTyping now...")
    
    keys = ['f', 'j', 'd', 'k']
    
    for key in keys:
        pyautogui.press(key)
        print(f"Pressed: {key}")
        time.sleep(0.5)
        
    print("\n✅ Test Complete. Did you see 'fjdk' appear?")
    
    # Extra check for osu!
    print("\nNOTE for osu! players:")
    print("If this works in Notepad but NOT in game, try running this script as Administrator/Root.")

if __name__ == "__main__":
    test_keyboard()