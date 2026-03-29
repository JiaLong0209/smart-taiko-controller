import serial
import time
import numpy as np
import joblib
import pyautogui

# 設定
PORT = 'COM3'
BAUD = 115200
WINDOW = 0.005  # <--- 這裡改成 5ms
COOLDOWN = 0.03 # 敲擊後的冷卻時間

# 載入模型 (記得要用 5ms 資料重新訓練過!)
model = joblib.load('taiko_model_5ms.pkl') 
ser = serial.Serial(PORT, BAUD)
last_hit = 0

print(f"極速 AI 太鼓啟動 (Window: {WINDOW*1000}ms)...")

try:
    while True:
        if ser.in_waiting:
            # 1. 觸發偵測 (簡單過濾底噪)
            # 先讀一行看看是否大於門檻，節省 CPU
            first_line = ser.readline().decode(errors='ignore').strip()
            if not first_line: continue
            
            try:
                vals = list(map(int, first_line.split(',')))
                if max(vals) > 50 and (time.time() - last_hit > COOLDOWN):
                    
                    # === 2. 開始 5ms 快速截取 ===
                    start = time.time()
                    batch = [vals] # 把第一筆加進去
                    
                    while (time.time() - start) < WINDOW:
                        if ser.in_waiting:
                            l = ser.readline().decode(errors='ignore').strip()
                            if l:
                                try:
                                    batch.append(list(map(int, l.split(','))))
                                except: pass
                    
                    # === 3. 找這 5ms 內的最大值 ===
                    # 雖然可能還沒到真正的最高點，但已經足夠判斷是誰了
                    peak = np.max(np.array(batch), axis=0)
                    
                    # === 4. AI 判斷 ===
                    pred = model.predict([peak])[0] # 預測
                    
                    # 執行按鍵
                    if pred == 'Left_Don': pyautogui.press('f')
                    elif pred == 'Right_Don': pyautogui.press('j')
                    elif pred == 'Left_Ka': pyautogui.press('d')
                    elif pred == 'Right_Ka': pyautogui.press('k')
                    
                    if pred != 'Noise':
                        print(f"判定: {pred} | 延遲: {(time.time()-start)*1000:.1f}ms")
                        last_hit = time.time()

            except ValueError:
                pass
except KeyboardInterrupt:
    ser.close()