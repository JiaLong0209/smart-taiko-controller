# src/config.py



class Config:
    # Serial Settings
    SUPPORTED_MODELS = ['threshold','xgb', 'svm', 'rf', 'taikonet']
    MODEL_NAME = 'taikonet'
    # MODEL_NAME = 'rf'
    SERIAL_PORT = '/dev/ttyACM0'  # Linux: /dev/ttyUSB0 or /dev/ttyACM0 | Windows: 'COM3'
    BAUD_RATE = 115200            # Must match Arduino
    
    
    # AI & Capture Settings
    CAPTURE_WINDOW = 0.0        # 5ms capture window (Low Latency)
    CROSSTALK_RATIO = 1.3

    TRIGGER_THRESHOLD = 70

    # THRESHOLDS = [40, 40, 40, 60]

    TIME_THRESHOLD = 0.035          # 30ms cooldown to prevent double hits
    
    # Hardware Map (Sensor Order coming from Arduino)
    # Arduino sends: Don_L, Don_R, Ka_L, Ka_R
    SENSOR_LABELS = ['Don_Left', 'Don_Right', 'Ka_Left', 'Ka_Right']
    
    # Labels for Classification
    CLASSES = ['Don_Left', 'Don_Right', 'Ka_Left', 'Ka_Right', 'Noise']
    TRAINING_DATA_NUMBER = 100
    
    # Key Mapping for osu! (Check your game settings)
    KEY_MAP = {
        'Don_Left': 'f',
        'Don_Right': 'j',
        'Ka_Left': 'd',
        'Ka_Right': 'k',
        'Noise': None
    }

    CSV_PATH = '../data/taiko_data.csv'

    # taikonet
    TAIKONET_EPOCHS = 250
    TAIKONET_LR = 0.01