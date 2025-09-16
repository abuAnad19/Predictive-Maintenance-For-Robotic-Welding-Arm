from pathlib import Path
import torch

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
VISION_RAW_DIR = DATA_RAW / "vision"
SENSORS_RAW_DIR = DATA_RAW / "sensors"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vision config
VISION_IMG_SIZE = (128, 128)   # H, W
VISION_NUM_CLASSES = 4         # update if dataset has different classes
VISION_BATCH_SIZE = 32
VISION_EPOCHS = 5
VISION_LR = 1e-3

# Sensor config
SENSOR_WINDOW = 256
SENSOR_STEP = 128
SENSOR_FEATURES = ["vibration", "current", "temp"]  # update when real CSV columns are known
SENSOR_CLASSES = ["normal", "fault"]
SENSOR_TEST_SIZE = 0.2
RANDOM_STATE = 42
SENSOR_LR = 1e-3
SENSOR_EPOCHS = 5
