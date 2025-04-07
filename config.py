# config.py
from pathlib import Path

# Base directories
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Model paths
MODEL_PATH = MODEL_DIR / "emotion_model.h5"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# WebSocket configuration
WS_SERVER_URI = "ws://localhost:8765/vnyan"

# Audio processing
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)