"""Central configuration. One source of truth for labels, sizes and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# Alphabetical order == the class order Keras' flow_from_directory / image_dataset_from_directory
# assigns to the folders of the Kaggle "face-expression-recognition-dataset".
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48

# BGR colours used when drawing results
COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 140, 0),
    "fear": (200, 0, 200),
    "happy": (0, 220, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 120, 0),
    "surprise": (0, 165, 255),
}

DEFAULT_MODEL = MODELS_DIR / "emotion_model.keras"
LEGACY_MODEL = MODELS_DIR / "legacy_model.h5"
CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
