"""Evaluate any saved model on a dataset split.

    python evaluate.py --data /path/to/images/validation [--model models/emotion_model.keras]
"""
import argparse
import json

import keras

from emotion.config import EMOTIONS, IMG_SIZE, OUTPUT_DIR
from emotion.evaluation import evaluate_model
from emotion.predictor import resolve_model_path

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="folder with one sub-folder per emotion")
p.add_argument("--model")
p.add_argument("--no-tta", action="store_true")
a = p.parse_args()

ds = keras.utils.image_dataset_from_directory(
    a.data, label_mode="categorical", class_names=EMOTIONS, color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=128, shuffle=False)
model = keras.models.load_model(resolve_model_path(a.model), compile=False)
rep = evaluate_model(model, ds, OUTPUT_DIR, tta=not a.no_tta)
print(json.dumps({k: v for k, v in rep.items() if k != "per_class"}, indent=2))
