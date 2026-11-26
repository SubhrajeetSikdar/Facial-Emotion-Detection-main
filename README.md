# Facial Emotion Detection (CNN, Keras 3)

Real-time emotion recognition for 7 classes: angry, disgust, fear, happy, neutral, sad, surprise.

## Setup
```bash
pip install -r requirements.txt
```
Dataset: Kaggle *face-expression-recognition-dataset* (FER2013 images). Expected layout:
`images/train/<emotion>/*.jpg` and `images/validation/<emotion>/*.jpg`.

## Train (best accuracy)
```bash
python train.py --data path/to/images --epochs 100 --mixup 0.2
```
Outputs `models/emotion_model.keras`, `outputs/training_curves.png`, `outputs/confusion_matrix.png`,
`outputs/metrics.json`. Model selection uses a 10% split held out of `train/`; the `validation/`
folder is used **only once**, for the final report.

## Run
```bash
python app.py                          # webcam
python app.py --source clip.mp4 --save out.mp4 --log detections.csv
python app.py --source photo.jpg
python evaluate.py --data path/to/images/validation
python -m unittest discover tests      # no TensorFlow needed
```
Keys: `q`/Esc quit, space pause, `s` screenshot, `h` probability bars, `m` mirror.
Without a trained `emotion_model.keras`, the app falls back to `models/legacy_model.h5` (your original model).

## What changed vs. the original
**Bugs fixed**
- **Train/inference mismatch (the big one):** training used raw 0-255 pixels, but `main.py` divided by 255. Normalisation now lives inside the model, so it cannot diverge.
- `ModelCheckpoint` monitored `val_acc` (doesn't exist in modern Keras) so the best model was never saved.
- `fit_generator`, `lr=`, and `keras.preprocessing.*` removed/deprecated; `ImageDataGenerator` import was commented out; dataset path was a broken string.
- Hard-coded `C:\Users\...` paths replaced with relative paths.
- Crash when the camera returned no frame; the "No Faces" message could never show; `np.sum` "empty face" check was wrong.
- One `predict()` call per face per frame (slow) -> one batched call.
- Validation set was also used for early stopping and reported accuracy (leakage) -> separate held-out split.

**Accuracy**
- Residual CNN with squeeze-excitation (~1.5M params vs 4.5M; the old one overfit: 72% train vs 61% val).
- Augmentation (flip, rotation, shift, zoom, contrast, brightness), optional MixUp, label smoothing, sqrt-balanced class weights (helps `disgust`, only ~436 images).
- AdamW + weight decay + weight EMA, cosine LR with warm-up, per-image standardisation (lighting robust).
- Flip test-time augmentation at evaluation and in the app.

**Features added**
- Multi-face tracking with temporal smoothing (no label flicker), confidence %, per-emotion probability bars, FPS counter.
- Image / video / webcam input, annotated output saving, CSV logging, screenshot, pause, `--process-every` for speed.
- Optional YuNet face detector (drop `face_detection_yunet_2023mar.onnx` into `models/`), Haar fallback with tuned parameters.
- Evaluation script with confusion matrix and per-class precision/recall/F1.

## Realistic accuracy
FER2013 labels are noisy: human agreement is ~65-72%. A well-trained model of this kind typically lands around the high-60s to low-70s % on the test folder. Going meaningfully beyond that requires cleaner labels (FER+), more data (AffectNet), or pretrained backbones — not a bigger CNN.
