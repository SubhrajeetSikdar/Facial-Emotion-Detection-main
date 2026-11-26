"""Real-time / image / video emotion detection.

    python app.py                       # webcam 0
    python app.py --source 1            # other camera
    python app.py --source clip.mp4 --save out.mp4
    python app.py --source photo.jpg --save annotated.jpg
    python app.py --log session.csv     # log every detection

Keys: q/ESC quit | space pause | s screenshot | h toggle probability bars | m mirror
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from emotion.config import COLORS, EMOTIONS, OUTPUT_DIR
from emotion.face_detection import FaceDetector
from emotion.predictor import EmotionPredictor
from emotion.preprocessing import expand_box
from emotion.tracking import FaceTracker, label_of

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_face(frame, track, probs, show_bars: bool):
    x, y, w, h = track.box
    label, conf = label_of(probs)
    color = COLORS[label]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} {conf * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = y - 8 if y - th - 12 > 0 else y + h + th + 10
    cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 8, ty + 6), color, -1)
    cv2.putText(frame, text, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if show_bars:
        bx, bw = x + w + 8, 90
        if bx + bw + 60 > frame.shape[1]:
            bx = max(0, x - bw - 70)
        for i, (e, p) in enumerate(zip(EMOTIONS, probs)):
            yy = y + i * 16
            cv2.rectangle(frame, (bx, yy), (bx + int(bw * p), yy + 11), COLORS[e], -1)
            cv2.putText(frame, f"{e[:3]} {p * 100:.0f}", (bx + bw + 4, yy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def process_frame(frame, detector, predictor, tracker, show_bars=True, writer=None, frame_idx=0, t0=0.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = [expand_box(*b, frame.shape) for b in detector.detect(frame, gray)]
    tracks = tracker.assign(boxes)
    if tracks:
        crops = [gray[t.box[1]:t.box[1] + t.box[3], t.box[0]:t.box[0] + t.box[2]] for t in tracks]
        probs = predictor.predict_faces(crops)          # one batched call for all faces
        for t, b, p in zip(tracks, boxes, probs):
            t.update(b, p)
            sm = t.smoothed()
            draw_face(frame, t, sm, show_bars)
            if writer:
                lab, conf = label_of(sm)
                writer.writerow([f"{time.time() - t0:.3f}", frame_idx, t.tid, lab, f"{conf:.3f}",
                                 *[f"{v:.3f}" for v in sm]])
    else:
        cv2.putText(frame, "No face detected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return len(tracks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file or image file")
    ap.add_argument("--model", default=None)
    ap.add_argument("--save", default=None, help="write annotated output to this file")
    ap.add_argument("--log", default=None, help="CSV file for per-frame detections")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--process-every", type=int, default=1, help="run the CNN every N frames (speed)")
    ap.add_argument("--width", type=int, default=960, help="downscale frames wider than this")
    args = ap.parse_args()

    detector = FaceDetector()
    predictor = EmotionPredictor(args.model, tta=not args.no_tta)
    tracker = FaceTracker()
    print(f"model={predictor.path.name} detector={detector.backend}")

    log_f = writer = None
    if args.log:
        log_f = open(args.log, "w", newline="")
        writer = csv.writer(log_f)
        writer.writerow(["time_s", "frame", "face_id", "emotion", "confidence", *EMOTIONS])

    src = Path(args.source)
    # --- single image ---------------------------------------------------------------
    if src.suffix.lower() in IMAGE_EXT:
        frame = cv2.imread(str(src))
        if frame is None:
            raise SystemExit(f"Cannot read image: {src}")
        n = process_frame(frame, detector, predictor, tracker, writer=writer, t0=time.time())
        print(f"{n} face(s) found")
        out = args.save or str(OUTPUT_DIR / f"{src.stem}_annotated.jpg")
        OUTPUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(out, frame)
        print("saved", out)
        if not args.no_display:
            cv2.imshow("Emotion Detector", frame); cv2.waitKey(0)
        if log_f: log_f.close()
        return

    # --- camera / video -------------------------------------------------------------
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    is_cam = isinstance(source, int)
    out_vw, show_bars, mirror, paused = None, True, is_cam, False
    fps_s, last, idx, t0, frame, n = 0.0, time.time(), 0, time.time(), None, 0

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:                      # BUG FIX: original crashed on a failed read
                    break
                if frame.shape[1] > args.width:
                    s = args.width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                if mirror:
                    frame = cv2.flip(frame, 1)
                if idx % args.process_every == 0:
                    n = process_frame(frame, detector, predictor, tracker, show_bars, writer, idx, t0)
                else:                           # skipped frame: re-draw last known results (no flicker)
                    live = [t for t in tracker.tracks.values() if t.missed == 0 and t.history]
                    for t in live:
                        draw_face(frame, t, t.smoothed(), show_bars)
                idx += 1
                now = time.time()
                fps_s = 0.9 * fps_s + 0.1 / max(now - last, 1e-6)
                last = now
                cv2.putText(frame, f"FPS {fps_s:4.1f}  faces {n}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
                if args.save:
                    if out_vw is None:
                        h, w = frame.shape[:2]
                        out_vw = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
                    out_vw.write(frame)
            if not args.no_display:
                cv2.imshow("Emotion Detector", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27): break
                elif k == ord(" "): paused = not paused
                elif k == ord("h"): show_bars = not show_bars
                elif k == ord("m"): mirror = not mirror
                elif k == ord("s"):
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    p = OUTPUT_DIR / f"shot_{int(time.time())}.png"
                    cv2.imwrite(str(p), frame); print("saved", p)
    finally:
        cap.release()
        if out_vw: out_vw.release()
        if log_f: log_f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
