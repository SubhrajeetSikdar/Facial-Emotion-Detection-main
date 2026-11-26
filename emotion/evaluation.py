"""Metrics, confusion matrix and training-curve plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import EMOTIONS


def plot_history(h: dict, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path):
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Normalised confusion matrix")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate_model(model, dataset, out_dir: Path, tta: bool = True) -> dict:
    """dataset must be un-shuffled and yield (images, one-hot labels)."""
    y_true, y_prob = [], []
    for x, y in dataset:
        x = np.asarray(x)
        p = np.asarray(model(x, training=False))
        if tta:
            p = (p + np.asarray(model(x[:, :, ::-1, :], training=False))) / 2
        y_prob.append(p); y_true.append(np.argmax(np.asarray(y), 1))
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    y_pred = y_prob.argmax(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))
    plot_confusion(cm, out_dir / "confusion_matrix.png")
    rep = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0))
    return {"accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "per_class": {k: rep[k] for k in EMOTIONS}}
