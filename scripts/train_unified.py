#!/usr/bin/env python3
"""
train_unified.py — Unified multi-dataset training for facial expression classification.

Combines CK+, FER2013, Custom Portraits, and YOLO expression datasets.
Extracts static 478-pt MediaPipe face mesh features (971-D) per image.
Trains an SVM classifier with extensive hyperparameter tuning.

Usage
-----
    python scripts/train_unified.py                              # All datasets
    python scripts/train_unified.py --skip-fer2013 --skip-yolo   # Fast (CK+ + Portraits)
    python scripts/train_unified.py --use-cache                  # Reuse cached features

Author : Micro-Expression Detection Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.microex.face_detector import FaceDetector
from src.microex.landmarks import LandmarkExtractor
from src.microex.static_features import FEATURE_DIM, extract_static_features
from src.microex.utils import ensure_directory

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Label harmonisation
# ═══════════════════════════════════════════════════════════════════════

EMOTION_LABELS: List[str] = [
    "happiness", "sadness", "surprise", "fear",
    "disgust", "anger", "contempt",
]

_LABEL_MAP: Dict[str, Optional[str]] = {
    # CK+ / Custom Portraits
    "anger":     "anger",
    "contempt":  "contempt",
    "disgust":   "disgust",
    "fear":      "fear",
    "happy":     "happiness",
    "happiness": "happiness",
    "sad":       "sadness",
    "sadness":   "sadness",
    "surprise":  "surprise",
    "surprised": "surprise",
    # FER2013
    "neutral":   None,   # skip — not an expression class
    # YOLO
    "angry":     "anger",
    "natural":   None,   # skip — equivalent to neutral
    "sleepy":    None,   # skip — not in our label set
}


def harmonise_label(raw: str) -> Optional[str]:
    """Map a raw dataset label to the unified 7-class label set.

    Returns ``None`` for labels that should be skipped.
    """
    return _LABEL_MAP.get(raw.strip().lower())


# ═══════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════

def load_ck_plus(root: Path) -> List[Tuple[str, str]]:
    """Load CK+ — emotion-labelled directories of face images.

    Returns list of ``(image_path, unified_label)``.
    """
    samples: List[Tuple[str, str]] = []
    if not root.exists():
        logger.warning("CK+ path not found: %s", root)
        return samples

    for emotion_dir in sorted(root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        label = harmonise_label(emotion_dir.name)
        if label is None:
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((str(img_path), label))

    logger.info("CK+: loaded %d samples across %d classes",
                len(samples), len(set(l for _, l in samples)))
    return samples


def load_fer2013(
    root: Path,
    max_per_class: int = 500,
) -> List[Tuple[str, str]]:
    """Load FER2013 Training — emotion-labelled directories of 48×48 images.

    Samples up to *max_per_class* images per class for tractability.
    """
    samples: List[Tuple[str, str]] = []
    if not root.exists():
        logger.warning("FER2013 path not found: %s", root)
        return samples

    for emotion_dir in sorted(root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        label = harmonise_label(emotion_dir.name)
        if label is None:
            continue
        files = [
            p for p in sorted(emotion_dir.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if len(files) > max_per_class:
            random.seed(42)
            files = random.sample(files, max_per_class)
        for img_path in files:
            samples.append((str(img_path), label))

    logger.info("FER2013: loaded %d samples (max %d/class) across %d classes",
                len(samples), max_per_class, len(set(l for _, l in samples)))
    return samples


def load_custom_portraits(root: Path) -> List[Tuple[str, str]]:
    """Load custom portrait data — per-subject dirs with named emotion images.

    Layout: ``root/images/{subject_id}/{Emotion}.jpg``
    """
    samples: List[Tuple[str, str]] = []
    images_dir = root / "images"
    if not images_dir.exists():
        logger.warning("Custom portraits path not found: %s", images_dir)
        return samples

    for subj_dir in sorted(images_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        for img_path in sorted(subj_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            # The stem is the emotion name (e.g. "Anger", "Happy")
            raw_label = img_path.stem.strip().lower()
            label = harmonise_label(raw_label)
            if label is None:
                continue
            samples.append((str(img_path), label))

    logger.info("Custom Portraits: loaded %d samples across %d classes",
                len(samples), len(set(l for _, l in samples)))
    return samples


def load_yolo_dataset(
    root: Path,
    split: str = "train",
    max_per_class: int = 500,
) -> List[Tuple[str, str, Tuple[float, float, float, float]]]:
    """Load YOLO-format dataset with bounding box annotations.

    Returns list of ``(image_path, label, (cx, cy, w, h))`` where box
    coordinates are normalised [0, 1].
    """
    samples: list = []
    yaml_path = root / "data.yaml"
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logger.warning("YOLO %s split not found: %s", split, root)
        return samples

    # Parse class names from data.yaml
    class_names: List[str] = []
    if yaml_path.exists():
        import re
        text = yaml_path.read_text()
        m = re.search(r"names:\s*\[(.+?)\]", text)
        if m:
            class_names = [n.strip().strip("'\"") for n in m.group(1).split(",")]

    if not class_names:
        logger.warning("Could not parse YOLO class names from %s", yaml_path)
        return samples

    # Group label files by class
    per_class: Dict[str, List[Tuple[str, Tuple[float, ...]]]] = {}
    for lbl_file in labels_dir.iterdir():
        if lbl_file.suffix != ".txt":
            continue
        img_stem = lbl_file.stem
        # Find matching image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = images_dir / (img_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        try:
            with open(lbl_file, "r") as f:
                line = f.readline().strip()
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            continue

        if class_id >= len(class_names):
            continue

        raw_label = class_names[class_id]
        label = harmonise_label(raw_label)
        if label is None:
            continue

        per_class.setdefault(label, []).append(
            (str(img_path), (cx, cy, bw, bh))
        )

    # Sample per class
    random.seed(42)
    for label, items in per_class.items():
        if len(items) > max_per_class:
            items = random.sample(items, max_per_class)
        for img_path, bbox in items:
            samples.append((img_path, label, bbox))

    logger.info("YOLO (%s): loaded %d samples (max %d/class) across %d classes",
                split, len(samples), max_per_class,
                len(set(l for _, l, *_ in samples)))
    return samples


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════

def _prepare_image(img: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Ensure image is RGB and at least *min_size* px on each side."""
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape) == 2:  # grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


def extract_features_single(
    image_path: str,
    detector: FaceDetector,
    landmarker: LandmarkExtractor,
    yolo_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[np.ndarray]:
    """Extract 971-D static landmark features from one face image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    detector : FaceDetector
        Initialised face detector.
    landmarker : LandmarkExtractor
        Initialised landmark extractor.
    yolo_bbox : tuple or None
        YOLO-format bbox ``(cx, cy, w, h)`` normalised [0, 1].
        If provided, crops the face before landmark extraction.

    Returns
    -------
    np.ndarray or None
        Feature vector of shape ``(971,)`` or ``None`` on failure.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # ── YOLO bbox crop ───────────────────────────────────────────
    if yolo_bbox is not None:
        ih, iw = img.shape[:2]
        cx, cy, bw, bh = yolo_bbox
        x1 = max(0, int((cx - bw / 2) * iw))
        y1 = max(0, int((cy - bh / 2) * ih))
        x2 = min(iw, int((cx + bw / 2) * iw))
        y2 = min(ih, int((cy + bh / 2) * ih))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        img = img[y1:y2, x1:x2]

    img = _prepare_image(img, min_size=200)

    # ── Detect face ──────────────────────────────────────────────
    from src.microex.face_detector import BoundingBox

    boxes = detector.detect(img)
    if not boxes:
        # Fallback: treat full image as face
        h, w = img.shape[:2]
        boxes = [BoundingBox(x=0, y=0, w=w, h=h, confidence=1.0)]

    bbox = boxes[0]  # largest face

    # ── Extract landmarks ────────────────────────────────────────
    landmarks = landmarker.extract(img, bbox)
    if landmarks is None:
        return None
    if landmarks.points.shape[0] < 478:
        return None

    # ── Compute static features ──────────────────────────────────
    try:
        fv = extract_static_features(
            landmarks.points,
            bbox.x, bbox.y, bbox.w, bbox.h,
        )
    except Exception:
        return None

    if fv.shape[0] != FEATURE_DIM:
        return None

    return fv


def extract_all_features(
    samples: List[Tuple[str, str]],
    yolo_samples: List[Tuple[str, str, Tuple[float, ...]]],
    detector: FaceDetector,
    landmarker: LandmarkExtractor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-extract features from all samples with progress logging.

    Returns (X, y) where X has shape ``(n_successful, 971)`` and y is
    a string array of unified emotion labels.
    """
    total = len(samples) + len(yolo_samples)
    features: List[np.ndarray] = []
    labels: List[str] = []
    skipped = 0
    t0 = time.time()

    # ── Standard samples (image_path, label) ─────────────────────
    for i, (img_path, label) in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "  [%d/%d] extracting features … (%.1f img/s, %d skipped)",
                i + 1, total, rate, skipped,
            )
        fv = extract_features_single(img_path, detector, landmarker)
        if fv is not None:
            features.append(fv)
            labels.append(label)
        else:
            skipped += 1

    # ── YOLO samples (image_path, label, bbox) ───────────────────
    offset = len(samples)
    for i, (img_path, label, bbox) in enumerate(yolo_samples):
        idx = offset + i
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "  [%d/%d] extracting features … (%.1f img/s, %d skipped)",
                idx + 1, total, rate, skipped,
            )
        fv = extract_features_single(img_path, detector, landmarker, yolo_bbox=bbox)
        if fv is not None:
            features.append(fv)
            labels.append(label)
        else:
            skipped += 1

    elapsed = time.time() - t0
    logger.info(
        "Feature extraction complete: %d successful, %d skipped, %.1fs total",
        len(features), skipped, elapsed,
    )

    if not features:
        return np.empty((0, FEATURE_DIM)), np.array([])

    X = np.vstack(features)
    y = np.array(labels)
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Train SVM with GridSearchCV and evaluate via stratified K-fold.

    Returns dict with *model*, *scaler*, and evaluation *metrics*.
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, f1_score, recall_score,
    )

    logger.info("Training on %d samples × %d features, %d classes",
                X.shape[0], X.shape[1], len(np.unique(y)))
    logger.info("Class distribution: %s", dict(Counter(y)))

    # ── Scale features ───────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Grid search ──────────────────────────────────────────────
    param_grid = {
        "C":      [0.1, 1, 10, 50, 100],
        "gamma":  ["scale", "auto", 0.01, 0.001, 0.0001],
        "kernel": ["rbf"],
    }

    min_class = min(Counter(y).values())
    cv_splits = min(n_folds, min_class)
    if cv_splits < 2:
        cv_splits = 2
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    logger.info("GridSearchCV: %d param combos × %d folds = %d fits",
                len(param_grid["C"]) * len(param_grid["gamma"]),
                cv_splits,
                len(param_grid["C"]) * len(param_grid["gamma"]) * cv_splits)

    svm = SVC(probability=True, class_weight="balanced", random_state=42)
    grid = GridSearchCV(
        svm, param_grid,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_scaled, y)
    train_time = time.time() - t0

    best_model = grid.best_estimator_
    logger.info("Best params: %s  (CV F1-macro: %.4f, %.1fs)",
                grid.best_params_, grid.best_score_, train_time)

    # ── Full cross-val evaluation ────────────────────────────────
    logger.info("Running %d-fold stratified evaluation …", cv_splits)
    fold_metrics: List[Dict[str, float]] = []
    skf2 = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=123)

    for fold_i, (train_idx, test_idx) in enumerate(skf2.split(X_scaled, y)):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        fold_clf = SVC(**grid.best_params_, probability=True,
                       class_weight="balanced", random_state=42)
        fold_clf.fit(X_tr, y_tr)
        y_pred = fold_clf.predict(X_te)

        fold_acc = accuracy_score(y_te, y_pred)
        fold_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        fold_uar = recall_score(y_te, y_pred, average="macro", zero_division=0)
        fold_metrics.append({
            "fold": fold_i + 1,
            "accuracy": float(fold_acc),
            "f1_macro": float(fold_f1),
            "uar": float(fold_uar),
        })
        logger.info("  Fold %d: Acc=%.4f  F1=%.4f  UAR=%.4f",
                     fold_i + 1, fold_acc, fold_f1, fold_uar)

    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    avg_f1  = np.mean([m["f1_macro"] for m in fold_metrics])
    avg_uar = np.mean([m["uar"] for m in fold_metrics])

    # ── Final model on all data ──────────────────────────────────
    logger.info("Training final model on ALL %d samples …", len(y))
    final_model = SVC(**grid.best_params_, probability=True,
                      class_weight="balanced", random_state=42)
    final_model.fit(X_scaled, y)

    # Full-data classification report (train set — for reference only)
    y_pred_all = final_model.predict(X_scaled)
    report = classification_report(y, y_pred_all, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred_all, labels=EMOTION_LABELS)

    metrics = {
        "best_params": grid.best_params_,
        "cv_best_f1_macro": float(grid.best_score_),
        "train_time_s": float(train_time),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "class_distribution": dict(Counter(y)),
        "cv_folds": cv_splits,
        "cv_avg_accuracy": float(avg_acc),
        "cv_avg_f1_macro": float(avg_f1),
        "cv_avg_uar": float(avg_uar),
        "cv_per_fold": fold_metrics,
        "train_set_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": EMOTION_LABELS,
    }

    return {
        "model": final_model,
        "scaler": scaler,
        "metrics": metrics,
    }


def save_model_bundle(
    model: Any,
    scaler: Any,
    output_path: str,
    feature_dim: int = FEATURE_DIM,
) -> None:
    """Save model in EmotionClassifier-compatible format."""
    ensure_directory(str(Path(output_path).parent))
    bundle = {
        "model": model,
        "scaler": scaler,
        "model_type": "svm",
        "emotion_labels": EMOTION_LABELS,
        "feature_dim": feature_dim,
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved to %s (%.2f MB)",
                output_path, os.path.getsize(output_path) / 1e6)


# ═══════════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════════

def save_feature_cache(
    X: np.ndarray, y: np.ndarray, cache_path: str,
) -> None:
    """Save extracted features + labels to compressed NumPy archive."""
    ensure_directory(str(Path(cache_path).parent))
    np.savez_compressed(cache_path, X=X, y=y)
    logger.info("Feature cache saved to %s (%.2f MB)",
                cache_path, os.path.getsize(cache_path) / 1e6)


def load_feature_cache(
    cache_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load cached features if available."""
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    logger.info("Loaded cached features: %d samples × %d dims", X.shape[0], X.shape[1])
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# Confusion matrix visualisation
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray, labels: List[str], output_path: str,
) -> None:
    """Save a confusion matrix heatmap to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            ylabel="True label", xlabel="Predicted label",
            title="Confusion Matrix (Final Model — All Training Data)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        ensure_directory(str(Path(output_path).parent))
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not plot confusion matrix: %s", exc)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified multi-dataset training for facial expression classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root data directory containing all datasets.",
    )
    parser.add_argument(
        "--output-model", type=str, default="models/classifier.pkl",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--output-metrics", type=str, default="output/training_metrics.json",
        help="Path to save evaluation metrics.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=500,
        help="Max samples per class from large datasets (FER2013, YOLO).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--face-backend", type=str, default="haar",
        choices=["mediapipe", "haar", "dlib"],
        help="Face detection backend (haar is fastest for batch).",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Load cached features if available.",
    )
    parser.add_argument("--skip-ck", action="store_true", help="Skip CK+ dataset.")
    parser.add_argument("--skip-fer2013", action="store_true", help="Skip FER2013 dataset.")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO dataset.")
    parser.add_argument("--skip-portraits", action="store_true", help="Skip Custom Portraits.")

    args = parser.parse_args()
    data_root = Path(args.data_root)
    cache_path = str(data_root / "feature_cache.npz")

    # ── Try loading cached features ──────────────────────────────
    if args.use_cache:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            X, y = cached
            results = train_and_evaluate(X, y, n_folds=args.cv_folds)
            save_model_bundle(results["model"], results["scaler"], args.output_model)

            ensure_directory(str(Path(args.output_metrics).parent))
            with open(args.output_metrics, "w") as f:
                json.dump(results["metrics"], f, indent=2, default=str)
            logger.info("Metrics saved to %s", args.output_metrics)

            cm = np.array(results["metrics"]["confusion_matrix"])
            plot_confusion_matrix(cm, EMOTION_LABELS, "output/confusion_matrix.png")
            return 0

    # ── Initialise detection pipeline ────────────────────────────
    logger.info("Initialising face detector (%s) and landmark extractor …",
                args.face_backend)
    detector = FaceDetector(backend=args.face_backend)
    landmarker = LandmarkExtractor(model="mediapipe_mesh", static_image_mode=True)

    # ── Load datasets ────────────────────────────────────────────
    standard_samples: List[Tuple[str, str]] = []
    yolo_samples: list = []

    if not args.skip_ck:
        ck_path = data_root / "MicroExpression" / "ck+" / "ck+"
        standard_samples.extend(load_ck_plus(ck_path))

    if not args.skip_portraits:
        portrait_path = data_root / "data"
        standard_samples.extend(load_custom_portraits(portrait_path))

    if not args.skip_fer2013:
        fer_path = data_root / "MicroExpression" / "fer2013" / "fer2013" / "Training"
        standard_samples.extend(load_fer2013(fer_path, max_per_class=args.max_per_class))

    if not args.skip_yolo:
        yolo_path = data_root / "9 Facial Expressions you need"
        yolo_samples = load_yolo_dataset(
            yolo_path, split="train", max_per_class=args.max_per_class,
        )

    total = len(standard_samples) + len(yolo_samples)
    if total == 0:
        logger.error("No samples loaded! Check --data-root and dataset paths.")
        return 1
    logger.info("Total samples to process: %d", total)

    # ── Extract features ─────────────────────────────────────────
    logger.info("Extracting 971-D static landmark features …")
    X, y = extract_all_features(standard_samples, yolo_samples, detector, landmarker)

    if X.shape[0] < 20:
        logger.error("Only %d features extracted — too few to train. "
                      "Check face detection and landmark extraction.", X.shape[0])
        return 1

    # Cache features for future runs
    save_feature_cache(X, y, cache_path)

    # ── Clean up detection modules ───────────────────────────────
    try:
        detector.release()
        landmarker.release()
    except Exception:
        pass

    # ── Train and evaluate ───────────────────────────────────────
    results = train_and_evaluate(X, y, n_folds=args.cv_folds)

    # ── Save model ───────────────────────────────────────────────
    save_model_bundle(results["model"], results["scaler"], args.output_model)

    # ── Save metrics ─────────────────────────────────────────────
    ensure_directory(str(Path(args.output_metrics).parent))
    with open(args.output_metrics, "w") as f:
        json.dump(results["metrics"], f, indent=2, default=str)
    logger.info("Metrics saved to %s", args.output_metrics)

    # ── Plot confusion matrix ────────────────────────────────────
    cm = np.array(results["metrics"]["confusion_matrix"])
    plot_confusion_matrix(cm, EMOTION_LABELS, "output/confusion_matrix.png")

    # ── Summary ──────────────────────────────────────────────────
    m = results["metrics"]
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  TRAINING COMPLETE")
    logger.info("  Samples:      %d", m["n_samples"])
    logger.info("  Features:     %d", m["n_features"])
    logger.info("  Best C:       %s", m["best_params"].get("C"))
    logger.info("  Best gamma:   %s", m["best_params"].get("gamma"))
    logger.info("  CV Accuracy:  %.4f", m["cv_avg_accuracy"])
    logger.info("  CV F1-macro:  %.4f", m["cv_avg_f1_macro"])
    logger.info("  CV UAR:       %.4f", m["cv_avg_uar"])
    logger.info("  Model saved:  %s", args.output_model)
    logger.info("═══════════════════════════════════════════════════")

    return 0


if __name__ == "__main__":
    sys.exit(main())
