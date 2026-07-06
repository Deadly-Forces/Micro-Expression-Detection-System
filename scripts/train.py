#!/usr/bin/env python3
"""
train.py — Training entry point for the Micro-Expression Detection System.

Loads dataset, extracts features, trains classifier, evaluates with LOSO-CV,
and saves the trained model.

Usage:
    python scripts/train.py --dataset-path data/CASME2 --output-model models/classifier.pkl
    python scripts/train.py --help
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.microex.face_detector import FaceDetector
from src.microex.landmarks import LandmarkExtractor
from src.microex.motion_features import MotionFeatureExtractor
from src.microex.classifier import EmotionClassifier
from src.microex.utils import apply_clahe, ensure_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def extract_features_from_sequence(
    frame_paths: List[str],
    detector: FaceDetector,
    landmark_ext: LandmarkExtractor,
    motion_ext: MotionFeatureExtractor,
) -> List[np.ndarray]:
    """
    Extract optical flow feature vectors from a sequence of frames.
    For CK+, we extract the flow between the onset (first frame) 
    and apex (last frame).

    Args:
        frame_paths: List of paths to image frames.
        detector: Face detection module.
        landmark_ext: Landmark extraction module.
        motion_ext: Motion feature extraction module.

    Returns:
        List containing a single feature vector (if successful).
    """
    if len(frame_paths) < 2:
        return []

    # Get onset frame (first)
    onset_frame = cv2.imread(frame_paths[0])
    if onset_frame is None: return []
    
    onset_boxes = detector.detect(onset_frame)
    if not onset_boxes: return []
    onset_lm = landmark_ext.extract(onset_frame, onset_boxes[0])
    if onset_lm is None: return []
    
    onset_rois = landmark_ext.get_rois(onset_frame, onset_lm)
    onset_gray = cv2.cvtColor(onset_rois.full_face, cv2.COLOR_BGR2GRAY)
    onset_gray = apply_clahe(onset_gray)

    # Get apex frame (last)
    apex_frame = cv2.imread(frame_paths[-1])
    if apex_frame is None: return []
    
    apex_boxes = detector.detect(apex_frame)
    if not apex_boxes: return []
    apex_lm = landmark_ext.extract(apex_frame, apex_boxes[0])
    if apex_lm is None: return []
    
    apex_rois = landmark_ext.get_rois(apex_frame, apex_lm)
    apex_gray = cv2.cvtColor(apex_rois.full_face, cv2.COLOR_BGR2GRAY)
    apex_gray = apply_clahe(apex_gray)

    # Ensure sizes match (bounding boxes might drift slightly)
    if onset_gray.shape != apex_gray.shape:
        apex_gray = cv2.resize(apex_gray, (onset_gray.shape[1], onset_gray.shape[0]))

    motion_ext.reset()
    flow = motion_ext.compute_flow(onset_gray, apex_gray)
    flow_feats = motion_ext.extract_features(flow, "full_face")
    
    # NEW: Also compute 3D landmark displacements for highly discriminative features
    face_w = float(onset_lm.face_bbox.w) if onset_lm.face_bbox.w > 0 else 1.0
    
    if len(onset_lm.points) == len(apex_lm.points):
        # shape is (N, 3), flattening to (N * 3,)
        disp = ((apex_lm.points - onset_lm.points) / face_w).flatten()
    else:
        disp = np.zeros(478 * 3, dtype=np.float32)

    # Combine optical flow magnitude stats with rich 3D structural displacements
    fv = np.concatenate([
        flow_feats.magnitude_histogram,
        flow_feats.angle_histogram,
        np.array([flow_feats.mean_magnitude / face_w, flow_feats.max_magnitude / face_w]),
        disp,
    ])

    return [fv]


def load_custom_dataset(
    dataset_path: str,
) -> Tuple[List[List[str]], List[str], List[str]]:
    root = Path(dataset_path)
    images_dir = root / "images"
    
    if not images_dir.exists():
        logger.error("No images/ directory found in dataset path.")
        return [], [], []
        
    frame_sequences = []
    labels = []
    subject_ids = []
    
    for subject_dir in images_dir.iterdir():
        if not subject_dir.is_dir():
            continue
            
        subject_id = subject_dir.name
        neutral_path = subject_dir / "Neutral.jpg"
        if not neutral_path.exists():
            # some sets might use lowercase, let's check
            neutral_path = subject_dir / "neutral.jpg"
            if not neutral_path.exists():
                continue
            
        for img_file in subject_dir.glob("*.[jJ][pP][gG]"):
            if img_file.stem.lower() == "neutral":
                continue
                
            label = img_file.stem.lower()
            frame_sequences.append([str(neutral_path), str(img_file)])
            labels.append(label)
            subject_ids.append(subject_id)
            
    logger.info(
        "Found %d sequences from %d subjects across %d emotion classes.",
        len(frame_sequences),
        len(set(subject_ids)),
        len(set(labels))
    )
    return frame_sequences, labels, subject_ids


def loso_cross_validation(
    features_by_video: Dict[str, np.ndarray],
    labels_by_video: Dict[str, str],
    subjects_by_video: Dict[str, str],
    classifier_type: str = "svm",
) -> Dict[str, Any]:
    """
    Perform Leave-One-Subject-Out cross-validation.

    Args:
        features_by_video: Video ID → feature matrix.
        labels_by_video: Video ID → label.
        subjects_by_video: Video ID → subject ID.
        classifier_type: Type of classifier to use.

    Returns:
        Dict with per-fold and aggregate metrics.
    """
    unique_subjects = sorted(set(subjects_by_video.values()))
    fold_results = []

    for held_out in unique_subjects:
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        for vid_id in features_by_video:
            feats = features_by_video[vid_id]
            label = labels_by_video[vid_id]
            subject = subjects_by_video[vid_id]

            if feats.ndim == 1:
                feats = feats.reshape(1, -1)

            if subject == held_out:
                test_features.append(feats)
                test_labels.extend([label] * len(feats))
            else:
                train_features.append(feats)
                train_labels.extend([label] * len(feats))

        if not train_features or not test_features:
            continue

        X_train = np.vstack(train_features)
        X_test = np.vstack(test_features)
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        clf = EmotionClassifier(model_type=classifier_type)
        clf.train(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)

        fold_results.append({
            "held_out_subject": held_out,
            "n_train": len(y_train),
            "n_test": len(y_test),
            **metrics,
        })

    # Aggregate
    if fold_results:
        avg_uf1 = np.mean([f.get("uf1", 0) for f in fold_results])
        avg_uar = np.mean([f.get("uar", 0) for f in fold_results])
        avg_accuracy = np.mean([f.get("accuracy", 0) for f in fold_results])
    else:
        avg_uf1 = avg_uar = avg_accuracy = 0.0

    return {
        "n_folds": len(fold_results),
        "avg_uf1": float(avg_uf1),
        "avg_uar": float(avg_uar),
        "avg_accuracy": float(avg_accuracy),
        "per_fold": fold_results,
    }


def main() -> int:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train the Micro-Expression Detection System classifier.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset root directory (e.g., data/CASME2).",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/classifier.pkl",
        help="Path to save trained model (default: models/classifier.pkl).",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["svm", "lstm", "cnn"],
        default="svm",
        help="Classifier type (default: svm).",
    )
    parser.add_argument(
        "--face-backend",
        type=str,
        choices=["mediapipe", "haar", "dlib"],
        default="haar",
        help="Face detection backend (default: haar).",
    )
    parser.add_argument(
        "--flow-method",
        type=str,
        choices=["farneback", "lucas_kanade"],
        default="farneback",
        help="Optical flow method (default: farneback).",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Perform Leave-One-Subject-Out cross-validation.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save evaluation metrics JSON.",
    )

    args = parser.parse_args()

    # Initialize modules
    detector = FaceDetector(backend=args.face_backend)
    landmark_ext = LandmarkExtractor()
    motion_ext = MotionFeatureExtractor(method=args.flow_method)

    # Load dataset
    seq_paths, labels, subject_ids = load_custom_dataset(args.dataset_path)
    if not seq_paths:
        logger.error("No valid sequences found in dataset path: %s", args.dataset_path)
        return 1

    # Extract features
    logger.info("Extracting features from %d sequences...", len(seq_paths))
    all_features = []
    all_labels = []
    features_by_video = {}
    labels_by_video = {}
    subjects_by_video = {}

    for i, (seq, label, subj) in enumerate(zip(seq_paths, labels, subject_ids)):
        seq_id = subj + "_" + str(i) # Use a generated ID based on subject
        logger.info("  [%d/%d] %s (Emotion: %s, Frames: %d)", i + 1, len(seq_paths), seq_id, label, len(seq))
        feats = extract_features_from_sequence(seq, detector, landmark_ext, motion_ext)
        if feats:
            stacked = np.vstack(feats)
            vid_id = seq_id
            features_by_video[vid_id] = stacked
            labels_by_video[vid_id] = label
            subjects_by_video[vid_id] = subj
            all_features.extend(feats)
            all_labels.extend([label] * len(feats))

    if not all_features:
        logger.error("No features extracted. Check dataset and detection pipeline.")
        return 1

    X = np.vstack(all_features)
    y = np.array(all_labels)
    logger.info("Total features: %d samples × %d dimensions", X.shape[0], X.shape[1])

    # Train
    if args.loso:
        logger.info("Running LOSO cross-validation...")
        cv_results = loso_cross_validation(
            features_by_video, labels_by_video, subjects_by_video, args.classifier,
        )
        logger.info("LOSO Results — UF1: %.4f, UAR: %.4f, Accuracy: %.4f",
                     cv_results["avg_uf1"], cv_results["avg_uar"], cv_results["avg_accuracy"])

        if args.output_metrics:
            ensure_directory(str(Path(args.output_metrics).parent))
            with open(args.output_metrics, "w") as f:
                json.dump(cv_results, f, indent=2, default=str)
            logger.info("Metrics saved to %s", args.output_metrics)

    # Train final model on all data
    logger.info("Training final model on all data...")
    classifier = EmotionClassifier(model_type=args.classifier)
    train_metrics = classifier.train(X, y)
    logger.info("Training metrics: %s", train_metrics)

    # Save
    ensure_directory(str(Path(args.output_model).parent))
    classifier.save_model(args.output_model)
    logger.info("Model saved to %s", args.output_model)

    # Cleanup
    detector.release()
    landmark_ext.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
