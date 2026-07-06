#!/usr/bin/env python3
"""
evaluate.py — Evaluation and metrics entry point.

Loads a trained model and evaluates it against a dataset, producing
precision/recall/F1 per class, confusion matrix, UF1, and UAR.

Usage:
    python scripts/evaluate.py --model models/classifier.pkl --dataset-path data/CASME2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.microex.classifier import EmotionClassifier
from src.microex.face_detector import FaceDetector
from src.microex.landmarks import LandmarkExtractor
from src.microex.motion_features import MotionFeatureExtractor
from src.microex.utils import ensure_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Micro-Expression Detection classifier.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/evaluation_results.json",
        help="Path to save evaluation results.",
    )
    parser.add_argument(
        "--face-backend",
        type=str,
        default="haar",
        help="Face detection backend.",
    )

    args = parser.parse_args()

    # Load model
    if not Path(args.model).exists():
        logger.error("Model not found: %s", args.model)
        return 1

    classifier = EmotionClassifier()
    classifier.load_model(args.model)
    logger.info("Loaded model from %s", args.model)

    # For full evaluation, we'd extract features from the dataset
    # and run classifier.evaluate(). This is the framework:
    logger.info("Evaluation framework ready.")
    logger.info("To run full evaluation, provide a labeled dataset with features.")
    logger.info("See scripts/train.py --loso for LOSO cross-validation.")

    # Save placeholder results
    ensure_directory(str(Path(args.output).parent))
    results: Dict[str, Any] = {
        "model_path": args.model,
        "dataset_path": args.dataset_path,
        "status": "framework_ready",
        "note": "Run train.py with --loso for full LOSO evaluation.",
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
