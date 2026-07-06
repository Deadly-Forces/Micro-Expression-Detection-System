"""
static_features.py — Static facial landmark features for expression classification.

Extracts geometric features from a single frame's MediaPipe 478-point face mesh,
enabling classification without requiring frame pairs or temporal data.

The feature vector consists of:
    1. Normalized (x, y) landmark coordinates: 478 × 2 = 956 floats
    2. Hand-crafted geometric features (AU proxies): 15 floats

Total: 971 features

Used by both ``scripts/train_unified.py`` (training) and
``src/microex/pipeline.py`` (real-time inference).

Author : Micro-Expression Detection Team
"""

from __future__ import annotations

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

FEATURE_DIM: int = 971  # 956 coords + 15 geometric

# MediaPipe face mesh key landmark indices
# (Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry)

# Eye corners + upper/lower lids
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # p1–p6 for EAR
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]  # p1–p6 for EAR

# Eyebrow arch points
_RIGHT_BROW = [46, 53, 52, 65, 70]
_LEFT_BROW  = [276, 283, 282, 295, 300]

# Mouth key indices
_MOUTH_INNER_TOP    = 13
_MOUTH_INNER_BOTTOM = 14
_MOUTH_LEFT         = 61
_MOUTH_RIGHT        = 291
_UPPER_LIP_TOP      = 0    # philtrum / cupid's bow center
_LOWER_LIP_BOTTOM   = 17   # lowest point of lower lip

# Other anchors
_NOSE_TIP     = 1
_NOSE_BRIDGE  = 6
_CHIN         = 152
_FOREHEAD     = 10
_LEFT_CHEEK   = 234
_RIGHT_CHEEK  = 454


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2-D (or 3-D) points."""
    return float(np.linalg.norm(p1 - p2))


def _eye_aspect_ratio(pts: np.ndarray, indices: list) -> float:
    """Eye Aspect Ratio — measures how open/closed the eye is.

    EAR = (|p2−p6| + |p3−p5|) / (2 · |p1−p4|)
    """
    p = pts[indices]
    v1 = _dist(p[1], p[5])
    v2 = _dist(p[2], p[4])
    h  = _dist(p[0], p[3])
    return (v1 + v2) / (2.0 * h + 1e-8)


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def extract_static_features(
    landmarks_points: np.ndarray,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
) -> np.ndarray:
    """Extract a fixed-length feature vector from 478-pt MediaPipe face mesh.

    Parameters
    ----------
    landmarks_points : np.ndarray
        Shape ``(478, 2+)``.  Only the first two columns (x, y in pixels)
        are used.
    bbox_x, bbox_y, bbox_w, bbox_h : float
        Face bounding box used for spatial normalisation.

    Returns
    -------
    np.ndarray
        1-D float64 array of length :pydata:`FEATURE_DIM` (971).
    """
    if landmarks_points.shape[0] < 478:
        raise ValueError(
            f"Expected ≥ 478 landmarks, got {landmarks_points.shape[0]}"
        )

    # Work with x, y only (z is relative depth and can be noisy)
    pts = landmarks_points[:478, :2].copy().astype(np.float64)

    # ── Normalise to face bounding box [0, 1] ──────────────────────
    w = max(float(bbox_w), 1.0)
    h = max(float(bbox_h), 1.0)
    pts[:, 0] = (pts[:, 0] - float(bbox_x)) / w
    pts[:, 1] = (pts[:, 1] - float(bbox_y)) / h

    # 1) Flattened normalised coordinates  → 956 features
    coords = pts.flatten()

    # 2) Geometric / Action-Unit proxy features  → 15 features
    geo = np.zeros(15, dtype=np.float64)
    try:
        # [0-2] Eye Aspect Ratios
        ear_r = _eye_aspect_ratio(pts, _RIGHT_EYE)
        ear_l = _eye_aspect_ratio(pts, _LEFT_EYE)
        geo[0] = ear_r
        geo[1] = ear_l
        geo[2] = (ear_r + ear_l) / 2.0          # mean EAR

        # [3] Mouth Aspect Ratio (vertical / horizontal)
        mouth_v = _dist(pts[_MOUTH_INNER_TOP], pts[_MOUTH_INNER_BOTTOM])
        mouth_h = _dist(pts[_MOUTH_LEFT], pts[_MOUTH_RIGHT])
        geo[3] = mouth_v / (mouth_h + 1e-8)

        # [4] Mouth width relative to face
        geo[4] = mouth_h                          # already normalised

        # [5-6] Brow height relative to eye center (positive = brow raised)
        r_brow_y = np.mean(pts[_RIGHT_BROW, 1])
        r_eye_y  = np.mean(pts[_RIGHT_EYE, 1])
        l_brow_y = np.mean(pts[_LEFT_BROW, 1])
        l_eye_y  = np.mean(pts[_LEFT_EYE, 1])
        geo[5] = r_eye_y - r_brow_y
        geo[6] = l_eye_y - l_brow_y

        # [7] Nose tip → upper lip distance
        geo[7] = _dist(pts[_NOSE_TIP], pts[_MOUTH_INNER_TOP])

        # [8] Chin → lower lip distance
        geo[8] = _dist(pts[_CHIN], pts[_MOUTH_INNER_BOTTOM])

        # [9] Face aspect ratio  (height / width)
        face_h = _dist(pts[_FOREHEAD], pts[_CHIN])
        face_w = _dist(pts[_LEFT_CHEEK], pts[_RIGHT_CHEEK])
        geo[9] = face_h / (face_w + 1e-8)

        # [10-11] Mouth corners relative to nose tip (vertical offset)
        geo[10] = pts[_MOUTH_LEFT, 1]  - pts[_NOSE_TIP, 1]
        geo[11] = pts[_MOUTH_RIGHT, 1] - pts[_NOSE_TIP, 1]

        # [12] Eye openness asymmetry
        geo[12] = abs(ear_r - ear_l)

        # [13] Lip stretch (outer lip width)
        geo[13] = _dist(pts[_UPPER_LIP_TOP], pts[_LOWER_LIP_BOTTOM])

        # [14] Brow asymmetry
        geo[14] = abs(geo[5] - geo[6])

    except (IndexError, ValueError):
        pass  # Return zeros for geometric features on error

    return np.concatenate([coords, geo])
