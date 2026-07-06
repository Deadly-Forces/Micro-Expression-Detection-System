"""
Shared utility functions for the Micro-Expression Detection System.

All helpers work CPU-only with standard OpenCV and NumPy.  Every function
carries full type-hints, docstrings, input validation, and exception
handling.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """Draw a bounding box (and optional label) on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H×W×3, uint8).
    bbox : tuple[int, int, int, int]
        ``(x, y, w, h)`` in pixel coordinates.
    color : tuple[int, int, int]
        BGR colour for the box.
    thickness : int
        Line thickness in pixels.
    label : str or None
        Text label drawn above the box.
    confidence : float or None
        If supplied alongside *label*, the confidence value is appended.

    Returns
    -------
    np.ndarray
        The *frame* with drawings applied (modified in-place).

    Raises
    ------
    ValueError
        If *frame* is not a valid image array or *bbox* has wrong length.
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be a valid image array (2-D or 3-D)")
    try:
        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
    except AttributeError:
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 elements (x,y,w,h), got {len(bbox)}")
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    if label is not None:
        text = label
        if confidence is not None:
            text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        # Background rectangle
        cv2.rectangle(
            frame,
            (x, y - th - baseline - 4),
            (x + tw + 4, y),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (x + 2, y - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """Draw facial landmark points on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image.
    landmarks : np.ndarray
        Shape ``(N, 2)`` — (x, y) pixel coordinates for *N* landmarks.
    color : tuple[int, int, int]
        BGR colour.
    radius : int
        Circle radius in pixels.

    Returns
    -------
    np.ndarray
        Modified *frame*.

    Raises
    ------
    ValueError
        If *landmarks* does not have shape ``(N, 2)``.
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be a valid image array")
    landmarks = np.asarray(landmarks)
    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError(
            f"landmarks must have shape (N, 2), got {landmarks.shape}"
        )

    for pt in landmarks:
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)

    return frame


def draw_flow_field(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Visualise a dense optical-flow field as arrows overlaid on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H×W×3).
    flow : np.ndarray
        Shape ``(H, W, 2)`` — optical-flow vectors (dx, dy) per pixel.
    step : int
        Grid spacing (in pixels) between arrow origins.
    color : tuple[int, int, int]
        BGR colour for arrows.

    Returns
    -------
    np.ndarray
        Modified *frame*.

    Raises
    ------
    ValueError
        If *flow* shape is incompatible with *frame*.
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be a valid image array")
    if flow is None or flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"flow must have shape (H, W, 2), got "
            f"{flow.shape if flow is not None else None}"
        )
    if flow.shape[0] != frame.shape[0] or flow.shape[1] != frame.shape[1]:
        raise ValueError(
            f"flow spatial dims {flow.shape[:2]} must match "
            f"frame dims {frame.shape[:2]}"
        )

    step = max(1, step)
    h, w = frame.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            end_x = int(x + dx)
            end_y = int(y + dy)
            cv2.arrowedLine(
                frame, (x, y), (end_x, end_y), color, 1, cv2.LINE_AA, tipLength=0.3
            )

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Image processing helpers
# ═══════════════════════════════════════════════════════════════════════════

def resize_with_aspect_ratio(
    frame: np.ndarray,
    target_width: int = 640,
) -> np.ndarray:
    """Resize *frame* so its width equals *target_width*, preserving the
    aspect ratio.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    target_width : int
        Desired output width in pixels.

    Returns
    -------
    np.ndarray
        Resized image.

    Raises
    ------
    ValueError
        If *target_width* is not positive or *frame* is invalid.
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be a valid image array")
    if target_width <= 0:
        raise ValueError(f"target_width must be > 0, got {target_width}")

    h, w = frame.shape[:2]
    if w == 0:
        raise ValueError("frame width is 0")
    scale = target_width / w
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def apply_clahe(
    frame_gray: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    to a grayscale image.

    Parameters
    ----------
    frame_gray : np.ndarray
        Single-channel (H×W) uint8 grayscale image.
    clip_limit : float
        CLAHE clip limit.
    grid_size : tuple[int, int]
        Tile grid size for local histogram equalisation.

    Returns
    -------
    np.ndarray
        Equalised grayscale image (same shape).

    Raises
    ------
    ValueError
        If *frame_gray* is not 2-D / single-channel.
    """
    if frame_gray is None or frame_gray.ndim != 2:
        raise ValueError(
            "frame_gray must be a 2-D (single-channel) array, "
            f"got ndim={getattr(frame_gray, 'ndim', None)}"
        )
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(frame_gray.astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic test data
# ═══════════════════════════════════════════════════════════════════════════

def create_synthetic_face_frame(
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Generate a synthetic BGR frame with drawn face-like features.

    Useful for unit / integration tests that need a realistic-ish input
    without requiring actual images or cameras.

    Parameters
    ----------
    width : int
        Frame width.
    height : int
        Frame height.

    Returns
    -------
    np.ndarray
        BGR uint8 image of shape ``(height, width, 3)``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"width and height must be > 0, got ({width}, {height})"
        )

    frame = np.full((height, width, 3), 200, dtype=np.uint8)  # light grey bg

    cx, cy = width // 2, height // 2
    face_rx, face_ry = width // 5, height // 4

    # Face oval
    cv2.ellipse(
        frame, (cx, cy), (face_rx, face_ry), 0, 0, 360,
        (180, 200, 220), -1, cv2.LINE_AA,
    )

    # Eyes
    eye_y = cy - face_ry // 3
    eye_offset = face_rx // 3
    eye_r = max(3, face_rx // 8)
    cv2.circle(frame, (cx - eye_offset, eye_y), eye_r, (50, 50, 50), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + eye_offset, eye_y), eye_r, (50, 50, 50), -1, cv2.LINE_AA)

    # Eyebrows
    brow_y = eye_y - eye_r - max(3, face_ry // 10)
    brow_half_w = max(5, face_rx // 4)
    cv2.line(
        frame,
        (cx - eye_offset - brow_half_w, brow_y),
        (cx - eye_offset + brow_half_w, brow_y - 3),
        (80, 60, 40), 2, cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (cx + eye_offset - brow_half_w, brow_y - 3),
        (cx + eye_offset + brow_half_w, brow_y),
        (80, 60, 40), 2, cv2.LINE_AA,
    )

    # Nose
    nose_top = cy - face_ry // 10
    nose_bot = cy + face_ry // 5
    cv2.line(frame, (cx, nose_top), (cx, nose_bot), (140, 130, 120), 2, cv2.LINE_AA)
    nose_w = max(3, face_rx // 6)
    cv2.line(
        frame, (cx - nose_w, nose_bot), (cx + nose_w, nose_bot),
        (140, 130, 120), 2, cv2.LINE_AA,
    )

    # Mouth
    mouth_y = cy + face_ry // 2
    mouth_half_w = max(5, face_rx // 3)
    cv2.ellipse(
        frame, (cx, mouth_y), (mouth_half_w, max(2, face_ry // 10)),
        0, 0, 180, (60, 60, 180), 2, cv2.LINE_AA,
    )

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# File-system / time helpers
# ═══════════════════════════════════════════════════════════════════════════

def ensure_directory(path: str) -> str:
    """Create *path* (and parents) if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to ensure.

    Returns
    -------
    str
        The normalised absolute path.

    Raises
    ------
    ValueError
        If *path* is empty.
    OSError
        If directory creation fails.
    """
    if not path or not path.strip():
        raise ValueError("path must be a non-empty string")
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return the current local time as a formatted string.

    Parameters
    ----------
    fmt : str
        ``strftime`` format string.

    Returns
    -------
    str
        Formatted timestamp.
    """
    return datetime.now().strftime(fmt)


# ═══════════════════════════════════════════════════════════════════════════
# Trial block
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    passed: int = 0
    failed: int = 0

    def _report(name: str, ok: bool, reason: str = "") -> None:
        global passed, failed
        tag = "PASS" if ok else "FAIL"
        suffix = f" — {reason}" if reason else ""
        print(f"  [{tag}] {name}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print("utils — Trial Block")
    print("=" * 60)

    # ── create_synthetic_face_frame ──────────────────────────────────────
    try:
        face = create_synthetic_face_frame(320, 240)
        _report(
            "create_synthetic_face_frame",
            face.shape == (240, 320, 3) and face.dtype == np.uint8,
            f"shape={face.shape}, dtype={face.dtype}",
        )
    except Exception as exc:
        _report("create_synthetic_face_frame", False, str(exc))

    # ── draw_bounding_box ────────────────────────────────────────────────
    try:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = draw_bounding_box(frame, (10, 10, 50, 50), label="test", confidence=0.95)
        _report(
            "draw_bounding_box",
            out.shape == (100, 100, 3),
            f"shape={out.shape}",
        )
    except Exception as exc:
        _report("draw_bounding_box", False, str(exc))

    # ── draw_bounding_box — bad bbox ─────────────────────────────────────
    try:
        draw_bounding_box(np.zeros((100, 100, 3), np.uint8), (1, 2, 3))
        _report("draw_bounding_box bad bbox", False, "No ValueError raised")
    except ValueError:
        _report("draw_bounding_box bad bbox", True)
    except Exception as exc:
        _report("draw_bounding_box bad bbox", False, f"Wrong error: {exc}")

    # ── draw_landmarks ───────────────────────────────────────────────────
    try:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pts = np.array([[25, 30], [50, 30], [37, 50], [30, 70], [45, 70]])
        out = draw_landmarks(frame, pts, radius=3)
        has_drawing = np.any(out > 0)
        _report("draw_landmarks", has_drawing, f"shape={out.shape}")
    except Exception as exc:
        _report("draw_landmarks", False, str(exc))

    # ── draw_flow_field ──────────────────────────────────────────────────
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        flow = np.random.randn(64, 64, 2).astype(np.float32) * 5
        out = draw_flow_field(frame, flow, step=8)
        _report("draw_flow_field", out.shape == (64, 64, 3), f"shape={out.shape}")
    except Exception as exc:
        _report("draw_flow_field", False, str(exc))

    # ── resize_with_aspect_ratio ─────────────────────────────────────────
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = resize_with_aspect_ratio(frame, target_width=320)
        expected_h = int(480 * (320 / 640))
        _report(
            "resize_with_aspect_ratio",
            out.shape[1] == 320 and out.shape[0] == expected_h,
            f"shape={out.shape}, expected_h={expected_h}",
        )
    except Exception as exc:
        _report("resize_with_aspect_ratio", False, str(exc))

    # ── apply_clahe ──────────────────────────────────────────────────────
    try:
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        out = apply_clahe(gray, clip_limit=3.0, grid_size=(4, 4))
        _report(
            "apply_clahe",
            out.shape == (100, 100) and out.dtype == np.uint8,
            f"shape={out.shape}, dtype={out.dtype}",
        )
    except Exception as exc:
        _report("apply_clahe", False, str(exc))

    # ── apply_clahe — bad input ──────────────────────────────────────────
    try:
        apply_clahe(np.zeros((10, 10, 3), np.uint8))
        _report("apply_clahe bad input", False, "No ValueError raised")
    except ValueError:
        _report("apply_clahe bad input", True)
    except Exception as exc:
        _report("apply_clahe bad input", False, f"Wrong error: {exc}")

    # ── ensure_directory ─────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="microex_util_trial_")
    try:
        test_dir = os.path.join(tmp_dir, "a", "b", "c")
        result = ensure_directory(test_dir)
        _report(
            "ensure_directory",
            os.path.isdir(result),
            f"path={result}",
        )
    except Exception as exc:
        _report("ensure_directory", False, str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── ensure_directory — empty path ────────────────────────────────────
    try:
        ensure_directory("")
        _report("ensure_directory empty", False, "No ValueError raised")
    except ValueError:
        _report("ensure_directory empty", True)
    except Exception as exc:
        _report("ensure_directory empty", False, f"Wrong error: {exc}")

    # ── get_timestamp_str ────────────────────────────────────────────────
    try:
        ts = get_timestamp_str()
        ts_ok = len(ts) == 15 and "_" in ts  # YYYYMMDD_HHMMSS
        _report("get_timestamp_str", ts_ok, f"value={ts!r}")
    except Exception as exc:
        _report("get_timestamp_str", False, str(exc))

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests PASSED.")
        sys.exit(0)
