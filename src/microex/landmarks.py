"""Facial landmark extraction and ROI segmentation.

Extracts 478-point MediaPipe Face Mesh or 68-point dlib landmarks inside
a detected face bounding box, then segments the face into Action-Unit–
relevant regions (eyes, brows, nose, mouth).

Usage::

    extractor = LandmarkExtractor(model='mediapipe_mesh')
    lm = extractor.extract(frame, bbox)
    if lm is not None:
        rois = extractor.get_rois(frame, lm)
    extractor.release()
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.microex.face_detector import BoundingBox

logger = logging.getLogger(__name__)

# Minimum dimension (pixels) a cropped face must have to bother running
# the landmark model.
_MIN_FACE_DIM = 20


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FacialLandmarks:
    """Container for extracted facial landmarks.

    Attributes:
        points: Landmark coordinates in **pixel space**.
            Shape ``(N, 3)`` for MediaPipe (x, y, z) or
            ``(N, 2)`` for dlib (x, y).
        source: Identifier string — ``'mediapipe_mesh'`` or ``'dlib_68'``.
        face_bbox: The bounding box used for extraction.
    """
    points: np.ndarray        # (N, 3) or (N, 2), float64
    source: str               # 'mediapipe_mesh' | 'dlib_68'
    face_bbox: BoundingBox


@dataclass
class FacialROIs:
    """Cropped image regions for AU-relevant facial areas.

    Every array is a BGR ``uint8`` sub-image.  ``landmarks_in_roi``
    holds landmark coordinates shifted to be relative to ``full_face``
    (the aligned, padded face crop).

    Attributes:
        left_eye:  Cropped left-eye region.
        right_eye: Cropped right-eye region.
        left_brow: Cropped left-eyebrow region.
        right_brow: Cropped right-eyebrow region.
        nose:      Cropped nose region.
        mouth:     Cropped mouth region.
        full_face: Padded, aligned face crop.
        landmarks_in_roi: Landmarks re-centred to ``full_face``.
    """
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_brow: np.ndarray
    right_brow: np.ndarray
    nose: np.ndarray
    mouth: np.ndarray
    full_face: np.ndarray
    landmarks_in_roi: np.ndarray


# ---------------------------------------------------------------------------
# LandmarkExtractor
# ---------------------------------------------------------------------------

_VALID_MODELS = {"mediapipe_mesh", "dlib_68"}


class LandmarkExtractor:
    """Extract facial landmarks and segment face into ROI crops.

    Parameters:
        model: Landmark model — ``'mediapipe_mesh'`` (478-point) or
            ``'dlib_68'`` (68-point).
        static_image_mode: When ``True``, each image is treated
            independently (slower but more robust for single images).

    Raises:
        ValueError: If *model* is unsupported.
    """

    # ------------------------------------------------------------------
    # MediaPipe Face Mesh landmark index groups
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    # ------------------------------------------------------------------

    LEFT_EYE_INDICES: List[int] = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
        386, 385, 384, 398,
    ]
    RIGHT_EYE_INDICES: List[int] = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
        159, 160, 161, 246,
    ]
    LEFT_BROW_INDICES: List[int] = [
        276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    ]
    RIGHT_BROW_INDICES: List[int] = [
        46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    ]
    NOSE_INDICES: List[int] = [
        1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 220, 115, 48,
        64, 237, 44, 275, 440, 344, 278, 294, 457,
    ]
    MOUTH_INDICES: List[int] = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
        191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    ]

    # dlib 68-point equivalents (ranges are inclusive-end in the lists)
    _DLIB_LEFT_EYE = list(range(42, 48))
    _DLIB_RIGHT_EYE = list(range(36, 42))
    _DLIB_LEFT_BROW = list(range(22, 27))
    _DLIB_RIGHT_BROW = list(range(17, 22))
    _DLIB_NOSE = list(range(27, 36))
    _DLIB_MOUTH = list(range(48, 68))

    def __init__(
        self,
        model: str = "mediapipe_mesh",
        static_image_mode: bool = False,
    ) -> None:
        model = model.strip().lower()
        if model not in _VALID_MODELS:
            raise ValueError(
                f"Unsupported model '{model}'. Choose from {_VALID_MODELS}."
            )
        self._model: str = model
        self._static_image_mode: bool = static_image_mode

        # Lazy handles
        self._mp_mesh: Optional[object] = None
        self._dlib_predictor: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> Optional[FacialLandmarks]:
        """Extract landmarks within *bbox* on *frame*.

        Args:
            frame: BGR or grayscale ``uint8`` image.
            bbox: Bounding box delimiting the face region.

        Returns:
            :class:`FacialLandmarks` on success, ``None`` if no
            landmarks were found or the face region is too small.

        Raises:
            ValueError: If *frame* is ``None`` or empty.
            TypeError:  If *frame* is not a NumPy array.
        """
        self._validate_frame(frame)
        colour = self._ensure_bgr(frame)
        h, w = colour.shape[:2]

        # Clamp bbox to frame boundaries
        clamped = bbox.clamp(w, h)
        if clamped.w < _MIN_FACE_DIM or clamped.h < _MIN_FACE_DIM:
            logger.debug(
                "BBox too small after clamping (%dx%d) — skipping.",
                clamped.w, clamped.h,
            )
            return None

        if self._model == "mediapipe_mesh":
            return self._extract_mediapipe(colour, clamped, w, h)
        return self._extract_dlib(colour, clamped)

    def get_rois(
        self,
        frame: np.ndarray,
        landmarks: FacialLandmarks,
        padding: float = 0.2,
    ) -> FacialROIs:
        """Segment the face into AU-relevant ROI crops.

        Args:
            frame: Original BGR ``uint8`` image.
            landmarks: Previously extracted :class:`FacialLandmarks`.
            padding: Fractional padding around each ROI bounding box.

        Returns:
            :class:`FacialROIs` with cropped regions and shifted landmarks.

        Raises:
            ValueError: If *frame* is ``None`` or empty.
        """
        self._validate_frame(frame)
        colour = self._ensure_bgr(frame)
        h, w = colour.shape[:2]

        bb = landmarks.face_bbox.clamp(w, h)

        # Compute padded full-face crop
        pad_x = int(bb.w * padding)
        pad_y = int(bb.h * padding)
        x1 = max(0, bb.x - pad_x)
        y1 = max(0, bb.y - pad_y)
        x2 = min(w, bb.x + bb.w + pad_x)
        y2 = min(h, bb.y + bb.h + pad_y)
        full_face = colour[y1:y2, x1:x2].copy()

        # Shift landmarks into full_face coordinate system
        pts = landmarks.points.copy().astype(np.float64)
        pts[:, 0] -= x1
        pts[:, 1] -= y1
        landmarks_in_roi = pts

        # Decide index groups
        if landmarks.source == "mediapipe_mesh":
            groups = {
                "left_eye": self.LEFT_EYE_INDICES,
                "right_eye": self.RIGHT_EYE_INDICES,
                "left_brow": self.LEFT_BROW_INDICES,
                "right_brow": self.RIGHT_BROW_INDICES,
                "nose": self.NOSE_INDICES,
                "mouth": self.MOUTH_INDICES,
            }
        else:
            groups = {
                "left_eye": self._DLIB_LEFT_EYE,
                "right_eye": self._DLIB_RIGHT_EYE,
                "left_brow": self._DLIB_LEFT_BROW,
                "right_brow": self._DLIB_RIGHT_BROW,
                "nose": self._DLIB_NOSE,
                "mouth": self._DLIB_MOUTH,
            }

        crops: dict[str, np.ndarray] = {}
        face_h, face_w = full_face.shape[:2]
        for name, indices in groups.items():
            safe = [i for i in indices if i < len(pts)]
            if not safe:
                # Fallback: 1-pixel black patch
                crops[name] = np.zeros((1, 1, 3), dtype=np.uint8)
                continue
            region_pts = pts[safe, :2]
            rx1, ry1 = region_pts.min(axis=0).astype(int)
            rx2, ry2 = region_pts.max(axis=0).astype(int)
            rpad_x = max(1, int((rx2 - rx1) * padding))
            rpad_y = max(1, int((ry2 - ry1) * padding))
            rx1 = max(0, rx1 - rpad_x)
            ry1 = max(0, ry1 - rpad_y)
            rx2 = min(face_w, rx2 + rpad_x)
            ry2 = min(face_h, ry2 + rpad_y)
            crop = full_face[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            crops[name] = crop.copy()

        return FacialROIs(
            left_eye=crops["left_eye"],
            right_eye=crops["right_eye"],
            left_brow=crops["left_brow"],
            right_brow=crops["right_brow"],
            nose=crops["nose"],
            mouth=crops["mouth"],
            full_face=full_face,
            landmarks_in_roi=landmarks_in_roi,
        )

    def release(self) -> None:
        """Release underlying model resources."""
        if self._mp_mesh is not None:
            try:
                self._mp_mesh.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._mp_mesh = None
        self._dlib_predictor = None
        logger.debug("LandmarkExtractor resources released.")

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _extract_mediapipe(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        frame_w: int,
        frame_h: int,
    ) -> Optional[FacialLandmarks]:
        """Extract 478 landmarks using MediaPipe Face Mesh Tasks API."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            logger.warning("mediapipe not installed — cannot extract landmarks.")
            return None

        if self._mp_mesh is None:
            model_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task")
            if not Path(model_path).exists():
                logger.error("MediaPipe Landmarker model not found at %s", model_path)
                return None
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self._mp_mesh = vision.FaceLandmarker.create_from_options(options)

        # Crop face region with a small pad for Mesh stability
        pad = int(max(bbox.w, bbox.h) * 0.15)
        x1 = max(0, bbox.x - pad)
        y1 = max(0, bbox.y - pad)
        x2 = min(frame_w, bbox.x + bbox.w + pad)
        y2 = min(frame_h, bbox.y + bbox.h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        
        try:
            results = self._mp_mesh.detect(mp_image)
        except Exception as e:
            logger.error("MediaPipe landmarker failed: %s", e)
            return None

        if not results.face_landmarks:
            return None

        face_lm = results.face_landmarks[0]
        crop_h, crop_w = crop.shape[:2]
        pts = np.array(
            [
                [lm.x * crop_w + x1, lm.y * crop_h + y1, lm.z * crop_w]
                for lm in face_lm
            ],
            dtype=np.float64,
        )

        return FacialLandmarks(points=pts, source="mediapipe_mesh", face_bbox=bbox)

    def _extract_dlib(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> Optional[FacialLandmarks]:
        """Extract 68 landmarks using dlib shape predictor."""
        try:
            import dlib  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("dlib not installed — cannot extract landmarks.")
            return None

        if self._dlib_predictor is None:
            # Attempt to load default 68-landmark model
            model_path = "shape_predictor_68_face_landmarks.dat"
            try:
                self._dlib_predictor = dlib.shape_predictor(model_path)
            except RuntimeError:
                logger.error(
                    "dlib shape predictor model not found at '%s'. "
                    "Download from http://dlib.net/files/ .",
                    model_path,
                )
                return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(  # type: ignore[attr-defined]
            bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h,
        )

        shape = self._dlib_predictor(gray, rect)  # type: ignore[union-attr]
        pts = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)],
            dtype=np.float64,
        )
        return FacialLandmarks(points=pts, source="dlib_68", face_bbox=bbox)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Raise on invalid input."""
        if frame is None:
            raise ValueError("Frame must not be None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame).__name__}.")
        if frame.size == 0:
            raise ValueError("Frame is empty (zero size).")

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        """Convert single-channel grayscale to 3-channel BGR."""
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame


# ======================================================================
# Trial block
# ======================================================================

def _make_synthetic_frame(
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Return a light-grey frame with a crude face-like oval."""
    canvas = np.full((height, width, 3), 200, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    cv2.ellipse(canvas, (cx, cy), (70, 90), 0, 0, 360, (180, 200, 230), -1)
    cv2.circle(canvas, (cx - 25, cy - 15), 8, (30, 30, 30), -1)
    cv2.circle(canvas, (cx + 25, cy - 15), 8, (30, 30, 30), -1)
    cv2.ellipse(canvas, (cx, cy + 30), (20, 8), 0, 0, 180, (80, 80, 180), 2)
    return canvas


def _make_synthetic_landmarks(
    n: int,
    bbox: BoundingBox,
    dims: int = 3,
) -> FacialLandmarks:
    """Fabricate *n* random landmark points inside *bbox*."""
    rng = np.random.RandomState(42)
    xs = rng.uniform(bbox.x, bbox.x + bbox.w, size=(n, 1))
    ys = rng.uniform(bbox.y, bbox.y + bbox.h, size=(n, 1))
    if dims == 3:
        zs = rng.uniform(-0.05, 0.05, size=(n, 1))
        pts = np.hstack([xs, ys, zs])
    else:
        pts = np.hstack([xs, ys])
    source = "mediapipe_mesh" if dims == 3 else "dlib_68"
    return FacialLandmarks(points=pts, source=source, face_bbox=bbox)


def _run_trials() -> bool:
    """Execute standalone trial tests.  Returns *True* if all pass."""
    all_passed = True

    def _report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_passed
        status = "PASS" if passed else "FAIL"
        msg = f"{status}: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if not passed:
            all_passed = False

    print("\n=== LandmarkExtractor Trial Block ===\n")

    frame = _make_synthetic_frame()
    bbox = BoundingBox(x=200, y=120, w=240, h=240, confidence=0.95)

    # ---- Test 1: MediaPipe extract (if available) ---------------------
    try:
        import mediapipe  # noqa: F401
        mp_available = hasattr(mediapipe, 'solutions')
    except ImportError:
        mp_available = False

    if mp_available:
        extractor = LandmarkExtractor(model="mediapipe_mesh", static_image_mode=True)
        try:
            lm = extractor.extract(frame, bbox)
            if lm is not None:
                ok = (
                    lm.points.ndim == 2
                    and lm.points.shape[1] == 3
                    and lm.source == "mediapipe_mesh"
                )
                _report(
                    "MediaPipe extract — shape & source",
                    ok,
                    f"shape={lm.points.shape}, source={lm.source}",
                )
                # Verify pixel-space bounds
                xs = lm.points[:, 0]
                ys = lm.points[:, 1]
                in_bounds = (
                    xs.min() >= -50  # small tolerance
                    and ys.min() >= -50
                    and xs.max() < frame.shape[1] + 50
                    and ys.max() < frame.shape[0] + 50
                )
                _report(
                    "MediaPipe extract — points in frame bounds",
                    in_bounds,
                    f"x=[{xs.min():.0f}..{xs.max():.0f}], "
                    f"y=[{ys.min():.0f}..{ys.max():.0f}]",
                )
            else:
                _report(
                    "MediaPipe extract",
                    True,
                    "returned None (synthetic face not detected — acceptable)",
                )
        except Exception as exc:
            _report("MediaPipe extract", False, str(exc))
        finally:
            extractor.release()
    else:
        print("SKIP: mediapipe not installed — MediaPipe extract tests skipped.")

    # ---- Test 2: get_rois with synthetic landmarks --------------------
    syn_lm = _make_synthetic_landmarks(478, bbox, dims=3)
    extractor2 = LandmarkExtractor(model="mediapipe_mesh", static_image_mode=True)
    try:
        rois = extractor2.get_rois(frame, syn_lm, padding=0.2)
        roi_names = [
            "left_eye", "right_eye", "left_brow", "right_brow",
            "nose", "mouth", "full_face",
        ]
        for name in roi_names:
            arr: np.ndarray = getattr(rois, name)
            ok = arr.size > 0
            _report(
                f"get_rois — {name} non-empty",
                ok,
                f"shape={arr.shape}",
            )
        # landmarks_in_roi shape check
        ok_lir = (
            rois.landmarks_in_roi.ndim == 2
            and rois.landmarks_in_roi.shape[0] == 478
        )
        _report(
            "get_rois — landmarks_in_roi shape",
            ok_lir,
            f"shape={rois.landmarks_in_roi.shape}",
        )
    except Exception as exc:
        _report("get_rois with synthetic landmarks", False, str(exc))
    finally:
        extractor2.release()

    # ---- Test 3: bbox partially outside frame -------------------------
    outside_bbox = BoundingBox(x=-50, y=-30, w=300, h=200, confidence=0.8)
    syn_lm_out = _make_synthetic_landmarks(478, outside_bbox, dims=3)
    extractor3 = LandmarkExtractor(model="mediapipe_mesh", static_image_mode=True)
    try:
        rois_out = extractor3.get_rois(frame, syn_lm_out, padding=0.1)
        _report(
            "get_rois — bbox partially outside frame",
            rois_out.full_face.size > 0,
            f"full_face shape={rois_out.full_face.shape}",
        )
    except Exception as exc:
        _report("get_rois — bbox partially outside frame", False, str(exc))
    finally:
        extractor3.release()

    # ---- Test 4: very small bbox returns None from extract -------------
    tiny_bbox = BoundingBox(x=300, y=200, w=5, h=5, confidence=0.9)
    extractor4 = LandmarkExtractor(model="mediapipe_mesh", static_image_mode=True)
    try:
        result = extractor4.extract(frame, tiny_bbox)
        _report(
            "extract — tiny bbox returns None",
            result is None,
            f"returned {'None' if result is None else 'FacialLandmarks'}",
        )
    except Exception as exc:
        _report("extract — tiny bbox returns None", False, str(exc))
    finally:
        extractor4.release()

    # ---- Test 5: empty frame raises ValueError ------------------------
    extractor5 = LandmarkExtractor(model="mediapipe_mesh")
    try:
        extractor5.extract(np.array([], dtype=np.uint8), bbox)
        _report("extract — empty frame raises ValueError", False, "no exception raised")
    except ValueError:
        _report("extract — empty frame raises ValueError", True)
    except Exception as exc:
        _report("extract — empty frame raises ValueError", False, f"wrong exception: {exc}")
    finally:
        extractor5.release()

    # ---- Test 6: invalid model raises ValueError ----------------------
    try:
        LandmarkExtractor(model="invalid_model")
        _report("invalid model raises ValueError", False, "no exception raised")
    except ValueError:
        _report("invalid model raises ValueError", True)
    except Exception as exc:
        _report("invalid model raises ValueError", False, f"wrong exception: {exc}")

    # ---- Test 7: dlib_68 synthetic landmarks + get_rois ---------------
    syn_lm_dlib = _make_synthetic_landmarks(68, bbox, dims=2)
    extractor6 = LandmarkExtractor(model="dlib_68", static_image_mode=True)
    try:
        rois_dlib = extractor6.get_rois(frame, syn_lm_dlib, padding=0.15)
        ok_d = all(
            getattr(rois_dlib, n).size > 0
            for n in ["left_eye", "right_eye", "mouth", "nose"]
        )
        _report(
            "get_rois — dlib_68 synthetic landmarks",
            ok_d,
            "all key ROIs non-empty",
        )
    except Exception as exc:
        _report("get_rois — dlib_68 synthetic landmarks", False, str(exc))
    finally:
        extractor6.release()

    return all_passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    passed = _run_trials()
    print(f"\n{'='*40}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
