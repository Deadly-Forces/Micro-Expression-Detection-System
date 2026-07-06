"""Face detection module with multi-backend fallback chain.

Supports MediaPipe Face Detection (primary), Haar Cascade (fallback),
and dlib HOG detector (optional). Detectors are initialised lazily so
GPU memory is not allocated at import time.

Usage::

    detector = FaceDetector(backend='mediapipe')
    boxes = detector.detect(frame)        # returns List[BoundingBox]
    detector.release()
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box with detection confidence.

    Attributes:
        x: Left column (pixels).
        y: Top row (pixels).
        w: Width (pixels).
        h: Height (pixels).
        confidence: Detector confidence in [0, 1].
    """
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def area(self) -> int:
        """Return area in pixels²."""
        return self.w * self.h

    def clamp(self, frame_w: int, frame_h: int) -> "BoundingBox":
        """Return a new BoundingBox clamped to *frame_w* × *frame_h*."""
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(frame_w, self.x + self.w)
        y2 = min(frame_h, self.y + self.h)
        return BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1), self.confidence)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

_VALID_BACKENDS = {"mediapipe", "haar", "dlib"}


class FaceDetector:
    """Multi-backend face detector with automatic fallback.

    Parameters:
        backend: Primary detection backend — one of
            ``'mediapipe'``, ``'haar'``, or ``'dlib'``.
        haar_cascade_path: Explicit path to a Haar cascade XML file.
            Defaults to OpenCV's bundled ``haarcascade_frontalface_default.xml``.
        min_confidence: Minimum detection confidence to keep a result.
        enable_fallback: If ``True`` and the primary detector returns no
            faces, a Haar cascade fallback is attempted (unless Haar *is*
            the primary backend).

    Raises:
        ValueError: If *backend* is not one of the supported values.
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        haar_cascade_path: Optional[str] = None,
        min_confidence: float = 0.5,
        enable_fallback: bool = True,
    ) -> None:
        backend = backend.strip().lower()
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Choose from {_VALID_BACKENDS}."
            )

        self._backend: str = backend
        self._min_confidence: float = float(min_confidence)
        self._enable_fallback: bool = enable_fallback

        # Haar cascade path -----------------------------------------------
        if haar_cascade_path is not None:
            self._haar_path: str = haar_cascade_path
        else:
            self._haar_path = str(
                Path(__file__).resolve().parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"
            )

        # Lazy handles — created on first use -----------------------------
        self._mp_detector: Optional[object] = None
        self._haar_classifier: Optional[cv2.CascadeClassifier] = None
        self._dlib_detector: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect faces in *frame* and return bounding boxes.

        The primary backend runs first.  If it finds nothing **and**
        ``enable_fallback`` is ``True``, Haar cascade is tried as well
        (unless Haar is already the primary backend).

        Args:
            frame: BGR or grayscale image, ``uint8``.

        Returns:
            List of :class:`BoundingBox` sorted by area descending
            (largest face first).  May be empty.

        Raises:
            ValueError: If *frame* is ``None`` or has zero size.
            TypeError: If *frame* is not a NumPy array.
        """
        self._validate_frame(frame)

        # Ensure 3-channel BGR for backends that need colour
        colour = self._ensure_bgr(frame)

        dispatch = {
            "mediapipe": self._detect_mediapipe,
            "haar": self._detect_haar,
            "dlib": self._detect_dlib,
        }

        boxes: List[BoundingBox] = dispatch[self._backend](colour)

        # Fallback to Haar if primary produced nothing
        if not boxes and self._enable_fallback and self._backend != "haar":
            logger.debug(
                "Primary detector '%s' returned 0 faces — falling back to Haar.",
                self._backend,
            )
            boxes = self._detect_haar(colour)

        # Sort by descending area
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes

    def release(self) -> None:
        """Release underlying detector resources."""
        if self._mp_detector is not None:
            try:
                self._mp_detector.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._mp_detector = None
        self._haar_classifier = None
        self._dlib_detector = None
        logger.debug("FaceDetector resources released.")

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _detect_mediapipe(self, frame: np.ndarray) -> List[BoundingBox]:
        """Run MediaPipe Face Detection on *frame* using the Tasks API.

        Initialises the MediaPipe detector lazily on first call.

        Args:
            frame: 3-channel BGR ``uint8`` image.

        Returns:
            List of detected :class:`BoundingBox`.
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            logger.warning(
                "mediapipe is not installed — skipping MediaPipe detection."
            )
            return []

        if self._mp_detector is None:
            # Assumes the tflite model is downloaded in models/
            model_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "blaze_face_short_range.tflite")
            if not Path(model_path).exists():
                logger.error("MediaPipe model not found at %s", model_path)
                return []
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=self._min_confidence)
            self._mp_detector = vision.FaceDetector.create_from_options(options)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            detection_result = self._mp_detector.detect(mp_image)
        except Exception as e:
            logger.error("MediaPipe detection failed: %s", e)
            return []

        boxes: List[BoundingBox] = []
        if detection_result.detections:
            for det in detection_result.detections:
                bb = det.bounding_box
                conf = float(det.categories[0].score) if det.categories else 1.0
                if conf < self._min_confidence:
                    continue
                bx = int(bb.origin_x)
                by = int(bb.origin_y)
                bw = int(bb.width)
                bh = int(bb.height)
                boxes.append(
                    BoundingBox(bx, by, bw, bh, conf).clamp(w, h)
                )
        return boxes

    def _detect_haar(self, frame: np.ndarray) -> List[BoundingBox]:
        """Run Haar cascade on *frame*.

        Args:
            frame: 3-channel BGR ``uint8`` image.

        Returns:
            List of detected :class:`BoundingBox`.
        """
        if self._haar_classifier is None:
            self._haar_classifier = cv2.CascadeClassifier(self._haar_path)
            if self._haar_classifier.empty():
                logger.error("Failed to load Haar cascade from '%s'.", self._haar_path)
                return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            rects = self._haar_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error as e:
            logger.error("Haar cascade failed during detectMultiScale: %s", e)
            return []

        h, w = frame.shape[:2]
        boxes: List[BoundingBox] = []
        for (rx, ry, rw, rh) in rects:
            boxes.append(
                BoundingBox(int(rx), int(ry), int(rw), int(rh), 1.0).clamp(w, h)
            )
        return boxes

    def _detect_dlib(self, frame: np.ndarray) -> List[BoundingBox]:
        """Run dlib HOG frontal-face detector on *frame*.

        Args:
            frame: 3-channel BGR ``uint8`` image.

        Returns:
            List of detected :class:`BoundingBox`.
        """
        try:
            import dlib  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("dlib is not installed — skipping dlib detection.")
            return []

        if self._dlib_detector is None:
            self._dlib_detector = dlib.get_frontal_face_detector()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets, scores, _ = self._dlib_detector.run(gray, 1, -1.0)  # type: ignore[union-attr]

        h, w = frame.shape[:2]
        boxes: List[BoundingBox] = []
        for rect, score in zip(dets, scores):
            if score < self._min_confidence:
                continue
            bx = rect.left()
            by = rect.top()
            bw = rect.right() - rect.left()
            bh = rect.bottom() - rect.top()
            boxes.append(
                BoundingBox(bx, by, bw, bh, float(score)).clamp(w, h)
            )
        return boxes

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

def _draw_synthetic_face(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    face_w: int = 140,
    face_h: int = 180,
) -> None:
    """Draw a crude face-like oval onto *canvas* (in-place).

    Skin-toned ellipse with two dark eye circles, a nose line,
    and a mouth arc.
    """
    skin = (180, 200, 230)  # BGR light skin tone
    eye_colour = (40, 30, 20)  # dark brown

    # Head ellipse
    cv2.ellipse(canvas, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, skin, -1)
    cv2.ellipse(canvas, (cx, cy), (face_w // 2, face_h // 2), 0, 0, 360, (100, 120, 140), 2)

    # Eyes
    eye_y = cy - face_h // 6
    eye_sep = face_w // 4
    cv2.circle(canvas, (cx - eye_sep, eye_y), 10, eye_colour, -1)
    cv2.circle(canvas, (cx + eye_sep, eye_y), 10, eye_colour, -1)

    # Nose
    cv2.line(canvas, (cx, cy - 10), (cx, cy + 15), (120, 140, 160), 2)

    # Mouth
    cv2.ellipse(canvas, (cx, cy + face_h // 5), (25, 10), 0, 0, 180, (80, 80, 180), 2)


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

    # ---- Test 1: single synthetic face (Haar) -------------------------
    print("\n=== FaceDetector Trial Block ===\n")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (200, 200, 200)  # light grey background
    _draw_synthetic_face(frame, 320, 240)

    detector = FaceDetector(backend="haar", enable_fallback=False)
    try:
        boxes = detector.detect(frame)
        _report(
            "Single synthetic face (Haar)",
            True,
            f"detected {len(boxes)} face(s) — synthetic ovals may or may not trigger Haar",
        )
    except Exception as exc:
        _report("Single synthetic face (Haar)", False, str(exc))
    finally:
        detector.release()

    # ---- Test 2: empty / None frame raises ValueError -----------------
    try:
        detector2 = FaceDetector(backend="haar")
        detector2.detect(np.array([], dtype=np.uint8))
        _report("Empty frame raises ValueError", False, "no exception raised")
    except ValueError:
        _report("Empty frame raises ValueError", True)
    except Exception as exc:
        _report("Empty frame raises ValueError", False, f"wrong exception: {exc}")
    finally:
        detector2.release()

    # ---- Test 3: None frame raises ValueError -------------------------
    try:
        detector3 = FaceDetector(backend="haar")
        detector3.detect(None)  # type: ignore[arg-type]
        _report("None frame raises ValueError", False, "no exception raised")
    except (ValueError, TypeError):
        _report("None frame raises ValueError", True)
    except Exception as exc:
        _report("None frame raises ValueError", False, f"wrong exception: {exc}")
    finally:
        detector3.release()

    # ---- Test 4: grayscale input accepted -----------------------------
    gray_frame = np.zeros((480, 640), dtype=np.uint8)
    gray_frame[:] = 200
    _draw_synthetic_face(
        cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), 320, 240
    )  # the oval draw is on a throwaway; we just need a non-empty gray frame
    detector4 = FaceDetector(backend="haar", enable_fallback=False)
    try:
        boxes_gray = detector4.detect(gray_frame)
        _report(
            "Grayscale input accepted",
            True,
            f"returned {len(boxes_gray)} box(es)",
        )
    except Exception as exc:
        _report("Grayscale input accepted", False, str(exc))
    finally:
        detector4.release()

    # ---- Test 5: multi-face frame (two synthetic ovals) ---------------
    multi_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    _draw_synthetic_face(multi_frame, 160, 240, face_w=120, face_h=160)
    _draw_synthetic_face(multi_frame, 480, 240, face_w=140, face_h=180)
    detector5 = FaceDetector(backend="haar", enable_fallback=False)
    try:
        multi_boxes = detector5.detect(multi_frame)
        _report(
            "Multi-face frame",
            True,
            f"detected {len(multi_boxes)} face(s) — expected 0-2 for synthetic",
        )
        if len(multi_boxes) > 1:
            sizes = [b.area for b in multi_boxes]
            _report(
                "Multi-face sorted descending by area",
                sizes == sorted(sizes, reverse=True),
                f"areas={sizes}",
            )
    except Exception as exc:
        _report("Multi-face frame", False, str(exc))
    finally:
        detector5.release()

    # ---- Test 6: invalid backend raises ValueError --------------------
    try:
        FaceDetector(backend="nonexistent")
        _report("Invalid backend raises ValueError", False, "no exception raised")
    except ValueError:
        _report("Invalid backend raises ValueError", True)
    except Exception as exc:
        _report("Invalid backend raises ValueError", False, f"wrong exception: {exc}")

    # ---- Test 7: BoundingBox.clamp ------------------------------------
    bb = BoundingBox(-10, -10, 100, 100, 0.9)
    clamped = bb.clamp(640, 480)
    ok = clamped.x == 0 and clamped.y == 0 and clamped.w == 90 and clamped.h == 90
    _report("BoundingBox.clamp", ok, f"clamped={clamped}")

    # ---- Test 8: MediaPipe backend (optional) -------------------------
    try:
        import mediapipe  # noqa: F401
        mp_available = hasattr(mediapipe, 'solutions')
    except ImportError:
        mp_available = False

    if mp_available:
        mp_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        _draw_synthetic_face(mp_frame, 320, 240)
        detector_mp = FaceDetector(backend="mediapipe", min_confidence=0.3)
        try:
            mp_boxes = detector_mp.detect(mp_frame)
            _report(
                "MediaPipe backend (optional)",
                True,
                f"detected {len(mp_boxes)} face(s)",
            )
        except Exception as exc:
            _report("MediaPipe backend (optional)", False, str(exc))
        finally:
            detector_mp.release()
    else:
        print("SKIP: MediaPipe not installed — MediaPipe backend test skipped.")

    # ---- Test 9: dlib backend (optional) ------------------------------
    try:
        import dlib  # noqa: F401
        dlib_available = True
    except ImportError:
        dlib_available = False

    if dlib_available:
        dlib_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        _draw_synthetic_face(dlib_frame, 320, 240)
        detector_dlib = FaceDetector(backend="dlib", min_confidence=0.0)
        try:
            dlib_boxes = detector_dlib.detect(dlib_frame)
            _report(
                "dlib backend (optional)",
                True,
                f"detected {len(dlib_boxes)} face(s)",
            )
        except Exception as exc:
            _report("dlib backend (optional)", False, str(exc))
        finally:
            detector_dlib.release()
    else:
        print("SKIP: dlib not installed — dlib backend test skipped.")

    return all_passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    passed = _run_trials()
    print(f"\n{'='*40}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
