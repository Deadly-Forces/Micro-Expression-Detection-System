"""
config.py — Centralized configuration for the Micro-Expression Detection System.

Defines a ``SystemConfig`` dataclass that holds every tuneable parameter the
pipeline needs, together with ``load_config`` / ``save_config`` helpers that
round-trip through JSON files.

Author : Micro-Expression Detection Team
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Literal, Optional

import cv2

logger = logging.getLogger(__name__)


def _default_haar_cascade_path() -> str:
    """Return the path to the built-in OpenCV Haar frontal-face cascade."""
    data_dir = cv2.data.haarcascades  # type: ignore[attr-defined]
    return os.path.join(data_dir, "haarcascade_frontalface_default.xml")


def _default_emotion_labels() -> List[str]:
    return [
        "happiness",
        "sadness",
        "surprise",
        "fear",
        "disgust",
        "anger",
        "contempt",
    ]


@dataclass
class SystemConfig:
    """Complete runtime configuration for the micro-expression pipeline.

    All fields carry sensible defaults so the system can start with zero
    external config files.  GPU acceleration is *opt-in* (``enable_gpu``).

    Attributes
    ----------
    camera_index : int
        Index passed to ``cv2.VideoCapture`` for webcam input.
    video_path : Optional[str]
        Path to a single video file (used when ``input_mode == 'video'``).
    dataset_path : Optional[str]
        Root directory of a dataset (e.g. CASME II layout).
    input_mode : Literal['webcam', 'video', 'dataset']
        Which frame source to activate.
    frame_width : int
        Desired capture width in pixels.
    frame_height : int
        Desired capture height in pixels.
    target_fps : int
        Target frames-per-second for processing / display loop.
    face_detection_backend : Literal['mediapipe', 'haar', 'dlib']
        Which face detector to use.
    landmark_model : Literal['mediapipe_mesh', 'dlib_68']
        Facial landmark model variant.
    flow_method : Literal['farneback', 'lucas_kanade']
        Optical-flow algorithm for motion estimation.
    classifier_type : Literal['svm', 'lstm', 'cnn']
        Back-end classifier for expression recognition.
    model_path : str
        File path to a serialised classifier model.
    haar_cascade_path : str
        Path to the Haar cascade XML (defaults to OpenCV's built-in).
    confidence_threshold : float
        Minimum confidence for reporting a detected expression.
    apex_onset_threshold : float
        Motion magnitude threshold that marks *onset* of a micro-expression.
    apex_offset_threshold : float
        Motion magnitude threshold that marks *offset*.
    sliding_window_size : int
        Number of frames in the temporal sliding window.
    output_dir : str
        Directory for saving results, logs, and visualisations.
    log_level : str
        Python logging level name (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    enable_gpu : bool
        Whether to attempt GPU-accelerated paths (CUDA / OpenCL).
    require_consent : bool
        If True the pipeline must obtain consent before recording faces.
    emotion_labels : List[str]
        Canonical ordered list of emotion class names.
    """

    # ── Input ────────────────────────────────────────────────────────
    camera_index: int = 0
    video_path: Optional[str] = None
    dataset_path: Optional[str] = None
    input_mode: Literal["webcam", "video", "dataset"] = "webcam"

    # ── Frame geometry ───────────────────────────────────────────────
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    # ── Detection / Landmarking ──────────────────────────────────────
    face_detection_backend: Literal["mediapipe", "haar", "dlib"] = "mediapipe"
    landmark_model: Literal["mediapipe_mesh", "dlib_68"] = "mediapipe_mesh"

    # ── Optical flow ─────────────────────────────────────────────────
    flow_method: Literal["farneback", "lucas_kanade"] = "farneback"

    # ── Classification ───────────────────────────────────────────────
    classifier_type: Literal["svm", "lstm", "cnn"] = "svm"
    model_path: str = "models/classifier.pkl"

    # ── Haar cascade ─────────────────────────────────────────────────
    haar_cascade_path: str = dataclasses.field(
        default_factory=_default_haar_cascade_path
    )

    # ── Thresholds ───────────────────────────────────────────────────
    confidence_threshold: float = 0.5
    apex_onset_threshold: float = 0.3
    apex_offset_threshold: float = 0.15
    sliding_window_size: int = 16

    # ── Output / logging ─────────────────────────────────────────────
    output_dir: str = "output"
    log_level: str = "INFO"

    # ── Hardware / ethics ────────────────────────────────────────────
    enable_gpu: bool = False
    require_consent: bool = True

    # ── Labels ───────────────────────────────────────────────────────
    emotion_labels: List[str] = dataclasses.field(
        default_factory=_default_emotion_labels
    )

    # ── Validation ───────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Validate field values immediately after construction."""
        if self.frame_width <= 0:
            raise ValueError(f"frame_width must be > 0, got {self.frame_width}")
        if self.frame_height <= 0:
            raise ValueError(f"frame_height must be > 0, got {self.frame_height}")
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if not 0.0 <= self.apex_onset_threshold <= 1.0:
            raise ValueError(
                f"apex_onset_threshold must be in [0, 1], got {self.apex_onset_threshold}"
            )
        if not 0.0 <= self.apex_offset_threshold <= 1.0:
            raise ValueError(
                f"apex_offset_threshold must be in [0, 1], got {self.apex_offset_threshold}"
            )
        if self.sliding_window_size < 1:
            raise ValueError(
                f"sliding_window_size must be >= 1, got {self.sliding_window_size}"
            )
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"log_level must be one of {valid_log_levels}, got '{self.log_level}'"
            )
        if self.input_mode == "video" and not self.video_path:
            logger.warning(
                "input_mode is 'video' but video_path is None — "
                "set video_path before starting the pipeline."
            )
        if self.input_mode == "dataset" and not self.dataset_path:
            logger.warning(
                "input_mode is 'dataset' but dataset_path is None — "
                "set dataset_path before starting the pipeline."
            )


# ═══════════════════════════════════════════════════════════════════════
# Persistence helpers
# ═══════════════════════════════════════════════════════════════════════


def load_config(path: Optional[str] = None) -> SystemConfig:
    """Load a ``SystemConfig`` from a JSON file, or return defaults.

    Parameters
    ----------
    path : Optional[str]
        Path to a JSON configuration file.  If *None* or the file does not
        exist, a default ``SystemConfig`` is returned.

    Returns
    -------
    SystemConfig
        The fully-validated configuration object.

    Raises
    ------
    json.JSONDecodeError
        If the file exists but contains malformed JSON.
    ValueError
        If field validation inside ``SystemConfig.__post_init__`` fails.
    """
    if path is None:
        logger.info("No config path supplied — using defaults.")
        return SystemConfig()

    resolved = Path(path)
    if not resolved.is_file():
        logger.warning("Config file '%s' not found — using defaults.", path)
        return SystemConfig()

    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Malformed JSON in '%s': %s", path, exc)
        raise

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at top level, got {type(data).__name__}"
        )

    # Only pass keys that SystemConfig actually declares so that stale /
    # unknown keys in older config files do not blow up the constructor.
    valid_keys = {f.name for f in dataclasses.fields(SystemConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    unknown = set(data.keys()) - valid_keys
    if unknown:
        logger.warning("Ignoring unknown config keys: %s", unknown)

    return SystemConfig(**filtered)


def save_config(config: SystemConfig, path: str) -> None:
    """Serialise a ``SystemConfig`` to a JSON file.

    Parameters
    ----------
    config : SystemConfig
        The configuration to persist.
    path : str
        Destination file path.  Parent directories are created if needed.

    Raises
    ------
    TypeError
        If *config* is not a ``SystemConfig`` instance.
    OSError
        On filesystem errors (permissions, disk full, …).
    """
    if not isinstance(config, SystemConfig):
        raise TypeError(
            f"Expected SystemConfig, got {type(config).__name__}"
        )

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2, ensure_ascii=False)

    logger.info("Configuration saved to '%s'.", dest)


# ═══════════════════════════════════════════════════════════════════════
# Trial block
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    all_passed = True

    # --- Test 1: default construction --------------------------------
    try:
        cfg = SystemConfig()
        print("TEST 1 — Default construction: PASS")
    except Exception as exc:
        print(f"TEST 1 — Default construction: FAIL ({exc})")
        all_passed = False

    # --- Test 2: round-trip save → load → compare --------------------
    try:
        cfg = SystemConfig(
            camera_index=2,
            input_mode="video",
            video_path="test.mp4",
            frame_width=1280,
            frame_height=720,
            target_fps=60,
            face_detection_backend="haar",
            landmark_model="dlib_68",
            flow_method="lucas_kanade",
            classifier_type="lstm",
            model_path="models/my_model.h5",
            confidence_threshold=0.75,
            apex_onset_threshold=0.25,
            apex_offset_threshold=0.10,
            sliding_window_size=32,
            output_dir="results",
            log_level="DEBUG",
            enable_gpu=True,
            require_consent=False,
            emotion_labels=["happy", "sad"],
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        save_config(cfg, tmp_path)
        loaded = load_config(tmp_path)

        mismatches: list[str] = []
        for f in dataclasses.fields(SystemConfig):
            orig = getattr(cfg, f.name)
            reloaded = getattr(loaded, f.name)
            if orig != reloaded:
                mismatches.append(
                    f"  {f.name}: expected {orig!r}, got {reloaded!r}"
                )

        if mismatches:
            print("TEST 2 — Round-trip save/load: FAIL")
            print("\n".join(mismatches))
            all_passed = False
        else:
            print("TEST 2 — Round-trip save/load: PASS")

        os.unlink(tmp_path)
    except Exception as exc:
        print(f"TEST 2 — Round-trip save/load: FAIL ({exc})")
        all_passed = False

    # --- Test 3: load_config with missing file returns defaults -------
    try:
        default_cfg = load_config("/nonexistent/path/config.json")
        assert default_cfg == SystemConfig(), "Should equal defaults"
        print("TEST 3 — Missing file returns defaults: PASS")
    except Exception as exc:
        print(f"TEST 3 — Missing file returns defaults: FAIL ({exc})")
        all_passed = False

    # --- Test 4: validation catches bad values -----------------------
    try:
        bad_passed = False
        try:
            SystemConfig(frame_width=-1)
        except ValueError:
            bad_passed = True
        assert bad_passed, "Should have raised ValueError for frame_width=-1"

        bad_passed = False
        try:
            SystemConfig(confidence_threshold=2.0)
        except ValueError:
            bad_passed = True
        assert bad_passed, "Should have raised ValueError for threshold=2.0"

        bad_passed = False
        try:
            SystemConfig(log_level="BOGUS")
        except ValueError:
            bad_passed = True
        assert bad_passed, "Should have raised ValueError for log_level='BOGUS'"

        print("TEST 4 — Validation rejects bad values: PASS")
    except Exception as exc:
        print(f"TEST 4 — Validation rejects bad values: FAIL ({exc})")
        all_passed = False

    # --- Summary -----------------------------------------------------
    print()
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    sys.exit(0 if all_passed else 1)
