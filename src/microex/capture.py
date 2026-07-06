"""
capture.py — Frame acquisition module for the Micro-Expression Detection System.

Provides abstract and concrete frame sources that yield ``FrameResult`` objects
from webcams, video files, and dataset directory trees.  All sources implement
the iterator and context-manager protocols for clean, Pythonic usage::

    with VideoSource("clip.mp4") as src:
        for result in src:
            cv2.imshow("frame", result.frame)

CPU-only by default; no GPU code-paths are required in this module.

Author : Micro-Expression Detection Team
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data envelope
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FrameResult:
    """Immutable container returned by every frame source on each iteration.

    Attributes
    ----------
    frame : np.ndarray
        BGR image of shape ``(H, W, 3)`` and dtype ``uint8``.
    frame_index : int
        Zero-based ordinal of this frame within its source.
    timestamp_ms : float
        Presentation timestamp in milliseconds (``0.0`` when unavailable).
    source_id : str
        Human-readable identifier of the source (camera index, file path, …).
    """

    frame: np.ndarray
    frame_index: int
    timestamp_ms: float
    source_id: str


# ═══════════════════════════════════════════════════════════════════════
# Abstract base class
# ═══════════════════════════════════════════════════════════════════════


class FrameSource(ABC):
    """Abstract iterator / context-manager for frame acquisition.

    Subclasses **must** implement ``__next__`` (and optionally override
    ``release``, ``get_fps``, ``get_frame_count``).
    """

    # ── Iterator protocol ────────────────────────────────────────────

    def __iter__(self) -> Iterator[FrameResult]:
        """Return *self* as the iterator."""
        return self

    @abstractmethod
    def __next__(self) -> FrameResult:
        """Yield the next ``FrameResult`` or raise ``StopIteration``."""
        ...

    # ── Resource management ──────────────────────────────────────────

    def release(self) -> None:
        """Release any underlying OS / hardware resources.

        The default implementation is a no-op; subclasses should override
        as needed.
        """

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    # ── Metadata ─────────────────────────────────────────────────────

    @abstractmethod
    def get_fps(self) -> float:
        """Return the nominal frames-per-second of this source."""
        ...

    @abstractmethod
    def get_frame_count(self) -> int:
        """Return total frame count (``-1`` if unknown, e.g. live camera)."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Concrete sources
# ═══════════════════════════════════════════════════════════════════════


class WebcamSource(FrameSource):
    """Live camera capture via ``cv2.VideoCapture``.

    Parameters
    ----------
    camera_index : int
        OS camera device index (typically 0 for the built-in webcam).
    width : int
        Desired capture width in pixels.
    height : int
        Desired capture height in pixels.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.
    ValueError
        If *camera_index* is negative.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        if camera_index < 0:
            raise ValueError(f"camera_index must be >= 0, got {camera_index}")
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got ({width}, {height})")

        self._camera_index: int = camera_index
        self._width: int = width
        self._height: int = height
        self._frame_index: int = 0
        self._released: bool = False

        self._cap: cv2.VideoCapture = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that the device is connected and not in use by another process."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        logger.info(
            "WebcamSource opened camera %d (%dx%d).",
            camera_index,
            width,
            height,
        )

    # ── Iterator ─────────────────────────────────────────────────────

    def __next__(self) -> FrameResult:
        if self._released:
            raise StopIteration("WebcamSource has been released.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise StopIteration("Failed to read frame from webcam.")
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        result = FrameResult(
            frame=frame,
            frame_index=self._frame_index,
            timestamp_ms=ts,
            source_id=f"webcam:{self._camera_index}",
        )
        self._frame_index += 1
        return result

    # ── Resource management ──────────────────────────────────────────

    def release(self) -> None:
        """Release the camera device."""
        if not self._released and self._cap.isOpened():
            self._cap.release()
            logger.info("WebcamSource released camera %d.", self._camera_index)
        self._released = True

    # ── Metadata ─────────────────────────────────────────────────────

    def get_fps(self) -> float:
        """Return the camera's reported FPS (may be 0 for some drivers)."""
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def get_frame_count(self) -> int:
        """Live cameras have no finite frame count; returns ``-1``."""
        return -1


class VideoSource(FrameSource):
    """Frame source backed by a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If OpenCV cannot decode the video.
    """

    def __init__(self, video_path: str) -> None:
        if not video_path:
            raise ValueError("video_path must be a non-empty string.")
        p = Path(video_path)
        if not p.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._video_path: str = str(p.resolve())
        self._frame_index: int = 0
        self._released: bool = False

        self._cap: cv2.VideoCapture = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video file '{self._video_path}'. "
                "The codec may not be supported or the file may be corrupt."
            )

        self._fps: float = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "VideoSource opened '%s' — %d frames @ %.1f FPS.",
            self._video_path,
            self._total_frames,
            self._fps,
        )

    # ── Iterator ─────────────────────────────────────────────────────

    def __next__(self) -> FrameResult:
        if self._released:
            raise StopIteration("VideoSource has been released.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise StopIteration
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        result = FrameResult(
            frame=frame,
            frame_index=self._frame_index,
            timestamp_ms=ts,
            source_id=self._video_path,
        )
        self._frame_index += 1
        return result

    # ── Resource management ──────────────────────────────────────────

    def release(self) -> None:
        if not self._released and self._cap.isOpened():
            self._cap.release()
            logger.info("VideoSource released '%s'.", self._video_path)
        self._released = True

    # ── Metadata ─────────────────────────────────────────────────────

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._total_frames


class DatasetSource(FrameSource):
    """Iterate over a directory tree of videos **and** images.

    Handles nested structures such as CASME II's
    ``<root>/<subject>/<emotion>/clip.avi`` as well as flat directories of
    images.

    Parameters
    ----------
    dataset_dir : str
        Root directory to walk.
    extensions : Tuple[str, ...]
        File extensions to include (case-insensitive).

    Raises
    ------
    FileNotFoundError
        If *dataset_dir* does not exist.
    ValueError
        If no matching files are found.
    """

    _VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv"})
    _IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

    def __init__(
        self,
        dataset_dir: str,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".jpg", ".png"),
    ) -> None:
        if not dataset_dir:
            raise ValueError("dataset_dir must be a non-empty string.")
        root = Path(dataset_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        self._dataset_dir: str = str(root.resolve())
        self._extensions: Tuple[str, ...] = tuple(e.lower() for e in extensions)

        # Discover all matching files, sorted for reproducibility.
        self._files: List[str] = sorted(
            str(p)
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self._extensions
        )
        if not self._files:
            raise ValueError(
                f"No files with extensions {self._extensions} found under '{dataset_dir}'."
            )

        logger.info(
            "DatasetSource found %d files under '%s'.",
            len(self._files),
            self._dataset_dir,
        )

        # Internal iteration state
        self._file_idx: int = 0
        self._frame_index: int = 0
        self._current_cap: Optional[cv2.VideoCapture] = None
        self._released: bool = False

    # ── Iterator ─────────────────────────────────────────────────────

    def __next__(self) -> FrameResult:
        if self._released:
            raise StopIteration("DatasetSource has been released.")

        while self._file_idx < len(self._files):
            filepath = self._files[self._file_idx]
            ext = Path(filepath).suffix.lower()

            # --- Image file -------------------------------------------
            if ext in self._IMAGE_EXTS:
                self._file_idx += 1
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("Skipping unreadable image: %s", filepath)
                    continue
                result = FrameResult(
                    frame=img,
                    frame_index=self._frame_index,
                    timestamp_ms=0.0,
                    source_id=filepath,
                )
                self._frame_index += 1
                return result

            # --- Video file -------------------------------------------
            if ext in self._VIDEO_EXTS:
                # Open video if not already open
                if self._current_cap is None:
                    cap = cv2.VideoCapture(filepath)
                    if not cap.isOpened():
                        logger.warning(
                            "Skipping unopenable video: %s", filepath
                        )
                        self._file_idx += 1
                        continue
                    self._current_cap = cap

                ret, frame = self._current_cap.read()
                if not ret or frame is None:
                    # Finished this video — move to next file
                    self._current_cap.release()
                    self._current_cap = None
                    self._file_idx += 1
                    continue

                ts = self._current_cap.get(cv2.CAP_PROP_POS_MSEC)
                result = FrameResult(
                    frame=frame,
                    frame_index=self._frame_index,
                    timestamp_ms=ts,
                    source_id=filepath,
                )
                self._frame_index += 1
                return result

            # --- Unknown extension (shouldn't happen) -----------------
            self._file_idx += 1

        raise StopIteration

    # ── Resource management ──────────────────────────────────────────

    def release(self) -> None:
        if self._current_cap is not None and self._current_cap.isOpened():
            self._current_cap.release()
            self._current_cap = None
        self._released = True
        logger.info("DatasetSource released.")

    # ── Metadata ─────────────────────────────────────────────────────

    def get_fps(self) -> float:
        """Dataset sources have no single FPS; returns ``0.0``."""
        return 0.0

    def get_frame_count(self) -> int:
        """Return the number of discovered *files* (not total video frames)."""
        return len(self._files)


# ═══════════════════════════════════════════════════════════════════════
# Trial block
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    all_passed: bool = True
    tmp_paths: list[str] = []

    WIDTH, HEIGHT, N_FRAMES = 640, 480, 10

    # -----------------------------------------------------------------
    # Helper: create a synthetic test video
    # -----------------------------------------------------------------
    def _make_test_video(path: str, width: int, height: int, n_frames: int) -> None:
        """Write *n_frames* BGR frames with a drawn circle to *path*."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(path, fourcc, 25.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open '{path}'.")
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw a circle to simulate a rough "face" blob
            cx, cy = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(frame, (cx, cy), radius, (0, 200, 255), -1)
            # Add a small per-frame variation so frames aren't identical
            cv2.putText(
                frame,
                f"F{i:03d}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
        writer.release()

    # -----------------------------------------------------------------
    # Prepare temp video
    # -----------------------------------------------------------------
    try:
        tmp_video_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_video_fd)
        tmp_paths.append(tmp_video_path)
        _make_test_video(tmp_video_path, WIDTH, HEIGHT, N_FRAMES)
    except Exception as exc:
        print(f"SETUP — Create test video: FAIL ({exc})")
        sys.exit(1)

    # =================================================================
    # TEST 1: VideoSource — read all frames
    # =================================================================
    try:
        frames_read: int = 0
        with VideoSource(tmp_video_path) as src:
            fps = src.get_fps()
            total = src.get_frame_count()
            for result in src:
                assert isinstance(result, FrameResult), "Expected FrameResult"
                assert result.frame.shape == (HEIGHT, WIDTH, 3), (
                    f"Shape mismatch: {result.frame.shape}"
                )
                assert result.frame.dtype == np.uint8, "dtype should be uint8"
                frames_read += 1

        assert frames_read == N_FRAMES, (
            f"Expected {N_FRAMES} frames, got {frames_read}"
        )
        assert fps > 0, f"FPS should be > 0, got {fps}"
        assert total == N_FRAMES, f"Total frames mismatch: {total} vs {N_FRAMES}"
        print(f"TEST 1 — VideoSource (read {frames_read} frames, fps={fps:.1f}): PASS")
    except Exception as exc:
        print(f"TEST 1 — VideoSource: FAIL ({exc})")
        all_passed = False

    # =================================================================
    # TEST 2: VideoSource — missing file raises FileNotFoundError
    # =================================================================
    try:
        raised = False
        try:
            VideoSource("/nonexistent/video.mp4")
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError for missing file"
        print("TEST 2 — VideoSource missing file error: PASS")
    except Exception as exc:
        print(f"TEST 2 — VideoSource missing file error: FAIL ({exc})")
        all_passed = False

    # =================================================================
    # TEST 3: DatasetSource — iterate over directory with test video
    # =================================================================
    try:
        tmp_dir = tempfile.mkdtemp()
        tmp_paths.append(tmp_dir)

        # Create nested structure: <root>/sub01/happy/clip.mp4
        nested = os.path.join(tmp_dir, "sub01", "happy")
        os.makedirs(nested, exist_ok=True)
        nested_video = os.path.join(nested, "clip.mp4")
        _make_test_video(nested_video, WIDTH, HEIGHT, N_FRAMES)

        ds_frames: int = 0
        with DatasetSource(tmp_dir, extensions=(".mp4",)) as ds:
            file_count = ds.get_frame_count()
            for result in ds:
                assert isinstance(result, FrameResult)
                ds_frames += 1

        assert ds_frames == N_FRAMES, f"Expected {N_FRAMES}, got {ds_frames}"
        assert file_count == 1, f"Expected 1 file, got {file_count}"
        print(
            f"TEST 3 — DatasetSource (files={file_count}, frames={ds_frames}): PASS"
        )
    except Exception as exc:
        print(f"TEST 3 — DatasetSource: FAIL ({exc})")
        all_passed = False

    # =================================================================
    # TEST 4: DatasetSource — empty dir raises ValueError
    # =================================================================
    try:
        empty_dir = tempfile.mkdtemp()
        tmp_paths.append(empty_dir)
        raised = False
        try:
            DatasetSource(empty_dir)
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for empty dataset dir"
        print("TEST 4 — DatasetSource empty dir error: PASS")
    except Exception as exc:
        print(f"TEST 4 — DatasetSource empty dir error: FAIL ({exc})")
        all_passed = False

    # =================================================================
    # TEST 5: DatasetSource — image iteration
    # =================================================================
    try:
        img_dir = tempfile.mkdtemp()
        tmp_paths.append(img_dir)
        # Write 3 synthetic PNG images
        for i in range(3):
            img = np.full((100, 100, 3), fill_value=(i * 80) % 256, dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, f"img_{i:02d}.png"), img)

        img_count: int = 0
        with DatasetSource(img_dir, extensions=(".png",)) as ds:
            for result in ds:
                assert result.frame.shape == (100, 100, 3)
                img_count += 1

        assert img_count == 3, f"Expected 3 images, got {img_count}"
        print(f"TEST 5 — DatasetSource images ({img_count} images): PASS")
    except Exception as exc:
        print(f"TEST 5 — DatasetSource images: FAIL ({exc})")
        all_passed = False

    # =================================================================
    # SKIP: WebcamSource
    # =================================================================
    print("SKIP: WebcamSource (requires physical camera)")

    # =================================================================
    # Cleanup
    # =================================================================
    import shutil

    for p in tmp_paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.unlink(p)
        except OSError:
            pass

    # =================================================================
    # Summary
    # =================================================================
    print()
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    sys.exit(0 if all_passed else 1)
