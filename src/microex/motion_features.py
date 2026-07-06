"""
motion_features.py — Optical flow and temporal feature extraction module.

Computes dense optical flow (Farneback) or sparse optical flow (Lucas-Kanade)
between consecutive grayscale frames and extracts magnitude/angle histograms
per facial Region of Interest (ROI) for downstream micro-expression analysis.

CPU-only by default; no GPU dependencies required.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FlowFeatures:
    """Container for optical-flow features extracted from a single ROI.

    Attributes:
        magnitude_histogram: Histogram of flow magnitudes (shape: n_magnitude_bins,).
        angle_histogram: Histogram of flow angles in degrees (shape: n_angle_bins,).
        mean_magnitude: Average flow magnitude across the ROI.
        max_magnitude: Peak flow magnitude across the ROI.
        flow_field: Raw (H, W, 2) flow field if retained, else ``None``.
        region_label: Human-readable label for the source ROI.
    """

    magnitude_histogram: np.ndarray
    angle_histogram: np.ndarray
    mean_magnitude: float
    max_magnitude: float
    flow_field: Optional[np.ndarray]
    region_label: str


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


class MotionFeatureExtractor:
    """Compute optical flow between grayscale frames and extract features.

    Parameters:
        method: ``'farneback'`` (dense) or ``'lk'`` (Lucas-Kanade sparse on grid).
        n_magnitude_bins: Number of histogram bins for flow magnitude.
        n_angle_bins: Number of histogram bins for flow angle.
        buffer_size: Maximum number of grayscale frames held in ring buffer.
        retain_flow: If ``True``, raw flow field is stored in ``FlowFeatures``.
        lk_grid_step: Grid spacing (px) when using Lucas-Kanade method.
    """

    _VALID_METHODS = frozenset({"farneback", "lk"})

    # Good defaults for Farneback
    _FARNEBACK_PARAMS: Dict[str, object] = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    # LK params
    _LK_WIN_SIZE: Tuple[int, int] = (15, 15)
    _LK_MAX_LEVEL: int = 3
    _LK_CRITERIA: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )

    def __init__(
        self,
        method: str = "farneback",
        n_magnitude_bins: int = 8,
        n_angle_bins: int = 8,
        buffer_size: int = 16,
        retain_flow: bool = False,
        lk_grid_step: int = 10,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown flow method '{method}'. Choose from {self._VALID_METHODS}."
            )
        if n_magnitude_bins < 1 or n_angle_bins < 1:
            raise ValueError("Histogram bin counts must be >= 1.")
        if buffer_size < 2:
            raise ValueError("Buffer size must be >= 2 to compute flow.")

        self.method: str = method
        self.n_magnitude_bins: int = n_magnitude_bins
        self.n_angle_bins: int = n_angle_bins
        self.buffer_size: int = buffer_size
        self.retain_flow: bool = retain_flow
        self.lk_grid_step: int = lk_grid_step

        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Ring-buffer management
    # ------------------------------------------------------------------

    def update(self, frame_gray: np.ndarray) -> None:
        """Add a grayscale frame to the internal ring buffer.

        Args:
            frame_gray: 2-D ``uint8`` grayscale image.

        Raises:
            TypeError: If *frame_gray* is not an ndarray.
            ValueError: If *frame_gray* is not 2-D or not ``uint8``.
        """
        if not isinstance(frame_gray, np.ndarray):
            raise TypeError("frame_gray must be a numpy ndarray.")
        if frame_gray.ndim != 2:
            raise ValueError(
                f"Expected 2-D grayscale frame, got shape {frame_gray.shape}."
            )
        if frame_gray.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 dtype, got {frame_gray.dtype}."
            )
        self._buffer.append(frame_gray.copy())

    def reset(self) -> None:
        """Clear the ring buffer and all internal state."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Optical-flow computation
    # ------------------------------------------------------------------

    def compute_flow(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> np.ndarray:
        """Compute optical flow between two grayscale frames.

        Args:
            prev_gray: Previous grayscale frame (H, W), ``uint8``.
            curr_gray: Current grayscale frame (H, W), ``uint8``.

        Returns:
            Dense flow field of shape ``(H, W, 2)`` with ``float32`` dtype.
            Channel 0 is horizontal (dx), channel 1 is vertical (dy).

        Raises:
            ValueError: If frames differ in shape or are not 2-D.
        """
        self._validate_frame_pair(prev_gray, curr_gray)

        # Ensure grayscale uint8
        prev_g = self._ensure_gray(prev_gray)
        curr_g = self._ensure_gray(curr_gray)

        if self.method == "farneback":
            return self._compute_farneback(prev_g, curr_g)
        else:
            return self._compute_lk(prev_g, curr_g)

    def _compute_farneback(
        self, prev_g: np.ndarray, curr_g: np.ndarray
    ) -> np.ndarray:
        """Dense optical flow via Farneback algorithm."""
        flow: np.ndarray = cv2.calcOpticalFlowFarneback(
            prev_g,
            curr_g,
            None,  # type: ignore[arg-type]
            pyr_scale=self._FARNEBACK_PARAMS["pyr_scale"],
            levels=self._FARNEBACK_PARAMS["levels"],
            winsize=self._FARNEBACK_PARAMS["winsize"],
            iterations=self._FARNEBACK_PARAMS["iterations"],
            poly_n=self._FARNEBACK_PARAMS["poly_n"],
            poly_sigma=self._FARNEBACK_PARAMS["poly_sigma"],
            flags=self._FARNEBACK_PARAMS["flags"],
        )
        return flow

    def _compute_lk(
        self, prev_g: np.ndarray, curr_g: np.ndarray
    ) -> np.ndarray:
        """Sparse optical flow on a regular grid via Lucas-Kanade.

        The result is interpolated into a dense (H, W, 2) field so the
        downstream API stays uniform regardless of method.
        """
        h, w = prev_g.shape[:2]
        step = self.lk_grid_step

        # Build grid of points
        ys = np.arange(0, h, step, dtype=np.float32)
        xs = np.arange(0, w, step, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        pts = np.stack(
            [grid_x.ravel(), grid_y.ravel()], axis=-1
        ).reshape(-1, 1, 2)

        if pts.shape[0] == 0:
            return np.zeros((h, w, 2), dtype=np.float32)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_g,
            curr_g,
            pts,
            None,
            winSize=self._LK_WIN_SIZE,
            maxLevel=self._LK_MAX_LEVEL,
            criteria=self._LK_CRITERIA,
        )

        # Build dense flow via nearest-neighbour scatter
        flow = np.zeros((h, w, 2), dtype=np.float32)
        if next_pts is None or status is None:
            return flow

        status_flat = status.ravel().astype(bool)
        good_prev = pts.reshape(-1, 2)[status_flat]
        good_next = next_pts.reshape(-1, 2)[status_flat]
        displacement = good_next - good_prev

        for (px, py), (dx, dy) in zip(
            good_prev.astype(int), displacement
        ):
            y_lo = max(py - step // 2, 0)
            y_hi = min(py + step // 2 + 1, h)
            x_lo = max(px - step // 2, 0)
            x_hi = min(px + step // 2 + 1, w)
            flow[y_lo:y_hi, x_lo:x_hi, 0] = dx
            flow[y_lo:y_hi, x_lo:x_hi, 1] = dy

        return flow

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        flow: np.ndarray,
        region_label: str = "full_face",
    ) -> FlowFeatures:
        """Extract magnitude and angle histograms from a flow field.

        Args:
            flow: Dense flow of shape ``(H, W, 2)``.
            region_label: Label for the source ROI.

        Returns:
            A populated :class:`FlowFeatures` instance.

        Raises:
            ValueError: If *flow* does not have the expected shape.
        """
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(
                f"Flow must be (H, W, 2), got shape {flow.shape}."
            )

        mag, ang_rad = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        ang_deg = np.degrees(ang_rad)

        mean_mag = float(np.mean(mag))
        max_mag = float(np.max(mag)) if mag.size > 0 else 0.0

        # Magnitude histogram — auto range [0, max_mag] or [0, 1] if flat
        mag_upper = max(max_mag, 1e-6)
        mag_hist, _ = np.histogram(
            mag.ravel(),
            bins=self.n_magnitude_bins,
            range=(0.0, mag_upper),
        )
        mag_hist = mag_hist.astype(np.float32)
        mag_total = mag_hist.sum()
        if mag_total > 0:
            mag_hist /= mag_total

        # Angle histogram [0, 360)
        ang_hist, _ = np.histogram(
            ang_deg.ravel(),
            bins=self.n_angle_bins,
            range=(0.0, 360.0),
        )
        ang_hist = ang_hist.astype(np.float32)
        ang_total = ang_hist.sum()
        if ang_total > 0:
            ang_hist /= ang_total

        return FlowFeatures(
            magnitude_histogram=mag_hist,
            angle_histogram=ang_hist,
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            flow_field=flow.copy() if self.retain_flow else None,
            region_label=region_label,
        )

    def compute_roi_features(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi_regions: Dict[str, Tuple[int, int, int, int]],
    ) -> Dict[str, FlowFeatures]:
        """Compute per-ROI optical flow features.

        Args:
            prev_gray: Previous grayscale frame.
            curr_gray: Current grayscale frame.
            roi_regions: Mapping of region label → ``(x, y, w, h)`` rectangle.

        Returns:
            Dict mapping each region label to its :class:`FlowFeatures`.
        """
        self._validate_frame_pair(prev_gray, curr_gray)
        prev_g = self._ensure_gray(prev_gray)
        curr_g = self._ensure_gray(curr_gray)

        full_flow = self.compute_flow(prev_g, curr_g)
        fh, fw = full_flow.shape[:2]

        results: Dict[str, FlowFeatures] = {}

        for label, (rx, ry, rw, rh) in roi_regions.items():
            # Clamp to frame bounds
            x1 = max(int(rx), 0)
            y1 = max(int(ry), 0)
            x2 = min(int(rx + rw), fw)
            y2 = min(int(ry + rh), fh)

            if (x2 - x1) < 3 or (y2 - y1) < 3:
                warnings.warn(
                    f"ROI '{label}' is too small ({x2 - x1}x{y2 - y1}), skipping.",
                    stacklevel=2,
                )
                continue

            roi_flow = full_flow[y1:y2, x1:x2]
            results[label] = self.extract_features(roi_flow, region_label=label)

        return results

    def get_feature_vector(
        self, flow_features: Dict[str, FlowFeatures]
    ) -> np.ndarray:
        """Concatenate all ROI features into a single 1-D vector.

        For each ROI (sorted by label for deterministic order) the vector
        contains: ``[mag_hist..., ang_hist..., mean_mag, max_mag]``.

        Args:
            flow_features: Dict of region label → :class:`FlowFeatures`.

        Returns:
            1-D ``float32`` feature vector.
        """
        parts: List[np.ndarray] = []
        for label in sorted(flow_features.keys()):
            ff = flow_features[label]
            parts.append(ff.magnitude_histogram)
            parts.append(ff.angle_histogram)
            parts.append(
                np.array([ff.mean_magnitude, ff.max_magnitude], dtype=np.float32)
            )

        if not parts:
            return np.array([], dtype=np.float32)

        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gray(frame: np.ndarray) -> np.ndarray:
        """Convert to single-channel uint8 if necessary."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return frame

    @staticmethod
    def _validate_frame_pair(
        prev: np.ndarray, curr: np.ndarray
    ) -> None:
        """Raise on mismatched or invalid frame pairs."""
        if not isinstance(prev, np.ndarray) or not isinstance(curr, np.ndarray):
            raise TypeError("Both frames must be numpy ndarrays.")
        if prev.ndim < 2 or curr.ndim < 2:
            raise ValueError("Frames must be at least 2-D.")
        if prev.shape[:2] != curr.shape[:2]:
            raise ValueError(
                f"Frame sizes differ: {prev.shape[:2]} vs {curr.shape[:2]}."
            )


# ======================================================================
# Trial block
# ======================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0

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
    print("MotionFeatureExtractor — Trial Block")
    print("=" * 60)

    try:
        # ----------------------------------------------------------
        # 1. Synthetic frames: second is first shifted 5px right
        # ----------------------------------------------------------
        H, W = 240, 320
        rng = np.random.RandomState(42)
        frame1 = rng.randint(0, 256, (H, W), dtype=np.uint8)
        frame2 = np.zeros_like(frame1)
        frame2[:, 5:] = frame1[:, : W - 5]  # shift right by 5

        extractor = MotionFeatureExtractor(
            method="farneback",
            n_magnitude_bins=8,
            n_angle_bins=8,
            buffer_size=16,
            retain_flow=True,
        )

        # ----------------------------------------------------------
        # 2. Compute flow
        # ----------------------------------------------------------
        flow = extractor.compute_flow(frame1, frame2)
        _report(
            "Flow computation (Farneback)",
            flow is not None and isinstance(flow, np.ndarray),
            f"type={type(flow).__name__}",
        )

        # ----------------------------------------------------------
        # 3. Flow field shape
        # ----------------------------------------------------------
        expected_shape = (H, W, 2)
        _report(
            "Flow shape is (240, 320, 2)",
            flow.shape == expected_shape,
            f"got {flow.shape}",
        )

        # ----------------------------------------------------------
        # 4. Mean flow magnitude > 0
        # ----------------------------------------------------------
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(np.mean(mag))
        _report(
            "Mean flow magnitude > 0 (motion present)",
            mean_mag > 0.0,
            f"mean_mag={mean_mag:.4f}",
        )

        # ----------------------------------------------------------
        # 5. Extract features — histogram shapes
        # ----------------------------------------------------------
        feats = extractor.extract_features(flow, region_label="full_face")
        ok_mag = feats.magnitude_histogram.shape == (8,)
        ok_ang = feats.angle_histogram.shape == (8,)
        _report(
            "Magnitude histogram shape (8,)",
            ok_mag,
            f"got {feats.magnitude_histogram.shape}",
        )
        _report(
            "Angle histogram shape (8,)",
            ok_ang,
            f"got {feats.angle_histogram.shape}",
        )

        # ----------------------------------------------------------
        # 6. get_feature_vector → 1-D array
        # ----------------------------------------------------------
        roi_feats = {
            "left_eye": feats,
            "right_eye": feats,
        }
        vec = extractor.get_feature_vector(roi_feats)
        _report(
            "get_feature_vector returns 1-D array",
            vec.ndim == 1 and vec.dtype == np.float32,
            f"ndim={vec.ndim}, dtype={vec.dtype}, len={len(vec)}",
        )
        expected_len = 2 * (8 + 8 + 2)  # 2 ROIs × (mag_bins + ang_bins + 2 scalars)
        _report(
            f"Feature vector length == {expected_len}",
            len(vec) == expected_len,
            f"got {len(vec)}",
        )

        # ----------------------------------------------------------
        # 7. Same frame twice → near-zero flow
        # ----------------------------------------------------------
        flow_zero = extractor.compute_flow(frame1, frame1)
        mag_zero = np.sqrt(flow_zero[..., 0] ** 2 + flow_zero[..., 1] ** 2)
        mean_zero = float(np.mean(mag_zero))
        _report(
            "Same frame → near-zero mean magnitude",
            mean_zero < 0.01,
            f"mean_mag={mean_zero:.6f}",
        )

        # ----------------------------------------------------------
        # 8. Edge case: mismatched frame sizes
        # ----------------------------------------------------------
        small = np.zeros((100, 100), dtype=np.uint8)
        try:
            extractor.compute_flow(frame1, small)
            _report("Mismatched frames raise ValueError", False, "no exception raised")
        except ValueError:
            _report("Mismatched frames raise ValueError", True)

        # ----------------------------------------------------------
        # 9. Ring buffer update
        # ----------------------------------------------------------
        extractor.reset()
        extractor.update(frame1)
        extractor.update(frame2)
        _report(
            "Ring buffer holds 2 frames after 2 updates",
            len(extractor._buffer) == 2,
            f"len={len(extractor._buffer)}",
        )
        extractor.reset()
        _report(
            "Ring buffer empty after reset",
            len(extractor._buffer) == 0,
        )

        # ----------------------------------------------------------
        # 10. ROI features with small ROI warning
        # ----------------------------------------------------------
        rois = {
            "forehead": (10, 10, 100, 60),
            "tiny": (0, 0, 2, 2),  # too small — should be skipped
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            roi_result = extractor.compute_roi_features(frame1, frame2, rois)
            got_warning = any("too small" in str(ww.message) for ww in w)
        _report(
            "Small ROI triggers warning and is skipped",
            "tiny" not in roi_result and got_warning,
            f"keys={list(roi_result.keys())}, warning={got_warning}",
        )

    except Exception as exc:
        print(f"\n  [FATAL] Unexpected exception: {exc}")
        traceback.print_exc()
        failed += 1

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
