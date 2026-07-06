"""
apex_spotter.py — Onset-apex-offset detection for micro-expressions.

Implements a lightweight state-machine that consumes per-frame
:class:`FlowFeatures` dictionaries (from :mod:`motion_features`) and emits
:class:`MicroExpressionEvent` objects whenever a complete onset → apex → offset
cycle is detected within the micro-expression temporal window.

CPU-only; no GPU dependencies.
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Avoid hard import — only needed for type hints at runtime
try:
    from microex.motion_features import FlowFeatures
except ImportError:
    # Allow standalone execution where the package layout isn't installed
    FlowFeatures = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MicroExpressionEvent:
    """Describes a single detected micro-expression event.

    Attributes:
        onset_frame: Global frame index where the onset was first detected.
        apex_frame: Global frame index of peak magnitude.
        offset_frame: Global frame index where magnitude fell below offset
            threshold.
        duration_frames: ``offset_frame - onset_frame + 1``.
        duration_ms: Duration converted to milliseconds using the capture FPS.
        peak_magnitude: Maximum aggregate magnitude at the apex.
        region_activations: Per-ROI magnitude at the apex frame.
        feature_vector: Optional concatenated feature vector at the apex.
    """

    onset_frame: int
    apex_frame: int
    offset_frame: int
    duration_frames: int
    duration_ms: float
    peak_magnitude: float
    region_activations: Dict[str, float]
    feature_vector: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

_STATE_IDLE: str = "idle"
_STATE_TRACKING_ONSET: str = "tracking_onset"
_STATE_TRACKING_APEX: str = "tracking_apex"


# ---------------------------------------------------------------------------
# Core spotter
# ---------------------------------------------------------------------------


class ApexSpotter:
    """Finite state-machine that detects micro-expression temporal events.

    The spotter maintains a running baseline of flow magnitude and transitions
    through three states:

    1. **IDLE** — waiting for magnitude to exceed *onset_threshold* above
       baseline.
    2. **TRACKING_ONSET / TRACKING_APEX** — recording the rising/falling
       phase and tracking peak magnitude.
    3. Back to **IDLE** once magnitude drops below *offset_threshold* above
       baseline **or** the event exceeds *max_duration_frames* (discarded
       as a macro-expression).

    Parameters:
        onset_threshold: Magnitude above baseline to trigger onset.
        offset_threshold: Magnitude above baseline that signals offset.
        min_duration_frames: Minimum event length (frames) to be accepted.
        max_duration_frames: Maximum event length; longer is discarded.
        window_size: Number of history frames retained for visualisation.
        fps: Capture frame-rate, used to convert duration to milliseconds.
        baseline_frames: Number of initial frames used to estimate baseline.
    """

    def __init__(
        self,
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.15,
        min_duration_frames: int = 2,
        max_duration_frames: int = 25,
        window_size: int = 16,
        fps: float = 30.0,
        baseline_frames: int = 5,
    ) -> None:
        if onset_threshold <= 0:
            raise ValueError("onset_threshold must be > 0.")
        if offset_threshold < 0:
            raise ValueError("offset_threshold must be >= 0.")
        if offset_threshold >= onset_threshold:
            raise ValueError(
                "offset_threshold must be strictly less than onset_threshold."
            )
        if min_duration_frames < 1:
            raise ValueError("min_duration_frames must be >= 1.")
        if max_duration_frames < min_duration_frames:
            raise ValueError(
                "max_duration_frames must be >= min_duration_frames."
            )
        if fps <= 0:
            raise ValueError("fps must be > 0.")
        if baseline_frames < 1:
            raise ValueError("baseline_frames must be >= 1.")

        self.onset_threshold: float = onset_threshold
        self.offset_threshold: float = offset_threshold
        self.min_duration_frames: int = min_duration_frames
        self.max_duration_frames: int = max_duration_frames
        self.window_size: int = window_size
        self.fps: float = fps
        self.baseline_frames: int = baseline_frames

        # Internal state
        self._state: str = _STATE_IDLE
        self._magnitude_history: deque[float] = deque(maxlen=window_size)
        self._baseline: float = 0.0
        self._baseline_accumulator: List[float] = []
        self._baseline_ready: bool = False

        # Tracking variables (active during non-IDLE states)
        self._onset_frame: int = -1
        self._peak_magnitude: float = 0.0
        self._apex_frame: int = -1
        self._apex_region_activations: Dict[str, float] = {}
        self._apex_feature_vector: Optional[np.ndarray] = None
        self._tracking_frames: int = 0

        # Stores per-frame region activations during tracking for apex lookup
        self._tracking_history: List[
            Tuple[int, float, Dict[str, float], Optional[np.ndarray]]
        ] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_index: int,
        flow_features: Dict[str, "FlowFeatures"],
    ) -> Optional[MicroExpressionEvent]:
        """Process a single frame's flow features.

        Args:
            frame_index: Global (0-based) frame counter.
            flow_features: ROI label → :class:`FlowFeatures` mapping for
                the current frame.

        Returns:
            A :class:`MicroExpressionEvent` when a complete micro-expression
            cycle is detected, otherwise ``None``.
        """
        if not isinstance(flow_features, dict):
            raise TypeError("flow_features must be a dict of FlowFeatures.")

        # Aggregate magnitude across all ROIs (mean of mean-magnitudes)
        region_mags: Dict[str, float] = {}
        for label, ff in flow_features.items():
            region_mags[label] = float(
                getattr(ff, "mean_magnitude", 0.0)
            )
        agg_magnitude: float = (
            float(np.mean(list(region_mags.values())))
            if region_mags
            else 0.0
        )

        self._magnitude_history.append(agg_magnitude)

        # Build baseline from first N frames
        if not self._baseline_ready:
            self._baseline_accumulator.append(agg_magnitude)
            if len(self._baseline_accumulator) >= self.baseline_frames:
                self._baseline = float(np.mean(self._baseline_accumulator))
                self._baseline_ready = True
            return None  # not enough data yet

        relative_mag: float = agg_magnitude - self._baseline

        # ---- State machine ----
        if self._state == _STATE_IDLE:
            return self._handle_idle(
                frame_index, relative_mag, agg_magnitude, region_mags
            )
        elif self._state in (_STATE_TRACKING_ONSET, _STATE_TRACKING_APEX):
            return self._handle_tracking(
                frame_index, relative_mag, agg_magnitude, region_mags
            )
        else:
            # Should never reach here
            logger.error("Unknown state '%s', resetting.", self._state)
            self._reset_tracking()
            return None

    def get_magnitude_history(self) -> np.ndarray:
        """Return the stored magnitude history as a 1-D float array."""
        return np.array(list(self._magnitude_history), dtype=np.float32)

    def reset(self) -> None:
        """Clear all internal state and buffers."""
        self._state = _STATE_IDLE
        self._magnitude_history.clear()
        self._baseline = 0.0
        self._baseline_accumulator.clear()
        self._baseline_ready = False
        self._reset_tracking()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_idle(
        self,
        frame_index: int,
        relative_mag: float,
        agg_magnitude: float,
        region_mags: Dict[str, float],
    ) -> Optional[MicroExpressionEvent]:
        """IDLE → begin tracking if magnitude exceeds onset threshold."""
        if relative_mag > self.onset_threshold:
            self._state = _STATE_TRACKING_ONSET
            self._onset_frame = frame_index
            self._peak_magnitude = agg_magnitude
            self._apex_frame = frame_index
            self._apex_region_activations = dict(region_mags)
            self._tracking_frames = 1
            self._tracking_history = [
                (frame_index, agg_magnitude, dict(region_mags), None)
            ]
            logger.debug(
                "Onset detected at frame %d (rel_mag=%.3f)",
                frame_index,
                relative_mag,
            )
        return None

    def _handle_tracking(
        self,
        frame_index: int,
        relative_mag: float,
        agg_magnitude: float,
        region_mags: Dict[str, float],
    ) -> Optional[MicroExpressionEvent]:
        """TRACKING → update peak / detect offset / discard macro."""
        self._tracking_frames += 1
        self._tracking_history.append(
            (frame_index, agg_magnitude, dict(region_mags), None)
        )

        # Update apex if new peak
        if agg_magnitude > self._peak_magnitude:
            self._peak_magnitude = agg_magnitude
            self._apex_frame = frame_index
            self._apex_region_activations = dict(region_mags)
            self._state = _STATE_TRACKING_APEX

        # Check for max duration exceeded → discard as macro-expression
        if self._tracking_frames > self.max_duration_frames:
            logger.debug(
                "Macro-expression discarded (duration %d > max %d).",
                self._tracking_frames,
                self.max_duration_frames,
            )
            self._reset_tracking()
            return None

        # Check for offset (magnitude has come back down after a peak was seen)
        if (
            self._state == _STATE_TRACKING_APEX
            and relative_mag < self.offset_threshold
        ):
            offset_frame = frame_index
            duration = offset_frame - self._onset_frame + 1

            if duration < self.min_duration_frames:
                logger.debug(
                    "Event too short (%d frames), discarding.", duration
                )
                self._reset_tracking()
                return None

            event = MicroExpressionEvent(
                onset_frame=self._onset_frame,
                apex_frame=self._apex_frame,
                offset_frame=offset_frame,
                duration_frames=duration,
                duration_ms=(duration / self.fps) * 1000.0,
                peak_magnitude=self._peak_magnitude,
                region_activations=dict(self._apex_region_activations),
                feature_vector=None,
            )

            logger.debug(
                "MicroExpressionEvent: onset=%d apex=%d offset=%d dur=%d",
                event.onset_frame,
                event.apex_frame,
                event.offset_frame,
                event.duration_frames,
            )

            self._reset_tracking()
            return event

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_tracking(self) -> None:
        """Reset tracking variables without touching baseline or history."""
        self._state = _STATE_IDLE
        self._onset_frame = -1
        self._peak_magnitude = 0.0
        self._apex_frame = -1
        self._apex_region_activations = {}
        self._apex_feature_vector = None
        self._tracking_frames = 0
        self._tracking_history.clear()


# ======================================================================
# Lightweight FlowFeatures stub for standalone testing
# ======================================================================


class _FlowFeaturesStub:
    """Minimal stand-in for :class:`FlowFeatures` used in the trial block."""

    __slots__ = (
        "magnitude_histogram",
        "angle_histogram",
        "mean_magnitude",
        "max_magnitude",
        "flow_field",
        "region_label",
    )

    def __init__(self, mean_magnitude: float, region_label: str = "full_face") -> None:
        self.magnitude_histogram = np.zeros(8, dtype=np.float32)
        self.angle_histogram = np.zeros(8, dtype=np.float32)
        self.mean_magnitude = mean_magnitude
        self.max_magnitude = mean_magnitude
        self.flow_field = None
        self.region_label = region_label


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
    print("ApexSpotter — Trial Block")
    print("=" * 60)

    try:
        # ----------------------------------------------------------
        # 1. Synthetic micro-expression magnitude sequence
        # ----------------------------------------------------------
        # baseline (10) → onset-rise (3) → fall/offset (3) → baseline (10)
        magnitudes: List[float] = (
            [0.05, 0.06, 0.07, 0.05, 0.08, 0.06, 0.07, 0.05, 0.09, 0.06]  # 0-9
            + [0.2, 0.5, 0.8]  # 10-12 (rise: onset→apex)
            + [0.4, 0.15, 0.08]  # 13-15 (fall: offset)
            + [0.05, 0.06, 0.07, 0.05, 0.08, 0.06, 0.07, 0.05, 0.09, 0.06]  # 16-25
        )

        spotter = ApexSpotter(
            onset_threshold=0.3,
            offset_threshold=0.15,
            min_duration_frames=2,
            max_duration_frames=25,
            window_size=64,
            fps=30.0,
            baseline_frames=5,
        )

        # ----------------------------------------------------------
        # 2. Feed sequence frame-by-frame
        # ----------------------------------------------------------
        detected_events: List[MicroExpressionEvent] = []

        for idx, mag in enumerate(magnitudes):
            stub = _FlowFeaturesStub(mean_magnitude=mag, region_label="full_face")
            features: Dict[str, _FlowFeaturesStub] = {"full_face": stub}
            event = spotter.update(idx, features)  # type: ignore[arg-type]
            if event is not None:
                detected_events.append(event)

        # ----------------------------------------------------------
        # 3. Exactly 1 event detected
        # ----------------------------------------------------------
        _report(
            "Exactly 1 MicroExpressionEvent detected",
            len(detected_events) == 1,
            f"got {len(detected_events)} events",
        )

        if detected_events:
            ev = detected_events[0]

            # ----------------------------------------------------------
            # 4. onset / apex / offset within ±1 frame
            # ----------------------------------------------------------
            # Expected: onset ~10, apex ~12, offset ~15
            onset_ok = abs(ev.onset_frame - 10) <= 1
            apex_ok = abs(ev.apex_frame - 12) <= 1
            offset_ok = abs(ev.offset_frame - 15) <= 1
            _report(
                f"Onset frame ≈ 10 (±1)",
                onset_ok,
                f"got {ev.onset_frame}",
            )
            _report(
                f"Apex frame ≈ 12 (±1)",
                apex_ok,
                f"got {ev.apex_frame}",
            )
            _report(
                f"Offset frame ≈ 15 (±1)",
                offset_ok,
                f"got {ev.offset_frame}",
            )

            # ----------------------------------------------------------
            # 5. Duration within micro-expression range
            # ----------------------------------------------------------
            dur_ok = 2 <= ev.duration_frames <= 25
            _report(
                "Duration in [2, 25] frames",
                dur_ok,
                f"got {ev.duration_frames} frames ({ev.duration_ms:.1f} ms)",
            )

            # ----------------------------------------------------------
            # Verify peak magnitude is the apex value
            # ----------------------------------------------------------
            _report(
                "Peak magnitude matches apex",
                ev.peak_magnitude >= 0.75,
                f"peak={ev.peak_magnitude:.3f}",
            )
        else:
            # Mark remaining tests as failed since no event was detected
            for name in [
                "Onset frame ≈ 10",
                "Apex frame ≈ 12",
                "Offset frame ≈ 15",
                "Duration in range",
                "Peak magnitude",
            ]:
                _report(name, False, "no event to check")

        # ----------------------------------------------------------
        # 6. No-expression sequence → no events
        # ----------------------------------------------------------
        spotter_quiet = ApexSpotter(
            onset_threshold=0.3,
            offset_threshold=0.15,
            baseline_frames=5,
            fps=30.0,
        )
        quiet_events: List[MicroExpressionEvent] = []
        for idx in range(30):
            stub = _FlowFeaturesStub(mean_magnitude=0.05)
            ev2 = spotter_quiet.update(idx, {"full_face": stub})  # type: ignore[arg-type]
            if ev2 is not None:
                quiet_events.append(ev2)

        _report(
            "No events for flat/quiet sequence",
            len(quiet_events) == 0,
            f"got {len(quiet_events)} events",
        )

        # ----------------------------------------------------------
        # 7. Macro-expression (>25 frames high magnitude) → discarded
        # ----------------------------------------------------------
        spotter_macro = ApexSpotter(
            onset_threshold=0.3,
            offset_threshold=0.15,
            min_duration_frames=2,
            max_duration_frames=25,
            baseline_frames=5,
            fps=30.0,
        )
        macro_events: List[MicroExpressionEvent] = []
        # 5 baseline frames + 30 high-magnitude frames + 5 low frames
        macro_mags: List[float] = (
            [0.05] * 5
            + [0.8] * 30  # sustained high — exceeds max_duration
            + [0.05] * 5
        )
        for idx, mag in enumerate(macro_mags):
            stub = _FlowFeaturesStub(mean_magnitude=mag)
            ev3 = spotter_macro.update(idx, {"full_face": stub})  # type: ignore[arg-type]
            if ev3 is not None:
                macro_events.append(ev3)

        _report(
            "Macro-expression (>25 frames) is discarded",
            len(macro_events) == 0,
            f"got {len(macro_events)} events",
        )

        # ----------------------------------------------------------
        # 8. Magnitude history accessible
        # ----------------------------------------------------------
        hist = spotter.get_magnitude_history()
        _report(
            "get_magnitude_history returns 1-D float array",
            isinstance(hist, np.ndarray) and hist.ndim == 1,
            f"shape={hist.shape}, dtype={hist.dtype}",
        )

        # ----------------------------------------------------------
        # 9. Reset clears state
        # ----------------------------------------------------------
        spotter.reset()
        hist_after = spotter.get_magnitude_history()
        _report(
            "Reset clears magnitude history",
            len(hist_after) == 0,
        )

    except Exception as exc:
        print(f"\n  [FATAL] Unexpected exception: {exc}")
        traceback.print_exc()
        failed += 1

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
