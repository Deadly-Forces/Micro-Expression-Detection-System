"""
Structured logging and result export module for the Micro-Expression
Detection System.

Provides:
* Per-frame status logging (JSONL)
* Detection-event logging (JSONL)
* Error logging (JSONL)
* CSV / JSON export
* Thread-safe writes
* Context-manager interface
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class DetectionLogger:
    """Thread-safe, structured logger for micro-expression detection sessions.

    Parameters
    ----------
    output_dir : str
        Root directory for session output.
    session_id : str or None
        Human-readable session identifier.  When *None* a timestamped name
        is generated automatically (``session_YYYYMMDD_HHMMSS``).
    """

    def __init__(
        self,
        output_dir: str = "output",
        session_id: Optional[str] = None,
    ) -> None:
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

        self._output_dir = output_dir
        self._session_id: str = session_id or datetime.now().strftime(
            "session_%Y%m%d_%H%M%S"
        )
        self._session_dir: str = os.path.join(self._output_dir, self._session_id)
        os.makedirs(self._session_dir, exist_ok=True)

        self._lock = threading.Lock()

        # ── file handles (lazy-open on first write) ─────────────────────
        self._frame_log_path = os.path.join(self._session_dir, "frame_log.jsonl")
        self._detections_path = os.path.join(self._session_dir, "detections.jsonl")
        self._errors_path = os.path.join(self._session_dir, "errors.jsonl")

        self._frame_fh: Optional[io.TextIOWrapper] = None
        self._detections_fh: Optional[io.TextIOWrapper] = None
        self._errors_fh: Optional[io.TextIOWrapper] = None

        # Accumulate detection records in memory for easy export
        self._detection_records: List[Dict[str, Any]] = []
        self._frame_count: int = 0

        # ── python logging setup ────────────────────────────────────────
        self._py_logger = logging.getLogger(f"microex.session.{self._session_id}")
        self._py_logger.setLevel(logging.DEBUG)
        # Avoid duplicate handlers when re-creating in tests
        if not self._py_logger.handlers:
            # File handler — JSON lines
            log_file = os.path.join(self._session_dir, "session.log")
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._py_logger.addHandler(fh)

            # Console handler — human-readable
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._py_logger.addHandler(ch)

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _iso_now() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def _ensure_fh(self, attr: str, path: str) -> io.TextIOWrapper:
        """Lazily open a file handle."""
        fh = getattr(self, attr)
        if fh is None or fh.closed:
            fh = open(path, "a", encoding="utf-8")  # noqa: SIM115
            setattr(self, attr, fh)
        return fh

    def _write_jsonl(self, attr: str, path: str, record: Dict[str, Any]) -> None:
        """Thread-safe append of a JSON record to a JSONL file."""
        with self._lock:
            fh = self._ensure_fh(attr, path)
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()

    # ── public API ───────────────────────────────────────────────────────
    def log_frame(
        self,
        frame_index: int,
        timestamp_ms: float,
        faces_detected: int,
        processing_time_ms: float,
    ) -> None:
        """Record per-frame processing status.

        Parameters
        ----------
        frame_index : int
            Zero-based frame number.
        timestamp_ms : float
            Position in the source video (milliseconds).
        faces_detected : int
            Number of faces found in this frame.
        processing_time_ms : float
            Wall-clock time spent processing the frame (ms).
        """
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")

        record: Dict[str, Any] = {
            "timestamp": self._iso_now(),
            "level": "INFO",
            "module": "pipeline",
            "frame": frame_index,
            "timestamp_ms": timestamp_ms,
            "faces_detected": faces_detected,
            "processing_time_ms": round(processing_time_ms, 3),
        }
        self._write_jsonl("_frame_fh", self._frame_log_path, record)
        self._frame_count += 1

    def log_detection(self, event: Dict[str, Any]) -> None:
        """Record a micro-expression detection event.

        The *event* dict should typically contain keys such as
        ``onset``, ``apex``, ``offset``, ``label``, and ``confidence``.

        Parameters
        ----------
        event : dict
            Detection payload.
        """
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")

        record: Dict[str, Any] = {
            "timestamp": self._iso_now(),
            "level": "INFO",
            "module": "detector",
            **event,
        }
        self._write_jsonl("_detections_fh", self._detections_path, record)
        self._detection_records.append(record)

        self._py_logger.info(
            json.dumps({"event": "detection", **event}, default=str)
        )

    def log_error(
        self,
        frame_index: int,
        error: str,
        module: str,
    ) -> None:
        """Record an error with context.

        Parameters
        ----------
        frame_index : int
            Frame at which the error occurred.
        error : str
            Human-readable error description.
        module : str
            Originating module / component name.
        """
        record: Dict[str, Any] = {
            "timestamp": self._iso_now(),
            "level": "ERROR",
            "module": module,
            "frame": frame_index,
            "error": error,
        }
        self._write_jsonl("_errors_fh", self._errors_path, record)
        self._py_logger.error(
            json.dumps({"event": "error", "frame": frame_index, "error": error, "module": module}, default=str)
        )

    # ── exports ──────────────────────────────────────────────────────────
    def export_csv(self, output_path: Optional[str] = None) -> str:
        """Export all detections to a CSV file.

        Parameters
        ----------
        output_path : str or None
            Destination path.  Defaults to ``<session_dir>/detections.csv``.

        Returns
        -------
        str
            Absolute path to the written CSV.
        """
        if output_path is None:
            output_path = os.path.join(self._session_dir, "detections.csv")

        # Collect all unique keys across records for the CSV header
        all_keys: List[str] = []
        seen: set = set()
        for rec in self._detection_records:
            for k in rec:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        if not all_keys:
            all_keys = ["timestamp", "level", "module", "label", "confidence"]

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for rec in self._detection_records:
                writer.writerow(rec)

        logger.info("CSV exported to %s", output_path)
        return os.path.abspath(output_path)

    def export_json(self, output_path: Optional[str] = None) -> str:
        """Export a full session summary as a single JSON document.

        Parameters
        ----------
        output_path : str or None
            Destination path.  Defaults to ``<session_dir>/summary.json``.

        Returns
        -------
        str
            Absolute path to the written JSON.
        """
        if output_path is None:
            output_path = os.path.join(self._session_dir, "summary.json")

        summary: Dict[str, Any] = {
            "session_id": self._session_id,
            "session_dir": self._session_dir,
            "frames_logged": self._frame_count,
            "detections_count": len(self._detection_records),
            "exported_at": self._iso_now(),
            "detections": self._detection_records,
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        logger.info("JSON summary exported to %s", output_path)
        return os.path.abspath(output_path)

    def get_session_dir(self) -> str:
        """Return the absolute session directory path."""
        return os.path.abspath(self._session_dir)

    # ── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        """Flush and close all open file handles."""
        with self._lock:
            for attr in ("_frame_fh", "_detections_fh", "_errors_fh"):
                fh = getattr(self, attr, None)
                if fh is not None and not fh.closed:
                    fh.flush()
                    fh.close()
            # Also close Python logging file handlers
            for handler in list(self._py_logger.handlers):
                handler.close()
                self._py_logger.removeHandler(handler)

    def __enter__(self) -> "DetectionLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


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
    print("DetectionLogger — Trial Block")
    print("=" * 60)

    tmp_root = tempfile.mkdtemp(prefix="microex_log_trial_")
    try:
        # ── 1. Create logger ─────────────────────────────────────────────
        with DetectionLogger(output_dir=tmp_root, session_id="test_session") as dlog:
            session_dir = dlog.get_session_dir()
            _report("Logger creation", os.path.isdir(session_dir),
                    f"dir={session_dir}")

            # ── 2. Log 10 frames ─────────────────────────────────────────
            for i in range(10):
                dlog.log_frame(
                    frame_index=i,
                    timestamp_ms=i * 33.33,
                    faces_detected=1,
                    processing_time_ms=5.0 + i * 0.1,
                )
            frame_log = os.path.join(session_dir, "frame_log.jsonl")
            with open(frame_log, "r", encoding="utf-8") as f:
                frame_lines = f.readlines()
            _report("frame_log.jsonl exists", os.path.isfile(frame_log))
            _report("frame_log has 10 lines", len(frame_lines) == 10,
                    f"got {len(frame_lines)}")

            # ── 3. Log 2 detections ──────────────────────────────────────
            dlog.log_detection({
                "onset": 10, "apex": 15, "offset": 20,
                "label": "surprise", "confidence": 0.87,
            })
            dlog.log_detection({
                "onset": 50, "apex": 55, "offset": 60,
                "label": "anger", "confidence": 0.72,
            })
            det_log = os.path.join(session_dir, "detections.jsonl")
            with open(det_log, "r", encoding="utf-8") as f:
                det_lines = f.readlines()
            _report("detections.jsonl has 2 lines", len(det_lines) == 2,
                    f"got {len(det_lines)}")

            # ── 4. Log 1 error ───────────────────────────────────────────
            dlog.log_error(
                frame_index=25,
                error="Landmark detection failed — face occluded",
                module="landmark_detector",
            )
            err_log = os.path.join(session_dir, "errors.jsonl")
            with open(err_log, "r", encoding="utf-8") as f:
                err_lines = f.readlines()
            _report("errors.jsonl has 1 line", len(err_lines) == 1,
                    f"got {len(err_lines)}")

            # ── 5. JSONL records are valid JSON ──────────────────────────
            try:
                parsed = json.loads(frame_lines[0])
                _report("Frame record is valid JSON",
                        "frame" in parsed and "timestamp" in parsed,
                        f"keys={list(parsed.keys())}")
            except json.JSONDecodeError as exc:
                _report("Frame record is valid JSON", False, str(exc))

            # ── 6. Export CSV ────────────────────────────────────────────
            csv_path = dlog.export_csv()
            csv_exists = os.path.isfile(csv_path)
            csv_cols_ok = False
            if csv_exists:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    cols = reader.fieldnames or []
                    csv_cols_ok = "label" in cols and "confidence" in cols
            _report("CSV export exists", csv_exists, f"path={csv_path}")
            _report("CSV has expected columns", csv_cols_ok,
                    f"columns={cols if csv_exists else 'N/A'}")

            # ── 7. Export JSON summary ───────────────────────────────────
            json_path = dlog.export_json()
            json_exists = os.path.isfile(json_path)
            json_valid = False
            if json_exists:
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        summary = json.load(f)
                        json_valid = (
                            summary.get("session_id") == "test_session"
                            and summary.get("detections_count") == 2
                        )
                    except json.JSONDecodeError:
                        pass
            _report("JSON export exists", json_exists, f"path={json_path}")
            _report("JSON content valid", json_valid,
                    f"session_id={summary.get('session_id', '?')}, "
                    f"det_count={summary.get('detections_count', '?')}"
                    if json_exists and json_valid else "")

    finally:
        # ── Cleanup ──────────────────────────────────────────────────────
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
            _report("Temp cleanup", True)
        except Exception as exc:
            _report("Temp cleanup", False, str(exc))

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests PASSED.")
        sys.exit(0)
