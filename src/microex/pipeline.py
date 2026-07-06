#!/usr/bin/env python3
"""
pipeline.py — End-to-end orchestration for the Micro-Expression Detection System.

Orchestrates: Capture → Face Detection → Landmarks → Motion Features →
              Apex Spotting → Classification → Output/Visualization.

Supports real-time mode (webcam with overlay) and offline batch mode (video/dataset
with structured output).
"""

import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.microex.capture import VideoSource, WebcamSource, DatasetSource, FrameResult
from src.microex.face_detector import FaceDetector, BoundingBox
from src.microex.landmarks import LandmarkExtractor, FacialLandmarks, FacialROIs
from src.microex.motion_features import MotionFeatureExtractor, FlowFeatures
from src.microex.apex_spotter import ApexSpotter, MicroExpressionEvent
from src.microex.classifier import EmotionClassifier, PredictionResult
from src.microex.logger import DetectionLogger
from src.microex.static_features import extract_static_features
from src.microex.utils import (
    draw_bounding_box,
    draw_landmarks,
    resize_with_aspect_ratio,
    apply_clahe,
    ensure_directory,
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────
@dataclass
class FrameProcessingResult:
    """Result of processing a single frame through the pipeline."""
    frame_index: int
    timestamp_ms: float
    processing_time_ms: float
    faces_detected: int
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    landmarks: List[Optional[FacialLandmarks]] = field(default_factory=list)
    flow_features: List[Dict[str, FlowFeatures]] = field(default_factory=list)
    events: List[MicroExpressionEvent] = field(default_factory=list)
    predictions: List[PredictionResult] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestration."""
    input_mode: str = "video"              # 'webcam', 'video', 'dataset'
    video_path: Optional[str] = None
    dataset_path: Optional[str] = None
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Detection
    face_detection_backend: str = "mediapipe"
    min_face_confidence: float = 0.5

    # Landmarks
    landmark_model: str = "mediapipe_mesh"

    # Motion
    flow_method: str = "farneback"

    # Apex spotting
    onset_threshold: float = 0.15
    offset_threshold: float = 0.08
    min_duration_frames: int = 2
    max_duration_frames: int = 25
    sliding_window_size: int = 16

    # Classifier
    classifier_type: str = "svm"
    model_path: Optional[str] = None
    confidence_threshold: float = 0.2

    # Output
    output_dir: str = "output"
    show_overlay: bool = True
    save_annotated_video: bool = False

    # Performance
    target_fps: float = 30.0
    max_faces: int = 3
    enable_gpu: bool = False
    skip_frames: int = 0  # Process every Nth frame (0 = no skip)


# ── Main Pipeline ─────────────────────────────────────────────────────────────
class MicroExpressionPipeline:
    """
    End-to-end micro-expression detection pipeline.

    Orchestrates all modules from frame capture through classification,
    with support for real-time overlay rendering and batch processing.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """
        Initialize the pipeline with all sub-modules.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        self._initialized = False
        self._frame_count = 0
        self._total_processing_time = 0.0

        # Sub-modules (initialized lazily)
        self._detector: Optional[FaceDetector] = None
        self._landmark_extractor: Optional[LandmarkExtractor] = None
        self._motion_extractors: Dict[int, MotionFeatureExtractor] = {}
        self._apex_spotters: Dict[int, ApexSpotter] = {}
        self._history_buffers: Dict[int, List[Tuple[int, Optional[FacialLandmarks], Dict[str, FlowFeatures]]]] = {}
        self._classifier: Optional[EmotionClassifier] = None
        self._logger: Optional[DetectionLogger] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._display_predictions: Dict[int, Tuple[str, float, int]] = {}

    def initialize(self) -> None:
        """Initialize all sub-modules. Call before processing."""
        logger.info("Initializing pipeline modules...")

        # Face detector
        self._detector = FaceDetector(
            backend=self.config.face_detection_backend,
            min_confidence=self.config.min_face_confidence,
        )

        # Landmark extractor
        self._landmark_extractor = LandmarkExtractor(
            model=self.config.landmark_model,
        )

        # Classifier (load pre-trained if path provided)
        self._classifier = EmotionClassifier(model_type=self.config.classifier_type)
        if self.config.model_path and Path(self.config.model_path).exists():
            self._classifier.load_model(self.config.model_path)
            logger.info("Loaded classifier model from %s", self.config.model_path)
        else:
            logger.warning(
                "No pre-trained model loaded. Classification will raise errors "
                "until a model is trained or loaded."
            )

        # Logger
        self._logger = DetectionLogger(output_dir=self.config.output_dir)

        self._initialized = True
        logger.info("Pipeline initialized successfully.")

    def _get_motion_extractor(self, face_id: int) -> MotionFeatureExtractor:
        """Get or create a motion feature extractor for a tracked face."""
        if face_id not in self._motion_extractors:
            self._motion_extractors[face_id] = MotionFeatureExtractor(
                method=self.config.flow_method,
            )
        return self._motion_extractors[face_id]

    def _get_apex_spotter(self, face_id: int) -> ApexSpotter:
        """Get or create an apex spotter for a tracked face."""
        if face_id not in self._apex_spotters:
            self._apex_spotters[face_id] = ApexSpotter(
                onset_threshold=self.config.onset_threshold,
                offset_threshold=self.config.offset_threshold,
                min_duration_frames=self.config.min_duration_frames,
                max_duration_frames=self.config.max_duration_frames,
                window_size=self.config.sliding_window_size,
                fps=self.config.target_fps,
            )
        return self._apex_spotters[face_id]

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float = 0.0,
    ) -> FrameProcessingResult:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR input frame (np.ndarray, uint8).
            frame_index: Frame sequence number.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            FrameProcessingResult with all detections and predictions.

        Raises:
            RuntimeError: If pipeline is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()
        result = FrameProcessingResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            processing_time_ms=0.0,
            faces_detected=0,
        )

        annotated = frame.copy() if self.config.show_overlay else None

        try:
            # ── Step 1: Face Detection ─────────────────────────────────────
            bounding_boxes = self._detector.detect(frame)
            bounding_boxes = bounding_boxes[: self.config.max_faces]
            result.faces_detected = len(bounding_boxes)
            result.bounding_boxes = bounding_boxes

            if not bounding_boxes:
                self._frame_count += 1
                elapsed = (time.perf_counter() - start_time) * 1000
                result.processing_time_ms = elapsed
                result.annotated_frame = annotated
                return result

            # ── Step 2–6: Per-face processing ──────────────────────────────
            for face_id, bbox in enumerate(bounding_boxes):
                # Step 2: Landmark extraction
                lm = self._landmark_extractor.extract(frame, bbox)
                result.landmarks.append(lm)

                if lm is None:
                    continue

                # Step 3: ROI segmentation
                rois = self._landmark_extractor.get_rois(frame, lm)

                # Step 4: Motion features (on grayscale full face)
                face_gray = cv2.cvtColor(rois.full_face, cv2.COLOR_BGR2GRAY)
                face_gray = apply_clahe(face_gray)
                motion_ext = self._get_motion_extractor(face_id)
                motion_ext.update(face_gray)

                # Need at least 2 frames for flow
                if len(motion_ext._buffer) < 2:
                    continue

                prev_gray = motion_ext._buffer[-2]
                if prev_gray.shape != face_gray.shape:
                    prev_gray = cv2.resize(prev_gray, (face_gray.shape[1], face_gray.shape[0]))
                
                flow = motion_ext.compute_flow(prev_gray, face_gray)
                flow_feats = motion_ext.extract_features(flow, region_label="full_face")
                roi_flow = {"full_face": flow_feats}
                result.flow_features.append(roi_flow)

                # Store history for feature extraction at apex
                if face_id not in self._history_buffers:
                    self._history_buffers[face_id] = []
                self._history_buffers[face_id].append((frame_index, lm, roi_flow))
                # keep last 100 frames
                if len(self._history_buffers[face_id]) > 100:
                    self._history_buffers[face_id].pop(0)

                # Step 5: Apex spotting
                spotter = self._get_apex_spotter(face_id)
                event = spotter.update(frame_index, roi_flow)

                if event is not None:
                    result.events.append(event)

                    # Assemble feature vector from apex frame's landmarks
                    # Uses static 971-D landmark geometry features for
                    # compatibility with multi-dataset trained classifier.
                    apex_lm = None
                    for f_idx, h_lm, _h_flow in self._history_buffers[face_id]:
                        if f_idx == event.apex_frame:
                            apex_lm = h_lm
                            break

                    if apex_lm is not None and apex_lm.points.shape[0] >= 478:
                        try:
                            event.feature_vector = extract_static_features(
                                apex_lm.points,
                                apex_lm.face_bbox.x,
                                apex_lm.face_bbox.y,
                                apex_lm.face_bbox.w,
                                apex_lm.face_bbox.h,
                            )
                        except Exception:
                            event.feature_vector = None

                    # Step 6: Classification
                    if event.feature_vector is not None and self._classifier is not None:
                        try:
                            pred = self._classifier.predict(event.feature_vector)
                            if pred.confidence >= self.config.confidence_threshold:
                                result.predictions.append(pred)
                                self._display_predictions[face_id] = (pred.label, pred.confidence, frame_index + 60)
                        except (RuntimeError, ValueError) as e:
                            logger.debug("Classification skipped: %s", e)

                # ── Overlay rendering ──────────────────────────────────────
                if annotated is not None:
                    # Bounding box
                    color = (0, 255, 0)
                    label_text = ""
                    conf = 0.0
                    
                    if face_id in self._display_predictions:
                        disp_label, disp_conf, expiry = self._display_predictions[face_id]
                        if frame_index <= expiry:
                            label_text = disp_label
                            conf = disp_conf
                            color = (0, 255, 255)  # Yellow for active expression
                        else:
                            del self._display_predictions[face_id]

                    draw_bounding_box(
                        annotated, bbox, color=color,
                        label=label_text, confidence=conf,
                    )

                    # Landmarks
                    if lm is not None:
                        draw_landmarks(annotated, lm.points[:, :2], color=(0, 200, 0), radius=1)

        except Exception as e:
            logger.error("Frame %d processing error: %s", frame_index, e, exc_info=True)
            if self._logger:
                self._logger.log_error(frame_index, str(e), module="pipeline")

        # ── Timing ─────────────────────────────────────────────────────────
        elapsed = (time.perf_counter() - start_time) * 1000
        result.processing_time_ms = elapsed
        result.annotated_frame = annotated
        self._frame_count += 1
        self._total_processing_time += elapsed

        # ── Logging ────────────────────────────────────────────────────────
        if self._logger:
            self._logger.log_frame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                faces_detected=result.faces_detected,
                processing_time_ms=elapsed,
            )
            for event in result.events:
                det_info = {
                    "onset_frame": event.onset_frame,
                    "apex_frame": event.apex_frame,
                    "offset_frame": event.offset_frame,
                    "duration_ms": event.duration_ms,
                    "peak_magnitude": event.peak_magnitude,
                }
                if result.predictions:
                    det_info["predicted_label"] = result.predictions[-1].label
                    det_info["confidence"] = result.predictions[-1].confidence
                self._logger.log_detection(det_info)

        return result

    def run_realtime(self) -> None:
        """
        Run the pipeline in real-time mode with webcam input and overlay display.

        Press 'q' to quit.
        """
        if not self._initialized:
            self.initialize()

        logger.info("Starting real-time mode (camera index: %d)", self.config.camera_index)

        source = WebcamSource(
            camera_index=self.config.camera_index,
            width=self.config.frame_width,
            height=self.config.frame_height,
        )

        try:
            with source:
                for frame_result in source:
                    if self.config.skip_frames > 0:
                        if frame_result.frame_index % (self.config.skip_frames + 1) != 0:
                            continue

                    result = self.process_frame(
                        frame_result.frame,
                        frame_result.frame_index,
                        frame_result.timestamp_ms,
                    )

                    if result.annotated_frame is not None:
                        # Add FPS counter
                        fps = 1000.0 / max(result.processing_time_ms, 0.001)
                        cv2.putText(
                            result.annotated_frame,
                            f"FPS: {fps:.1f} | Faces: {result.faces_detected}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow("Micro-Expression Detector", result.annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User requested quit.")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cv2.destroyAllWindows()
            self._finalize()

    def run_video(self, video_path: str) -> List[FrameProcessingResult]:
        """
        Run the pipeline on a video file in offline mode.

        Args:
            video_path: Path to the video file.

        Returns:
            List of FrameProcessingResult for each processed frame.
        """
        if not self._initialized:
            self.initialize()

        logger.info("Processing video: %s", video_path)
        results: List[FrameProcessingResult] = []

        source = VideoSource(video_path)
        try:
            with source:
                total = source.get_frame_count()
                for frame_result in source:
                    if self.config.skip_frames > 0:
                        if frame_result.frame_index % (self.config.skip_frames + 1) != 0:
                            continue

                    result = self.process_frame(
                        frame_result.frame,
                        frame_result.frame_index,
                        frame_result.timestamp_ms,
                    )
                    results.append(result)

                    # Progress reporting
                    if total > 0 and frame_result.frame_index % 100 == 0:
                        pct = (frame_result.frame_index / total) * 100
                        logger.info("Progress: %.1f%% (%d/%d frames)", pct, frame_result.frame_index, total)

        finally:
            self._finalize()

        return results

    def run_batch(self, dataset_path: str) -> Dict[str, List[FrameProcessingResult]]:
        """
        Run the pipeline on a dataset directory in batch mode.

        Args:
            dataset_path: Path to dataset root directory.

        Returns:
            Dict mapping source_id to list of FrameProcessingResult.
        """
        if not self._initialized:
            self.initialize()

        logger.info("Batch processing dataset: %s", dataset_path)
        all_results: Dict[str, List[FrameProcessingResult]] = {}
        current_source_id = ""
        current_results: List[FrameProcessingResult] = []

        source = DatasetSource(dataset_path)
        try:
            with source:
                for frame_result in source:
                    if frame_result.source_id != current_source_id:
                        if current_source_id and current_results:
                            all_results[current_source_id] = current_results
                        current_source_id = frame_result.source_id
                        current_results = []
                        # Reset per-video state
                        self._motion_extractors.clear()
                        self._apex_spotters.clear()
                        self._history_buffers.clear()

                    result = self.process_frame(
                        frame_result.frame,
                        frame_result.frame_index,
                        frame_result.timestamp_ms,
                    )
                    current_results.append(result)

                # Save last video's results
                if current_source_id and current_results:
                    all_results[current_source_id] = current_results

        finally:
            self._finalize()

        return all_results

    def _finalize(self) -> None:
        """Clean up resources and export results."""
        if self._logger:
            try:
                self._logger.export_csv()
                self._logger.export_json()
            except Exception as e:
                logger.warning("Error exporting results: %s", e)
            self._logger.close()

        if self._detector:
            self._detector.release()
        if self._landmark_extractor:
            self._landmark_extractor.release()
        if self._video_writer:
            self._video_writer.release()

        avg_time = (
            self._total_processing_time / self._frame_count
            if self._frame_count > 0
            else 0
        )
        logger.info(
            "Pipeline finalized. Processed %d frames, avg %.1f ms/frame.",
            self._frame_count,
            avg_time,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline performance statistics."""
        avg_time = (
            self._total_processing_time / self._frame_count
            if self._frame_count > 0
            else 0
        )
        return {
            "frames_processed": self._frame_count,
            "total_time_ms": self._total_processing_time,
            "avg_time_per_frame_ms": avg_time,
            "avg_fps": 1000.0 / max(avg_time, 0.001),
        }

    def release(self) -> None:
        """Release all resources."""
        self._finalize()
        self._initialized = False


# ── Trial Block ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("TRIAL: pipeline.py — End-to-end pipeline orchestration")
    print("=" * 60)

    all_passed = True

    # Test 1: Pipeline instantiation with config
    try:
        config = PipelineConfig(
            input_mode="video",
            face_detection_backend="haar",
            flow_method="farneback",
            show_overlay=False,
        )
        pipeline = MicroExpressionPipeline(config)
        assert pipeline.config.face_detection_backend == "haar"
        assert not pipeline._initialized
        print("PASS: Pipeline instantiation with custom config")
    except Exception as e:
        print(f"FAIL: Pipeline instantiation — {e}")
        all_passed = False

    # Test 2: Pipeline initialization
    try:
        pipeline.initialize()
        assert pipeline._initialized
        print("PASS: Pipeline initialization (all modules created)")
    except Exception as e:
        print(f"FAIL: Pipeline initialization — {e}")
        all_passed = False

    # Test 3: Process a synthetic frame
    try:
        # Create a synthetic frame with a face-like shape
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw skin-tone oval
        cv2.ellipse(frame, (320, 200), (100, 130), 0, 0, 360, (180, 200, 220), -1)
        # Draw eyes
        cv2.circle(frame, (280, 170), 12, (50, 50, 50), -1)
        cv2.circle(frame, (360, 170), 12, (50, 50, 50), -1)
        # Draw mouth
        cv2.ellipse(frame, (320, 240), (30, 10), 0, 0, 360, (50, 50, 150), -1)

        result = pipeline.process_frame(frame, frame_index=0, timestamp_ms=0.0)
        assert isinstance(result, FrameProcessingResult)
        assert result.frame_index == 0
        assert result.processing_time_ms >= 0
        assert result.faces_detected >= 0  # Haar may or may not detect synthetic
        print(f"PASS: Frame processing — {result.faces_detected} faces, "
              f"{result.processing_time_ms:.1f}ms")
    except Exception as e:
        print(f"FAIL: Frame processing — {e}")
        all_passed = False

    # Test 4: Process multiple frames (temporal coherence)
    try:
        for i in range(5):
            shifted = np.roll(frame, i * 2, axis=1)  # Slight horizontal shift
            result = pipeline.process_frame(shifted, frame_index=i + 1, timestamp_ms=(i + 1) * 33.3)
        stats = pipeline.get_stats()
        assert stats["frames_processed"] >= 5
        assert stats["avg_time_per_frame_ms"] >= 0
        print(f"PASS: Multi-frame processing — {stats['frames_processed']} frames, "
              f"avg {stats['avg_time_per_frame_ms']:.1f}ms/frame")
    except Exception as e:
        print(f"FAIL: Multi-frame processing — {e}")
        all_passed = False

    # Test 5: Process empty frame (no face)
    try:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(blank, frame_index=99, timestamp_ms=3300.0)
        assert result.faces_detected == 0
        assert result.bounding_boxes == []
        print("PASS: Empty frame handling — 0 faces correctly")
    except Exception as e:
        print(f"FAIL: Empty frame handling — {e}")
        all_passed = False

    # Test 6: Uninitialised pipeline guard
    try:
        fresh_pipeline = MicroExpressionPipeline()
        try:
            fresh_pipeline.process_frame(frame, 0, 0.0)
            print("FAIL: Uninitialised guard — should have raised RuntimeError")
            all_passed = False
        except RuntimeError:
            print("PASS: Uninitialised pipeline correctly raises RuntimeError")
    except Exception as e:
        print(f"FAIL: Uninitialised guard — {e}")
        all_passed = False

    # Test 7: Pipeline stats
    try:
        stats = pipeline.get_stats()
        assert "frames_processed" in stats
        assert "avg_fps" in stats
        assert stats["avg_fps"] > 0
        print(f"PASS: Pipeline stats — {stats['avg_fps']:.0f} avg FPS")
    except Exception as e:
        print(f"FAIL: Pipeline stats — {e}")
        all_passed = False

    # Cleanup
    try:
        pipeline.release()
        print("PASS: Pipeline release (resources freed)")
    except Exception as e:
        print(f"FAIL: Pipeline release — {e}")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL PIPELINE TRIALS PASSED")
    else:
        print("RESULT: SOME PIPELINE TRIALS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)
