# Micro-Expression Detection System — Technical Blueprint

> **Version**: 1.0  
> **Status**: Architecture Approved — Pre-Implementation  
> **Classification**: Internal Engineering Document  
> **Date**: 2026-07-05

---

## 1. Requirements Breakdown

### 1.1 Functional Requirements

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-01 | **Frame Acquisition** — Ingest from webcam (live), video file (MP4/AVI), or dataset directories (CASME II/SAMM/SMIC) | P0 | Must handle missing camera, corrupt files gracefully |
| FR-02 | **Face Detection** — Detect and localize faces per frame, return bounding boxes | P0 | Support zero-face and multi-face scenarios |
| FR-03 | **Facial Landmark Tracking** — Extract 68-point (dlib) or 478-point (MediaPipe) landmarks | P0 | Track across frames for temporal coherence |
| FR-04 | **ROI Segmentation** — Isolate AU-relevant facial regions (brows, eyes, nose, mouth, cheeks) | P0 | Based on landmark geometry |
| FR-05 | **Temporal Micro-Expression Spotting** — Detect onset-apex-offset within sliding windows (40–500ms / 2–25 frames @ 30fps) | P0 | Peak detection on feature magnitude curves |
| FR-06 | **Classification** — Categorize into 7 classes: happiness, sadness, surprise, fear, disgust, anger, contempt | P0 | Confidence score per prediction |
| FR-07 | **Real-Time Mode** — Live overlay rendering with bounding box, landmarks, label, confidence | P1 | Target <100ms/frame |
| FR-08 | **Offline Batch Mode** — Process video files or dataset directories, output structured results | P1 | CSV/JSON export |
| FR-09 | **Result Logging & Visualization** — Per-frame structured logs, detection timeline, confusion matrix rendering | P1 | SQLite optional |
| FR-10 | **Model Persistence** — Save/load trained models (weights, scaler, label encoder) | P0 | Pickle/ONNX/SavedModel |

### 1.2 Non-Functional Requirements

| ID | Requirement | Target | Constraint |
|----|-------------|--------|------------|
| NF-01 | **Latency** (real-time mode) | <100ms per frame | On consumer CPU (i5-10th gen+) |
| NF-02 | **Accuracy** | UF1 ≥ 0.65, UAR ≥ 0.60 on CASME II | Competitive with published baselines |
| NF-03 | **Hardware** | CPU-only primary | GPU optional via CUDA/TensorRT |
| NF-04 | **Privacy** | Local processing default | No cloud transmission of biometric data |
| NF-05 | **Biometric Data Handling** | GDPR/BIPA-aligned practices | Encryption at rest, consent mechanism, auto-purge |
| NF-06 | **Portability** | Python 3.10+ | Windows/Linux/macOS |
| NF-07 | **Testability** | Every module ships with trial block | Standalone PASS/FAIL verification |

---

## 2. Dataset Strategy

### 2.1 Dataset Comparison Matrix

| Dataset | Subjects | Samples | FPS | Resolution | Emotions | Licensing | Pros | Cons |
|---------|----------|---------|-----|------------|----------|-----------|------|------|
| **CASME II** | 26 | 247 | 200 | 640×480 | 5 classes (+ others) | Academic (request-based) | High temporal resolution, well-annotated, most cited | Small sample count, imbalanced classes |
| **SAMM** | 32 | 159 | 200 | 2040×1088 | 7 AU-coded classes | Academic (request-based) | High spatial resolution, diverse ethnicities, AU-annotated | Very small, requires significant preprocessing |
| **SMIC** | 16 | 164 | 100 | 640×480 | 3 classes (pos/neg/surprise) | Academic (request-based) | Three sub-databases (HS/VIS/NIR) | Coarse labels, lower FPS than CASME II |

### 2.2 Preprocessing Pipeline

1. **Face Cropping** — Detect face in first frame, apply 20% padding, crop all subsequent frames to same ROI (handle drift with periodic re-detection)
2. **Spatial Normalization** — Resize to 224×224 (for CNN) or 128×128 (for flow), apply affine alignment using eye coordinates
3. **Temporal Normalization** — Interpolate or subsample to fixed sequence length (e.g., 16 frames) using cubic interpolation for short clips
4. **Intensity Normalization** — Histogram equalization (CLAHE) to handle lighting variation
5. **Augmentation** — Horizontal flip (preserving AU symmetry awareness), slight rotation (±5°), brightness jitter, temporal jitter (shift onset by ±1 frame)

### 2.3 Licensing Note

All three datasets require institutional affiliation and signed data usage agreements. No redistribution. Models trained on these datasets may be distributed; raw data may not. System must support a `--dataset-path` config pointing to the user's local copy.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   CAPTURE    │───▶│ FACE DETECT  │───▶│  LANDMARKS  │───▶│     ROI      │
│  (camera/    │    │ (Haar/dlib/  │    │ (68pt/478pt)│    │ SEGMENTATION │
│  video/dir)  │    │  MediaPipe)  │    │             │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   OUTPUT /   │◀──│  CLASSIFIER  │◀──│ APEX SPOTTER│◀──│   TEMPORAL    │
│  REPORTING   │    │ (SVM/CNN/   │    │ (peak det.) │    │   FEATURES   │
│  (log/viz)   │    │   LSTM)     │    │             │    │ (OptFlow/LBP)│
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### 3.2 Detection Method Tradeoffs

| Method | Face Detection | Speed (CPU) | Accuracy | Robustness | Best For |
|--------|---------------|-------------|----------|------------|----------|
| **Haar Cascade** | OpenCV built-in | ~15ms/frame | Moderate | Poor (pose/occlusion) | Fallback, embedded |
| **dlib HOG** | dlib | ~40ms/frame | Good | Moderate | CPU-only deployment |
| **dlib CNN** | dlib (MMOD) | ~200ms/frame CPU, ~20ms GPU | Excellent | Good | Accuracy-critical |
| **MediaPipe Face Detection** | MediaPipe | ~10ms/frame | Excellent | Good | Primary choice |
| **MediaPipe Face Mesh** | MediaPipe | ~15ms/frame | Excellent + 478 landmarks | Good | Landmark-rich analysis |

**Decision**: MediaPipe Face Mesh as primary (speed + 478 landmarks), Haar cascade as fallback for environments where MediaPipe fails to initialize.

### 3.3 Temporal Feature Extraction Tradeoffs

| Method | Description | Speed | ME Discriminability | Implementation Complexity |
|--------|-------------|-------|---------------------|---------------------------|
| **Optical Flow (Lucas-Kanade)** | Sparse flow on landmark points | Fast (~5ms) | Moderate | Low |
| **Optical Flow (Farneback)** | Dense flow on full ROI | Moderate (~20ms) | Good | Low |
| **LBP-TOP** | Texture dynamics in XY/XT/YT planes | Moderate (~15ms) | Good (classical baseline) | Medium |
| **3D-CNN (C3D/SlowFast)** | End-to-end spatiotemporal | Slow (~50ms GPU) | High | High |
| **Vision Transformer (TimeSformer)** | Attention-based temporal | Slow (~80ms GPU) | State-of-art | Very High |

**Decision**: Farneback dense optical flow as MVP feature extractor (good balance), with LBP-TOP as supplementary feature. CNN/Transformer as Phase 3 upgrade.

### 3.4 Classifier Tradeoffs

| Classifier | Training Data Needed | Speed (Inference) | Micro-Expression Performance | Overfitting Risk |
|------------|---------------------|-------------------|------------------------------|------------------|
| **SVM (RBF)** | Low (~200 samples) | <1ms | Baseline (UF1 ~0.55-0.65) | Low with proper C/gamma tuning |
| **Random Forest** | Low | <1ms | Similar to SVM | Low |
| **LSTM** | Medium (~500+) | ~5ms | Good temporal modeling | Medium |
| **CNN (ResNet-18)** | High (needs augmentation) | ~10ms CPU, ~2ms GPU | Good with transfer learning | High on small datasets |
| **Transformer** | Very High | ~20ms GPU | State-of-art | Very High |

**Decision**: SVM (RBF) as MVP classifier on optical flow features. LSTM upgrade in Phase 3. CNN with transfer learning (ImageNet pretrained) as Phase 4 option.

---

## 4. Tech Stack Justification

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Language** | Python 3.10+ | Ecosystem maturity, CV/ML library support, type hint features (ParamSpec, TypeAlias) |
| **Core CV** | OpenCV 4.8+ (`opencv-python-headless` for server, `opencv-python` for GUI) | Industry standard, Haar/optical flow built-in, video I/O |
| **Face/Landmarks** | MediaPipe 0.10+ | 478-point face mesh, fast CPU inference, cross-platform |
| **Fallback Detection** | dlib 19.24+ (optional) | 68-point landmarks, HOG detector, well-tested |
| **Numerical** | NumPy 1.24+ | Array operations, feature vector manipulation |
| **ML Classical** | scikit-learn 1.3+ | SVM, StandardScaler, metrics, model persistence |
| **ML Deep** | PyTorch 2.0+ (optional) | LSTM/CNN training, ONNX export, GPU acceleration |
| **Visualization** | Matplotlib 3.7+ | Confusion matrices, flow visualization, timelines |
| **Serving** | FastAPI 0.100+ (optional) | REST API for batch processing, WebSocket for streaming |
| **Storage** | SQLite (built-in) | Result persistence, zero-config |
| **Config** | dataclasses + JSON | Type-safe config, no heavy dependencies |
| **Testing** | pytest 7+ | Unit/integration testing, fixtures |
| **Logging** | Python `logging` + structured JSON | Built-in, configurable, parseable |

---

## 5. Project Structure

```
micro_expression_detector/
├── README.md                          # Setup, usage, trial instructions
├── requirements.txt                   # Pinned dependencies
├── setup.py                           # Package installation
├── config.py                          # Centralized configuration (dataclass)
├── docs/
│   ├── ARCHITECTURE.md                # This document
│   ├── MODEL_CARD.md                  # Model performance & limitations
│   └── SECURITY.md                    # Biometric data handling policy
├── src/
│   └── microex/
│       ├── __init__.py
│       ├── capture.py                 # Frame acquisition (webcam/video/dataset)
│       ├── face_detector.py           # Face detection with fallback chain
│       ├── landmarks.py               # Landmark extraction & ROI cropping
│       ├── motion_features.py         # Optical flow / LBP-TOP features
│       ├── apex_spotter.py            # Onset-apex-offset detection
│       ├── classifier.py              # Train/inference with model persistence
│       ├── pipeline.py                # End-to-end orchestration
│       ├── logger.py                  # Structured logging & export
│       └── utils.py                   # Shared utilities (drawing, transforms)
├── models/                            # Saved model weights (gitignored)
│   └── .gitkeep
├── data/                              # Dataset symlinks/paths (gitignored)
│   └── .gitkeep
├── output/                            # Results, logs, exports (gitignored)
│   └── .gitkeep
├── tests/
│   ├── conftest.py                    # Shared fixtures (sample frames, etc.)
│   ├── test_capture.py
│   ├── test_face_detector.py
│   ├── test_landmarks.py
│   ├── test_motion_features.py
│   ├── test_apex_spotter.py
│   ├── test_classifier.py
│   ├── test_pipeline.py
│   └── test_integration.py           # Full pipeline integration test
├── scripts/
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation/metrics entry point
│   └── run_trial.py                   # Execute all module trials sequentially
└── assets/
    └── sample_face.jpg                # Bundled test image for trials
```

---

## 6. Data Flow Description

### 6.1 Real-Time Mode

1. `capture.py` opens webcam via `cv2.VideoCapture(0)`, yields frames as `np.ndarray` (BGR, uint8).
2. Each frame enters `face_detector.py`. MediaPipe Face Detection runs first; if zero detections, Haar cascade fallback triggers. Output: list of `BoundingBox(x, y, w, h, confidence)`.
3. For each detected face, `landmarks.py` runs MediaPipe Face Mesh on the cropped+padded face ROI. Output: `np.ndarray` shape `(478, 3)` (x, y, z normalized). Extracts sub-ROIs for brows (landmarks 65–68, 105, 107), eyes (33, 133, 159, 145, etc.), nose (1, 2, 98, 327), mouth (61, 291, 0, 17, 78, 308).
4. `motion_features.py` maintains a ring buffer of the last N frames' landmarks/ROIs. Computes Farneback dense optical flow between frame(t) and frame(t-1) for each ROI. Output: flow magnitude + angle histograms as feature vector per ROI.
5. `apex_spotter.py` monitors the sliding window of flow magnitudes. When magnitude exceeds onset threshold, begins tracking. Identifies apex (peak magnitude) and offset (return to baseline). Outputs `MicroExpressionEvent(onset_frame, apex_frame, offset_frame, duration_ms, region_activations)`.
6. On apex detection, `classifier.py` receives the feature vector (stacked flow features from onset→apex) and predicts emotion class + confidence. Loads pre-trained SVM model from disk.
7. `pipeline.py` composites the overlay: bounding box (green), landmarks (dots), predicted label + confidence text, temporal bar showing detection timeline.
8. `logger.py` writes structured JSON log per detection event and per-frame status to `output/session_YYYYMMDD_HHMMSS/`.

### 6.2 Offline Batch Mode

Same pipeline, but `capture.py` iterates over video files or dataset directory structures. Results aggregate into CSV with columns: `video_id, subject_id, onset_frame, apex_frame, offset_frame, predicted_label, confidence, ground_truth_label (if available)`.

---

## 7. Model Selection Decision Matrix

| Criteria (Weight) | SVM+OptFlow (MVP) | LBP-TOP+SVM | CNN (ResNet-18) | LSTM+OptFlow | Transformer |
|--------------------|--------------------|-------------|-----------------|--------------|-------------|
| **Accuracy** (0.30) | 6 | 6 | 8 | 8 | 9 |
| **Speed CPU** (0.25) | 9 | 8 | 4 | 6 | 2 |
| **Data Efficiency** (0.20) | 8 | 8 | 4 | 5 | 2 |
| **Implementation Effort** (0.15) | 9 | 7 | 5 | 5 | 3 |
| **Interpretability** (0.10) | 7 | 6 | 3 | 4 | 2 |
| **Weighted Score** | **7.65** | **7.15** | **5.15** | **6.10** | **3.85** |

**Winner for MVP**: SVM on Optical Flow features. Upgrade path to LSTM for temporal modeling.

---

## 8. Security & Privacy Plan

### 8.1 Principles

- **Local-first**: All processing happens on-device by default. No frame data leaves the machine.
- **No persistence of raw frames**: Frames are processed in-memory and discarded. Only extracted features and metadata are optionally stored.
- **Consent mechanism**: When webcam mode is activated, system displays a consent banner. Configurable via `config.REQUIRE_CONSENT = True`.

### 8.2 Data at Rest

| Data Type | Storage | Encryption | Retention |
|-----------|---------|------------|-----------|
| Raw frames | **Never stored** (RAM only) | N/A | Discarded after processing |
| Extracted features | Optional (SQLite/CSV) | AES-256 via `cryptography` lib (optional) | User-configured TTL |
| Detection events | JSON logs | Filesystem permissions | Session-scoped, auto-purge configurable |
| Trained models | Pickle/ONNX files | Not encrypted (no PII in weights) | Permanent |

### 8.3 Deployment as Service (Optional)

- Require TLS for all API endpoints
- No frame data in API responses (only labels + confidence)
- Rate limiting on video upload endpoints
- Authentication via API key
- Audit logging for all inference requests

---

## 9. Scalability Plan

| Mode | Architecture | Concurrency | Hardware |
|------|-------------|-------------|----------|
| **Single-user real-time** | Single process, main thread loop | 1 camera | Consumer CPU |
| **Batch video processing** | `multiprocessing.Pool` with N workers | N videos in parallel | Multi-core CPU |
| **Multi-camera** | Async frame capture (threading per camera), shared inference queue | M cameras, 1 inference thread | CPU + optional GPU |
| **Service deployment** | FastAPI + async workers, Redis queue for batch jobs | Concurrent API requests | Cloud VM / GPU instance |

---

## 10. Phased Development Roadmap

### Phase 1 — MVP (Weeks 1–2)
- [x] Project structure, config, capture module
- [x] Face detection (MediaPipe + Haar fallback)
- [x] Landmark extraction + ROI segmentation
- [x] Basic optical flow feature extraction
- [x] SVM classifier with synthetic/manual labels
- [x] Pipeline integration (offline mode)
- [x] All module trial blocks passing
- **Gate**: All trial blocks PASS, pipeline processes a sample video end-to-end

### Phase 2 — Optical Flow + Apex Spotting (Weeks 3–4)
- [ ] Farneback dense optical flow integration
- [ ] Sliding window apex spotter with configurable thresholds
- [ ] Real-time overlay rendering
- [ ] Structured logging and CSV/JSON export
- [ ] Dataset loader for CASME II format
- **Gate**: Apex spotter detects synthetic micro-expression events with >80% recall

### Phase 3 — Deep Model Integration (Weeks 5–7)
- [ ] LSTM classifier on temporal flow sequences
- [ ] Transfer learning pipeline (pretrained CNN encoder)
- [ ] Training script with CASME II/SAMM
- [ ] LOSO cross-validation implementation
- [ ] Model evaluation metrics (UF1, UAR, confusion matrix)
- **Gate**: UF1 ≥ 0.55 on CASME II with LOSO-CV

### Phase 4 — Optimization & Hardening (Weeks 8–9)
- [ ] Frame skipping / adaptive processing
- [ ] ROI-only optical flow (skip background)
- [ ] Model quantization (INT8 via ONNX Runtime)
- [ ] Memory profiling and leak fixes
- [ ] Edge case handling (occlusion, lighting, pose)
- **Gate**: <100ms/frame on target CPU, no memory leaks over 1hr session

### Phase 5 — Deployment & Documentation (Week 10)
- [ ] README with full setup/run/trial instructions
- [ ] Docker containerization (optional)
- [ ] pytest suite passing
- [ ] Security review complete
- [ ] Model card published
- **Gate**: All audit items from Prompt 3 addressed

---

## 11. Testing Strategy

### 11.1 Module Trial Blocks (Non-Negotiable)

Every module in `src/microex/` **must** include a `if __name__ == "__main__":` block that:

1. Creates or loads a sample input (synthetic frame, bundled test image, or generated data)
2. Exercises the module's primary function(s)
3. Validates output shape, type, and sanity (e.g., landmarks count == 478, flow magnitude > 0)
4. Prints `PASS: <description>` or `FAIL: <description with reason>`
5. Returns exit code 0 on PASS, 1 on FAIL

**Trial execution order** (via `scripts/run_trial.py`):
```
capture → face_detector → landmarks → motion_features → apex_spotter → classifier → logger → pipeline
```
Each module's trial is independent. A module may not be wired into `pipeline.py` until its trial passes.

### 11.2 pytest Suite

- **Unit tests**: One test file per module, testing core functions with synthetic inputs
- **Integration test**: `test_integration.py` — feeds a sample video through the full pipeline, asserts structured output is produced
- **Fixtures**: Shared synthetic frames, landmark arrays, flow fields in `conftest.py`
- **Coverage target**: ≥80% line coverage on `src/microex/`

### 11.3 Model Evaluation Protocol

- **Primary metric**: Unweighted F1 (UF1) — handles class imbalance in micro-expression datasets
- **Secondary metric**: Unweighted Average Recall (UAR)
- **Cross-validation**: Leave-One-Subject-Out (LOSO) — prevents data leakage from same-subject samples
- **Baseline comparison**: Report against published results on same dataset split
- **Overfitting detection**: Compare train vs validation loss curves; flag if train accuracy > val accuracy by >15%

---

*End of Architecture Document — Proceed to Implementation (Prompt 2)*
