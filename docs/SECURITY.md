# Security & Privacy Policy — Micro-Expression Detection System

> **Classification**: Internal / Deployment-Required Reading  
> **Last Updated**: 2026-07-05

---

## 1. Biometric Data Classification

This system processes **facial biometric data** as defined under:
- **GDPR** Article 9 (Special Categories of Personal Data)
- **Illinois BIPA** (Biometric Information Privacy Act)
- **CCPA/CPRA** (California Consumer Privacy Act — biometric identifiers)

All facial images, landmark coordinates, and emotion classifications derived from facial analysis constitute **sensitive personal data** requiring heightened protection.

---

## 2. Data Handling Principles

### 2.1 Local-First Processing

| Principle | Implementation |
|-----------|---------------|
| **No cloud transmission** | All inference runs on-device. No frames, features, or results are transmitted over network by default. |
| **No raw frame persistence** | Frames exist only in RAM during processing. No automatic saving to disk. |
| **Minimal data retention** | Only extracted metadata (labels, timestamps, confidence scores) is stored. Feature vectors are discarded post-classification unless explicitly configured. |
| **Consent-gated acquisition** | Webcam mode requires explicit opt-in. `config.require_consent = True` by default. |

### 2.2 Data at Rest

| Data Type | Default State | Encryption Option | Retention Policy |
|-----------|--------------|-------------------|-----------------|
| Raw video frames | **Never stored** | N/A | Immediate discard |
| Facial landmarks | **RAM only** | N/A | Discarded per-frame |
| Optical flow features | **RAM only** | N/A | Discarded per-frame |
| Detection events (JSON logs) | Stored in session dir | AES-256 via `cryptography` (opt-in) | Session-scoped, configurable auto-purge |
| Trained model weights | Stored on disk | Not encrypted (no PII embedded) | Permanent |
| Evaluation metrics | Stored in session dir | Not encrypted (aggregate, non-PII) | Permanent |

### 2.3 Data in Transit

- Default deployment: **No network communication**
- If FastAPI service is enabled:
  - TLS 1.3 required for all endpoints
  - No raw frame data in API responses (labels + confidence only)
  - API key authentication required
  - Rate limiting enforced (default: 60 req/min)

---

## 3. Consent Handling

### 3.1 Webcam Mode

When `config.require_consent = True` (default):

1. System displays console banner: *"This application will access your camera for facial analysis. Press 'y' to consent, 'n' to exit."*
2. If GUI mode: overlay consent dialog on first frame with Accept/Decline buttons
3. Consent state is logged with timestamp
4. User can revoke consent at any time by pressing 'q' (stops processing, deletes session data)

### 3.2 Video/Dataset Mode

- Assumes consent was obtained during data collection
- System logs a warning: *"Processing pre-recorded video. Ensure appropriate consent was obtained during recording."*

---

## 4. Threat Model

| Threat | Risk Level | Mitigation |
|--------|-----------|------------|
| Unauthorized camera access | High | Consent gate, OS-level camera permissions |
| Frame data exfiltration | High | Local-only processing, no network calls |
| Model inversion (reconstructing faces from model) | Low | SVM/shallow models have low inversion risk |
| Session log exposure | Medium | Filesystem permissions, optional encryption |
| Temp file leakage | Medium | No temp files with frame data; temp dirs cleaned on exit |
| Adversarial input (fooling classifier) | Medium | Input validation, confidence thresholds |
| Inference on unconsenting subjects | High | Consent mechanism, deployment guidelines |

---

## 5. Deployment Guidelines

### 5.1 Acceptable Use

- **Research**: Academic study of micro-expressions with IRB-approved protocols
- **Training**: Self-directed emotion recognition training tools
- **Accessibility**: Assistive technology for individuals with alexithymia

### 5.2 Prohibited Use

- Covert surveillance or monitoring without consent
- Employment screening or hiring decisions
- Law enforcement interrogation without legal basis
- Deception detection claims (micro-expressions are not reliable lie detectors)

### 5.3 Deployment Checklist

- [ ] Consent mechanism enabled and tested
- [ ] No raw frames stored to disk
- [ ] Session logs configured with appropriate retention
- [ ] If serving via API: TLS enabled, auth configured, rate limiting active
- [ ] Privacy policy/notice provided to end users
- [ ] Data processing agreement in place (if processing third-party data)

---

## 6. Incident Response

If a data breach involving biometric data is suspected:

1. **Contain**: Stop all processing, preserve logs
2. **Assess**: Determine scope (which sessions, which data types)
3. **Notify**: GDPR requires notification within 72 hours
4. **Remediate**: Purge affected session data, rotate API keys if service-deployed
5. **Document**: Record incident details, root cause, and corrective actions
