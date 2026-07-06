# Model Card — Micro-Expression Detection System

> **Model Version**: 1.0 (MVP — SVM Baseline)  
> **Last Updated**: 2026-07-05

---

## Model Details

| Field | Value |
|-------|-------|
| **Model Type** | Support Vector Machine (RBF kernel) |
| **Input** | Optical flow feature vectors (concatenated ROI histograms) |
| **Output** | Emotion class + confidence score |
| **Feature Dimension** | Variable (depends on ROI count × histogram bins) |
| **Training Framework** | scikit-learn 1.3+ |
| **Inference Latency** | <1ms per prediction (CPU) |

---

## Intended Use

### Primary Use Cases
- Academic research on micro-expression recognition
- Benchmarking against CASME II / SAMM / SMIC datasets
- Prototyping real-time facial analysis pipelines

### Out-of-Scope Use Cases
- Deception / lie detection (micro-expressions are not reliable indicators)
- Clinical diagnosis of emotional disorders
- Covert surveillance or monitoring
- High-stakes decision-making (hiring, legal proceedings)

---

## Training Data

| Dataset | Samples Used | Classes | Cross-Validation |
|---------|-------------|---------|-------------------|
| CASME II | Up to 247 | happiness, surprise, disgust, repression, others | LOSO (26 subjects) |
| SAMM | Up to 159 | happiness, surprise, anger, contempt, fear, sadness, disgust | LOSO (32 subjects) |

### Data Limitations
- Very small sample sizes (100-250 total)
- Significant class imbalance (surprise/disgust overrepresented)
- Lab-controlled conditions only (consistent lighting, frontal pose)
- Limited ethnic/demographic diversity
- Elicited (not spontaneous) expressions in some cases

---

## Evaluation Results

### Expected Performance Range (SVM Baseline)

| Metric | CASME II | SAMM | Notes |
|--------|----------|------|-------|
| **UF1 (Unweighted F1)** | 0.55 – 0.65 | 0.45 – 0.55 | Macro-averaged across classes |
| **UAR (Unweighted Avg Recall)** | 0.50 – 0.60 | 0.40 – 0.50 | Mean per-class recall |
| **Accuracy** | 0.60 – 0.70 | 0.50 – 0.60 | Misleading due to class imbalance |

### Known Weaknesses
- Poor discrimination between fear and surprise (similar AU activation patterns)
- Contempt class has very few training samples → low recall
- Performance degrades significantly outside lab conditions

---

## Ethical Considerations

- **Bias**: Training data primarily from East Asian (CASME II) and European (SAMM) subjects. Performance on other demographics is untested and likely degraded.
- **Fairness**: No fairness auditing has been performed across demographic groups.
- **Privacy**: See [SECURITY.md](./SECURITY.md) for biometric data handling policies.
- **Misuse potential**: Emotion classification from facial expressions is inherently noisy. Results should never be treated as ground truth for any individual's emotional state.

---

## Limitations and Recommendations

1. **Do not deploy for deception detection** — the scientific basis for micro-expression-based lie detection is disputed
2. **Expect high variance** — small datasets and LOSO-CV produce high variance in metrics across folds
3. **Validate on your target population** — if deploying for a specific demographic, collect and validate on representative data
4. **Use confidence thresholds** — filter predictions below 0.6 confidence for any downstream application
5. **Human-in-the-loop** — always pair automated predictions with human review for any consequential use
