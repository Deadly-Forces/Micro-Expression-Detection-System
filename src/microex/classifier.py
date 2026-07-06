"""
Trainable emotion classifier with model persistence for the Micro-Expression
Detection System.

Supports SVM (scikit-learn, CPU-only) out of the box, with optional LSTM/CNN
paths gated behind a ``torch`` import guard.  Every public method carries full
type-hints, input validation, and production-grade error handling.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Optional deep-learning imports — guarded so the module works CPU-only
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE: bool = False
try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Default emotion labels (FACS 7-class) ──────────────────────────────────
_DEFAULT_EMOTIONS: List[str] = [
    "happiness",
    "sadness",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "contempt",
]


# ── Data-classes ────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    """Structured container for a single emotion prediction."""

    label: str  # predicted emotion
    confidence: float  # probability / decision-function score
    all_scores: Dict[str, float]  # scores for every class
    model_type: str  # 'svm', 'lstm', 'cnn'


# ── Classifier ──────────────────────────────────────────────────────────────
class EmotionClassifier:
    """Trainable emotion classifier supporting SVM (default) with optional
    LSTM / CNN backends.

    Parameters
    ----------
    model_type : str
        One of ``'svm'``, ``'lstm'``, ``'cnn'``.
    emotion_labels : list[str] or None
        Class names.  Falls back to the 7-class FACS set.
    """

    _VALID_MODEL_TYPES = {"svm", "lstm", "cnn"}

    def __init__(
        self,
        model_type: str = "svm",
        emotion_labels: Optional[List[str]] = None,
    ) -> None:
        if model_type not in self._VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self._VALID_MODEL_TYPES}, "
                f"got '{model_type}'"
            )
        if model_type in ("lstm", "cnn") and not _TORCH_AVAILABLE:
            raise ImportError(
                f"model_type='{model_type}' requires PyTorch.  "
                "Install it with `pip install torch`."
            )

        self.model_type: str = model_type
        self.emotion_labels: List[str] = list(emotion_labels or _DEFAULT_EMOTIONS)

        # Internal state — populated by train() or load_model()
        self._model: Any = None
        self._scaler: Optional[StandardScaler] = None
        self._is_trained: bool = False
        self._feature_dim: Optional[int] = None

    # ── helpers ──────────────────────────────────────────────────────────
    @property
    def is_trained(self) -> bool:
        """Whether the classifier has a fitted model."""
        return self._is_trained

    def _validate_features(
        self, features: np.ndarray, *, allow_single: bool = False
    ) -> np.ndarray:
        """Ensure *features* is a 2-D float array with the right width."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1 and allow_single:
            features = features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError(
                f"Expected 2-D feature array, got shape {features.shape}"
            )
        if self._feature_dim is not None and features.shape[1] != self._feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects "
                f"{self._feature_dim}, got {features.shape[1]}"
            )
        return features

    # ── training ─────────────────────────────────────────────────────────
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Train the classifier.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_samples, n_features)``.
        labels : np.ndarray
            Shape ``(n_samples,)`` — integer or string class labels.

        Returns
        -------
        dict[str, float]
            Training metrics including ``'accuracy'`` and optionally
            ``'val_accuracy'``.

        Raises
        ------
        ValueError
            If shapes are inconsistent or a non-SVM backend is requested
            without PyTorch.
        """
        features = self._validate_features(features)
        labels = np.asarray(labels)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features ({features.shape[0]}) and labels "
                f"({labels.shape[0]}) length mismatch"
            )

        self._feature_dim = features.shape[1]

        if self.model_type == "svm":
            return self._train_svm(features, labels)

        # Placeholder for future LSTM / CNN (torch already validated)
        raise NotImplementedError(
            f"Training for model_type='{self.model_type}' is not yet "
            "implemented."
        )

    def _train_svm(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Fit StandardScaler + RBF-SVM pipeline with GridSearchCV."""
        from sklearn.model_selection import GridSearchCV
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(features)

        # Base model
        base_svm = SVC(kernel="rbf", probability=True, random_state=42)

        # Parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        }

        # Grid search
        # If there are fewer than 3 samples in the least populated class, cv=3 might fail for StratifiedKFold
        # Fallback to fewer splits if necessary.
        unique, counts = np.unique(labels, return_counts=True)
        cv_splits = min(3, counts.min())
        if cv_splits < 2:
            logger.warning("Not enough samples per class for GridSearchCV. Falling back to single fit.")
            self._model = base_svm
            self._model.fit(X_scaled, labels)
            metrics = {"accuracy": float(accuracy_score(labels, self._model.predict(X_scaled)))}
        else:
            grid_search = GridSearchCV(
                base_svm, param_grid, cv=cv_splits, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_scaled, labels)
            self._model = grid_search.best_estimator_
            
            best_gamma = grid_search.best_params_['gamma']
            metrics = {
                "accuracy": float(grid_search.best_score_),
                "best_C": float(grid_search.best_params_['C']),
                "best_gamma": best_gamma if isinstance(best_gamma, str) else float(best_gamma)
            }

        self._is_trained = True
        logger.info("SVM training complete — %s", metrics)
        return metrics

    # ── inference ─────────────────────────────────────────────────────────
    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """Predict the emotion label for a single feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray
            Shape ``(n_features,)`` or ``(1, n_features)``.

        Returns
        -------
        PredictionResult

        Raises
        ------
        RuntimeError
            If the model has not been trained / loaded.
        ValueError
            If the feature dimension does not match.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained.  Call train() or load_model() first."
            )

        feature_vector = self._validate_features(
            feature_vector, allow_single=True
        )

        if self.model_type == "svm":
            return self._predict_svm(feature_vector)

        raise NotImplementedError(
            f"Prediction for model_type='{self.model_type}' is not yet "
            "implemented."
        )

    def _predict_svm(self, X: np.ndarray) -> PredictionResult:
        """Run SVM prediction on a single (already-validated) sample."""
        assert self._scaler is not None and self._model is not None  # noqa: S101
        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)[0]

        classes = list(self._model.classes_)
        all_scores: Dict[str, float] = {
            str(cls): float(p) for cls, p in zip(classes, proba)
        }
        best_idx = int(np.argmax(proba))
        label = str(classes[best_idx])
        confidence = float(proba[best_idx])

        return PredictionResult(
            label=label,
            confidence=confidence,
            all_scores=all_scores,
            model_type=self.model_type,
        )

    # ── persistence ──────────────────────────────────────────────────────
    def save_model(self, path: str) -> None:
        """Serialize the trained model, scaler, and metadata to *path*.

        Uses :mod:`pickle` (standard-library) so ``joblib`` is not required.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save — model is not trained.")

        payload = {
            "model": self._model,
            "scaler": self._scaler,
            "model_type": self.model_type,
            "emotion_labels": self.emotion_labels,
            "feature_dim": self._feature_dim,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load a previously-saved model from *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the payload is corrupted or incompatible.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301

        required_keys = {"model", "scaler", "model_type", "emotion_labels", "feature_dim"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(
                f"Model file is missing keys: {missing}"
            )

        self._model = payload["model"]
        self._scaler = payload["scaler"]
        self.model_type = payload["model_type"]
        self.emotion_labels = payload["emotion_labels"]
        self._feature_dim = payload["feature_dim"]
        self._is_trained = True
        logger.info("Model loaded from %s", path)

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate the classifier on held-out data.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``per_class`` (precision / recall / f1 per
            class), ``confusion_matrix``, ``uf1``, ``uar``.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained.  Call train() or load_model() first."
            )
        features = self._validate_features(features)
        labels = np.asarray(labels)

        preds = np.array(
            [self.predict(features[i]).label for i in range(features.shape[0])]
        )

        # Force string comparison
        labels_str = np.array([str(l) for l in labels])

        acc = float(accuracy_score(labels_str, preds))
        uf1 = self._compute_uf1(labels_str, preds)
        uar = self._compute_uar(labels_str, preds)
        cm = confusion_matrix(labels_str, preds).tolist()

        # Per-class metrics
        unique_labels = sorted(set(labels_str) | set(preds))
        prec, rec, f1, sup = precision_recall_fscore_support(
            labels_str, preds, labels=unique_labels, zero_division=0
        )
        per_class: Dict[str, Dict[str, float]] = {}
        for idx, lbl in enumerate(unique_labels):
            per_class[lbl] = {
                "precision": float(prec[idx]),
                "recall": float(rec[idx]),
                "f1": float(f1[idx]),
                "support": int(sup[idx]),
            }

        return {
            "accuracy": acc,
            "uf1": uf1,
            "uar": uar,
            "per_class": per_class,
            "confusion_matrix": cm,
        }

    # ── metric helpers ───────────────────────────────────────────────────
    @staticmethod
    def _compute_uf1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unweighted F1 — macro-averaged F1 across all classes."""
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    @staticmethod
    def _compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unweighted Average Recall — macro-averaged recall."""
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════
# Trial block
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import shutil
    import sys

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
    print("EmotionClassifier — Trial Block")
    print("=" * 60)

    # ── 1. Synthetic data (separable) ────────────────────────────────────
    rng = np.random.RandomState(42)
    n_per_class = 25
    n_features = 32
    n_classes = 4
    class_labels = [str(i) for i in range(n_classes)]

    X_blocks, y_blocks = [], []
    for cls_idx in range(n_classes):
        mean = rng.randn(n_features) * 3 + cls_idx * 4
        X_blocks.append(mean + rng.randn(n_per_class, n_features) * 0.5)
        y_blocks.append(np.full(n_per_class, str(cls_idx)))

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)

    _report("Synthetic data generation", X.shape == (100, 32) and y.shape == (100,),
            f"X={X.shape}, y={y.shape}")

    # ── 2. Train SVM ─────────────────────────────────────────────────────
    clf = EmotionClassifier(model_type="svm", emotion_labels=class_labels)
    try:
        metrics = clf.train(X, y)
        train_ok = "accuracy" in metrics and metrics["accuracy"] > 0
        _report("SVM training", train_ok, f"metrics={metrics}")
    except Exception as exc:
        _report("SVM training", False, str(exc))

    # ── 3. Predict on a sample ───────────────────────────────────────────
    try:
        result = clf.predict(X[0])
        struct_ok = (
            isinstance(result.label, str)
            and isinstance(result.confidence, float)
            and 0.0 <= result.confidence <= 1.0
            and isinstance(result.all_scores, dict)
            and result.model_type == "svm"
        )
        _report(
            "Prediction structure",
            struct_ok,
            f"label={result.label!r}, conf={result.confidence:.4f}, "
            f"model_type={result.model_type!r}",
        )
    except Exception as exc:
        _report("Prediction structure", False, str(exc))

    # ── 4. Save → load → re-predict ────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="microex_trial_")
    model_path = os.path.join(tmp_dir, "model.pkl")
    try:
        clf.save_model(model_path)
        clf2 = EmotionClassifier(model_type="svm")
        clf2.load_model(model_path)
        result2 = clf2.predict(X[0])
        same = result.label == result2.label and np.isclose(
            result.confidence, result2.confidence, atol=1e-9
        )
        _report(
            "Save / load round-trip",
            same,
            f"original={result.label}/{result.confidence:.6f}, "
            f"reloaded={result2.label}/{result2.confidence:.6f}",
        )
    except Exception as exc:
        _report("Save / load round-trip", False, str(exc))

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    try:
        eval_metrics = clf.evaluate(X, y)
        expected_keys = {"accuracy", "uf1", "uar", "per_class", "confusion_matrix"}
        keys_ok = expected_keys.issubset(eval_metrics.keys())
        _report(
            "evaluate() keys",
            keys_ok,
            f"keys={set(eval_metrics.keys())}",
        )
        _report(
            "evaluate() accuracy > 0",
            eval_metrics["accuracy"] > 0,
            f"accuracy={eval_metrics['accuracy']:.4f}, "
            f"uf1={eval_metrics['uf1']:.4f}, uar={eval_metrics['uar']:.4f}",
        )
    except Exception as exc:
        _report("evaluate()", False, str(exc))

    # ── 6. Error: predict before training ────────────────────────────────
    try:
        fresh = EmotionClassifier(model_type="svm")
        fresh.predict(X[0])
        _report("RuntimeError on untrained predict", False, "No error raised")
    except RuntimeError:
        _report("RuntimeError on untrained predict", True)
    except Exception as exc:
        _report("RuntimeError on untrained predict", False, f"Wrong error: {exc}")

    # ── 7. Error: wrong feature dimension ────────────────────────────────
    try:
        bad_vec = np.zeros(999)
        clf.predict(bad_vec)
        _report("ValueError on wrong dim", False, "No error raised")
    except ValueError:
        _report("ValueError on wrong dim", True)
    except Exception as exc:
        _report("ValueError on wrong dim", False, f"Wrong error: {exc}")

    # ── Cleanup ──────────────────────────────────────────────────────────
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests PASSED.")
        sys.exit(0)
