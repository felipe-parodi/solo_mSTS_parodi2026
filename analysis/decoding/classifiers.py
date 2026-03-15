"""Classifier wrappers for neural decoding.

Provides consistent interface for LogisticRegression and LinearSVM.
Both are recommended for neural decoding due to interpretability.

Usage:
    from analysis.decoding.classifiers import LogisticClassifier, SVMClassifier

    clf = LogisticClassifier(C=1.0)
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    print(result.y_pred, result.y_prob)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class ClassifierResult:
    """Result container for classifier predictions."""

    y_pred: np.ndarray  # Predicted class labels
    y_prob: Optional[np.ndarray]  # Class probabilities (n_samples, n_classes)
    classes: np.ndarray  # Class labels in order


@dataclass
class ClassifierConfig:
    """Configuration for classifiers."""

    C: float = 1.0  # Regularization (inverse of strength)
    class_weight: str = "balanced"  # Handle class imbalance
    max_iter: int = 1000  # Max iterations for convergence
    random_state: int = 42


class LogisticClassifier:
    """Multinomial logistic regression with L2 regularization.

    Preferred for decoding because:
    - Gives class probabilities (useful for SPE, uncertainty)
    - Handles multi-class naturally (softmax)
    - Balanced class weights for imbalanced data
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
        standardize: bool = True,
    ):
        """Initialize logistic classifier.

        Args:
            C: Regularization parameter (higher = less regularization).
            class_weight: 'balanced' to handle class imbalance.
            max_iter: Maximum iterations for solver.
            random_state: Random seed.
            standardize: Whether to standardize features internally.
        """
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.standardize = standardize

        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        """Fit classifier on training data.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Labels, shape (n_samples,). Can be strings or integers.

        Returns:
            Self for chaining.
        """
        # Encode labels if needed
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Standardize features
        if self.standardize:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit model (multinomial is default in sklearn >= 1.0)
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> ClassifierResult:
        """Predict class labels and probabilities.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            ClassifierResult with predictions and probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        # Apply same scaling
        if self.standardize and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Predict
        y_pred_encoded = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)

        # Decode labels back to original
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return ClassifierResult(
            y_pred=y_pred,
            y_prob=y_prob,
            classes=self.label_encoder.classes_,
        )

    def get_coefficients(self) -> Optional[np.ndarray]:
        """Get model coefficients (for interpretability)."""
        if self.model is None:
            return None
        return self.model.coef_


class SVMClassifier:
    """Linear SVM for multi-class classification.

    Uses Crammer-Singer formulation (one-vs-one internally).
    Good fallback if LogisticRegression is unstable.
    Does not provide class probabilities.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 2000,
        random_state: int = 42,
        standardize: bool = True,
    ):
        """Initialize SVM classifier.

        Args:
            C: Regularization parameter.
            class_weight: 'balanced' to handle class imbalance.
            max_iter: Maximum iterations.
            random_state: Random seed.
            standardize: Whether to standardize features.
        """
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.standardize = standardize

        self.model: Optional[LinearSVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Fit classifier on training data.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Labels, shape (n_samples,).

        Returns:
            Self for chaining.
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Standardize
        if self.standardize:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit model
        self.model = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual="auto",
        )
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> ClassifierResult:
        """Predict class labels.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            ClassifierResult with predictions (no probabilities for SVM).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        if self.standardize and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return ClassifierResult(
            y_pred=y_pred,
            y_prob=None,  # SVM doesn't provide probabilities
            classes=self.label_encoder.classes_,
        )

    def get_coefficients(self) -> Optional[np.ndarray]:
        """Get model coefficients."""
        if self.model is None:
            return None
        return self.model.coef_


def create_classifier(
    classifier_type: str = "logistic",
    **kwargs,
) -> LogisticClassifier | SVMClassifier:
    """Factory function to create classifier by name.

    Args:
        classifier_type: 'logistic' or 'svm'.
        **kwargs: Passed to classifier constructor.

    Returns:
        Classifier instance.
    """
    if classifier_type.lower() in ("logistic", "logreg", "lr"):
        return LogisticClassifier(**kwargs)
    elif classifier_type.lower() in ("svm", "linearsvc"):
        return SVMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
