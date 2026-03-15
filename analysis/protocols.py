"""Protocol definitions for model interfaces to enable dependency inversion.

This module defines abstract protocols that models must conform to, enabling
loose coupling and easy swapping of implementations. Following SOLID principles,
high-level logic depends on these abstractions, not concrete implementations.

Example:
    >>> from data_utils.model_protocols import Encoder, Decoder
    >>> 
    >>> # Any class conforming to the protocol can be used
    >>> class MyCustomGLM:
    >>>     def fit(self, X, y): ...
    >>>     def predict(self, X): ...
    >>>     def score(self, X, y): ...
    >>> 
    >>> encoder: Encoder = MyCustomGLM()  # Works if it matches protocol
"""

from typing import Protocol, Any, Optional, Tuple, Dict
import numpy as np


class Encoder(Protocol):
    """Protocol for neural encoding models (GLM, etc.).
    
    Any model implementing these methods can be used for encoding,
    enabling easy swapping between Poisson GLM, negative binomial GLM,
    linear models, or even deep learning approaches.
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Encoder":
        """Fit the encoding model.
        
        Args:
            X: Design matrix of shape (n_samples, n_features).
            y: Target neural activity of shape (n_samples,).
            
        Returns:
            Self for method chaining.
        """
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict neural activity.
        
        Args:
            X: Design matrix of shape (n_samples, n_features).
            
        Returns:
            Predicted activity of shape (n_samples,).
        """
        ...
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model performance score (e.g., D² or R²).
        
        Args:
            X: Design matrix of shape (n_samples, n_features).
            y: True neural activity of shape (n_samples,).
            
        Returns:
            Performance score.
        """
        ...
    
    @property
    def coef_(self) -> np.ndarray:
        """Model coefficients."""
        ...
    
    @property
    def intercept_(self) -> float:
        """Model intercept."""
        ...


class Decoder(Protocol):
    """Protocol for behavioral decoding models (SVM, etc.).
    
    Any classifier implementing these methods can be used for decoding,
    including SVM, logistic regression, random forests, or neural networks.
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Decoder":
        """Fit the decoder model.
        
        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Behavioral labels of shape (n_samples,).
            
        Returns:
            Self for method chaining.
        """
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict behavioral states.
        
        Args:
            X: Neural features of shape (n_samples, n_features).
            
        Returns:
            Predicted labels of shape (n_samples,).
        """
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Neural features of shape (n_samples, n_features).
            
        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        ...
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.
        
        Args:
            X: Neural features of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).
            
        Returns:
            Accuracy score.
        """
        ...


class ModelFactory(Protocol):
    """Protocol for model factories enabling dependency injection.
    
    Factories create models with specific configurations, allowing
    the high-level code to remain agnostic about concrete implementations.
    """
    
    def create_encoder(self, **kwargs) -> Encoder:
        """Create an encoder model with given parameters.
        
        Args:
            **kwargs: Model-specific parameters.
            
        Returns:
            Encoder instance.
        """
        ...
    
    def create_decoder(self, **kwargs) -> Decoder:
        """Create a decoder model with given parameters.
        
        Args:
            **kwargs: Model-specific parameters.
            
        Returns:
            Decoder instance.
        """
        ...


class EncoderFactory:
    """Concrete factory for creating encoding models."""
    
    def __init__(self, model_type: str = "poisson_glm"):
        """Initialize factory with default model type.
        
        Args:
            model_type: Type of encoder ('poisson_glm', 'linear', 'neural_net').
        """
        self.model_type = model_type
    
    def create_encoder(self, **kwargs) -> Encoder:
        """Create encoder based on configured type.
        
        Args:
            **kwargs: Model parameters (e.g., alpha for regularization).
            
        Returns:
            Encoder instance.
        """
        if self.model_type == "poisson_glm":
            from . import gpu_utils
            gpu_utils.initialize(use_gpu=False)  # Initialize if needed
            
            # Extract relevant parameters
            alpha = kwargs.get('alpha', 1.0)
            max_iter = kwargs.get('max_iter', 1000)
            
            return gpu_utils.PoissonRegressor(
                alpha=alpha,
                max_iter=max_iter
            )
        
        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge
            alpha = kwargs.get('alpha', 1.0)
            return Ridge(alpha=alpha)
        
        elif self.model_type == "neural_net":
            # Placeholder for neural network implementation
            raise NotImplementedError("Neural network encoder not yet implemented")
        
        else:
            raise ValueError(f"Unknown encoder type: {self.model_type}")


class DecoderFactory:
    """Concrete factory for creating decoding models."""
    
    def __init__(self, model_type: str = "svm"):
        """Initialize factory with default model type.
        
        Args:
            model_type: Type of decoder ('svm', 'logistic', 'random_forest').
        """
        self.model_type = model_type
    
    def create_decoder(self, **kwargs) -> Decoder:
        """Create decoder based on configured type.
        
        Args:
            **kwargs: Model parameters (e.g., C for SVM regularization).
            
        Returns:
            Decoder instance.
        """
        if self.model_type == "svm":
            from . import gpu_utils
            gpu_utils.initialize(use_gpu=False)  # Initialize if needed
            
            # Extract relevant parameters
            class_weight = kwargs.get('class_weight', 'balanced')
            max_iter = kwargs.get('max_iter', 2000)
            C = kwargs.get('C', 1.0)
            
            return gpu_utils.LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter
            )
        
        elif self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            
            class_weight = kwargs.get('class_weight', 'balanced')
            max_iter = kwargs.get('max_iter', 1000)
            C = kwargs.get('C', 1.0)
            
            return LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                multi_class='ovr'
            )
        
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            class_weight = kwargs.get('class_weight', 'balanced')
            
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight
            )
        
        else:
            raise ValueError(f"Unknown decoder type: {self.model_type}")


def get_default_encoder_factory() -> EncoderFactory:
    """Get the default encoder factory.
    
    Returns:
        EncoderFactory configured with default settings.
    """
    return EncoderFactory(model_type="poisson_glm")


def get_default_decoder_factory() -> DecoderFactory:
    """Get the default decoder factory.
    
    Returns:
        DecoderFactory configured with default settings.
    """
    return DecoderFactory(model_type="svm")