"""
Modular Loss Functions
Core Stack: Pure NumPy (Zero PyTorch, TensorFlow, or autograd libraries)
"""

import numpy as np


class Loss:
    """Base interface for loss functions."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)


class MSELoss(Loss):
    """
    Mean Squared Error Loss:
    L = (1 / N) * sum((y_pred - y_true)^2)
    dL/dy_pred = (2 / N) * (y_pred - y_true)
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred.astype(np.float64)
        self.y_true = y_true.astype(np.float64)
        n = self.y_pred.shape[0]
        loss_val = np.sum(np.square(self.y_pred - self.y_true)) / n
        return float(loss_val)

    def backward(self) -> np.ndarray:
        n = self.y_pred.shape[0]
        return (2.0 / n) * (self.y_pred - self.y_true)

    def __repr__(self):
        return "MSELoss()"


class BCELoss(Loss):
    """
    Binary Cross-Entropy Loss with Numerical Stability Epsilon:
    L = - (1 / N) * sum( y * log(p) + (1 - y) * log(1 - p) )
    dL/dp = (1 / N) * [ (p - y) / (p * (1 - p)) ]
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps
        self.y_pred = None
        self.y_true = None
        self.y_pred_clipped = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred.astype(np.float64)
        self.y_true = y_true.astype(np.float64)
        n = self.y_pred.shape[0]

        # Clip predictions to prevent log(0) and division by zero
        self.y_pred_clipped = np.clip(self.y_pred, self.eps, 1.0 - self.eps)

        loss_terms = self.y_true * np.log(self.y_pred_clipped) + (1.0 - self.y_true) * np.log(1.0 - self.y_pred_clipped)
        loss_val = -np.sum(loss_terms) / n
        return float(loss_val)

    def backward(self) -> np.ndarray:
        n = self.y_pred.shape[0]
        p = self.y_pred_clipped
        y = self.y_true
        denom = np.maximum(p * (1.0 - p), self.eps)
        grad = (1.0 / n) * ((p - y) / denom)
        return grad

    def __repr__(self):
        return f"BCELoss(eps={self.eps})"
