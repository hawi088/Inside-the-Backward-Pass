"""
Modular Neural Engine - Base Layers & Activations
Core Stack: Pure NumPy (Zero PyTorch, TensorFlow, or autograd libraries)
"""

import numpy as np


class Layer:
    """Base abstract interface for all neural network layers."""

    def __init__(self):
        self.trainable = False
        self._input_cache = None
        self._output_cache = None
        self._grad_cache = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self):
        """Returns list of tuples: (param_name, param_array, grad_array)"""
        return []

    def zero_grad(self):
        """Reset parameter gradients to zero."""
        pass


class Linear(Layer):
    """
    Fully Connected (Affine) Transformation Layer: Z = X @ W + b.
    Tracks weights W, bias b, and gradients dW, db.
    Supports He (Kaiming) and Xavier (Glorot) normal initialization.
    """

    def __init__(self, in_features: int, out_features: int, init: str = "he", seed: int = None):
        super().__init__()
        self.trainable = True
        self.in_features = in_features
        self.out_features = out_features
        self.init_type = init.lower()

        rng = np.random.default_rng(seed)

        # Weight initialization
        if self.init_type in ("he", "kaiming"):
            # He normal: std = sqrt(2 / in_features)
            std = np.sqrt(2.0 / in_features)
        elif self.init_type in ("xavier", "glorot"):
            # Xavier normal: std = sqrt(2 / (in_features + out_features))
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            std = 0.01

        self.W = rng.normal(loc=0.0, scale=std, size=(in_features, out_features)).astype(np.float64)
        self.b = np.zeros((1, out_features), dtype=np.float64)

        # Gradient buffers
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward Pass:
        Z = X @ W + b
        """
        self._input_cache = inputs.astype(np.float64)
        self._output_cache = np.matmul(self._input_cache, self.W) + self.b
        return self._output_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward Pass:
        Given dZ (grad_output):
        dW = X^T @ dZ
        db = sum(dZ, axis=0, keepdims=True)
        dX = dZ @ W^T
        """
        dZ = grad_output.astype(np.float64)
        self._grad_cache = dZ

        # Gradients wrt parameters
        self.dW = np.matmul(self._input_cache.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # Gradient wrt inputs for preceding layer
        dX = np.matmul(dZ, self.W.T)
        return dX

    def get_params(self):
        return [
            ("W", self.W, self.dW),
            ("b", self.b, self.db),
        ]

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, init='{self.init_type}')"


class Sigmoid(Layer):
    """
    Sigmoid Activation Layer:
    A = 1 / (1 + exp(-Z))
    dA/dZ = A * (1 - A)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_cache = inputs.astype(np.float64)
        # Numerical stability clip to prevent float overflow
        clipped = np.clip(self._input_cache, -60.0, 60.0)
        self._output_cache = 1.0 / (1.0 + np.exp(-clipped))
        return self._output_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        dZ = dA * sigma'(Z) = dA * (A * (1 - A))
        """
        A = self._output_cache
        grad_activation = A * (1.0 - A)
        dZ = grad_output * grad_activation
        self._grad_cache = dZ
        return dZ

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Layer):
    """
    Hyperbolic Tangent Activation Layer:
    A = tanh(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))
    dA/dZ = 1 - A^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_cache = inputs.astype(np.float64)
        self._output_cache = np.tanh(self._input_cache)
        return self._output_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        dZ = dA * (1 - A^2)
        """
        A = self._output_cache
        grad_activation = 1.0 - np.square(A)
        dZ = grad_output * grad_activation
        self._grad_cache = dZ
        return dZ

    def __repr__(self):
        return "Tanh()"


class ReLU(Layer):
    """
    Rectified Linear Unit Activation Layer:
    A = max(0, Z)
    dA/dZ = 1 if Z > 0 else 0
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_cache = inputs.astype(np.float64)
        self._output_cache = np.maximum(0.0, self._input_cache)
        return self._output_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        dZ = dA * (Z > 0)
        """
        grad_activation = (self._input_cache > 0.0).astype(np.float64)
        dZ = grad_output * grad_activation
        self._grad_cache = dZ
        return dZ

    def __repr__(self):
        return "ReLU()"
