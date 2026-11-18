"""
Custom Optimizers from Scratch
Core Stack: Pure NumPy (Zero PyTorch, TensorFlow, or autograd libraries)

Module 4 Implementation:
- Standard SGD
- SGD with Momentum (velocity buffers)
- Adam (Adaptive Moment Estimation with bias corrections)
"""

import numpy as np


class Optimizer:
    """Base class for all first-principles optimizers."""

    def __init__(self, params, lr: float = 0.01):
        """
        Parameters:
        -----------
        params : list of Layer or list of (name, param, grad)
        lr : float
            Learning rate (alpha).
        """
        self.params = params
        self.lr = lr
        self.t = 0  # Timestep counter

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for layer in self.params:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()


class SGD(Optimizer):
    """
    Standard Stochastic Gradient Descent:
    theta = theta - alpha * grad(theta)
    """

    def __init__(self, params, lr: float = 0.01):
        super().__init__(params, lr)

    def step(self):
        self.t += 1
        for layer in self.params:
            if hasattr(layer, "get_params"):
                for name, p, grad in layer.get_params():
                    p -= self.lr * grad

    def __repr__(self):
        return f"SGD(lr={self.lr})"


class SGDMomentum(Optimizer):
    """
    SGD with Classical Momentum:
    Maintains velocity buffer v to barrel through flat saddles and damp oscillations.
    v = beta * v - alpha * grad(theta)
    theta = theta + v
    """

    def __init__(self, params, lr: float = 0.01, beta: float = 0.9):
        super().__init__(params, lr)
        self.beta = beta
        # Initialize velocity buffers matching parameter shapes
        self.velocities = []
        for layer in self.params:
            if hasattr(layer, "get_params"):
                layer_vels = {}
                for name, p, _ in layer.get_params():
                    layer_vels[name] = np.zeros_like(p)
                self.velocities.append(layer_vels)
            else:
                self.velocities.append({})

    def step(self):
        self.t += 1
        for layer_idx, layer in enumerate(self.params):
            if hasattr(layer, "get_params"):
                layer_vels = self.velocities[layer_idx]
                for name, p, grad in layer.get_params():
                    v = layer_vels[name]
                    # v = beta * v - alpha * grad
                    v = self.beta * v - self.lr * grad
                    layer_vels[name] = v
                    p += v

    def __repr__(self):
        return f"SGDMomentum(lr={self.lr}, beta={self.beta})"


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation):
    Combines momentum (first moment m) and RMSProp (second raw moment v)
    with bias-correction terms to accelerate convergence across ill-conditioned ravines.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2
    m_hat_t = m_t / (1 - beta1^t)
    v_hat_t = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - (alpha / (sqrt(v_hat_t) + eps)) * m_hat_t
    """

    def __init__(self, params, lr: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # First and second moment buffers
        self.m_buffers = []
        self.v_buffers = []

        for layer in self.params:
            if hasattr(layer, "get_params"):
                m_layer = {}
                v_layer = {}
                for name, p, _ in layer.get_params():
                    m_layer[name] = np.zeros_like(p)
                    v_layer[name] = np.zeros_like(p)
                self.m_buffers.append(m_layer)
                self.v_buffers.append(v_layer)
            else:
                self.m_buffers.append({})
                self.v_buffers.append({})

    def step(self):
        self.t += 1
        # Precompute bias correction divisors
        bias_corr1 = 1.0 - (self.beta1 ** self.t)
        bias_corr2 = 1.0 - (self.beta2 ** self.t)

        for layer_idx, layer in enumerate(self.params):
            if hasattr(layer, "get_params"):
                m_layer = self.m_buffers[layer_idx]
                v_layer = self.v_buffers[layer_idx]

                for name, p, grad in layer.get_params():
                    m = m_layer[name]
                    v = v_layer[name]

                    # Update biased first and second moments
                    m = self.beta1 * m + (1.0 - self.beta1) * grad
                    v = self.beta2 * v + (1.0 - self.beta2) * np.square(grad)

                    m_layer[name] = m
                    v_layer[name] = v

                    # Compute bias-corrected moments
                    m_hat = m / bias_corr1
                    v_hat = v / bias_corr2

                    # Parameter update with epsilon scaling
                    update = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
                    p -= update

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"
