"""
Modular Neural Engine - Sequential Network Container & PyTorch-Style Summary
Core Stack: Pure NumPy (Zero PyTorch, TensorFlow, or autograd libraries)
"""

from typing import List, Tuple
import numpy as np
from .layers import Layer, Linear, Sigmoid, Tanh, ReLU


class Sequential:
    """
    Sequential Container for Neural Network Architectures.
    Executes forward passes left-to-right and backward passes right-to-left.
    """

    def __init__(self, layers: List[Layer] = None):
        self.layers = list(layers) if layers is not None else []

    def add(self, layer: Layer):
        """Append a layer to the network sequence."""
        self.layers.append(layer)
        return self

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Pass (Left-to-Right):
        X -> Layer 0 -> Layer 1 -> ... -> Layer L-1 -> Output
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward Pass (Right-to-Left):
        dL/dOut -> Layer L-1 -> Layer L-2 -> ... -> Layer 0 -> dL/dX
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        """Reset parameter gradients across all layers."""
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self) -> List[Layer]:
        """Return list of trainable layers with parameters."""
        return [layer for layer in self.layers if layer.trainable]

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (trainable_params, non_trainable_params)."""
        trainable_count = 0
        for layer in self.layers:
            if layer.trainable:
                for _, param, _ in layer.get_params():
                    trainable_count += param.size
        return trainable_count, 0

    def summary(self, input_shape: Tuple[int, ...] = (1, 2), verbose: bool = True) -> str:
        """
        PyTorch / TorchSummary style model inspection tool.
        Traces dimensions through the computational graph, computing shapes and parameter counts.
        """
        # Header formatting
        line_length = 90
        sep_double = "=" * line_length
        sep_single = "-" * line_length

        lines = [
            sep_double,
            f"{'Layer (type)':<30} {'Input Shape':<22} {'Output Shape':<22} {'Param #':<12}",
            sep_double,
        ]

        # Dummy tensor to trace shapes
        if len(input_shape) == 1:
            dummy_shape = (1, input_shape[0])
        else:
            dummy_shape = input_shape

        current_x = np.zeros(dummy_shape, dtype=np.float64)
        total_params = 0
        total_mem_bytes = current_x.nbytes

        for idx, layer in enumerate(self.layers):
            in_shape_str = str(current_x.shape).replace("1,", "None,")
            
            # Layer forward shape evaluation
            # Clone input to avoid mutating real cache
            temp_x = np.copy(current_x)
            out = layer.forward(temp_x)
            out_shape_str = str(out.shape).replace("1,", "None,")

            # Count params
            layer_params = 0
            if layer.trainable:
                for _, param, _ in layer.get_params():
                    layer_params += param.size
                    total_mem_bytes += param.nbytes * 2  # Param + Grad

            total_params += layer_params
            total_mem_bytes += out.nbytes * 2  # Forward activation + backward grad

            layer_name = f"{idx}: {layer.__class__.__name__}"
            lines.append(f"{layer_name:<30} {in_shape_str:<22} {out_shape_str:<22} {layer_params:<12,}")
            current_x = out

        lines.append(sep_double)
        lines.append(f"Total params: {total_params:,}")
        lines.append(f"Trainable params: {total_params:,}")
        lines.append(f"Non-trainable params: 0")
        lines.append(sep_single)

        input_size_mb = (np.prod(dummy_shape) * 8) / (1024 * 1024)
        params_size_mb = (total_params * 8) / (1024 * 1024)
        total_size_mb = total_mem_bytes / (1024 * 1024)

        lines.append(f"Input size (MB):                {input_size_mb:.6f}")
        lines.append(f"Params size (MB):               {params_size_mb:.6f}")
        lines.append(f"Estimated Total Size (MB):      {total_size_mb:.6f}")
        lines.append(sep_double)

        summary_text = "\n".join(lines)
        if verbose:
            print(summary_text)
        return summary_text

    def __repr__(self):
        desc = "Sequential(\n"
        for i, l in enumerate(self.layers):
            desc += f"  ({i}): {l}\n"
        desc += ")"
        return desc
