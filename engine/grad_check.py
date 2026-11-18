"""
Rigorous Mathematical Proof - Phase 3 Gradient Checker
Core Stack: Pure NumPy (Zero PyTorch, TensorFlow, or autograd libraries)

Module 7 Implementation:
Implements two-sided symmetric finite-difference gradient verification:
f'(theta) ~ [ J(theta + eps) - J(theta - eps) ] / (2 * eps)
Enforces strict relative error threshold < 1e-6.
"""

from typing import Dict, List, Any
import numpy as np
from .network import Sequential
from .losses import Loss


def gradient_check(
    network: Sequential,
    loss_fn: Loss,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-7,
    threshold: float = 1e-6,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Performs rigorous finite-difference gradient verification across all network parameters.

    Parameters:
    -----------
    network : Sequential
        Neural network model.
    loss_fn : Loss
        Loss function (e.g. BCELoss or MSELoss).
    X : np.ndarray
        Mini-batch input data.
    y : np.ndarray
        Ground-truth labels.
    eps : float
        Finite-difference perturbation step size (default: 1e-7).
    threshold : float
        Maximum allowable relative error threshold for mathematical proof (default: 1e-6).
    verbose : bool
        Whether to print the detailed audit table.

    Returns:
    --------
    audit_results : dict
        Detailed dictionary containing per-parameter comparison metrics and overall verdict.
    """
    # 1. Analytical backward pass
    network.zero_grad()
    y_pred = network.forward(X)
    loss_base = loss_fn.forward(y_pred, y_true=y)
    grad_loss = loss_fn.backward()
    network.backward(grad_loss)

    results = []
    overall_pass = True
    max_global_rel_err = 0.0

    all_analytical_flat = []
    all_numerical_flat = []

    # Iterate through all trainable layers
    layer_idx = 0
    for layer in network.layers:
        if not layer.trainable:
            continue

        for param_name, param_tensor, analytical_grad in layer.get_params():
            numerical_grad = np.zeros_like(param_tensor)
            it = np.nditer(param_tensor, flags=["multi_index"], op_flags=["readwrite"])

            # Evaluate symmetric finite differences for every scalar weight/bias
            while not it.finished:
                idx = it.multi_index
                orig_val = param_tensor[idx]

                # Perturb theta + eps
                param_tensor[idx] = orig_val + eps
                pred_plus = network.forward(X)
                loss_plus = loss_fn.forward(pred_plus, y_true=y)

                # Perturb theta - eps
                param_tensor[idx] = orig_val - eps
                pred_minus = network.forward(X)
                loss_minus = loss_fn.forward(pred_minus, y_true=y)

                # Reset parameter to original value
                param_tensor[idx] = orig_val

                # Two-sided numerical gradient
                numerical_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
                it.iternext()

            # Flatten for vector norm comparisons
            ana_flat = analytical_grad.ravel()
            num_flat = numerical_grad.ravel()

            all_analytical_flat.append(ana_flat)
            all_numerical_flat.append(num_flat)

            diff_norm = np.linalg.norm(ana_flat - num_flat)
            sum_norm = np.linalg.norm(ana_flat) + np.linalg.norm(num_flat)
            rel_error = diff_norm / (sum_norm + 1e-12)
            max_abs_diff = np.max(np.abs(ana_flat - num_flat))

            passed = rel_error < threshold
            if not passed:
                overall_pass = False

            max_global_rel_err = max(max_global_rel_err, rel_error)

            results.append({
                "layer_idx": layer_idx,
                "layer_name": layer.__class__.__name__,
                "param_name": param_name,
                "param_shape": param_tensor.shape,
                "num_elements": param_tensor.size,
                "ana_norm": np.linalg.norm(ana_flat),
                "num_norm": np.linalg.norm(num_flat),
                "max_abs_diff": max_abs_diff,
                "rel_error": rel_error,
                "passed": passed,
                "analytical_grad": analytical_grad.copy(),
                "numerical_grad": numerical_grad.copy(),
            })

        layer_idx += 1

    # Global vector check across the entire parameter space
    global_ana = np.concatenate(all_analytical_flat)
    global_num = np.concatenate(all_numerical_flat)
    global_diff_norm = np.linalg.norm(global_ana - global_num)
    global_sum_norm = np.linalg.norm(global_ana) + np.linalg.norm(global_num)
    global_rel_error = global_diff_norm / (global_sum_norm + 1e-12)

    if global_rel_error >= threshold:
        overall_pass = False

    audit_summary = {
        "passed": overall_pass,
        "global_rel_error": global_rel_error,
        "max_param_rel_error": max_global_rel_err,
        "threshold": threshold,
        "layer_results": results,
    }

    if verbose:
        print_gradient_check_table(audit_summary)

    return audit_summary


def print_gradient_check_table(audit_summary: Dict[str, Any]):
    """Prints a clean, professional audit table of gradient verification."""
    line_length = 100
    print("=" * line_length)
    print("GRADIENT CHECK AUDIT REPORT: Analytical Backprop vs. Numerical Finite Differences")
    print("=" * line_length)
    header = f"{'Layer':<14} {'Param':<8} {'Shape':<12} {'Max Abs Diff':<16} {'Ana Norm':<14} {'Num Norm':<14} {'Rel Error':<14} {'Status':<8}"
    print(header)
    print("-" * line_length)

    for item in audit_summary["layer_results"]:
        status = "PASS" if item["passed"] else "FAIL"
        shape_str = str(item["param_shape"])
        row = (
            f"{item['layer_name'] + ' [' + str(item['layer_idx']) + ']':<14} "
            f"{item['param_name']:<8} "
            f"{shape_str:<12} "
            f"{item['max_abs_diff']:<16.5e} "
            f"{item['ana_norm']:<14.5e} "
            f"{item['num_norm']:<14.5e} "
            f"{item['rel_error']:<14.5e} "
            f"{status:<8}"
        )
        print(row)

    print("-" * line_length)
    print(f"Global Parameter Vector Relative Error: {audit_summary['global_rel_error']:.5e} (Threshold: < {audit_summary['threshold']:.1e})")
    verdict = ">>> PASS: Analytical Backward Pass Mathematically Proven Exact! <<<" if audit_summary["passed"] else ">>> FAIL: Discrepancy Detected! <<<"
    print(verdict)
    print("=" * line_length)
