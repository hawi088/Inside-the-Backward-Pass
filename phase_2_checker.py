"""
Module 3: The Modular Neural Engine & PyTorch-Style Summary
Capstone Project: "Inside the Backward Pass"
Core Stack: Pure NumPy & Matplotlib (Zero PyTorch, TensorFlow, or autograd libraries)

Phase 2 Tiny Architecture:
- 2 Inputs: (x1, x2)
- 1 Hidden Layer: 2 neurons with Sigmoid (h1, h2)
- 1 Output: (y_hat)
- Loss: MSELoss (and BCELoss)
- Network: calls forward on layers in order, backward in reverse order.
- Runs single data point forward and backward, printing every value and gradient
  so it can be directly verified against the Phase 1 hand-calculation paper.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dataset.two_moons import make_two_moons
from engine.layers import Linear, ReLU, Tanh, Sigmoid
from engine.losses import BCELoss, MSELoss
from engine.network import Sequential


def create_phase2_network(init: str = "xavier", seed: int = 42, output_sigmoid: bool = False) -> Sequential:
    """
    Builds the exact Phase 2 architecture:
    2 inputs -> 1 hidden layer with 2 neurons (Sigmoid) -> 1 output.
    Total parameters: 9 (6 in Layer 1, 3 in Layer 2).
    """
    layers = [
        Linear(in_features=2, out_features=2, init=init, seed=seed),      # Layer 0: W1 (2,2), b1 (1,2) -> 6 params
        Sigmoid(),                                                         # Layer 1: h1, h2 activations -> 0 params
        Linear(in_features=2, out_features=1, init=init, seed=seed + 1),  # Layer 2: W2 (2,1), b2 (1,1) -> 3 params
    ]
    if output_sigmoid:
        layers.append(Sigmoid())
    return Sequential(layers)


def plot_module_3_inspection(model: Sequential, X_sample: np.ndarray, y_sample: np.ndarray, save_path: str = None, show: bool = False):
    """
    Renders visual inspection of weight matrices (2x2 and 2x1), biases, and gradient magnitudes.
    """
    loss_fn = MSELoss()
    y_pred = model.forward(X_sample)
    loss = loss_fn.forward(y_pred, y_sample)
    dLoss = loss_fn.backward()
    model.backward(dLoss)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 7.5), facecolor="#0B0E14")
    fig.suptitle(
        "Module 3 (Phase 2 Architecture): 2 Inputs -> 1 Hidden Layer (2 Neurons) -> 1 Output\n"
        "Weights (W1: 2x2, W2: 2x1), Biases (b1: 1x2, b2: 1x1), and Analytical Gradients (dW, db)",
        fontsize=13, fontweight="bold", color="#F8FAFC", y=0.98
    )

    linear_layers = [l for l in model.layers if isinstance(l, Linear)]
    num_linear = len(linear_layers)

    gs = fig.add_gridspec(2, num_linear, height_ratios=[1.1, 1.0], wspace=0.35, hspace=0.35)

    # Top Row: Weight matrix heatmaps
    for idx, layer in enumerate(linear_layers):
        ax_w = fig.add_subplot(gs[0, idx])
        ax_w.set_facecolor("#121620")
        im = ax_w.imshow(layer.W, cmap="magma", aspect="auto")
        title_str = (
            rf"$\bf{{Layer\ {idx+1}:}}$ Linear({layer.in_features}$\to${layer.out_features})" + "\n"
            rf"$W \in \mathbb{{R}}^{{{layer.in_features} \times {layer.out_features}}}$ ($\|W\|_F={np.linalg.norm(layer.W):.3f}$)"
        )
        ax_w.set_title(title_str, fontsize=10, color="#38BDF8", pad=8)
        ax_w.set_xlabel("Output Neurons", color="#94A3B8", fontsize=9)
        ax_w.set_ylabel("Input Neurons", color="#94A3B8", fontsize=9)
        cbar = fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors="#94A3B8")

        # Overlay text values inside cells
        for r in range(layer.W.shape[0]):
            for c in range(layer.W.shape[1]):
                val = layer.W[r, c]
                ax_w.text(c, r, f"{val:.3f}", ha="center", va="center", color="#F8FAFC", fontsize=9, fontweight="bold")

    # Bottom Row: Gradient norm & distribution
    for idx, layer in enumerate(linear_layers):
        ax_grad = fig.add_subplot(gs[1, idx])
        ax_grad.set_facecolor("#121620")
        dw_norm = np.linalg.norm(layer.dW)
        db_norm = np.linalg.norm(layer.db)

        bars = ax_grad.bar(["$\\|dW\\|_F$", "$\\|db\\|_2$"], [dw_norm, db_norm],
                           color=["#F59E0B", "#10B981"], width=0.50, edgecolor="#F8FAFC", lw=0.6)
        ax_grad.set_title(rf"$\bf{{Grad\ Norms}}$: $\|dW\|={dw_norm:.3e}$" + "\n"
                          rf"$\|db\|={db_norm:.3e}$", fontsize=9.5, color="#F59E0B", pad=8)
        ax_grad.set_ylabel("Norm Magnitude", color="#94A3B8", fontsize=9)
        ax_grad.grid(True, linestyle=":", color="#1E293B", alpha=0.6, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax_grad.text(bar.get_x() + bar.get_width() / 2.0, height * 1.02,
                         f"{height:.2e}", ha="center", va="bottom", fontsize=8, color="#F8FAFC")

    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.10, top=0.88)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[Module 3] Saved architecture inspection figure: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def run_phase1_paper_trace_check(model: Sequential):
    """
    Runs a single fixed data point through the 2->2->1 network and prints every single
    forward value and every single gradient for 1:1 paper comparison.
    """
    print("\n" + "=" * 80)
    print("PHASE 1 PAPER TRACE VERIFICATION: SINGLE DATA POINT STEP-BY-STEP")
    print("=" * 80)

    # Fixed single data point and known weights
    x_single = np.array([[0.5, 0.8]], dtype=np.float64)  # 2 inputs
    y_single = np.array([[1.0]], dtype=np.float64)        # 1 target

    # Explicit known weights
    l1 = model.layers[0]  # Linear(2, 2)
    l1.W = np.array([[0.5, 0.8], [-0.6, 0.4]], dtype=np.float64)
    l1.b = np.array([[0.1, -0.2]], dtype=np.float64)

    act = model.layers[1] # Sigmoid()

    l2 = model.layers[2]  # Linear(2, 1)
    l2.W = np.array([[0.7], [-0.5]], dtype=np.float64)
    l2.b = np.array([[0.3]], dtype=np.float64)

    # 1. Forward Pass
    y_hat = model.forward(x_single)
    loss_fn = MSELoss()
    loss = loss_fn.forward(y_hat, y_single)

    # Intermediate cached values
    z1 = l1._output_cache  # Shape (1, 2)
    a1 = act._output_cache # Shape (1, 2)
    z2 = l2._output_cache  # Shape (1, 1)

    print(">>> 1. FORWARD PASS VALUES (Paper vs Code) <<<")
    print(f"  Input x:                [{x_single[0,0]:.4f}, {x_single[0,1]:.4f}]")
    print(f"  Hidden Pre-act z1:      [{z1[0,0]:.6f}, {z1[0,1]:.6f}]")
    print(f"  Hidden Sigmoid h (a1):  [{a1[0,0]:.6f}, {a1[0,1]:.6f}]")
    print(f"  Output Pre-act z2:      {z2[0,0]:.6f}")
    print(f"  Output Prediction y_hat:{y_hat[0,0]:.6f}")
    print(f"  Target y:               {y_single[0,0]:.4f}")
    print(f"  MSE Loss:               {loss:.6f}")

    # 2. Backward Pass
    model.zero_grad()
    dLoss = loss_fn.backward()  # 2 * (y_hat - y)
    dX = model.backward(dLoss)

    print("\n>>> 2. BACKWARD PASS GRADIENTS (Paper vs Code) <<<")
    print(f"  dL/dy_hat:              {dLoss[0,0]:+.6f}")
    print(f"  Output Layer dW2:       [{l2.dW[0,0]:+.6f}, {l2.dW[1,0]:+.6f}]^T")
    print(f"  Output Layer db2:       [{l2.db[0,0]:+.6f}]")
    print(f"  Hidden Activation dZ1:  [{act._grad_cache[0,0]:+.6f}, {act._grad_cache[0,1]:+.6f}]")
    print(f"  Hidden Layer dW1:       [[{l1.dW[0,0]:+.6f}, {l1.dW[0,1]:+.6f}], [{l1.dW[1,0]:+.6f}, {l1.dW[1,1]:+.6f}]]")
    print(f"  Hidden Layer db1:       [{l1.db[0,0]:+.6f}, {l1.db[0,1]:+.6f}]")
    print(f"  Input Gradient dX:      [{dX[0,0]:+.6f}, {dX[0,1]:+.6f}]")
    print("=" * 80 + "\n")


def run_module_3(save_dir: str = "exports", show: bool = False):
    print("=" * 80)
    print("MODULE 3 (Phase 2): Modular Neural Engine & PyTorch-Style Summary")
    print("=" * 80)

    # 1. Instantiate Fixed Tiny Architecture (2 -> 2 -> 1)
    model = create_phase2_network(init="xavier", seed=42)

    # 2. PyTorch-style summary
    print("\n--- Architecture Summary Inspection (Fixed 1 Hidden Layer, 2 Neurons) ---")
    summary_str = model.summary(input_shape=(1, 2), verbose=True)

    # 3. Phase 1 Paper Hand-Trace Verification (Single Data Point)
    run_phase1_paper_trace_check(model)

    # 4. Batch evaluation
    X, y = make_two_moons(n_samples=64, noise=0.2, seed=42)
    y_pred = model.forward(X)
    loss_fn = MSELoss()
    loss = loss_fn.forward(y_pred, y)
    print(f"Batch Forward verification:  Input shape {X.shape} -> Output shape {y_pred.shape}")
    print(f"Batch MSE Loss:              {loss:.5f}")

    grad_loss = loss_fn.backward()
    grad_input = model.backward(grad_loss)
    print(f"Batch Backward verification: Loss grad {grad_loss.shape} -> Input grad {grad_input.shape}")

    # 5. Visual parameter & gradient audit
    save_path = os.path.join(save_dir, "module3_engine_summary.png")
    plot_module_3_inspection(model, X, y, save_path=save_path, show=show)
    print("[Module 3 Complete] Phase 2 network (2 -> 2 -> 1) verified.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3: Phase 2 Modular Engine (2 -> 2 -> 1)")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    parser.add_argument("--save-dir", type=str, default="exports", help="Directory for output figure")
    args = parser.parse_args()
    run_module_3(save_dir=args.save_dir, show=args.show)
