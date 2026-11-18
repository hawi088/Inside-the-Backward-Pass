"""
Module 1: The Non-Linear Dataset & Linear Model Failure
Capstone Project: "Inside the Backward Pass"
Core Stack: Pure NumPy & Matplotlib (Zero PyTorch, TensorFlow, or autograd libraries)

Objective:
Establish the core classification problem and prove the fundamental limitation of linear models.
- Generates 500 samples of the Two Moons dataset (noise=0.2).
- Class 0 (rose pink: #E83E8C) and Class 1 (slate indigo: #3F51B5).
- Implements a pure first-principles Linear Classifier (Logistic Regression) from raw NumPy:
  1. Affine Projection: z = X @ W + b
  2. Sigmoid Link: p = 1 / (1 + exp(-z))
  3. Binary Cross-Entropy Loss: L = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
  4. Analytical Gradients: dL/dW = (1/N) * X^T @ (p - y), dL/db = (1/N) * sum(p - y)
  5. Gradient Descent Update: W <- W - alpha * dL/dW, b <- b - alpha * dL/db
- Visual Proof: Renders high-res scatter plot comparing data distribution against the straight
  decision boundary w1*x1 + w2*x2 + b = 0, visually capturing why points remain trapped
  on the wrong side and training accuracy strictly caps out (~86.8%).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset.two_moons import make_two_moons, COLOR_CLASS_0, COLOR_CLASS_1

class PureLinearClassifier:
    """
    Pure Linear Classifier (Logistic Regression) built from raw mathematical first principles.
    Zero high-level abstraction: uses raw NumPy matrix multiplications, analytical gradients,
    and gradient descent parameter updates.

    Mathematical Model:
    -------------------
    Forward:
        z = X @ W + b                           (Affine projection: (N, 2) @ (2, 1) + (1, 1) -> (N, 1))
        p = 1 / (1 + exp(-z))                   (Sigmoid activation probability: (N, 1))

    Loss (Binary Cross-Entropy):
        L = - (1 / N) * sum( y * log(p) + (1 - y) * log(1 - p) )

    Analytical Gradients (Chain Rule):
        dL/dz = (1 / N) * (p - y)               (Residual error: (N, 1))
        dL/dW = X^T @ dL/dz = (1 / N) * X^T @ (p - y)   (Weight gradient: (2, 1))
        dL/db = sum(dL/dz, axis=0) = (1 / N) * sum(p - y) (Bias gradient: (1, 1))

    Parameter Update:
        W = W - alpha * dL/dW
        b = b - alpha * dL/db
    """

    def __init__(self, in_features: int = 2, lr: float = 0.1, seed: int = 42):
        self.in_features = in_features
        self.lr = lr
        rng = np.random.default_rng(seed)

        # Xavier/Glorot Normal initialization
        std = np.sqrt(2.0 / in_features)
        self.W = rng.normal(loc=0.0, scale=std, size=(in_features, 1)).astype(np.float64)
        self.b = np.zeros((1, 1), dtype=np.float64)

        # Training history buffers
        self.loss_history = []
        self.acc_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z_clip = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes predicted probabilities: p = sigma(X @ W + b)."""
        z = np.matmul(X, self.W) + self.b
        return self.sigmoid(z)

    def compute_loss(self, p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
        """Computes Binary Cross-Entropy loss with numerical stability epsilon."""
        n = len(y)
        p_clipped = np.clip(p, eps, 1.0 - eps)
        loss_val = -np.sum(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)) / n
        return float(loss_val)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, verbose: bool = False):
        """
        Fits the linear classifier using raw Gradient Descent.
        Runs full-batch backpropagation step-by-step for the specified number of epochs.
        """
        n = len(X)
        self.loss_history = []
        self.acc_history = []

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            p = self.forward(X)

            # 2. Compute Loss & Accuracy
            loss = self.compute_loss(p, y)
            preds_binary = (p >= 0.5).astype(np.float64)
            acc = float(np.mean(preds_binary == y)) * 100.0

            self.loss_history.append(loss)
            self.acc_history.append(acc)

            # 3. Backward Pass (Analytical Gradients)
            # Derivative of BCE wrt logit z: dz = (p - y) / N
            dz = (p - y) / n
            dW = np.matmul(X.T, dz)                    # Shape (2, 1)
            db = np.sum(dz, axis=0, keepdims=True)     # Shape (1, 1)

            # 4. Gradient Descent Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if verbose and (epoch % 50 == 0 or epoch == epochs):
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.2f}% | "
                      f"W: [{self.W[0, 0]:.3f}, {self.W[1, 0]:.3f}], b: {self.b[0, 0]:.3f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) >= 0.5).astype(np.float64)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        p = self.forward(X)
        loss = self.compute_loss(p, y)
        preds = (p >= 0.5).astype(np.float64)
        acc = float(np.mean(preds == y)) * 100.0
        return loss, acc


# ==============================================================================
# Visualization Suite & Proof of Linear Failure
# ==============================================================================

def plot_module_1(X: np.ndarray, y: np.ndarray, clf: PureLinearClassifier, save_path: str = None, show: bool = False):
    """
    Renders a clean, minimalist two-panel figure in the Module 2 paper-white aesthetic.
    Panel 1: Scatter + learned linear decision boundary + trapped error points.
    Panel 2: Loss & accuracy convergence plateau curves.
    """
    # ── Module 2 palette ──────────────────────────────────────────────────────
    BG    = "#FBFAF7"   # paper white
    INK   = "#1C1C1C"   # axes / labels
    MID   = "#5C5A55"   # secondary text
    GREY  = "#9A978F"   # tick marks
    LIGHT = "#D6D3CA"   # hairlines
    GRID  = "#ECE9E2"   # grid
    BLUE  = "#2D5B9A"   # Class 1 / loss curve
    GREEN = "#3C7A5A"   # accuracy curve
    RED   = "#C4452F"   # decision boundary / trapped points

    W = clf.W.flatten()
    b = clf.b.flatten()[0]
    preds_binary = clf.predict(X)
    loss_val, final_acc = clf.evaluate(X, y)

    misclassified_mask = (preds_binary != y).flatten()
    correct_mask       = ~misclassified_mask
    num_trapped        = int(np.sum(misclassified_mask))

    # ── Reset to default (undo dark_background) ──────────────────────────────
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset":     "cm",
        "axes.unicode_minus":   False,
        "figure.facecolor":     BG,
        "axes.facecolor":       BG,
        "axes.edgecolor":       LIGHT,
        "axes.linewidth":       0.8,
        "axes.labelcolor":      INK,
        "xtick.color":          GREY,
        "ytick.color":          GREY,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "grid.color":           GRID,
        "grid.linewidth":       0.6,
        "text.color":           INK,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.25, 0.95], "wspace": 0.28})
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    # ── Panel 1: Decision Boundary ────────────────────────────────────────────
    x_min = X[:, 0].min() - 0.5;  x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5;  y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 260),
                         np.linspace(y_min, y_max, 260))
    prob_grid = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Very light probability wash — single desaturated colormap
    ax1.contourf(xx, yy, prob_grid, levels=30, cmap="RdBu_r", alpha=0.12)
    ax1.contour(xx,  yy, prob_grid, levels=[0.5], colors=[RED],
                linewidths=1.6, linestyles="--")

    # Data points — correct
    mask_c0_ok = (y.flatten() == 0) & correct_mask
    mask_c1_ok = (y.flatten() == 1) & correct_mask
    ax1.scatter(X[mask_c0_ok, 0], X[mask_c0_ok, 1],
                color=RED,  s=28, alpha=0.70, linewidths=0, label="Class 0")
    ax1.scatter(X[mask_c1_ok, 0], X[mask_c1_ok, 1],
                color=BLUE, s=28, alpha=0.70, linewidths=0, label="Class 1")

    # Trapped / misclassified — open circles, same colour, louder
    mask_c0_err = (y.flatten() == 0) & misclassified_mask
    mask_c1_err = (y.flatten() == 1) & misclassified_mask
    ax1.scatter(X[mask_c0_err, 0], X[mask_c0_err, 1],
                facecolors="none", edgecolors=RED,  s=72, linewidths=1.4,
                label=f"Trapped ({np.sum(mask_c0_err)} pts)")
    ax1.scatter(X[mask_c1_err, 0], X[mask_c1_err, 1],
                facecolors="none", edgecolors=BLUE, s=72, linewidths=1.4,
                label=f"Trapped ({np.sum(mask_c1_err)} pts)")

    # Explicit boundary line annotation
    x_vals = np.array([x_min, x_max])
    if abs(W[1]) > 1e-6:
        y_bnd = -(W[0] * x_vals + b) / W[1]
        ax1.plot(x_vals, y_bnd, color=INK, lw=1.0, linestyle="--", alpha=0.5)

    ax1.set_xlim(x_min, x_max);  ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel(r"$x_1$",  fontsize=11, color=INK)
    ax1.set_ylabel(r"$x_2$",  fontsize=11, color=INK)
    ax1.set_title("Linear boundary on Two Moons — geometric failure",
                  fontsize=11, color=INK, pad=10, fontweight="normal")
    ax1.grid(True, lw=0.5)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper left",
               labelcolor=INK)

    # Annotation box — minimal ink
    note = (f"Decision boundary:  ${W[0]:.2f}\\,x_1 + {W[1]:.2f}\\,x_2 + {b:.2f} = 0$\n"
            f"Trapped points: {num_trapped} / {len(y)}    "
            f"Accuracy ceiling: {final_acc:.1f}%")
    ax1.text(0.98, 0.03, note, transform=ax1.transAxes, fontsize=8,
             color=MID, ha="right", va="bottom",
             bbox=dict(boxstyle="square,pad=0.4", facecolor=BG,
                       edgecolor=LIGHT, linewidth=0.8))

    # ── Panel 2: Convergence Plateau ──────────────────────────────────────────
    epochs_arr = np.arange(1, len(clf.loss_history) + 1)
    acc_pct    = np.array(clf.acc_history)

    ax2.plot(epochs_arr, clf.loss_history, color=RED,  lw=1.8, label="BCE Loss")
    ax2.set_xlabel("Epoch",  fontsize=11, color=INK)
    ax2.set_ylabel("BCE Loss", fontsize=11, color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(bottom=0.0)

    ax2b = ax2.twinx()
    ax2b.plot(epochs_arr, acc_pct, color=BLUE, lw=1.8, linestyle="--",
              label="Accuracy (%)")
    ax2b.set_ylabel("Accuracy (%)", fontsize=11, color=BLUE)
    ax2b.tick_params(axis="y", labelcolor=BLUE)
    ax2b.set_ylim(60, 100)

    # Plateau shading — hairline grey, not solid
    plateau_start = 50
    ax2.axvspan(plateau_start, len(clf.loss_history),
                color=GRID, alpha=0.9, zorder=0)
    ax2.text(plateau_start + 5, max(clf.loss_history) * 0.95,
             "Plateau", fontsize=8, color=GREY, va="top")

    # Ceiling annotation
    ax2b.axhline(final_acc, color=LIGHT, lw=0.8, linestyle=":")
    ax2b.text(len(clf.loss_history) * 0.6, final_acc + 0.5,
              f"{final_acc:.1f}% ceiling", fontsize=8, color=MID)

    ax2.set_title("Loss & accuracy plateau — model capacity saturated",
                  fontsize=11, color=INK, pad=10, fontweight="normal")
    ax2.grid(True, lw=0.5)

    lines  = [plt.Line2D([0], [0], color=RED,  lw=1.8),
              plt.Line2D([0], [0], color=BLUE, lw=1.8, linestyle="--")]
    labels = ["BCE Loss", "Accuracy (%)"]
    ax2.legend(lines, labels, fontsize=8.5, frameon=False, loc="center right",
               labelcolor=INK)

    fig.suptitle("Module 1 — Geometric Proof: Linear Models Cannot Classify Two Moons",
                 fontsize=12, color=INK, fontweight="normal", y=1.01)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, facecolor=BG, bbox_inches="tight")
        print(f"[Module 1] Saved figure: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return final_acc


# ==============================================================================
# Module Entrypoint
# ==============================================================================

def run_module_1(save_dir: str = "exports", show: bool = False):
    print("=" * 80)
    print("MODULE 1: The Non-Linear Dataset & Linear Model Failure")
    print("=" * 80)

    # 1. Generate 500 samples of Two Moons (noise=0.2)
    X, y = make_two_moons(n_samples=500, noise=0.2, seed=42)
    print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class 0 (Rose Pink): {np.sum(y == 0)} samples | Class 1 (Slate Indigo): {np.sum(y == 1)} samples")

    # 2. Fit pure first-principles linear classifier
    print("\n--- Training Pure Linear Classifier from Scratch (300 Epochs Gradient Descent) ---")
    clf = PureLinearClassifier(in_features=2, lr=0.15, seed=42)
    clf.fit(X, y, epochs=300, verbose=True)

    W, b = clf.W.flatten(), clf.b.flatten()[0]
    loss, final_acc = clf.evaluate(X, y)
    print(f"\nFinal Learned Parameters: W = [{W[0]:.4f}, {W[1]:.4f}], b = {b:.4f}")
    print(f"Final Linear Accuracy:    {final_acc:.2f}% (Loss: {loss:.4f})")

    # 3. Visual Proof
    save_path = os.path.join(save_dir, "module1_linear_failure.png")
    plot_module_1(X, y, clf, save_path=save_path, show=show)
    print(f"[Module 1 Proof Complete] As expected, linear classifier strictly plateaued at {final_acc:.2f}%.")
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1: Linear Model Failure on Two Moons")
    parser.add_argument("--show", action="store_true", help="Display interactive plot window")
    parser.add_argument("--save-dir", type=str, default="exports", help="Directory for output figure")
    args = parser.parse_args()
    run_module_1(save_dir=args.save_dir, show=args.show)
