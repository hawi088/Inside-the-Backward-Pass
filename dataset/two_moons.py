"""
Two Moons Non-Linear Dataset Generator
Core Stack: Pure NumPy (Zero Scikit-Learn / PyTorch)
"""

import numpy as np

# Standard palette for Capstone Project
COLOR_CLASS_0 = "#E83E8C"  # Rose Pink
COLOR_CLASS_1 = "#3F51B5"  # Slate Indigo
COLOR_BG = "#12141D"       # Dark Slate Canvas
COLOR_TEXT = "#E0E6ED"     # Light Text
COLOR_GRID = "#2A2E3D"     # Subtle Grid

def make_two_moons(n_samples: int = 500, noise: float = 0.2, seed: int = 42):
    """
    Generates two interleaving half circles (moons) with Gaussian noise.
    Pure NumPy implementation of the Two Moons benchmark dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of sample points generated (default: 500).
    noise : float
        Standard deviation of Gaussian noise added to data (default: 0.2).
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    X : np.ndarray of shape (n_samples, 2)
        Input coordinate points (x1, x2).
    y : np.ndarray of shape (n_samples, 1)
        Binary ground-truth labels (0 for outer moon, 1 for inner moon).
    """
    rng = np.random.default_rng(seed)
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Outer moon: semi-circle centered at (0, 0)
    theta_out = np.linspace(0, np.pi, n_samples_out)
    outer_circ_x = np.cos(theta_out)
    outer_circ_y = np.sin(theta_out)
    
    # Inner moon: inverted semi-circle offset to (1, -0.5)
    theta_in = np.linspace(0, np.pi, n_samples_in)
    inner_circ_x = 1.0 - np.cos(theta_in)
    inner_circ_y = 1.0 - np.sin(theta_in) - 0.5
    
    # Stack features into (N, 2)
    x1 = np.concatenate([outer_circ_x, inner_circ_x])
    x2 = np.concatenate([outer_circ_y, inner_circ_y])
    X = np.column_stack([x1, x2])
    
    # Binary targets (N, 1)
    y = np.concatenate([
        np.zeros(n_samples_out, dtype=np.float64),
        np.ones(n_samples_in, dtype=np.float64)
    ]).reshape(-1, 1)
    
    # Add Gaussian perturbation
    if noise > 0:
        X += rng.normal(loc=0.0, scale=noise, size=X.shape)
        
    return X, y

def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """Splits dataset into stratified train and test partitions in pure NumPy."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    split_idx = int(n * (1.0 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Two Moons Non-Linear Dataset Preview")
    print("=" * 60)

    X, y = make_two_moons(n_samples=500, noise=0.2, seed=42)
    print(f"Total samples:       {len(X)}")
    print(f"Feature matrix (X):  shape {X.shape}, dtype {X.dtype}")
    print(f"Target vector  (y):  shape {y.shape}, dtype {y.dtype}")
    print(f"Class 0 (Rose Pink): {np.sum(y == 0)} samples")
    print(f"Class 1 (Slate):     {np.sum(y == 1)} samples")
    print(f"X feature ranges:    x1 in [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], "
          f"x2 in [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    print("=" * 60)

    # Render a quick scatter preview
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=COLOR_BG)
    ax.set_facecolor("#181B26")

    c0 = (y.flatten() == 0)
    c1 = (y.flatten() == 1)
    ax.scatter(X[c0, 0], X[c0, 1], color=COLOR_CLASS_0, s=35, alpha=0.85, label="Class 0 (Rose Pink)")
    ax.scatter(X[c1, 0], X[c1, 1], color=COLOR_CLASS_1, s=35, alpha=0.85, label="Class 1 (Slate Indigo)")

    ax.set_title("Two Moons Dataset Preview (500 samples, noise=0.2)", color=COLOR_TEXT, fontsize=12, pad=10)
    ax.set_xlabel(r"Feature $x_1$", color="#94A3B8")
    ax.set_ylabel(r"Feature $x_2$", color="#94A3B8")
    ax.grid(True, linestyle=":", color=COLOR_GRID, alpha=0.6)
    ax.legend(facecolor="#181B26", edgecolor="#334155")

    import os
    import sys

    os.makedirs("exports", exist_ok=True)
    preview_path = "exports/dataset_two_moons_preview.png"
    plt.savefig(preview_path, dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
    print(f"Saved dataset preview figure to: {preview_path}")

    # Display plot window if running interactively
    if "--no-show" not in sys.argv:
        try:
            plt.show()
        except Exception:
            pass
