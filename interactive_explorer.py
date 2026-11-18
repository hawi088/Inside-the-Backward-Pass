"""
Interactive Explorer & Visual Dashboard
Capstone Project: "Inside the Backward Pass"
Core Stack: Pure NumPy & Matplotlib (Zero PyTorch, TensorFlow, or autograd libraries)

Interactive Matplotlib GUI featuring:
- Live Two Moons dataset generator with adjustable noise slider
- Dynamic decision boundary comparator (Linear vs. Deep MLP)
- Real-time calculus tangent slope explorer with slider for logit z
- Activation function switcher (ReLU, Tanh, Sigmoid)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from dataset.two_moons import make_two_moons, COLOR_CLASS_0, COLOR_CLASS_1
from engine.layers import Linear, ReLU, Tanh, Sigmoid
from engine.losses import BCELoss
from engine.network import Sequential
from engine.optimizers import Adam


def train_interactive_model(X, y, act_name="relu", epochs=120):
    """Trains a quick interactive model on current data."""
    if act_name == "relu":
        act1, act2 = ReLU(), ReLU()
    elif act_name == "tanh":
        act1, act2 = Tanh(), Tanh()
    else:
        act1, act2 = Sigmoid(), Sigmoid()

    model = Sequential([
        Linear(2, 16, init="he" if act_name == "relu" else "xavier", seed=42),
        act1,
        Linear(16, 12, init="he" if act_name == "relu" else "xavier", seed=43),
        act2,
        Linear(12, 1, init="xavier", seed=44),
        Sigmoid()
    ])
    loss_fn = BCELoss()
    opt = Adam(model.parameters(), lr=0.03)

    for _ in range(epochs):
        p = model.forward(X)
        loss_fn.forward(p, y)
        opt.zero_grad()
        model.backward(loss_fn.backward())
        opt.step()

    return model


def launch_interactive_dashboard():
    """Launches the unified interactive Matplotlib dashboard."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.5), facecolor="#0B0E14")
    fig.suptitle(
        "Inside the Backward Pass: Interactive Laboratory\n"
        "Pure NumPy & Matplotlib Neural Mechanics",
        fontsize=14, fontweight="bold", color="#F8FAFC", y=0.97
    )

    # Initial state
    current_noise = 0.20
    current_act = "relu"
    X, y = make_two_moons(n_samples=400, noise=current_noise, seed=42)
    model = train_interactive_model(X, y, act_name=current_act, epochs=100)

    # Subplot 1: Decision Boundary
    ax_boundary = fig.add_axes([0.08, 0.22, 0.45, 0.65])
    ax_boundary.set_facecolor("#121620")

    # Subplot 2: Calculus & Derivative Tangent
    ax_calc = fig.add_axes([0.60, 0.22, 0.36, 0.65])
    ax_calc.set_facecolor("#121620")

    # Widget controls axes
    ax_slider_noise = fig.add_axes([0.10, 0.08, 0.38, 0.03], facecolor="#1E293B")
    slider_noise = Slider(ax_slider_noise, "Noise", 0.05, 0.45, valinit=current_noise, valstep=0.01, color="#38BDF8")

    ax_slider_z = fig.add_axes([0.60, 0.08, 0.35, 0.03], facecolor="#1E293B")
    slider_z = Slider(ax_slider_z, "Logit z", -4.0, 4.0, valinit=1.0, valstep=0.1, color="#F43F5E")

    ax_radio_act = fig.add_axes([0.48, 0.03, 0.10, 0.13], facecolor="#1A202C")
    radio_act = RadioButtons(ax_radio_act, ("Quadratic", "ReLU", "Tanh", "Sigmoid"), active=0, activecolor="#10B981")

    # Meshgrid setup
    xx, yy = np.meshgrid(np.linspace(-2.0, 3.0, 100), np.linspace(-1.5, 2.0, 100))
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Plot initial boundary
    def draw_boundary():
        ax_boundary.clear()
        ax_boundary.set_facecolor("#121620")
        prob_grid = model.forward(grid_coords).reshape(xx.shape)
        ax_boundary.contourf(xx, yy, prob_grid, levels=30, cmap="coolwarm", alpha=0.45)
        ax_boundary.contour(xx, yy, prob_grid, levels=[0.5], colors=["#FFD166"], linewidths=2.2)

        c0 = (y.flatten() == 0)
        c1 = (y.flatten() == 1)
        ax_boundary.scatter(X[c0, 0], X[c0, 1], color=COLOR_CLASS_0, s=28, label="Class 0 (Rose Pink)")
        ax_boundary.scatter(X[c1, 0], X[c1, 1], color=COLOR_CLASS_1, s=28, label="Class 1 (Slate Indigo)")

        preds = (model.forward(X) >= 0.5).astype(np.float64)
        acc = np.mean(preds == y) * 100.0
        ax_boundary.set_title(f"Learned Boundary ({current_act.upper()})\nAccuracy: {acc:.1f}%",
                              fontsize=11, fontweight="bold", color="#38BDF8")
        ax_boundary.set_xlabel(r"Feature $x_1$", color="#94A3B8")
        ax_boundary.set_ylabel(r"Feature $x_2$", color="#94A3B8")
        ax_boundary.grid(True, linestyle=":", color="#1E293B", alpha=0.5)
        ax_boundary.legend(loc="upper left", fontsize=8, facecolor="#1A202C", edgecolor="#334155")

    # Plot calculus curves
    z_vals = np.linspace(-4.5, 4.5, 250)
    def draw_calculus(z0):
        ax_calc.clear()
        ax_calc.set_facecolor("#121620")

        if current_act == "quadratic":
            f_vals = z_vals ** 2
            f0 = z0 ** 2
            slope = 2.0 * z0
            func_name = r"f(x) = x^2 \implies f'(x) = 2x"
        elif current_act == "sigmoid":
            f_vals = 1.0 / (1.0 + np.exp(-z_vals))
            f0 = 1.0 / (1.0 + np.exp(-z0))
            slope = f0 * (1.0 - f0)
            func_name = r"\sigma(z)"
        elif current_act == "tanh":
            f_vals = np.tanh(z_vals)
            f0 = np.tanh(z0)
            slope = 1.0 - f0 ** 2
            func_name = r"\tanh(z)"
        else:  # relu
            f_vals = np.maximum(0.0, z_vals)
            f0 = max(0.0, z0)
            slope = 1.0 if z0 > 0.0 else 0.0
            func_name = r"\mathrm{ReLU}(z)"

        ax_calc.plot(z_vals, f_vals, color="#38BDF8", lw=2.4, label=f"${func_name}$")
        ax_calc.axhline(0.0, color="#334155", lw=0.8, linestyle="--")
        ax_calc.axvline(0.0, color="#334155", lw=0.8, linestyle="--")

        # Tangent line
        t_span = np.linspace(z0 - 1.5, z0 + 1.5, 30)
        ax_calc.plot(t_span, f0 + slope * (t_span - z0), color="#F43F5E", lw=2.2, label=f"Tangent (Slope = {slope:.3f})")
        ax_calc.plot([z0], [f0], marker="o", color="#F43F5E", markersize=8)

        ax_calc.set_title(f"Dynamic Calculus & Tangent Slopes\n$z_0 = {z0:+.2f} \\implies f'(z_0) = {slope:.3f}$",
                          fontsize=11, fontweight="bold", color="#F43F5E")
        ax_calc.set_xlabel(r"Logit $z$", color="#94A3B8")
        ax_calc.set_ylabel(r"Output", color="#94A3B8")
        ax_calc.grid(True, linestyle=":", color="#1E293B", alpha=0.5)
        ax_calc.legend(loc="upper left", fontsize=8.5, facecolor="#1A202C", edgecolor="#334155")

    # Initial render
    draw_boundary()
    draw_calculus(1.0)

    # Event callbacks
    def on_noise_change(val):
        nonlocal X, y, model, current_noise
        current_noise = val
        X, y = make_two_moons(n_samples=400, noise=current_noise, seed=42)
        model = train_interactive_model(X, y, act_name=current_act, epochs=80)
        draw_boundary()
        fig.canvas.draw_idle()

    def on_act_change(label):
        nonlocal current_act, model
        current_act = label.lower()
        if current_act != "quadratic":
            model = train_interactive_model(X, y, act_name=current_act, epochs=100)
            draw_boundary()
        draw_calculus(slider_z.val)
        fig.canvas.draw_idle()

    def on_z_change(val):
        draw_calculus(val)
        fig.canvas.draw_idle()

    slider_noise.on_changed(on_noise_change)
    slider_z.on_changed(on_z_change)
    radio_act.on_clicked(on_act_change)

    plt.show()


if __name__ == "__main__":
    launch_interactive_dashboard()
