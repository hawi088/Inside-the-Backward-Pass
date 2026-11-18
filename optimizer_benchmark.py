"""
Module 4: Custom Optimizers from Scratch
Capstone Project: "Inside the Backward Pass"
Core Stack: Pure NumPy & Matplotlib (Zero PyTorch, TensorFlow, or autograd libraries)

Canonical Baseline:
- Architecture: 2 Inputs -> 1 Hidden Layer (2 neurons, Sigmoid) -> 1 Output (MSE setup)
- Single Canonical Data Point: x = [0.5, 0.8], target y = 1.0
- Exact Phase 1 Initial Weights (W1: 2x2, b1: 1x2, W2: 2x1, b2: 1x1)
- Loss: MSELoss (L = (y_hat - y)^2)

Compares:
1. Standard SGD: theta <- theta - alpha * grad
2. SGD + Momentum: v <- beta * v - alpha * grad; theta <- theta + v
3. Adam: First (m) and second raw (v) moment estimators with bias corrections

Two parts live in this file:
  A. The engine-based benchmark (4-panel PNG)                      -> run_module_4()
  B. VISUALIZATION SECTION: a minimalist film, in the same visual
     language as the Module 2 calculus film, that shows how each
     optimizer takes its 35 steps, one at a time, then all together   -> render_film()

Usage:
    python module4_optimizers.py                        # benchmark PNG (needs engine/) + cross-check
    python module4_optimizers.py --film                 # benchmark + full film
    python module4_optimizers.py --film-only            # film only (pure NumPy, no engine needed)
    python module4_optimizers.py --film-only --film-scene adam
    python module4_optimizers.py --film-only --preview  # PNG stills of every chapter
    Chapters: intro, sgd, momentum, adam, together, summary
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from engine.layers import Linear, Sigmoid
from engine.losses import MSELoss
from engine.network import Sequential
from engine.optimizers import SGD, SGDMomentum, Adam


# Canonical Phase 1 & 2 Constants
X_SINGLE = np.array([[0.5, 0.8]], dtype=np.float64)  # (1, 2)
Y_SINGLE = np.array([[1.0]], dtype=np.float64)        # (1, 1)

W1_INIT = np.array([[0.5, 0.8], [-0.6, 0.4]], dtype=np.float64)
B1_INIT = np.array([[0.1, -0.2]], dtype=np.float64)
W2_INIT = np.array([[0.7], [-0.5]], dtype=np.float64)
B2_INIT = np.array([[0.3]], dtype=np.float64)


def build_canonical_model() -> Sequential:
    """Builds the canonical Phase 2 network with exact Phase 1 initial weights."""
    model = Sequential([
        Linear(in_features=2, out_features=2),
        Sigmoid(),
        Linear(in_features=2, out_features=1),
    ])
    # Set exact Phase 1 weights
    l1 = model.layers[0]
    l1.W = np.copy(W1_INIT)
    l1.b = np.copy(B1_INIT)

    l2 = model.layers[2]
    l2.W = np.copy(W2_INIT)
    l2.b = np.copy(B2_INIT)
    return model


def train_single_point_optimizer(opt_name: str, opt_class, opt_kwargs: dict, steps: int = 35):
    """Trains the canonical network on the single Phase 1 point, recording every step."""
    model = build_canonical_model()
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    loss_fn = MSELoss()

    history = {
        "name": opt_name,
        "loss": [],
        "y_hat": [],
        "step_norms": [],
        "w1_w2_path": [],  # Track output weights (w1, w2)
        "step1_params": {},
    }

    l2 = model.layers[2]

    for step in range(steps):
        # Record weights before update
        history["w1_w2_path"].append((float(l2.W[0, 0]), float(l2.W[1, 0])))
        params_before = [np.copy(p) for layer in model.parameters() for _, p, _ in layer.get_params()]

        # 1. Forward
        y_pred = model.forward(X_SINGLE)
        loss = loss_fn.forward(y_pred, Y_SINGLE)
        history["loss"].append(float(loss))
        history["y_hat"].append(float(y_pred[0, 0]))

        # 2. Backward
        optimizer.zero_grad()
        grad = loss_fn.backward()
        model.backward(grad)

        # 3. Update
        optimizer.step()

        # Measure parameter displacement ||delta theta||
        params_after = [p for layer in model.parameters() for _, p, _ in layer.get_params()]
        step_sq = sum(np.sum(np.square(after - before)) for before, after in zip(params_before, params_after))
        history["step_norms"].append(float(np.sqrt(step_sq)))

        if step == 0:
            # Save Step 1 updated weights for ledger comparison
            history["step1_params"] = {
                "W1": np.copy(model.layers[0].W),
                "b1": np.copy(model.layers[0].b),
                "W2": np.copy(model.layers[2].W),
                "b2": np.copy(model.layers[2].b),
            }

    history["w1_w2_path"].append((float(l2.W[0, 0]), float(l2.W[1, 0])))
    return history, model


def plot_single_point_benchmark(trials: list, save_path: str = None, show: bool = False):
    """
    Renders professional 4-panel visual comparison of optimizer dynamics on the single Phase 1 point.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0B0E14")
    fig.suptitle(
        "Module 4: Optimizer Dynamics on the Canonical Single Data Point [x1=0.50, x2=0.80, target y=1.0]\n"
        "Tracking SGD vs. SGD+Momentum vs. Adam from Exact Phase 1 Initial Weights",
        fontsize=13, fontweight="bold", color="#F8FAFC", y=0.98
    )

    colors = {
        "Standard SGD": "#EF4444",      # Red
        "SGD + Momentum": "#3B82F6",    # Blue
        "Adam (Adaptive)": "#10B981",   # Green
    }

    # 1. Loss Descent (L -> 0)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#121620")
    for t in trials:
        name = t["name"]
        ax1.plot(t["loss"], color=colors[name], lw=2.4, marker="o", markersize=3.5,
                 label=f"{name} (Initial: {t['loss'][0]:.4f} -> Final: {t['loss'][-1]:.6f})")
    ax1.set_title("Single-Point MSE Loss Descent ($L = (\\hat{y} - 1.0)^2$)", fontsize=11, color="#38BDF8", pad=8)
    ax1.set_xlabel("Descent Step", color="#94A3B8", fontsize=9)
    ax1.set_ylabel("MSE Loss", color="#94A3B8", fontsize=9)
    ax1.grid(True, linestyle=":", color="#1E293B", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5, facecolor="#151A24", edgecolor="#334155")

    # 2. Prediction Convergence (y_hat -> 1.0)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#121620")
    ax2.axhline(1.0, color="#F59E0B", linestyle="--", lw=1.6, label="Ground Truth Target y = 1.0")
    for t in trials:
        name = t["name"]
        ax2.plot(t["y_hat"], color=colors[name], lw=2.4, marker="s", markersize=3.5,
                 label=f"{name} (Start: {t['y_hat'][0]:.4f} -> Peak: {t['y_hat'][-1]:.4f})")
    ax2.set_title("Model Prediction Convergence ($\\hat{y} \\to 1.0$)", fontsize=11, color="#38BDF8", pad=8)
    ax2.set_xlabel("Descent Step", color="#94A3B8", fontsize=9)
    ax2.set_ylabel("Predicted Output $\\hat{y}$", color="#94A3B8", fontsize=9)
    ax2.grid(True, linestyle=":", color="#1E293B", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5, facecolor="#151A24", edgecolor="#334155")

    # 3. Update Displacement Magnitude ||Delta theta||
    ax3 = axes[1, 0]
    ax3.set_facecolor("#121620")
    for t in trials:
        name = t["name"]
        ax3.plot(t["step_norms"], color=colors[name], lw=2.0, marker="^", markersize=3.5, label=name)
    ax3.set_title(r"Parameter Step Norm $\|\Delta \theta\|_2 = \|\theta_{t+1} - \theta_t\|_2$", fontsize=11, color="#38BDF8", pad=8)
    ax3.set_xlabel("Descent Step", color="#94A3B8", fontsize=9)
    ax3.set_ylabel(r"$\|\Delta \theta\|_2$", color="#94A3B8", fontsize=9)
    ax3.grid(True, linestyle=":", color="#1E293B", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5, facecolor="#151A24", edgecolor="#334155")

    # 4. Trajectory in Output Weight Subspace (w1, w2)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#121620")
    for t in trials:
        name = t["name"]
        coords = np.array(t["w1_w2_path"])
        ax4.plot(coords[:, 0], coords[:, 1], color=colors[name], lw=2.2, alpha=0.9, label=name)
        ax4.scatter(coords[0, 0], coords[0, 1], color="#F8FAFC", s=50, marker="o", zorder=5)
        ax4.scatter(coords[-1, 0], coords[-1, 1], color=colors[name], s=80, marker="*", zorder=6)

    ax4.text(coords[0, 0], coords[0, 1] - 0.02, "  Start (0.7, -0.5)", color="#F8FAFC", fontsize=8.5, va="top")
    ax4.set_title("Optimization Path in Output Weight Subspace $(w_1, w_2)$", fontsize=11, color="#38BDF8", pad=8)
    ax4.set_xlabel(r"Output Weight $w_1$ (synapse $h_1 \to \hat{y}$)", color="#94A3B8", fontsize=9)
    ax4.set_ylabel(r"Output Weight $w_2$ (synapse $h_2 \to \hat{y}$)", color="#94A3B8", fontsize=9)
    ax4.grid(True, linestyle=":", color="#1E293B", alpha=0.6)
    ax4.legend(loc="best", fontsize=8.5, facecolor="#151A24", edgecolor="#334155")

    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.90, wspace=0.20, hspace=0.28)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"[Module 4] Saved single-point optimizer benchmark figure: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def run_module_4(save_dir: str = "exports", show: bool = False):
    print("=" * 80)
    print("MODULE 4: Custom Optimizers on the Canonical Single Data Point")
    print(f"Data Point: x = [{X_SINGLE[0,0]:.2f}, {X_SINGLE[0,1]:.2f}], target y = {Y_SINGLE[0,0]:.1f}")
    print("=" * 80)

    steps = 35

    # 1. Standard SGD
    print("\n[1/3] Running Standard SGD (lr=0.10)...")
    sgd_res, _ = train_single_point_optimizer("Standard SGD", SGD, {"lr": 0.10}, steps=steps)
    print(f"  -> SGD Step 1 Loss: {sgd_res['loss'][0]:.6f} | Step {steps} Loss: {sgd_res['loss'][-1]:.6f} (y_hat: {sgd_res['y_hat'][-1]:.4f})")

    # 2. SGD with Momentum
    print("\n[2/3] Running SGD with Momentum (lr=0.10, beta=0.9)...")
    mom_res, _ = train_single_point_optimizer("SGD + Momentum", SGDMomentum, {"lr": 0.10, "beta": 0.90}, steps=steps)
    print(f"  -> Momentum Step 1 Loss: {mom_res['loss'][0]:.6f} | Step {steps} Loss: {mom_res['loss'][-1]:.6f} (y_hat: {mom_res['y_hat'][-1]:.4f})")

    # 3. Adam
    print("\n[3/3] Running Adam (lr=0.05, beta1=0.9, beta2=0.999)...")
    adam_res, _ = train_single_point_optimizer("Adam (Adaptive)", Adam, {"lr": 0.05, "beta1": 0.90, "beta2": 0.999, "eps": 1e-8}, steps=steps)
    print(f"  -> Adam Step 1 Loss: {adam_res['loss'][0]:.6f} | Step {steps} Loss: {adam_res['loss'][-1]:.6f} (y_hat: {adam_res['y_hat'][-1]:.4f})")

    # Step 1 Parameter Update Ledger
    print("\n" + "=" * 80)
    print("STEP 1 PARAMETER UPDATE LEDGER: EXACT PAPER COMPARISON (Initial -> Step 1)")
    print("=" * 80)
    print(f"{'Parameter':<12} | {'Initial':<10} | {'SGD (lr=0.1)':<14} | {'Momentum':<14} | {'Adam (lr=0.05)':<14}")
    print("-" * 80)
    p_sgd = sgd_res["step1_params"]
    p_mom = mom_res["step1_params"]
    p_adm = adam_res["step1_params"]

    print(f"{'w1 (h1->y)':<12} | {W2_INIT[0,0]:<10.4f} | {p_sgd['W2'][0,0]:<14.4f} | {p_mom['W2'][0,0]:<14.4f} | {p_adm['W2'][0,0]:<14.4f}")
    print(f"{'w2 (h2->y)':<12} | {W2_INIT[1,0]:<10.4f} | {p_sgd['W2'][1,0]:<14.4f} | {p_mom['W2'][1,0]:<14.4f} | {p_adm['W2'][1,0]:<14.4f}")
    print(f"{'b2 (bias)':<12} | {B2_INIT[0,0]:<10.4f} | {p_sgd['b2'][0,0]:<14.4f} | {p_mom['b2'][0,0]:<14.4f} | {p_adm['b2'][0,0]:<14.4f}")
    print(f"{'w11 (x1->h1)':<12} | {W1_INIT[0,0]:<10.4f} | {p_sgd['W1'][0,0]:<14.4f} | {p_mom['W1'][0,0]:<14.4f} | {p_adm['W1'][0,0]:<14.4f}")
    print(f"{'w21 (x2->h1)':<12} | {W1_INIT[1,0]:<10.4f} | {p_sgd['W1'][1,0]:<14.4f} | {p_mom['W1'][1,0]:<14.4f} | {p_adm['W1'][1,0]:<14.4f}")
    print(f"{'b1_1 (h1)':<12} | {B1_INIT[0,0]:<10.4f} | {p_sgd['b1'][0,0]:<14.4f} | {p_mom['b1'][0,0]:<14.4f} | {p_adm['b1'][0,0]:<14.4f}")
    print(f"{'w12 (x1->h2)':<12} | {W1_INIT[0,1]:<10.4f} | {p_sgd['W1'][0,1]:<14.4f} | {p_mom['W1'][0,1]:<14.4f} | {p_adm['W1'][0,1]:<14.4f}")
    print(f"{'w22 (x2->h2)':<12} | {W1_INIT[1,1]:<10.4f} | {p_sgd['W1'][1,1]:<14.4f} | {p_mom['W1'][1,1]:<14.4f} | {p_adm['W1'][1,1]:<14.4f}")
    print(f"{'b1_2 (h2)':<12} | {B1_INIT[0,1]:<10.4f} | {p_sgd['b1'][0,1]:<14.4f} | {p_mom['b1'][0,1]:<14.4f} | {p_adm['b1'][0,1]:<14.4f}")
    print("=" * 80 + "\n")

    save_path = os.path.join(save_dir, "module4_optimizer_benchmark.png")
    plot_single_point_benchmark([sgd_res, mom_res, adam_res], save_path=save_path, show=show)

    # Does the engine behave exactly like the textbook update rules used by the film below?
    crosscheck_with_engine([sgd_res, mom_res, adam_res])

    return [sgd_res, mom_res, adam_res]


# ==============================================================================
# ==============================================================================
#  VISUALIZATION SECTION  --  "Three Ways to Take a Step"
#
#  A minimalist film in the same visual language as the Module 2 calculus film:
#  paper canvas, ink text, hairline axes, Computer Modern math, one idea at a time.
#
#  Colour meaning (fixed in every chapter)
#      BLUE   = Standard SGD
#      OCHRE  = SGD + Momentum
#      GREEN  = Adam            (also: the target y = 1)
#      RED    = the slope / gradient / the answer of a worked step
#
#  The film is self-contained: it re-implements the network, the backward pass and the three
#  update rules in plain NumPy (see ref_train), so it runs without engine/. When the engine
#  benchmark runs, crosscheck_with_engine() confirms both give the same numbers.
#
#  Chapters: intro, sgd, momentum, adam, together, summary
# ==============================================================================
# ==============================================================================
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import animation

# ------------------------------------------------------------------ palette (quiet, print-like)
BG = "#FBFAF7"      # paper
INK = "#1C1C1C"     # primary text and the loss curve itself
MID = "#5C5A55"     # secondary text
GREY = "#9A978F"    # axes, ticks, captions
LIGHT = "#D6D3CA"   # hairlines
GRID = "#ECE9E2"    # grid
BLUE = "#2D5B9A"    # SGD
OCHRE = "#B5832A"   # Momentum
GREEN = "#3C7A5A"   # Adam, target
RED = "#C4452F"     # slope / gradient / answer

# ------------------------------------------------------------------ hyper-parameters (same as run_module_4)
FILM_STEPS = 35
SGD_LR = 0.10
MOM_LR, MOM_BETA = 0.10, 0.90
ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS = 0.05, 0.90, 0.999, 1e-8

# parameter order; w_ij is the weight on the synapse x_i -> h_j, i.e. W1_INIT[i, j]
PARAM_KEYS = ["w11", "w21", "w12", "w22", "bh1", "bh2", "w1", "w2", "bo"]
IW1 = PARAM_KEYS.index("w1")      # the output weight we follow through the worked examples


def _theta0():
    return np.array([W1_INIT[0, 0], W1_INIT[1, 0], W1_INIT[0, 1], W1_INIT[1, 1],
                     B1_INIT[0, 0], B1_INIT[0, 1], W2_INIT[0, 0], W2_INIT[1, 0], B2_INIT[0, 0]], dtype=float)


# ------------------------------------------------------------------ reference math (analytical, no autograd)
def _fwd_bwd(th):
    """One forward + backward pass of the canonical network. Returns (y_hat, loss, gradient[9])."""
    w11, w21, w12, w22, bh1, bh2, w1, w2, bo = th
    x1, x2, y = float(X_SINGLE[0, 0]), float(X_SINGLE[0, 1]), float(Y_SINGLE[0, 0])
    z1, z2 = x1 * w11 + x2 * w21 + bh1, x1 * w12 + x2 * w22 + bh2
    h1, h2 = 1.0 / (1.0 + np.exp(-z1)), 1.0 / (1.0 + np.exp(-z2))
    yh = h1 * w1 + h2 * w2 + bo
    gy = 2.0 * (yh - y)
    gz1, gz2 = gy * w1 * h1 * (1 - h1), gy * w2 * h2 * (1 - h2)
    g = np.array([x1 * gz1, x2 * gz1, x1 * gz2, x2 * gz2, gz1, gz2, h1 * gy, h2 * gy, gy])
    return yh, (yh - y) ** 2, g


def ref_train(key, steps=FILM_STEPS):
    """Textbook SGD / Momentum / Adam on the canonical point. States 0..steps, updates 0..steps-1."""
    th = _theta0()
    v, m, s = np.zeros(9), np.zeros(9), np.zeros(9)
    H = {k: [] for k in ("params", "grads", "loss", "yhat", "delta", "vel", "m", "v", "mhat", "vhat")}
    for t in range(steps + 1):
        yh, L, g = _fwd_bwd(th)
        H["params"].append(th.copy()); H["grads"].append(g); H["loss"].append(L); H["yhat"].append(yh)
        if t == steps:
            break
        if key == "sgd":
            d = -SGD_LR * g
        elif key == "momentum":
            v = MOM_BETA * v - MOM_LR * g
            d = v.copy()
            H["vel"].append(v.copy())
        else:
            m = ADAM_B1 * m + (1 - ADAM_B1) * g
            s = ADAM_B2 * s + (1 - ADAM_B2) * g * g
            mh, sh = m / (1 - ADAM_B1 ** (t + 1)), s / (1 - ADAM_B2 ** (t + 1))
            d = -ADAM_LR * mh / (np.sqrt(sh) + ADAM_EPS)
            H["m"].append(m.copy()); H["v"].append(s.copy()); H["mhat"].append(mh.copy()); H["vhat"].append(sh.copy())
        th = th + d
        H["delta"].append(d.copy())
    out = {k: np.array(val) for k, val in H.items()}
    out["step_norm"] = np.linalg.norm(out["delta"], axis=1)
    return out


HIST = {k: ref_train(k) for k in ("sgd", "momentum", "adam")}

OPTS = {
    "sgd": dict(num="1", short="SGD", engine="Standard SGD", color=BLUE,
                title="Standard SGD: step against the slope"),
    "momentum": dict(num="2", short="Momentum", engine="SGD + Momentum", color=OCHRE,
                     title="SGD + Momentum: keep the velocity"),
    "adam": dict(num="3", short="Adam", engine="Adam (Adaptive)", color=GREEN,
                 title="Adam: scale every step by its own history"),
}
ORDER = ("sgd", "momentum", "adam")


def crosscheck_with_engine(engine_results, tol=1e-8):
    """Compare the engine's loss / y_hat histories with the textbook rules used by the film."""
    key_of = {o["engine"]: k for k, o in OPTS.items()}
    print("=" * 80)
    print("CROSS-CHECK: engine optimizers vs. the textbook update rules used in the film")
    print("=" * 80)
    for res in engine_results:
        k = key_of.get(res["name"])
        if k is None:
            continue
        n = len(res["loss"])
        dl = float(np.max(np.abs(np.array(res["loss"]) - HIST[k]["loss"][:n])))
        dy = float(np.max(np.abs(np.array(res["y_hat"]) - HIST[k]["yhat"][:n])))
        flag = "OK" if max(dl, dy) < tol else "MISMATCH (check the engine's update rule / hyper-parameters)"
        print(f"  {res['name']:<18} max|dLoss| = {dl:.2e}   max|dy_hat| = {dy:.2e}   {flag}")
    print("=" * 80 + "\n")


# ------------------------------------------------------------------ tiny animation toolkit
def seg(t, a, b):
    """Progress 0..1 of time t inside [a, b]."""
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def ease(u):
    return u * u * (3 - 2 * u)


def E(t, a, b):
    return ease(seg(t, a, b))


def T(fig, x, y, s, size=15, color=INK, a=1.0, ha="left", weight="normal"):
    """Text in figure coordinates with alpha (skipped when invisible)."""
    if a <= 0.01:
        return
    fig.text(x, y, s, fontsize=size, color=color, alpha=float(a), ha=ha, va="center", fontweight=weight)


def rule(fig, x0, x1, y, a=1.0, color=LIGHT, lw=1.0):
    if a <= 0.01:
        return
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, lw=lw, alpha=float(a)))


def header(fig, num, title, a=1.0, col=MID):
    T(fig, 0.05, 0.94, num, size=22, color=col, a=a, weight="bold")
    T(fig, 0.075, 0.94, title, size=22, color=INK, a=a)
    rule(fig, 0.05, 0.95, 0.90, a=a)


def mk_ax(fig, rect, xlim, ylim, xl="", yl="", title="", xticks=None, yticks=None, logy=False):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=GREY, labelsize=10, length=3, width=0.8)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.yaxis.set_tick_params(which="minor", left=False)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xl, color=MID, fontsize=12)
    ax.set_ylabel(yl, color=MID, fontsize=12)
    if title:
        ax.set_title(title, color=INK, fontsize=13, pad=8, loc="left")
    return ax


def dot(ax, x, y, color=INK, ms=9, z=8):
    ax.plot([x], [y], "o", color=color, ms=ms, mec=BG, mew=1.6, zorder=z, clip_on=False)


def fL(x):
    """loss formatting: plain for ordinary values, scientific for tiny ones"""
    return "%.4f" % x if x >= 1e-3 else "%.1e" % x


# ------------------------------------------------------------------ shared pieces
def _progress(t, t0, dt, n=FILM_STEPS):
    """Step progress (0..n) with a smooth ease inside every single step."""
    if t <= t0:
        return 0.0
    q = (t - t0) / dt
    if q >= n:
        return float(n)
    i = np.floor(q)
    return float(i + ease(q - i))


def _ball(H, p, n=FILM_STEPS):
    """(completed steps k, prediction at progress p)"""
    k = min(int(np.floor(p)), n)
    i = min(k, n - 1)
    fr = 1.0 if p >= n else p - i
    return k, H["yhat"][i] + (H["yhat"][i + 1] - H["yhat"][i]) * fr


def _bowl(ax, t, t0=1.0, t1=2.5):
    """The loss bowl L = (y_hat - 1)^2 and the target line."""
    yy = np.linspace(0.15, 1.65, 300)
    nn = max(2, int(300 * E(t, t0, t1)))
    ax.plot(yy[:nn], (yy[:nn] - 1) ** 2, color=INK, lw=2.4)
    a = E(t, t0 + 0.5, t1)
    if a > 0.02:
        ax.axvline(1.0, color=GREEN, ls="--", lw=1.2, alpha=a)
        ax.text(1.02, 0.52, r"target  $y=1$", color=GREEN, fontsize=12, alpha=a)


def _bowl_axes(fig, rect, title):
    return mk_ax(fig, rect, (0.15, 1.65), (-0.03, 0.56), r"prediction $\hat y$", r"loss $L$", title=title,
                 xticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], yticks=[0, 0.2, 0.4])


LOGLIM = (1e-14, 2.0)
LOGTICKS = [1e-12, 1e-8, 1e-4, 1.0]


# ================================================================== SCENE 0: intro
def scene_intro(fig, t):
    T(fig, 0.5, 0.86, "M O D U L E   4", size=12, color=GREY, ha="center", a=E(t, 0.0, 0.8))
    T(fig, 0.5, 0.77, "Three Ways to Take a Step", size=44, ha="center", a=E(t, 0.2, 1.2))
    T(fig, 0.5, 0.68, "The gradient says which way is downhill. The optimizer decides how far to go.",
      size=15, color=MID, ha="center", a=E(t, 1.0, 2.0))
    rule(fig, 0.08, 0.92, 0.62, a=E(t, 1.5, 2.5))

    cols = [
        (0.20, "Standard SGD", BLUE, [r"$\theta\ \leftarrow\ \theta-\alpha\,g$"],
         r"$\alpha=%.2f$" % SGD_LR, "step against the slope", 3.0),
        (0.50, "SGD + Momentum", OCHRE, [r"$v\ \leftarrow\ \beta v-\alpha g$", r"$\theta\ \leftarrow\ \theta+v$"],
         r"$\alpha=%.2f,\ \ \beta=%.2f$" % (MOM_LR, MOM_BETA), "…and remember where you were heading", 4.6),
        (0.80, "Adam", GREEN, [r"$\theta\ \leftarrow\ \theta-\alpha\,\hat m\,/\,(\sqrt{\hat v}+\epsilon)$",
                               r"$\hat m,\ \hat v$: averages of $g$ and $g^2$"],
         r"$\alpha=%.2f,\ \beta_1=%.2f,\ \beta_2=%.3f$" % (ADAM_LR, ADAM_B1, ADAM_B2),
         "…and rescale every parameter's step", 6.2),
    ]
    for x, name, col, forms, hyper, desc, t0 in cols:
        T(fig, x, 0.54, name, size=21, color=col, ha="center", a=E(t, t0, t0 + 0.8), weight="bold")
        for j, f in enumerate(forms):
            T(fig, x, 0.45 - 0.07 * j, f, size=18 if j == 0 else 15, color=INK if j == 0 else MID, ha="center",
              a=E(t, t0 + 0.6 + 0.5 * j, t0 + 1.4 + 0.5 * j))
        T(fig, x, 0.30, hyper, size=14, color=MID, ha="center", a=E(t, t0 + 1.8, t0 + 2.6))
        T(fig, x, 0.24, desc, size=13, color=GREY, ha="center", a=E(t, t0 + 2.2, t0 + 3.0))

    rule(fig, 0.08, 0.92, 0.16, a=E(t, 9.0, 10.0))
    y0, L0 = HIST["sgd"]["yhat"][0], HIST["sgd"]["loss"][0]
    T(fig, 0.5, 0.10,
      r"Same network  ·  same start  ·  same data point $x=(%.2f,\ %.2f),\ y=%.0f$   $\Rightarrow$   $\hat y_0=%.4f,\ L_0=%.4f$   ·   %d steps each"
      % (X_SINGLE[0, 0], X_SINGLE[0, 1], Y_SINGLE[0, 0], y0, L0, FILM_STEPS),
      size=14, color=MID, ha="center", a=E(t, 9.5, 10.8))


# ================================================================== SCENES 1-3: one optimizer at a time
T_RUN0 = 13.0       # the 35 steps start here
STEP_DT = 0.42      # seconds per step


def _worked(key):
    """The worked first step on w1 as (y, text, size, colour, t_start) rows."""
    H = HIST[key]
    g = H["grads"][0][IW1]
    w0, w1n = H["params"][0][IW1], H["params"][1][IW1]
    rows = [(0.68, r"Worked example: output weight $w_1$ (synapse $h_1\to\hat y$)", 14, MID, 2.6),
            (0.62, "1.  gradient from the backward pass", 13, MID, 3.0),
            (0.57, r"$g=\partial L/\partial w_1=%.4f$" % g, 17, INK, 3.6)]
    if key == "sgd":
        rows += [(0.50, "2.  scale by the learning rate", 13, MID, 5.0),
                 (0.45, r"$\alpha g=%.2f\cdot(%.4f)=%.4f$" % (SGD_LR, g, SGD_LR * g), 17, INK, 5.6),
                 (0.38, "3.  step against the gradient", 13, MID, 7.2),
                 (0.33, r"$w_1=%.4f-(%.4f)=%.4f$" % (w0, SGD_LR * g, w1n), 17, RED, 7.8),
                 (0.24, "No memory: every step looks only at the current slope.", 14, MID, 9.5)]
        rule_txt, size, hyper = r"$\theta\ \leftarrow\ \theta-\alpha\,g$", 28, r"learning rate $\alpha=%.2f$,   applied to all 9 parameters" % SGD_LR
    elif key == "momentum":
        v1 = H["vel"][0][IW1]
        v2, g2 = H["vel"][1][IW1], H["grads"][1][IW1]
        rows += [(0.50, "2.  update the velocity (it starts at 0)", 13, MID, 5.0),
                 (0.45, r"$v=%.2f\cdot0-%.2f\cdot(%.4f)=%+.4f$" % (MOM_BETA, MOM_LR, g, v1), 17, INK, 5.6),
                 (0.38, "3.  move by the velocity", 13, MID, 7.2),
                 (0.33, r"$w_1=%.4f+%.4f=%.4f$" % (w0, v1, w1n), 17, RED, 7.8),
                 (0.25, "Step 1 equals SGD: there is no past yet.", 14, MID, 9.2),
                 (0.19, "Step 2 already differs, because $v$ remembers step 1:", 14, MID, 10.2),
                 (0.13, r"$v=%.2f\cdot(%.4f)-%.2f\cdot(%.4f)=%+.4f$" % (MOM_BETA, v1, MOM_LR, g2, v2), 16, OCHRE, 11.0)]
        rule_txt, size, hyper = r"$v\leftarrow\beta v-\alpha g,\qquad\theta\leftarrow\theta+v$", 24, r"$\alpha=%.2f,\ \ \beta=%.2f$,   $v$ starts at 0" % (MOM_LR, MOM_BETA)
    else:
        m, v, mh, vh = H["m"][0][IW1], H["v"][0][IW1], H["mhat"][0][IW1], H["vhat"][0][IW1]
        rows += [(0.50, "2.  running averages of $g$ and $g^2$ (start at 0)", 13, MID, 4.6),
                 (0.45, r"$m=%.1fg=%.4f,\quad v=%.3fg^2=%.5f$" % (1 - ADAM_B1, m, 1 - ADAM_B2, v), 16, INK, 5.2),
                 (0.38, "3.  bias correction: divide by $1-\\beta^t$", 13, MID, 6.4),
                 (0.33, r"$\hat m=%.4f,\quad \hat v=%.4f$" % (mh, vh), 16, INK, 7.0),
                 (0.26, r"4.  step = $\alpha\,\hat m\,/\,\sqrt{\hat v}$", 13, MID, 8.2),
                 (0.20, r"$w_1=%.4f-%.2f\cdot\dfrac{%.4f}{\sqrt{%.4f}}=%.4f$" % (w0, ADAM_LR, mh, vh, w1n), 16, RED, 8.8),
                 (0.11, r"First step is $\alpha\cdot\mathrm{sign}(g)$ for every parameter.", 13, MID, 10.2)]
        rule_txt, size, hyper = (r"$\theta\ \leftarrow\ \theta-\alpha\,\hat m\,/\,(\sqrt{\hat v}+\epsilon)$", 21,
                                 r"$\alpha=%.2f,\ \ \beta_1=%.2f,\ \ \beta_2=%.3f$" % (ADAM_LR, ADAM_B1, ADAM_B2))
    return rule_txt, size, hyper, rows


def _comments(key):
    """Commentary shown during the run: (step_threshold, text, colour)."""
    H = HIST[key]
    n = FILM_STEPS
    pk = int(np.argmax(H["yhat"]))
    out = []
    if key == "sgd":
        out = [(0, "No memory: every step uses only the current slope.", MID),
               (6, "Near the bottom the slope flattens, so the steps shrink by themselves.", MID),
               (n, r"After %d steps:  $L=%s$,  $\hat y=%.6f$" % (n, fL(H["loss"][-1]), H["yhat"][-1]), RED)]
    elif key == "momentum":
        out = [(0, "The velocity $v$ adds up past gradients.", MID),
               (max(pk, 3), r"$\hat y$ passes the target, but $v$ keeps pushing: peak $\hat y=%.3f$ at step %d." % (H["yhat"][pk], pk), MID),
               (n, r"After %d steps:  $L=%s$,  still swinging around the target." % (n, fL(H["loss"][-1])), RED)]
    else:
        out = [(0, r"Step size is set by $\alpha$ and $\hat m/\sqrt{\hat v}$, not by the raw gradient.", MID),
               (max(pk, 3), r"$\hat y$ overshoots to %.3f at step %d, then settles slowly." % (H["yhat"][pk], pk), MID),
               (n, r"After %d steps:  $L=%s$" % (n, fL(H["loss"][-1])), RED)]
    return out


def _closing(key):
    return {"sgd": "One rule: look at the slope, step the other way, repeat.",
            "momentum": "Momentum builds speed, then has to unwind it.",
            "adam": "Adam rescales each parameter's step: handy when gradients differ wildly in size."}[key]


def scene_optimizer(fig, t, key):
    H, M = HIST[key], OPTS[key]
    col = M["color"]
    n = FILM_STEPS
    header(fig, M["num"], M["title"], E(t, 0, 0.8), col=col)
    T(fig, 0.05, 0.86, r"The dot is the network's $(\hat y,\ L)$; each step moves all nine parameters.",
      size=15, color=MID, a=E(t, 0.8, 1.6))

    ax = _bowl_axes(fig, [0.07, 0.40, 0.48, 0.40], "")
    _bowl(ax, t)
    p = _progress(t, T_RUN0, STEP_DT)
    k, yv = _ball(H, p)
    y0, L0 = H["yhat"][0], H["loss"][0]

    # slope of the bowl at the start (this is the gradient the backward pass found)
    ta = E(t, 3.5, 4.5) * (1 - E(t, T_RUN0 - 1.0, T_RUN0))
    if ta > 0.02:
        sl = 2 * (y0 - 1)
        xs = np.array([y0 - 0.30, y0 + 0.30])
        ax.plot(xs, L0 + sl * (xs - y0), color=RED, lw=2.6, alpha=ta)
        ax.text(y0 + 0.10, L0 + 0.015, r"slope $=%.3f$" % sl, color=RED, fontsize=12, alpha=ta, va="center")
    if t >= T_RUN0:
        ax.plot(H["yhat"][:k + 1], H["loss"][:k + 1], "-o", color=col, lw=0.8, ms=4, mec=col, alpha=0.6)
    if E(t, 2.0, 3.0) > 0.02:
        dot(ax, yv, (yv - 1) ** 2, col if t >= T_RUN0 else INK, ms=12 * E(t, 2.0, 3.0), z=9)
        if t < T_RUN0:
            ax.text(y0 - 0.03, L0 + 0.03, r"start", color=INK, fontsize=12, ha="right")

    # ---- the two small charts under the bowl
    if t >= T_RUN0 - 0.5:
        a1 = mk_ax(fig, [0.09, 0.08, 0.19, 0.20], (0, n), LOGLIM, "step  t", "", title="loss (log scale)",
                   xticks=[0, 10, 20, 30], yticks=LOGTICKS, logy=True)
        a1.plot(np.arange(k + 1), np.maximum(H["loss"][:k + 1], 1e-14), "-o", color=col, lw=2.0, ms=3, mec=col)
        dmax = float(np.abs(H["delta"][:, IW1]).max()) * 1.15
        a2 = mk_ax(fig, [0.36, 0.08, 0.19, 0.20], (0, n), (0, dmax), "step  t", "", title=r"$|\Delta w_1|$ per step",
                   xticks=[0, 10, 20, 30])
        kk = min(k, n)
        a2.plot(np.arange(1, kk + 1), np.abs(H["delta"][:kk, IW1]), "-o", color=col, lw=2.0, ms=3, mec=col)

    # ---- right column
    X = 0.60
    rule_txt, size, hyper, rows = _worked(key)
    T(fig, X, 0.84, rule_txt, size=size, a=E(t, 0.8, 1.8))
    T(fig, X, 0.77, hyper, size=14, color=MID, a=E(t, 1.8, 2.8))
    a_w = 1 - E(t, T_RUN0 - 1.0, T_RUN0)
    for y, s, sz, c, t0 in rows:
        T(fig, X, y, s, size=sz, color=c, a=E(t, t0, t0 + 0.7) * a_w)

    a_r = E(t, T_RUN0, T_RUN0 + 0.8)
    if a_r > 0.02:
        kk = min(k, n)
        T(fig, X, 0.68, r"step  $t=%d\ /\ %d$" % (kk, n), size=19, color=col, a=a_r, weight="bold")
        T(fig, X, 0.61, r"$\hat y=%.4f$" % H["yhat"][kk], size=17, a=a_r)
        T(fig, X, 0.55, r"loss  $L=%s$" % fL(H["loss"][kk]), size=17, a=a_r)
        T(fig, X, 0.49, r"$\|\Delta\theta\|=%.4f$" % (H["step_norm"][kk - 1] if kk > 0 else 0.0), size=17, a=a_r)
        if key == "sgd":
            extra = r"$g_{w_1}=%+.4f$" % H["grads"][kk][IW1]
        elif key == "momentum":
            extra = r"$v_{w_1}=%+.4f$" % (H["vel"][kk - 1][IW1] if kk > 0 else 0.0)
        else:
            extra = r"$\hat m/\sqrt{\hat v}$ for $w_1=%+.3f$" % (H["mhat"][kk - 1][IW1] / (np.sqrt(H["vhat"][kk - 1][IW1]) + ADAM_EPS) if kk > 0 else 0.0)
        T(fig, X, 0.43, extra, size=15, color=RED, a=a_r)
        yy_ = 0.34
        for thr, txt, c in _comments(key):
            if kk >= thr:
                T(fig, X, yy_, txt, size=13, color=c, a=a_r)
                yy_ -= 0.065
    T(fig, 0.5, 0.03, _closing(key), size=15, color=INK, ha="center",
      a=E(t, T_RUN0 + n * STEP_DT + 0.8, T_RUN0 + n * STEP_DT + 2.0))


# ================================================================== SCENE 4: together
T_TOG0, TOG_DT = 3.0, 0.5


def scene_together(fig, t):
    n = FILM_STEPS
    header(fig, "4", "Side by side: same start, three behaviours", E(t, 0, 0.8))
    p = _progress(t, T_TOG0, TOG_DT)
    k = min(int(np.floor(p)), n)
    T(fig, 0.05, 0.86, r"All three begin at $\hat y_0=%.4f$ and take %d steps." % (HIST["sgd"]["yhat"][0], n),
      size=15, color=MID, a=E(t, 0.8, 1.6))
    T(fig, 0.95, 0.86, r"step  $%d\ /\ %d$" % (k, n), size=17, color=INK, ha="right", a=E(t, 2.0, 3.0), weight="bold")

    ax = _bowl_axes(fig, [0.07, 0.42, 0.48, 0.40], "")
    _bowl(ax, t, 0.8, 2.0)
    for key in ORDER:
        H, col = HIST[key], OPTS[key]["color"]
        _, yv = _ball(H, p)
        if t >= T_TOG0:
            ax.plot(H["yhat"][:k + 1], H["loss"][:k + 1], "-o", color=col, lw=0.7, ms=3.5, mec=col, alpha=0.5)
        a = E(t, 1.8, 2.8)
        if a > 0.02:
            dot(ax, yv, (yv - 1) ** 2, col, ms=12 * a, z=9)

    # live readout under the bowl
    for j, key in enumerate(ORDER):
        H, col, x = HIST[key], OPTS[key]["color"], 0.07 + 0.165 * j
        a = E(t, 2.0, 3.0)
        T(fig, x, 0.30, OPTS[key]["short"], size=16, color=col, a=a, weight="bold")
        T(fig, x, 0.24, r"$L=%s$" % fL(H["loss"][k]), size=14, a=a)
        T(fig, x, 0.19, r"$\hat y=%.3f$" % H["yhat"][k], size=14, a=a)
        T(fig, x, 0.14, r"$\|\Delta\theta\|=%.3f$" % (H["step_norm"][k - 1] if k > 0 else 0.0), size=14, a=a)

    # right: three stacked charts
    if t >= 2.0:
        a1 = mk_ax(fig, [0.66, 0.62, 0.29, 0.21], (0, n), LOGLIM, "", "", title="loss (log scale)",
                   xticks=[0, 10, 20, 30], yticks=LOGTICKS, logy=True)
        a2 = mk_ax(fig, [0.66, 0.36, 0.29, 0.17], (0, n), (0.2, 1.6), "", "", title=r"prediction $\hat y$  (target 1)",
                   xticks=[0, 10, 20, 30], yticks=[0.5, 1.0, 1.5])
        a2.axhline(1.0, color=GREEN, ls="--", lw=1.1)
        a3 = mk_ax(fig, [0.66, 0.10, 0.29, 0.17], (0, n), (0, 0.5), "step  t", "", title=r"step size $\|\Delta\theta\|$",
                   xticks=[0, 10, 20, 30], yticks=[0, 0.2, 0.4])
        for key in ORDER:
            H, col = HIST[key], OPTS[key]["color"]
            a1.plot(np.arange(k + 1), np.maximum(H["loss"][:k + 1], 1e-14), "-", color=col, lw=2.2)
            a2.plot(np.arange(k + 1), H["yhat"][:k + 1], "-", color=col, lw=2.2)
            if k > 0:
                a3.plot(np.arange(1, k + 1), H["step_norm"][:k], "-", color=col, lw=2.2)

    end = T_TOG0 + n * TOG_DT
    T(fig, 0.5, 0.05, r"SGD slides straight in. Momentum and Adam overshoot $\hat y=1$ and swing back.",
      size=15, color=INK, ha="center", a=E(t, end + 0.8, end + 2.0))


# ================================================================== SCENE 5: summary
def _metrics(key):
    H = HIST[key]
    first = next((i for i, l in enumerate(H["loss"]) if l < 1e-3), None)
    return dict(yfin=H["yhat"][-1], Lfin=H["loss"][-1], ypk=float(H["yhat"].max()),
                first=first, path=float(H["step_norm"].sum()), ups=int((np.diff(H["loss"]) > 0).sum()))


def scene_summary(fig, t):
    n = FILM_STEPS
    header(fig, "5", "The verdict, on this one problem", E(t, 0, 0.8))
    allw = np.vstack([HIST[k]["params"][:, [IW1, IW1 + 1]] for k in ORDER])
    lo, hi = allw.min(0), allw.max(0)
    pad = 0.12 * (hi - lo)
    ax = mk_ax(fig, [0.07, 0.15, 0.44, 0.67], (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]),
               r"output weight $w_1$  (synapse $h_1\to\hat y$)", r"output weight $w_2$  (synapse $h_2\to\hat y$)",
               title=r"paths in the $(w_1,\ w_2)$ plane")
    for key in ORDER:
        P_ = HIST[key]["params"][:, [IW1, IW1 + 1]]
        m = max(2, int((n + 1) * E(t, 1.5, 11.0)))
        ax.plot(P_[:m, 0], P_[:m, 1], "-o", color=OPTS[key]["color"], lw=1.8, ms=3, mec=OPTS[key]["color"], alpha=0.9)
        if m >= n + 1:
            ax.plot([P_[-1, 0]], [P_[-1, 1]], "*", color=OPTS[key]["color"], ms=15, mec=BG, mew=1.0, zorder=7)
    dot(ax, *HIST["sgd"]["params"][0, [IW1, IW1 + 1]], INK, ms=10)
    ax.text(HIST["sgd"]["params"][0, IW1] + 0.01, HIST["sgd"]["params"][0, IW1 + 1] - 0.02, "start", color=INK, fontsize=12)
    T(fig, 0.07, 0.04, "A projection: the other seven parameters move too.", size=12, color=GREY, a=E(t, 2.0, 3.0))

    X0 = 0.58
    cols = [0.74, 0.85, 0.96]
    for c, key in zip(cols, ORDER):
        T(fig, c, 0.80, OPTS[key]["short"], size=15, color=OPTS[key]["color"], ha="right", a=E(t, 1.5, 2.3), weight="bold")
    rule(fig, X0, 0.96, 0.765, a=E(t, 1.5, 2.3))
    M = {k: _metrics(k) for k in ORDER}
    rows = [("final prediction  $\\hat y$", lambda m: "%.4f" % m["yfin"]),
            ("final loss  $L$", lambda m: fL(m["Lfin"])),
            ("peak prediction", lambda m: "%.3f" % m["ypk"]),
            ("first step with $L<10^{-3}$", lambda m: "never" if m["first"] is None else "%d" % m["first"]),
            ("steps where $L$ went up", lambda m: "%d" % m["ups"]),
            ("total distance moved", lambda m: "%.3f" % m["path"])]
    for i, (lab, fn) in enumerate(rows):
        y = 0.71 - 0.07 * i
        a = E(t, 2.5 + 0.6 * i, 3.2 + 0.6 * i)
        T(fig, X0, y, lab, size=13, color=MID, a=a)
        for c, key in zip(cols, ORDER):
            T(fig, c, y, fn(M[key]), size=14, ha="right", a=a)

    best = min(ORDER, key=lambda k: M[k]["Lfin"])
    over = [OPTS[k]["short"] for k in ORDER if M[k]["ypk"] > 1.001]
    T(fig, X0, 0.27, r"Lowest final loss after %d steps:  %s  ($L=%s$)" % (n, OPTS[best]["short"], fL(M[best]["Lfin"])),
      size=14, color=OPTS[best]["color"], a=E(t, 7.0, 8.0), weight="bold")
    if over:
        T(fig, X0, 0.20, r"%s overshoot, so their loss rises and falls." % " and ".join(over),
          size=13, color=INK, a=E(t, 8.5, 9.5))
    T(fig, X0, 0.12, "One point, one smooth bowl, hand-picked learning rates:", size=11.5, color=MID, a=E(t, 10.5, 11.5))
    T(fig, X0, 0.08, "this is not a ranking of optimizers in general.", size=11.5, color=MID, a=E(t, 10.5, 11.5))
    T(fig, X0, 0.04, "Their strengths show up with noisy, badly scaled gradients.", size=11.5, color=MID, a=E(t, 11.5, 12.5))


# ------------------------------------------------------------------ film runner
SCENES = [("intro", 14.0, scene_intro),
          ("sgd", 31.0, lambda f, t: scene_optimizer(f, t, "sgd")),
          ("momentum", 31.0, lambda f, t: scene_optimizer(f, t, "momentum")),
          ("adam", 31.0, lambda f, t: scene_optimizer(f, t, "adam")),
          ("together", 26.0, scene_together),
          ("summary", 16.0, scene_summary)]
PREVIEW_TIMES = {"intro": [3.0, 12.0],
                 "sgd": [4.5, 8.5, 20.0, 30.0],
                 "momentum": [8.5, 12.0, 20.0, 30.0],
                 "adam": [4.5, 10.5, 20.0, 30.0],
                 "together": [8.0, 14.0, 24.0],
                 "summary": [12.0, 15.0]}


def _film_style():
    plt.style.use("default")      # the benchmark above switches matplotlib to a dark theme
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


def draw_frame(fig, scene, dur, t):
    fig.clf()
    fig.patch.set_facecolor(BG)
    scene(fig, t)
    a = max(1 - E(t, 0, 0.5), E(t, dur - 0.5, dur))
    if a > 0.01:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor="none",
                                 alpha=a, zorder=1000))


def render_film(scene="all", preview=False, fps=20, dpi=80, fmt="auto", out="exports"):
    _film_style()
    os.makedirs(out, exist_ok=True)
    todo = [s for s in SCENES if scene in ("all", s[0])]
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor=BG)

    if preview:
        for name, dur, fn in todo:
            for t in PREVIEW_TIMES[name]:
                draw_frame(fig, fn, dur, t)
                path = os.path.join(out, f"module4_preview_{name}_{t:05.1f}.png")
                fig.savefig(path, facecolor=BG)
                print("saved", path)
        plt.close(fig)
        return

    if fmt == "auto":
        fmt = "mp4" if animation.writers.is_available("ffmpeg") else "gif"
    tag = scene if scene != "all" else "full"
    path = os.path.join(out, f"module4_optimizers_film_{tag}.{fmt}")
    writer = (animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
              if fmt == "mp4" else animation.PillowWriter(fps=min(fps, 15)))
    total = sum(int(d * fps) for _, d, _ in todo)
    done = 0
    with writer.saving(fig, path, dpi):
        for name, dur, fn in todo:
            for i in range(int(dur * fps)):
                draw_frame(fig, fn, dur, i / fps)
                writer.grab_frame(facecolor=BG)
                done += 1
                if done % 40 == 0:
                    print(f"\r[{name:<9}] {done}/{total} frames", end="", flush=True)
    print(f"\nSaved {path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4: Single-Point Optimizer Benchmark + Film")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    parser.add_argument("--save-dir", type=str, default="exports", help="Directory for output figure")
    parser.add_argument("--film", action="store_true", help="also render the optimizer film")
    parser.add_argument("--film-only", action="store_true", help="render only the film (no engine run)")
    parser.add_argument("--film-scene", default="all", choices=["all"] + [s[0] for s in SCENES])
    parser.add_argument("--preview", action="store_true", help="film: save PNG stills instead of a video")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=80, help="80 -> 1280x720, 120 -> 1920x1080")
    parser.add_argument("--format", default="auto", choices=["auto", "mp4", "gif"])
    args = parser.parse_args()

    if not args.film_only:
        run_module_4(save_dir=args.save_dir, show=args.show)
    if args.film or args.film_only or args.preview:
        render_film(scene=args.film_scene, preview=args.preview, fps=args.fps, dpi=args.dpi,
                    fmt=args.format, out=args.save_dir)