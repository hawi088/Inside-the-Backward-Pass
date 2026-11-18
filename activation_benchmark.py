"""
Module 5 - "Inside the Backward Pass"  |  minimalist activation-function film
Pure NumPy + Matplotlib (no autograd / DL frameworks, no Manim, no engine package).

Same design language as the Module 2 calculus film:
  * One paper-white canvas, ink-black text, hairline axes, Computer Modern math.
  * One idea at a time. Every step is written on the right, in order.
  * Colour carries meaning, and the meaning never changes between chapters:
        AMBER = Sigmoid        GREEN = ReLU        BLUE = Tanh      (the three contestants)
        RED   = slope / gradient / "the thing we are after"
        INK   = the canonical neuron h1,  GREY = the canonical neuron h2

Canonical data point (identical to the capstone):
    x = [0.5, 0.8], target y = 1.0, network 2 -> 2 -> 1, MSE loss, Adam lr = 0.03
    W1 = [[0.5, 0.8], [-0.6, 0.4]]  (one ROW per hidden neuron)   b1 = [0.1, -0.2]
    W2 = [0.7, -0.5]                                              b2 = 0.3
    => z1 = +0.990, z2 = -0.180  (the same operating points your static figure marks)

Eight chapters:
  1. Why an activation function at all   two linear layers collapse into one
  2. Sigmoid   definition, graph, slope traced live, closed-form derivative, our two neurons
  3. ReLU      same story
  4. Tanh      same story
  5. Side by side   the three curves and the three slopes on the same axes
  6. Step 1, forward    the same weights give three different predictions
  7. Step 1, backward   the same loss gives three different gradients (ReLU kills neuron 2)
  8. Training   30 Adam steps: loss, prediction, h1, h2, all three activations live

Usage
  python module5_activation_film.py                       # full film (mp4 if ffmpeg exists, else gif)
  python module5_activation_film.py --scene relu          # one chapter only
  python module5_activation_film.py --preview             # PNG stills of every chapter
  python module5_activation_film.py --fps 24 --dpi 100    # higher quality
Chapters: title, why, sigmoid, relu, tanh, compare, forward, backward, train, outro
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import animation

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

# ------------------------------------------------------------------ palette (quiet, print-like)
BG = "#FBFAF7"
INK = "#1C1C1C"
MID = "#5C5A55"
GREY = "#9A978F"
LIGHT = "#D6D3CA"
GRID = "#ECE9E2"
BLUE = "#2D5B9A"      # Tanh
GREEN = "#3C7A5A"     # ReLU
AMBER = "#B7832F"     # Sigmoid
RED = "#C4452F"       # slope / gradient


# ------------------------------------------------------------------ math (analytical, no autograd)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    return (np.asarray(z) > 0).astype(float)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2


NAMES = ["Sigmoid", "ReLU", "Tanh"]
ACT = {
    "Sigmoid": (sigmoid, d_sigmoid, AMBER),
    "ReLU": (relu, d_relu, GREEN),
    "Tanh": (np.tanh, d_tanh, BLUE),
}

# canonical network (rows of W1 = neurons)
X0 = np.array([0.5, 0.8])
Y0 = 1.0
W1_0 = np.array([[0.5, 0.8], [-0.6, 0.4]])
B1_0 = np.array([0.1, -0.2])
W2_0 = np.array([0.7, -0.5])
B2_0 = 0.3
Z_CAN = W1_0 @ X0 + B1_0            # [+0.990, -0.180]


def audit(name):
    """Exact Step-1 forward + backward numbers for one activation."""
    f, d, _ = ACT[name]
    z = W1_0 @ X0 + B1_0
    h = f(z)
    yh = float(W2_0 @ h + B2_0)
    loss = (yh - Y0) ** 2
    dL = 2.0 * (yh - Y0)
    dh = dL * W2_0
    sl = d(z)
    dz = dh * sl
    return dict(z=z, h=h, yh=yh, loss=loss, dL=dL, dh=dh, sl=sl, dz=dz)


def train(name, steps=30, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
    """Adam on the single point, exactly like the capstone's train_single_point_activation."""
    f, d, _ = ACT[name]
    P = [W1_0.copy(), B1_0.copy(), W2_0.copy(), np.array(B2_0)]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    H = dict(loss=[], yh=[], h1=[], h2=[])
    for s in range(1, steps + 1):
        W1, bb1, W2, bb2 = P
        z = W1 @ X0 + bb1
        h = f(z)
        yh = float(W2 @ h + bb2)
        H["loss"].append((yh - Y0) ** 2)
        H["yh"].append(yh)
        H["h1"].append(float(h[0]))
        H["h2"].append(float(h[1]))
        dL = 2.0 * (yh - Y0)
        dz = dL * W2 * d(z)
        G = [np.outer(dz, X0), dz, dL * h, np.array(dL)]
        for i in range(4):
            m[i] = b1 * m[i] + (1 - b1) * G[i]
            v[i] = b2 * v[i] + (1 - b2) * G[i] ** 2
            P[i] -= lr * (m[i] / (1 - b1 ** s)) / (np.sqrt(v[i] / (1 - b2 ** s)) + eps)
    return {k: np.array(x) for k, x in H.items()}


AUD = {n: audit(n) for n in NAMES}
HIST = {n: train(n) for n in NAMES}
N_STEPS = 30


# ------------------------------------------------------------------ tiny animation toolkit (same as Module 2)
def seg(t, a, b):
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def ease(u):
    return u * u * (3 - 2 * u)


def E(t, a, b):
    return ease(seg(t, a, b))


def T(fig, x, y, s, size=15, color=INK, a=1.0, ha="left", weight="normal"):
    if a <= 0.01:
        return
    fig.text(x, y, s, fontsize=size, color=color, alpha=float(a), ha=ha, va="center", fontweight=weight)


def rule(fig, x0, x1, y, a=1.0, color=LIGHT, lw=1.0):
    if a <= 0.01:
        return
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, lw=lw, alpha=float(a)))


def header(fig, num, title, a=1.0):
    T(fig, 0.05, 0.94, num, size=22, color=BLUE, a=a, weight="bold")
    T(fig, 0.075, 0.94, title, size=22, color=INK, a=a)
    rule(fig, 0.05, 0.95, 0.90, a=a)


def mk_ax(fig, rect, xlim, ylim, xl="", yl="", center=False, title="", grid=True, xticks=None, yticks=None):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
        ax.spines[s].set_linewidth(1.0)
        if center:
            ax.spines[s].set_position("zero")
    ax.tick_params(colors=GREY, labelsize=10, length=3, width=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if grid:
        ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if center:
        ax.text(xlim[1], 0, xl, ha="right", va="bottom", color=MID, fontsize=13, clip_on=False)
        ax.text(0, ylim[1], " " + yl, ha="left", va="top", color=MID, fontsize=13, clip_on=False)
    else:
        ax.set_xlabel(xl, color=MID, fontsize=12)
        ax.set_ylabel(yl, color=MID, fontsize=12)
    if title:
        ax.set_title(title, color=INK, fontsize=13, pad=8, loc="left")
    return ax


def dot(ax, x, y, color=INK, ms=9, z=8):
    ax.plot([x], [y], "o", color=color, ms=ms, mec=BG, mew=1.6, zorder=z, clip_on=False)


# ================================================================== SCENE 0: title
def scene_title(fig, t):
    T(fig, 0.5, 0.74, "M O D U L E   5", size=12, color=GREY, ha="center", a=E(t, 0.0, 0.8))
    T(fig, 0.5, 0.64, "Inside the Backward Pass", size=44, ha="center", a=E(t, 0.2, 1.2))
    T(fig, 0.5, 0.55, "Activation functions   ·   Sigmoid   ·   ReLU   ·   Tanh",
      size=15, color=MID, ha="center", a=E(t, 1.0, 2.0))
    ax = fig.add_axes([0.34, 0.08, 0.32, 0.36])
    ax.axis("off")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.3, 2.4)
    zs = np.linspace(-3, 3, 200)
    for (nm, a0) in (("Sigmoid", 1.0), ("ReLU", 1.8), ("Tanh", 2.6)):
        n = int(200 * E(t, a0, a0 + 1.2))
        if n > 1:
            ax.plot(zs[:n], ACT[nm][0](zs[:n]), color=ACT[nm][2], lw=2.6)


# ================================================================== SCENE 1: why an activation function
def scene_why(fig, t):
    header(fig, "1", "Why a network needs an activation function", E(t, 0, 0.8))
    ax = mk_ax(fig, [0.06, 0.12, 0.46, 0.68], (-3, 3), (-2.2, 2.2), r"$x$", r"output", center=True)
    xs = np.linspace(-3, 3, 300)

    nl = int(300 * E(t, 6.0, 9.0))
    if nl > 1:
        ax.plot(xs[:nl], 0.6 * xs[:nl] + 0.2, color=BLUE, lw=2.8)
    if t > 8.5:
        ax.text(1.0, -1.35, "without " + r"$\varphi$" + ": always a straight line", color=BLUE, fontsize=12,
                ha="center", alpha=E(t, 8.5, 9.5))
    nc = int(300 * E(t, 12.5, 15.5))
    if nc > 1:
        curve = 1.2 * np.tanh(1.8 * xs + 0.6) - 0.6 * np.tanh(2.0 * xs - 1.5)
        ax.plot(xs[:nc], curve[:nc], color=GREEN, lw=2.8)
    if t > 15.2:
        ax.text(-0.15, 1.95, "with " + r"$\varphi$" + ": it can bend", color=GREEN, fontsize=12,
                ha="right", alpha=E(t, 15.2, 16.2))

    X = 0.60
    T(fig, X, 0.80, r"A neuron:   $z=w\cdot x+b$,   then   $h=\varphi(z)$", size=18, a=E(t, 1.0, 2.0))
    T(fig, X, 0.71, r"Remove $\varphi$ and stack two layers:", size=14, color=MID, a=E(t, 3.0, 4.0))
    T(fig, X, 0.63, r"$W_2(W_1x+b_1)+b_2$", size=19, a=E(t, 4.0, 5.0))
    T(fig, X, 0.56, r"$=(W_2W_1)\,x+(W_2b_1+b_2)$", size=19, a=E(t, 5.0, 6.0))
    T(fig, X, 0.47, "That is one linear layer. The extra depth bought nothing.", size=14, color=RED, a=E(t, 8.0, 9.0))
    T(fig, X, 0.38, r"Put $\varphi$ between the layers and they cannot be merged:", size=14, color=MID, a=E(t, 10.5, 11.5))
    T(fig, X, 0.31, r"$W_2\,\varphi(W_1x+b_1)+b_2$", size=19, a=E(t, 11.5, 12.5))
    T(fig, X, 0.22, "Now the model can bend.", size=16, color=GREEN, a=E(t, 14.0, 15.0))
    T(fig, X, 0.12, "Next: three choices of  " + r"$\varphi$" + ".   Their shape, their slope, their fate.",
      size=14, color=MID, a=E(t, 16.5, 17.5))


# ================================================================== SCENES 2-4: one activation each
SPEC = {
    "Sigmoid": dict(
        num="2", title="Sigmoid: the smooth squash",
        sub="Any real number goes in. A value between 0 and 1 comes out.",
        ylim=(-0.25, 1.25), yt=[0, 0.5, 1], dylim=(-0.06, 0.32), dyt=[0, 0.25],
        formula=r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$", fsize=26,
        dformula=r"$\sigma'(z)=\sigma(z)\,(1-\sigma(z))$", dsize=21,
        rng=r"output range:   $(0,\ 1)$,  never negative", shape="a smooth S, steepest in the middle",
        fact=r"the largest slope is only $0.25$, at $z=0$",
        comment=lambda z: ("flat tail: almost no gradient flows" if abs(z) >= 2.5 else
                           "steepest zone: gradient flows best" if abs(z) < 1.0 else "slope is fading"),
        neuron_note="Both slopes sit under the 0.25 cap: gradients shrink.",
        take="Smooth and bounded, but flat at both ends, so gradients shrink.",
    ),
    "ReLU": dict(
        num="3", title="ReLU: the one-way gate",
        sub="Positive numbers pass through untouched. Negative numbers are cut to zero.",
        ylim=(-0.8, 4.4), yt=[0, 2, 4], dylim=(-0.3, 1.45), dyt=[0, 1],
        formula=r"$\mathrm{ReLU}(z)=\max(0,\ z)$", fsize=24,
        dformula=r"$h'(z)=1\ \ (z>0),\qquad 0\ \ (z\leq0)$", dsize=20,
        rng=r"output range:   $[0,\ \infty)$,  unbounded above", shape="a flat floor, then a straight ramp",
        fact="the slope is exactly 1 or exactly 0, nothing in between",
        comment=lambda z: ("dead side: slope 0, gradient blocked" if z <= 0 else "slope 1: gradient passes untouched"),
        neuron_note=None,
        take="Cheap, never saturates for z > 0, but a neuron stuck at z < 0 gets no gradient.",
    ),
    "Tanh": dict(
        num="4", title="Tanh: the centered squash",
        sub="Any real number goes in. A value between -1 and 1 comes out, centered on 0.",
        ylim=(-1.4, 1.4), yt=[-1, 0, 1], dylim=(-0.25, 1.35), dyt=[0, 1],
        formula=r"$\tanh(z)=\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$", fsize=24,
        dformula=r"$\tanh'(z)=1-\tanh^2(z)$", dsize=22,
        rng=r"output range:   $(-1,\ 1)$,  centered on 0", shape="an S like sigmoid, but stretched and centered",
        fact=r"the largest slope is $1$, at $z=0$: four times sigmoid's",
        comment=lambda z: ("flat tail: almost no gradient flows" if abs(z) >= 2.2 else
                           "steepest zone: slope close to 1" if abs(z) < 0.8 else "slope is fading"),
        neuron_note=None,
        take="Like sigmoid but centered at 0 and four times steeper in the middle.",
    ),
}
SW0, SW1 = 8.5, 16.5     # sweep window


def scene_act(fig, t, name):
    sp = SPEC[name]
    f, d, col = ACT[name]
    header(fig, sp["num"], sp["title"], E(t, 0, 0.8))
    T(fig, 0.05, 0.865, sp["sub"], size=15, color=MID, a=E(t, 0.8, 1.6))

    zs = np.linspace(-4, 4, 400)
    # ---- top axes: h(z)
    ax = mk_ax(fig, [0.06, 0.50, 0.48, 0.31], (-4, 4), sp["ylim"], r"$z$", r"$h$", center=True,
               xticks=[-4, -2, 0, 2, 4], yticks=sp["yt"])
    n = max(2, int(400 * E(t, 2.0, 5.0)))
    ax.plot(zs[:n], f(zs[:n]), color=col, lw=2.8)

    # sweep position
    if t < SW0:
        z0 = -3.8
    elif t < SW1:
        z0 = -3.8 + 7.6 * E(t, SW0, SW1)
    else:
        z0 = 3.8
    sweeping = SW0 - 0.3 <= t < 17.6
    a_sw = E(t, SW0 - 0.3, SW0 + 0.4) * (1 - E(t, 17.0, 17.6))
    if sweeping and a_sw > 0.02:
        f0, m = float(f(z0)), float(d(z0))
        ends = np.array([z0 - 1.3, z0 + 1.3])
        ax.plot(ends, f0 + m * (ends - z0), color=RED, lw=2.4, alpha=a_sw)
        dot(ax, z0, f0, INK, ms=9 * a_sw, z=9)

    # ---- bottom axes: h'(z), traced as the point sweeps
    if t >= SW0 - 0.3:
        ab = mk_ax(fig, [0.06, 0.10, 0.48, 0.30], (-4, 4), sp["dylim"], r"$z$", r"$h'(z)$", center=True,
                   xticks=[-4, -2, 0, 2, 4], yticks=sp["dyt"])
        if t < SW1:
            zz = np.linspace(-3.8, z0, 160)
        else:
            zz = zs
        ab.plot(zz, d(zz), color=RED, lw=2.8)
        if t < 17.6:
            ab.plot([z0, z0], [0, float(d(z0))], color=RED, lw=1.2, ls=":")
            dot(ab, z0, float(d(z0)), RED, ms=9)

    X = 0.60
    # ---- right column phase A: definition
    T(fig, X, 0.80, "Definition", size=14, color=MID, a=E(t, 2.2, 3.0) * (1 - E(t, 7.6, 8.4)))
    T(fig, X, 0.71, sp["formula"], size=sp["fsize"], a=E(t, 2.8, 3.8))
    aA = E(t, 4.8, 5.6) * (1 - E(t, 7.6, 8.4))
    T(fig, X, 0.62, sp["rng"], size=15, color=col, a=aA)
    T(fig, X, 0.55, "shape: " + sp["shape"], size=14, color=MID, a=E(t, 5.6, 6.4) * (1 - E(t, 7.6, 8.4)))

    # ---- phase B: live readout
    a_r = E(t, SW0, SW0 + 0.8) * (1 - E(t, 17.0, 17.8))
    if a_r > 0.02:
        zr = z0
        T(fig, X, 0.62, r"$z=%+.2f$" % zr, size=18, a=a_r)
        T(fig, X, 0.55, r"$h(z)=%+.3f$" % float(f(zr)), size=18, color=col, a=a_r)
        T(fig, X, 0.48, r"slope $=%.3f$" % float(d(zr)), size=18, color=RED, a=a_r)
        T(fig, X, 0.39, sp["comment"](zr), size=14, color=MID, a=a_r)
    T(fig, X, 0.80, "Red tangent = slope at z.  Bottom graph records it.", size=13, color=RED, a=E(t, SW0 - 0.2, SW0 + 0.8) * (1 - E(t, 17.0, 17.8)))

    # ---- phase C: closed form
    T(fig, X, 0.62, "The red curve, in closed form", size=14, color=MID, a=E(t, 18.0, 18.8))
    T(fig, X, 0.54, sp["dformula"], size=sp["dsize"], color=RED, a=E(t, 18.8, 19.8))
    T(fig, X, 0.45, sp["fact"], size=14, color=MID, a=E(t, 20.0, 21.0))

    # ---- phase D: the two canonical hidden neurons
    if t >= 21.5:
        a_c = E(t, 21.5, 22.5)
        for k, (zc, cc, lab) in enumerate(zip(Z_CAN, (INK, GREY), (r"$h_1$", r"$h_2$"))):
            hc, sc = float(f(zc)), float(d(zc))
            ax.plot([zc, zc], [0, hc], color=cc, lw=1.3, ls=":", alpha=a_c)
            dot(ax, zc, hc, cc, ms=9 * a_c, z=9)
            ax.text(zc + (0.18 if k == 0 else -0.18), hc + (0.12 if sp["ylim"][1] < 3 else 0.3), lab, color=cc,
                    fontsize=13, ha="left" if k == 0 else "right", alpha=a_c)
            if t >= SW0 - 0.3:
                ab.plot([zc, zc], [0, sc], color=cc, lw=1.3, ls=":", alpha=a_c)
                dot(ab, zc, sc, cc, ms=9 * a_c, z=9)
        T(fig, X, 0.32, "Our two hidden neurons at Step 1", size=14, color=MID, a=E(t, 21.5, 22.5))
        for k, (cc, ya, t0) in enumerate(((INK, 0.25, 22.8), (GREY, 0.18, 24.6))):
            zc = Z_CAN[k]
            T(fig, X, ya, r"$h_%d$:  $z=%+.3f\ \to\ h=%+.4f$,  slope $%.4f$" % (k + 1, zc, float(f(zc)), float(d(zc))),
              size=14, color=cc if k == 0 else MID, a=E(t, t0, t0 + 0.9))
        note = sp["neuron_note"]
        if name == "ReLU":
            note = r"$h_2$ is exactly 0 with slope 0: no gradient reaches it."
        if name == "Tanh":
            note = r"$h_2$ keeps slope %.2f: nearly all gradient passes." % float(d(Z_CAN[1]))
        T(fig, X, 0.11, note, size=13, color=RED, a=E(t, 26.4, 27.4))
    T(fig, 0.5, 0.04, sp["take"], size=15, color=INK, ha="center", a=E(t, 28.5, 29.8))


def scene_sigmoid(fig, t):
    scene_act(fig, t, "Sigmoid")


def scene_relu(fig, t):
    scene_act(fig, t, "ReLU")


def scene_tanh(fig, t):
    scene_act(fig, t, "Tanh")


# ================================================================== SCENE 5: side by side
def scene_compare(fig, t):
    header(fig, "5", "Side by side: one input z, three different answers", E(t, 0, 0.8))
    for nm, x in (("Sigmoid", 0.05), ("ReLU", 0.14), ("Tanh", 0.205)):
        T(fig, x, 0.865, nm, size=16, color=ACT[nm][2], weight="bold", a=E(t, 1.0, 1.8))

    zs = np.linspace(-3, 3, 300)
    a1 = mk_ax(fig, [0.06, 0.19, 0.40, 0.60], (-3, 3), (-1.5, 3), r"$z$", r"$h$", center=True,
               title=r"what each one outputs,  $h(z)$", xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-1, 0, 1, 2, 3])
    a2 = mk_ax(fig, [0.56, 0.19, 0.40, 0.60], (-3, 3), (-0.3, 1.4), r"$z$", r"$h'(z)$", center=True,
               title=r"how steep each one is,  $h'(z)$", xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[0, 0.25, 0.5, 1])
    for i, nm in enumerate(NAMES):
        f, d, col = ACT[nm]
        n1 = int(300 * E(t, 1.5 + 2 * i, 3.5 + 2 * i))
        if n1 > 1:
            a1.plot(zs[:n1], f(zs[:n1]), color=col, lw=2.8)
        n2 = int(300 * E(t, 8.0 + 2 * i, 10.0 + 2 * i))
        if n2 > 1:
            a2.plot(zs[:n2], d(zs[:n2]), color=col, lw=2.8)
    T(fig, 0.5, 0.865, "Same z in, three different curves out. The slopes decide how gradients travel.",
      size=13, color=MID, ha="center", a=E(t, 8.0, 9.0))

    # canonical operating points
    ac = E(t, 14.5, 15.5)
    if ac > 0.02:
        for k, (zc, cc) in enumerate(zip(Z_CAN, (INK, GREY))):
            a1.axvline(zc, color=cc, ls=":", lw=1.3, alpha=ac)
            a2.axvline(zc, color=cc, ls=":", lw=1.3, alpha=ac)
            a1.text(zc + (0.07 if k == 0 else -0.07), -1.35, (r"$z_1=+0.99$" if k == 0 else r"$z_2=-0.18$"),
                    color=cc, fontsize=11, ha="left" if k == 0 else "right", alpha=ac)
            for nm in NAMES:
                f, d, col = ACT[nm]
                if f(zc) < 3:
                    dot(a1, zc, float(f(zc)), col, ms=8 * ac)
                dot(a2, zc, float(d(zc)), col, ms=8 * ac)
    xs3 = (0.20, 0.51, 0.82)
    msgs = [r"Sigmoid:   slope $\leq 0.25$", r"ReLU:   slope is $0$ or $1$", r"Tanh:   slope $\leq 1$"]
    for i, nm in enumerate(NAMES):
        T(fig, xs3[i], 0.07, msgs[i], size=16, color=ACT[nm][2], ha="center", a=E(t, 17.5 + 1.2 * i, 18.5 + 1.2 * i),
          weight="bold")


# ================================================================== table helper (forward / backward)
def table(fig, t, rows, cols=(0.34, 0.47, 0.60), y0=0.72, dy=0.085, expl_y=0.13):
    for j, nm in enumerate(NAMES):
        T(fig, cols[j], 0.80, nm, size=16, color=ACT[nm][2], weight="bold", ha="right", a=E(t, 1.0, 1.8))
    rule(fig, 0.05, 0.62, 0.765, a=E(t, 1.0, 1.8))
    for i, r in enumerate(rows):
        a = E(t, r["t"], r["t"] + 0.8)
        y = y0 - dy * i
        T(fig, 0.05, y, r["label"], size=16, color=MID, a=a)
        for j, v in enumerate(r["vals"]):
            txt, colr, wt = v if isinstance(v, tuple) else (v, INK, "normal")
            T(fig, cols[j], y, txt, size=17, color=colr, a=a, ha="right", weight=wt)
        nxt = rows[i + 1]["t"] if i + 1 < len(rows) else None
        ae = E(t, r["t"] + 0.4, r["t"] + 1.2) * (1 - (E(t, nxt - 0.5, nxt) if nxt is not None else 0.0))
        T(fig, 0.05, expl_y, r["expl"], size=14.5, color=MID, a=ae)


def fmt(v, p=4, sign=False):
    v = float(v) + 0.0
    return ("%+." + str(p) + "f") % v if sign else ("%." + str(p) + "f") % v


def zero_flag(v, text):
    return (text, RED, "bold") if abs(v) < 1e-12 else (text, INK, "normal")


# ================================================================== SCENE 6: forward
def scene_forward(fig, t):
    header(fig, "6", "Step 1, forward: same weights, three different predictions", E(t, 0, 0.8))
    T(fig, 0.05, 0.865, r"$x=[0.5,\ 0.8]$,   target $y=1.0$,   $W_1,b_1,W_2,b_2$ identical in all three networks",
      size=14, color=MID, a=E(t, 0.8, 1.6))
    A = AUD
    rows = [
        dict(t=2.0, label=r"$z_1$", vals=[fmt(A[n]["z"][0], 3, True) for n in NAMES],
             expl=r"$z_1=0.5\cdot0.5+0.8\cdot0.8+0.1=+0.990$     (same weights, same input: identical for all three)"),
        dict(t=5.5, label=r"$z_2$", vals=[fmt(A[n]["z"][1], 3, True) for n in NAMES],
             expl=r"$z_2=(-0.6)\cdot0.5+0.4\cdot0.8-0.2=-0.180$"),
        dict(t=9.0, label=r"$h_1=\varphi(z_1)$", vals=[fmt(A[n]["h"][0]) for n in NAMES],
             expl=r"$h_1=\varphi(0.990)$:   the three functions disagree for the first time"),
        dict(t=12.5, label=r"$h_2=\varphi(z_2)$", vals=[zero_flag(A[n]["h"][1], fmt(A[n]["h"][1])) for n in NAMES],
             expl=r"$h_2=\varphi(-0.180)$:   ReLU clamps the negative input to exactly 0"),
        dict(t=16.0, label=r"$\hat{y}$", vals=[fmt(A[n]["yh"]) for n in NAMES],
             expl=r"$\hat{y}=0.7\,h_1-0.5\,h_2+0.3$"),
        dict(t=20.0, label=r"$L=(\hat{y}-1)^2$", vals=[fmt(A[n]["loss"], 6) for n in NAMES],
             expl=r"ReLU starts almost on target. Sigmoid starts furthest away."),
    ]
    table(fig, t, rows)

    if t >= 16.0:
        ax = mk_ax(fig, [0.71, 0.20, 0.24, 0.52], (-0.6, 2.6), (0, 1.18), "", "", grid=False,
                   title=r"prediction $\hat{y}$ vs target", xticks=[0, 1, 2], yticks=[0, 0.5, 1])
        ax.set_xticklabels(NAMES)
        for tl, nm in zip(ax.get_xticklabels(), NAMES):
            tl.set_color(ACT[nm][2])
            tl.set_fontsize(11)
        ax.axhline(1.0, color=INK, ls="--", lw=1.4)
        ax.text(2.55, 1.03, "target 1.0", color=INK, fontsize=11, ha="right")
        p = E(t, 16.5, 18.5)
        for i, nm in enumerate(NAMES):
            ax.bar(i, A[nm]["yh"] * p, width=0.55, color=ACT[nm][2], alpha=0.9)
            if p > 0.95:
                ax.text(i, A[nm]["yh"] + 0.03, "%.3f" % A[nm]["yh"], ha="center", color=INK, fontsize=11)
    T(fig, 0.5, 0.05, "One input, one set of weights: the activation alone changes the answer.",
      size=15, color=INK, ha="center", a=E(t, 22.5, 24.0))


# ================================================================== SCENE 7: backward
def scene_backward(fig, t):
    header(fig, "7", "Step 1, backward: same loss, three different gradients", E(t, 0, 0.8))
    T(fig, 0.05, 0.865, r"Chain rule:   $\partial L/\partial z=\partial L/\partial\hat{y}\cdot w_2\cdot\varphi'(z)$",
      size=15, color=MID, a=E(t, 0.8, 1.6))
    A = AUD

    def row(i, key, k=None, p=4, sign=True):
        out = []
        for n in NAMES:
            v = A[n][key] if k is None else A[n][key][k]
            out.append(zero_flag(v, fmt(v, p, sign)))
        return out

    rows = [
        dict(t=2.0, label=r"$\partial L/\partial\hat{y}$", vals=row(0, "dL"),
             expl=r"$\partial L/\partial\hat{y}=2(\hat{y}-1)$:   negative everywhere, so every net must push $\hat{y}$ up"),
        dict(t=5.0, label=r"$\partial L/\partial h_1$", vals=row(1, "dh", 0),
             expl=r"$\partial L/\partial h_1=\partial L/\partial\hat{y}\cdot0.7$"),
        dict(t=8.0, label=r"$\partial L/\partial h_2$", vals=row(2, "dh", 1),
             expl=r"$\partial L/\partial h_2=\partial L/\partial\hat{y}\cdot(-0.5)$"),
        dict(t=11.0, label=r"$\varphi'(z_1)$", vals=row(3, "sl", 0, sign=False),
             expl=r"$\varphi'(z_1)$:   the slope from the activation chapters, read off at $z_1=0.990$"),
        dict(t=14.0, label=r"$\varphi'(z_2)$", vals=row(4, "sl", 1, sign=False),
             expl=r"$\varphi'(z_2)$:   read off at $z_2=-0.180$.   ReLU is flat there: slope exactly 0"),
        dict(t=17.5, label=r"$\partial L/\partial z_1$", vals=row(5, "dz", 0),
             expl=r"$\partial L/\partial z_1=\partial L/\partial h_1\cdot\varphi'(z_1)$:   the chain rule at work"),
        dict(t=21.0, label=r"$\partial L/\partial z_2$", vals=row(6, "dz", 1),
             expl=r"$\partial L/\partial z_2$:   ReLU gives exactly 0, so row 2 of $W_1$ receives no update"),
    ]
    table(fig, t, rows, dy=0.078, y0=0.72)

    if t >= 17.5:
        mx = max(abs(A[n]["dz"][k]) for n in NAMES for k in (0, 1))
        ax = mk_ax(fig, [0.71, 0.22, 0.24, 0.50], (-0.8, 6.8), (0, mx * 1.3), "", "", grid=False,
                   title=r"$|\partial L/\partial z|$ reaching each neuron", xticks=[1, 5], yticks=None)
        ax.set_xticklabels(["neuron 1", "neuron 2"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        p = E(t, 18.0, 20.0)
        for g in (0, 1):
            if g == 1 and t < 21.0:
                continue
            pg = p if g == 0 else E(t, 21.5, 23.0)
            for i, nm in enumerate(NAMES):
                v = abs(A[nm]["dz"][g])
                ax.bar(g * 4 + i, v * pg, width=0.8, color=ACT[nm][2], alpha=0.9)
                if pg > 0.95:
                    if v < 1e-12:
                        ax.text(g * 4 + i, mx * 0.04, "0", ha="center", color=RED, fontsize=13, fontweight="bold")
                    else:
                        ax.text(g * 4 + i, v + mx * 0.03, "%.3f" % v, ha="center", color=INK, fontsize=9)
        if t >= 23.5:
            ax.text(4 + 1, mx * 0.30, "dead", ha="center", color=RED, fontsize=12, alpha=E(t, 23.5, 24.5))
    T(fig, 0.5, 0.05, "Same weights, same data, same loss: the activation alone decides how much gradient each neuron gets.",
      size=14, color=INK, ha="center", a=E(t, 25.5, 27.0))


# ================================================================== SCENE 8: training
def scene_train(fig, t):
    header(fig, "8", "Training: 30 Adam steps on the single point", E(t, 0, 0.8))
    k = int(np.clip(int((t - 2.0) / 0.8) + 1, 0, N_STEPS)) if t >= 2.0 else 0
    a_hd = E(t, 1.0, 2.0)
    T(fig, 0.07, 0.862, "step  %d / %d" % (k, N_STEPS), size=15, color=INK, a=a_hd, weight="bold")
    if k > 0:
        for nm, x in zip(NAMES, (0.24, 0.50, 0.76)):
            T(fig, x, 0.862, "%s   L = %.2e" % (nm, HIST[nm]["loss"][k - 1]), size=14, color=ACT[nm][2], a=a_hd)

    allh = np.concatenate([HIST[n]["yh"] for n in NAMES])
    yl = (min(allh.min(), 0.55) - 0.03, max(allh.max(), 1.0) + 0.05)
    h1all = np.concatenate([HIST[n]["h1"] for n in NAMES])
    h2all = np.concatenate([HIST[n]["h2"] for n in NAMES])

    def pad(a, lo=None):
        r = a.max() - a.min()
        return (a.min() - 0.12 * r, a.max() + 0.12 * r)

    panels = [
        ([0.07, 0.50, 0.40, 0.30], "loss  (log scale)", "loss", "loss", (1e-8, 0.5)),
        ([0.57, 0.50, 0.39, 0.30], r"prediction $\hat{y}\ \to\ 1.0$", r"$\hat{y}$", "yh", yl),
        ([0.07, 0.12, 0.40, 0.27], r"hidden neuron $h_1$  (z=+0.99 at start)", r"$h_1$", "h1", pad(h1all)),
        ([0.57, 0.12, 0.39, 0.27], r"hidden neuron $h_2$  (z=-0.18 at start)", r"$h_2$", "h2", pad(h2all)),
    ]
    for rect, title, ylab, key, lim in panels:
        ax = mk_ax(fig, rect, (0, N_STEPS - 1), lim, "step" if rect[1] < 0.3 else "", ylab, title=title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if key == "loss":
            ax.set_yscale("log")
            ax.minorticks_off()
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if key == "yh":
            ax.axhline(1.0, color=INK, ls="--", lw=1.3)
        if key == "h2" and HIST["ReLU"]["h2"].max() == 0:
            pass
        if k > 0:
            for nm in NAMES:
                data = HIST[nm][key]
                ax.plot(np.arange(k), data[:k], color=ACT[nm][2], lw=2.2, marker="o", ms=3, mec=ACT[nm][2])
                dot(ax, k - 1, data[k - 1], ACT[nm][2], ms=8)
        if key == "h2" and t >= 27.5:
            ax.text(N_STEPS - 1.5, HIST["ReLU"]["h2"][-1] + 0.03 * (lim[1] - lim[0]),
                    "ReLU h2: stuck at 0, never learns", color=GREEN, fontsize=11, ha="right",
                    alpha=E(t, 27.5, 28.5))

    final = sorted(NAMES, key=lambda n: HIST[n]["loss"][-1])
    msg = "Final loss:   " + "   <   ".join("%s %.1e" % (n, HIST[n]["loss"][-1]) for n in final)
    T(fig, 0.5, 0.035, msg, size=14, color=INK, ha="center", a=E(t, 28.5, 30.0))


# ================================================================== SCENE 9: outro
def scene_outro(fig, t):
    T(fig, 0.06, 0.88, "Recap", size=34, a=E(t, 0, 0.8))
    rule(fig, 0.06, 0.94, 0.82, a=E(t, 0, 0.8))
    rows = [
        ("Sigmoid", AMBER, r"squashes to $(0,1)$,   slope $\leq0.25$,   flat at both ends: gradients shrink layer by layer"),
        ("ReLU", GREEN, r"passes $z>0$ at slope 1,   cuts $z\leq0$ to 0:   cheap and sharp, but a neuron can go dead"),
        ("Tanh", BLUE, r"squashes to $(-1,1)$,   slope $\leq1$,   centered:   the steadiest of the three on our point"),
    ]
    for i, (name, c, eq) in enumerate(rows):
        a = E(t, 1 + 1.3 * i, 1.9 + 1.3 * i)
        y = 0.68 - 0.15 * i
        T(fig, 0.06, y, name, size=21, color=c, a=a, weight="bold")
        T(fig, 0.06, y - 0.055, eq, size=16, color=INK, a=a)
        rule(fig, 0.06, 0.94, y - 0.10, a=a * 0.8, color=GRID)
    T(fig, 0.06, 0.20, r"In the backward pass every layer multiplies the gradient by $\varphi'(z)$.",
      size=17, color=RED, a=E(t, 5.2, 6.4))
    T(fig, 0.06, 0.13, "Choose the activation and you choose how much learning signal survives the trip back.",
      size=16, color=MID, a=E(t, 6.4, 7.6))


SCENES = [("title", 5.0, scene_title), ("why", 20.0, scene_why),
          ("sigmoid", 33.0, scene_sigmoid), ("relu", 33.0, scene_relu), ("tanh", 33.0, scene_tanh),
          ("compare", 24.0, scene_compare), ("forward", 26.0, scene_forward),
          ("backward", 29.0, scene_backward), ("train", 32.0, scene_train), ("outro", 9.0, scene_outro)]
PREVIEW_TIMES = {"title": [3.8], "why": [9.5, 17.0],
                 "sigmoid": [4.5, 11.0, 19.5, 27.5, 31.0],
                 "relu": [4.5, 11.0, 19.5, 27.5, 31.0],
                 "tanh": [4.5, 11.0, 19.5, 27.5, 31.0],
                 "compare": [13.0, 22.0], "forward": [8.0, 22.0], "backward": [12.0, 27.0],
                 "train": [12.0, 31.0], "outro": [8.0]}


def draw_frame(fig, scene, dur, t):
    fig.clf()
    fig.patch.set_facecolor(BG)
    scene(fig, t)
    a = max(1 - E(t, 0, 0.5), E(t, dur - 0.5, dur))
    if a > 0.01:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor="none",
                                 alpha=a, zorder=1000))


def main():
    ap = argparse.ArgumentParser(description="Minimalist activation-function film (NumPy + Matplotlib)")
    ap.add_argument("--scene", default="all", choices=["all"] + [s[0] for s in SCENES])
    ap.add_argument("--preview", action="store_true", help="save PNG stills instead of a video")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=80, help="80 -> 1280x720, 120 -> 1920x1080")
    ap.add_argument("--format", default="auto", choices=["auto", "mp4", "gif"])
    ap.add_argument("--out", default="exports")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    todo = [s for s in SCENES if args.scene in ("all", s[0])]
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=BG)

    if args.preview:
        for name, dur, fn in todo:
            for t in PREVIEW_TIMES[name]:
                draw_frame(fig, fn, dur, t)
                path = os.path.join(args.out, f"preview_{name}_{t:05.1f}.png")
                fig.savefig(path, facecolor=BG)
                print("saved", path)
        return

    fmt_ = args.format
    if fmt_ == "auto":
        fmt_ = "mp4" if animation.writers.is_available("ffmpeg") else "gif"
    tag = args.scene if args.scene != "all" else "full"
    path = os.path.join(args.out, f"module5_activation_{tag}.{fmt_}")
    writer = (animation.FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
              if fmt_ == "mp4" else animation.PillowWriter(fps=min(args.fps, 15)))
    total = sum(int(d * args.fps) for _, d, _ in todo)
    done = 0
    with writer.saving(fig, path, args.dpi):
        for name, dur, fn in todo:
            for i in range(int(dur * args.fps)):
                draw_frame(fig, fn, dur, i / args.fps)
                writer.grab_frame(facecolor=BG)
                done += 1
                if done % 40 == 0:
                    print(f"\r[{name:<10}] {done}/{total} frames", end="", flush=True)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()