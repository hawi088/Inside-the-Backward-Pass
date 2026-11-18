#!/usr/bin/env python3
"""
Module 8 - "Inside the Backward Pass"  |  minimalist loss-landscape film
Pure NumPy + Matplotlib (no autograd / DL frameworks, no Manim, no engine package).

Same design language as the Module 2 and Module 5 films:
  * One paper-white canvas, ink-black text, hairline axes, Computer Modern math.
  * One idea at a time, written on the right in order.
  * Colour carries one meaning in every chapter:
        INK   = the loss surface / the loss itself
        RED   = gradient, slope, step (the thing we are after)
        GREEN = the bottom of the landscape ("good")
        BLUE  = SGD          AMBER = Adam

Everything on screen is computed, nothing is drawn by hand. The network is the capstone's
canonical one:  x = [0.5, 0.8], y = 1.0, 2 -> 2 -> 1, sigmoid hidden layer, MSE.
    W1 = [[0.5, 0.8], [-0.6, 0.4]]  (in x out: z = x @ W1 + b1, column j feeds hidden neuron j)
    z1 = -0.130, z2 = +0.520   =>   h1 = 0.4675, h2 = 0.6271
    W2 = [0.7, -0.5], b2 = 0.3, and W1[0,0] = 0.5 (x1 -> h1), W1[1,0] = -0.6 (x2 -> h1)

Two 2-D slices of the full loss landscape are explored, everything else held at its canonical value:
  Slice A  output weights (w1, w2) = W2[0], W2[1]     L = (w1*h1 + w2*h2 + b2 - 1)^2
  Slice B  hidden weights (a, b)   = W1[0,0], W1[1,0] L = (0.7*sigmoid(0.5a + 0.8b + 0.1) + c)^2

Four chapters:
  1. One weight       a parabola, its slope, the first gradient steps
  2. Two output w.    a surface, then a flat view: the bottom is a LINE, not a point
  3. Two hidden w.    everything depends on one number z1: a ramp between two plateaus
  4. Optimizers       SGD vs Adam walking both landscapes (descent restricted to the two swept weights)

Usage
  python module8_loss_landscape_film.py                     # full film (mp4 if ffmpeg exists, else gif)
  python module8_loss_landscape_film.py --scene valley      # one chapter only
  python module8_loss_landscape_film.py --preview           # PNG stills of every chapter
  python module8_loss_landscape_film.py --fps 24 --dpi 100  # higher quality
Chapters: title, curve, valley, ramp, walk, outro
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
BLUE = "#2D5B9A"      # SGD
GREEN = "#3C7A5A"     # the bottom
AMBER = "#B7832F"     # Adam
RED = "#C4452F"       # gradient / step


# ------------------------------------------------------------------ the canonical network, analytically
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


X_IN = np.array([0.5, 0.8])
W1_CAN = np.array([[0.5, 0.8], [-0.6, 0.4]])                                      # in x out: z = x @ W1 + b1
B1_CAN = np.array([0.1, -0.2])
Z_CAN = X_IN @ W1_CAN + B1_CAN                                                    # [-0.130, +0.520]
H1, H2 = sigmoid(Z_CAN)                                                           # 0.4675, 0.6271
W2_0 = np.array([0.7, -0.5])
B2 = 0.3
A0, B0 = W1_CAN[0, 0], W1_CAN[1, 0]                                               # W1[0,0] = 0.5, W1[1,0] = -0.6
K_B = W2_0[1] * H2 + B2 - 1.0                                                     # slice B constant, -1.0136
L_FLOOR_B = (W2_0[0] + K_B) ** 2                                                  # 0.0983: z1 -> +inf
L_TOP_B = K_B ** 2                                                                # 1.0273: z1 -> -inf


def L_A(w1, w2):
    return (w1 * H1 + w2 * H2 + B2 - 1.0) ** 2


def grad_A(p):
    e = p[0] * H1 + p[1] * H2 + B2 - 1.0
    return 2.0 * e * np.array([H1, H2])


def z1_of(a, b):
    return 0.5 * a + 0.8 * b + 0.1


def L_B(a, b):
    return (W2_0[0] * sigmoid(z1_of(a, b)) + K_B) ** 2


def grad_B(p):
    z = z1_of(*p)
    s = sigmoid(z)
    e = W2_0[0] * s + K_B
    return 2.0 * e * W2_0[0] * s * (1 - s) * np.array([0.5, 0.8])


def descend(gradf, p0, opt, lr, steps=50, b1=0.9, b2=0.999, eps=1e-8):
    """SGD or Adam on the two swept weights only (all other weights frozen at canonical values)."""
    p = np.array(p0, float)
    path = [p.copy()]
    m = np.zeros(2)
    v = np.zeros(2)
    for s in range(1, steps + 1):
        g = gradf(p)
        if opt == "sgd":
            p = p - lr * g
        else:
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            p = p - lr * (m / (1 - b1 ** s)) / (np.sqrt(v / (1 - b2 ** s)) + eps)
        path.append(p.copy())
    return np.array(path)


N_WALK = 50
PA_SGD = descend(grad_A, W2_0, "sgd", 0.3, N_WALK)
PA_ADAM = descend(grad_A, W2_0, "adam", 0.1, N_WALK)
PB_SGD = descend(grad_B, [A0, B0], "sgd", 0.3, N_WALK)
PB_ADAM = descend(grad_B, [A0, B0], "adam", 0.1, N_WALK)
LA_SGD, LA_ADAM = L_A(PA_SGD[:, 0], PA_SGD[:, 1]), L_A(PA_ADAM[:, 0], PA_ADAM[:, 1])
LB_SGD, LB_ADAM = L_B(PB_SGD[:, 0], PB_SGD[:, 1]), L_B(PB_ADAM[:, 0], PB_ADAM[:, 1])

# 1-D slice: only W2[0] free
K_1D = W2_0[1] * H2 + B2 - 1.0                                                    # -1.0136
W1D = [0.7]
for _ in range(10):
    w = W1D[-1]
    W1D.append(w - 0.3 * 2.0 * (H1 * w + K_1D) * H1)
W1D = np.array(W1D)


def L_1D(w):
    return (H1 * w + K_1D) ** 2


# grids
XA, XB_ = (-2.0, 3.5), (-3.0, 2.0)
GA1, GA2 = np.meshgrid(np.linspace(*XA, 220), np.linspace(*XB_, 200))
GAZ = L_A(GA1, GA2)
SA1, SA2 = np.meshgrid(np.linspace(-1.0, 3.0, 41), np.linspace(-2.5, 1.5, 41))     # 3-D patch of slice A
SAZ = L_A(SA1, SA2)
RB = (-3.0, 4.5)
GBa, GBb = np.meshgrid(np.linspace(*RB, 220), np.linspace(*RB, 220))
GBZ = L_B(GBa, GBb)
SBa, SBb = np.meshgrid(np.linspace(*RB, 41), np.linspace(*RB, 41))
SBZ = L_B(SBa, SBb)


# ------------------------------------------------------------------ tiny animation toolkit (same as Modules 2 and 5)
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


def arrow(ax, p0, p1, color, prog=1.0, lw=2.6, alpha=1.0, ls="-"):
    if prog <= 0.02:
        return
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.annotate("", xy=p0 + (p1 - p0) * prog, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha, ls=ls,
                                mutation_scale=18, shrinkA=0, shrinkB=0))


def dot(ax, x, y, color=INK, ms=9, z=8):
    ax.plot([x], [y], "o", color=color, ms=ms, mec=BG, mew=1.6, zorder=z, clip_on=False)


def style3d(ax):
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.label.set_color(MID)
        try:
            axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
            axis._axinfo["axisline"].update(color=GREY)
        except Exception:
            pass
    ax.tick_params(colors=GREY, labelsize=8)
    ax.set_box_aspect((1, 1, 0.7))
    try:
        ax.computed_zorder = False
    except Exception:
        pass


def veil(fig, t, tc, rect=(0.0, 0.06, 0.60, 0.82)):
    """Short paper-white dip over the left panel, hiding a cut between two views."""
    a = E(t, tc - 0.6, tc) if t < tc else 1 - E(t, tc, tc + 0.6)
    if a > 0.01:
        fig.add_artist(Rectangle(rect[:2], rect[2], rect[3], transform=fig.transFigure, facecolor=BG,
                                 edgecolor="none", alpha=a, zorder=900))


def surface3d(fig, rect, X, Y, Z, xlim, ylim, zlim, azim, alpha, xl, yl, zl):
    ax = fig.add_axes(rect, projection="3d")
    style3d(ax)
    ax.view_init(elev=28, azim=azim)
    if alpha > 0.02:
        ax.plot_surface(X, Y, Z, color="#E2DFD6", alpha=0.62 * alpha, linewidth=0.3, edgecolor="#B9B5AA",
                        rstride=2, cstride=2, antialiased=True, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel(xl, labelpad=-4)
    ax.set_ylabel(yl, labelpad=-4)
    ax.set_zlabel(zl, labelpad=-6)
    return ax


# ================================================================== SCENE 0: title
def scene_title(fig, t):
    T(fig, 0.5, 0.74, "M O D U L E   8", size=12, color=GREY, ha="center", a=E(t, 0.0, 0.8))
    T(fig, 0.5, 0.64, "The Loss Landscape", size=44, ha="center", a=E(t, 0.2, 1.2))
    T(fig, 0.5, 0.55, "Every weight setting has a loss   ·   training is a walk downhill",
      size=15, color=MID, ha="center", a=E(t, 1.0, 2.0))
    ax = fig.add_axes([0.36, 0.08, 0.28, 0.36])
    ax.axis("off")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.4, 4.6)
    xs = np.linspace(-2, 2, 200)
    n = max(2, int(200 * E(t, 1.0, 2.2)))
    ax.plot(xs[:n], xs[:n] ** 2, color=INK, lw=2.2)
    pa = E(t, 1.8, 2.4)
    if pa > 0.02:
        x = -1.8 * np.exp(-max(t - 2.4, 0.0) * 1.2)
        dot(ax, x, x ** 2, RED, ms=10 * pa)


# ================================================================== SCENE 1: one weight
def scene_curve(fig, t):
    header(fig, "1", "One weight: the loss is a curve", E(t, 0, 0.8))
    T(fig, 0.05, 0.865, r"Freeze everything except one output weight $w=W_2[0]$.  The rest keeps its canonical value.",
      size=15, color=MID, a=E(t, 0.8, 1.6))

    ax = mk_ax(fig, [0.06, 0.10, 0.50, 0.70], (-1, 3), (-0.25, 3.0), r"$w$", r"$L$", center=True,
               xticks=[-1, 0, 1, 2, 3], yticks=[1, 2, 3])
    ws = np.linspace(-1, 3, 300)
    n = max(2, int(300 * E(t, 2.5, 5.0)))
    ax.plot(ws[:n], L_1D(ws[:n]), color=INK, lw=2.6)

    w0 = 0.7
    pa = E(t, 5.0, 5.8)
    k = 0
    frac = 0.0
    if t >= 19.5:
        q = (t - 19.5) / 0.8
        k = int(min(np.floor(q), 10))
        frac = 0.0 if k >= 10 else q - k
    if k >= 10:
        wcur = W1D[10]
    else:
        wcur = W1D[k] + (W1D[k + 1] - W1D[k]) * ease(frac)
    if pa > 0.02:
        pos = w0 if t < 19.5 else wcur
        dot(ax, pos, float(L_1D(pos)), INK, ms=9 * pa, z=8)
    if t < 19.5 and pa > 0.02:
        ax.text(0.55, float(L_1D(w0)) + 0.35, r"$P$", color=INK, fontsize=14, ha="right", alpha=pa)

    wstar = -K_1D / H1
    pm = E(t, 8.5, 9.5)
    if pm > 0.02:
        ax.plot([wstar], [0], "o", ms=10 * pm, mfc=BG, mec=GREEN, mew=2.0, zorder=7, clip_on=False)
        ax.text(wstar, 0.32, "bottom", color=GREEN, fontsize=12, ha="center", alpha=pm)

    # trail of steps
    if t >= 19.5:
        pts = W1D[:min(k, 10) + 1]
        ax.plot(pts, L_1D(pts), "-o", color=BLUE, lw=1.3, ms=3.5, mec=BLUE, zorder=6)

    # tangent
    ta = E(t, 12.0, 13.0)
    if ta > 0.02:
        wt = w0 if t < 19.5 else wcur
        m = 2.0 * (H1 * wt + K_1D) * H1
        ends = np.array([wt - 0.9, wt + 0.9])
        ax.plot(ends, float(L_1D(wt)) + m * (ends - wt), color=RED, lw=2.4, alpha=ta)

    X = 0.62
    T(fig, X, 0.80, r"prediction, with $w$ free", size=14, color=MID, a=E(t, 2.6, 3.4))
    T(fig, X, 0.73, r"$\hat{y}(w)=%.4f\,w%+.4f$" % (H1, W2_0[1] * H2 + B2), size=19, a=E(t, 3.2, 4.2))
    T(fig, X, 0.64, r"$L(w)=(\hat{y}-1)^2=(%.4f\,w-%.4f)^2$" % (H1, -K_1D), size=17, a=E(t, 5.5, 6.5))
    T(fig, X, 0.55, "A parabola: one bottom, at  w = %.3f" % wstar, size=15, color=GREEN, a=E(t, 8.5, 9.5))

    e0 = H1 * w0 + K_1D
    m0 = 2.0 * e0 * H1
    T(fig, X, 0.45, r"slope at $P$ ($w=0.7$,  $L=%.3f$)" % float(L_1D(w0)), size=14, color=MID, a=E(t, 11.0, 12.0))
    T(fig, X, 0.38, r"$\dfrac{dL}{dw}=2(\hat{y}-1)\,h_1=2(%.4f)(%.4f)=%.3f$" % (e0, H1, m0), size=15, color=RED,
      a=E(t, 12.5, 13.5))
    T(fig, X, 0.30, "negative slope: the bottom is to the right.", size=14, color=MID, a=E(t, 15.0, 16.0))

    T(fig, X, 0.22, r"$w\ \leftarrow\ w-\alpha\,\dfrac{dL}{dw}$,    $\alpha=0.3$", size=17, color=INK, a=E(t, 17.0, 18.0))
    if t >= 19.5:
        kk = min(k, 10)
        T(fig, X, 0.14, r"step $%d$:   $w=%.3f$,   $L=%.4f$" % (kk, W1D[kk], float(L_1D(W1D[kk]))), size=16,
          color=BLUE, a=E(t, 19.5, 20.3))
    T(fig, 0.5, 0.04, "Near the bottom the slope shrinks, so the steps shrink too.   That is gradient descent on one weight.",
      size=14, color=INK, ha="center", a=E(t, 27.0, 28.5))


# ================================================================== SCENE 2: two output weights
TC2 = 17.5


def scene_valley(fig, t):
    header(fig, "2", "Two output weights: the loss is a surface", E(t, 0, 0.8))
    X = 0.62
    p0 = W2_0
    L0 = float(L_A(*p0))
    ln_w1 = np.linspace(-2.0, 3.5, 100)
    ln_w2 = (0.7 - ln_w1 * H1) / H2                      # valley floor: w1*h1 + w2*h2 = 0.7

    if t < TC2:
        ax = surface3d(fig, [0.0, 0.08, 0.58, 0.76], SA1, SA2, SAZ, (-1, 3), (-2.5, 1.5), (0, 8),
                       -62 + 22 * seg(t, 0, TC2), E(t, 0.4, 2.0), r"$w_1$", r"$w_2$", r"$L$")
        if t > 9.0:
            a9 = E(t, 9.0, 10.0)
            ax.scatter([p0[0]], [p0[1]], [L0 + 0.1], color=INK, s=55 * a9, depthshade=False, zorder=10)
        if t > 12.5:
            n = int(100 * E(t, 12.5, 15.0))
            msk = (ln_w2 >= -2.5) & (ln_w2 <= 1.5) & (ln_w1 >= -1) & (ln_w1 <= 3)
            xs_, ys_ = ln_w1[msk][:max(2, int(msk.sum() * E(t, 12.5, 15.0)))], ln_w2[msk][:max(2, int(msk.sum() * E(t, 12.5, 15.0)))]
            if n > 1:
                ax.plot(xs_, ys_, np.zeros_like(xs_) + 0.02, color=GREEN, lw=3, zorder=6)
        aout = 1 - E(t, 16.6, 17.4)
        T(fig, X, 0.80, "Now free two output weights", size=15, color=MID, a=E(t, 1.0, 2.0) * aout)
        T(fig, X, 0.71, r"$\hat{y}=%.4f\,w_1+%.4f\,w_2+0.3$" % (H1, H2), size=19, a=E(t, 2.5, 3.5) * aout)
        T(fig, X, 0.62, r"$L=(\hat{y}-1)^2$", size=21, a=E(t, 4.5, 5.5) * aout)
        T(fig, X, 0.52, "Each pair  (w1, w2)  is a point on the floor.", size=14, color=MID, a=E(t, 6.0, 7.0) * aout)
        T(fig, X, 0.46, "Its loss is the height: a surface.", size=14, color=MID, a=E(t, 7.0, 8.0) * aout)
        T(fig, X, 0.37, "Our start  (0.7, -0.5)  has  L = %.3f" % L0, size=14, color=INK, a=E(t, 9.0, 10.0) * aout)
        T(fig, X, 0.28, "Look at the shape: not a round bowl, a long trough.", size=15, color=GREEN, a=E(t, 11.5, 12.5) * aout)
    veil(fig, t, TC2)

    if t >= TC2:
        ax = mk_ax(fig, [0.05, 0.10, 0.52, 0.76], XA, XB_, r"$w_1$", r"$w_2$", grid=False)
        ax.set_aspect("equal", adjustable="box")
        ca = E(t, TC2, TC2 + 1.0)
        if ca > 0.02:
            ax.contour(GA1, GA2, GAZ, levels=[0.05, 0.2, 0.5, 1, 2, 3, 5, 8, 12], colors=LIGHT, linewidths=1.0, alpha=ca)
        dot(ax, p0[0], p0[1], INK, ms=9, z=8)
        ax.text(p0[0] - 0.12, p0[1] - 0.3, r"$P$", color=INK, fontsize=13, ha="right")
        # valley line
        vp = E(t, 22.0, 24.5)
        if vp > 0.02:
            msk = (ln_w2 >= XB_[0]) & (ln_w2 <= XB_[1])
            xs_, ys_ = ln_w1[msk], ln_w2[msk]
            nn = max(2, int(len(xs_) * vp))
            ax.plot(xs_[:nn], ys_[:nn], color=GREEN, lw=3.0, zorder=5)
        # gradient and step
        g0 = grad_A(p0)
        a_g = E(t, 28.0, 29.2)
        s_ = 0.9
        if t < 32.5:
            arrow(ax, p0, p0 + s_ * g0, GREY, a_g, lw=2.6)
            arrow(ax, p0, p0 - s_ * g0, RED, E(t, 29.5, 30.8), lw=3.2)
        # SGD walk
        if t >= 33.0:
            p = 50.0 * seg(t, 33.0, 41.0)
            i = int(min(np.floor(p), 49))
            fr = p - i
            pos = PA_SGD[i] + (PA_SGD[i + 1] - PA_SGD[i]) * fr
            trail = np.vstack([PA_SGD[:i + 1], pos[None, :]])
            ax.plot(trail[:, 0], trail[:, 1], "-o", color=BLUE, lw=1.6, ms=3.5, mec=BLUE, zorder=6)
            dot(ax, pos[0], pos[1], BLUE, ms=10, z=9)

        T(fig, X, 0.84, "Seen from straight above: contour lines", size=14, color=MID, a=E(t, 18.0, 19.0))
        T(fig, X, 0.77, "Every contour is a straight line. Why?", size=16, a=E(t, 19.5, 20.5))
        T(fig, X, 0.69, "L depends on one number only:", size=14, color=MID, a=E(t, 20.5, 21.5))
        T(fig, X, 0.62, r"$s=%.4f\,w_1+%.4f\,w_2$" % (H1, H2), size=19, a=E(t, 21.0, 22.0))
        T(fig, X, 0.53, r"bottom:   $L=0\ \Leftrightarrow\ s=0.7$", size=18, color=GREEN, a=E(t, 23.0, 24.0))
        T(fig, X, 0.46, "The minimum is a whole line, not a point.", size=15, color=GREEN, a=E(t, 24.5, 25.5))
        T(fig, X, 0.37, r"$\nabla L=2(\hat{y}-1)\,[h_1,h_2]=[%.3f,\ %.3f]$" % (g0[0], g0[1]), size=14, color=MID,
          a=E(t, 27.5, 28.5))
        T(fig, X, 0.31, r"grey: $\nabla L$ (uphill),    red: $-\nabla L$ (the step)", size=13, color=RED, a=E(t, 30.0, 31.0))
        T(fig, X, 0.25, "perpendicular to the contours, as always.", size=13, color=MID, a=E(t, 31.0, 32.0))
        if t >= 33.0:
            kk = min(int(50 * seg(t, 33.0, 41.0)), 50)
            T(fig, X, 0.17, r"SGD, step $%d$:   $L=%.2e$" % (kk, float(LA_SGD[kk])), size=15, color=BLUE, a=E(t, 33.0, 34.0))
        fin = PA_SGD[-1]
        T(fig, X, 0.10, r"It lands at $(%.2f,\ %.2f)$: the nearest point of the valley." % (fin[0], fin[1]),
          size=13.5, color=INK, a=E(t, 41.5, 42.8))
        T(fig, X, 0.05, "Another start would land somewhere else on the line.", size=13, color=MID, a=E(t, 42.5, 43.6))


# ================================================================== SCENE 3: two hidden weights
TC3 = 18.0


def scene_ramp(fig, t):
    header(fig, "3", "Two hidden weights: the loss behind a sigmoid", E(t, 0, 0.8))
    X = 0.62
    p0 = np.array([A0, B0])
    L0 = float(L_B(*p0))
    if t < TC3:
        ax = surface3d(fig, [0.0, 0.08, 0.58, 0.76], SBa, SBb, SBZ, RB, RB, (0, 1.1),
                       -50 + 20 * seg(t, 0, TC3), E(t, 0.4, 2.0), r"$a$", r"$b$", r"$L$")
        if t > 11.0:
            a9 = E(t, 11.0, 12.0)
            ax.scatter([p0[0]], [p0[1]], [L0 + 0.02], color=INK, s=55 * a9, depthshade=False, zorder=10)
        aout = 1 - E(t, 17.0, 17.8)
        T(fig, X, 0.80, "Now free the two weights feeding  h1", size=15, color=MID, a=E(t, 1.0, 2.0) * aout)
        T(fig, X, 0.72, r"$a=W_1[0,0]\ (x_1\to h_1),\qquad b=W_1[1,0]\ (x_2\to h_1)$", size=13.5, a=E(t, 2.5, 3.5) * aout)
        T(fig, X, 0.63, r"$z_1=0.5\,a+0.8\,b+0.1$", size=20, a=E(t, 4.5, 5.5) * aout)
        T(fig, X, 0.54, r"$\hat{y}=0.7\,\sigma(z_1)%+.4f$" % (W2_0[1] * H2 + B2), size=19, a=E(t, 7.0, 8.0) * aout)
        T(fig, X, 0.45, r"Again one number, $z_1$, carries everything.", size=14, color=MID, a=E(t, 9.0, 10.0) * aout)
        T(fig, X, 0.35, "The surface is a ramp between two flat plateaus.", size=15, color=RED, a=E(t, 13.0, 14.0) * aout)
    veil(fig, t, TC3)

    if t >= TC3:
        ax = mk_ax(fig, [0.05, 0.10, 0.52, 0.76], RB, RB, r"$a$", r"$b$", grid=False)
        ax.set_aspect("equal", adjustable="box")
        ca = E(t, TC3, TC3 + 1.0)
        if ca > 0.02:
            ax.contour(GBa, GBb, GBZ, levels=[0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], colors=LIGHT,
                       linewidths=1.0, alpha=ca)
        dot(ax, p0[0], p0[1], INK, ms=9, z=8)
        ax.text(p0[0] - 0.15, p0[1] + 0.25, r"$P$", color=INK, fontsize=13, ha="right")
        u = np.array([0.5, 0.8]) / np.hypot(0.5, 0.8)
        arrow(ax, p0, p0 + 2.3 * u, RED, E(t, 22.0, 23.2), lw=3.2)
        if t > 23:
            ax.text(p0[0] + 2.3 * u[0] + 0.1, p0[1] + 2.3 * u[1] - 0.1, r"$-\nabla L$ (downhill)", color=RED, fontsize=12,
                    alpha=E(t, 23.0, 24.0))

        T(fig, X, 0.84, "Contour lines: straight and parallel", size=15, a=E(t, 18.8, 19.8))
        T(fig, X, 0.77, r"each one is a set of equal $z_1$.", size=14, color=MID, a=E(t, 20.2, 21.2))
        T(fig, X, 0.69, r"$\nabla L\ \parallel\ (0.5,\ 0.8)$  everywhere (up to sign)", size=17, color=RED, a=E(t, 23.5, 24.5))

        if t >= 26.0:
            am = E(t, 26.0, 27.0)
            zz = np.linspace(-6, 8, 300)
            ab = mk_ax(fig, [0.66, 0.10, 0.29, 0.27], (-6, 8), (0, 1.15), r"$z_1$", r"$L$",
                       title=r"cut along that direction:  $L(z_1)$", xticks=[-6, -3, 0, 3, 6], yticks=[0, 0.5, 1])
            ab.plot(zz, (0.7 * sigmoid(zz) + K_B) ** 2, color=INK, lw=2.4)
            ab.axhline(L_TOP_B, color=GREY, ls=":", lw=1.2)
            ab.axhline(L_FLOOR_B, color=GREEN, ls=":", lw=1.4)
            ab.text(-5.8, L_TOP_B + 0.04, "%.3f" % L_TOP_B, color=GREY, fontsize=11, ha="left")
            ab.text(7.8, L_FLOOR_B + 0.05, "%.4f" % L_FLOOR_B, color=GREEN, fontsize=11, ha="right")
            dot(ab, float(z1_of(*p0)), L0, INK, ms=8)
        T(fig, X, 0.58, r"Even $z_1\to\infty$ only reaches  $L=%.4f$,  never 0." % L_FLOOR_B, size=14.5, color=GREEN,
          a=E(t, 29.0, 30.0))
        T(fig, X, 0.51, r"Because  $\hat{y}\leq0.7%+.4f=%.4f<1$." % (W2_0[1] * H2 + B2, 0.7 + W2_0[1] * H2 + B2),
          size=14, color=MID, a=E(t, 31.0, 32.0))
        T(fig, X, 0.44, "Hidden weights alone cannot fix it: h2 is frozen here.", size=13, color=MID, a=E(t, 33.0, 34.0))
        T(fig, 0.5, 0.012, "No bottom on this slice: descent just slides toward the plateau, and the slope fades as it goes.",
          size=13.5, color=INK, ha="center", a=E(t, 38.0, 39.5))


# ================================================================== SCENE 4: optimizers
def draw_walk(ax, P, col, k, frac):
    kk = int(min(k, N_WALK))
    if kk >= N_WALK:
        pos = P[N_WALK]
    else:
        pos = P[kk] + (P[kk + 1] - P[kk]) * frac
    trail = np.vstack([P[:kk + 1], pos[None, :]])
    ax.plot(trail[:, 0], trail[:, 1], "-o", color=col, lw=1.6, ms=3, mec=col, zorder=6)
    dot(ax, pos[0], pos[1], col, ms=10, z=9)


def scene_walk(fig, t):
    header(fig, "4", "Walking the landscape: SGD vs Adam", E(t, 0, 0.8))
    T(fig, 0.05, 0.865, "SGD  (lr 0.3)", size=15, color=BLUE, weight="bold", a=E(t, 0.8, 1.6))
    T(fig, 0.19, 0.865, "Adam  (lr 0.1)", size=15, color=AMBER, weight="bold", a=E(t, 1.0, 1.8))
    T(fig, 0.34, 0.865, "same start, same landscape, only the update rule differs", size=13, color=MID, a=E(t, 1.4, 2.2))

    p = N_WALK * seg(t, 3.0, 28.0)
    k = int(min(np.floor(p), N_WALK))
    frac = 0.0 if k >= N_WALK else p - k
    kk = min(k, N_WALK)

    axa = mk_ax(fig, [0.06, 0.40, 0.40, 0.40], XA, XB_, r"$w_1$", r"$w_2$", grid=False,
                title="Slice A:  output weights")
    axa.set_aspect("equal", adjustable="box")
    axa.contour(GA1, GA2, GAZ, levels=[0.05, 0.2, 0.5, 1, 2, 3, 5, 8, 12], colors=LIGHT, linewidths=1.0)
    ln1 = np.linspace(-2.0, 3.5, 100)
    ln2 = (0.7 - ln1 * H1) / H2
    m = (ln2 >= XB_[0]) & (ln2 <= XB_[1])
    axa.plot(ln1[m], ln2[m], color=GREEN, lw=2.4, zorder=4)
    dot(axa, W2_0[0], W2_0[1], INK, ms=8, z=7)

    axb = mk_ax(fig, [0.56, 0.40, 0.38, 0.40], RB, RB, r"$a$", r"$b$", grid=False,
                title="Slice B:  hidden weights")
    axb.set_aspect("equal", adjustable="box")
    axb.contour(GBa, GBb, GBZ, levels=[0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], colors=LIGHT, linewidths=1.0)
    dot(axb, A0, B0, INK, ms=8, z=7)

    if t >= 3.0:
        draw_walk(axa, PA_SGD, BLUE, k, frac)
        draw_walk(axa, PA_ADAM, AMBER, k, frac)
        draw_walk(axb, PB_SGD, BLUE, k, frac)
        draw_walk(axb, PB_ADAM, AMBER, k, frac)

    # loss curves
    la = mk_ax(fig, [0.06, 0.17, 0.40, 0.12], (0, N_WALK), (1e-8, 0.5), "step", "loss")
    la.set_yscale("log")
    la.minorticks_off()
    la.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    lb = mk_ax(fig, [0.56, 0.17, 0.38, 0.12], (0, N_WALK), (0.08, 0.5), "step", "loss")
    lb.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    lb.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if t >= 3.0:
        for ax_, LS, LD in ((la, LA_SGD, LA_ADAM), (lb, LB_SGD, LB_ADAM)):
            ax_.plot(np.arange(kk + 1), np.maximum(LS[:kk + 1], 1e-9), color=BLUE, lw=2.0)
            ax_.plot(np.arange(kk + 1), np.maximum(LD[:kk + 1], 1e-9), color=AMBER, lw=2.0)
        T(fig, 0.88, 0.865, "step  %d / %d" % (kk, N_WALK), size=14, color=INK, ha="right", a=E(t, 3.0, 4.0), weight="bold")

    T(fig, 0.5, 0.055,
      "A:  SGD lands on the valley floor  (L = %.0e);   Adam keeps circling it  (L = %.0e)" % (LA_SGD[-1], LA_ADAM[-1]),
      size=14, color=INK, ha="center", a=E(t, 29.0, 30.0))
    T(fig, 0.5, 0.020,
      "B:  Adam strides across the plateau  (L = %.4f),   SGD crawls as the slope fades  (L = %.4f)" % (LB_ADAM[-1], LB_SGD[-1]),
      size=14, color=INK, ha="center", a=E(t, 31.0, 32.0))


# ================================================================== SCENE 5: outro
def scene_outro(fig, t):
    T(fig, 0.06, 0.88, "Recap", size=34, a=E(t, 0, 0.8))
    rule(fig, 0.06, 0.94, 0.82, a=E(t, 0, 0.8))
    rows = [
        ("Output weights", r"a trough, not a bowl:  $L=0$ along a whole line,  so the landing point depends on the start"),
        ("Hidden weights", r"a ramp between two plateaus:  one gradient direction, and a floor of $L=%.3f$ on this slice" % L_FLOOR_B),
        ("SGD and Adam", r"SGD follows $-\nabla L$ exactly.  Adam rescales each weight, so it strides on plateaus and leaves the gradient line"),
    ]
    for i, (name, eq) in enumerate(rows):
        a = E(t, 1 + 1.2 * i, 1.8 + 1.2 * i)
        y = 0.70 - 0.13 * i
        T(fig, 0.06, y, name, size=19, color=BLUE, a=a, weight="bold")
        T(fig, 0.06, y - 0.05, eq, size=15, color=INK, a=a)
        rule(fig, 0.06, 0.94, y - 0.09, a=a * 0.8, color=GRID)
    T(fig, 0.06, 0.20, "The landscape is the loss at every possible weight setting.  The backward pass gives the slope; training walks.",
      size=15, color=RED, a=E(t, 5.2, 6.4))


SCENES = [("title", 5.0, scene_title), ("curve", 30.0, scene_curve), ("valley", 45.0, scene_valley),
          ("ramp", 44.0, scene_ramp), ("walk", 34.0, scene_walk), ("outro", 9.0, scene_outro)]
PREVIEW_TIMES = {"title": [3.5], "curve": [7.0, 14.0, 24.0],
                 "valley": [8.0, 14.5, 26.0, 36.0, 44.0],
                 "ramp": [8.0, 14.0, 24.0, 35.0, 42.0],
                 "walk": [10.0, 20.0, 33.0], "outro": [8.0]}


def draw_frame(fig, scene, dur, t):
    fig.clf()
    fig.patch.set_facecolor(BG)
    scene(fig, t)
    a = max(1 - E(t, 0, 0.5), E(t, dur - 0.5, dur))
    if a > 0.01:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor="none",
                                 alpha=a, zorder=1000))


def main():
    ap = argparse.ArgumentParser(description="Minimalist loss-landscape film (NumPy + Matplotlib)")
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

    fmt = args.format
    if fmt == "auto":
        fmt = "mp4" if animation.writers.is_available("ffmpeg") else "gif"
    tag = args.scene if args.scene != "all" else "full"
    path = os.path.join(args.out, f"module8_landscape_{tag}.{fmt}")
    writer = (animation.FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
              if fmt == "mp4" else animation.PillowWriter(fps=min(args.fps, 15)))
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