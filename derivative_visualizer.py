#!/usr/bin/env python3
"""
Module 2 - "Inside the Backward Pass"  |  minimalist calculus film
Pure NumPy + Matplotlib (no autograd / DL frameworks, no Manim).

Design rules
  * One paper-white canvas, ink-black text, hairline axes, Computer Modern math.
  * Three muted accents, each with ONE fixed meaning in every chapter:
        BLUE  = the thing we vary / measure   (secant, x-direction, cost curve)
        GREEN = the second input / "good"     (y-direction, cost going down)
        RED   = the answer we are after       (tangent, slope, gradient, step)
  * One idea at a time. Every step is written on the right, in order, and
    stays on screen until the next idea needs the space.

Four chapters, each with a worked numeric example:
  1. Derivative          f(x) = x^2         secant -> tangent, algebra, f'(x) = 2x
  2. Partial derivatives f(x,y) = x^2+2y^2  freeze y, freeze x, slice the surface
  3. Gradient            grad f = [2x, 4y]  stack partials, directional derivative, perpendicular to level curves
  4. Gradient descent    w <- w - a*grad f  worked first step, then the full descent

Usage
  python module2_calculus_minimal.py                      # full film (mp4 if ffmpeg exists, else gif)
  python module2_calculus_minimal.py --scene derivative   # one chapter only
  python module2_calculus_minimal.py --preview            # PNG stills of every chapter
  python module2_calculus_minimal.py --fps 24 --dpi 100   # higher quality
Chapters: title, derivative, partial, gradient, descent, outro
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
BG = "#FBFAF7"      # paper
INK = "#1C1C1C"     # primary text and the function itself
MID = "#5C5A55"     # secondary text
GREY = "#9A978F"    # axes, ticks, captions
LIGHT = "#D6D3CA"   # contour lines, hairlines
GRID = "#ECE9E2"    # grid
BLUE = "#2D5B9A"
GREEN = "#3C7A5A"
RED = "#C4452F"


# ------------------------------------------------------------------ math (analytical, no autograd)
def f_1d(x):
    return x ** 2


def f_2d(x, y):
    return x ** 2 + 2.0 * y ** 2


def grad_2d(x, y):
    return np.array([2.0 * x, 4.0 * y])


ALPHA, START, N_STEPS = 0.15, (2.3, 2.0), 24


def gd_path():
    w = [np.array(START, dtype=float)]
    for _ in range(N_STEPS):
        w.append(w[-1] - ALPHA * grad_2d(*w[-1]))
    return np.array(w)


PATH = gd_path()
COSTS = f_2d(PATH[:, 0], PATH[:, 1])

_s = np.linspace(-2.5, 2.5, 41)
SX, SY = np.meshgrid(_s, _s)
SZ = f_2d(SX, SY)
GX, GY = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2.56, 2.56, 200))
GZ = f_2d(GX, GY)
DX, DY = np.meshgrid(np.linspace(-3.2, 3.8, 220), np.linspace(-2.6, 3.4, 220))
DZ = f_2d(DX, DY)


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
    ax.set_box_aspect((1, 1, 0.75))
    try:
        ax.computed_zorder = False
    except Exception:
        pass


def contour_lines(ax, X, Y, Z, levels, a=1.0):
    if a > 0.02:
        ax.contour(X, Y, Z, levels=levels, colors=LIGHT, linewidths=1.0, alpha=a)


def tangent_ends(x0, m, half=1.7):
    """Tangent segment with roughly constant on-screen length, however steep."""
    dx = half / np.sqrt(1 + (0.43 * m) ** 2)
    return np.array([x0 - dx, x0 + dx])


# ================================================================== SCENE 0: title
def scene_title(fig, t):
    T(fig, 0.5, 0.74, "M O D U L E   2", size=12, color=GREY, ha="center", a=E(t, 0.0, 0.8))
    T(fig, 0.5, 0.64, "Inside the Backward Pass", size=44, ha="center", a=E(t, 0.2, 1.2))
    T(fig, 0.5, 0.55, "Derivative   ·   Partial derivative   ·   Gradient   ·   Gradient descent",
      size=15, color=MID, ha="center", a=E(t, 1.0, 2.0))
    ax = fig.add_axes([0.38, 0.10, 0.24, 0.32])
    ax.axis("off")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.4, 4.5)
    xs = np.linspace(-2, 2, 200)
    n = max(2, int(200 * E(t, 1.0, 2.2)))
    ax.plot(xs[:n], xs[:n] ** 2, color=INK, lw=2.2)
    pa = E(t, 1.8, 2.4)
    if pa > 0.02:
        x = -1.8 * np.exp(-max(t - 2.2, 0.0))
        dot(ax, x, x ** 2, RED, ms=10 * pa)


# ================================================================== SCENE 1: derivative
DT = 2.2          # seconds per value of h
T_SEC, T_TAN, T_ALG, T_SWEEP = 4.0, 15.0, 21.0, 31.0


def scene_derivative(fig, t):
    header(fig, "1", "The derivative: the slope at a single point", E(t, 0, 0.8))
    T(fig, 0.05, 0.86, r"How steep is $f(x)=x^2$ at the point $x=1$?", size=15, color=MID, a=E(t, 0.8, 1.6))

    ax = mk_ax(fig, [0.06, 0.10, 0.50, 0.70], (-3.1, 3.1), (-1.2, 10), r"$x$", r"$f(x)$", center=True)
    xs = np.linspace(-3.1, 3.1, 300)
    n = max(2, int(300 * E(t, 1.5, 3.0)))
    ax.plot(xs[:n], xs[:n] ** 2, color=INK, lw=2.4)

    # operating point x0
    if t < T_SWEEP:
        x0 = 1.0
    elif t < T_SWEEP + 3:
        x0 = 1.0 - 3.6 * E(t, T_SWEEP, T_SWEEP + 3)
    else:
        x0 = -2.6 + 5.2 * E(t, T_SWEEP + 3, T_SWEEP + 8)
    f0 = x0 ** 2
    pa = E(t, 2.5, 3.3)
    if pa > 0.02:
        dot(ax, x0, f0, INK, ms=9 * pa, z=7)
    if t < T_SWEEP and pa > 0.02:
        ax.text(0.8, 1.45, "P = (1, 1)", color=INK, fontsize=13, ha="right", va="center", alpha=pa)

    # ---- secant phase: h = 2, 1, 0.5, 0.1, 0.01
    hs = [2.0, 1.0, 0.5, 0.1, 0.01]
    k, local = -1, 0.0
    if t >= T_SEC - 0.4:
        tt = max(t - T_SEC, 0.0)
        k = min(int(tt / DT), 4)
        local = tt - DT * k
    sec_a = E(t, T_SEC - 0.4, T_SEC + 0.4) * (1 - E(t, T_TAN, T_TAN + 1))
    if sec_a > 0.02 and k >= 0:
        h = hs[0] if k == 0 else hs[k - 1] + (hs[k] - hs[k - 1]) * E(local, 0, 1.0)
        fq = (x0 + h) ** 2
        sl = (fq - f0) / h
        xl = np.array([-3.1, 3.1])
        ax.plot(xl, f0 + sl * (xl - x0), color=BLUE, lw=2.0, alpha=sec_a)
        ax.plot([x0, x0 + h], [f0, f0], color=BLUE, lw=1.3, ls=":", alpha=sec_a)
        ax.plot([x0 + h, x0 + h], [f0, fq], color=BLUE, lw=1.3, ls=":", alpha=sec_a)
        dot(ax, x0 + h, fq, BLUE, ms=8 * sec_a, z=6)
        if h >= 0.5:
            ax.text(x0 + h / 2, f0 - 0.55, r"$h$", color=BLUE, fontsize=14, ha="center", va="center", alpha=sec_a)
            if x0 + h > 2.5:
                ax.text(x0 + h - 0.12, (f0 + fq) / 2, r"$\Delta f$", color=BLUE, fontsize=14, ha="right", va="center", alpha=sec_a)
            else:
                ax.text(x0 + h + 0.12, (f0 + fq) / 2, r"$\Delta f$", color=BLUE, fontsize=14, ha="left", va="center", alpha=sec_a)

    # ---- right column, phase A: table of shrinking h
    a_tab = E(t, T_SEC - 0.4, T_SEC + 0.4) * (1 - E(t, T_TAN, T_TAN + 0.8))
    X = 0.64
    T(fig, X, 0.80, "Shrink h and watch the slope settle", size=14, color=MID, a=a_tab)
    T(fig, X, 0.72, r"slope $=\dfrac{f(1+h)-f(1)}{h}$", size=20, a=a_tab)
    cols = (0.70, 0.82, 0.94)
    T(fig, cols[0], 0.62, r"$h$", size=14, color=GREY, a=a_tab, ha="right")
    T(fig, cols[1], 0.62, r"$f(1+h)$", size=14, color=GREY, a=a_tab, ha="right")
    T(fig, cols[2], 0.62, "slope", size=13, color=GREY, a=a_tab, ha="right")
    rule(fig, X, 0.95, 0.595, a=a_tab)
    for i, hv in enumerate(hs):
        if i < k or (i == k and local > 1.0):
            y = 0.55 - 0.06 * i
            last = (i == 4)
            T(fig, cols[0], y, f"{hv:g}", size=15, a=a_tab, ha="right", color=INK)
            T(fig, cols[1], y, f"{(1 + hv) ** 2:.4f}", size=15, a=a_tab, ha="right", color=INK)
            T(fig, cols[2], y, f"{((1 + hv) ** 2 - 1) / hv:.3f}", size=15, a=a_tab, ha="right",
              color=RED if last else BLUE, weight="bold" if last else "normal")
    if k >= 0 and local > 1.0:
        hv = hs[k]
        T(fig, X, 0.20,
          r"$(%g)^2-1=%.4f$,   then divide by $h=%g$   $\Rightarrow$   $%.3f$"
          % (1 + hv, (1 + hv) ** 2 - 1, hv, ((1 + hv) ** 2 - 1) / hv),
          size=13, color=MID, a=a_tab)

    # ---- phase B: limit -> tangent
    tan_a = E(t, T_TAN, T_TAN + 1)
    if tan_a > 0.02:
        xl = tangent_ends(x0, 2 * x0)
        ax.plot(xl, f0 + 2 * x0 * (xl - x0), color=RED, lw=2.6, alpha=tan_a)
        dot(ax, x0, f0, INK, ms=9, z=9)
    a_lim = E(t, T_TAN, T_TAN + 1) * (1 - E(t, T_ALG - 0.6, T_ALG + 0.2))
    T(fig, X, 0.80, r"As $h\to0$, the secant becomes the tangent.", size=16, a=a_lim)
    T(fig, X, 0.72, "At x = 1 its slope is exactly 2.", size=16, color=RED, a=E(t, T_TAN + 1.5, T_TAN + 2.5) * a_lim)
    T(fig, X, 0.62, r"$\dfrac{f(1+h)-f(1)}{h}\ \to\ 2$", size=22, a=E(t, T_TAN + 3, T_TAN + 4) * a_lim)

    # ---- phase C: algebra for ANY x
    a_alg = 1 - E(t, T_SWEEP - 0.6, T_SWEEP + 0.2)
    T(fig, X, 0.80, "Do it for any x, not only x = 1", size=14, color=MID, a=E(t, T_ALG + 0.2, T_ALG + 1) * a_alg)
    T(fig, X, 0.70, r"$\dfrac{(x+h)^2-x^2}{h}$", size=22, a=E(t, T_ALG + 1.5, T_ALG + 2.5) * a_alg)
    T(fig, X, 0.58, r"$=\dfrac{x^2+2xh+h^2-x^2}{h}$", size=22, a=E(t, T_ALG + 3, T_ALG + 4) * a_alg)
    T(fig, X, 0.46, r"$=\dfrac{2xh+h^2}{h}=2x+h$", size=22, a=E(t, T_ALG + 5, T_ALG + 6) * a_alg)
    T(fig, X, 0.34, r"$h\to0:\qquad f'(x)=2x$", size=28, color=RED, a=E(t, T_ALG + 7, T_ALG + 8) * a_alg)
    T(fig, X, 0.24, r"check:  $x=1\ \Rightarrow\ f'(1)=2$", size=15, color=GREEN, a=E(t, T_ALG + 8.5, T_ALG + 9.5) * a_alg)

    # ---- phase D: sweep. slope at every x, traced on its own graph
    if t >= T_SWEEP:
        a_s = E(t, T_SWEEP, T_SWEEP + 1)
        ax2 = mk_ax(fig, [0.66, 0.10, 0.29, 0.34], (-3, 3), (-6.5, 6.5), r"$x$", r"slope", center=True,
                    title=r"slope at every $x$:  $f'(x)=2x$")
        ax2.plot([-3, 3], [-6, 6], color=LIGHT, lw=1.2)
        lo = min(1.0, x0) if t < T_SWEEP + 3 else -2.6
        hi = 1.0 if t < T_SWEEP + 3 else max(1.0, x0)
        xx = np.linspace(lo, hi, 80)
        ax2.plot(xx, 2 * xx, color=RED, lw=2.6)
        ax2.plot([x0, x0], [0, 2 * x0], color=RED, lw=1.2, ls=":")
        dot(ax2, x0, 2 * x0, RED, ms=9)
        T(fig, X, 0.72, r"$x_0=%+.2f$" % x0, size=18, a=a_s)
        T(fig, X, 0.65, r"slope $=2x_0=%+.2f$" % (2 * x0), size=18, color=RED, a=a_s)
        if x0 < -0.05:
            msg = "slope < 0: the curve is falling"
        elif x0 > 0.05:
            msg = "slope > 0: the curve is rising"
        else:
            msg = "slope = 0: the bottom of the bowl"
        T(fig, X, 0.57, msg, size=14, color=MID, a=a_s)


# ================================================================== SCENE 2: partial derivatives
def scene_partial(fig, t):
    header(fig, "2", "Partial derivatives: move one input, freeze the other", E(t, 0, 0.8))
    T(fig, 0.05, 0.86, r"$f(x,y)=x^2+2y^2$: two inputs, one output (a surface).   At the point $(1,1)$ the height is $f=3$.",
      size=15, color=MID, a=E(t, 0.8, 1.6))

    ax = fig.add_axes([0.0, 0.08, 0.58, 0.76], projection="3d")
    style3d(ax)
    ax.view_init(elev=26, azim=-62 + 22 * seg(t, 0, 30))
    fade = E(t, 0.4, 2.0)
    if fade > 0.02:
        ax.plot_surface(SX, SY, SZ, color="#E2DFD6", alpha=0.55 * fade, linewidth=0.3, edgecolor="#B9B5AA",
                        rstride=2, cstride=2, antialiased=True, zorder=1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(0, 19)
    ax.set_xlabel(r"$x$", labelpad=-4)
    ax.set_ylabel(r"$y$", labelpad=-4)
    ax.set_zlabel(r"$f$", labelpad=-6)

    plane_x = np.array([[-2.5, 2.5], [-2.5, 2.5]])
    plane_z = np.array([[0, 0], [19, 19]])
    xs = np.linspace(-2.5, 2.5, 80)
    TA, TB = 4.0, 15.5

    # Part A: freeze y = 1
    if TA <= t < TB - 0.5:
        n = max(2, int(80 * E(t, TA, TA + 2)))
        ax.plot_surface(plane_x, np.ones((2, 2)), plane_z, color=BLUE, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(xs[:n], np.ones(n), xs[:n] ** 2 + 2 + 0.15, color=BLUE, lw=3, zorder=5)
        if t > 11:
            s = np.linspace(-1.0, 1.0, 2)
            ax.plot(1 + s, np.ones(2), 3 + 2 * s + 0.15, color=RED, lw=2.4, zorder=6)
    # Part B: freeze x = 1
    if t >= TB:
        n = max(2, int(80 * E(t, TB, TB + 2)))
        ax.plot_surface(np.ones((2, 2)), plane_x, plane_z, color=GREEN, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(np.ones(n), xs[:n], 1 + 2 * xs[:n] ** 2 + 0.15, color=GREEN, lw=3, zorder=5)
        if t > 22:
            s = np.linspace(-0.6, 0.6, 2)
            ax.plot(np.ones(2), 1 + s, 3 + 4 * s + 0.15, color=RED, lw=2.4, zorder=6)
    if t > 2.5:
        ax.scatter([1], [1], [3.15], color=INK, s=55, depthshade=False, zorder=10)

    # right: the 1D slice, which is just a chapter-1 problem
    X = np.linspace(-2.5, 2.5, 200)
    if TA + 0.5 <= t < TB:
        a2 = mk_ax(fig, [0.66, 0.50, 0.29, 0.30], (-2.5, 2.5), (0, 14), r"$x$", r"value",
                   title=r"the slice  $g(x)=f(x,1)$")
        a2.plot(X, X ** 2 + 2, color=BLUE, lw=2.6)
        if t > 10.5:
            xl = np.array([-0.4, 2.4])
            a2.plot(xl, 3 + 2 * (xl - 1), color=RED, lw=2.4)
        dot(a2, 1, 3, INK, ms=8)
    if t >= TB:
        a2 = mk_ax(fig, [0.66, 0.50, 0.29, 0.30], (-2.5, 2.5), (0, 14), r"$y$", r"value",
                   title=r"the slice  $h(y)=f(1,y)$")
        a2.plot(X, 1 + 2 * X ** 2, color=GREEN, lw=2.6)
        if t > 21.5:
            xl = np.array([-0.2, 2.2])
            a2.plot(xl, 3 + 4 * (xl - 1), color=RED, lw=2.4)
        dot(a2, 1, 3, INK, ms=8)

    # right: line-by-line reasoning
    Xt = 0.62
    aA = 1 - E(t, TB - 0.8, TB - 0.2)
    T(fig, Xt, 0.38, r"Freeze $y=1$. Only $x$ can move.", size=15, color=BLUE, a=E(t, 4.8, 5.8) * aA)
    T(fig, Xt, 0.31, r"$g(x)=f(x,1)=x^2+2(1)^2=x^2+2$", size=17, a=E(t, 7, 8) * aA)
    T(fig, Xt, 0.24, r"$g'(x)=2x$        (the rule from chapter 1)", size=17, a=E(t, 9, 10) * aA)
    T(fig, Xt, 0.16, r"$\partial f/\partial x$ at $(1,1)\ =\ 2\cdot1\ =\ 2$", size=18, color=RED, a=E(t, 11.5, 12.5) * aA)

    T(fig, Xt, 0.38, r"Freeze $x=1$. Only $y$ can move.", size=15, color=GREEN, a=E(t, TB + 0.8, TB + 1.8))
    T(fig, Xt, 0.31, r"$h(y)=f(1,y)=1^2+2y^2=1+2y^2$", size=17, a=E(t, TB + 3, TB + 4))
    T(fig, Xt, 0.24, r"$h'(y)=4y$", size=17, a=E(t, TB + 5.5, TB + 6.5))
    T(fig, Xt, 0.16, r"$\partial f/\partial y$ at $(1,1)\ =\ 4\cdot1\ =\ 4$", size=18, color=RED, a=E(t, TB + 8, TB + 9))

    T(fig, 0.5, 0.04, r"$\partial f/\partial x=2x$  (treat $y$ as a constant)            "
                      r"$\partial f/\partial y=4y$  (treat $x$ as a constant)",
      size=17, color=INK, ha="center", a=E(t, 26.0, 27.5))


# ================================================================== SCENE 3: gradient
def scene_gradient(fig, t):
    header(fig, "3", "The gradient: every slope packed into one arrow", E(t, 0, 0.8))
    ax = mk_ax(fig, [0.05, 0.10, 0.50, 0.76], (-3, 3), (-2.56, 2.56), r"$x$", r"$y$", grid=False)
    ax.set_aspect("equal", adjustable="box")
    contour_lines(ax, GX, GY, GZ, np.arange(1, 19, 2), E(t, 0.3, 1.5))
    P = np.array([1.0, 1.0])
    pa = E(t, 1.5, 2.3)
    if pa > 0.02:
        dot(ax, P[0], P[1], INK, ms=9 * pa, z=8)
        ax.text(-0.02, 1.0, r"$P=(1,1)$", color=INK, fontsize=12, ha="right", va="center", alpha=pa)

    X = 0.60
    T(fig, X, 0.82, "Stack the two partial derivatives into one vector", size=15, color=MID, a=E(t, 2.5, 3.3))
    T(fig, X, 0.74, r"$\nabla f=\left[\dfrac{\partial f}{\partial x},\ \dfrac{\partial f}{\partial y}\right]=[\,2x,\ 4y\,]$",
      size=21, a=E(t, 3.0, 4.0))
    T(fig, X, 0.65, r"at $P$:   $\nabla f(1,1)=[\,2\cdot1,\ 4\cdot1\,]=[\,2,\ 4\,]$", size=18, color=RED, a=E(t, 5, 6))

    s = 0.35
    ga = np.array([2.0, 4.0]) * s
    arrow(ax, P, P + [ga[0], 0], BLUE, E(t, 3, 4.2))
    arrow(ax, P + [ga[0], 0], P + ga, GREEN, E(t, 4.4, 5.8))
    arrow(ax, P, P + ga, RED, E(t, 6, 7), lw=3.4)
    if t > 3.5:
        ax.text(P[0] + 0.05, P[1] - 0.3, r"$\partial f/\partial x=2$", color=BLUE, fontsize=11, alpha=E(t, 3.5, 4.5) * (1 - E(t, 20, 20.8)))
    if t > 5:
        ax.text(P[0] + ga[0] + 0.1, P[1] + 0.7, r"$\partial f/\partial y=4$", color=GREEN, fontsize=11, alpha=E(t, 5, 6) * (1 - E(t, 20, 20.8)))
    if t > 6.5:
        ax.text(P[0] + ga[0] / 2 - 0.12, P[1] + ga[1] / 2 + 0.1, r"$\nabla f$", color=RED, fontsize=16, ha="right", va="center", alpha=E(t, 6.5, 7.5))
        ax.text(-2.9, 2.3, "arrows drawn at 0.35 scale", color=GREY, fontsize=10, alpha=E(t, 6.5, 7.5))

    # ---- why is it the STEEPEST direction? rotate a unit vector u
    th_max = np.arctan2(4, 2)
    a_u = E(t, 8, 9) * (1 - E(t, 20, 20.8))
    if a_u > 0.02:
        th = (th_max + 2 * np.pi) * E(t, 9, 17)
        u = np.array([np.cos(th), np.sin(th)])
        ang = np.linspace(0, 2 * np.pi, 120)
        ax.plot(P[0] + 0.9 * np.cos(ang), P[1] + 0.9 * np.sin(ang), color=GREY, lw=1, ls=":", alpha=a_u)
        arrow(ax, P, P + 0.9 * u, INK, 1.0, lw=2.4, alpha=a_u)
        D = 2 * u[0] + 4 * u[1]
        T(fig, X, 0.55, r"Take a unit step in direction $u=(\cos\theta,\ \sin\theta)$", size=14, color=MID, a=a_u)
        T(fig, X, 0.48, r"slope along $u$:   $D_uf=\nabla f\cdot u=2\cos\theta+4\sin\theta$", size=16, a=a_u)
        T(fig, X, 0.41, r"$\theta=%5.1f^\circ$        $D_uf=%+.2f$" % (np.degrees(th) % 360, D), size=15, color=BLUE, a=a_u)
        ai = mk_ax(fig, [0.66, 0.08, 0.29, 0.16], (0, 360), (-5.5, 5.5), r"$\theta$ (degrees)", r"$D_uf$",
                   xticks=[0, 90, 180, 270, 360], yticks=[-4, 0, 4])
        tt = np.linspace(0, 360, 240)
        dd = 2 * np.cos(np.radians(tt)) + 4 * np.sin(np.radians(tt))
        ai.plot(tt, dd, color=LIGHT, lw=2.0)
        m = tt <= (np.degrees(th) % 360 if th < 2 * np.pi else 360)
        ai.plot(tt[m], dd[m], color=BLUE, lw=2.4)
        pk = E(t, 17.2, 18.2)
        if pk > 0.02:
            ai.axvline(63.43, color=RED, ls=":", lw=1.3, alpha=pk)
        dot(ai, np.degrees(th) % 360, D, RED, ms=8)
        T(fig, X, 0.34, r"Peak at $\theta=63.4^\circ$: exactly the direction of $\nabla f$.", size=14, color=RED, a=E(t, 17.2, 18.2) * a_u)
        T(fig, X, 0.29, r"$\max D_uf=\|\nabla f\|=\sqrt{20}\approx4.47$", size=15, color=RED, a=E(t, 17.8, 18.8) * a_u)

    # ---- orthogonality to the level curve
    if t >= 20.5:
        phi = np.linspace(0, 2 * np.pi, 240)
        n = max(2, int(240 * E(t, 20.5, 22.5)))
        ax.plot(np.sqrt(3) * np.cos(phi[:n]), np.sqrt(1.5) * np.sin(phi[:n]), color=INK, lw=2.2, zorder=5)
        T(fig, X, 0.55, "Level curve through P: every point with f = 3", size=15, color=INK, a=E(t, 20.5, 21.5))
        tau = np.array([-4.0, 2.0]) / np.hypot(4, 2)
        gh = np.array([2.0, 4.0]) / np.hypot(2, 4)
        a_t = E(t, 22.5, 23.5)
        ax.plot([P[0] - tau[0], P[0] + tau[0]], [P[1] - tau[1], P[1] + tau[1]], color=MID, lw=1.5, ls="--", alpha=a_t)
        sq = np.array([P + 0.2 * tau, P + 0.2 * tau + 0.2 * gh, P + 0.2 * gh])
        ax.plot(sq[:, 0], sq[:, 1], color=MID, lw=1.3, alpha=a_t)
        T(fig, X, 0.46, r"tangent direction along the curve:   $\tau=(-4,\ 2)$", size=15, a=E(t, 22.5, 23.5))
        T(fig, X, 0.37, r"$\nabla f\cdot\tau=2(-4)+4(2)=0\ \Rightarrow\ \nabla f\perp$ level curve", size=15, color=RED, a=E(t, 24.5, 25.5))
        arrow(ax, P, P - ga, MID, E(t, 26.5, 28), lw=3.0)
        T(fig, X, 0.27, r"$\nabla f$ points straight uphill.", size=16, color=RED, a=E(t, 26, 27))
        T(fig, X, 0.20, r"$-\nabla f$ (grey arrow) points straight downhill.", size=16, color=MID, a=E(t, 27.5, 28.5))


# ================================================================== SCENE 4: gradient descent
def scene_descent(fig, t):
    header(fig, "4", "Gradient descent: walk downhill, one step at a time", E(t, 0, 0.8))
    ax = mk_ax(fig, [0.05, 0.10, 0.50, 0.76], (-3.2, 3.8), (-2.6, 3.4), r"$w_1$", r"$w_2$", grid=False)
    ax.set_aspect("equal", adjustable="box")
    contour_lines(ax, DX, DY, DZ, [0.5, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36], E(t, 0.0, 1.0))
    ax.plot([0], [0], "o", ms=9, mfc=BG, mec=INK, mew=1.6, zorder=6)
    ax.text(0.2, -0.45, "minimum", color=MID, fontsize=11)

    # progress along the path
    if t < 7:
        p = 0.0
    elif t < 8.2:
        p = E(t, 7, 8.2)
    elif t < 10:
        p = 1.0
    else:
        q = (t - 10) / 0.6
        qi = np.floor(q)
        p = 1 + qi + ease(q - qi)
    p = min(p, float(N_STEPS))
    pi = int(np.floor(p))
    fr = p - pi
    pos = PATH[N_STEPS] if pi >= N_STEPS else PATH[pi] + (PATH[pi + 1] - PATH[pi]) * fr
    trail = np.vstack([PATH[:pi + 1], pos[None, :]])
    ax.plot(trail[:, 0], trail[:, 1], "-o", color=INK, lw=1.4, ms=3.5, mec=INK)

    # the first step, drawn explicitly: uphill gradient (grey), downhill step (red)
    if 4.7 <= t < 10.5:
        a_g = (1 - E(t, 9.5, 10.5))
        arrow(ax, PATH[0], PATH[0] + ALPHA * grad_2d(*PATH[0]), GREY, E(t, 4.7, 5.5), lw=2.4, alpha=a_g)
        if t > 5.2:
            ax.text(2.55, 3.25, r"$\alpha\nabla f$", color=MID, fontsize=13, alpha=E(t, 5.2, 5.8) * a_g)
    if 5.9 <= t < 10.5:
        arrow(ax, PATH[0], PATH[1], RED, E(t, 5.9, 6.9), lw=3.0, alpha=(1 - E(t, 9.5, 10.5)))
        if t > 6.5:
            ax.text(1.0, 1.9, r"$-\alpha\nabla f$", color=RED, fontsize=13, ha="right", alpha=E(t, 6.5, 7.2) * (1 - E(t, 9.5, 10.5)))
    dot(ax, pos[0], pos[1], RED, ms=11, z=9)
    if t < 7:
        ax.text(START[0] + 0.15, START[1] - 0.45, r"$w_0$", color=INK, fontsize=13)

    X = 0.60
    T(fig, X, 0.84, r"$w_{t+1}\;=\;w_t\;-\;\alpha\,\nabla f(w_t)$", size=28, color=INK, a=E(t, 0.8, 1.8))
    T(fig, X, 0.77, r"step size $\alpha=%.2f$,   start $w_0=(%.1f,\ %.1f)$" % (ALPHA, START[0], START[1]),
      size=14, color=MID, a=E(t, 1.8, 2.8))

    # worked first step
    g0 = grad_2d(*PATH[0])
    a_w = E(t, 2.5, 3.0) * (1 - E(t, 9.5, 10.5))
    w1 = PATH[1]
    T(fig, X, 0.68, r"1.  Gradient at $w_0$", size=14, color=MID, a=E(t, 2.6, 3.2) * a_w)
    T(fig, X, 0.63, r"$\nabla f(w_0)=[2\cdot%.1f,\ 4\cdot%.1f]=[%.2f,\ %.2f]$" % (START[0], START[1], g0[0], g0[1]),
      size=16, a=E(t, 3.4, 4.2) * a_w)
    T(fig, X, 0.55, r"2.  Scale by $\alpha$", size=14, color=MID, a=E(t, 4.2, 4.7) * a_w)
    T(fig, X, 0.50, r"$\alpha\nabla f=0.15\cdot[%.2f,\ %.2f]=[%.2f,\ %.2f]$" % (g0[0], g0[1], ALPHA * g0[0], ALPHA * g0[1]),
      size=16, a=E(t, 4.9, 5.7) * a_w)
    T(fig, X, 0.42, r"3.  Step against the gradient", size=14, color=MID, a=E(t, 5.9, 6.4) * a_w)
    T(fig, X, 0.37, r"$w_1=[%.2f,\ %.2f]-[%.2f,\ %.2f]=[%.2f,\ %.2f]$"
      % (START[0], START[1], ALPHA * g0[0], ALPHA * g0[1], w1[0], w1[1]), size=16, color=RED, a=E(t, 6.5, 7.3) * a_w)
    T(fig, X, 0.29, r"cost:   $f(w_0)=%.2f\ \to\ f(w_1)=%.2f$,  lower" % (COSTS[0], COSTS[1]),
      size=15, color=GREEN, a=E(t, 8.0, 9.0) * a_w)

    # live readout + cost curve
    a_l = E(t, 10, 11)
    k = min(pi, N_STEPS)
    if a_l > 0.02:
        w = PATH[k]
        g = grad_2d(*w)
        T(fig, X, 0.67, r"iteration  $t=%d$" % k, size=17, a=a_l)
        T(fig, X, 0.61, r"$w=(%+.3f,\ %+.3f)$" % (w[0], w[1]), size=16, a=a_l)
        T(fig, X, 0.55, r"$\nabla f=[%+.3f,\ %+.3f]$" % (g[0], g[1]), size=16, a=a_l)
        T(fig, X, 0.49, r"cost  $f(w)=%.5f$" % COSTS[k], size=16, color=GREEN, a=a_l)
        T(fig, X, 0.42, "Near the bottom the gradient shrinks, so the steps shrink too.", size=13, color=MID, a=E(t, 14, 15))
    if t >= 8:
        ac = mk_ax(fig, [0.66, 0.14, 0.29, 0.20], (0, N_STEPS), (0, 14), "iteration  t", r"cost $f(w)$")
        ac.plot(np.arange(k + 1), COSTS[:k + 1], "-o", color=GREEN, lw=2.2, ms=3.5, mec=GREEN)
    T(fig, 0.5, 0.04, "Look at the slope, step the other way, repeat.   That loop is how a network learns.",
      size=15, color=INK, ha="center", a=E(t, 24.5, 26))


# ================================================================== SCENE 5: outro
def scene_outro(fig, t):
    T(fig, 0.06, 0.88, "Recap", size=34, a=E(t, 0, 0.8))
    rule(fig, 0.06, 0.94, 0.82, a=E(t, 0, 0.8))
    rows = [
        (r"Derivative", r"slope of a one-input function:   $f(x)=x^2\ \Rightarrow\ f'(x)=2x$"),
        (r"Partial derivative", r"slope along one input, the others frozen:   $\partial f/\partial x=2x,\ \ \partial f/\partial y=4y$"),
        (r"Gradient", r"all the partials in one vector:   $\nabla f=[2x,\ 4y]$,  points uphill,  $\perp$ level curves"),
        (r"Gradient descent", r"repeat   $w\leftarrow w-\alpha\nabla f(w)$   until the bottom"),
    ]
    for i, (name, eq) in enumerate(rows):
        a = E(t, 1 + 1.2 * i, 1.8 + 1.2 * i)
        y = 0.70 - 0.13 * i
        T(fig, 0.06, y, name, size=19, color=BLUE, a=a, weight="bold")
        T(fig, 0.06, y - 0.05, eq, size=16, color=INK, a=a)
        rule(fig, 0.06, 0.94, y - 0.09, a=a * 0.8, color=GRID)
    T(fig, 0.06, 0.12, "The backward pass is this gradient, computed for millions of weights with the chain rule.",
      size=16, color=RED, a=E(t, 6.2, 7.5))


SCENES = [("title", 5.0, scene_title), ("derivative", 41.0, scene_derivative),
          ("partial", 30.0, scene_partial), ("gradient", 33.0, scene_gradient),
          ("descent", 29.0, scene_descent), ("outro", 9.0, scene_outro)]
PREVIEW_TIMES = {"title": [3.5], "derivative": [3.5, 7.0, 12.8, 17.5, 28.5, 36.5],
                 "partial": [10.0, 13.0, 21.0, 28.0],
                 "gradient": [6.8, 13.0, 19.0, 28.5],
                 "descent": [6.5, 9.0, 12.0, 20.0, 27.0], "outro": [8.0]}


def draw_frame(fig, scene, dur, t):
    fig.clf()
    fig.patch.set_facecolor(BG)
    scene(fig, t)
    a = max(1 - E(t, 0, 0.5), E(t, dur - 0.5, dur))
    if a > 0.01:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor="none",
                                 alpha=a, zorder=1000))


def main():
    ap = argparse.ArgumentParser(description="Minimalist calculus film (NumPy + Matplotlib)")
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
    path = os.path.join(args.out, f"module2_calculus_{tag}.{fmt}")
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