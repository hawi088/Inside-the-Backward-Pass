"""
INSIDE THE BACKWARD PASS  --  a quiet, editorial animation suite
Pure NumPy + Matplotlib. No autograd, no PyTorch, no TensorFlow.

Design language: warm paper background, ink type, hairline rules, flat shapes,
earth-tone accents (slate / sage / plum / ochre / brick). No glow, no gradients,
no neon. Colour is used for meaning only: slate = forward, brick = backward,
sage = update.

Every operation is its own animated beat:

  Part 1  Two Moons manifold, spotlight on the sample [x1, x2]
  Part 2  FORWARD PASS  (12 beats)
  Part 3  LOSS LANDSCAPE (parabola, residual, the square whose area is L, slope)
  Part 4  BACKWARD PASS (15 beats)
  Part 5  GRADIENT DESCENT (forward -> backward -> update, every iteration)

The parameters match the hand calculation exactly:
    z1 = 0.5*x1 + (-0.6)*x2 + 0.1  = -0.130      h1 = 0.4675
    z2 = 0.8*x1 +   0.4*x2  - 0.2  =  0.520      h2 = 0.6275
    yhat = 0.7*h1 - 0.5*h2 + 0.3   =  0.3135     L  = 0.4713

Usage:
    python 3b1b_backprop_suite.py                    # all parts -> GIF + PNG
    python 3b1b_backprop_suite.py --parts 2 4        # only some parts
    python 3b1b_backprop_suite.py --mp4              # also export MP4 (needs ffmpeg)
    python 3b1b_backprop_suite.py --still            # PNG stills only (fast preview)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba, to_rgb

try:
    from dataset.two_moons import make_two_moons
except Exception:  # self-contained fallback so the script always runs
    def make_two_moons(n_samples=500, noise=0.2, seed=42):
        rng = np.random.default_rng(seed)
        n = n_samples // 2
        t1 = rng.uniform(0, np.pi, n)
        t0 = rng.uniform(0, np.pi, n)
        up = np.c_[np.cos(t1), np.sin(t1)]                  # class 1 (upper moon)
        lo = np.c_[1 - np.cos(t0), 0.5 - np.sin(t0)]        # class 0 (lower moon)
        X = np.vstack([lo, up]) + rng.normal(0, noise, (2 * n, 2))
        y = np.r_[np.zeros(n), np.ones(n)].reshape(-1, 1)
        return X, y

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "axes.unicode_minus": True,
    "font.family": "DejaVu Sans",
})
SERIF = "DejaVu Serif"
MONO = "DejaVu Sans Mono"

# ==============================================================================
# PALETTE  (paper + ink, muted earth tones)
# ==============================================================================
C_BG = "#F4F1EA"            # paper
C_TEXT = "#1E1E1E"          # ink
C_SUB = "#7B776F"           # secondary text
C_HAIR = "#CFC9BC"          # hairlines
C_LINE = "#E2DDD2"          # faint grid

C_X1, C_X2 = "#0070C0", "#5B7F6A"      # inputs  : slate, sage
C_H1, C_H2 = "#3F7090", "#7D5A78"      # hidden  : lighter slate, plum
C_Y = "#A8782F"                         # output  : ochre
C_ACC = "#B2432F"                       # accent  : brick (loss / backward)
C_GREEN = "#4B7B5E"                     # update / converged
C_GOLD = C_PINK = C_ACC                 # legacy names -> single accent
C_POS, C_NEG = "#6F7F8C", "#B2432F"
EDGE_C = "#9C968A"

# ==============================================================================
# THE MATH  (all hand-written)
# ==============================================================================
X_PT = np.array([0.50, 0.80])
Y_TRUE = 1.0
# Naming convention in code:  w_ij  =  weight on the synapse  x_i -> h_j
#   z1 = x1*w11 + x2*w21 + bh1   ->  0.5*x1 + (-0.6)*x2 + 0.1
#   z2 = x1*w12 + x2*w22 + bh2   ->  0.8*x1 +  0.4*x2  - 0.2
# (the paper's W1 rows are neurons, so its w12 is this code's w21, and vice versa)
P0 = dict(w11=0.50, w21=-0.60, w12=0.80, w22=0.40,    # layer 1  (w_ij : x_i -> h_j)
          bh1=0.10, bh2=-0.20,
          w1=0.70, w2=-0.50, bo=0.30)                 # layer 2
PARAM_ORDER = ["w11", "w21", "w12", "w22", "bh1", "bh2", "w1", "w2", "bo"]
PARAM_COL = dict(w11=C_X1, w21=C_X2, bh1=C_H1, w12=C_X1, w22=C_X2, bh2=C_H2,
                 w1=C_H1, w2=C_H2, bo=C_Y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(P, x):
    z1 = x[0] * P["w11"] + x[1] * P["w21"] + P["bh1"]
    z2 = x[0] * P["w12"] + x[1] * P["w22"] + P["bh2"]
    h1, h2 = sigmoid(z1), sigmoid(z2)
    yh = h1 * P["w1"] + h2 * P["w2"] + P["bo"]
    return dict(x=x, z1=z1, z2=z2, h1=h1, h2=h2, yhat=yh, L=(yh - Y_TRUE) ** 2)


def backward(P, c):
    x = c["x"]
    g = {}
    g["yhat"] = 2.0 * (c["yhat"] - Y_TRUE)
    g["bo"] = g["yhat"]
    g["w1"], g["w2"] = c["h1"] * g["yhat"], c["h2"] * g["yhat"]
    g["h1"], g["h2"] = g["yhat"] * P["w1"], g["yhat"] * P["w2"]
    g["s1"], g["s2"] = c["h1"] * (1 - c["h1"]), c["h2"] * (1 - c["h2"])
    g["z1"], g["z2"] = g["h1"] * g["s1"], g["h2"] * g["s2"]
    g["w11"], g["w21"], g["bh1"] = x[0] * g["z1"], x[1] * g["z1"], g["z1"]
    g["w12"], g["w22"], g["bh2"] = x[0] * g["z2"], x[1] * g["z2"], g["z2"]
    return g


C0 = forward(P0, X_PT)
G0 = backward(P0, C0)

# ==============================================================================
# SMALL UTILITIES
# ==============================================================================
def ramp(t, a, b):
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a * (1 - t) + b * t)


def par(v, d=3):
    return f"({v:.{d}f})" if v < 0 else f"{v:.{d}f}"


def fmtL(L):
    return f"{L:.4f}" if L >= 1e-3 else f"{L:.1e}"


def halo(lw=4):
    """paper-coloured knockout so text stays legible over lines (not a glow)"""
    return [pe.withStroke(linewidth=lw, foreground=C_BG)]


def drift(t, x, y, a, dy=0.03):
    """fade in while drifting up a hair"""
    t.set_alpha(a)
    t.set_position((x, y - dy * (1 - a)))


def add_background(fig):
    fig.patch.set_facecolor(C_BG)


def panel(ax, ec=C_HAIR):
    """hairline frame, no fill"""
    p = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.012",
                               transform=ax.transAxes, fc="none", ec=ec, lw=0.9,
                               zorder=0, clip_on=False)
    ax.add_patch(p)
    ax.axis("off")
    return p


def style_axes(ax, xlabel="", ylabel=""):
    ax.set_facecolor(C_BG)
    for k, s in ax.spines.items():
        s.set_color(C_HAIR)
        s.set_linewidth(0.9)
        if k in ("top", "right"):
            s.set_visible(False)
    ax.tick_params(colors=C_SUB, labelsize=8, length=3, width=0.6, color=C_HAIR)
    ax.grid(True, color=C_LINE, alpha=0.9, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=C_SUB, fontsize=9)
    ax.set_ylabel(ylabel, color=C_SUB, fontsize=9)


def header(fig, title, subtitle="INSIDE THE BACKWARD PASS"):
    fig.text(0.035, 0.945, subtitle, color=C_SUB, fontsize=9.5, fontweight="bold")
    fig.text(0.035, 0.893, title, color=C_TEXT, fontsize=24, family=SERIF)
    fig.add_artist(Line2D([0.035, 0.968], [0.868, 0.868], transform=fig.transFigure,
                          color=C_HAIR, lw=0.9))


def save_outputs(fig, update, frames, save_gif, save_mp4, save_static, still_frame, fps, dpi, tag):
    if save_static:
        update(still_frame)
        fig.savefig(save_static, dpi=150, facecolor=C_BG)
        print(f"[{tag}] PNG  -> {save_static}")
    if save_gif or save_mp4:
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
        kw = dict(savefig_kwargs=dict(facecolor=C_BG))
        if save_gif:
            ani.save(save_gif, writer="pillow", fps=fps, dpi=dpi, **kw)
            print(f"[{tag}] GIF  -> {save_gif}  ({frames} frames)")
        if save_mp4:
            ani.save(save_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=6000), dpi=dpi, **kw)
            print(f"[{tag}] MP4  -> {save_mp4}")
    plt.close(fig)


# ==============================================================================
# THE NETWORK CANVAS  (flat neurons, thin synapses, travelling dots)
# ==============================================================================
NODE = dict(x1=(0.16, 0.72), x2=(0.16, 0.28), h1=(0.74, 0.72), h2=(0.74, 0.28), y=(1.27, 0.50))
LOSS_XY = (1.52, 0.50)
NCOL = dict(x1=C_X1, x2=C_X2, h1=C_H1, h2=C_H2, y=C_Y)
SYM = dict(x1=r"$x_1$", x2=r"$x_2$", h1=r"$h_1$", h2=r"$h_2$", y=r"$\hat{y}$")
EDGE = dict(w11=("x1", "h1"), w21=("x2", "h1"), w12=("x1", "h2"), w22=("x2", "h2"),
            w1=("h1", "y"), w2=("h2", "y"))
BIAS_OF = dict(h1="bh1", h2="bh2", y="bo")
R = 0.085
TAG_T = dict(w11=0.70, w21=0.70, w12=0.70, w22=0.70, w1=0.66, w2=0.66)
ANN_T = dict(w11=0.30, w21=0.30, w12=0.30, w22=0.30, w1=0.34, w2=0.34)
OFF = dict(inn=(0, -0.034), z=(0, -R - 0.05), g1=(0, -R - 0.05), g2=(0, -R - 0.12),
           b=(0, R + 0.28), gb=(0.17, R + 0.12))
BIAS_PT_DY = R + 0.22


class NetCanvas:
    def __init__(self, ax, P, nd=2, slots=6):
        self.ax, self.nd = ax, nd
        ax.set_xlim(0, 1.62)
        ax.set_ylim(-0.10, 1.18)
        ax.set_aspect("equal")
        ax.axis("off")
        self.base, self.hl, self.tag = {}, {}, {}
        for k, (a, b) in EDGE.items():
            p0, p1 = self.seg(a, b)
            self.base[k], = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], zorder=2,
                                    solid_capstyle="round", color=EDGE_C)
            self.hl[k], = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=3.2, alpha=0, zorder=2.5,
                                  solid_capstyle="round", color=C_ACC)
            tx, ty = self.epos(k, TAG_T[k])
            self.tag[k] = ax.text(tx, ty, "", fontsize=8.5, ha="center", va="center", zorder=6,
                                  family=MONO, path_effects=halo(6))
        # loss link
        p0, p1 = self.seg("y", "L")
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C_SUB, lw=0.9, ls=(0, (3, 3)), alpha=0.8, zorder=2)
        # nodes
        self.body, self.ring, self.btag = {}, {}, {}
        for n, (cx, cy) in NODE.items():
            col = NCOL[n]
            if n in BIAS_OF:   # hairline from the bias label down into the neuron
                ax.plot([cx, cx], [cy + R * 1.04, cy + BIAS_PT_DY], color=C_HAIR, lw=0.9, zorder=1)
            self.ring[n] = patches.Circle((cx, cy), R, fc="none", ec=col, lw=1.0, alpha=0, zorder=3)
            ax.add_patch(self.ring[n])
            self.body[n] = patches.Circle((cx, cy), R, fc=C_BG, ec=col, lw=1.6, zorder=4)
            ax.add_patch(self.body[n])
            ax.text(cx, cy + 0.024, SYM[n], color=C_TEXT, fontsize=15, ha="center", va="center", zorder=5)
            if n in BIAS_OF:
                self.btag[n] = ax.text(cx + OFF["b"][0], cy + OFF["b"][1], "", color=C_SUB, fontsize=8.5,
                                       family=MONO, ha="center", va="center", zorder=5)
        # travelling dots: a short tail + a solid head
        self.slots = []
        for _ in range(slots):
            ln, = ax.plot([], [], lw=2.2, alpha=0, zorder=8.5, solid_capstyle="round")
            dt, = ax.plot([], [], "o", ms=6.5, alpha=0, zorder=9, mec=C_BG, mew=1.0)
            self.slots.append((ln, dt))
        self.set_weights(P)

    # ---- geometry
    def pt(self, name):
        if name == "L":
            return np.array(LOSS_XY)
        if name.startswith("bias:"):
            cx, cy = NODE[name[5:]]
            return np.array([cx, cy + BIAS_PT_DY])
        return np.array(NODE[name])

    def seg(self, a, b):
        A, B = self.pt(a), self.pt(b)
        u = (B - A) / np.linalg.norm(B - A)
        if a in NODE:
            A = A + u * R * 1.04
        if b in NODE:
            B = B - u * R * 1.04
        if b == "L":
            B = B - u * 0.06
        if a == "L":
            A = A + u * 0.06
        return A, B

    def epos(self, k, t):
        a, b = EDGE[k]
        A, B = self.pt(a), self.pt(b)
        return A + (B - A) * t

    # ---- text artists
    def text(self, node, slot, s, color=C_TEXT, alpha=0.0):
        if node == "L":
            t = self.ax.text(LOSS_XY[0], LOSS_XY[1], s, color=C_ACC, fontsize=12, ha="center", va="center",
                             zorder=6, bbox=dict(boxstyle="round,pad=0.4", fc=C_BG, ec=C_ACC, lw=1.0))
        else:
            cx, cy = NODE[node]
            ox, oy = OFF[slot]
            t = self.ax.text(cx + ox, cy + oy, s, color=color, fontsize=9.5 if slot == "inn" else 10.5,
                             ha="center", va="center", zorder=7, family=MONO if slot == "inn" else None,
                             path_effects=halo(5))
        t.set_alpha(alpha)
        return t

    def ann(self, edge, s, color, alpha=0.0):
        x, y = self.epos(edge, ANN_T[edge])
        t = self.ax.text(x, y, s, color=color, fontsize=9.5, ha="center", va="center", zorder=7,
                         family=MONO, fontweight="bold", path_effects=halo(6))
        t.set_alpha(alpha)
        return t

    # ---- dynamic state
    def set_weights(self, P, nd=None):
        nd = self.nd if nd is None else nd
        for k in EDGE:
            w = P[k]
            ln = self.base[k]
            ln.set_color(EDGE_C)
            ln.set_alpha(0.95)
            ln.set_linewidth(0.9 + 3.4 * min(abs(w), 1.0))
            ln.set_linestyle("-" if w >= 0 else (0, (4, 3)))     # dashed = negative weight
            self.tag[k].set_text(f"{w:+.{nd}f}")
            self.tag[k].set_color(C_SUB)
        for n, kb in BIAS_OF.items():
            self.btag[n].set_text(f"b = {P[kb]:+.{nd}f}")

    def reset(self):
        for k in EDGE:
            self.hl[k].set_alpha(0)
        for ln, dt in self.slots:
            ln.set_alpha(0)
            dt.set_alpha(0)
        for n in NODE:
            self.flash(n, 0.0)

    def highlight(self, k, s, color=C_ACC):
        self.hl[k].set_color(color)
        self.hl[k].set_alpha(0.9 * s)

    def flash(self, n, s):
        self.body[n].set_facecolor(mix(C_BG, NCOL[n], 0.22 * s))
        self.body[n].set_linewidth(1.6 + 1.4 * s)
        self.ring[n].set_radius(R * (1 + 0.55 * s))
        self.ring[n].set_alpha(0.55 * s)

    def pulse(self, slot, a, b, prog, color):
        """prog in (0,1): a short tail + dot travel along the synapse."""
        ln, dt = self.slots[slot]
        if prog <= 0 or prog >= 1:
            ln.set_alpha(0)
            dt.set_alpha(0)
            return
        A, B = self.seg(a, b)
        e = smooth(prog)
        t0 = max(0.0, e - 0.35)
        p0, p1 = A + (B - A) * t0, A + (B - A) * e
        ln.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        ln.set_color(color)
        ln.set_alpha(0.6)
        dt.set_data([p1[0]], [p1[1]])
        dt.set_color(color)
        dt.set_alpha(1.0)


# ==============================================================================
# STEP-BY-STEP STAGE ENGINE  (used by Part 2 forward and Part 4 backward)
# ==============================================================================
def St(title, color, sym, sub, res, note="", pulses=(), flash=None, texts=(), anns=(),
       mini=None, ledger=""):
    return dict(title=title, color=color, sym=sym, sub=sub, res=res, note=note, pulses=list(pulses),
                flash=flash, texts=list(texts), anns=list(anns), mini=mini, ledger=ledger)


def re_plain(s):
    """strip mathtext for the ledger's plain-text roadmap"""
    return (s.replace("$", "").replace("\\hat{y}", "ŷ").replace("\\sigma", "σ").replace("\\delta", "δ")
             .replace("\\partial", "∂").replace("_", "").replace("{", "").replace("}", "").replace("\\", ""))


def build_stage(stage_title, steps, setup_canvas, mini_cfg, caption, step_frames=30, tail=44):
    N = len(steps)
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    add_background(fig)
    header(fig, stage_title)
    counter = fig.text(0.968, 0.90, "", color=C_TEXT, fontsize=12, family=MONO, ha="right")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 0.88], height_ratios=[1.75, 0.95, 1.7],
                          left=0.035, right=0.968, bottom=0.085, top=0.845, wspace=0.05, hspace=0.13)
    ax_net = fig.add_subplot(gs[:, 0])
    ax_card = fig.add_subplot(gs[0, 1])
    ax_mini = fig.add_subplot(gs[1, 1])
    ax_led = fig.add_subplot(gs[2, 1])

    canvas = NetCanvas(ax_net, P0, nd=2, slots=4)
    setup_canvas(canvas)
    ax_net.text(0.81, -0.05, caption, color=C_SUB, fontsize=10, ha="center", va="center", style="italic")

    # artists per (node,slot) key, with the step that owns them
    art = {}
    for si, s in enumerate(steps):
        for node, slot, txt, col in s["texts"]:
            art.setdefault((node, slot), []).append((si, canvas.text(node, slot, txt, col)))
        for edge, txt, col in s["anns"]:
            art.setdefault((edge, "ann"), []).append((si, canvas.ann(edge, txt, col)))

    # ---- card
    panel(ax_card)
    ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1)
    ttl = ax_card.text(0.05, 0.94, "", fontsize=12.5, va="top")
    POS = dict(sym=(0.5, 0.72), sub=(0.5, 0.46), res=(0.5, 0.24))
    sym = ax_card.text(*POS["sym"], "", fontsize=17, ha="center", va="center", color=C_TEXT)
    sub = ax_card.text(*POS["sub"], "", fontsize=15, ha="center", va="center", color=C_TEXT)
    res = ax_card.text(*POS["res"], "", fontsize=22, ha="center", va="center")
    ul, = ax_card.plot([0.5, 0.5], [0.135, 0.135], lw=1.6, solid_capstyle="round")
    note = ax_card.text(0.5, 0.03, "", fontsize=9.5, ha="center", va="bottom", color=C_SUB, style="italic")

    # ---- mini plot
    style_axes(ax_mini, "z", "")
    zz = np.linspace(-4.5, 4.5, 300)
    ax_mini.plot(zz, mini_cfg["fn"](zz), color=C_TEXT, lw=1.6)
    ax_mini.set_xlim(-4.5, 4.5)
    ax_mini.set_ylim(*mini_cfg["ylim"])
    ax_mini.text(0.02, 0.9, mini_cfg["label"], transform=ax_mini.transAxes, color=C_TEXT, fontsize=12)
    dots = {}
    for name, (z, v, col, lab, ha) in mini_cfg["dots"].items():
        stem, = ax_mini.plot([z, z], [0, 0], color=col, lw=1.1, ls=(0, (3, 3)))
        dot, = ax_mini.plot([z], [0], "o", color=col, ms=9, mec=C_BG, mew=1.5, zorder=5)
        lbl = ax_mini.text(z + (0.2 if ha == "left" else -0.2), v * 0.5, lab, color=col, fontsize=11, ha=ha, va="center")
        lbl.set_alpha(0)
        dots[name] = (stem, dot, lbl, z, v)

    # ---- ledger
    panel(ax_led)
    ax_led.set_xlim(0, 1); ax_led.set_ylim(0, 1)
    ax_led.text(0.03, 0.965, "LEDGER", color=C_SUB, fontsize=8.5, fontweight="bold", va="top")
    rh = min(0.062, 0.90 / N)
    fs = 9.0 if N <= 13 else 8.4
    rows = [ax_led.text(0.03, 0.905 - i * rh, "", fontsize=fs, family=MONO, va="top") for i in range(N)]

    # ---- progress bar
    ax_pb = fig.add_axes([0.035, 0.040, 0.933, 0.006])
    ax_pb.axis("off"); ax_pb.set_xlim(0, N); ax_pb.set_ylim(0, 1)
    fills = []
    for i in range(N):
        ax_pb.add_patch(patches.Rectangle((i + 0.05, 0), 0.9, 1, fc=C_LINE, ec="none"))
        f = patches.Rectangle((i + 0.05, 0), 0.0001, 1, fc=steps[i]["color"], ec="none")
        ax_pb.add_patch(f); fills.append(f)

    total = N * step_frames + tail

    def update(frame):
        frame = int(frame)
        cur = min(frame // step_frames, N - 1)
        p = 1.0 if frame >= N * step_frames else (frame - cur * step_frames) / step_frames
        s = steps[cur]
        canvas.reset()

        # crossfading text artists
        for key, lst in art.items():
            vis = [1.0 if si < cur else (smooth(ramp(p, .62, 1)) if si == cur else 0.0) for si, _ in lst]
            for i, (_, a) in enumerate(lst):
                nxt = vis[i + 1] if i + 1 < len(lst) else 0.0
                al = vis[i] * (1 - nxt)
                a.set_alpha(al)
                if a.get_bbox_patch() is not None:
                    a.get_bbox_patch().set_alpha(al)

        # pulses + synapse highlight + neuron breathe
        ps = ramp(p, 0, .62)
        hl = 1 - ramp(p, .62, .92)
        for i, (a, b, col) in enumerate(s["pulses"]):
            canvas.pulse(i, a, b, ps, col)
            for k, (ea, eb) in EDGE.items():
                if {a, b} == {ea, eb}:
                    canvas.highlight(k, hl if ps > 0 else 0, col)
        if s["flash"]:
            canvas.flash(s["flash"], np.sin(np.pi * ramp(p, .42, .98)))

        # card: every line drifts in, one after the other
        ttl.set_text(s["title"]); ttl.set_color(s["color"]); ttl.set_alpha(smooth(ramp(p, 0, .15)))
        sym.set_text(s["sym"]); drift(sym, *POS["sym"], smooth(ramp(p, 0, .22)))
        sub.set_text(s["sub"]); drift(sub, *POS["sub"], smooth(ramp(p, .22, .45)))
        res.set_text(s["res"]); res.set_color(s["color"]); drift(res, *POS["res"], smooth(ramp(p, .58, .82)))
        pr_ul = smooth(ramp(p, .62, .95))
        ul.set_data([0.5 - 0.22 * pr_ul, 0.5 + 0.22 * pr_ul], [0.135, 0.135]); ul.set_color(s["color"])
        note.set_text(s["note"]); note.set_alpha(smooth(ramp(p, 0, .3)))

        # mini plot dots
        for name, (stem, dot, lbl, z, v) in dots.items():
            owner = [i for i, st in enumerate(steps) if st["mini"] == name]
            if not owner:
                continue
            oi = owner[0]
            pr = 1.0 if oi < cur else (smooth(ramp(p, 0, .6)) if oi == cur else 0.0)
            stem.set_data([z, z], [0, v * pr]); dot.set_data([z], [v * pr])
            stem.set_alpha(1.0 if pr > 0 else 0.0); dot.set_alpha(1.0 if pr > 0 else 0.0)
            lbl.set_alpha(ramp(pr, .85, 1))

        # ledger: current row writes itself in
        for i, r_ in enumerate(rows):
            st = steps[i]
            if i < cur:
                r_.set_text(f"{i + 1:>2}. {st['ledger']}"); r_.set_color(to_rgba(C_TEXT, 0.9))
            elif i == cur and p > .75:
                r_.set_text(f"{i + 1:>2}. {st['ledger']}")
                r_.set_color(to_rgba(st["color"], 0.35 + 0.65 * ramp(p, .75, .95)))
            elif i == cur:
                r_.set_text(f"{i + 1:>2}. {re_plain(st['title'])}"); r_.set_color(to_rgba(st["color"], 1.0))
            else:
                r_.set_text(f"{i + 1:>2}. {re_plain(st['title'])}"); r_.set_color(to_rgba(C_SUB, 0.4))
            r_.set_fontweight("bold" if i == cur else "normal")

        counter.set_text(f"{cur + 1:02d} / {N:02d}")
        for i, f in enumerate(fills):
            frac = 1.0 if i < cur else (ramp(p, 0, 1) if i == cur else 0.0)
            f.set_width(max(0.0001, 0.9 * frac))
        return []

    return fig, update, total


# ==============================================================================
# PART 2: THE FORWARD PASS  (12 beats)
# ==============================================================================
def forward_steps(P, c):
    x1, x2 = c["x"]
    p11, p21, p12, p22 = x1 * P["w11"], x2 * P["w21"], x1 * P["w12"], x2 * P["w22"]
    q1, q2 = c["h1"] * P["w1"], c["h2"] * P["w2"]
    z1, z2, h1, h2, yh, L = c["z1"], c["z2"], c["h1"], c["h2"], c["yhat"], c["L"]
    S = []
    # ---- hidden neuron h1
    S.append(St(r"Multiply $x_1$ by $w_{11}$", C_X1, r"$x_1\cdot w_{11}$",
                rf"${x1:.2f}\times{par(P['w11'], 2)}$", rf"$=\;{p11:.3f}$",
                "Every input is scaled by the weight on its synapse.",
                [("x1", "h1", C_X1)], "h1", [("h1", "z", rf"$\Sigma={p11:.3f}$", C_H1)],
                [("w11", f"{p11:+.3f}", C_X1)], None, f"x1·w11 = {p11:+.3f}"))
    S.append(St(r"Multiply $x_2$ by $w_{21}$", C_X2, r"$x_2\cdot w_{21}$",
                rf"${x2:.2f}\times{par(P['w21'], 2)}$", rf"$=\;{p21:.3f}$",
                "A negative weight: this synapse is inhibitory (dashed).",
                [("x2", "h1", C_X2)], "h1", [("h1", "z", rf"$\Sigma={p11 + p21:.3f}$", C_H1)],
                [("w21", f"{p21:+.3f}", C_X2)], None, f"x2·w21 = {p21:+.3f}"))
    S.append(St(r"Add the bias $b_1$  →  pre-activation $z_1$", C_ACC,
                r"$z_1=x_1w_{11}+x_2w_{21}+b_1$",
                rf"$z_1={p11:.3f}+{par(p21)}+{par(P['bh1'], 3)}$", rf"$z_1={z1:.3f}$",
                "The bias nudges the sum, but z₁ stays slightly negative.",
                [("bias:h1", "h1", C_ACC)], "h1", [("h1", "z", rf"$z_1={z1:.3f}$", C_H1)], [], None,
                f"z1 = {p11:.3f}{p21:+.3f}{P['bh1']:+.3f} = {z1:+.3f}"))
    # ---- hidden neuron h2
    S.append(St(r"Multiply $x_1$ by $w_{12}$", C_X1, r"$x_1\cdot w_{12}$",
                rf"${x1:.2f}\times{par(P['w12'], 2)}$", rf"$=\;{p12:.3f}$",
                "A strong positive weight: this synapse is excitatory.",
                [("x1", "h2", C_X1)], "h2", [("h2", "z", rf"$\Sigma={p12:.3f}$", C_H2)],
                [("w12", f"{p12:+.3f}", C_X1)], None, f"x1·w12 = {p12:+.3f}"))
    S.append(St(r"Multiply $x_2$ by $w_{22}$", C_X2, r"$x_2\cdot w_{22}$",
                rf"${x2:.2f}\times{par(P['w22'], 2)}$", rf"$=\;{p22:.3f}$",
                "Another excitatory pull, from the second input.",
                [("x2", "h2", C_X2)], "h2", [("h2", "z", rf"$\Sigma={p12 + p22:.3f}$", C_H2)],
                [("w22", f"{p22:+.3f}", C_X2)], None, f"x2·w22 = {p22:+.3f}"))
    S.append(St(r"Add the bias $b_2$  →  pre-activation $z_2$", C_ACC,
                r"$z_2=x_1w_{12}+x_2w_{22}+b_2$",
                rf"$z_2={p12:.3f}+{p22:.3f}+{par(P['bh2'])}$", rf"$z_2={z2:.3f}$",
                "The negative bias trims the sum, but the net stimulus is positive.",
                [("bias:h2", "h2", C_ACC)], "h2", [("h2", "z", rf"$z_2={z2:.3f}$", C_H2)], [], None,
                f"z2 = {p12:.3f}{p22:+.3f}{P['bh2']:+.3f} = {z2:+.3f}"))
    # ---- sigmoid squashing
    S.append(St(r"Squash $z_1$ through the sigmoid", C_H1,
                r"$h_1=\sigma(z_1)=\dfrac{1}{1+e^{-z_1}}$",
                rf"$\dfrac{{1}}{{1+e^{{{-z1:.3f}}}}}=\dfrac{{1}}{{1+{np.exp(-z1):.4f}}}$", rf"$h_1={h1:.4f}$",
                "Near zero input, σ sits close to one half.",
                [], "h1", [("h1", "inn", f"{h1:.3f}", C_H1)], [], "h1", f"h1 = σ({z1:+.3f}) = {h1:.4f}"))
    S.append(St(r"Squash $z_2$ through the sigmoid", C_H2,
                r"$h_2=\sigma(z_2)=\dfrac{1}{1+e^{-z_2}}$",
                rf"$\dfrac{{1}}{{1+e^{{{-z2:.3f}}}}}=\dfrac{{1}}{{1+{np.exp(-z2):.4f}}}$",
                rf"$h_2={h2:.4f}$", "Positive input pushes σ above one half.",
                [], "h2", [("h2", "inn", f"{h2:.3f}", C_H2)], [], "h2", f"h2 = σ({z2:+.3f}) = {h2:.4f}"))
    # ---- output layer
    S.append(St(r"Send $h_1$ through $w_1$", C_H1, r"$h_1\cdot w_1$",
                rf"${h1:.4f}\times{par(P['w1'], 2)}$", rf"$=\;{q1:.4f}$",
                "Hidden activations become the next layer's inputs.",
                [("h1", "y", C_H1)], "y", [("y", "z", rf"$\Sigma={q1:.3f}$", C_Y)],
                [("w1", f"{q1:+.3f}", C_H1)], None, f"h1·w1 = {q1:+.4f}"))
    S.append(St(r"Send $h_2$ through $w_2$", C_H2, r"$h_2\cdot w_2$",
                rf"${h2:.4f}\times{par(P['w2'], 2)}$", rf"$=\;{q2:.4f}$",
                "This neuron votes against the output.",
                [("h2", "y", C_H2)], "y", [("y", "z", rf"$\Sigma={q1 + q2:.3f}$", C_Y)],
                [("w2", f"{q2:+.3f}", C_H2)], None, f"h2·w2 = {q2:+.4f}"))
    S.append(St(r"Add output bias  →  prediction $\hat{y}$", C_Y,
                r"$\hat{y}=h_1w_1+h_2w_2+b_{out}$",
                rf"$\hat{{y}}={q1:.4f}+{par(q2, 4)}+{P['bo']:.2f}$", rf"$\hat{{y}}={yh:.4f}$",
                "The network's answer for this sample.",
                [("bias:y", "y", C_ACC)], "y",
                [("y", "inn", f"{yh:.3f}", C_Y), ("y", "z", rf"$\hat y={yh:.3f}$", C_Y)], [], None,
                f"ŷ = {q1:.4f}{q2:+.4f}+{P['bo']:.2f} = {yh:.4f}"))
    S.append(St(r"Compare with the truth  →  loss $L$", C_ACC, r"$L=(\hat{y}-y)^2$",
                rf"$({yh:.4f}-{Y_TRUE:.3f})^2$", rf"$L={L:.4f}$",
                f"Prediction is too low by {Y_TRUE - yh:.4f}.",
                [("y", "L", C_ACC)], None, [("L", "badge", rf"$L={L:.4f}$", C_ACC)], [], None,
                f"L = ({yh - Y_TRUE:+.4f})² = {L:.4f}"))
    return S


def render_part2_forward(save_gif=None, save_mp4=None, save_static=None, fps=24, dpi=80, step_frames=30):
    print("[Part 2] Forward pass: 12 beats")
    steps = forward_steps(P0, C0)

    def setup(cv):
        for n, v in (("x1", X_PT[0]), ("x2", X_PT[1])):
            cv.text(n, "inn", f"{v:.2f}", NCOL[n], alpha=1.0)

    # z1 < z2, so h1's label goes on the left of its dot and h2's on the right
    sig = dict(fn=sigmoid, color=C_H1, ylim=(-0.05, 1.08), label=r"$\sigma(z)$",
               dots=dict(h1=(C0["z1"], C0["h1"], C_H1, rf"$h_1={C0['h1']:.4f}$", "right"),
                         h2=(C0["z2"], C0["h2"], C_H2, rf"$h_2={C0['h2']:.4f}$", "left")))
    fig, update, total = build_stage("The Forward Pass", steps, setup, sig,
                                     "z = Σ (input × weight) + bias        h = σ(z)", step_frames)
    save_outputs(fig, update, total, save_gif, save_mp4, save_static, step_frames * 7 + 22, fps, dpi, "Part 2")


# ==============================================================================
# PART 4: THE BACKWARD PASS  (15 beats)
# ==============================================================================
def backward_steps(P, c, g):
    x1, x2 = c["x"]
    yh, h1, h2 = c["yhat"], c["h1"], c["h2"]
    gy = g["yhat"]
    A = C_ACC
    S = []
    S.append(St(r"Start at the loss: how does $L$ react to $\hat{y}$?", A,
                r"$\delta_{\hat y}\equiv\dfrac{\partial L}{\partial\hat y}=2(\hat y-y)$",
                rf"$2\,({yh:.4f}-{Y_TRUE:.3f})$", rf"$\delta_{{\hat y}}={gy:.4f}$",
                "Negative: nudging ŷ upward would reduce the loss.  (δ ≡ ∂L/∂·)",
                [("L", "y", A)], "y", [("y", "g1", rf"$\delta_{{\hat y}}={gy:.3f}$", A)], [], None,
                f"δŷ = 2(ŷ-y) = {gy:+.4f}"))
    S.append(St(r"Output bias $b_{out}$", C_Y, r"$\dfrac{\partial L}{\partial b_{out}}=\delta_{\hat y}\cdot 1$",
                rf"${par(gy, 4)}\times 1$", rf"$={g['bo']:.4f}$", "The bias adds directly to ŷ, so its local slope is 1.",
                [("y", "bias:y", A)], "y", [("y", "gb", rf"$\partial b={g['bo']:.3f}$", C_Y)], [], None,
                f"∂L/∂bo = {g['bo']:+.4f}"))
    S.append(St(r"Output weight $w_1$", C_H1, r"$\dfrac{\partial L}{\partial w_1}=\delta_{\hat y}\cdot h_1$",
                rf"${par(gy, 4)}\times{h1:.4f}$", rf"$={g['w1']:.4f}$", "The weight's local slope is the activation that flowed through it.",
                [("y", "h1", A)], None, [], [("w1", f"{g['w1']:+.3f}", C_H1)], None, f"∂L/∂w1 = {g['w1']:+.4f}"))
    S.append(St(r"Output weight $w_2$", C_H2, r"$\dfrac{\partial L}{\partial w_2}=\delta_{\hat y}\cdot h_2$",
                rf"${par(gy, 4)}\times{h2:.4f}$", rf"$={g['w2']:.4f}$", "Same rule, other synapse.",
                [("y", "h2", A)], None, [], [("w2", f"{g['w2']:+.3f}", C_H2)], None, f"∂L/∂w2 = {g['w2']:+.4f}"))
    S.append(St(r"Push the signal back into $h_1$", C_H1, r"$\delta_{h_1}=\delta_{\hat y}\cdot w_1$",
                rf"${par(gy, 4)}\times{par(P['w1'], 2)}$", rf"$\delta_{{h_1}}={g['h1']:.4f}$",
                "Chain rule: upstream sensitivity × local slope (the weight).",
                [("y", "h1", A)], "h1", [("h1", "g1", rf"$\delta_{{h_1}}={g['h1']:.3f}$", C_H1)], [], None,
                f"δh1 = δŷ·w1 = {g['h1']:+.4f}"))
    S.append(St(r"Push the signal back into $h_2$", C_H2, r"$\delta_{h_2}=\delta_{\hat y}\cdot w_2$",
                rf"${par(gy, 4)}\times{par(P['w2'], 2)}$", rf"$\delta_{{h_2}}={g['h2']:+.4f}$",
                "Sign flips: w2 is negative, so h2 wants to move the OTHER way.",
                [("y", "h2", A)], "h2", [("h2", "g1", rf"$\delta_{{h_2}}={g['h2']:+.3f}$", C_H2)], [], None,
                f"δh2 = δŷ·w2 = {g['h2']:+.4f}"))
    S.append(St(r"Through the sigmoid at $h_1$", C_H1, r"$\delta_{z_1}=\delta_{h_1}\cdot\sigma'(z_1)=\delta_{h_1}\,h_1(1-h_1)$",
                rf"${par(g['h1'], 4)}\times{g['s1']:.4f}$", rf"$\delta_{{z_1}}={g['z1']:.4f}$",
                "σ′(z) = σ(1-σ): steepest at z = 0, flat in the tails.",
                [], "h1", [("h1", "g2", rf"$\delta_{{z_1}}={g['z1']:.3f}$", C_H1)], [], "h1", f"δz1 = δh1·σ' = {g['z1']:+.4f}"))
    S.append(St(r"Through the sigmoid at $h_2$", C_H2, r"$\delta_{z_2}=\delta_{h_2}\cdot\sigma'(z_2)=\delta_{h_2}\,h_2(1-h_2)$",
                rf"${g['h2']:.4f}\times{g['s2']:.4f}$", rf"$\delta_{{z_2}}={g['z2']:+.4f}$",
                "Slightly less steep: z₂ is further from 0 than z₁.",
                [], "h2", [("h2", "g2", rf"$\delta_{{z_2}}={g['z2']:+.3f}$", C_H2)], [], "h2", f"δz2 = δh2·σ' = {g['z2']:+.4f}"))
    S.append(St(r"First-layer weight $w_{11}$", C_X1, r"$\dfrac{\partial L}{\partial w_{11}}=\delta_{z_1}\cdot x_1$",
                rf"${par(g['z1'], 4)}\times{x1:.2f}$", rf"$={g['w11']:.4f}$", "Local slope of z₁ w.r.t. w₁₁ is the input x₁.",
                [("h1", "x1", A)], None, [], [("w11", f"{g['w11']:+.3f}", C_X1)], None, f"∂L/∂w11 = {g['w11']:+.4f}"))
    S.append(St(r"First-layer weight $w_{21}$", C_X2, r"$\dfrac{\partial L}{\partial w_{21}}=\delta_{z_1}\cdot x_2$",
                rf"${par(g['z1'], 4)}\times{x2:.2f}$", rf"$={g['w21']:.4f}$", "Larger input, larger gradient.",
                [("h1", "x2", A)], None, [], [("w21", f"{g['w21']:+.3f}", C_X2)], None, f"∂L/∂w21 = {g['w21']:+.4f}"))
    S.append(St(r"Bias $b_1$", C_H1, r"$\dfrac{\partial L}{\partial b_1}=\delta_{z_1}\cdot 1$",
                rf"${par(g['z1'], 4)}\times 1$", rf"$={g['bh1']:.4f}$", "Bias slopes are just the neuron's δz.",
                [("h1", "bias:h1", A)], "h1", [("h1", "gb", rf"$\partial b={g['bh1']:.3f}$", C_H1)], [], None,
                f"∂L/∂bh1 = {g['bh1']:+.4f}"))
    S.append(St(r"First-layer weight $w_{12}$", C_X1, r"$\dfrac{\partial L}{\partial w_{12}}=\delta_{z_2}\cdot x_1$",
                rf"${g['z2']:.4f}\times{x1:.2f}$", rf"$={g['w12']:+.4f}$", "Opposite sign to w₁₁: this path pulls the other way.",
                [("h2", "x1", A)], None, [], [("w12", f"{g['w12']:+.3f}", C_X1)], None, f"∂L/∂w12 = {g['w12']:+.4f}"))
    S.append(St(r"First-layer weight $w_{22}$", C_X2, r"$\dfrac{\partial L}{\partial w_{22}}=\delta_{z_2}\cdot x_2$",
                rf"${g['z2']:.4f}\times{x2:.2f}$", rf"$={g['w22']:+.4f}$", "Same pattern, scaled by x₂.",
                [("h2", "x2", A)], None, [], [("w22", f"{g['w22']:+.3f}", C_X2)], None, f"∂L/∂w22 = {g['w22']:+.4f}"))
    S.append(St(r"Bias $b_2$", C_H2, r"$\dfrac{\partial L}{\partial b_2}=\delta_{z_2}\cdot 1$",
                rf"${g['z2']:.4f}\times 1$", rf"$={g['bh2']:+.4f}$", "Last of the nine gradients.",
                [("h2", "bias:h2", A)], "h2", [("h2", "gb", rf"$\partial b={g['bh2']:+.3f}$", C_H2)], [], None,
                f"∂L/∂bh2 = {g['bh2']:+.4f}"))
    gn = np.sqrt(sum(g[k] ** 2 for k in PARAM_ORDER))
    S.append(St(r"All nine gradients are ready", C_GREEN, r"$\nabla_\theta L=\left(\dfrac{\partial L}{\partial\theta_1},\ldots,\dfrac{\partial L}{\partial\theta_9}\right)$",
                rf"$\|\nabla_\theta L\|=\sqrt{{\sum_i g_i^2}}$", rf"$\|\nabla L\|={gn:.4f}$",
                "Next: step every parameter a little in the opposite direction.",
                [("y", "h1", C_GREEN), ("y", "h2", C_GREEN), ("h1", "x1", C_GREEN), ("h1", "x2", C_GREEN)], None, [], [], None,
                f"‖∇L‖ = {gn:.4f}"))
    return S


def render_part4_backward(save_gif=None, save_mp4=None, save_static=None, fps=24, dpi=80, step_frames=30):
    print("[Part 4] Backward pass: 15 beats")
    steps = backward_steps(P0, C0, G0)

    def setup(cv):
        for n, v in (("x1", f"{X_PT[0]:.2f}"), ("x2", f"{X_PT[1]:.2f}"), ("h1", f"{C0['h1']:.3f}"),
                     ("h2", f"{C0['h2']:.3f}"), ("y", f"{C0['yhat']:.3f}")):
            cv.text(n, "inn", v, NCOL[n], alpha=1.0)
        cv.text("L", "badge", rf"$L={C0['L']:.4f}$", C_ACC, alpha=1.0)

    dsig = lambda z: sigmoid(z) * (1 - sigmoid(z))
    # z1 < z2, so h1's label goes on the left of its dot and h2's on the right
    cfg = dict(fn=dsig, color=C_ACC, ylim=(-0.01, 0.30), label=r"$\sigma'(z)=\sigma(z)\,(1-\sigma(z))$",
               dots=dict(h1=(C0["z1"], G0["s1"], C_H1, rf"$\sigma'={G0['s1']:.4f}$", "right"),
                         h2=(C0["z2"], G0["s2"], C_H2, rf"$\sigma'={G0['s2']:.4f}$", "left")))
    fig, update, total = build_stage("The Backward Pass", steps, setup, cfg,
                                     "δ = upstream sensitivity  ×  local derivative   (the chain rule)", step_frames)
    save_outputs(fig, update, total, save_gif, save_mp4, save_static, step_frames * 6 + 24, fps, dpi, "Part 4")


# ==============================================================================
# PART 1: THE TWO MOONS
# ==============================================================================
def render_part1_manifold(save_gif=None, save_mp4=None, save_static=None, fps=24, dpi=80, frames=130):
    print("[Part 1] Two moons")
    X, y = make_two_moons(n_samples=500, noise=0.2, seed=42)
    y = np.asarray(y).flatten()
    n = len(X)
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    add_background(fig)
    header(fig, "The Data: Two Moons")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1], left=0.035, right=0.968, bottom=0.085, top=0.845, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax, r"feature $x_1$", r"feature $x_2$")
    ax.set_xlim(-1.8, 2.8); ax.set_ylim(-1.3, 1.8); ax.set_aspect("equal", adjustable="box")
    base = np.array([to_rgba(C_X1 if v == 1 else C_ACC) for v in y])
    delay = np.empty(n); delay[np.random.default_rng(1).permutation(n)] = np.linspace(0, 34, n)
    pts = ax.scatter(X[:, 0], X[:, 1], s=26, c=base, linewidths=0, zorder=3)
    ax.scatter([], [], c=C_X1, s=26, label="class 1  (target y = 1)")
    ax.scatter([], [], c=C_ACC, s=26, label="class 0  (target y = 0)")
    ax.legend(loc="lower left", frameon=False, labelcolor=C_TEXT, fontsize=10)
    ring = patches.Circle(tuple(X_PT), 0.1, fill=False, ec=C_TEXT, lw=1.2, zorder=8)
    ax.add_patch(ring)
    core, = ax.plot([X_PT[0]], [X_PT[1]], "o", color=C_TEXT, ms=9, mec=C_BG, mew=1.6, zorder=9)
    vx, = ax.plot([X_PT[0], X_PT[0]], [-1.3, -1.3], color=C_X1, lw=1.2, ls=(0, (4, 3)), zorder=7)
    vy, = ax.plot([-1.8, -1.8], [X_PT[1], X_PT[1]], color=C_X2, lw=1.2, ls=(0, (4, 3)), zorder=7)
    lx = ax.text(X_PT[0], -1.17, rf"$x_1={X_PT[0]:.2f}$", color=C_X1, fontsize=14, ha="center", path_effects=halo())
    ly = ax.text(-1.72, X_PT[1] + 0.09, rf"$x_2={X_PT[1]:.2f}$", color=C_X2, fontsize=14, ha="left", path_effects=halo())

    axc = fig.add_subplot(gs[0, 1]); panel(axc); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.text(0.5, 0.93, "THE TRAINING SAMPLE", color=C_SUB, fontsize=10.5, fontweight="bold", ha="center", va="top")
    cpos = [(0.5, 0.74), (0.5, 0.56), (0.5, 0.38)]
    cl = [axc.text(*cpos[0], rf"$x_1=\;{X_PT[0]:.2f}$", color=C_X1, fontsize=30, ha="center", va="center"),
          axc.text(*cpos[1], rf"$x_2=\;{X_PT[1]:.2f}$", color=C_X2, fontsize=30, ha="center", va="center"),
          axc.text(*cpos[2], rf"$y\;=\;{Y_TRUE:.0f}$", color=C_Y, fontsize=30, ha="center", va="center")]
    foot = axc.text(0.5, 0.17, "two numbers in, one number out:\nthe network must learn to say 1 here",
                    color=C_SUB, fontsize=11, ha="center", va="center", style="italic")

    def update(f):
        f = int(f)
        spot = smooth(ramp(f, 40, 62))
        a = np.array([ramp(f - d, 0, 7) for d in delay])
        c1 = base.copy(); c1[:, 3] = a * (1 - 0.82 * spot)
        pts.set_facecolors(c1)
        # a single ripple expanding from the sample, then settling to a thin ring
        rip = ramp(f, 40, 70)
        ring.set_radius(0.06 + 0.12 * smooth(rip))
        ring.set_alpha(spot)
        core.set_alpha(spot)
        pr = smooth(ramp(f, 62, 82))
        vx.set_data([X_PT[0], X_PT[0]], [X_PT[1] + (-1.3 - X_PT[1]) * pr, X_PT[1]])
        vy.set_data([X_PT[0] + (-1.8 - X_PT[0]) * pr, X_PT[0]], [X_PT[1], X_PT[1]])
        lx.set_alpha(ramp(f, 76, 88)); ly.set_alpha(ramp(f, 76, 88))
        for i, t in enumerate(cl):
            drift(t, *cpos[i], smooth(ramp(f, 84 + i * 10, 96 + i * 10)))
        foot.set_alpha(smooth(ramp(f, 112, 124)))
        return []

    save_outputs(fig, update, frames, save_gif, save_mp4, save_static, 125, fps, dpi, "Part 1")


# ==============================================================================
# PART 3: THE LOSS LANDSCAPE
# ==============================================================================
def render_part3_loss(save_gif=None, save_mp4=None, save_static=None, fps=24, dpi=80, frames=150):
    print("[Part 3] Loss landscape")
    yh, L, gy = C0["yhat"], C0["L"], G0["yhat"]
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    add_background(fig)
    header(fig, "The Loss Landscape")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1.35, 1], left=0.035, right=0.968,
                          bottom=0.085, top=0.845, wspace=0.07, hspace=0.14)
    ax = fig.add_subplot(gs[:, 0])
    style_axes(ax, r"prediction $\hat y$", r"loss $L$")
    ax.set_xlim(-0.05, 1.75); ax.set_ylim(-0.08, 1.15)
    yy = np.linspace(-0.05, 1.75, 400); LL = (yy - Y_TRUE) ** 2
    ax.axvline(Y_TRUE, color=C_GREEN, ls=(0, (2, 3)), lw=1.2)
    ax.text(Y_TRUE + 0.02, 1.07, r"target  $y=1$", color=C_GREEN, fontsize=12)
    cl, = ax.plot([], [], color=C_TEXT, lw=2.0, solid_capstyle="round")
    ymark, = ax.plot([yh], [0], "v", color=C_X1, ms=11, mec=C_BG, mew=1.2, zorder=6)
    ylab = ax.text(yh, -0.065, rf"$\hat y={yh:.4f}$", color=C_X1, fontsize=13, ha="center", va="top")
    res_line, = ax.plot([], [], color=C_ACC, lw=3.5, solid_capstyle="round", zorder=5)
    res_lab = ax.text((yh + 1) / 2, 0.035, "", color=C_ACC, fontsize=12, ha="center", va="bottom", path_effects=halo())
    drop, = ax.plot([], [], color=C_SUB, ls=(0, (3, 3)), lw=1.0)
    ball, = ax.plot([yh], [L], "o", color=C_ACC, ms=11, mec=C_BG, mew=1.6, zorder=7)
    tan, = ax.plot([], [], color=C_GREEN, lw=1.8, zorder=5)
    arrow = patches.FancyArrowPatch((yh, L + 0.03), (yh, L + 0.03), arrowstyle="-|>", mutation_scale=20,
                                    color=C_GREEN, lw=2.2, zorder=8)
    ax.add_patch(arrow)
    tlab = ax.text(yh + 0.04, L + 0.2, "", color=C_GREEN, fontsize=12, ha="left", path_effects=halo())
    Llab = ax.text(yh - 0.04, L + 0.015, "", color=C_ACC, fontsize=14, ha="right", va="bottom", path_effects=halo())

    axs = fig.add_subplot(gs[0, 1]); panel(axs)
    axs2 = axs.inset_axes([0.0, 0.0, 1, 1]); axs2.axis("off")
    axs2.set_xlim(-0.30, 1.15); axs2.set_ylim(-0.25, 1.10); axs2.set_aspect("equal")
    axs2.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, ec=C_HAIR, ls=(0, (3, 3)), lw=1.0))
    axs2.text(1.02, 1.0, "unit square", color=C_SUB, fontsize=9, va="top")
    s_ = abs(yh - Y_TRUE)
    sq = patches.Rectangle((0, 0), 0.0001, 0.0001, fc=to_rgba(C_ACC, 0.14), ec=C_ACC, lw=1.8)
    axs2.add_patch(sq)
    sqb = axs2.text(0, -0.1, "", color=C_ACC, fontsize=12, ha="left", va="center")
    sql = axs2.text(-0.03, 0, "", color=C_ACC, fontsize=12, ha="right", va="center")
    sqc = axs2.text(s_ / 2, s_ / 2, "", color=C_ACC, fontsize=17, ha="center", va="center")
    axs.set_title("squared error is literally an area", color=C_SUB, fontsize=10.5, pad=-14, style="italic")

    axb = fig.add_subplot(gs[1, 1]); panel(axb); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    lines = [
        (rf"$\hat y - y \;=\; {yh:.4f}-{Y_TRUE:.1f}\;=\;{yh - Y_TRUE:.4f}$", C_ACC, 76),
        (rf"$L \;=\;({yh - Y_TRUE:.4f})^2\;=\;{L:.4f}$", C_TEXT, 96),
        (rf"$\dfrac{{dL}}{{d\hat y}}=2(\hat y-y)=2\,({yh - Y_TRUE:.4f})={gy:.4f}$", C_GREEN, 116),
        (r"negative slope  ⇒  push $\hat y$ UP", C_TEXT, 130)]
    bpos = [(0.5, 0.82 - i * 0.22) for i in range(4)]
    bt = [axb.text(*bpos[i], s, color=c, fontsize=17 if i < 3 else 15, ha="center", va="center")
          for i, (s, c, _) in enumerate(lines)]

    def update(f):
        f = int(f)
        k = int(ramp(f, 0, 38) * len(yy))
        cl.set_data(yy[:k], LL[:k])
        a = ramp(f, 40, 52)
        ymark.set_alpha(a); ylab.set_alpha(a)
        pr = smooth(ramp(f, 54, 76))
        res_line.set_data([yh, yh + (1 - yh) * pr], [0.012, 0.012])
        res_lab.set_text(rf"residual $=\hat y-y={yh - 1:.4f}$"); res_lab.set_alpha(ramp(pr, .6, 1))
        d = smooth(ramp(f, 78, 94))
        drop.set_data([yh, yh], [0, L * d])
        ball.set_alpha(ramp(f, 86, 94))
        Llab.set_text(rf"$L={L:.4f}$"); Llab.set_alpha(ramp(f, 90, 100))
        sp = smooth(ramp(f, 78, 108))
        sq.set_width(max(1e-4, s_ * sp)); sq.set_height(max(1e-4, s_ * sp))
        sqb.set_text(rf"${s_:.4f}$"); sqb.set_alpha(ramp(sp, .8, 1)); sqb.set_position((s_ / 2 - 0.05, -0.1))
        sql.set_text(rf"${s_:.4f}$"); sql.set_alpha(ramp(sp, .8, 1)); sql.set_position((-0.03, s_ / 2))
        sqc.set_text(r"area $=L$"); sqc.set_alpha(ramp(sp, .85, 1))
        tp = smooth(ramp(f, 104, 124))
        span = 0.42 * tp
        tx = np.array([yh - span, yh + span]); tan.set_data(tx, L + gy * (tx - yh)); tan.set_alpha(1.0 if tp > 0 else 0.0)
        arrow.set_positions((yh, L + 0.05), (yh + 0.28 * tp, L + 0.05))
        arrow.set_alpha(tp)
        tlab.set_text(rf"slope $={gy:.4f}$"); tlab.set_alpha(ramp(tp, .6, 1))
        for i, (t, (_, _, f0)) in enumerate(zip(bt, lines)):
            drift(t, *bpos[i], smooth(ramp(f, f0 - 10, f0)))
        return []

    save_outputs(fig, update, frames, save_gif, save_mp4, save_static, 140, fps, dpi, "Part 3")


# ==============================================================================
# PART 5: GRADIENT DESCENT  --  every iteration, forward -> backward -> update
# ==============================================================================
def render_part5_descent(save_gif=None, save_mp4=None, save_static=None, fps=24, dpi=80,
                         lr=0.10, T=12, slow=76, fast=34, tail=54):
    print(f"[Part 5] Gradient descent: {T} full iterations")
    Ps, caches, grads = [dict(P0)], [forward(P0, X_PT)], []
    for _ in range(T):
        g = backward(Ps[-1], caches[-1]); grads.append(g)
        Pn = {k: Ps[-1][k] - lr * g[k] for k in P0}
        Ps.append(Pn); caches.append(forward(Pn, X_PT))
    its = [slow if t < 2 else fast for t in range(T)]
    starts = np.cumsum([0] + its)
    total = int(starts[-1]) + tail

    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    add_background(fig)
    header(fig, "Gradient Descent")
    phase_txt = fig.text(0.968, 0.905, "", fontsize=13, fontweight="bold", ha="right")
    itr_txt = fig.text(0.968, 0.945, "", fontsize=9.5, color=C_SUB, fontweight="bold", ha="right")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 0.88], height_ratios=[1.15, 0.85, 1.55], left=0.035,
                          right=0.968, bottom=0.085, top=0.845, wspace=0.05, hspace=0.20)
    ax_net = fig.add_subplot(gs[:, 0])
    cv = NetCanvas(ax_net, P0, nd=3, slots=6)
    inn = {n: cv.text(n, "inn", "", NCOL[n], alpha=1.0) for n in NODE}
    badge = cv.text("L", "badge", "", C_ACC, alpha=1.0)
    banner = ax_net.text(0.81, -0.05, "", color=C_GREEN, fontsize=13, fontweight="bold", ha="center", va="center")

    # --- loss parabola in yhat space
    ax_p = fig.add_subplot(gs[0, 1]); style_axes(ax_p, r"prediction $\hat y$", r"$L$")
    yy = np.linspace(0.3, 1.25, 200)
    ax_p.plot(yy, (yy - 1) ** 2, color=C_TEXT, lw=1.8)
    ax_p.axvline(1.0, color=C_GREEN, ls=(0, (2, 3)), lw=1.1)
    ax_p.set_xlim(0.3, 1.25); ax_p.set_ylim(-0.02, 0.52)
    trail, = ax_p.plot([], [], "o", color=C_SUB, ms=5, alpha=0.6)
    ball, = ax_p.plot([], [], "o", color=C_ACC, ms=11, mec=C_BG, mew=1.5, zorder=6)
    tan, = ax_p.plot([], [], color=C_GREEN, lw=1.6)
    blab = ax_p.text(0.5, 0.88, "", transform=ax_p.transAxes, color=C_TEXT, fontsize=10.5, family=MONO, va="top")

    # --- loss vs iteration
    ax_i = fig.add_subplot(gs[1, 1]); style_axes(ax_i, "iteration", "")
    ax_i.set_title("loss per iteration (log scale)", color=C_SUB, fontsize=9, pad=3)
    Ls = np.array([c["L"] for c in caches])
    ax_i.set_yscale("log"); ax_i.set_xlim(-0.3, T + 0.3); ax_i.set_ylim(max(Ls.min() / 4, 1e-9), Ls.max() * 3)
    iline, = ax_i.plot([], [], "-o", color=C_TEXT, lw=1.6, ms=5, mec=C_BG, mew=0.8)
    ax_i.grid(True, which="major", alpha=0.9)
    ax_i.grid(False, which="minor")
    ax_i.yaxis.set_tick_params(which="minor", left=False)

    # --- parameter ledger
    ax_t = fig.add_subplot(gs[2, 1]); panel(ax_t); ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
    thead = ax_t.text(0.03, 0.955, "", color=C_SUB, fontsize=9, family=MONO, fontweight="bold", va="top")
    trows = [ax_t.text(0.03, 0.83 - i * 0.092, "", fontsize=9.4, family=MONO, va="top") for i in range(9)]

    ax_pb = fig.add_axes([0.035, 0.040, 0.933, 0.006]); ax_pb.axis("off"); ax_pb.set_xlim(0, T); ax_pb.set_ylim(0, 1)
    fills = []
    for i in range(T):
        ax_pb.add_patch(patches.Rectangle((i + 0.05, 0), 0.9, 1, fc=C_LINE, ec="none"))
        f = patches.Rectangle((i + 0.05, 0), 0.0001, 1, fc=C_GREEN, ec="none")
        ax_pb.add_patch(f); fills.append(f)

    order = ["w11", "w21", "w12", "w22", "w1", "w2"]
    fcol = dict(w11=C_X1, w21=C_X2, w12=C_X1, w22=C_X2, w1=C_H1, w2=C_H2)

    def update(frame):
        frame = int(frame)
        final = frame >= starts[-1]
        if final:
            t, u = T - 1, 1.0
        else:
            t = int(np.searchsorted(starts, frame, side="right") - 1); u = (frame - starts[t]) / its[t]
        fw, bw, up = ramp(u, 0, .3), ramp(u, .3, .6), ramp(u, .6, .95)
        phase = 0 if u < .3 else (1 if u < .6 else 2)
        if final:
            phase = 2
        wprog = smooth(up)
        Pv = {k: Ps[t][k] + (Ps[t + 1][k] - Ps[t][k]) * wprog for k in P0}
        dc = caches[T] if final else (caches[t] if u >= .1 else caches[max(t - 1, 0)])

        cv.reset(); cv.set_weights(Pv, nd=3)
        for n, v in (("x1", X_PT[0]), ("x2", X_PT[1]), ("h1", dc["h1"]), ("h2", dc["h2"]), ("y", dc["yhat"])):
            inn[n].set_text(f"{v:.3f}" if n not in ("x1", "x2") else f"{v:.2f}")
        badge.set_text(rf"$L={fmtL(dc['L'])}$")

        # travelling dots
        if not final:
            if 0 < fw < 1:
                for i, k in enumerate(order):
                    a, b = EDGE[k]
                    pr = ramp(fw, 0, .5) if i < 4 else ramp(fw, .5, 1)
                    cv.pulse(i, a, b, pr, fcol[k])
                    if 0 < pr < 1:
                        cv.highlight(k, 1.0, C_X1)
            if 0 < bw < 1:
                for i, k in enumerate(order):
                    a, b = EDGE[k]
                    pr = ramp(bw, 0, .5) if i >= 4 else ramp(bw, .5, 1)
                    cv.pulse(i, b, a, pr, C_ACC)
                    if 0 < pr < 1:
                        cv.highlight(k, 1.0, C_ACC)
            if 0 < up < 1:
                s_ = np.sin(np.pi * up)
                for k in order:
                    cv.highlight(k, s_, C_GREEN)
                for n in ("h1", "h2", "y"):
                    cv.flash(n, 0.5 * s_)
            if 0 < fw < 1:
                for n in ("h1", "h2", "y"):
                    cv.flash(n, 0.5 * np.sin(np.pi * ramp(fw, .4, 1)))

        # phase labels
        names = [("1   FORWARD", C_X1), ("2   BACKWARD", C_ACC), (r"3   UPDATE   θ ← θ − α∇L", C_GREEN)]
        if final:
            phase_txt.set_text("CONVERGED"); phase_txt.set_color(C_GREEN)
            itr_txt.set_text(f"{T} ITERATIONS   ·   α = {lr}")
        else:
            phase_txt.set_text(names[phase][0]); phase_txt.set_color(names[phase][1])
            itr_txt.set_text(f"ITERATION {t + 1:02d} / {T:02d}   ·   α = {lr}")
        banner.set_text(f"ŷ = {caches[T]['yhat']:.4f}   (target 1)    L = {fmtL(caches[T]['L'])}" if final else "")

        # parabola
        y0, y1 = caches[t]["yhat"], caches[t + 1]["yhat"]
        yv = y0 + (y1 - y0) * wprog
        if final:
            yv = caches[T]["yhat"]
        Lv = (yv - 1) ** 2
        ball.set_data([yv], [Lv])
        shown = [caches[i]["yhat"] for i in range(T + 1 if final else t + 1)]
        trail.set_data(shown, [(v - 1) ** 2 for v in shown])
        if bw > 0 or phase >= 1:
            sl = 2 * (y0 - 1) if not final else 0
            xs = np.array([yv - .18, yv + .18]); tan.set_data(xs, Lv + sl * (xs - yv)); tan.set_alpha(1.0 if not final else 0.0)
        else:
            tan.set_alpha(0)
        blab.set_text(f"ŷ = {yv:.4f}\nL = {fmtL(Lv)}")

        # loss curve
        nshow = T + 1 if final else (t + 1 if u >= .15 else t)
        nshow = max(nshow, 1)
        iline.set_data(np.arange(nshow), Ls[:nshow])

        # ledger
        thead.set_text({0: "PARAMETERS θ  (current values)",
                        1: "GRADIENTS  ∂L/∂θ  (from the backward pass)",
                        2: f"UPDATE   θ_new = θ − {lr:.2f} × ∂L/∂θ"}[phase])
        g = grads[t]
        for i, k in enumerate(PARAM_ORDER):
            old = Ps[t][k]
            if phase == 0:
                s = f"{k:>4} = {old:+.4f}"
            elif phase == 1:
                s = f"{k:>4} = {old:+.4f}     ∂L/∂{k:<3} = {g[k]:+.4f}" if bw * 9 > i else f"{k:>4} = {old:+.4f}"
            else:
                cur = Ps[t][k] + (Ps[t + 1][k] - Ps[t][k]) * wprog
                s = f"{k:>4}: {old:+.4f} − {lr:.2f}×({g[k]:+.4f}) = {cur:+.4f}"
            trows[i].set_text(s); trows[i].set_color(to_rgba(PARAM_COL[k], 0.97))

        for i, f in enumerate(fills):
            frac = 1.0 if (i < t or final) else (u if i == t else 0.0)
            f.set_width(max(0.0001, 0.9 * frac))
        return []

    save_outputs(fig, update, total, save_gif, save_mp4, save_static, int(starts[1]) + 50, fps, dpi, "Part 5")


# ==============================================================================
# MASTER RUNNER
# ==============================================================================
def run_all(save_dir, parts, still, mp4, fps, dpi, no_gif=False):
    os.makedirs(save_dir, exist_ok=True)
    P = lambda name, ext: os.path.join(save_dir, f"3b1b_{name}.{ext}")
    plan = [
        (1, "1_two_moons", render_part1_manifold),
        (2, "2_forward_pass", render_part2_forward),
        (3, "3_loss_landscape", render_part3_loss),
        (4, "4_backward_pass", render_part4_backward),
        (5, "5_gradient_descent", render_part5_descent),
    ]
    print("=" * 78 + "\n  INSIDE THE BACKWARD PASS  -  animation suite\n" + "=" * 78)
    for num, name, fn in plan:
        if num in parts:
            fn(save_gif=None if (still or no_gif) else P(name, "gif"),
               save_mp4=P(name, "mp4") if (mp4 and not still) else None,
               save_static=P(name, "png"), fps=fps, dpi=dpi)
    print("=" * 78 + f"\n  DONE -> {os.path.abspath(save_dir)}\n" + "=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Minimalist backprop animation suite")
    ap.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--save-dir", default="exports")
    ap.add_argument("--still", action="store_true", help="PNG stills only (fast preview)")
    ap.add_argument("--mp4", action="store_true", help="also export MP4 (needs ffmpeg)")
    ap.add_argument("--no-gif", action="store_true", help="skip GIF export (faster)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=80, help="animation dpi (lower = smaller GIF)")
    a = ap.parse_args()
    run_all(a.save_dir, a.parts, a.still, a.mp4, a.fps, a.dpi, a.no_gif)