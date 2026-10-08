"""Build the four main-text figures for "Valid but Vacuous".

    python make_figures.py      # writes ../paper/figs/fig1_overview.pdf ... fig4_transport.pdf

All numbers come from paper_data.py and data/*.json. Figures are drawn at their
printed size (6.5 in wide), so 7 pt here is 7 pt on the page.
"""
import json
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

import paper_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ palette
BLUE = "#2a78d6"      # answered and kept the promise / certified / same benchmark
ORANGE = "#eb6834"    # no guarantee / new benchmark
AQUA = "#1baf7a"      # no guarantee, all data
RED = "#d03b3b"       # broke the promise
RED_WASH = "#f5d8d8"  # most a valid certificate allows
SILENT = "#dddcd6"    # answered nothing
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RAMP = {0.1: "#86b6ef", 0.2: "#3987e5", 0.3: "#1c5cab", 0.4: "#0d366b"}
BOLD = 600

plt.rcParams.update({
    "font.family": "FreeSans",
    "font.size": 7.4,
    "axes.labelsize": 7.4,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "ytick.major.pad": 2.5,
    "legend.fontsize": 6.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})

BUDGETS = ["5%", "10%", "20%", "30%", "40%"]
X = np.arange(5)
VAL = dict(fontsize=6.8, color=INK2)


def r1(v, nd=1):
    """Round half up (10.85 -> 10.9), as in the text."""
    return Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def pct(v, nd=1):
    return f"{r1(100 * v, nd)}%"


def yaxis(ax, lo, hi, step, fmt=lambda t: f"{t * 100:.0f}%"):
    ax.set_ylim(lo, hi)
    ticks = np.arange(0, hi + 1e-9, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt(t) for t in ticks])
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def budget_axis(ax, xs=X, labels=BUDGETS, pad=0.6):
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("error budget", labelpad=2)
    ax.set_xlim(xs[0] - pad, xs[-1] + pad)


HEAD = []


def header(ax, letter, title):
    HEAD.append((ax, letter, title))


def apply_headers(fig):
    """Letter + title, left-aligned with the y tick labels."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, letter, title in HEAD:
        if ax.figure is not fig:
            continue
        boxes = [t.get_window_extent(rend) for t in ax.get_yticklabels() if t.get_text()]
        x0 = min(b.x0 for b in boxes) if boxes else ax.get_window_extent().x0
        xa = ax.transAxes.inverted().transform((x0, 0))[0]
        kw = dict(xycoords="axes fraction", textcoords="offset points", ha="left",
                  va="bottom", annotation_clip=False)
        ax.annotate(letter, xy=(xa, 1), xytext=(0, 7), fontsize=9.5, fontweight=BOLD,
                    color=INK, **kw)
        ax.annotate(title, xy=(xa, 1), xytext=(10.5, 7), fontsize=8.3, fontweight=BOLD,
                    color=INK, **kw)


def rbar(ax, x, h, w, color, r_pt=1.5, z=3):
    """Bar with a rounded top, square at the baseline (data coordinates)."""
    if h <= 0:
        return
    fig = ax.figure
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px = r_pt * fig.dpi / 72.0
    rx = min(px * (x1 - x0) / bbox.width, w / 2)
    ry = min(px * (y1 - y0) / bbox.height, h)
    L, R = x - w / 2, x + w / 2
    k = 0.5523
    verts = [(L, 0), (L, h - ry), (L, h - ry + k * ry), (L + rx - k * rx, h), (L + rx, h),
             (R - rx, h), (R - rx + k * rx, h), (R, h - ry + k * ry), (R, h - ry),
             (R, 0), (L, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=z))


def dashed(ax, y, text=None, x=None, ha="left", va="bottom"):
    ax.axhline(y, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    if text:
        ax.text(x, y, text, color=INK, fontsize=6.7, ha=ha, va=va, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def chip(fig, x, y, color, w=0.0115, h=0.04):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.002",
                                      transform=fig.transFigure, facecolor=color,
                                      edgecolor="none"))


def save(fig, name):
    apply_headers(fig)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig)


# =================================================================== Figure 1
# Unchanged from the version the author approved.
def fig1():
    fired, broke, n = D.WITHIN_FIRED[2], D.WITHIN_BROKE[2], D.N_WITHIN   # budget 20%
    raw = np.array([broke, fired - broke, n - fired]) / n * 1000
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: 1000 - counts.sum()]:
        counts[i] += 1
    n_red, n_blue, n_grey = (int(c) for c in counts)                       # 14, 94, 892

    fig = plt.figure(figsize=(6.5, 2.55))
    W_FIG, H_FIG = fig.get_size_inches()
    cols, rows = 40, 25
    s, gap = 1.0, 0.26
    ax_w = 0.415
    ax_h = ax_w * W_FIG / H_FIG * (rows * (s + gap)) / (cols * (s + gap))
    ax = fig.add_axes([0.0, 0.205, ax_w, ax_h])
    colors = [RED] * n_red + [BLUE] * n_blue + [SILENT] * n_grey
    for i, c in enumerate(colors):
        r, q = divmod(i, cols)
        ax.add_patch(FancyBboxPatch((q * (s + gap), -r * (s + gap)), s, s,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    facecolor=c, edgecolor="none"))
    ax.set_xlim(-gap / 2, cols * (s + gap) - gap / 2)
    ax.set_ylim(-(rows - 1) * (s + gap) - gap / 2, s + gap / 2)
    ax.axis("off")
    top = 0.205 + ax_h
    fig.text(0.002, top + 0.06, "40,000 simulated deployments, one square per 40",
             fontsize=8.2, fontweight=BOLD, color=INK, va="bottom")
    fig.text(0.002, top + 0.012, "error budget \u03b1 = 20%: at most 1 in 5 answers may be wrong",
             fontsize=6.7, color=MUTED, va="bottom")

    leg = [(SILENT, f"answered nothing ({n_grey})", 0.002, 0.105),
           (BLUE, f"answered and kept the promise ({n_blue})", 0.19, 0.105),
           (RED, f"answered and broke the promise ({n_red})", 0.19, 0.035)]
    for c, t, x, y in leg:
        chip(fig, x, y, c)
        fig.text(x + 0.017, y + 0.02, t, fontsize=6.8, color=INK2, va="center")

    x0 = 0.535
    fig.add_artist(Line2D([x0 - 0.025] * 2, [0.04, 0.96], color=GRID, lw=0.8,
                          transform=fig.transFigure))
    blocks = [
        (0.965, "WHAT GETS REPORTED", pct(broke / n),
         ["of all deployments broke", "the promise."],
         "\u2713  Valid: under the 5% target.", [RED]),
        (0.655, "HOW OFTEN THE MODEL MAY SPEAK", pct(fired / n),
         ["of deployments answered", "any question at all."],
         f"The other {pct(1 - fired / n)} answered nothing.", [BLUE, RED]),
        (0.345, "WHAT A HOSPITAL GETTING ANSWERS SEES", pct(broke / fired),
         ["of the deployments that", "answered broke the promise."],
         f"\u2717  {r1(broke / fired / D.DELTA)}\u00d7 the 5% target.", [RED]),
    ]
    for y, head, big, lines, verdict, marks in blocks:
        for j, m in enumerate(marks):
            chip(fig, x0 + j * 0.016, y - 0.035, m, w=0.0105, h=0.033)
        fig.text(x0 + 0.016 * len(marks) + 0.004, y - 0.0185, head, fontsize=6.3,
                 color=MUTED, va="center", fontweight=BOLD)
        fig.text(x0, y - 0.065, big, fontsize=20, color=INK, va="top")
        tx = x0 + 0.165
        fig.text(tx, y - 0.07, "\n".join(lines), fontsize=7.2, color=INK2, va="top",
                 linespacing=1.2)
        fig.text(tx, y - 0.205, verdict, fontsize=7.2, color=INK, va="top")
    save(fig, "fig1_overview")


# =================================================================== Figure 2
def fig2():
    n = D.N_WITHIN
    fired = np.array(D.WITHIN_FIRED) / n
    broke = np.array(D.WITHIN_BROKE) / n
    lo = np.array([c[0] for c in D.WITHIN_BROKE_CI])
    hi = np.array([c[1] for c in D.WITHIN_BROKE_CI])
    cond = D.within_conditional()
    ceil = D.ceiling()

    fig = plt.figure(figsize=(6.5, 2.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.12, 1.3], wspace=0.3,
                          left=0.04, right=0.995, bottom=0.19, top=0.85)

    # (a)
    ax = fig.add_subplot(gs[0])
    yaxis(ax, 0, 0.06, 0.02)
    budget_axis(ax)
    ax.vlines(X, lo, hi, color=RED, alpha=0.3, linewidth=3.0, zorder=3, capstyle="round")
    ax.plot(X, broke, color=RED, linewidth=1.4, zorder=4, solid_capstyle="round")
    ax.scatter(X, broke, s=20, color=RED, edgecolor="white", linewidth=1.0, zorder=5)
    dashed(ax, D.DELTA, "5% limit", x=-0.6)
    ax.text(4, hi[4] + 0.002, pct(broke[4]), ha="center", va="bottom", **VAL)
    header(ax, "a", "Broke the promise")

    # (b)
    ax = fig.add_subplot(gs[1])
    yaxis(ax, 0, 0.5, 0.1)
    budget_axis(ax)
    for i, v in enumerate(fired):
        rbar(ax, i, v, 0.58, BLUE)
        ax.text(i, v + 0.012, "0%" if v == 0 else pct(v), ha="center", va="bottom", **VAL)
    header(ax, "b", "Answered at all")

    # (c)
    ax = fig.add_subplot(gs[2])
    yaxis(ax, 0, 0.5, 0.1)
    budget_axis(ax)
    ax.set_xlim(-0.6, 4.85)
    ax.text(0, 0.1, "silent", ha="center", va="bottom", fontsize=6.6, color=MUTED)
    for i in range(1, 5):
        rbar(ax, i, min(ceil[i], 0.5), 0.58, RED_WASH, z=2)
        rbar(ax, i, cond[i], 0.58, RED, z=3)
        ax.text(i, cond[i] + 0.01, pct(cond[i]), ha="center", va="bottom", zorder=6, **VAL)
        if ceil[i] > 0.5:
            ax.annotate("", xy=(i, 0.515), xytext=(i, 0.455),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6),
                        annotation_clip=False, zorder=7)
        else:
            lab = pct(ceil[i], 0) + (" allowed" if i == 2 else "")
            ax.text(i + 0.33, ceil[i], lab, ha="left", va="center", fontsize=6.5, color=MUTED)
    dashed(ax, D.DELTA)
    header(ax, "c", "Broke it, when answering")

    # (d)
    ax = fig.add_subplot(gs[3])
    yaxis(ax, 0, 0.7, 0.1)
    budget_axis(ax)
    w = 0.25
    series = [(D.COV_CERTIFIED, BLUE, "certified"),
              (D.COV_PLUGIN, ORANGE, "no guarantee"),
              (D.COV_PLUGIN_FULL, AQUA, "no guarantee, all data")]
    for k, (vals, col, _) in enumerate(series):
        for i, v in enumerate(vals):
            rbar(ax, i + (k - 1) * (w + 0.03), v, w, col, r_pt=1.2)
    for i, v in enumerate(D.COV_ORACLE):
        ax.plot([i - 0.45, i + 0.45], [v, v], color=INK, linewidth=1.1, zorder=6,
                solid_capstyle="butt")
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none", label=l)
               for _, c, l in series]
    handles.append(Line2D([], [], color=INK, lw=1.1, label="best possible"))
    ax.legend(handles=handles, loc="upper left", handlelength=0.9, handleheight=0.9,
              borderaxespad=0.0, labelspacing=0.3, handletextpad=0.5)
    header(ax, "d", "Questions answered")
    save(fig, "fig2_within")


# =================================================================== Figure 3
def fig3():
    xs = np.arange(4)
    labs = ["10%", "20%", "30%", "40%"]
    plus = lambda t: "0" if t == 0 else f"+{t * 100:.0f}"
    fig = plt.figure(figsize=(6.5, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.26,
                          left=0.04, right=0.975, bottom=0.19, top=0.85)

    ax = fig.add_subplot(gs[0])
    yaxis(ax, 0, 0.25, 0.05, fmt=plus)
    budget_axis(ax, xs, labs)
    g = np.array(D.SURV_GAP)
    ax.vlines(xs, D.SURV_GAP_LO, D.SURV_GAP_HI, color=BLUE, alpha=0.3, linewidth=3.0,
              capstyle="round", zorder=3)
    ax.plot(xs, g, color=BLUE, lw=1.4, zorder=4)
    ax.scatter(xs, g, s=20, color=BLUE, edgecolor="white", linewidth=1.0, zorder=5)
    for i, v in enumerate(g):
        ax.text(i, D.SURV_GAP_HI[i] + 0.006, f"+{r1(100 * v)}", ha="center", va="bottom", **VAL)
    header(ax, "a", "Extra errors on new questions")

    ax = fig.add_subplot(gs[1])
    yaxis(ax, 0, 1.1, 0.25)
    budget_axis(ax, xs, labs)
    ax.set_xlim(-0.6, 4.35)
    for i, v in enumerate(D.SURV_SHARE_WORSE):
        rbar(ax, i, v, 0.58, BLUE)
        ax.text(i, v + 0.025, f"{100 * v:.0f}%", ha="center", va="bottom", **VAL)
    dashed(ax, 0.5, "chance", x=4.35, ha="right")
    header(ax, "b", "Did worse on new questions")

    ax = fig.add_subplot(gs[2])
    pts = json.load(open(os.path.join(HERE, "data", "fig3c.json")))
    yaxis(ax, -0.04, 0.7, 0.1, fmt=plus)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for a in [0.1, 0.2, 0.3, 0.4]:
        P = [p for p in pts if p["alpha"] == a]
        ax.scatter([p["fire"] for p in P], [p["gap"] for p in P], s=14, color=RAMP[a],
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"{int(round(a * 100))}% budget")
    ax.set_xlim(-0.03, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("how often the model spoke", labelpad=2)
    ax.legend(loc="upper right", handletextpad=0.1, borderaxespad=0.0, labelspacing=0.25)
    header(ax, "c", "Rare passes are the luckiest")
    save(fig, "fig3_luck")


# =================================================================== Figure 4
def fig4():
    n = D.N_TRANSPORT
    new_all = np.array(D.TR_BROKE_NEW) / n
    same_all = np.array(D.TR_BROKE_SAME) / n
    new_c = np.array([b / f if f else np.nan for b, f in zip(D.TR_BROKE_NEW, D.TR_FIRED)])
    same_c = np.array([b / f if f else np.nan
                       for b, f in zip(D.TR_BROKE_SAME, D.TR_FIRED_CONTROL)])

    fig = plt.figure(figsize=(6.5, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.26,
                          left=0.04, right=0.99, bottom=0.21, top=0.85)

    ax = fig.add_subplot(gs[0])
    yaxis(ax, 0, 0.2, 0.05)
    budget_axis(ax)
    w = 0.32
    for i in range(5):
        rbar(ax, i - w / 2 - 0.02, same_all[i], w, BLUE, r_pt=1.2)
        rbar(ax, i + w / 2 + 0.02, new_all[i], w, ORANGE, r_pt=1.2)
    ax.text(4 + w / 2 + 0.02, new_all[4] + 0.004, pct(new_all[4]), ha="center",
            va="bottom", **VAL)
    dashed(ax, D.DELTA, "5% limit", x=-0.6)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE, ec="none", label="same benchmark"),
                       Rectangle((0, 0), 1, 1, fc=ORANGE, ec="none", label="new benchmark")],
              loc="upper left", handlelength=0.9, handleheight=0.9, borderaxespad=0.0,
              labelspacing=0.3)
    header(ax, "a", "Broke the promise")

    ax = fig.add_subplot(gs[1])
    yaxis(ax, 0, 0.8, 0.2)
    xs = np.arange(1, 5)
    budget_axis(ax, xs, BUDGETS[1:])
    ax.set_xlim(0.55, 4.55)
    ax.fill_between(xs, same_c[1:], new_c[1:], color=ORANGE, alpha=0.12, lw=0, zorder=2)
    for vals, col in [(new_c, ORANGE), (same_c, BLUE)]:
        ax.plot(xs, vals[1:], color=col, lw=1.6, zorder=4)
        ax.scatter(xs, vals[1:], s=20, color=col, edgecolor="white", lw=1.0, zorder=5)
    ax.text(4, new_c[4] + 0.035, pct(new_c[4], 0), va="bottom", ha="center", **VAL)
    ax.text(4, same_c[4] + 0.035, pct(same_c[4], 0), va="bottom", ha="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.3), **VAL)
    dashed(ax, D.DELTA)
    header(ax, "b", "Broke it, when answering")

    ax = fig.add_subplot(gs[2])
    pts = json.load(open(os.path.join(HERE, "data", "fig4d.json")))
    auc = np.array([p["auc"] for p in pts])
    sh = np.array([p["shift"] for p in pts])
    yaxis(ax, -0.12, 1.0, 0.25, fmt=lambda t: "0" if t == 0 else f"+{t * 100:.0f}")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.scatter(auc, sh, s=13, color=ORANGE, edgecolor="white", lw=0.6, zorder=3)
    b1, b0 = np.polyfit(auc, sh, 1)
    xx = np.array([0.49, 0.92])
    ax.plot(xx, b0 + b1 * xx, color=INK, lw=1.1, zorder=4)
    r2 = np.corrcoef(auc, sh)[0, 1] ** 2
    ax.text(0.925, 0.95, f"r² = {r2:.2f}", ha="right", va="top", fontsize=7.2, color=INK)
    ax.set_xlim(0.48, 0.93)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("how different the two benchmarks are", labelpad=2)
    header(ax, "c", "Extra failures from switching")
    save(fig, "fig4_transport")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("wrote figures to", os.path.abspath(OUT))
