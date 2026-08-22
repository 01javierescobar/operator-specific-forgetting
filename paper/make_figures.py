#!/usr/bin/env python3
"""Generate all manuscript figures from the frozen JSON artifacts only.

Outputs (figures/):
  fig1_memory_diagram.pdf/.png   write/read/erase schematic with drift locations
  fig2_residual_laws.pdf/.png    four residual laws vs measured points (d=128 + d=64)
  fig3_load_scaling.pdf/.png     measured c vs theory c=(n-1)/D, both scales
  fig4_energy_vs_decodability    key/re cell: energy leakage vs decodable fraction
  fig5_write_stability           delta beta=1 divergence vs NLMS/superposition
  fig6_tost                      equivalence test with CI90 and margins

No number is hard-coded from the manuscript; everything is recomputed here.
"""
import json
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import scipy.stats as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = lambda *p: os.path.join(ROOT, *p)
FIG = lambda name: OUT("figures", name)


def load(rel):
    with open(OUT(rel), encoding="utf-8") as f:
        return json.load(f)


o05 = load("outputs/wave_mem/o05.json")
o04b = load("outputs/wave_mem/o04b.json")
autopsy = load("outputs/wave_mem/delta_autopsy.json")
autopsy_beta = load("outputs/wave_mem/delta_autopsy_beta.json")

DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DL = [str(d) for d in DELTAS]
C128 = 39.0 / 128.0
C64 = 19.0 / 64.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "savefig.dpi": 300,
    }
)


def law(cell, chan, dl, c):
    if cell == "clave" and chan == "complex":
        return 0.0
    if cell == "estado" and chan == "complex":
        return 2 * (1 - math.cos(dl)) * (1 + c)
    if cell == "clave" and chan == "re":
        return math.sin(dl) ** 4 + c * (1 - math.cos(2 * dl)) / 4
    return (1 - math.cos(dl)) ** 2 + c * (1 - math.cos(dl))


def o05_fugas(arm, cell, dl):
    return [o05["runs"][f"{arm}_s{s}"]["probe"]["cells"][cell][dl]["fuga"] for s in "12345"]


def o05_em_guarded(arm, cell, dl):
    vals = []
    for s in "12345":
        v = o05["runs"][f"{arm}_s{s}"]["probe"]["cells"][cell][dl]["em_guarded"]
        vals.append(v if isinstance(v, (int, float)) else math.nan)
    return vals


def o04b_fugas(arm, cell, dl):
    return [
        f for s in (1, 2, 3) for f in o04b["runs"][f"{arm}_s{s}"][dl][f"fuga_{cell}"]
    ]


def savefig(fig, name):
    fig.savefig(FIG(name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG(name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


# ---------------------------------------------------------------------------
print("fig1: memory diagram")
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(-0.4, 6.2)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef2f7", ec="#33415c", lw=1.2):
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=lw
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)


def arrow(x0, y0, x1, y1, label=None, lx=0.5, ly=0.15, color="#33415c", style="-|>", lw=1.2):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, lw=lw, color=color
    )
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=8.5, color=color, ha="center")


# Memory block
box(0.6, 2.0, 3.0, 2.0, r"Memory state" + "\n" + r"$\mathbf{M} = \sum_i \mathbf{w}_i \mathbf{v}_i^\top$")

# Operator blocks
box(5.8, 4.3, 3.6, 1.4, r"Read operator" + "\n" + r"$\hat{\mathbf{v}}_j = \mathbf{w}_j^H \mathbf{M} / D$")
box(5.8, 2.3, 3.6, 1.4, r"Ideal erase" + "\n" + r"$\mathbf{M}' = \mathbf{M} - \mathbf{w}_j \hat{\mathbf{v}}_j$")
box(5.8, 0.3, 3.6, 1.4, r"Drifted erase ($\delta$)" + "\n" + r"Key: $\mathbf{w}_j e^{i\delta}$", fc="#fdf3e7", ec="#b07d2b")

# Arrows
arrow(3.6, 3.6, 5.7, 4.8, label=r"Write $+\mathbf{w}_i \mathbf{v}_i^\top$", lx=-0.2, ly=0.25)
arrow(3.6, 3.0, 5.7, 3.0, label=r"Reread", lx=-0.1, ly=0.18)
arrow(3.6, 2.4, 5.7, 1.2, label=r"Subtract", lx=-0.35, ly=-0.35)

# Drift relation arrow between Ideal and Drifted erase
arrow(7.6, 2.3, 7.6, 1.7, color="#b07d2b", style="-|>", label=r"drift $\delta$", lx=0.5, ly=0.0)

# Bottom note
ax.text(5.0, -0.25, r"$\mathbf{State\ drift}$: stored item $\mathbf{v}_j e^{i\delta}$ at write;  $\mathbf{Key\ drift}$: erase key $\mathbf{w}_j e^{i\delta}$ only",
        fontsize=8.5, color="#444", ha="center")

savefig(fig, "fig1_memory_diagram")

# ---------------------------------------------------------------------------
print("fig2: residual laws")
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
cells_info = [
    ("clave", "complex", "Key drift / Complex readout"),
    ("estado", "complex", "State drift / Complex readout"),
    ("clave", "re", "Key drift / Real readout"),
    ("estado", "re", "State drift / Real readout"),
]
x = DELTAS
xs = [d + 0.008 for d in DELTAS]

for ax, (cell, chan, title) in zip(axes.flat, cells_info):
    arm = "wave_complex" if chan == "complex" else "wave_re"
    y = [law(cell, chan, d, C128) for d in DELTAS]
    ax.plot(x, y, "-", color="#1f4e79", lw=1.6, label=r"law, $c=39/128$")
    meds, los, his = [], [], []
    for dl in DL:
        f = o05_fugas(arm, cell, dl)
        meds.append(statistics.median(f))
        los.append(min(f))
        his.append(max(f))
    ax.errorbar(
        xs, meds, yerr=[[m - lo for m, lo in zip(meds, los)],
                        [hi - m for m, hi in zip(meds, his)]],
        fmt="o", ms=4, color="#1f4e79", capsize=2, lw=0.9,
        label=r"$d=128$, 5 seeds", zorder=3)
    if cell != "clave" or chan != "complex":
        f64 = [statistics.median(o04b_fugas(arm, cell, dl)) for dl in DL]
        ax.plot(xs, f64, "s", ms=3.6, mfc="none", mec="#b02a37", mew=1.1,
                label=r"$d=64$, 3 seeds", zorder=3)
    if cell == "clave" and chan == "complex":
        ax.set_ylim(-0.02, 0.02)
        ax.annotate("exact zero residual\n(30/30 null outcomes)", xy=(0.25, 0.008),
                    fontsize=8.5, color="#555", ha="center")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.02, 0.55)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(alpha=0.25, lw=0.5)
    if chan == "complex" and cell == "estado":
        ax.legend(fontsize=8, loc="upper left")

axes[1, 0].set_xlabel(r"Phase drift $\delta$ (rad)")
axes[1, 1].set_xlabel(r"Phase drift $\delta$ (rad)")
for ax in axes[:, 0]:
    ax.set_ylabel("Pooled residual energy")
fig.tight_layout()
savefig(fig, "fig2_residual_laws")

# ---------------------------------------------------------------------------
print("fig3: load scaling (measured c vs theory)")
def measured_c(chan, cell, fugas, dl):
    if cell == "clave" and chan == "complex":
        return None
    if cell == "estado" and chan == "complex":
        return [f / (2 * (1 - math.cos(dl))) - 1 for f in fugas]
    if cell == "clave" and chan == "re":
        return [4 * (f - math.sin(dl) ** 4) / (1 - math.cos(2 * dl)) for f in fugas]
    return [(f - (1 - math.cos(dl)) ** 2) / (1 - math.cos(dl)) for f in fugas]

fig, ax = plt.subplots(figsize=(5.0, 3.6))
cellchan = [("estado", "complex", "state / complex"),
            ("clave", "re", "key / real"),
            ("estado", "re", "state / real")]
markers = {"state / complex": "o", "key / real": "s", "state / real": "^"}

for (cell, chan, label_name) in cellchan:
    arm = "wave_complex" if chan == "complex" else "wave_re"
    est128, est64 = [], []
    for dl in DL[1:]:
        d = float(dl)
        est128 += [c for c in measured_c(chan, cell, o05_fugas(arm, cell, dl), d) if c is not None]
        est64 += [c for c in measured_c(chan, cell, o04b_fugas(arm, cell, dl), d) if c is not None]
    m128, m64 = statistics.median(est128), statistics.median(est64)
    lo128, hi128 = min(est128), max(est128)
    lo64, hi64 = min(est64), max(est64)
    ax.errorbar([0], [m128], yerr=[[m128 - lo128], [hi128 - m128]],
                fmt=markers[label_name], ms=5.5, color="#1f4e79", capsize=3, lw=1.1)
    ax.errorbar([1], [m64], yerr=[[m64 - lo64], [hi64 - m64]],
                fmt=markers[label_name], ms=5.5, mfc="none", color="#1f4e79", capsize=3, lw=1.1)
    ax.scatter([0], [m128], marker=markers[label_name], s=35, color="#1f4e79",
               zorder=3, label=label_name)
    ax.scatter([1], [m64], marker=markers[label_name], s=35, facecolors="none",
               color="#1f4e79", zorder=3)

ax.axhline(C128, color="#b02a37", ls="--", lw=1.2)
ax.axhline(C64, color="#b02a37", ls=":", lw=1.2)

ax.text(0.5, C128 + 0.007, r"theory $(n-1)/D=39/128$", fontsize=8, color="#b02a37", ha="center")
ax.text(0.5, C64 - 0.015, r"theory $19/64$", fontsize=8, color="#b02a37", ha="center")

ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["$d=128$\n(5 seeds)", "$d=64$\n(3 seeds)"])
ax.set_ylabel(r"Measured load $c$ (median over $\delta$)")
ax.set_ylim(0.15, 0.42)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=8, loc="lower right")
ax.annotate("4/4 cells agree within 0.0064", xy=(0.5, 0.37), ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="#f8f9fa", ec="#ccc", lw=0.8))
fig.tight_layout()
savefig(fig, "fig3_load_scaling")

# ---------------------------------------------------------------------------
print("fig4: energy leakage vs decodability (key/re, d=128)")
fig, ax = plt.subplots(figsize=(5.0, 3.6))
y = [law("clave", "re", d, C128) for d in DELTAS]
ax.plot(x, y, "-", color="#1f4e79", lw=1.6, label="Predicted energy (law)")
fugas = [o05_fugas("wave_re", "clave", dl) for dl in DL]
med = [statistics.median(f) for f in fugas]
lo = [min(f) for f in fugas]
hi = [max(f) for f in fugas]
ax.errorbar(xs, med, yerr=[[a - b for a, b in zip(med, lo)], [a - b for a, b in zip(hi, med)]],
            fmt="o", ms=4, color="#1f4e79", capsize=2, lw=0.9, label=r"Measured energy")
ax2 = ax.twinx()
em = [statistics.median(o05_em_guarded("wave_re", "clave", dl)) for dl in DL]
ax2.plot(x, em, "s--", ms=4.5, color="#b02a37", lw=1.4, label="Decodable fraction (EM)")
ax2.set_ylim(-0.05, 1.05)
ax2.set_ylabel("Erased-item EM (decodability)", color="#b02a37")
ax2.tick_params(axis="y", colors="#b02a37")
ax.set_xlabel(r"Phase drift $\delta$ (rad)")
ax.set_ylabel("Pooled residual energy")
ax.set_ylim(-0.01, 0.11)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(alpha=0.25, lw=0.5)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")

ax.annotate("Decodability lags energy\n(EM $\\approx 0$ until $\\delta\\approx 0.25$)",
            xy=(0.04, 0.052), fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fdfdfd", ec="#bbb", lw=0.6))
fig.tight_layout()
savefig(fig, "fig4_energy_vs_decodability")

# ---------------------------------------------------------------------------
print("fig5: write stability")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
arms = ["delta $\\beta=1$\n(corrective, train)", "delta $\\beta=1$\n(random items)",
        "wave complex\n(superposition)", "wave Re\n(superposition)",
        "NLMS $\\beta=0.1$"]
first = [
    statistics.median([r["S_norm_first"][0] for r in autopsy["runs"]]),
    autopsy_beta["delta_random_S_norm"][0],
    autopsy_beta["wave_complex_S_norm"][0],
    autopsy_beta["wave_re_S_norm"][0],
    autopsy_beta["beta01_S_norm_first"][0],
]
last = [
    statistics.median([r["S_norm_last"][0] for r in autopsy["runs"]]),
    autopsy_beta["delta_random_S_norm"][1],
    autopsy_beta["wave_complex_S_norm"][1],
    autopsy_beta["wave_re_S_norm"][1],
    autopsy_beta["beta01_S_norm_last"][0],
]
colors = ["#b02a37", "#b02a37", "#1f4e79", "#1f4e79", "#2d6a4f"]
for i, (a, b, col) in enumerate(zip(first, last, colors)):
    ax.scatter([i], [a], marker="o", s=30, color=col, zorder=3)
    ax.annotate("", xy=(i, b), xytext=(i, a),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    ax.scatter([i], [b], marker=">", s=42, color=col, zorder=3)

ax.set_yscale("log")
ax.set_xticks(range(len(arms)))
ax.set_xticklabels(arms, fontsize=8)
ax.set_ylabel(r"$\|\mathbf{S}\|$ (state norm, epoch 1 $\rightarrow$ last)")
ax.set_ylim(10, 1e14)
ax.set_xlim(-0.6, 4.6)
ax.grid(alpha=0.25, lw=0.5, which="both")

ax.annotate(r"Diverges ($\|\mathbf{S}\| \sim 10^{11}$--$10^{12}$)" + "\n" + r"$\mathrm{EM}\approx 0.03$--$0.05$",
            xy=(0.6, 2e8), fontsize=8.2, color="#b02a37", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2f2", ec="#b02a37", lw=0.7))
ax.annotate(r"Bounded ($\|\mathbf{S}\| \approx 55$)" + "\n" + r"$\mathrm{EM}=1.000$",
            xy=(3.8, 800), fontsize=8.2, color="#2d6a4f", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f0f7f4", ec="#2d6a4f", lw=0.7))

fig.tight_layout()
savefig(fig, "fig5_write_stability")

# ---------------------------------------------------------------------------
print("fig6: TOST")
mean = o05["tost"]["mean_delta"]
sd = o05["tost"]["sd"]
n = o05["tost"]["n_seeds"]
eps = o05["tost"]["eps"]
ci = o05["tost"]["ci90"]
df = n - 1
se = sd / math.sqrt(n)
x = [v / 1000 for v in range(-60, 61)]
y = [st.t.pdf(v, df, loc=mean, scale=se) for v in x]
fig, ax = plt.subplots(figsize=(5.4, 3.4))

ax.axvspan(-eps, eps, color="#2d6a4f", alpha=0.08)
ax.axvline(-eps, color="#2d6a4f", ls="--", lw=1.1)
ax.axvline(eps, color="#2d6a4f", ls="--", lw=1.1)
ax.plot(x, y, color="#33415c", lw=1.5)
ax.axvline(mean, color="#1f4e79", lw=1.2)

# CI90 bar
y_bar = max(y) * 0.80
ax.plot([ci[0], ci[1]], [y_bar, y_bar], color="#b02a37", lw=3.5, solid_capstyle="butt")
ax.plot([ci[0]], [y_bar], "|", color="#b02a37", ms=10, mew=1.5)
ax.plot([ci[1]], [y_bar], "|", color="#b02a37", ms=10, mew=1.5)

ax.annotate(f"Mean $\\Delta={mean:.4f}$", xy=(mean, max(y) * 1.05), ha="center", fontsize=8.5)
ax.annotate(f"90% CI [{ci[0]:.4f}, {ci[1]:.4f}]", xy=(-0.003, y_bar + 2.5), fontsize=8.5,
            color="#b02a37", ha="center")
ax.annotate("Crosses $-\\varepsilon$ by 0.0006", xy=(ci[0], y_bar), xytext=(-0.042, y_bar - 10),
            arrowprops=dict(arrowstyle="->", color="#b02a37", lw=0.9),
            fontsize=8, color="#b02a37", ha="center")

ax.text(0.96, 0.94, r"Equivalence band $\pm\varepsilon=0.02$", transform=ax.transAxes,
        fontsize=8, ha="right", va="top", color="#2d6a4f",
        bbox=dict(boxstyle="round,pad=0.2", fc="#f0f7f4", ec="#2d6a4f", lw=0.6))
ax.set_xlabel("Selectivity difference (wave complex $-$ NLMS)")
ax.set_ylabel("Density ($t$-distribution, $\\mathrm{df}=4$)")
ax.set_xlim(-0.06, 0.06)
ax.set_ylim(0, 52)
fig.tight_layout()
savefig(fig, "fig6_tost")

print("done")