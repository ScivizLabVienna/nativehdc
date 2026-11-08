"""Paper #2 figures from frozen G13 stdout (raw/g13_run.txt). Run with SYSTEM python."""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "raw", "g13_run.txt")
FIGS = os.path.join(os.path.dirname(__file__), "figs"); os.makedirs(FIGS, exist_ok=True)
txt = open(RAW, encoding="utf-8").read()
num = lambda s: [float(x) for x in re.findall(r"[\d.]+", s)]

x1 = {int(m[0]): (num(m[1]), num(m[2])) for m in re.findall(r"D=\s*(\d+)\s+FLAT=\[([^\]]*)\]\s+SHEAF=\[([^\]]*)\]", txt)}
x2 = {int(m[0]): (num(m[1]), num(m[2])) for m in re.findall(r"nrel=\s*(\d+)\s+FLAT=\[([^\]]*)\]\s+SHEAF=(.*)", txt)}
x4 = [(float(a), float(b)) for a, b in re.findall(r"seed=\d+: FLAT@3=([\d.]+)\s+SHARDchart@3=([\d.]+)", txt)]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#3b6ea5", "#c96f4a"

# --- Fig 1: X1 capacity curves ----------------------------------------------
D = sorted(x1)
fm = [np.mean(x1[d][0]) for d in D]
flo = [min(x1[d][0]) for d in D]; fhi = [max(x1[d][0]) for d in D]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.errorbar(D, fm, yerr=[np.array(fm) - flo, np.array(fhi) - np.array(fm)],
            fmt="o-", color=ORANGE, capsize=3, label="flat bundle")
for d in D:
    ax.scatter([d] * 3, x1[d][0], s=8, color=ORANGE, alpha=.5)
ax.axhline(1.0, color=BLUE, lw=1.6, label=r"chart arms (SHEAF $=$ SHARD $=1.000$)")
for d, y in zip(D, fm):
    ax.annotate(f"{y:.3f}", (d, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
ax.axvline(246, ls=":", lw=.9, color="k")
ax.annotate("D/2lnD knee heuristic\n(underestimates by ~3x)", (246, .45), fontsize=7, rotation=90, ha="right")
ax.set_xscale("log", base=2); ax.set_xticks(D + [246]); ax.set_xticklabels([str(d) for d in D] + ["knee"])
ax.axvspan(2560, 6144, alpha=.06, color=ORANGE)
ax.set_xlabel("dimension D (unique-tag charts vs flat, K=512)")
ax.set_ylabel("depth-2 recall @K=512"); ax.set_ylim(-.05, 1.12)
ax.legend(frameon=False, loc="center left")
ax.set_title("G13/X1: chart-local partitioning vs flat bundling (3 seeds)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_x1.png"), dpi=200); plt.close(fig)

# --- Fig 2: X2 granularity continuum ----------------------------------------
nr = sorted(x2)
sm = [np.mean(x2[n][1]) for n in nr]; slo = [min(x2[n][1]) for n in nr]; shi = [max(x2[n][1]) for n in nr]
flm = [np.mean(x2[n][0]) for n in nr]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.errorbar(nr, sm, yerr=[np.array(sm) - slo, np.array(shi) - np.array(sm)], fmt="o-", color=BLUE, capsize=3, label="relation-pool charts")
ax.plot(nr, flm, "s--", color=ORANGE, label="flat bundle (nrel-independent)")
for n, y in zip(nr, sm): ax.annotate(f"{y:.3f}", (n, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7.5)
ax.set_xscale("log", base=4); ax.set_xticks(nr); ax.set_xticklabels([str(n) for n in nr])
ax.set_xlabel(r"relations per pool $n_{\mathrm{rel}}$ (K=512, D=1024)")
ax.set_ylabel("depth-2 recall"); ax.set_ylim(-.05, 1.12); ax.legend(frameon=False, loc="center left")
ax.set_title(r"G13/X2: key-aliasing continuum ($n_{\mathrm{rel}} \to K$: no ceiling)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_x2.png"), dpi=200); plt.close(fig)

# --- Fig 3: X4 depth-3 -------------------------------------------------------
f3 = np.mean([a for a, _ in x4]); s3 = np.mean([b for _, b in x4])
fig, ax = plt.subplots(figsize=(3.6, 2.8))
bars = ax.bar(["flat", "chart"], [f3, s3], width=.5, color=[ORANGE, BLUE])
for b, v in zip(bars, [f3, s3]): ax.text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("depth-3 chain recall")
ax.set_title("G13/X4: three-hop chains (unique tags, D=1024)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_x4.png"), dpi=200); plt.close(fig)

print("wrote:", *[f for f in sorted(os.listdir(FIGS))], "| x1 D:", D, "| x4 means:", round(f3, 3), round(s3, 3))
