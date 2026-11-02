"""G2 verdict helper: parse runs/{arm}.jsonl, print PPL curves side-by-side + decision rule."""
import json, os, sys

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def curve(arm):
    p = os.path.join(RUNS, f"{arm}.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        r = json.loads(line)
        if "ppl" in r:
            out[r["step"]] = r["ppl"]
    return out


g, v = curve("ghrr"), curve("vanilla")
steps = sorted(set(g) & set(v))
print(f"{'step':>7} {'ghrr_ppl':>9} {'van_ppl':>9} {'ghrr_vs_van':>12}")
for s in steps:
    print(f"{s:>7} {g[s]:>9.2f} {v[s]:>9.2f} {100 * (g[s] / v[s] - 1):>+11.1f}%")
if steps:
    bg, bv = min(g.values()), min(v.values())
    d = 100 * (bg / bv - 1)
    print(f"\nBEST ghrr={bg:.2f} vanilla={bv:.2f} delta={d:+.1f}%")
    if os.path.exists(os.path.join(RUNS, "vanilla_best.pt")):
        verdict = ("KEEP GHRR (within +/-10%) -> Phase 4" if abs(d) <= 10 else
                   "GHRR WORSE >10% -> one LR re-run (3e-4/1e-3), then m>1 pivot")
        print(f"G2 VERDICT: {verdict}")
    else:
        print("(vanilla arm still running — rerun after both DONE lines in runs/bakeoff.log)")
