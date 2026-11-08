"""G12 unique-edge-tag control (see docs/g12-tags-spec.md). Imports frozen g11_sheaf
build/run — arms paired by construction; only change: per-edge unique relation tags.
Trivialization control G12d: wrong-hop-tag must land on the wrong intermediate >=99%."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from ghrr_attn import hash_phasors
from g11_sheaf import D, SEEDS, KS, build, run, top1


def wrong_tag_check(N, src, tgt, ridx, h, RH, out, qchains):
    """rev-1a (pre-verdict; skill rule 'sanity anchors test mechanism, not contrast arm'):
    moved to SHEAF readout — at K=512 the FLAT arm is noise-saturated so wrong-tag correctness
    there is chance-level (observed 1.3%), an invalid instrument. Per chain: query the source
    chart with a DIFFERENT edge's tag (different target) and require recovery of a node != m."""
    charts = {}
    for e in range(len(src)):
        charts.setdefault(int(src[e]), []).append(e)
    wrong, attempted = 0, 0
    for (s, e1, m, e2, t) in qchains:
        alts = [e for (e, _) in out[s] if e != e1 and int(tgt[e]) != m]  # same-target alt = legit correct landing
        e_alt = alts[0] if alts else None
        if e_alt is None:
            continue
        attempted += 1
        eids = charts[s]
        chart = sum((h[s] * RH[int(ridx[e])] * (R[int(e)] * h[int(tgt[e])]) for e in eids),
                    torch.zeros(D, dtype=torch.complex128)) * torch.conj(RH[int(ridx[e_alt])])
        cands = [int(tgt[e]) for e in eids]
        sims = (chart * torch.conj(torch.stack([R[int(e)] * h[c] for e, c in zip(eids, cands)]))).real.sum(1)
        if cands[int(sims.argmax())] != m:
            wrong += 1
    return wrong / max(attempted, 1), attempted


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        for K in KS:
            N, srcs, tgts, ridx, h, RH, out, qc = build(K, seed)
            # THE only deviation from G11: unique per-edge tags
            ridx = torch.arange(K)
            RH = hash_phasors(1_000_000 + torch.arange(K), D).to(torch.complex128)
            gR = torch.Generator().manual_seed(seed + 10_000)
            R = torch.polar(torch.ones(K, D, dtype=torch.float64),
                            6.283185307179586 * torch.rand(K, D, dtype=torch.float64, generator=gR)).to(torch.complex128)
            f, _, _ = run("flat", N, srcs, tgts, ridx, h, RH, out, qc, R)
            sh, _, _ = run("shard", N, srcs, tgts, ridx, h, RH, out, qc, R)
            sf, gl, thr = run("sheaf", N, srcs, tgts, ridx, h, RH, out, qc, R)
            wt, na = wrong_tag_check(N, srcs, tgts, ridx, h, RH, out, qc)
            rows.append((seed, K, f, sh, sf, gl, thr, wt, na))
            print(f"seed={seed:5} K={K:4}  FLAT={f:.3f}  SHARD={sh:.3f}  SHEAF={sf:.3f}  glue={gl:.4f}(thr{thr:.2f})  wrongtag_wrong={wt:.3f}({na})")
    ok_a = all(r[4] >= 0.95 for r in rows if r[1] == 64) and all(r[5] > r[6] for r in rows)
    ok_b = all(r[4] - r[2] >= 0.30 for r in rows if r[1] in (256, 512))
    attr = sum(r[4] - r[3] for r in rows) / len(rows)
    ok_c = abs(attr) <= 0.05
    ok_d = all(r[7] >= 0.99 or r[8] == 0 for r in rows)
    checks = {"G12a_sanity": ok_a, "G12b_delta_ge_030": ok_b, "G12c_attr_pm005": ok_c, "G12d_trivialization": ok_d}
    low = [r for r in rows if r[4] < 0.90]
    verdict = "CONFIRMED" if all(checks.values()) and not low else ("NEEDS-INVESTIGATION" if low else "NULL")
    print(f"\nverdict={verdict}  attribution(sheaf-shard)={attr:+.3f}")
    bad = [k for k, v in checks.items() if not v]
    print(f"{len(checks)-len(bad)}/{len(checks)} gates " + (f"FAILED: {bad}" if bad else "ALL PASS"))
    sys.exit(0 if verdict == "CONFIRMED" else 1)
