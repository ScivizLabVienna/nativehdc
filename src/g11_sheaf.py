"""G11 sheaf-gluing probe — see docs/g11-sheaf-spec.md incl. revision log (rev 1).
Arms (identical stored content; only READOUT structure differs). Each edge carries a
relation key rel_e from a pool of 8; queries expose (source, per-hop relation) to ALL arms:
  FLAT  : one global bundle sum_e h(src_e)*rel_e*val_e; cue-cancel both, cleanup vs global dict.
  SHARD : per-source bundles sum_e rel_e*val_e; cue-cancel rel only (partitioning, no gluing).
  SHEAF : same charts over SECTIONS R_e x h(tgt); chart-local section cleanup, gluing
          conj(R_e) x q verified vs h(m) (theory-anchored axiom check), hop-2 from chart(m).
Metric: depth-2 top-1 recall, 150 valid chains, attempted-only. CPU $0."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ghrr_attn import hash_phasors

torch.set_num_threads(8)
D = 1024
SEEDS = (7, 13, 1337)
KS = (64, 256, 512)
NREL = 8


def top1(q, cands):
    return int((q * torch.conj(cands)).sum(1).real.argmax())


def build(K, seed):
    g = torch.Generator().manual_seed(seed)
    N = K // 4
    src = torch.randint(0, N, (K,), generator=g)
    tgt = torch.randint(0, N, (K,), generator=g)
    same = src == tgt
    tgt[same] = (tgt[same] + 1) % N
    ridx = torch.randint(0, NREL, (K,), generator=g)
    h = hash_phasors(torch.arange(N), D).to(torch.complex128)
    RH = hash_phasors(1_000_000 + torch.arange(NREL), D).to(torch.complex128)
    out = [[] for _ in range(N)]
    for eid in range(K):
        out[int(src[eid])].append((eid, int(tgt[eid])))
    chains = [(s, e1, m, e2, t) for s in range(N) for (e1, m) in out[s]
              for (e2, t) in out[m] if s != t]
    perm = torch.randperm(len(chains), generator=g).tolist()
    return N, src, tgt, ridx, h, RH, out, [chains[i] for i in perm[:150]]


def run(arm, N, src, tgt, ridx, h, RH, out, qchains, R):
    val = (lambda e: R[e] * h[int(tgt[e])]) if arm == "sheaf" else (lambda e: h[int(tgt[e])])
    sec = lambda e: RH[int(ridx[e])] * val(e)                      # keyed stored item
    charts = {}
    for s in range(N):
        if out[s]:
            eids = torch.tensor([e for e, _ in out[s]])
            cands = torch.stack([val(e) for e in eids])
            charts[s] = (sum((sec(e) for e in eids), torch.zeros(D, dtype=torch.complex128)), eids, cands)
    mem_flat = sum((h[int(src[e])] * sec(e) for e in range(len(src))), torch.zeros(D, dtype=torch.complex128))
    dmax = max(len(o) for o in out if o)
    thr = 1 - 4 * (2 * dmax / D) ** 0.5                            # theory: sim ~ N(1, sqrt(2*deg/D))
    hit, sims = 0, []
    for (s, e1, m, e2, t) in qchains:
        r1, r2 = RH[int(ridx[e1])], RH[int(ridx[e2])]
        if arm == "flat":                                          # hop-1: cancel src+rel globally
            q = mem_flat * torch.conj(h[s]) * torch.conj(r1)
            m_hat = top1(q, h)
        else:                                                      # charts keyed externally by source
            if s not in charts:
                continue
            bundle, eids, cands = charts[s]
            q = bundle * torch.conj(r1)
            j = top1(q, cands)
            m_hat = int(tgt[int(eids[j])])   # eids maps winning candidate -> edge -> node (all chart arms)
        if arm == "sheaf" and s in charts:
            _, eids1, _ = charts[s]
            glued = torch.conj(R[int(eids1[j])]) * (charts[s][0] * torch.conj(r1))
            sims.append(float((glued * torch.conj(h[int(tgt[int(eids1[j])])])).sum().real) / D)
        if m_hat != m:
            continue
        if arm == "flat":
            q2 = mem_flat * torch.conj(h[m]) * torch.conj(r2)
            t_hat = top1(q2, h)
        else:
            if m not in charts:
                continue
            b2, eids2, cands2 = charts[m]
            q2 = b2 * torch.conj(r2)
            j2 = top1(q2, cands2)
            t_hat = int(tgt[int(eids2[j2])])
            if arm == "sheaf":
                glued2 = torch.conj(R[int(eids2[j2])]) * q2
                sims.append(float((glued2 * torch.conj(h[int(tgt[int(eids2[j2])])])).sum().real) / D)
        if t_hat == t:
            hit += 1
    mean_glue = sum(sims) / max(len(sims), 1)
    return hit / max(len(qchains), 1), mean_glue, thr


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        for K in KS:
            args = build(K, seed)
            N, srcs = args[0], args[1]
            gR = torch.Generator().manual_seed(seed + 10_000)
            R = torch.polar(torch.ones(len(srcs), D, dtype=torch.float64),
                            6.283185307179586 * torch.rand(len(srcs), D, dtype=torch.float64, generator=gR)).to(torch.complex128)
            f, _, _ = run("flat", *args, R)
            sh, _, _ = run("shard", *args, R)
            sf, gl, thr = run("sheaf", *args, R)
            rows.append((seed, K, f, sh, sf, gl, thr))
            print(f"seed={seed:5} K={K:4}  FLAT={f:.3f}  SHARD={sh:.3f}  SHEAF={sf:.3f}  glue_mean={gl:.4f} (thr {thr:.3f})")
    ok_a = all(r[4] >= 0.90 for r in rows if r[1] == 64)
    ok_b = all(r[4] - r[2] >= 0.30 for r in rows if r[1] in (256, 512))
    ok_c = all(r[5] > r[6] for r in rows)
    checks = {"G11a_sanity_K64": ok_a, "G11b_sheaf_minus_flat_ge_030": ok_b, "G11c_gluing_axiom_theory": ok_c}
    attr = sum(r[4] - r[3] for r in rows) / len(rows)
    verdict = "KEEP-structural" if all(checks.values()) else "NULL"
    print(f"\nverdict={verdict}  attribution(sheaf-shard mean)={attr:+.3f}")
    bad = [k for k, v in checks.items() if not v]
    print(f"{len(checks)-len(bad)}/{len(checks)} gates " + (f"FAILED: {bad}" if bad else "ALL PASS"))
    sys.exit(1 if (bad or verdict == "NULL") else 0)
