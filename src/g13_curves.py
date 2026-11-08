"""G13 capacity-curve sweeps (docs/g13-curves-spec.md). Frozen evaluator g11_sheaf driven
via module globals only. Comparability contract: chart/flat hop semantics and TRUE-intermediate
query convention are mirrored from run() verbatim; X4 extends one more hop with the same rules."""
import os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import g11_sheaf as m
from ghrr_attn import hash_phasors

SEEDS, K = [7, 13, 1337], 512


def make_R(n, d, seed):
    gR = torch.Generator().manual_seed(seed + 10_000)
    return torch.polar(torch.ones(n, d, dtype=torch.float64),
                       6.283185307179586 * torch.rand(n, d, dtype=torch.float64, generator=gR)).to(torch.complex128)


def tagged_args(d, seed):
    """g11 build then the G12 deviation: unique per-edge tags at size d."""
    m.D = d
    N, src, tgt, ridx, h, RH, out, qc = m.build(K, seed)
    return (N, src, tgt, torch.arange(K), h,
            hash_phasors(1_000_000 + torch.arange(K), d).to(torch.complex128), out, qc)


def depth3(args, R):
    N, src, tgt, ridx, h, RH, out, qc = args
    charts = {}
    for s in range(N):
        if out[s]:
            eids = [e for e, _ in out[s]]
            charts[s] = (sum((RH[int(ridx[e])] * h[int(tgt[e])] for e in eids), torch.zeros(m.D, dtype=torch.complex128)), eids)
    mem = sum((h[int(src[e])] * RH[int(ridx[e])] * h[int(tgt[e])] for e in range(len(src))), torch.zeros(m.D, dtype=torch.complex128))
    rng = random.Random(args[0] + 777)
    f3 = sh3 = n3 = 0
    for (s, e1, mid, e2, t) in qc:
        if not out[t]:
            continue
        e3 = rng.choice(out[t])[0]
        if int(tgt[e3]) == t:
            continue  # self-loop target keeps query trivially addressable; skip for honesty
        n3 += 1
        r1, r2, r3 = (RH[int(ridx[e])] for e in (e1, e2, e3))
        ok_f = True
        for cue_key, node in ((s, None),):
            pass
        # hop convention mirrored from run(): each hop queried at the TRUE previous node
        if int(top_local(mem * torch.conj(h[s]) * torch.conj(r1), h)) != mid:
            ok_f = False
        if ok_f and top_local(mem * torch.conj(h[mid]) * torch.conj(r2), h) != t:
            ok_f = False
        if ok_f and top_local(mem * torch.conj(h[t]) * torch.conj(r3), h) == int(tgt[e3]):
            f3 += 1
        ok_s = True
        for (node, cue, step) in ((s, r1, mid), (mid, r2, t), (t, r3, int(tgt[e3]))):
            if node not in charts:
                ok_s = False
                break
            b, eids = charts[node]
            q = b * torch.conj(cue)
            cands = [int(tgt[e]) for e in eids]
            sims = (q * torch.conj(torch.stack([h[c] for c in cands]))).real.sum(1)
            if cands[int(sims.argmax())] != step:
                ok_s = False
                break
        if ok_s:
            sh3 += 1
    return f3 / max(n3, 1), sh3 / max(n3, 1), n3


def top_local(q, mat):
    return int((q * torch.conj(mat)).real.sum(1).argmax())


if __name__ == "__main__":
    print("== X1: unique-tag charts vs flat, K=512, D sweep ==")
    x1 = {}
    for d in (512, 1024, 4096):
        accs = {"flat": [], "shard": [], "sheaf": []}
        for seed in SEEDS:
            args = tagged_args(d, seed)
            R = make_R(len(args[1]), d, seed)
            for arm in accs:
                accs[arm].append(m.run(arm, *args, R)[0])
        x1[d] = accs
        print(f"D={d:5}  FLAT={['%.3f' % v for v in accs['flat']]}  SHEAF={['%.3f' % v for v in accs['sheaf']]}  SHARD={['%.3f' % v for v in accs['shard']]}")
    print("== X2: rel-pool granularity, K=512, D=1024 ==")
    m.D = 1024
    x2 = {}
    for nr in (2, 8, 32):
        m.NREL = nr
        sf = []
        for seed in SEEDS:
            args = m.build(K, seed)
            R = make_R(len(args[1]), 1024, seed)
            sf.append((m.run("flat", *args, R)[0], m.run("sheaf", *args, R)[0]))
        x2[nr] = sf
        print(f"nrel={nr:3}  FLAT={['%.3f' % a for a, _ in sf]}  SHEAF={['%.3f' % _, a in [(b, a) for a, b in sf]] if False else ' '.join('%.3f' % b for _, b in sf)}")
    m.NREL = 8
    print("== X3 (exploratory): identity-map vs random-phase maps, unique tags D=1024 ==")
    for seed in SEEDS[:2]:
        args = tagged_args(1024, seed)
        Rr = make_R(len(args[1]), 1024, seed)
        Ro = torch.ones(len(args[1]), 1024, dtype=torch.complex128)
        a_r = m.run("sheaf", *args, Rr)[0]
        a_o = m.run("sheaf", *args, Ro)[0]
        print(f"seed={seed}: sheaf(random)={a_r:.3f}  sheaf(identity)={a_o:.3f}  diff={abs(a_r - a_o):.4f}")
    print("== X4 (exploratory): depth-3, unique tags, D=1024 ==")
    for seed in SEEDS:
        args = tagged_args(1024, seed)
        f3, s3, n3 = depth3(args, None)
        print(f"seed={seed}: FLAT@3={f3:.3f}  SHARDchart@3={s3:.3f}  (chains={n3})")

    # --- gates from emitted rows (rev-free, spec G13a-f) ---
    ok_a = all(v >= 0.97 for d in x1 for v in x1[d]["sheaf"] + x1[d]["shard"])
    shape = sum(1 for s in range(3) if x1[4096]["flat"][s] > x1[1024]["flat"][s]) >= 2
    ok_c = all(min(x1[d]["sheaf"][s] - x1[d]["flat"][s] for s in range(3)) > 0 for d in x1)
    ok_d = all(x2[nr][s][1] >= x2[prev][s][1] - 0.05 for prev, nr in ((2, 8), (8, 32)) for s in range(3))
    ok_e = all(0.5 <= x2[8][s][1] <= 0.8 for s in range(3))
    ok_f = all(a < 0.15 for nr in x2 for a, _ in x2[nr]) and all(max(x1[d]["flat"]) < 0.60 for d in x1)
    checks = {"G13a_charts": ok_a, "G13b_flat_shape": shape, "G13c_delta_pos": ok_c,
              "G13d_nrel_monotone": ok_d, "G13e_anchor_band": ok_e, "G13f_flat_bands": ok_f}
    bad = [k for k, v in checks.items() if not v]
    print(f"\nverdict={'ALLPASS' if not bad else 'CHECK'}  failed={bad}")
