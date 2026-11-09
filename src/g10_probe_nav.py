"""G10 probe: is the HDC layer fit for GPS-denied drone nav / Hermes memory roles?
Measures what the decision actually turns on: encode rate, place-memory capacity
at BOTH champion geometries (D=1024 and 4096), and recall tick latency. CPU-only."""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ghrr_attn import hash_phasors, bind, unbind

torch.set_num_threads(8)


def enc_rate(D, n=4096):
    hash_phasors(torch.arange(512), D)  # warm
    t0 = time.perf_counter()
    hash_phasors(torch.arange(1_000_000, 1_000_000 + n), D)
    return n / (time.perf_counter() - t0)


def pose_recall(K, D):
    sig = hash_phasors(torch.arange(K), D)
    pose = hash_phasors(9_000_000 + torch.arange(K) * 7, D)
    mem = bind(sig, pose).sum(0)
    r = unbind(mem.unsqueeze(0).expand(K, D), sig)
    top = (r @ pose.conj().T).real.argmax(1)
    return float((top == torch.arange(K)).float().mean())


if __name__ == "__main__":
    if "--content" in sys.argv:  # G15 content-encoder regression gate (G14 champion); exit 0 = all asserts
        import e1_encoder as E1
        import numpy as np
        torch.set_num_threads(8)

        def ck(name, cond):
            print(f"{'PASS' if cond else 'FAIL'}  {name}")
            return cond

        gates, t0 = [], __import__("time").time()
        data = E1.load_sentences(os.path.join(os.path.dirname(__file__), "..", "runs", "g4_docs.db"))
        sigs = [E1.tk(s) for _, _, s in data]
        bins = E1.corpus_bins(sigs)
        gates.append(ck("corpus_stats_match_campaign", len(sigs) == 2935 and abs(np.mean([len(b) for b in bins]) - 20.7) < 0.5))
        X4096 = E1.rf_encode(bins, 4096)
        gates.append(ck("determinism_byte_identical", bool((E1.rf_encode(bins[:64], 4096) == E1.rf_encode(bins[:64], 4096)).all())))
        docid = np.array([d for d, _, _ in data])
        idxs = np.array([i for _, i, _ in data])
        key2row = {(docid[r], int(idxs[r])): r for r in range(len(sigs))}
        rel = [(r, key2row[(docid[r], int(idxs[r]) + 1)]) for r in range(len(sigs)) if (docid[r], int(idxs[r]) + 1) in key2row]
        rng = np.random.default_rng(7)
        unrel = []
        while len(unrel) < 3 * len(rel):
            x, y = rng.integers(0, len(sigs), 2)
            if docid[x] != docid[y]:
                unrel.append((int(x), int(y)))
        unrel = unrel[:len(rel)]
        Xhc = E1.rf_encode([["h|" + "|".join(sorted(s))[:200]] for s in sigs], 4096)

        def auc(X):
            rs = np.array([(X[a].conj() * X[b]).real.mean() for a, b in rel])
            us = np.array([(X[a].conj() * X[b]).real.mean() for a, b in unrel])
            return float((rs[:, None] > us[None, :]).mean() + 0.5 * (rs[:, None] == us[None, :]).mean())

        a_lm, a_hc = auc(X4096), auc(Xhc)
        print(f"AUC lm+tok {a_lm:.3f} vs hctl {a_hc:.3f}")
        gates.append(ck("separation_auc", a_lm > 0.62 and a_lm > a_hc))
        rng9 = np.random.default_rng(9)
        ii, jj = rng9.integers(0, len(sigs), 4000), rng9.integers(0, len(sigs), 4000)
        keep = ii != jj
        o4096 = float(np.abs((X4096[ii[keep]].conj() * X4096[jj[keep]]).real.mean(1)).mean())
        print(f"ortho content @D4096 {o4096:.4f} (campaign 0.0115; baseline hash-ids 0.0092)")
        gates.append(ck("ortho_D4096_within_2x_baseline", o4096 < 0.0184))
        X1024 = E1.rf_encode(bins, 1024)
        o1024 = float(np.abs((X1024[ii[keep]].conj() * X1024[jj[keep]]).real.mean(1)).mean())
        print(f"ortho content @D1024 {o1024:.4f} — G14 caveat: monitor, not gate (adopt D=4096 only)")
        del X1024
        accs = {}
        for K in (512, 1024, 2048):
            for sd in (7, 13, 1337):
                pose = np.asarray(hash_phasors(
                    torch.tensor(9_000_000 + sd * 10_000_000 + 7 * np.arange(K)), 4096)).astype(np.complex64)
                Xr = X4096[:K]
                r = Xr.conj() * (Xr * pose).sum(0)
                acc = float((np.argmax((r @ pose.conj().T).real, 1) == np.arange(K)).mean())
                print(f"content recall K={K} seed={sd}: {acc:.4f}")
                accs.setdefault(K, []).append(acc)
        # fixed-rows[:K] protocol drifts slightly vs campaign random-subset rows (harness convention
        # keeps it deterministic); sanity band = regression detector, exact numbers: p2b_run.log.
        print(f"mean over seeds: " + " ".join(f"K{K}:{np.mean(v):.3f}" for K, v in accs.items())
              + " (campaign random-row means 0.776/0.345/0.087)")
        gates.append(ck("recall_sanity_band", 0.72 < np.mean(accs[512]) < 0.85 and 0.28 < np.mean(accs[1024]) < 0.42))
        n_bad = len([g for g in gates if not g])
        print(f"\n{len(gates) - n_bad}/{len(gates)} gates PASS | {__import__('time').time() - t0:.0f}s" + (" -> G15 OK" if not n_bad else f" FAILED: {n_bad}"))
        sys.exit(0 if not n_bad else 1)
    for D in (1024, 4096):
        print(f"encode rate @D={D}: {enc_rate(D):,.0f} novel-places/s")
    for D in (1024, 4096):
        print("loop-closure recall @D=%d: " % D + " ".join(
            f"K{k}:{pose_recall(k, D):.2f}" for k in (64, 128, 256, 512, 1024)))
    sig = hash_phasors(torch.arange(128), 1024)
    mem = bind(sig, hash_phasors(9_000_000 + torch.arange(128) * 7, 1024)).sum(0)
    q = sig[3:4].expand(128, 1024)
    t0 = time.perf_counter()
    for _ in range(2000):
        unbind(mem.expand(128, 1024), q)
    dt = (time.perf_counter() - t0) / 2000
    print(f"recall tick (K=128 places, D=1024): {dt*1e3:.2f} ms -> ~{1/dt:,.0f} Hz on 8 CPU threads")
