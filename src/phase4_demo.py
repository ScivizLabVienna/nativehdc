"""Phase-4 associative-memory demo (nativehdc). exit 0 = all asserts pass.

A) true HDC superposition memory: K bindings bundled into ONE D=4096 complex
   vector (ghrr_attn bind/unbind/similarity, CPU); capacity vs noise sweep.
B) end-to-end: champion TinyLM via Harness gate answers a grounded question
   with doc citation (same live path as g3_test).
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "..", "runs")
checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


if __name__ == "__main__":
    import torch
    from ghrr_attn import hash_phasors, bind, unbind, similarity

    torch.manual_seed(0)
    D = 4096

    # similarity(a,b)=mean Re conj(a*b) -> ranking over candidates is the same inner product;
    # compute matrix directly to keep cleanup one matmul (shortcut: skips mean-scale, rank-invariant).
    def acc(K, cue_noise=0.0):
        a = hash_phasors(torch.arange(1_000_000, 1_000_000 + K), D)
        b = hash_phasors(torch.arange(2_000_000, 2_000_000 + K), D)
        mem = bind(a, b).sum(0)
        cues = a if not cue_noise else (1 - cue_noise) * a + cue_noise * a.roll(1, 0)
        r = unbind(mem.unsqueeze(0).expand(K, D), cues)
        top = (r @ b.conj().T).real.argmax(1)
        return float((top == torch.arange(K)).float().mean())

    ks = [16, 64, 256, 1024, 2048]
    curve = {k: acc(k) for k in ks}
    for k in ks:
        print(f"K={k:>5}  cleanup top-1 acc = {curve[k]:.3f}")
    # knee matches HDC capacity theory: ~D/(2 ln D) ~ 246 bundled items at D=4096
    check("bundled_memory_perfect_to_64", all(curve[k] == 1.0 for k in (16, 64)))
    check("knee_at_theory_256_just_past", curve[256] >= 0.99 and curve[1024] < curve[256])

    n20 = acc(64, cue_noise=0.2)
    print(f"K=64, cue mixed 20% wrong-cue noise -> acc = {n20:.3f}")
    check("robust_to_20pct_cue_noise", n20 == 1.0)

    # single-binding exactness sanity (no crosstalk baseline)
    a1, b1 = hash_phasors(torch.tensor([42]), D), hash_phasors(torch.tensor([99]), D)
    check("single_bind_unbind_exact", float(similarity(unbind(bind(a1, b1), a1), b1)) > 0.999)

    # ---- B) live end-to-end through the gate (g3_test pattern) ----
    best = os.environ.get("GHRR_CKPT") or os.path.join(RUNS, "ghrr_best.pt")
    if "--no-live" in sys.argv or not os.path.exists(best):
        print("live part skipped (--no-live or no checkpoint)")
    else:
        import sqlite3
        from transformers import AutoTokenizer
        from harness import Harness
        from model import TinyLM
        torch.set_num_threads(6)
        DB = os.path.join(RUNS, "g4_store.db") if os.path.exists(os.path.join(RUNS, "g4_store.db")) else os.path.join(RUNS, "g3_fixture.db")
        if not os.path.exists(DB):  # rebuild g3 fixture db (same triples as g3_test)
            DB = os.path.join(RUNS, "phase4_demo.db")
            con = sqlite3.connect(DB)
            con.execute("CREATE TABLE IF NOT EXISTS triples(s TEXT,p TEXT,o TEXT,doc TEXT)")
            con.execute("DELETE FROM triples")
            con.executemany("INSERT INTO triples VALUES(?,?,?,?)", [
                ("mary", "found", "a red ball", "doc-7"), ("john", "ate", "the blue cake", "doc-12"),
                ("sue", "played", "in the green garden", "doc-3"), ("bob", "lost", "his yellow kite", "doc-9"),
                ("ana", "drew", "a big purple house", "doc-5"), ("tom", "walked", "to the small lake", "doc-1")])
            con.commit(); con.close()
        tk = AutoTokenizer.from_pretrained("gpt2")
        m = TinyLM(attn_type="ghrr", tau=float(os.environ.get("GHRR_TAU", "16")), hv_dim=int(os.environ.get("GHRR_HV", "1024")))  # ckpt carries no tau; env must match training value
        m.load_state_dict(torch.load(best, map_location="cpu")); m.eval()

        @torch.inference_mode()
        def live_g(text):
            ids = tk.encode(text)[:m.ctx - 24] or [tk.eos_token_id]
            out = []
            for _ in range(24):
                lg, _ = m(torch.tensor([(ids + out)[-m.ctx:]]))
                out.append(int(lg[0, -1].argmax()))
            return tk.decode(out)

        h = Harness(live_g, DB, os.path.join(HERE, "injection_exemplars.txt"),
                    trace_path=os.path.join(RUNS, "phase4_trace.jsonl"))
        r = h.ask("who found a red ball")
        print(f"live[{r['decision']}] cites={r['cites_doc']} out={r['answer']!r}")
        check("live_model_answers_via_assoc_gate", r["answerable"] and r["cites_doc"])

    bad = [k for k, v in checks.items() if not v]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks PASS" + (f" | FAILED: {bad}" if bad else " -> PHASE-4 DEMO OK"))
    sys.exit(1 if bad else 0)
