# STATUS — nativehdc (single source of truth per session; update before ending a session)

Current phase: **PRE-A / bootstrap → G1** (2026-11-02: G0 green; GHRR paper + HoloVec API mined into docs/research-ghrr-holovec.md)

## Next actionable item
- [ ] **G1 — differentiable GHRR-attention module.** `src/ghrr_attn.py` per PLAN + research-note deltas: base HVs as Q·Λ (hash-seeded Λ phases only), W *non-unitary trainable* (paper §IV-E relaxes unitarity), m=1 scalar path = FHRR cos-phase kernel, fixed random per-head/per-QKV positional maps (research note 4). Unit tests cross-check bind/unbind/similarity vs holovec within float tol + named asserts + exit 0.

## Gate log (fill as gates run: command, date, measured numbers, verdict)
- **G0 PASS** 2026-11-02 — `uv venv .venv --python 3.13`; `uv pip install torch --index-url .../cu126` + holovec 1.0.2, datasets, tokenizers; reproduce: `S:/nativehdc/.venv/Scripts/python.exe S:/nativehdc/src/g0_verify.py` → exit 0 first try. torch 2.14.1+cu126, cuda=RTX 3090; GHRR(dim=256,m=3,diag=0) bind/unbind rec_sim **1.000000**, non-comm sim(ab,ba) **0.1606**, m=1 commutes (FHRR-equiv) sim **1.000000**. Verdict: env ready, proceed to G1.
