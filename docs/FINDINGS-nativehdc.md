# GHRR/HDC × TinyLM — Findings (as of 2026-11-03)

Compact results report. Live state: STATUS.md. Plan: docs/PLAN-nativehdc.md.

## Setup
- GPU RTX 3090, co-resident with Strata inference (:8080) — throughput swings 1.6k–33k tok/s, expected.
- TinyLM d384 / L6 / H8 + hv-dim 4096 complex (≈64.2 M params), GPT-2 BPE, TinyStories, ctx=64, batch×ctx=1024 tok/step; corpus deviation (181.8 M train / 4.0 M val) logged & accepted.
- GHRR attention = hash-phasor binding matrices as RoPE-style score rotations; verified vs HoloVec oracle: bind err 2.4e-7, sim err 1.0e-8 (G1).

## G2 — quality gate (KEEP GHRR)
- ghrr@3e-4 val PPL **5.966** vs vanilla@6e-4 **5.511** at same token budget → **+8.3 %, inside pre-registered ±10 %**. Run 1 (lr 6e-4) had been +12.7 %; the gap is an optimization-conditioning effect (gentler LR shrank it +31 %→+8 % monotonically), not a capacity wall.
- Champion = deterministic seed-1337 replay, byte-exact DONE line `best_ppl=5.966` (clobber incident recovered; `--nobest`/`--suffix` guards now structural).

## G3/G4 — harness gates
- SQLite doc store + HV retrieval: stopword/stem-filtered hashes resolve phrase-similarity saturation; adversarial decoys pass.
- Live gate accuracy top-1 0.109 vs random ≈0.01 → >10× baseline; g3_test asserts 10/10 PASS against champion.

## G5 — temperature & context ablation (8 points + control, all @20k steps unless stated)
best val PPL grid (ghrr@3e-4, --nobest):

| tau | ctx64 | ctx128 |
|-----|-------|--------|
| 2   | 10.10 | 9.08   |
| 4   | 9.21  | 8.04   |
| 8   | 8.75  | 7.50   |
| 16  | 8.67  | 7.36   |

- **tau: monotone better up to ~8–16, then flat** (−9 %, −5–7 %, then −1–2 % per doubling). τ=8 promoted as safe default; score temp is a real but bounded lever (HV-phase softmax sharpening), no stability break observed.
- **ctx128's 9–11 % edge was an artifact of double tokens/step:** equal-token control (ctx64/τ8 @40 k steps = 82 M tok) gives **7.45 vs 7.50** — −0.7 %, noise. Method rule learned: *context-length ablations must be token-matched, never step-matched.* For short-document corpora (TinyStories) ctx=64 is optimal; longer ctx costs compute without benefit at this scale.
- All grid curves still descending at 20 k steps → lower bounds; ordering used only within-grid (not comparable to full-budget 5.966).

## Phase-4 — associative memory + grounded citation demo (`src/phase4_demo.py`, exit 0)
- True superposition memory: K pair-binds bundled into ONE D=4096 complex vector; unbind+cleanup readout. Measured knee acc 1.0 @≤64, 0.996 @256, 0.36 @1024 = HDC theory D/(2 ln D) ≈ 246 items; noise-robust (20 % wrong cue still 1.0 at K=64).
- Production harness ItemStore stores per-item vectors → not bundling-limited; m>1 binding matrices would raise the bundled-vector ceiling (memory-capacity lever = G2 pivot lever).
- End-to-end: champion TinyLM behind harness gate answers grounded question, cites doc-7, coherent continuation. Trace: runs/phase4_trace.jsonl.

## Champion promotion (2026-11-03, verified)
- Full-budget τ8 retrain: best val PPL **5.853** (t=4.37 h) vs τ4 champion 5.966 → **−1.9 %, strictly better → PROMOTED**. Old champion preserved (`runs/ghrr_tau4_champ_best.pt`). Verified under `GHRR_TAU=8`: g3_test 10/10, phase4_demo exit 0.
- Realized gain (−1.9 %) again undershot the 20 k-probe extrapolation (−3–5 %): third confirmation that short-probe deltas compress at full budget — discount probe-prize estimates roughly half.
- **Pitfall: checkpoints carry no τ scalar** (construct-time arg only). g3_test/phase4_demo now honor `GHRR_TAU` env; always match it to the training value, else scores are mis-scaled while load_state_dict succeeds silently.

## Standing ops lessons (enforced in tooling)
- `--nobest` for any short run; `--suffix` for full runs that must not touch champion paths (both smoke-proven via mtime isolation).
- Watch-pattern notices can replay stale lines — always verify live before acting.
