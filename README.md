# nativehdc — GHRR phase-kernel attention, honestly measured

A **gate-disciplined bake-off** of FHRR/Holonomic (GHRR) phase-kernel attention inside a 36 M-parameter
TinyLM trained on TinyStories (181.8 M tokens), against the same model with standard attention.
Every experiment declared its accept/reject rule *before* launch; every verdict is in `STATUS.md`; null and
marginal results are reported as such.

**TL;DR numbers (all val-PPL on one corpus, seed-swept):**

| claim | result |
|---|---|
| G2 bake-off (GHRR @ D=4096 vs vanilla) | 5.966 vs 5.511 achieved baseline → **+8.3 %, inside pre-registered ±10 %** — keep |
| Hypervisor shrink D 4096→1024 | perplexity equal-or-better (paired, 3 seeds) at **−44 % params**; costs associative capacity (bundled recall 1.00 → 0.54 @ K=256) |
| Score temperature τ 4→8→16 | cumulative **−2.7 %**, wins all 3 paired seeds at τ=16 |
| Champion (τ=16, D=1024) | mean **5.809 ± 0.014** over 3 seeds; still behind vanilla's tuned 5.511 — the honest headline is *feasible + useful gates*, not SOTA |
| Associative gate (g3/g4/phase-4) | answerability refusal + doc citation work at 5/5; downstream multi-hop transfer: **null** |

White paper (PDF, incl. methodology + all failures): `report/nativehdc-whitepaper-2026-11.pdf`
— arXiv entry pending.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate   # or bin/activate on Linux
pip install torch transformers numpy                 # bf16, CUDA
.venv/Scripts/python.exe src/train.py --attn ghrr --lr 3e-4 --tau 16 --hv-dim 1024 --seed 1337
# gates: src/g0_verify.py, g1_test.py, g3_test.py, g4_eval.py, phase4_demo.py (exit 0 = pass)
```

Checkpoints and run logs (`runs/`, ~3.4 GB) are not in git; `report/figs/` embeds the curves.

## License
Code: MIT. White paper text: CC BY 4.0.
