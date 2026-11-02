# Research notes — GHRR (arXiv 2405.09689v2, IEEE TAI 2026) + HoloVec API

Read in full 2026-11-02 (cache: S:/hermes/cache/web/arxiv.org-2e4ceae5e2.md).
Deltas to the plan that these notes force or inform, all marked **[G1]** / **[G2]**.

## Paper facts that change/shape our design

1. **GHRR base-HV construction (Prop. 1):** each component is `A = Q Λ` with `Q ∈ U(m)`
   random basis, `Λ = diag(λ_1..λ_m)`, λ_j ∈ U(1) independent uniform phases.
   Quasi-orthogonality holds iff E[η λ*] = 0 (independent phases). Similarity is
   `δ(H1,H2) = Re Σ_j tr(A_j B_j†) / (D·m)`. Property 3 (binding preserves similarity)
   follows from trace cyclicity — free. **[G1]** this is exactly what to implement;
   hash-seed only the phases of Λ (PYTHONHASHSEED rule), Q may stay a fixed random draw.

2. **Attention equivalence (§IV-E, Eq. 22):** `[σ(Q∗K†)∗V]_j = σ(W_qj Λ_qj Λ_kj† W_kj†) W_vj Λ_vj`
   with `σ := softmax(Re[·])`. Unitarity of W_j may be **relaxed** "for expressivity
   comparable to a traditional Transformer". **[G1]** our differentiable module should keep
   W as a plain (not unitarized) trainable linear map — matches paper's own recommendation
   and avoids SVD-in-backward.

3. **Parameter semantics (§IV-E):** **D = number of attention heads, m = max tokens the
   GHRR-attention handles independently** (more than m tokens entangle). Attention range is
   bounded by m. **[G2]** our PLAN's "D=4096 HV dim" must be re-scaled to this view: with
   paper-style block attention we need m ≈ context length n, D = heads — but our G2 is the
   **m=1 scalar path**, so per-token attention stays standard-sequence-shaped and m=1 buys
   nothing structurally except the FHRR cos-phase kernel + hash encoding. Keep PLAN as-is;
   a later experiment can try m>1 with n=m (short ctx 64, D small).

4. **LM experiment (§V-E, Table III):** Wikitext2 PPL 29.16±0.13 → 27.54 (−5.47%),
   PTB 94.78 → 91.51 (best); small models, mean of 5 runs. **Fixed random PE ≈ trainable PE**
   (negligible difference) but PEs must **vary across Q/K/V and heads** (+2.56% PTB).
   **[G2]** use fixed random per-head/per-QKV position matrices — saves params and a
   gradient path. Encoders: **W trainable, Λ fixed.** Improvement is small and dataset-small;
   paste's "improved vs vanilla" claim = true but within ~5%, so our ±10%/±20% G2 gate is
   calibrated right (we bet GHRR lands near parity on TinyStories).

5. **Cost (§IV-F):** binding O(D·m³) vs FHRR O(D·m²) — extra factor m. Trivial at m=1,
   quadratic ctx attention at m=n; irrelevant for G2 sizes on a 3090.

6. **Diagonality knob** = Σ|Q_jj| / Σ_{j,k}|Q_jk|: 1.0 → commutative FHRR-like, 0.0 → max
   non-commutativity; interpolates decoding perf between HRR-ish and TPP-ish (§V-C).
   Cheap G2 ablation axis if GHRR underperforms baseline: try diagonality sweep at m=2–3.

## HoloVec oracle API (v1.0.2, verified installed; repo branch is `master` not `main`)

```python
from holovec import VSA
m = VSA.create("GHRR", dim=256, matrix_size=3, diagonality=0.0, seed=42)  # or FHRR/MAP/HRR/VTB/BSC/BSDC
a = m.random(seed=1); b = m.random(seed=2)
m.bind(a, b); m.unbind(m.bind(a,b), b)     # exact recovery (conjugate-transpose inverse)
m.bundle([...]); m.similarity(x, y)        # bundling sums + re-unitarizes via polar/SVD
from holovec.retrieval import Codebook, ItemStore   # fit(dict name->hv), query(hv, k) -> [(name, sim)]
from holovec.encoders import FractionalPowerEncoder # bandwidth/min_val/max_val/seed
```
NumPy is the release-blocking backend; torch/JAX are supported optional backends (cuda OK).
`m=1` GHRR == FHRR confirmed empirically (commutativity sim 1.0, see g0_verify.py).

## G0 measured (2026-11-02)

`S:/nativehdc/.venv/Scripts/python.exe src/g0_verify.py` → exit 0:
torch 2.14.1+cu126 on RTX 3090; GHRR m=3 rec_sim 1.000000, nc_sim 0.1606 (threshold <0.3),
m=1 commutes (sim 1.0). Env is ready for G1.
