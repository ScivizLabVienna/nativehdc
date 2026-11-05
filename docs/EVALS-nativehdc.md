# G7 — external eval: ProofWriter/RuleTaker depth-reasoning (spec, 2026-11-04)

Chosen from the sciknoworg/awesome-neurosymbolic-ai pool (see external-landscape doc). Why this one: NL rule-chains match our binding-capacity story; published "depth wall" gives a ready-made differentiating axis for GHRR phase attention vs vanilla; text is TinyStories-level simple; zero-shot LM-native scoring possible.

Data (verified live via HF datasets-server, no auth): `tasksource/proofwriter`, single config, splits train/test/validation, 174k test rows, columns: `theory` (NL story+rules), `question`, `answer ∈ {True,False,Unknown}`, `config=depth-N` strata, `QDep/maxD`. Apache-2.0 (inherited allenai/ruletaker). Access: paginated `rows` API or one-time 42.9 MB parquet pull (`/resolve` paths once listed via `info`).

Scoring (no architecture change): conditional likelihood of the answer string given `theory + " Question: … Answer:"`, softmax over the three continuations; accuracy per depth stratum. Arms: ghrr tau8 champion (GHRR_TAU=8) vs vanilla G2 arm under identical subset/budget.

Known bias, accepted deliberately: many theories exceed ctx=64 tokens (pos table is 64, no eval-time extension without surgery). We filter to tokenized theory+question ≤ ~55 tok ("RuleTaker-Short"), stratified ~500/depth. Both arms see the identical subset — relative comparison stands; absolute numbers and depth-curves carry a short-theory selection bias. Note in any writeup.

Gate (pre-registered): G7 passes if ghrr beats vanilla on aggregate accuracy by >1pp on Short-Q AND the per-depth decay slope is shallower for ghrr; ties => report both, no promotion pressure from G7 alone.
Run cost: inference-only, minutes per arm. Script will be `src/g7_eval.py` (~60 lines) once probes land and the harness pattern is copied from g3_test.

## Result — zero-shot run (2026-11-04, src/g7_eval.py, n=500/depth x 5 depths)
- ghrr (tau8 champ): overall **0.271**; vanilla (G2 arm): **0.266**. Both FLAT across depth-0..5. Verdict: **zero-shot discriminates nothing** at TinyStories scale — degenerate majority-class regime (constant "True" ~= 27 % base rate). +0.5 pp ghrr = noise, not the hypothesized binding advantage. Null result, reported as-is; G7-gate inconclusive by construction, not failed.
- **G7b (required follow-up)**: fine-tune both arms on proofwriter train split (~20k steps, TinyStories-mixed replay to avoid PPL catastrophic drift), then re-score test by depth. Only trained models can hit the depth-wall axis; published depth results are all finetuned too. ~1 GPU-day total for two 39M arms — queue decision with G6 verdict.
- Blocked-on: none. Data cached at data/proofwriter_test.parquet (gitignored).

## G7b Results (2026-11-04, executed)
Both arms warm-started from champions, fine-tuned 2500 steps on ProofWriter FT bins (data/proofwriter_ft, 3.5M tok), best checkpoint gated on ProofWriter val (val ppl: ghrr 2.314, vanilla 2.301).

| arm | overall | d0 | d1 | d2 | d3 | d5 |
|---|---|---|---|---|---|---|
| ghrr@t8/hv1024-FT | 0.501 | 0.540 | 0.458 | 0.496 | 0.488 | 0.524 |
| vanilla-FT | 0.499 | 0.520 | 0.460 | 0.502 | 0.500 | 0.514 |

Gate (ghrr > vanilla by >1pp AND shallower depth slope): **FAIL — null**. Delta +0.2pp, both slopes flat (sigma/stratum ~2.2pp). depth-4 absent in parquet shard 0; subset identical across arms. Interpretation: +23pp over zero-shot is domain adaptation; neither architecture reaches multi-hop regime at these arm sizes (ghrr 36M / vanilla 64M). Follow-up candidates (not queued): >=100M-param arms, or synthetic-rule FT with held-out rule sets.
