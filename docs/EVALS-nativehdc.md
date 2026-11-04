# G7 — external eval: ProofWriter/RuleTaker depth-reasoning (spec, 2026-11-04)

Chosen from the sciknoworg/awesome-neurosymbolic-ai pool (see external-landscape doc). Why this one: NL rule-chains match our binding-capacity story; published "depth wall" gives a ready-made differentiating axis for GHRR phase attention vs vanilla; text is TinyStories-level simple; zero-shot LM-native scoring possible.

Data (verified live via HF datasets-server, no auth): `tasksource/proofwriter`, single config, splits train/test/validation, 174k test rows, columns: `theory` (NL story+rules), `question`, `answer ∈ {True,False,Unknown}`, `config=depth-N` strata, `QDep/maxD`. Apache-2.0 (inherited allenai/ruletaker). Access: paginated `rows` API or one-time 42.9 MB parquet pull (`/resolve` paths once listed via `info`).

Scoring (no architecture change): conditional likelihood of the answer string given `theory + " Question: … Answer:"`, softmax over the three continuations; accuracy per depth stratum. Arms: ghrr tau8 champion (GHRR_TAU=8) vs vanilla G2 arm under identical subset/budget.

Known bias, accepted deliberately: many theories exceed ctx=64 tokens (pos table is 64, no eval-time extension without surgery). We filter to tokenized theory+question ≤ ~55 tok ("RuleTaker-Short"), stratified ~500/depth. Both arms see the identical subset — relative comparison stands; absolute numbers and depth-curves carry a short-theory selection bias. Note in any writeup.

Gate (pre-registered): G7 passes if ghrr beats vanilla on aggregate accuracy by >1pp on Short-Q AND the per-depth decay slope is shallower for ghrr; ties => report both, no promotion pressure from G7 alone.
Run cost: inference-only, minutes per arm. Script will be `src/g7_eval.py` (~60 lines) once probes land and the harness pattern is copied from g3_test.

Blocked-on: none (data verified reachable 2026-11-04). Deferred until G6 hv_dim probes + their verdict commit are done, to keep one variable per step.
