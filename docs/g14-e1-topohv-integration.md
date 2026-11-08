# G14 — E1 topology-HV encoder: champion integration record (P3). External campaign, $0 CPU, 2026-11-08.

Cross-repo note: produced by the E1 series in S:/research/e1-topology-hv (rounds 1-9;
findings.md carries all tables/commands). No nativehdc code changed here — this is the
integration SPEC + accepted-champion record, written per P3 after delegated operator accept.

## Champion change (measured, pre-registered gate)
Encoder proposal `lm+tok` replaces pure-id ghrr encoding for navigation memory items:
item HV = exp(i·arg Σ_b bin(b)) over deterministic content bins b of the item text —
landmark-keyed persistence merge bins (top-15 sentence-embedding kNN clouds, sigma=0.3)
+ hard top-idf token bins; bins hashed to full-dim unit phasors via blake2b (same seam as
hash_phasors). Frozen g10 bind/unbind/pose_recall algebra untouched; encoder is a drop-in
at every hash_phasors(ids, D) call site.

Gate (pre-registered P2, task_plan.md) — PASSED 2026-11-08 r8 via frozen-algebra replica
(validated mirror .7988/.3418; zero harness edits), paired seeds {7,13,1337}, K{512,1024,2048},
N=2935 g4-docs sentences:
- recall vs hash-ids: 0.776/0.345/0.087 vs 0.807/0.354/0.084 → worst −3.1 pts = −3.9% rel (≥ −5%)
- separation AUC@D4096: 0.695 vs whole-signature control 0.508 (strict >; gate ≥ .62)
- encoder byte-deterministic on double-encode (G1 champion requirement holds)
Accept posture: operator reply "continue the way you think is best" = delegated accept;
recorded as such in campaign gate log (not a verbatim "accept").

## Why this matters for nativehdc
Id-hash HVs are capacity-optimal but similarity-blind (flat cross-item sim; g10/phase4 demo
retrieval cannot rank by meaning). lm+tok buys graded retrieval (next-sentence AUC ~.7) at a
small, capacity-gated cost — consistent with the measured tradeoff law of P1 (shared content
mass → quasi-orthogonality 0.0115 vs 0.0092 at D=4096, degrading with K).

## Known caveats / non-goals
- NRFF sum/sqrt(n) normalization is a NO-OP post-summation here: exp(i·arg) is scale-invariant;
  only per-bin normalization or no-projection variants could change anything.
- Corpus statistics (df/idf, top-k neighbors) enter the encoder: recompute bins on corpus rebuild;
  bin lists are pure functions of (text, corpus index) — deterministic cache key = item sig + corpus hash.
- ortho nonuniformity grows with shared-token density; at D=1024 monitor before adoption
  (P1 curve: separation rises AND recall-margin shrinks vs id-hash as K→knee).
- Integration code (wiring rf_encode into g10 harness entry) is an authorized nativehdc edit
  to be taken in a separate G-step with its own regression gate; not done here.

## Cross-cites
- arXiv:2605.16785 — original topology-derived HV proposal motivating E1 (P0 planner reference).
- Campaign evidence: S:/research/e1-topology-hv/findings.md sections "P2b FULL GATE MEASUREMENT"
  and per-round lit-scan records; probe = p2b_p2gate_probe.py (exit 0, 82 s).
