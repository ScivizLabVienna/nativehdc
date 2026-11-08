# Paper #2 seed — "Partition over Gluing: capacity structure in FHRR multi-hop recall" (draft skeleton, 2026-11-08)

Status: experiments G10/G11/G12 done at D=1024; writing NOT started; this file is the claims/evidence ledger, not prose.

## Claims (evidence-linked)
C1 — Partitioning dominates flat bundling at saturation: chart-local (per-source) storage holds 1.000 depth-2 recall to K=512 while the flat global bundle collapses along the known pose-recall knee (1.0/0.5/0.07-0.1 @64/256/512); delta +0.37..+0.93 abs, 9/9 cells >= gate 0.30 (G12; G11 shows the effect at rel-pool granularity +0.23..+0.62 despite intra-cell aliasing).
C2 — Diagonal restriction ("gluing") maps contribute EXACTLY nothing measurable: |SHEAF-SHARD| = 0.000 (G12) / -0.007 (G11); predicted by m=1 FHRR rotation commutativity; glue axiom itself holds (sim ~1.00 vs floor ~0.45) so the machinery works, it just doesn't pay at stalk size 1.
C3 — Operating granularities have floors (MEASURED curves, G13): flat@K=512 depth-2 recall = 0.01/0.08/0.81 at D=512/1024/4096 (heuristic knee D/2lnD UNDERestimates usable capacity ~3x at K=512 — audit finding, quote measured curve not the formula); rel-pool charts scale with pool size 0.16/0.66/0.86 for nrel=2/8/32 (aliasing continuum, G13d monotone per seed); unique-tag charts = 1.000 from D=512 up; depth-3 chains: charts 1.000 vs flat 0.02-0.03.
C4 — Addressing budget is matched, not smuggled: every arm's query exposes (source, hop tag) identically; G12d sensitivity control: wrong tag moves correct-recovery from 1.000 to ~0.2 (>= 0.687 gap per cell), chance floor ~0.1-0.2 quantified.

## Required before drafting
X1 DONE (G13: flat curve above, charts 1.000 all D; delta +0.15..+0.99). X2 DONE (nrel continuum 0.16/0.66/0.86 monotone; flat nrel-independent <0.10). X3 DONE as structural check: identity-map arm == random-phase-map arm EXACTLY (diff 0.0000) — stalk-size-1 maps cancel by paired-conj construction, so C2 holds for ALL per-edge map families at m=1; learned maps only meaningful for shared stalks (m>1), out of scope. X4 DONE: depth-3 charts 1.000 vs flat ~0.025.
Remaining before draft: lit-scan + id verification (hooks unverified!), figure generation from G10-G13 tables, one abstract/limits pass.
## Framing hooks (verify ids before citing!)
HDC capacity/binding literature (Kleyko et al.); sharded/partitioned index practice (LSM, Plateau/ACR-style collision management) as the systems analogy; "sheaf memory" formal-lineage claim must cite carefully or stay out of the title.
## Honest limits to state up front
Synthetic random graphs; entity vectors are given hashes (no semantic encoder axis tested here — that's E1/G6 territory); charts assume known edge inventory per source (graph-structured memory, not free-lance retrieval).
