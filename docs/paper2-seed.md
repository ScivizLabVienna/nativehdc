# Paper #2 seed — "Partition over Gluing: capacity structure in FHRR multi-hop recall" (draft skeleton, 2026-11-08)

Status: experiments G10/G11/G12 done at D=1024; writing NOT started; this file is the claims/evidence ledger, not prose.

## Claims (evidence-linked)
C1 — Partitioning dominates flat bundling at saturation: chart-local (per-source) storage holds 1.000 depth-2 recall to K=512 while the flat global bundle collapses along the known pose-recall knee (1.0/0.5/0.07-0.1 @64/256/512); delta +0.37..+0.93 abs, 9/9 cells >= gate 0.30 (G12; G11 shows the effect at rel-pool granularity +0.23..+0.62 despite intra-cell aliasing).
C2 — Diagonal restriction ("gluing") maps contribute EXACTLY nothing measurable: |SHEAF-SHARD| = 0.000 (G12) / -0.007 (G11); predicted by m=1 FHRR rotation commutativity; glue axiom itself holds (sim ~1.00 vs floor ~0.45) so the machinery works, it just doesn't pay at stalk size 1.
C3 — Operating granularities have floors: flat = interference knee (~D/2lnD unique keys); shared-rel-pool charts = Poisson(deg/n_rel>0) key-aliasing ceiling (G11 residual ~0.65-0.79); unique-tag charts = no floor in tested range (G12 all 1.000, deg<=dmax<<knee).
C4 — Addressing budget is matched, not smuggled: every arm's query exposes (source, hop tag) identically; G12d sensitivity control: wrong tag moves correct-recovery from 1.000 to ~0.2 (>= 0.687 gap per cell), chance floor ~0.1-0.2 quantified.

## Required before drafting
X1 D-sweep {512,1024,4096} for the partitioning curve; X2 rel-pool granularity continuum (nrel in {2,8,32,K}) mapping the aliasing continuum against C3 theory; X3 one learned-R_e positive-control attempt (expect negative per C2; if it wins, C2 weakens and paper reframes); X4 hop-depth 3 spot-check at K=512.
## Framing hooks (verify ids before citing!)
HDC capacity/binding literature (Kleyko et al.); sharded/partitioned index practice (LSM, Plateau/ACR-style collision management) as the systems analogy; "sheaf memory" formal-lineage claim must cite carefully or stay out of the title.
## Honest limits to state up front
Synthetic random graphs; entity vectors are given hashes (no semantic encoder axis tested here — that's E1/G6 territory); charts assume known edge inventory per source (graph-structured memory, not free-lance retrieval).
