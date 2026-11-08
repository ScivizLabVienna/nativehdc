# G13 — capacity-curve sweeps for C1/C3 (X1+X2) + structural-map check (X3) + hop-3 spot (X4). CPU $0, 2026-11-08.

Operator go = "continue". Evaluator frozen: everything via module globals of g11_sheaf
(D, NREL patched at call time; R recipe identical to G11/G12 main -> anchor cells must
reproduce). No prose changes to build/run.

## X1 (gated): unique-tag charts vs flat, K=512 fixed, D in {512, 1024, 4096}, seeds 7/13/1337
Prediction: SHEAF=SHARD >= 0.97 every cell (chart item count <= dmax << knee for all D).
FLAT(tagged) follows unique-key knee ~ D/(2 ln D): 512->~0.00-0.10, 1024->0.07-0.10 (known),
4096->0.25-0.55. Delta chart-flat > 0 in every cell; delta shrinks as D grows.
Gates: G13a charts>=0.97 all cells; G13b flat non-increasing then improving pattern OR monotone
increase with D on >=2/3 seeds (shape check, tolerant); G13c min delta > 0.

## X2 (gated): rel-pool granularity continuum, K=512, D=1024, nrel in {2,8,32} (+unique anchor), 3 seeds
Prediction (aliasing model lambda = avg_deg/nrel ~ 4/nrel; wrong-node via same-cell duplicate targets):
nrel=2 ~0.3-0.5; nrel=8 ~0.5-0.8 (anchor band vs G11 emitted K=512 sheaf cells); nrel=32 ~0.85-1.0; unique=1.0 (X1).
Flat unchanged ~0.04-0.10 (independent of nrel: global superposition dominates).
Gates: G13d SHEAF(recall) non-decreasing in nrel per seed (tolerance -0.05); G13e anchor cell nrel=8 inside band; G13f flat < 0.15 all cells.

## X3 (exploratory, NO gate): identity-map arm vs random-phase maps, unique tags, K=512, D=1024
Structural argument: sheaf storage tag*R_e*val is queried with paired conj(R_e), so ANY per-edge
map family cancels exactly at stalk size 1 -> prediction accuracy IDENTICAL to shard (<=0.005 diff).
Learned maps matter only for shared stalks (overlap m>1) -- out of scope here, stated in seed file.

## X4 (exploratory, NO gate): depth-3 chains, unique tags, K=512, D=1024, sheaf vs flat
Prediction: sheaf >= 0.85 (clean per-chart lookups compose), flat <= 0.03 (third cleanup on noise).
## Response rule (fixed pre-run)
Gated failure -> NULL/NEEDS-INV table recorded, HOLD; do not re-shape gates post-measurement.
Exploratory cells reported with numbers only. All PASS -> C1/C3 curves final in paper2-seed ledger.
