# Partition over Gluing: Where Capacity Lives in Hyperdimensional Multi-Hop Memory (draft v0.1, 2026-11-08)

Working draft. Every number below is transcribed from the frozen ledgers (`STATUS.md`, `raw/g13_run.txt`) of pre-registered probes G10-G14; no result in this document was produced after a verdict was locked. Figures: `report2/figs/fig1_x1.png`, `fig2_x2.png`, `fig3_x4.png`.

## Abstract

Fixed-capacity vector memories built from holographic reduced representations (FHRR/GHRR at stalk size m=1) are usually analyzed as one global ensemble: capacity is a property of dimension and decoder. We test the complementary architectural question at constant dimension and constant item count: does *where phasors live* matter more than *how memories are composed*? On synthetic directed multi-relational graphs with two- and three-hop cue chains, chart-local partitioning (one bundle per source node, top-1 readout restricted to degree-sized candidate lists) dominates flat global bundling by one to two orders of magnitude at equal D and K (G12: 1.000 vs 0.07-0.10 at K=512, D=1024; G13 X1: delta +0.15 to +0.99 across D in {512, 1024, 4096}), while sheaf-style per-edge restriction ("gluing") maps contribute exactly zero beyond partitioning (G11 attribution -0.007; G12 |SHEAF-SHARD| = 0.000 at nine cells; X3: identity-map and random-phase-map arms agree to 0.0000 because paired conjugate cleanup cancels any per-edge unit-modulus map at m=1). We further locate the two distinct floors that govern chart designs - source-key aliasing as a Poisson(λ = deg/n_rel) cell-collision continuum (X2: recall 0.147-0.180 / 0.627-0.687 / 0.840-0.893 for n_rel in {2, 8, 32} with per-seed monotonicity) and the flat-ensemble interference knee - and report a correction to the D/(2 ln D) knee heuristic, which underestimates usable pair-binding recall roughly threefold at K=512, D=4096.

## 1. Claims under test

C1: at fixed (D, K), source-indexed chart partitioning strictly dominates flat bundling for depth-2 and depth-3 multi-hop recall, once per-cell addressing is unambiguous.
C2: diagonal per-edge restriction maps (the "gluing" layer of a sheaf-style memory) add no measurable recall at stalk size m=1; they are structurally null under paired-conjugate cleanup.
C3: each design has its own capacity floor and it comes from a different mechanism: aliasing (charts) vs interference (flat); the correct fix differs accordingly.
C4: charts' sensitivity to correct addressing tags is high (right-tag minus wrong-tag recall gap >= 0.5), i.e., partitioned readout genuinely uses the tag, not context residue.

## 2. Setup (frozen before runs)

Evaluator (`src/g11_sheaf.py` + `src/g12_tags.py`, complex128 FHRR: hash phasors exp(2πi·h(id)/D)-style hashes; bind = coordinate product; unbind = multiply by conjugate; top-1 real-dot cleanup). Three arms share identical item content: FLAT (one global bundle Σ h(src)·r·h(tgt)), SHARD (per-source charts, bare bundles), SHEAF (charts with per-edge target rotations R[e] applied to both stored and candidate sides). Multi-hop queries chain cues through true intermediate nodes (oracle hop convention fixed in the evaluator; identical across arms). Graphs are random regular directed graphs, 3 seeds {7, 13, 1337}, K = number of edges {64, 256, 512} unless swept. Distinguishing control (G12): per-edge unique tags r_e remove source-relation cell collisions present in G11's shared relation pool. Spec revisions rev 1c and rev 2a were documented before final runs; predictions were written into the specs before execution (audit below).

## 3. Results

### 3.1 Partitioning dominates bundling (C1)
G12 (CONFIRMED, 4/4 gates, 9/9 cells): with unique tags, SHEAF = SHARD = 1.000 at every cell; FLAT follows the interference knee 1.000 / 0.47-0.51 / 0.07-0.10 at K = 64/256/512 (D=1024), i.e., deltas +0.37 to +0.93. Depth 3 (G13 X4): chart recall remains 1.000 across ~148 chains per seed while flat collapses to 0.020-0.027 [fig3]. Dimension sweep (G13 X1, fig1): charts remain 1.000 at D in {512, 1024, 4096}; flat recovers monotonically with D (means 0.016 / 0.084 / 0.804) but never overtakes; delta > 0 in all 9 cells (G13c pass).

### 3.2 Two distinct floors (C3)
Charts degrade only through key aliasing: with a shared relation pool of size n_rel, two edges colliding in the same (source, relation) cell are unresolvable, giving a Poisson-style ceiling that moves continuously with n_rel (X2, fig2): sheaf recall 0.147-0.180 (n_rel=2), 0.627-0.687 (8), 0.840-0.893 (32), monotone per seed; flat stays below 0.10 everywhere at D=1024, i.e., its interference floor is independent of granularity. Correction: the bundled-knee heuristic D/(2 ln D) predicts <= 0.60 recall at K=512, D=4096; measured flat recall is 0.77-0.85 (G13f prediction-band miss, recorded as a calibration finding, not a verdict change: quote measured curves).

### 3.3 Gluing is exactly null at m=1 (C2)
G11 attribution |SHEAF-SHARD| = -0.007 on the collision-prone graph; G12 with tags |SHEAF-SHARD| = 0.000 at all cells; X3 shows identity-phase and random-phase map arms agree to 0.0000, which follows structurally: a per-edge unit-modulus diagonal map applied identically to stored values and candidate list cancels in the conjugate cleanup product, so no member of that map family can change top-1 order. Consequence: sheaf-style restriction layers become informative only for shared stalks (m>1 matrix bindings), which our stack measures elsewhere (GHRR m>1 program); at m=1 they are bookkeeping overhead.

### 3.4 Readout uses the tag (C4)
Tag-sensitivity gap (right-tag minus wrong-tag correct rate on the mechanism arm) >= 0.687 in every G12 cell, with wrong-tag recall near but above chance (~0.2 vs ~0.1). Per rev 2a the control is gated as a sensitivity gap rather than an absolute threshold because the FLAT arm's noise-saturated floor made absolute comparison uninformative; revision predates the verdict.

### 3.5 Prediction audit
Pre-run predictions matched measurements within ±0.03 everywhere except two documented misses, both favorable or neutral: G12 delta at K=512 overshot (+0.93 vs predicted +0.4-0.8), and the G13f flat-capacity band was too pessimistic (§3.2). No gate was reshaped after measurement.

## 4. Related work, and where this does not live

VSA capacity theory bounds what a single holistic ensemble can represent (Clarkson et al., arXiv:2301.10352; the VSA-sketching/Bloom-filter connection) and decoder-side interference cancellation raises flat-ensemble bits-per-dimension ceilings (Kleyko et al., arXiv:2305.16873 / Neural Comput. 2023). These are complementary axes: they optimize ensembles we deliberately abandon; partitioning instead reduces per-query readout to degree-sized candidate lists at the cost of addressable storage, which is what our deltas measure. Binding-capacity curves measured at language-model interfaces (arXiv:2609.30634, K50 = cN^α) show the same interference phenomenology without an explicit memory partition to compare against; framing support is available from attention-as-binding position work (arXiv:2512.14709). Holographic declarative memory adapted to ACT-R (arXiv:2508.15630) documents the practical addressing problem we abstract - slot enumeration for chunk retrieval; layered associative-memory architectures (MeMo, arXiv:2502.12851, ACL Findings 2025) pursue hierarchy rather than granularity comparisons; hypertokens (arXiv:2507.00002) address structure in latent codes rather than storage layout; Rel-SAR (arXiv:2501.11896) uses relation *contents* where our relations are collision-relevant *indices*. The distributed-systems task-sheaf program (arXiv:2503.02556) and the sheaf survey literature (arXiv:1303.3255) motivate gluing axioms as consistency criteria for local sections; our result is their empirical counterpart in memory design: when charts already store per-source readouts, the consistency layer adds zero recall. Practice-side partitioning benchmarks (consistent hashing et al., arXiv:2211.00741) inform the systems analogy at load-balancing level only.

## 5. Threats to validity

(1) Synthetic random regular graphs; no natural semantic structure, degree distribution, or query mix yet - external validity is currently mechanism-level only. (2) Complex-valued FHRR with exact hashes and top-1 cleanup against a clean candidate list; noisy channels, sparse-binary VSAs, and learned indexes are outside scope. (3) C2 is proven for the tested family (diagonal unit-modulus maps at m=1); it does not constrain non-diagonal or shared-stalk maps, which are future work. (4) Oracle hop chaining rewards a correct first hop; end-to-end self-verified chaining (refusal gating) changes absolute numbers, not the arm ordering (tested pattern in earlier POC work). (5) X1/X2 report 3 seeds without variance inflation checks at K=512 saturation (all arms pinned at 0 or 1, so ranking effects are unmeasurable there). (6) Figure values are parsed from one deterministic re-run of `src/g13_curves.py` reproducing the banked verdict line; the G11/G12 tables in §3 come from the locked ledgers, not re-runs.

## 6. Conclusion

At fixed dimension and item count, the storage layout decision (flat vs chart-local partitioning) moves multi-hop recall by one to two orders of magnitude, while the composition-decision layer studied most in VSA capacity work (binding maps, decoders on one ensemble) is second-order or exactly null at stalk size one. Capacity in this regime is an addressing property before it is a coding property.

---
*Draft bookkeeping: cite-key check pending; LaTeX conversion and Zenodo minting deferred until operator review of register/structure, per the polish-before-DOI rule.*
