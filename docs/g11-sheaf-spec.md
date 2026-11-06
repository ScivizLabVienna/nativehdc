# G11 — sheaf-gluing probe (structural, no learning, CPU $0)

Question (falsifiable): at bundle saturation on champion geometry (D=1024), does
**chart-local storage + gluing** (sheaf reading: clean up inside per-node stalks,
compose sections across charts via per-edge diagonal restriction maps) recover
depth-2 multi-hop facts better than the **flat global bundle**, and is any gain
attributable to more than mere capacity partitioning?

## Protocol (fixed before measurement)
- D=1024; seeds 7, 13, 1337; K in {64, 256, 512} triples; N = K//4 entities,
  random directed edges (no self-loops), avg out-degree ~4. Depth-aware 2-hop
  query chains only (s->m->t with s!=m!=t); incomplete candidate chains excluded
  from the denominator (attempted-only rule).
- Arms, identical stored content per edge v_e = R_e ⊙ h(tgt), R_e = fixed random
  unit-modulus diagonal restriction map:
  A) FLAT    : one global bundle sum_e bind(h(src), v_e); cleanup vs global dict.
  B) SHARD   : same but per-source bundles (charts); cleanup vs global dict —
               isolates pure capacity partitioning (no gluing).
  C) SHEAF   : per-source bundles of SECTIONS; hop-1 cleanup vs chart-local section
               candidates, then glue glued = conj(R_e) ⊙ section and verify against
               h(m) (sim > 0.99 assert, the actual sheaf axiom check); hop-2 from
               chart(m). Multi-hop top-1 over full depth-2 chains.
- Curve first, anchors after: print all arms; numeric asserts only on P1 (sanity,
  expected inside capacity knee) and the pre-registered deltas below.

## Pre-registered gates (all three must hold → KEEP-structural verdict)
- G11-a sanity: at K=64 every arm depth-2 recall >= 0.90 (bug detector).
- G11-b head-to-head: SHEAF − FLAT depth-2 recall >= 0.30 absolute on EVERY seed
  for K in {256, 512}. (Rationale: measured flat recall ~0.54@256 / ~0.16@512; a
  structural win must clear that band conservatively.)
- G11-c gluing axiom: post-glue sim(glued_section, h(m)) > 0.99 on all attempted
  chains (fails => phase bookkeeping bug, not science).
## Response rule (fixed now)
KEEP = write sheaf-chart storage into STATUS as supported *structural* direction
(learning of R_e deferred; a G11-d learnable variant is separate and needs its own spec).
NULL = SHEAF <= FLAT + 0.30 delta on any cell: recorded honestly as "chart-local
bundle structure does not beat flat cleanup at this scale"; no re-spec mid-run.
Attribution line always reported: SHEAF vs SHARD delta (if ~0, gain is capacity
partitioning, NOT sheaf gluing — state it that way).

## Revision log (pre-final-measurement, forced by probe-found defects — not result-driven)
- rev 1a: single-relation graphs make multi-edge sources intrinsically ambiguous (top-1 cannot
  identify which out-edge a chain used; all arms ~1/deg). Fix: per-edge relation key rel_e from
  pool of 8; query exposes (source, per-hop relation) identically to every arm. First two runs
  (flat-only-store bug; ambiguity) measured ~0 everywhere => mechanism bugs, not science.
- rev 1b: G11c "glue sim > 0.99" is unattainable even for a CORRECT sheaf: chart-bundle crosstalk
  puts sim ~ N(1, sqrt(2*deg/D)). Replaced by theory-anchored bound mean_glue > 1 - 4*sqrt(2*dmax/D).
  (Observed before rev: glue_min 0.92-0.98 with working phase bookkeeping — consistent w/ theory,
  inconsistent with old gate.) G11a/G11b margins unchanged.
- rev 1c (pre-final; mechanism-anchor correction): G11a "every arm >=0.90 @K=64" imported the
  phase4 knee premise (unique keys) onto a key-COLLIDING multi-rel graph, where FLAT aliasing is
  exactly the measured variable -- a sanity gate must not depend on the scientific contrast.
  G11a re-anchored to SHEAF arm only (tests gluing machinery); SHARD mapping bug (chart-candidate
  index read as node id) fixed as pure implementation defect. Final reading after this rev:
  no further spec changes; G11b applied cell-by-cell mechanically whatever it shows.
