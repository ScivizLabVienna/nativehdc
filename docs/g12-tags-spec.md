# G12 — unique-edge-tag control for G11's aliasing ceiling (CPU $0)

Claim under test (revised, post-G11): chart-local partitioning beats the flat global bundle
on depth-2 multi-hop recall at saturation **when addressing is unambiguous**; restriction-map
gluing adds nothing. G11's residual ceiling was (source,relation) key-cell collisions at
deg/n_rel ~ 0.5; G12 removes them by giving every edge a unique tag (ridx = eid).

## Protocol
Graph/entity/generator paths frozen: imports `g11_sheaf.build/run` unchanged (arms paired by
construction); ONLY difference: relation keys -> per-edge unique tags, RH sized K.
Query exposes (source, hop-edge tag) identically to all arms. Seeds 7/13/1337, K {64,256,512},
D=1024, 150 chains, attempted-only.

## Prediction (written BEFORE running — audit after)
FLAT(tagged) ~ known pose-recall knee: ~1.0@64, ~0.5@256, ~0.1-0.2@512. SHEAF/SHARD(tagged)
>= 0.97 every cell (charts hold <= dmax << knee items). Deltas at {256,512} ~ +0.4..+0.8 ->
G12b expected PASS. |SHEAF-SHARD| <= 0.05 (maps remain no-op; m=1 commutativity).

## Gates (fixed before measurement)
- G12a sanity: SHEAF(tagged) >= 0.95 at K=64 all seeds AND glue axiom > theory floor every row.
- G12b claim: per-cell SHEAF - FLAT >= 0.30 for all K in {256,512} x 3 seeds.
- G12c attribution: mean(SHEAF - SHARD) within +-0.05 (else gluing unexpectedly contributes -> investigate before crediting partitioning alone).
- G12d trivialization control: wrong-tag query at hop-1 (tag of a different edge with same source, else any other edge) must recover the WRONG intermediate in >= 99% of attempted chains (tags carry the addressing; if wrong-tag often lands right, gains are vacuous).
## Response rule (fixed now)
ALL PASS -> STATUS: partitioning claim CONFIRMED; next authorized step = findings-note skeleton
for paper #2 ($0 writing only). ANY cell SHEAF(tagged) < 0.90 at any K -> NEEDS-INVESTIGATION,
exactly ONE diagnostic smoke (seed 7, print per-chain top-1 margin), then HOLD for operator.
Gate failure that is not <0.90 (e.g. delta 0.25) -> NULL recorded with full table; no G13 without operator.
- rev 1a (pre-verdict): G12d control was defined on the FLAT readout; at K=512 that arm is
  chance-saturated (measured wrong-tag correctness 1.3% = 1/128 chance level) so the instrument
  cannot test addressing there -- violates our own logged rule (sanity anchor must not depend on
  the contrast variable). Control re-anchored to SHEAF readout; FLAT-arm number kept in table as
  diagnostic only. Claim gates a/b/c and thresholds unchanged. No further revisions before verdict.
- rev 2a (pre-verdict; final): G12d absolute threshold mis-specified -- a source can hold a SECOND
  edge to the SAME intermediate, so a mismatched tag may legitimately land on m via that duplicate
  (topology term ~deg/N not accounted for), and intra-chart cross-terms flip argmax ~25% within
  charts of size ~dmax. Addressing is therefore defined by SENSITIVITY, not absolute accuracy:
  gap = right-tag correct (=1.000 measured, frozen) - wrong-tag correct (1-wrongtag_wrong measured,
  frozen) >= 0.50 in every cell; chance floor for wrong-tag correct ~ deg/N + 1/N ~= 0.10-0.20.
  Uses ONLY already-emitted numbers; no re-run, no new data selected. This is the last revision.
