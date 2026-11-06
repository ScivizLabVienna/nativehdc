# arXiv submission dossier (ready whenever)

Title: GHRR Phase-Kernel Attention at TinyLM Scale: a Gate-Disciplined Bake-off on TinyStories
Authors: Experimental Agentic AI under human guidance (ScivizLab, University of Applied Arts Vienna)
Primary category: cs.CL  Cross: cs.AI, cs.LG
License choice on arXiv form: CC BY 4.0
PDF to upload: report/nativehdc-whitepaper-2026-11.pdf (arXiv accepts PDF-only submissions; TeX not installed here)
Repo for the "code availability" note: https://github.com/ScivizLabVienna/nativehdc

Abstract (validated numbers, no spin):
We report a pre-registered, gate-disciplined evaluation of GHRR/FHRR phase-kernel attention in a
36 M-parameter language model trained on 181.8 M TinyStories tokens, benchmarked against the same
architecture with standard attention. Under identical budgets the phase-kernel variant reached
validation perplexity 5.966 against the baseline's achieved 5.511 (+8.3%, within our pre-registered
+-10% keep rule). Geometry ablations were seed-swept: shrinking the hypervisor dimension 4x costs no
perplexity (equal-or-better on 3 paired seeds, -44% parameters) but halves usable associative capacity
(bundled recall 1.00 -> 0.54 at K=256); raising score temperature fourfold improved perplexity a
cumulative 2.7%, winning all three paired seeds. The champion (tau=16, D=1024) averages 5.809 +-
0.014 over three seeds - still behind tuned vanilla, which we state as the headline. Associative
operations attached to the same lattice support answerability refusal and document citation (5/5),
while multi-hop downstream transfer is null. All accept/reject rules were fixed before measurement;
null and marginal results are reported without promotion.

Blockers when ready: arXiv account (email verification) + possible endorsement for first-time cs.CL posters.
