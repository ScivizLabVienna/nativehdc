# G14 literature scan — round 1 (2026-11-08)

Tiered strategy per campaign doctrine; ≤6 searches used (3 queries + API verification). All cited ids below were **verified by exact-title match** against `export.arxiv.org/api/query?id_list=...` on 2026-11-08. Search engine served via keyless fallback (configured ddgs backend down — noted, not blocking).

## Verified anchors (citable)

| id | title | relevance / distinction |
|---|---|---|
| arXiv:2301.10352v2 (Clarkson, Ubaru, Yang 2023) | Capacity Analysis of Vector Symbolic Architectures | THE capacity theory hook: representation-capacity bounds for MAP-I/B + VSAs≡sketching/Bloom-filters connection. They bound dimensions needed for set tasks on **flat ensembles**; we measure recall at a **query interface** under fixed D and show chart-local partitioning dominates flat bundling by 1-2 orders of magnitude — orthogonal axes, no overlap claimed. |
| arXiv:2305.16873v1 = Neural Comput. 2023, doi 10.1162/neco_a_01590 (Kleyko, Bybee, Huang, Kymn, Olshausen, Frady, Sommer) | Efficient Decoding of Compositional Structure in Holistic Representations | Interference-cancellation / compressed-sensing decoding raises flat-ensemble bits/dim bounds (1.20→1.40; 0.60→1.26). **Must address:** their decoder is global and linear-algebraic; our partitioning argument concerns addressing cost per query (top-1 within degree-sized chart vs whole-memory scan) — different resource being optimized; flat decoders still O(D·K) per query while chart lookup is O(deg). |
| arXiv:2609.30634v1 (Ravulapalli, Chadha 2026-09) | In-Context Binding Capacity in Language Models | K50=cN^α binding-capacity curve for LMs; "interference lowers measured capacity by reducing single-binding recall". **Strongest adjacent collision** — same family of curves, but LLM-internal attention binding, not explicit VSA memory structure. Cite as parallel-evidence; our charts-vs-flat axis is absent there. |
| arXiv:2512.14709v1 (Dhayalkar 2025-12, AAAI'26 WS under review) | Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning | VSA-lens framing + failure modes (variable confusion); position paper, no measurements — cite for framing only. |

## Candidates found but NOT yet id-verified (do not cite until verified)
- HDM/ACT-R adaptation, arXiv:2508.15630 — chunk retrieval needs slot enumeration ⇒ supports our addressing-budget framing; verify before use.
- MeMo layered associative memories, arXiv:2502.12851 — layered AM memorization, no capacity curves; verify scope claim first.
- Hypertokens/HDRAM, arXiv:2507.00002 — ECC+holobasis K:V in latent space; adjacent only.
- Rel-SAR diverse relation representations, arXiv:2501.11896 — relations as VSA vectors (their relations are *content*; ours are *indices* + stalk maps) — disambiguate explicitly if cited.
- Hash-sharding practice: consistent-hashing/Maglev/AnchorHash benchmark arXiv:2211.00741; LSM survey arXiv:2507.09642 — systems-side partitioning literature; transfer is analogy-level (uniform hashing vs our phasor-addressed charts), one sentence max.

## Positioning statement for paper #2 draft (from verified set)
Prior VSA capacity work characterizes **what a single holistic ensemble can hold** (capacity bounds 2301.10352; decoder-side interference cancellation 2305.16873). LM binding-capacity curves (2609.30634) show the same interference phenomenology at the attention interface. We ask the complementary **architectural** question at fixed dimension and total stored items: does *where phasors live* (flat sum vs source-indexed charts) matter more than *how memories are composed* (relation tags, stalk maps)? Answer: yes, 1-2 orders of magnitude; relation-tag and per-edge-map costs are secondary/structural-null at m=1.

## Round 2 (same day, 2026-11-08)
All five former candidates **id-verified exact-title** via `export.arxiv.org/api/query?id_list=`: 2508.15630 (HDM/ACT-R), 2502.12851 (MeMo, doi 10.18653/v1/2025.findings-acl.785), 2507.00002 (Hypertokens, QNLP'25), 2501.11896 (Rel-SAR), 2211.00741 (hashing benchmark). All citable now.
FAR lens A (sheaf/CRDT reconciliation) — productive: **arXiv:2503.02556v1** "A Sheaf-Theoretic Characterization of Tasks in Distributed Systems" (verified; task sheaf, solvable tasks = global sections, cohomology = obstructions) and **arXiv:1303.3255v1** "Sheaves, Cosheaves and Applications" (Ghrist–Šťatunov–Zanin survey, verified; gluing axiom as limit/colimit). Transfer in one sentence: distributed-computing sheaves use *consistency of local sections* as the solvability criterion; our G11/G12 result is the empirical counterpart for memory: when charts already store per-source readouts, the consistency (restriction-map) layer adds **zero** recall — structure pays only where addressing pays.
FAR lens B (compiler symbol tables / register allocation) — **dud**: hits are textbook/blog material (ChrisOS chapter, lecture PDFs), nothing arXiv-citable; dropped honestly rather than force-fit.
## Verified corpus for Related Work (final list)
2301.10352, 2305.16873(+doi neco_a_01590), 2609.30634, 2512.14709, 2508.15630, 2502.12851(+ACL doi), 2507.00002, 2501.11896, 2211.00741, 2503.02556, 1303.3255 — eleven verified ids. Lit-scan rounds 1-2 CLOSED; round 3 optional only if draft reviewers flag a missing cluster.
