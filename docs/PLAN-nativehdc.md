# PLAN: native HDC local model + harness (from binpaste.xyz/nativehdc)

Source paste: 26.6 KB implementation guide "HDC-native local model + agent harness"
(GHRR attention inside + Hillock/HDC-Harness outside). Fetched 2026-11-01 via
Firebase RTDB: `pastebin-9967b-default-rtdb.../pastes/nativehdc.json` (SPA hides it; web_extract fails, curl the .json).

## Research verdict (verified today)

| Paste claim | Reality |
|---|---|
| GHRR arXiv 2405.09689 "2026" | **2024** (Yeung/Zou/Imani, UCI+West Point). v2 confirms: replacing transformer attention with the GHRR equivalent *improved* LM results vs vanilla — but only on a small experiment. Risk R1 in paste is real. |
| "copy reference impl if available" | Official UCI code not found; **HoloVec** (Twistient/HoloVec, v1.0.2, `pip install holovec`) implements GHRR (`VSA.create('GHRR', dim, matrix_size, diagonality)`), Apache-2.0, numpy/torch/jax backends incl. cuda. Use as oracle for bind/unbind/similarity cross-checks — not for the training path (need differentiable custom module). |
| Hillock = memory layer | roandejager/Hillock, **AGPL-3.0** → process boundary only (subprocess/HTTP), ideas not code. Already tracked in NOTES-hdc-harness.md. Optional tier; a holovec ItemStore + SQLite triples replicates the *verified-facts-before-generation* gate for the MVP. |
| HDC Harness = Rust binary (#811) | Pattern already distilled in our notes (stateless subprocess tool). cargo 1.99 present; building it is optional polish — a Python gate over holovec covers detect_adversarial/abstain for smoke purposes. |

## Machine fit (checked today)

- GPU: RTX 3090 24 GB, driver 617 — substitutes the guide's "1× RTX 4090" (~1.5× slower; 10M model ≈ 18–24 h on 1B tokens). $0 training possible locally; no cloud needed.
- RAM/disk: 127 GB RAM, D: 1.8 TB free — every RAM/VRAM target in the guide is trivially met.
- **No CUDA torch anywhere** (crisp-env/cv2-env import-fail; LongSplat venv dead-broken). Python 3.13 + pip OK, creating-intelligence 0.1.0a10 + numpy present, HF token at S:/hermes/secrets/hf-token.key.txt.
- Gap = one fresh venv with torch cu12x wheels (~2.5 GB download).

## Plan (4 gates, each independently verifiable; kill at first red)

**G0 — env (~30 min).** `uv venv D:/AI/nativehdc/.venv` on py313; `pip install torch --index-url https://download.pytorch.org/whl/cu126 holovec datasets tokenizers`. Verify: `torch.cuda.is_available()` True + GHRR bind/unbind exact-recovery assert in holovec.

**G1 — differentiable GHRR-attention module (day 1).** Implement m=1 scalar path (= FHRR kernel: score = Re(z_q·conj(z_k)) = cos(θq−θk) summed over D) as an nn.Module (~50 lines, torch.complex, no FFT needed for m=1); token→HV via **seeded hash encoding** (token_id → phases via hashlib/blake2b PRNG — never builtin `hash()`, PYTHONHASHSEED rule from our CI protocol notes). Unit test: cross-check similarity/bind/unbind against holovec GHRR within float tolerance; non-commutativity assert for the m>1 mode behind a flag.

**G2 — 10M-param bake-off (≈1 GPU-day + idle).** Decoder-only, 6 layers × d_model 384, D=4096 HV dim, softmax kept (per paste: binarize only in Phase 4). Data: TinyStories subset via HF token, GPT-2 BPE tokenizer (lazy; skip training a 32K BPE). Train GHRR-attn **and** vanilla baseline same seed/params/budget (start 300M tokens; extend to 1B only if delta ok). Gate: perplexity within 10% of baseline → proceed; >20% worse → debug phase-handling/binding-direction per paste before rerun.

**G3 — harness (~1 evening).** 50-line Python loop, observability-first (one JSONL trace line per step, per risk R3): detect_adversarial (holovec n-gram HV match vs ~200 injection exemplars) → facts gate (SQLite triples + holovec ItemStore; refusal = system-level `Answerable: no` protocol from our #536 notes) → generate → abstain check. Harness *always* runs the gates; the 10M model never calls tools (paste option 2.4c).

**G4 — paste's 7-point smoke test.** Grounded answer cites doc; ungrounded question refuses; injection never reaches the model. All 7 pass → the "first HDC-native local model with harness" MVP exists at ~10M params.

Deferred (not skipped-by-accident): Hillock-as-sidecar, Rust hdc_harness build, softmax removal, THDC binarization, tool-calling fine-tune, 300M scale-up, FHRR v2 phasors — all post-MVP per paste's Phase 4–6 and our existing skip list.

## Costs
$0: TinyStories + GPT-2 tokenizer via HF token (Pro $2/mo already running), everything else local/Apache-licensed. Only electricity (~1–2 GPU-days total for MVP).

## Timeline vs paste
Paste says "2 weeks on a 4090". On this box: **~3 working days** wall time (G0+G1 day 1, G2 day 2 incl. overnight training run, G3+G4 day 3), assuming the G2 gate goes green first try.
