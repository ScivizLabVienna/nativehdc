# nativehdc — HDC-native full stack (model + harness + tools)

**Project:** long-term R&D track from today's Hermes stack to an HDC-native equivalent.
Phases: **A** HDC-inside-Hermes router/memory server → **B** GHRR-LM + CI-kernel-planner mini-agent → **C** full native claim (measurements, not a leap).

## Bootstrap for a fresh session (resume procedure)
1. Read `docs/PLAN-nativehdc.md` (verified machine facts + G0–G4 MVP gates, 2026-11-01).
2. Load skill `hdc-associative-memory-work` (CI kernel protocol rules, memory-budget discipline).
3. Check `STATUS.md` for current phase/step; continue from the first unchecked item there.

## Layout
- `docs/` — plans + research notes (this repo is git-tracked; agent-sleep snapshots give rollback)
- `src/` — modules: ghrr_attn, lm, harness, hdc_router (stage A)
- `runs/` — training runs, eval outputs (gitignored)
- `data/` — TinyStories + tokenizer caches (gitignored)
- venv: `.venv` here (`uv venv --python 3.13`, torch cu126 from download.pytorch.org)

## Standing rules (do not re-litigate)
- $0 spend; HF token at `S:/hermes/secrets/hf-token.key.txt`; GPU = local 3090 only, train overnight, never block daytime work >~4 h.
- Determinism: never key persisted hypervectors by builtin `hash()` (PYTHONHASHSEED); use hashlib/flyhash.
- Every gate ends in named asserts + `exit 0` = pass; reproduce commands go in STATUS.md lines.
- Kill discipline from PLAN: G2 perplexity delta >20% after one debug cycle ⇒ report "measured negative", don't grind.
- AGPL isolation: roandejager/Hillock only ever as sidecar process; HoloVec (Apache-2.0) is the licensed dependency of record.

## Phase map (detail lives in STATUS.md)
| Phase | Deliverable | Est. |
|---|---|---|
| A | HDC router/memory MCP server on :8787 pattern — skill/tool/memory selection by HV recall, zero context tokens | days after G0–G1 |
| B | 3-day MVP (PLAN G0–G4) + CI-kernel planner choosing among 5–10 tools; Qwen-27B escape-hatch ratio logged = key metric | ~2 wk |
| C | go/no-go on full native: scale G2 to 100M, CI planning capacity sweep at D≥4096, escape-hatch ratio < threshold | months, measurement-driven |
