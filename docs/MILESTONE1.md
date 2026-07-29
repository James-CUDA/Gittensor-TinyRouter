# Milestone 1 — Prompt triage (domain + difficulty)

Miners submit **only a tiny head + weights**. The host runs the frozen
Qwen3-0.6B encoder. Default head architecture is TinyRouter’s
`TriageHead` (optional `AttentivePool`). You design / train your head however
you like; the submission is weights that fit the I/O contract.

## Workflow

```
  miner                          validator (host)
  ─────                          ────────────────
  train head  →  pack weights  →  gates (<1M, 1/day)
                               →  score KING on fixed test data
                               →  score CHALLENGER on same data
                               →  if challenger ≥ king + 0.02 → MERGE (new king)
                                  else REJECT
```

1. **Train** — optimize domain + difficulty on `James-Cuda/tinyrouter-m1`
   (`5-domain` or `20-domain`). Encoder may be frozen locally.
2. **Pack** — write weights only (no encoder checkpoint):

   ```bash
   python scripts/pack_milestone1.py \
     --miner-name alice --config 5-domain --pool penultimate \
     --weights experiments/m1/head.npz
   ```

3. **Preflight** (offline):

   ```bash
   python scripts/preflight_milestone1.py \
     --submission submissions/alice/m1 --miner-name alice
   ```

4. **Validator** (host — king vs challenger):

   ```bash
   python scripts/validate_milestone1.py \
     --challenger submissions/bob/m1 \
     --miner-name bob \
     --config 5-domain \
     --features experiments/m1/test_features.npy \
     --promote
   ```

   - Picks the fixed test split (same rows for king and challenger).
   - Rescores the current **king** pack and the **challenger** pack.
   - **MERGE** if `challenger.composite >= king.composite + 0.02` (or no king yet).
   - With `--promote`, writes `submissions/m1_leaderboard_<config>.json`.

## Rules

| Rule | Limit |
| --- | --- |
| What you submit | Head + weights only (not the 0.6B encoder) |
| Default arch | `TriageHead` (± attentive pool) — same I/O as TinyRouter linear style |
| **Size** | **< 1,000,000** trainable parameters in the pack |
| **Rate** | **1 submission per miner per day** |
| **Win margin** | Challenger must beat king composite by **≥ 0.02** |
| API | None (offline) |

## Pack layout

```
submissions/<miner>/m1/
  config.json           # config, pool, d_h
  W_domain.npy          # (C, 1024)
  W_diff.npy            # (5, 1024)
  attention_query.npy   # optional (1024,) if pool=attentive
```

Default `5-domain` penultimate pack ≈ **10,240** params (≪ 1M).

## Scoring

```
composite = 0.7 × domain_accuracy + 0.3 × difficulty_exact
```

Also reported: domain macro-F1, difficulty ±1, joint accuracy.

Dataset: https://huggingface.co/datasets/James-Cuda/tinyrouter-m1
