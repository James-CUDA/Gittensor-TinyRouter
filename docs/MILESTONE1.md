# Milestone 1 — Prompt triage (domain + difficulty)

Miners submit **only a tiny head + weights**. The host runs the frozen
Qwen3-0.6B encoder. Default head architecture is TinyRouter’s
`TriageHead` (optional `AttentivePool`). You design / train your head however
you like; the submission is weights that fit the I/O contract.

| | |
| --- | --- |
| Dataset | [James-Cuda/tinyrouter-m1](https://huggingface.co/datasets/James-Cuda/tinyrouter-m1) |
| Configs | `5-domain` · `20-domain` (GCI-Bench topics) |
| Schema | `id \| domain \| difficulty \| prompt` |
| Encoder VRAM | ~2–4 GB (`bfloat16`); RTX 3090 is enough; on `trinity-gpu` use **GPU 5** |

## Workflow

```
  miner                          validator (host)
  ─────                          ────────────────
  train head  →  pack weights  →  gates (<1M params, 1/day)
                               →  score KING on fixed test data
                               →  score CHALLENGER on same data
                               →  if challenger ≥ king + 0.02 → MERGE (new king)
                                  else REJECT
```

1. **Train** — optimize domain + difficulty on `tinyrouter-m1`
   (`5-domain` or `20-domain`). Freeze the encoder locally.
2. **Pack** — weights only (no encoder checkpoint):

   ```bash
   python scripts/pack_milestone1.py \
     --miner-name alice --config 5-domain --pool penultimate \
     --weights experiments/m1/head.npz
   ```

3. **Preflight** (offline, no GPU required for gates):

   ```bash
   python scripts/preflight_milestone1.py \
     --submission submissions/alice/m1 --miner-name alice
   ```

4. **Validator** (host — king vs challenger):

   ```bash
   export CUDA_VISIBLE_DEVICES=5   # trinity-gpu only
   python scripts/validate_milestone1.py \
     --challenger submissions/bob/m1 \
     --miner-name bob \
     --config 5-domain \
     --device cuda:0 \
     --features experiments/m1/test_features.npy \
     --promote
   ```

   - Same fixed test split for king and challenger.
   - **MERGE** if `challenger.composite >= king.composite + 0.02` (or no king yet).
   - `--promote` writes `submissions/m1_leaderboard_<config>.json`.

Self-check without promoting:

```bash
python scripts/eval_milestone1.py \
  --submission submissions/alice/m1 --config 5-domain --baseline majority
```

## Rules

| Rule | Limit |
| --- | --- |
| What you submit | Head + weights only (not the 0.6B encoder) |
| Default arch | `TriageHead` (± attentive pool) |
| **Size** | **under 1,000,000** trainable parameters |
| **Rate** | **1 submission per miner per day** |
| **Win margin** | Challenger ≥ king composite + **0.02** |
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

## Code map

| Path | Role |
| --- | --- |
| `src/trinity/coordinator/triage_head.py` | Default `TriageHead5Domain` / `TriageHead20Domain` |
| `src/trinity/coordinator/attention_pool.py` | Optional encode → attention → head |
| `src/trinity/m1/` | Pack, gates, metrics, leaderboard, scoring |
| `scripts/build_tinyrouter_m1_slim.py` | Rebuild Hub `5-domain` / `20-domain` |
| `scripts/legacy/` | Older `router-bench` Hub one-shots |

## Related

- Live TAO track (model × role): [`SUBMITTING.md`](../SUBMITTING.md)
- Repo overview: [`README.md`](../README.md)
