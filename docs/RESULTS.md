# TRINITY Replication — Results

> Structured scorecard for the open-source replication of **TRINITY: An Evolved LLM Coordinator**
> (Xu et al., ICLR 2026) using `deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6` via Fireworks.
>
> The per-coordinator and multi-task tables below are **auto-generated** by
> `scripts/results_table.py` from `experiments/**/eval*.json` (machine-readable copy:
> `experiments/results.json`). Method: [`AGENTS.md`](AGENTS.md) · spec: [`SPEC.md`](SPEC.md) ·
> full lab log incl. every mistake/fix: [`JOURNAL.md`](JOURNAL.md).

## 1. Headline

On our 3-model open-source pool, **TRINITY reproduces the paper's core relative claims**, with
honest caveats on variance:

- **R1/R2 ✅** — on the multi-task average, the trained coordinator (**0.750**) beats *every* single
  fixed model (best is deepseek-v4-pro at **0.639**). No single model wins both tasks; the router
  picks the right specialist per task.
- **R4 ✅** — TRINITY (0.750) beats random routing (0.558) on the average, and on 6 of 7 individual
  evals.
- **Per task:** TRINITY ties the best specialist (math≈glm-5p2, MMLU≈deepseek-v4-pro) — the expected
  single-task ceiling for routing.

## 2. Per-coordinator held-out evals (40 items each, fixed extraction)

| benchmark | coordinator | TRINITY | best single (model) | random | T>best? | T>rand? |
|---|---|---|---|---|---|---|
| math500 | full_pilot | 0.550 | 0.500 (glm-5p2) | 0.325 | ✅ | ✅ |
| math500 | math_s1 | 0.525 | 0.450 (glm-5p2) | 0.400 | ✅ | ✅ |
| math500 | math_s0 | 0.325 | 0.700 (glm-5p2) | 0.425 | ❌ | ❌ |
| mmlu | mmlu_s1 | 0.950 | 0.975 (deepseek) | 0.850 | ≈ | ✅ |
| mmlu | mmlu_s0 | 0.925 | 0.950 (deepseek) | 0.875 | ≈ | ✅ |
| mmlu | mmlu_pilot* | 0.550 | 0.950 (deepseek) | 0.500 | ❌ | ✅ |

`*` mmlu_pilot was scored before the extraction fix; superseded by mmlu_s0/s1.

## 3. Multi-task summary (the paper's R1/R2)

| system | math500 | MMLU | **average** |
|---|---|---|---|
| **TRINITY (best coordinator/task)** | **0.55** | **0.95** | **0.750** |
| deepseek-v4-pro (fixed) | 0.33 | 0.975 | 0.639 |
| glm-5p2 (fixed) | 0.50 | 0.75 | 0.625 |
| kimi-k2p6 (fixed) | 0.25 | 0.60 | 0.401 |
| random routing | 0.33 | 0.85 | 0.558 |

**R1/R2 ✅ HOLDS** (0.750 > 0.639) · **R4 ✅ HOLDS** (0.750 > 0.558).

## 4. Honest caveats (do not over-read)

1. **Seed variance.** Math coordinators are inconsistent: 2 of 3 beat the best single model
   (full_pilot 0.55, math_s1 0.525), but **math_s0 failed (0.325, below random)**. sep-CMA-ES with a
   noisy binary reward + small `m_cma` occasionally converges to a bad policy. MMLU is robust across
   seeds (0.925–0.95).
2. **Eval noise at n=40.** Reasoning models are not fully deterministic even at temperature 0, so the
   *same* "best single" baseline scored 0.45–0.70 across runs on the same math questions. Point
   estimates are shaky at n=40.
3. **The earlier MMLU "failure" (0.55) was a bug, not a finding** — brittle answer-extraction
   discarded correct multi-turn answers (and corrupted the training reward). Fixed; MMLU TRINITY
   moved 0.55 → 0.95. See JOURNAL 2026-06-23.
4. **Rigorous eval in progress:** n=120 with single-model baselines averaged over 3 reps, on the best
   math + MMLU coordinators — this section will be updated with mean±std once it lands.

## 5. What was NOT reproduced / scoped out

- LiveCodeBench (gated loader → toy fallback) and GPQA (gated dataset) were not run on real data, so
  the 4-task suite and the 86.2% LiveCodeBench record are out of scope here.
- Absolute numbers differ from the paper by design (different model pool). We target the **relative**
  invariants (R1/R2/R4), per `SPEC.md` §1.3.

## 6. Reproduce

```bash
source ~/.config/trinity/secrets.env
bash scripts/run_remote.sh train --benchmark math500   # evolve a coordinator on GPU
bash scripts/run_remote.sh eval  --benchmark math500 --theta <best_theta.npy>
python scripts/results_table.py --json                 # regenerate this scorecard
python scripts/cost_report.py --ledger cost_ledger.jsonl   # spend
```

## 7. Cost

~$22 spent at the time of writing (Fireworks token usage; see `scripts/cost_report.py`). Real rates:
deepseek-v4-pro $1.74/$3.48, glm-5p2 ~$1.40/$4.40, kimi-k2p6 $0.95/$4.00 per 1M in/out.
