# Roadmap

TinyRouter is a routing accuracy competition. Miners compete to train the best
coordinator head. This roadmap shows where the competition is headed.

## Competition benchmarks

| Benchmark | Current Best | Oracle Ceiling | Headroom |
|---|---|---|---|
| math500 | [see leaderboard.json](leaderboard.json) | 0.856 | +0.049 |
| MMLU | [see leaderboard.json](leaderboard.json) | ≥0.939 | near-ceiling |

## Now (active competition)

- **math500** — beat the current best routing accuracy
- **MMLU** — near ceiling (deepseek dominates), but routing can still improve cost-efficiency
- **Infrastructure** — pr_eval.py, leaderboard, anti-cheat checks

## Next (opening soon)

- **AIME benchmark** — harder math problems where routing matters more (higher variance across models)
- **GPQA Diamond** — harder knowledge benchmark
- **Cost-aware scoring** — reward heads that achieve the same accuracy with fewer API calls
- **Multi-benchmark combined score** — best average across math + knowledge + code

## Later (planned)

- **GRPO trainer support** — submit heads trained with GRPO (currently in `src/trinity/fugu/`)
- **Live API endpoint** — serve the current best head as a public routing API
- **Multi-pool competition** — route across different model providers
- **Community-voted pool** — let miners vote on which models to add/remove from the pool

## How to propose new directions

Open a PR against this file with your proposal. The maintainer reviews and merges
if it aligns with the competition's goals. These PRs are general contributions
(no TAO reward) but shape the future of the project.

## Maintainer's priority list

1. Keep the hidden benchmark secure and un-leaked
2. Run pr_eval.py on every submission PR within 48 hours
3. Keep the pool models current (update when Fireworks adds/removes models)
4. Expand benchmarks to increase routing headroom (harder = more room to improve)
5. Improve anti-cheat defenses as miners get more sophisticated
