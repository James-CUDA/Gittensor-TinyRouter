# Roadmap

TinyRouter is an SN74-style routing competition. The near-term goal is not to
maximize benchmark count; it is to build a **credible, frozen, miner-facing
competition loop** around one shared router that can be evaluated fairly across
software engineering, coding, and reasoning tasks.

## Current Objective

Train and evaluate **one shared routing head** across:

- SWE-bench Verified
- LiveCodeBench v6
- MMLU-Pro

Current model pool:

- Qwen3.5-35B-A3B
- MiniMax M3
- DeepSeek V4 Flash

Current training path:

1. **TRINITY-style head on RTX 5090**
2. **Conductor-style orchestrator on H100**

## Competition Principles

Everything below follows four constraints:

- **Frozen protocol** — fixed dataset revisions, task manifests, harness versions, decode settings, and seeds
- **Verified frontier only** — PRs win by beating the current benchmark frontier, not by making claims
- **Miner simplicity** — branch, submit, auto-eval, leaderboard
- **Reproducibility first** — all promoted results need receipts, manifests, and cost/accounting

## Phase 0 — Freeze the Protocol

Before expanding the competition, lock the benchmark contract.

Deliverables:

- `docs/PROTOCOL.md`
- frozen task manifests for each benchmark
- pinned dataset revisions / commits
- pinned harness versions
- fixed scoring rules
- sealed evaluation seed

Target evaluation shape:

- `n = 100–120` hidden eval items per benchmark
- optional hidden audit set per benchmark
- equal benchmark weighting in the composite score

## Phase 1 — Benchmark Adapters

Implement and harden the current benchmark stack:

- SWE-bench Verified adapter
- LiveCodeBench v6 adapter
- MMLU-Pro adapter

Requirements:

- unified task schema
- benchmark-specific scoring
- deterministic loading
- no silent fallback from real benchmark data to toy data in competition eval

## Phase 2 — Oracle Headroom Analysis

Before training miners against the pool, measure whether routing headroom is real.

Per benchmark and for the 3-benchmark union:

- best single model
- routing oracle
- disagreement rate
- estimated headroom

Decision gate:

- if routing headroom is weak, do not expand the pool blindly
- if one model dominates a benchmark, treat that benchmark as a ceiling/efficiency case

## Phase 3 — Shared TRINITY Head

Train the first credible shared router across all three benchmarks.

Scope:

- frozen encoder
- lightweight routing head
- Thinker / Worker / Verifier roles
- multi-turn loop
- separable CMA-ES
- one head for the union, not per-benchmark specialists

Hardware:

- RTX 5090

Success criteria:

- beats the best fixed single model on the frozen composite average
- beats random routing on the same protocol
- produces reproducible held-out results

## Phase 4 — Miner PR Competition

Ship the miner-facing evaluation loop.

Needed pieces:

- submission artifact contract
- hidden eval runner
- hidden audit runner
- PR validation checks
- leaderboard update path
- anti-cheat checks

Desired miner flow:

1. train router
2. package submission artifact
3. open PR
4. auto-eval runs
5. verified result posted
6. frontier update if the PR wins

## Phase 5 — Conductor

Once the TRINITY baseline is stable, expand from routing to orchestration.

Scope:

- workflow generation
- subtask decomposition
- worker assignment
- context/access structure
- verifier/refinement patterns
- recursion only if justified by benchmark gains

Hardware:

- H100

This phase should start only after the shared TRINITY head establishes the clean baseline and the remaining oracle gap is large enough to justify the added complexity.

## Phase 6 — Benchmark Expansion

After the 3-benchmark competition is stable, expand in controlled steps.

Priority expansion list:

1. MATH 500
2. GPQA-D
3. AIME
4. BigCodeBench
5. MT-Bench
6. RLPR
7. TerminalBench 2.1
8. Long Context Reasoning
9. Humanity's Last Exam
10. SciCode
11. CharXiv Reasoning

Rule:

- add benchmarks only when the harness is pinned and the scoring contract is clear
- prefer benchmarks that increase routing headroom or stress a new orchestration ability

## Model Pool Expansion

The current pool is intentionally small. The long-term target pool is:

- Qwen3.5-35B-A3B
- MiniMax M3
- DeepSeek V4 Pro
- GLM 5.2
- Opus 4.5
- Sonnet 5
- Gemini 3.5 Pro
- Fable 5
- GPT 5.5
- Kimi 2.7 Code

Expansion rule:

- models are added for **complementarity**, not prestige
- every new model must justify itself by increasing oracle headroom or efficiency options

## Infrastructure Priorities

Near-term infrastructure work:

- evaluator robustness
- benchmark harness reproducibility
- artifact validation
- leaderboard automation
- cost tracking
- audit-set integrity

Later infrastructure work:

- stronger evaluator isolation
- richer score breakdowns
- cost-aware scoring
- latency-aware scoring
- public frontier API

## How to Propose New Directions

Open a PR against this file with a concrete proposal. Strong proposals usually include:

- the benchmark or infrastructure target
- why it increases routing headroom or competition quality
- what must be frozen or pinned
- what the maintainer burden will be

## Maintainer Priorities

1. Keep hidden eval and audit sets secure
2. Keep the benchmark protocol frozen and legible
3. Prefer verified benchmark gains over broad but unstable scope expansion
4. Make miner submission and evaluation simple
5. Expand only when the next phase is measurable and reproducible
