#!/usr/bin/env python3
"""Rebuild Milestone-1 as two separate slim triage datasets.

  datasets/tinyrouter-m1/5-domain/   — 5 domains on full M1 pool
      math | code | knowledge | commonsense | instruction

  datasets/tinyrouter-m1/20-domain/  — exact Glint GCI-Bench 20 topics
      (gci_bench rows only; domain = official GCI ``topic``)

Schema (both): id | domain | difficulty | prompt
Difficulty: integer 1–5.

Optional: --push → James-Cuda/tinyrouter-m1 configs ``5-domain`` and ``20-domain``.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo

REPO = Path(__file__).resolve().parents[1]

# Official GCI-Bench topics (Glint-Research/GCI_Bench) — 20-domain taxonomy.
DOMAINS_FINE: tuple[str, ...] = (
    "agriculture",
    "archaeology",
    "architecture",
    "astronomy",
    "automotive",
    "aviation",
    "chemistry",
    "cooking",
    "energy",
    "finance",
    "gardening",
    "hardware",
    "marine",
    "music",
    "photography",
    "physiology",
    "sports",
    "textiles",
    "weather",
    "wildlife",
)
DOMAIN_FINE_SET = set(DOMAINS_FINE)

# Human-readable GCI topicLabel for docs (from Glint-Research/GCI_Bench).
GCI_TOPIC_LABEL: dict[str, str] = {
    "agriculture": "Agriculture & Crops",
    "archaeology": "Archaeology",
    "architecture": "Architecture & Construction",
    "astronomy": "Astronomy & Space",
    "automotive": "Automotive Mechanics",
    "aviation": "Aviation",
    "chemistry": "Chemistry Lab Processes",
    "cooking": "Culinary Science",
    "energy": "Renewable Energy",
    "finance": "Personal Finance",
    "gardening": "Home Gardening",
    "hardware": "Computer Hardware",
    "marine": "Marine Biology",
    "music": "Music Theory & Instruments",
    "photography": "Photography",
    "physiology": "Human Physiology",
    "sports": "Sports Training",
    "textiles": "Textile & Fashion",
    "weather": "Weather & Climate",
    "wildlife": "Wildlife & Animal Behavior",
}

COARSE_DOMAIN_NOTE: dict[str, str] = {
    "math": "GSM8K, MATH-500, AIME, AQuA-RAT, …",
    "code": "HumanEval, MBPP, BigCodeBench, …",
    "knowledge": "MMLU, MMLU-Pro, GPQA, ARC, TruthfulQA, GCI, …",
    "commonsense": "HellaSwag",
    "instruction": "Dolly-15k, IFEval",
}

DOMAINS_COARSE: tuple[str, ...] = (
    "math",
    "code",
    "knowledge",
    "commonsense",
    "instruction",
)

# Source → coarse domain
SOURCE_COARSE: dict[str, str] = {
    "aime_2024": "math",
    "aqua_rat": "math",
    "gsm8k": "math",
    "math500": "math",
    "humaneval": "code",
    "mbpp": "code",
    "bigcodebench": "code",
    "hellaswag": "commonsense",
    "dolly-15k": "instruction",
    "ifeval": "instruction",
    "truthfulqa": "knowledge",
    "arc": "knowledge",
    "mmlu": "knowledge",
    "mmlu_pro": "knowledge",
    "gpqa": "knowledge",
    "gci_bench": "knowledge",
    "supra_prompt_routing": "knowledge",
    "query_complexity": "knowledge",
}

# Fine domain_label / subject cues → coarse (when source default is weak)
COARSE_LABEL: dict[str, str] = {
    "math": "math",
    "mathematics": "math",
    "code": "code",
    "coding": "code",
    "programming": "code",
    "cs_software": "code",
    "statistics_ml": "code",
    "commonsense": "commonsense",
    "instruction": "instruction",
    "knowledge": "knowledge",
}

KEYWORD_COARSE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(math|algebra|geometry|calculus|probability|statistic)\b", re.I), "math"),
    (re.compile(r"\b(program|coding|software|algorithm|javascript|python)\b", re.I), "code"),
    (re.compile(r"\b(writing|instruction|chat|customer|education)\b", re.I), "instruction"),
]


def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", s)


def map_difficulty(x) -> int:
    try:
        v = int(float(str(x).strip()))
    except Exception:
        v = 3
    return max(1, min(5, v))


def map_coarse(source: str, domain_label: str) -> str:
    key = _norm(domain_label)
    if key in COARSE_LABEL:
        return COARSE_LABEL[key]
    if domain_label:
        for pat, dom in KEYWORD_COARSE:
            if pat.search(domain_label):
                return dom
    return SOURCE_COARSE.get(source, "knowledge")


def load_m1(path: Path | None, hub_repo: str) -> pd.DataFrame:
    if path is not None and path.exists():
        print(f"[load] local {path}")
        return pd.read_parquet(path)
    print(f"[load] hub {hub_repo} config=milestone1")
    return load_dataset(hub_repo, "milestone1", split="train").to_pandas()


def load_gci_fine(m1: pd.DataFrame | None) -> pd.DataFrame:
    """Fine rows = official GCI topics only."""
    if m1 is not None and "source" in m1.columns:
        gci = m1[m1["source"] == "gci_bench"].copy()
        if len(gci) > 0:
            print(f"[fine] from local m1 gci_bench: {len(gci)}")
            return gci
    print("[fine] load Glint-Research/GCI_Bench")
    raw = load_dataset("Glint-Research/GCI_Bench", split="test")
    rows = []
    diff_map = {"easy": 2, "medium": 3, "hard": 4}
    for i, ex in enumerate(raw):
        topic = str(ex.get("topic") or "").strip()
        q = str(ex.get("question") or "").strip()
        ctx = str(ex.get("context") or "").strip()
        prompt = f"{q}\n\n{ctx}".strip() if ctx else q
        if not prompt or topic not in DOMAIN_FINE_SET:
            continue
        rows.append(
            {
                "id": f"gci_bench:test:{i}",
                "source": "gci_bench",
                "domain_label": topic,
                "difficulty": str(diff_map.get(str(ex.get("difficulty") or "").lower(), 3)),
                "prompt": prompt,
            }
        )
    return pd.DataFrame(rows)


def build_coarse(m1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in m1.itertuples(index=False):
        prompt = (getattr(r, "prompt", None) or "").strip()
        if not prompt:
            continue
        source = getattr(r, "source", "") or ""
        # fine/GCI rows still contribute to coarse as knowledge
        domain = map_coarse(source, getattr(r, "domain_label", "") or "")
        assert domain in DOMAINS_COARSE, domain
        rid = getattr(r, "id", None) or f"{source}:row:{len(rows)}"
        rows.append(
            {
                "id": str(rid),
                "domain": domain,
                "difficulty": map_difficulty(getattr(r, "difficulty", 3)),
                "prompt": prompt,
            }
        )
    out = pd.DataFrame(rows, columns=["id", "domain", "difficulty", "prompt"])
    before = len(out)
    out = out.drop_duplicates(subset=["prompt"], keep="first").reset_index(drop=True)
    print(f"[coarse dedupe] {before} → {len(out)}")
    return out


def build_fine(gci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in gci.itertuples(index=False):
        prompt = (getattr(r, "prompt", None) or "").strip()
        topic = (getattr(r, "domain_label", None) or getattr(r, "topic", None) or "").strip()
        if not prompt or topic not in DOMAIN_FINE_SET:
            continue
        rid = getattr(r, "id", None) or f"gci_bench:row:{len(rows)}"
        rows.append(
            {
                "id": str(rid),
                "domain": topic,
                "difficulty": map_difficulty(getattr(r, "difficulty", 3)),
                "prompt": prompt,
            }
        )
    out = pd.DataFrame(rows, columns=["id", "domain", "difficulty", "prompt"])
    before = len(out)
    out = out.drop_duplicates(subset=["prompt"], keep="first").reset_index(drop=True)
    missing = DOMAIN_FINE_SET - set(out["domain"].unique())
    if missing:
        print(f"[warn] fine missing topics: {sorted(missing)}")
    print(f"[fine] {before} → {len(out)} rows; n_domains={out['domain'].nunique()}")
    return out


def split_df(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(round(len(shuf) * test_size)))
    test = shuf.iloc[:n_test].reset_index(drop=True)
    train = shuf.iloc[n_test:].reset_index(drop=True)
    return train, test, shuf


def _diff_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["difficulty"].value_counts().sort_index().items()}


def _domain_table_5(all_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    lines = [
        "| domain | description / sources | train | test | total |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for d in DOMAINS_COARSE:
        note = COARSE_DOMAIN_NOTE.get(d, "")
        n_tr = int((train_df["domain"] == d).sum())
        n_te = int((test_df["domain"] == d).sum())
        lines.append(f"| `{d}` | {note} | {n_tr:,} | {n_te:,} | {n_tr + n_te:,} |")
    return lines


def _domain_table_20(all_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    lines = [
        "| domain | topic label | train | test | total | difficulty (2/3/4) |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for d in DOMAINS_FINE:
        label = GCI_TOPIC_LABEL.get(d, d)
        sub = all_df[all_df["domain"] == d]
        n_tr = int((train_df["domain"] == d).sum())
        n_te = int((test_df["domain"] == d).sum())
        dc = _diff_counts(sub)
        diff_s = f"{dc.get(2, 0)}/{dc.get(3, 0)}/{dc.get(4, 0)}"
        lines.append(f"| `{d}` | {label} | {n_tr:,} | {n_te:,} | {n_tr + n_te:,} | {diff_s} |")
    return lines


def write_variant(
    out_dir: Path,
    *,
    name: str,
    domains: tuple[str, ...],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_df: pd.DataFrame,
    hub_repo: str,
    blurb: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    all_df.to_parquet(out_dir / "all.parquet", index=False)
    diff = _diff_counts(all_df)
    lines = [
        "---",
        f"pretty_name: tinyrouter-m1-{name}",
        "task_categories:",
        "  - text-classification",
        "tags:",
        "  - tinyrouter",
        "  - milestone-1",
        f"  - {name}",
        "license: other",
        "---",
        "",
        f"# tinyrouter-m1 / {name}",
        "",
        blurb,
        "",
        "## Schema",
        "",
        "| column | type | meaning |",
        "| --- | --- | --- |",
        "| `id` | string | stable row id |",
        "| `domain` | string | closed domain label |",
        "| `difficulty` | int | 1–5 |",
        "| `prompt` | string | user / problem text |",
        "",
        "## Split",
        "",
        f"- train: **{len(train_df):,}**",
        f"- test: **{len(test_df):,}**",
        f"- total: **{len(all_df):,}**",
        "",
        "## Difficulty",
        "",
        "| level | rows |",
        "| ---: | ---: |",
    ]
    for k, v in sorted(diff.items()):
        lines.append(f"| {k} | {v:,} |")
    lines += ["", f"## Domains ({len(domains)})", ""]
    if name == "5-domain":
        lines.extend(_domain_table_5(all_df, train_df, test_df))
    else:
        lines.extend(_domain_table_20(all_df, train_df, test_df))
        lines += [
            "",
            "Source: [Glint-Research/GCI_Bench](https://huggingface.co/datasets/Glint-Research/GCI_Bench) "
            "(GPL-3.0). Difficulty mapped easy→2 / medium→3 / hard→4.",
        ]
    lines += [
        "",
        "```python",
        "from datasets import load_dataset",
        f'ds = load_dataset("{hub_repo}", "{name}", split="train")',
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {out_dir}/ ({len(train_df)} train / {len(test_df)} test)")


def write_root_readme(
    root: Path,
    hub_repo: str,
    *,
    c_train: pd.DataFrame,
    c_test: pd.DataFrame,
    c_all: pd.DataFrame,
    f_train: pd.DataFrame,
    f_test: pd.DataFrame,
    f_all: pd.DataFrame,
) -> None:
    d5 = _diff_counts(c_all)
    d20 = _diff_counts(f_all)
    table_5 = "\n".join(_domain_table_5(c_all, c_train, c_test))
    table_20 = "\n".join(_domain_table_20(f_all, f_train, f_test))
    text = f"""---
pretty_name: tinyrouter-m1
task_categories:
  - text-classification
tags:
  - tinyrouter
  - milestone-1
  - gci-bench
license: other
configs:
  - config_name: 5-domain
    data_files:
      - split: train
        path: 5-domain/train.parquet
      - split: test
        path: 5-domain/test.parquet
  - config_name: 20-domain
    data_files:
      - split: train
        path: 20-domain/train.parquet
      - split: test
        path: 20-domain/test.parquet
  - config_name: default
    data_files:
      - split: train
        path: 5-domain/train.parquet
      - split: test
        path: 5-domain/test.parquet
---

# tinyrouter-m1

Two **separate** Milestone-1 triage datasets. Same schema; different domain taxonomies.

## Overview

| config | #domains | train | test | total | difficulty levels present |
| --- | ---: | ---: | ---: | ---: | --- |
| **`5-domain`** | 5 | {len(c_train):,} | {len(c_test):,} | {len(c_all):,} | {", ".join(str(k) for k in sorted(d5))} |
| **`20-domain`** | 20 | {len(f_train):,} | {len(f_test):,} | {len(f_all):,} | {", ".join(str(k) for k in sorted(d20))} |

## Schema (both)

| column | type | meaning |
| --- | --- | --- |
| `id` | string | stable row id (`source:split:idx`) |
| `domain` | string | closed domain label for that config |
| `difficulty` | int | 1–5 |
| `prompt` | string | user / problem text |

## `5-domain` — detailed

Full Milestone-1 pool remeshed to 5 router buckets.

**Difficulty:** {", ".join(f"{k}={v:,}" for k, v in sorted(d5.items()))}

{table_5}

## `20-domain` — detailed

[GCI-Bench](https://huggingface.co/datasets/Glint-Research/GCI_Bench) rows only
(GPL-3.0). `domain` = official GCI `topic`; topic labels from upstream `topicLabel`.
Difficulty: easy→2 / medium→3 / hard→4.

**Difficulty:** {", ".join(f"{k}={v:,}" for k, v in sorted(d20.items()))}

{table_20}

## Load

```python
from datasets import load_dataset
ds5  = load_dataset("{hub_repo}", "5-domain", split="train")
ds20 = load_dataset("{hub_repo}", "20-domain", split="train")
print(ds5[0]["domain"], ds5[0]["difficulty"])
print(ds20[0]["domain"], ds20[0]["difficulty"])
```

Built by `scripts/build_tinyrouter_m1_slim.py`.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def push_variant(repo: str, config: str, train_df: pd.DataFrame, test_df: pd.DataFrame, token: str) -> None:
    DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    ).push_to_hub(repo, config_name=config, token=token)
    print(f"[push] {repo} config={config}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=REPO / "datasets/router-bench/milestone1/all.parquet")
    ap.add_argument("--hub-repo", default="James-Cuda/router-bench")
    ap.add_argument("--out", type=Path, default=REPO / "datasets/tinyrouter-m1")
    ap.add_argument("--test-size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-repo", default="James-Cuda/tinyrouter-m1")
    args = ap.parse_args()

    m1 = load_m1(args.input if args.input.exists() else None, args.hub_repo)

    ds5 = build_coarse(m1)
    print("[5-domain]")
    for d in DOMAINS_COARSE:
        print(f"  {d:16s} {(ds5['domain']==d).sum():7d}")

    gci = load_gci_fine(m1)
    ds20 = build_fine(gci)
    print("[20-domain = GCI-Bench topics]")
    for d in DOMAINS_FINE:
        print(f"  {d:16s} {(ds20['domain']==d).sum():7d}")

    c_train, c_test, c_all = split_df(ds5, args.test_size, args.seed)
    f_train, f_test, f_all = split_df(ds20, args.test_size, args.seed)

    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    # drop old folder names if present
    for legacy in ("coarse", "fine"):
        legacy_dir = root / legacy
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)
            print(f"[clean] removed {legacy_dir}")

    write_variant(
        root / "5-domain",
        name="5-domain",
        domains=DOMAINS_COARSE,
        train_df=c_train,
        test_df=c_test,
        all_df=c_all,
        hub_repo=args.push_repo,
        blurb="**5-domain** triage over the full Milestone-1 pool.",
    )
    write_variant(
        root / "20-domain",
        name="20-domain",
        domains=DOMAINS_FINE,
        train_df=f_train,
        test_df=f_test,
        all_df=f_all,
        hub_repo=args.push_repo,
        blurb=(
            "**20-domain** = **exact** Glint "
            "[GCI-Bench](https://huggingface.co/datasets/Glint-Research/GCI_Bench) "
            "topics (gci_bench rows only)."
        ),
    )
    write_root_readme(
        root,
        args.push_repo,
        c_train=c_train,
        c_test=c_test,
        c_all=c_all,
        f_train=f_train,
        f_test=f_test,
        f_all=f_all,
    )
    print(f"[write] {root}/README.md")

    if args.push:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN not set")
        create_repo(args.push_repo, repo_type="dataset", exist_ok=True, token=token)
        # README-only refresh by default when data already pushed; still push data for consistency
        push_variant(args.push_repo, "5-domain", c_train, c_test, token)
        push_variant(args.push_repo, "20-domain", f_train, f_test, token)
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(root / "README.md"),
            path_in_repo="README.md",
            repo_id=args.push_repo,
            repo_type="dataset",
        )
        for name in ("5-domain", "20-domain"):
            api.upload_file(
                path_or_fileobj=str(root / name / "README.md"),
                path_in_repo=f"{name}/README.md",
                repo_id=args.push_repo,
                repo_type="dataset",
            )
        print(f"[push] → https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()
