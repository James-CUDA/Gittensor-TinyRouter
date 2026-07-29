#!/usr/bin/env python3
"""Reorganize router-bench into milestone1/ (domain+difficulty) and milestone2/ (routing).

Milestone 1 — prompt triage: predict domain + difficulty (not model/role).
Milestone 2 — future TinyRouter pool routing / multi-model router corpora.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfApi, create_repo


# Clear provenance for domain + difficulty classification
M1_SOURCES: dict[str, tuple[str, str]] = {
    # name: (relative path under out/, default difficulty 1-5)
    "dolly-15k": ("data/dolly-15k.parquet", "2"),
    "mbpp": ("data/mbpp.parquet", "2"),
    "humaneval": ("data/humaneval.parquet", "3"),
    "aqua_rat": ("data/aqua_rat.parquet", "3"),
    "gsm8k": ("data/gsm8k.parquet", "2"),
    "mmlu": ("data/mmlu.parquet", "3"),
    "math500": ("data/math500.parquet", "4"),
    "aime_2024": ("data2/aime_2024.parquet", "5"),
    "bigcodebench": ("data2/bigcodebench.parquet", "4"),
    "mmlu_pro": ("data2/mmlu_pro.parquet", "4"),
    "gpqa": ("data2/gpqa.parquet", "5"),
    "arc": ("data2/arc.parquet", "2"),
    "truthfulqa": ("data2/truthfulqa.parquet", "2"),
    "hellaswag": ("data2/hellaswag.parquet", "2"),
    "ifeval": ("data2/ifeval.parquet", "3"),
}

# Extra M1 sources loaded from the Hub (not under data/ / data2/)
M1_HUB_SOURCES = ("gci_bench",)

# Mixed / other-pool / no-ref router corpora — model routing later
M2_SOURCES: dict[str, str] = {
    "routerbench": "data/routerbench.parquet",
    "embedllm": "data2/embedllm.parquet",
    "sprout": "data2/sprout.parquet",
    "mix_instruct": "data2/mix_instruct.parquet",
    "llmrouterbench": "data2/llmrouterbench.parquet",
    "routereval": "data2/routereval.parquet",
}

GCI_DIFF = {"easy": "2", "medium": "3", "hard": "4"}

DOMAIN_NOTES = {
    "dolly-15k": "instruction — Databricks Dolly",
    "mbpp": "code — MBPP",
    "humaneval": "code — HumanEval",
    "aqua_rat": "math — AQuA-RAT",
    "gsm8k": "math — GSM8K",
    "mmlu": "knowledge — MMLU",
    "math500": "math — MATH-500",
    "aime_2024": "math — AIME 2024 (hard)",
    "bigcodebench": "code — BigCodeBench (hard)",
    "mmlu_pro": "knowledge — MMLU-Pro (hard)",
    "gpqa": "knowledge — GPQA (hard)",
    "arc": "knowledge — AI2 ARC",
    "truthfulqa": "knowledge — TruthfulQA",
    "hellaswag": "commonsense — HellaSwag",
    "ifeval": "instruction — IFEval",
    "gci_bench": (
        "topic domains (×20) — Glint-Research/GCI_Bench; "
        "domain_label=topic; difficulty from easy/medium/hard"
    ),
    "routerbench": "mixed — Martian RouterBench (other pool scores)",
    "embedllm": "mixed — EmbedLLM prompts (no reference)",
    "sprout": "mixed — SPROUT multi-model scores (other pool)",
    "mix_instruct": "instruction — MixInstruct (weak auto-grade)",
    "llmrouterbench": "mixed — LLMRouterBench prompt index (no reference)",
    "routereval": "mixed — RouterEval thin prompt index (no reference)",
}


def _difficulty_for_row(source: str, default: str, meta_s: str, domain: str) -> str:
    """Assign difficulty 1–5 from source defaults + light metadata rules."""
    try:
        meta = json.loads(meta_s) if meta_s else {}
    except Exception:
        meta = {}

    if source == "math500":
        level = meta.get("level") or meta.get("Level")
        if level is not None:
            try:
                lv = int(level)
                return str(min(5, max(1, lv)))
            except Exception:
                pass
    if source == "gpqa":
        cfg = str(meta.get("config") or "").lower()
        if "diamond" in cfg:
            return "5"
        if "extended" in cfg:
            return "4"
        return "5"
    if source == "arc":
        d = str(domain or "").lower()
        if "challenge" in d:
            return "3"
        return "2"
    if source == "mmlu":
        # keep mid; subject hardness ignored for v1
        return default
    return default


def load_gci_bench() -> pd.DataFrame:
    """Glint GCI_Bench → M1 rows; domain_label = topic (20 domains).

    Official GCI scores attention×gradient; we only reuse prompt + topic +
    difficulty for Milestone-1 triage (not the mechanistic harness).
    """
    from datasets import load_dataset

    raw = load_dataset("Glint-Research/GCI_Bench", split="test")
    rows: list[dict[str, str]] = []
    for i, ex in enumerate(raw):
        topic = str(ex.get("topic") or "").strip()
        diff_name = str(ex.get("difficulty") or "medium").strip().lower()
        question = str(ex.get("question") or "").strip()
        context = str(ex.get("context") or "").strip()
        prompt = f"{question}\n\n{context}".strip() if context else question
        if not prompt or not topic:
            continue
        meta = {
            "topicLabel": ex.get("topicLabel"),
            "templateId": ex.get("templateId"),
            "gci_difficulty": diff_name,
            "gci_id": ex.get("id"),
            "meta": ex.get("meta"),
        }
        rows.append(
            {
                "id": f"gci_bench:test:{i}",
                "source": "gci_bench",
                "split": "test",
                "prompt": prompt,
                "domain_label": topic,  # expands M1 beyond math/code/knowledge
                "difficulty": GCI_DIFF.get(diff_name, "3"),
                "provenance_label": topic,
                "reference": str(ex.get("referenceAnswer") or "").strip(),
                "domain": str(ex.get("topicLabel") or topic),
                "metadata_json": json.dumps(meta, ensure_ascii=False),
                "milestone": "1",
            }
        )
    return pd.DataFrame(rows)


def _enrich_m1(df: pd.DataFrame, source: str, default_diff: str) -> pd.DataFrame:
    df = df.copy()
    df["milestone"] = "1"
    df["domain_label"] = df["provenance_label"]
    diffs = [
        _difficulty_for_row(source, default_diff, str(m), str(d))
        for m, d in zip(df["metadata_json"], df["domain"])
    ]
    df["difficulty"] = diffs
    # keep column order
    cols = [
        "id",
        "source",
        "split",
        "prompt",
        "domain_label",
        "difficulty",
        "provenance_label",
        "reference",
        "domain",
        "metadata_json",
        "milestone",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def _enrich_m2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["milestone"] = "2"
    df["domain_label"] = df.get("provenance_label", "")
    df["difficulty"] = ""  # not the M2 target
    cols = [
        "id",
        "source",
        "split",
        "prompt",
        "domain_label",
        "difficulty",
        "provenance_label",
        "reference",
        "domain",
        "metadata_json",
        "milestone",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def reorganize(out_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    m1_dir = out_dir / "milestone1"
    m2_dir = out_dir / "milestone2"
    m1_dir.mkdir(parents=True, exist_ok=True)
    m2_dir.mkdir(parents=True, exist_ok=True)

    counts1: dict[str, int] = {}
    parts1: list[Dataset] = []
    for name, (rel, default_diff) in M1_SOURCES.items():
        src = out_dir / rel
        if not src.exists():
            print(f"[m1] MISSING {src}")
            continue
        df = _enrich_m1(pd.read_parquet(src), name, default_diff)
        dest = m1_dir / f"{name}.parquet"
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.to_parquet(str(dest))
        counts1[name] = len(ds)
        parts1.append(ds)
        print(f"[m1] {name}: {len(ds)}  domain={df['domain_label'].mode().iloc[0]}  "
              f"diff_dist={df['difficulty'].value_counts().to_dict()}")

    # Hub extras (GCI_Bench, …)
    if "gci_bench" in M1_HUB_SOURCES:
        print("[m1] loading gci_bench from Glint-Research/GCI_Bench ...")
        gci_df = load_gci_bench()
        dest = m1_dir / "gci_bench.parquet"
        ds = Dataset.from_pandas(gci_df, preserve_index=False)
        ds.to_parquet(str(dest))
        counts1["gci_bench"] = len(ds)
        parts1.append(ds)
        print(
            f"[m1] gci_bench: {len(ds)}  "
            f"n_domains={gci_df['domain_label'].nunique()}  "
            f"diff_dist={gci_df['difficulty'].value_counts().to_dict()}"
        )

    counts2: dict[str, int] = {}
    parts2: list[Dataset] = []
    for name, rel in M2_SOURCES.items():
        src = out_dir / rel
        if not src.exists():
            print(f"[m2] MISSING {src}")
            continue
        df = _enrich_m2(pd.read_parquet(src))
        dest = m2_dir / f"{name}.parquet"
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.to_parquet(str(dest))
        counts2[name] = len(ds)
        parts2.append(ds)
        print(f"[m2] {name}: {len(ds)}")

    if parts1:
        merged = concatenate_datasets(parts1).shuffle(seed=271828182)
        merged.to_parquet(str(m1_dir / "all.parquet"))
        print(f"[m1] all.parquet: {len(merged)}")
    if parts2:
        merged2 = concatenate_datasets(parts2).shuffle(seed=271828182)
        merged2.to_parquet(str(m2_dir / "all.parquet"))
        print(f"[m2] all.parquet: {len(merged2)}")

    (out_dir / "counts_milestone1.json").write_text(json.dumps(counts1, indent=2))
    (out_dir / "counts_milestone2.json").write_text(json.dumps(counts2, indent=2))
    return counts1, counts2


def write_readme(out_dir: Path, counts1: dict[str, int], counts2: dict[str, int]) -> None:
    def cfg(name: str, path: str) -> list[str]:
        return [
            f"  - config_name: {name}",
            "    data_files:",
            "      - split: train",
            f"        path: {path}",
        ]

    yaml: list[str] = [
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        "        path: milestone1/all.parquet",
        "  - config_name: milestone1",
        "    data_files:",
        "      - split: train",
        "        path: milestone1/all.parquet",
        "  - config_name: milestone2",
        "    data_files:",
        "      - split: train",
        "        path: milestone2/all.parquet",
    ]
    for src in counts1:
        yaml.extend(cfg(f"m1_{src}", f"milestone1/{src}.parquet"))
    for src in counts2:
        yaml.extend(cfg(f"m2_{src}", f"milestone2/{src}.parquet"))

    def count_table(title: str, counts: dict[str, int]) -> str:
        lines = [f"### {title}", "", "| source | rows |", "| --- | ---: |"]
        for k, v in counts.items():
            lines.append(f"| `{k}` | {v:,} |")
        lines.append(f"| **subtotal** | **{sum(counts.values()):,}** |")
        return "\n".join(lines)

    src_lines = [
        "| source | folder | domain_label | notes |",
        "| --- | --- | --- | --- |",
    ]
    for src in counts1:
        note = DOMAIN_NOTES.get(src, src)
        lab = note.split("—")[0].strip() if "—" in note else "mixed"
        src_lines.append(f"| `{src}` | `milestone1/` | {lab} | {note} |")
    for src in counts2:
        note = DOMAIN_NOTES.get(src, src)
        src_lines.append(f"| `{src}` | `milestone2/` | mixed | {note} |")

    total = sum(counts1.values()) + sum(counts2.values())
    text = f"""---
pretty_name: router-bench
task_categories:
  - text-classification
  - text-generation
language:
  - en
tags:
  - llm-routing
  - router
  - benchmark
  - tinyrouter
  - milestone-1
  - milestone-2
size_categories:
  - 100K<n<1M
license: other
{chr(10).join(yaml)}
---

# router-bench

Open corpus for [Gittensor-TinyRouter](https://github.com/James-CUDA/Gittensor-TinyRouter)
by [James-Cuda](https://huggingface.co/James-Cuda).

Split by **product milestone** (not dump chronology):

| Folder | Goal | Predict |
| --- | --- | --- |
| **`milestone1/`** | Prompt **triage** (Glint-style) | **domain** + **difficulty** |
| **`milestone2/`** | Future **pool routing** / router benches | model (/ role) later — not M1 |

Milestone 1 deliberately does **not** score TinyRouter `(model, role)`.
That comes in Milestone 2 once answer caches / live loops exist.

## Milestone 1 — domain + difficulty

Gold labels:

- `domain_label` — coarse provenance (`math` / `code` / `knowledge` / `commonsense` /
  `instruction`) **plus** 20 [GCI_Bench](https://huggingface.co/datasets/Glint-Research/GCI_Bench)
  topics (`cooking`, `astronomy`, `finance`, …) when `source=gci_bench`
- `difficulty` — `1`–`5` (source defaults + metadata; GCI maps easy→2, medium→3, hard→4)

`m1_gci_bench` adds **5,000** Glint items (`prompt = question + context`). We use
topic/difficulty for **triage only** — not Glint’s attention×gradient GCI score.

Dropped from M1: mixed router dumps (`routerbench`, `sprout`, `embedllm`,
`llmrouterbench`, `routereval`, `mix_instruct`) → see Milestone 2.

## Milestone 2 — routing corpora

Multi-model / prompt-index sets for later TinyRouter pool routing experiments.
Many rows lack a gradable `reference` or only carry **other pools'** scores — build
your own A/B/C cache before claiming routing accuracy.

## Sources

{chr(10).join(src_lines)}

## Counts

{count_table("milestone1/ (triage)", counts1)}

{count_table("milestone2/ (routing corpora)", counts2)}

| | rows |
| --- | ---: |
| **grand total** | **{total:,}** |

## Schema

| column | meaning |
| --- | --- |
| `id` | `source:split:local_index` |
| `source` | origin dataset |
| `split` | upstream split name |
| `prompt` | user / problem text |
| `domain_label` | M1 gold domain (`math` / `code` / …) |
| `difficulty` | M1 gold difficulty `1`–`5` (empty on M2) |
| `provenance_label` | same as `domain_label` (compat) |
| `reference` | gold answer when available |
| `domain` | finer tag (subject / suite) |
| `metadata_json` | source-specific JSON |
| `milestone` | `"1"` or `"2"` |

## Load

```python
from datasets import load_dataset

# Milestone 1 merge (default)
m1 = load_dataset("James-Cuda/router-bench", split="train")
# or: load_dataset("James-Cuda/router-bench", "milestone1", split="train")

# One M1 source
gpqa = load_dataset("James-Cuda/router-bench", "m1_gpqa", split="train")
math = load_dataset("James-Cuda/router-bench", "m1_math500", split="train")

# Milestone 2 merge
m2 = load_dataset("James-Cuda/router-bench", "milestone2", split="train")
rb = load_dataset("James-Cuda/router-bench", "m2_routerbench", split="train")

# M1 labels
print(m1[0]["domain_label"], m1[0]["difficulty"])
```

## Difficulty rules (M1)

| difficulty | Typical sources |
| ---: | --- |
| 1–2 | `dolly-15k`, `gsm8k`, `arc` (Easy), `hellaswag`, `truthfulqa`, GCI `easy` |
| 3 | `aqua_rat`, `mbpp`, `mmlu`, `ifeval`, `arc` (Challenge), GCI `medium` |
| 4 | `math500` (by MATH level when present), `humaneval`, `mmlu_pro`, `bigcodebench`, GCI `hard` |
| 5 | `aime_2024`, `gpqa` (esp. diamond) |

## License

Upstream licenses apply per `source`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")
    print(f"[readme] wrote {out_dir / 'README.md'}")


def push(out_dir: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    api.upload_file(
        path_or_fileobj=str(out_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Reorganize into milestone1 (domain+difficulty) and milestone2",
    )
    for name in ("counts_milestone1.json", "counts_milestone2.json"):
        p = out_dir / name
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {name}",
            )
    for folder in ("milestone1", "milestone2"):
        api.upload_folder(
            folder_path=str(out_dir / folder),
            path_in_repo=folder,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {folder}/ corpora",
        )

    # Remove legacy layouts from the Hub so the card matches the new split
    legacy = []
    try:
        for info in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
            path = getattr(info, "path", "") or ""
            if path.startswith("data/") or path.startswith("data2/"):
                legacy.append(path)
            if path in ("counts.json", "counts_data2.json"):
                legacy.append(path)
    except Exception as e:
        print(f"[push] list legacy failed: {e}")
    for path in sorted(set(legacy)):
        try:
            api.delete_file(
                path_in_repo=path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Remove legacy {path}",
            )
            print(f"[push] deleted {path}")
        except Exception as e:
            print(f"[push] delete {path} failed: {e}")

    print(f"[push] → https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/router-bench"))
    ap.add_argument("--repo-id", default="James-Cuda/router-bench")
    ap.add_argument("--push", action="store_true")
    ap.add_argument(
        "--remove-legacy-local",
        action="store_true",
        help="Delete local data/ and data2/ after building milestone folders",
    )
    args = ap.parse_args()
    counts1, counts2 = reorganize(args.out)
    write_readme(args.out, counts1, counts2)
    if args.remove_legacy_local:
        for folder in ("data", "data2"):
            p = args.out / folder
            if p.exists():
                shutil.rmtree(p)
                print(f"[local] removed {p}")
    if args.push:
        push(args.out, args.repo_id)


if __name__ == "__main__":
    main()
