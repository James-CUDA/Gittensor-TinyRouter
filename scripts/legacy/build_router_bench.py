#!/usr/bin/env python3
"""Build James-Cuda/router-bench — unified open router-eval corpus for Milestone 1.

Merges:
  - withmartian/routerbench (precomputed multi-LLM outcomes)
  - databricks/databricks-dolly-15k
  - google-research-datasets/mbpp
  - openai/openai_humaneval
  - deepmind/aqua_rat
  - openai/gsm8k
  - cais/mmlu
  - HuggingFaceH4/MATH-500

Usage:
  python scripts/build_router_bench.py --out datasets/router-bench
  HF_TOKEN=... python scripts/build_router_bench.py --out datasets/router-bench --push
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, create_repo


SCHEMA_DESC = """\
Unified prompt rows for open LLM-router evaluation (Milestone 1).

Columns:
  id                 unique row id (source:split:local_index)
  source             origin dataset name
  split              train|validation|test|...
  prompt             user / problem text
  provenance_label   coarse label from dataset membership
                     (code|math|knowledge|instruction|commonsense|conversation|rag|other)
  reference          gold answer / tests / rationale when available (else "")
  domain             finer tag (e.g. MMLU subject); else ""
  metadata_json      JSON blob of source-specific fields
"""


def _clean(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _row(
    *,
    source: str,
    split: str,
    idx: int,
    prompt: str,
    provenance_label: str,
    reference: str = "",
    domain: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    return {
        "id": f"{source}:{split}:{idx}",
        "source": source,
        "split": split,
        "prompt": _clean(prompt),
        "provenance_label": provenance_label,
        "reference": _clean(reference),
        "domain": _clean(domain),
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    }


def _from_rows(rows: list[dict[str, str]]) -> Dataset:
    # Drop empties
    rows = [r for r in rows if r["prompt"]]
    return Dataset.from_list(rows)


def load_dolly() -> Dataset:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for i, ex in enumerate(ds):
        instr = _clean(ex.get("instruction"))
        context = _clean(ex.get("context"))
        prompt = f"{instr}\n\n{context}".strip() if context else instr
        rows.append(
            _row(
                source="dolly-15k",
                split="train",
                idx=i,
                prompt=prompt,
                provenance_label="instruction",
                reference=_clean(ex.get("response")),
                domain=_clean(ex.get("category")),
                metadata={"category": ex.get("category")},
            )
        )
    return _from_rows(rows)


def load_mbpp() -> Dataset:
    # Prefer sanitized split when present
    try:
        raw = load_dataset("google-research-datasets/mbpp", "sanitized")
    except Exception:
        raw = load_dataset("google-research-datasets/mbpp")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            prompt = _clean(ex.get("text") or ex.get("prompt"))
            ref = _clean(ex.get("code"))
            rows.append(
                _row(
                    source="mbpp",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    provenance_label="code",
                    reference=ref,
                    domain="code",
                    metadata={
                        "task_id": ex.get("task_id"),
                        "test_list": ex.get("test_list"),
                    },
                )
            )
    return _from_rows(rows)


def load_humaneval() -> Dataset:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _row(
                source="humaneval",
                split="test",
                idx=i,
                prompt=_clean(ex.get("prompt")),
                provenance_label="code",
                reference=_clean(ex.get("canonical_solution")),
                domain="code",
                metadata={
                    "task_id": ex.get("task_id"),
                    "entry_point": ex.get("entry_point"),
                    "test": ex.get("test"),
                },
            )
        )
    return _from_rows(rows)


def load_aqua() -> Dataset:
    raw = load_dataset("deepmind/aqua_rat")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            opts = ex.get("options") or []
            opt_txt = "\n".join(opts) if isinstance(opts, list) else _clean(opts)
            prompt = f"{_clean(ex.get('question'))}\n\nOptions:\n{opt_txt}".strip()
            rows.append(
                _row(
                    source="aqua_rat",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    provenance_label="math",
                    reference=_clean(ex.get("correct")),
                    domain="math",
                    metadata={"rationale": ex.get("rationale"), "options": opts},
                )
            )
    return _from_rows(rows)


def load_gsm8k() -> Dataset:
    raw = load_dataset("openai/gsm8k", "main")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            rows.append(
                _row(
                    source="gsm8k",
                    split=str(split),
                    idx=i,
                    prompt=_clean(ex.get("question")),
                    provenance_label="math",
                    reference=_clean(ex.get("answer")),
                    domain="math",
                )
            )
    return _from_rows(rows)


def load_mmlu() -> Dataset:
    # all subjects via 'all' config
    raw = load_dataset("cais/mmlu", "all")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            choices = ex.get("choices") or []
            if isinstance(choices, list):
                labeled = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
            else:
                labeled = _clean(choices)
            prompt = f"{_clean(ex.get('question'))}\n\n{labeled}".strip()
            ans = ex.get("answer")
            if isinstance(ans, int) and 0 <= ans < 26:
                ref = chr(65 + ans)
            else:
                ref = _clean(ans)
            rows.append(
                _row(
                    source="mmlu",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    provenance_label="knowledge",
                    reference=ref,
                    domain=_clean(ex.get("subject")),
                    metadata={"subject": ex.get("subject"), "choices": choices},
                )
            )
    return _from_rows(rows)


def load_math500() -> Dataset:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _row(
                source="math500",
                split="test",
                idx=i,
                prompt=_clean(ex.get("problem")),
                provenance_label="math",
                reference=_clean(ex.get("answer") or ex.get("solution")),
                domain=_clean(ex.get("subject") or ex.get("level") or "math"),
                metadata={
                    "level": ex.get("level"),
                    "subject": ex.get("subject"),
                    "unique_id": ex.get("unique_id"),
                },
            )
        )
    return _from_rows(rows)


def _flatten_prompt(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        parts = [_flatten_prompt(x) for x in val]
        return "\n".join(p for p in parts if p)
    if isinstance(val, dict):
        if "content" in val:
            return _flatten_prompt(val.get("content"))
        return _clean(val)
    return _clean(val)


def _routerbench_prompt(sample: Any) -> str:
    if isinstance(sample, dict):
        for key in ("prompt", "question", "input", "query", "text"):
            if key in sample and sample[key] is not None:
                return _flatten_prompt(sample[key])
        if "messages" in sample and isinstance(sample["messages"], list):
            return _flatten_prompt(sample["messages"])
    return _flatten_prompt(sample)


def _routerbench_label(sample: dict[str, Any]) -> str:
    blob = json.dumps(sample, ensure_ascii=False).lower()
    # coarse provenance from eval-set name / tags when present
    for key in ("eval_name", "dataset", "task", "category", "domain"):
        val = _clean(sample.get(key)).lower()
        if not val:
            continue
        if any(x in val for x in ("mbpp", "code", "humaneval", "program")):
            return "code"
        if any(x in val for x in ("gsm", "math", "aqua")):
            return "math"
        if "mmlu" in val or "knowledge" in val:
            return "knowledge"
        if any(x in val for x in ("mt-bench", "mt_bench", "chat", "conversation")):
            return "conversation"
        if "rag" in val:
            return "rag"
        if any(x in val for x in ("hellaswag", "winogrande", "commonsense")):
            return "commonsense"
    if "mbpp" in blob or "humaneval" in blob:
        return "code"
    if "gsm8k" in blob or "gsm-8k" in blob:
        return "math"
    return "other"


def load_routerbench() -> Dataset:
    """Load Martian RouterBench pickles from the Hub (no live LLM calls)."""
    # Download raw files via HF hub snapshot
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, str]] = []
    for fname, tag in (
        ("routerbench_0shot.pkl", "0shot"),
        ("routerbench_5shot.pkl", "5shot"),
    ):
        path = hf_hub_download(
            "withmartian/routerbench",
            filename=fname,
            repo_type="dataset",
        )
        with open(path, "rb") as f:
            obj = pickle.load(f)

        # RouterBench formats vary: DataFrame, list[dict], dict of lists, nested
        samples: Iterable[Any]
        if hasattr(obj, "to_dict"):
            # pandas
            samples = obj.to_dict(orient="records")
        elif isinstance(obj, dict):
            # maybe {split: [...]} or columnar
            if all(isinstance(v, list) for v in obj.values()):
                keys = list(obj.keys())
                n = len(obj[keys[0]])
                samples = [{k: obj[k][i] for k in keys} for i in range(n)]
            else:
                # values are samples
                flat = []
                for k, v in obj.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                item = {**item, "_rb_key": k}
                            flat.append(item)
                    elif isinstance(v, dict):
                        flat.append({**v, "_rb_key": k})
                samples = flat
        elif isinstance(obj, list):
            samples = obj
        else:
            raise TypeError(f"Unsupported RouterBench type: {type(obj)}")

        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                sample = {"value": sample}
            prompt = _routerbench_prompt(sample)
            # Compact metadata: ids + oracle + per-model quality/cost scores.
            # Omit |model_response text to keep the corpus small.
            meta: dict[str, Any] = {}
            for k, v in sample.items():
                if k in ("prompt", "question", "input", "query", "text", "messages"):
                    continue
                if "|model_response" in str(k):
                    continue
                if k in (
                    "sample_id",
                    "eval_name",
                    "oracle_model_to_route_to",
                    "dataset",
                    "task",
                ) or str(k).endswith("|total_cost"):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        meta[k] = v
                    else:
                        meta[k] = _clean(v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    # model quality score columns
                    meta[k] = float(v)
            rows.append(
                _row(
                    source="routerbench",
                    split=tag,
                    idx=i,
                    prompt=prompt,
                    provenance_label=_routerbench_label(sample),
                    reference=_clean(
                        sample.get("oracle_model_to_route_to")
                        or sample.get("answer")
                        or sample.get("target")
                        or ""
                    ),
                    domain=_clean(
                        sample.get("eval_name")
                        or sample.get("dataset")
                        or sample.get("task")
                        or ""
                    ),
                    metadata=meta,
                )
            )
    return _from_rows(rows)


LOADERS = (
    ("dolly-15k", load_dolly),
    ("mbpp", load_mbpp),
    ("humaneval", load_humaneval),
    ("aqua_rat", load_aqua),
    ("gsm8k", load_gsm8k),
    ("mmlu", load_mmlu),
    ("math500", load_math500),
    ("routerbench", load_routerbench),
)


def build() -> DatasetDict:
    parts: list[Dataset] = []
    counts: dict[str, int] = {}
    for name, fn in LOADERS:
        print(f"[router-bench] loading {name} ...")
        ds = fn()
        counts[name] = len(ds)
        print(f"[router-bench]   {name}: {len(ds)} rows")
        parts.append(ds)
    merged = concatenate_datasets(parts)
    # stable shuffle for a shared train view; keep full as 'train' (eval corpus)
    merged = merged.shuffle(seed=271828182)
    print("[router-bench] totals:", counts, "sum=", len(merged))
    return DatasetDict({"train": merged}), counts


SOURCE_NOTES = (
    ("dolly-15k", "instruction", "Databricks Dolly"),
    ("mbpp", "code", "MBPP (sanitized when available)"),
    ("humaneval", "code", "OpenAI HumanEval"),
    ("aqua_rat", "math", "AQuA-RAT"),
    ("gsm8k", "math", "Grade-school math"),
    ("mmlu", "knowledge", "CAIS MMLU (all)"),
    ("math500", "math", "HuggingFaceH4/MATH-500"),
    ("routerbench", "mixed", "Martian RouterBench 0-shot + 5-shot prompts"),
)


def _yaml_configs(sources: list[str]) -> str:
    """Hub dataset configs: default=all, plus one config per source."""
    lines = [
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        "        path: data/all.parquet",
    ]
    for src in sources:
        lines.extend(
            [
                f"  - config_name: {src}",
                "    data_files:",
                "      - split: train",
                f"        path: data/{src}.parquet",
            ]
        )
    return "\n".join(lines)


def write_readme(out_dir: Path, counts: dict[str, int]) -> None:
    sources = list(counts.keys())
    count_lines = [
        "| source | rows |",
        "| --- | ---: |",
    ]
    for k, v in counts.items():
        count_lines.append(f"| `{k}` | {v:,} |")
    count_lines.append(f"| **total** | **{sum(counts.values()):,}** |")

    source_lines = [
        "| source | provenance_label | notes |",
        "| --- | --- | --- |",
    ]
    notes = {n: (lab, note) for n, lab, note in SOURCE_NOTES}
    for src in sources:
        lab, note = notes.get(src, ("mixed", src))
        source_lines.append(f"| `{src}` | {lab} | {note} |")

    load_examples = "\n".join(
        [
            "```python",
            "from datasets import load_dataset",
            "",
            "# All sources merged",
            'ds = load_dataset("James-Cuda/router-bench", split="train")',
            "",
            "# One source at a time (named config)",
            'mmlu = load_dataset("James-Cuda/router-bench", "mmlu", split="train")',
            'code = load_dataset("James-Cuda/router-bench", "humaneval", split="train")',
            "print(len(ds), len(mmlu), mmlu[0]['provenance_label'])",
            "```",
        ]
    )

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
size_categories:
  - 100K<n<1M
license: other
{_yaml_configs(sources)}
---

# router-bench

Unified **open LLM-router evaluation corpus** for TinyRouter Milestone 1
(no live API required for triage / cached-routing experiments).

Built by [James-Cuda](https://huggingface.co/James-Cuda) for
[Gittensor-TinyRouter](https://github.com/James-CUDA/Gittensor-TinyRouter).

## Sources

{chr(10).join(source_lines)}

## Counts

{chr(10).join(count_lines)}

## Schema

{SCHEMA_DESC}

## Load

{load_examples}

Per-source parquet files also live under `data/<source>.parquet`
(and the merged corpus at `data/all.parquet`).

## License

Upstream licenses apply per `source` (Dolly, MBPP, HumanEval, AQuA, GSM8K, MMLU,
MATH-500, RouterBench). This repo only redistributes prompts/labels needed for
router evaluation; respect each source license for redistribution.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_source_splits(out_dir: Path, merged: Dataset, counts: dict[str, int]) -> Path:
    """Write data/all.parquet + data/<source>.parquet for Hub configs."""
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(data_dir / "all.parquet"))
    for src in counts:
        subset = merged.filter(lambda ex, s=src: ex["source"] == s)
        subset.to_parquet(str(data_dir / f"{src}.parquet"))
        print(f"[router-bench] wrote data/{src}.parquet ({len(subset)} rows)")
    return data_dir


def push(out_dir: Path, repo_id: str, *, commit_message: str) -> None:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not set. Export a Hugging Face write token, then re-run with --push:\n"
            "  export HF_TOKEN=hf_...\n"
            "  python scripts/build_router_bench.py --out datasets/router-bench --push"
        )
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    # Drop legacy root train.parquet so Hub configs use data/*.parquet only.
    try:
        api.delete_file(
            "train.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Remove legacy root train.parquet",
        )
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        print(f"[router-bench] note: could not delete legacy train.parquet ({exc})")
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    print(f"[router-bench] pushed → https://huggingface.co/datasets/{repo_id}")


def prepare_hub_dir(out_dir: Path) -> Path:
    """Assemble hub/ with README + counts + data/*.parquet."""
    import shutil

    hub_dir = out_dir / "hub"
    data_src = out_dir / "data"
    if not (out_dir / "README.md").exists() or not data_src.is_dir():
        raise SystemExit(f"missing README.md or data/ under {out_dir}")
    if hub_dir.exists():
        shutil.rmtree(hub_dir)
    hub_dir.mkdir(parents=True)
    shutil.copy2(out_dir / "README.md", hub_dir / "README.md")
    if (out_dir / "counts.json").exists():
        shutil.copy2(out_dir / "counts.json", hub_dir / "counts.json")
    shutil.copytree(data_src, hub_dir / "data")
    return hub_dir


def split_from_existing(out_dir: Path) -> dict[str, int]:
    """Re-split an existing train.parquet / dataset_dict into data/<source>.parquet."""
    parquet = out_dir / "train.parquet"
    disk = out_dir / "dataset_dict"
    if parquet.exists():
        print(f"[router-bench] loading {parquet} ...")
        merged = Dataset.from_parquet(str(parquet))
    elif disk.exists():
        print(f"[router-bench] loading {disk} ...")
        merged = DatasetDict.load_from_disk(str(disk))["train"]
    else:
        raise SystemExit(f"no train.parquet or dataset_dict in {out_dir}")

    from collections import Counter

    counts = dict(Counter(merged["source"]))
    # stable key order matching SOURCE_NOTES when present
    order = [n for n, _, _ in SOURCE_NOTES if n in counts]
    order += [k for k in counts if k not in order]
    counts = {k: counts[k] for k in order}

    write_source_splits(out_dir, merged, counts)
    # keep a convenience copy at root for local use
    merged.to_parquet(str(out_dir / "train.parquet"))
    write_readme(out_dir, counts)
    with open(out_dir / "counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"[router-bench] split complete → {out_dir}/data/ ({sum(counts.values())} rows)")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/router-bench"))
    ap.add_argument("--repo-id", default="James-Cuda/router-bench")
    ap.add_argument("--push", action="store_true")
    ap.add_argument(
        "--push-only",
        action="store_true",
        help="Upload an already-built hub layout (README + data/)",
    )
    ap.add_argument(
        "--split-only",
        action="store_true",
        help="Re-split existing train.parquet into data/<source>.parquet + fix README",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="source names to skip (e.g. routerbench mmlu)",
    )
    args = ap.parse_args()

    global LOADERS
    if args.skip:
        skip = set(args.skip)
        LOADERS = tuple((n, f) for n, f in LOADERS if n not in skip)

    if args.split_only:
        split_from_existing(args.out)
    elif not args.push_only:
        args.out.mkdir(parents=True, exist_ok=True)
        dsd, counts = build()
        dsd.save_to_disk(str(args.out / "dataset_dict"))
        dsd["train"].to_parquet(str(args.out / "train.parquet"))
        write_source_splits(args.out, dsd["train"], counts)
        write_readme(args.out, counts)
        with open(args.out / "counts.json", "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2)
        print(f"[router-bench] wrote {args.out}")

    if args.push or args.push_only:
        hub_dir = prepare_hub_dir(args.out)
        push(
            hub_dir,
            args.repo_id,
            commit_message="Split per-source parquet files and fix README tables",
        )


if __name__ == "__main__":
    main()

