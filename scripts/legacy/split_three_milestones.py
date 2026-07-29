#!/usr/bin/env python3
"""Split James-Cuda/router-bench into three product milestones and fetch Hub extras.

Milestone 1 — prompt triage: domain + difficulty (no model/role)
Milestone 2 — model↔prompt scoring / difficulty routing corpora (no live API)
Milestone 3 — live OpenRouter: 3 pool models × Thinker/Worker/Verifier
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, create_repo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLS = [
    "id", "source", "split", "prompt", "domain_label", "difficulty",
    "provenance_label", "reference", "domain", "metadata_json", "milestone",
]

# Already-local M1 triage sources (domain + difficulty)
M1_LOCAL = {
    "dolly-15k", "mbpp", "humaneval", "aqua_rat", "gsm8k", "mmlu", "math500",
    "aime_2024", "bigcodebench", "mmlu_pro", "gpqa", "arc", "truthfulqa",
    "hellaswag", "ifeval", "gci_bench", "supra_prompt_routing", "query_complexity",
}

# Model-score / router corpora (M2)
M2_LOCAL = {
    "routerbench", "embedllm", "sprout", "mix_instruct", "llmrouterbench", "routereval",
}

# Small held-out suite for live 3×3 API eval (M3) — copied from M1 sources
M3_LIVE_SOURCES = ("math500", "aime_2024", "humaneval", "mbpp", "gpqa")

NO_ROBOTS_DIFF = {
    "Coding": "4", "Closed QA": "3", "Open QA": "2", "Classify": "2",
    "Extract": "2", "Summarize": "2", "Rewrite": "2", "Generation": "2",
    "Brainstorm": "1", "Chat": "1",
}
COMPLEXITY_DIFF = {"LOW": "2", "MEDIUM": "3", "HIGH": "5"}

NOTES = {
    "dolly-15k": "M1 instruction — Dolly",
    "mbpp": "M1/M3 code — MBPP",
    "humaneval": "M1/M3 code — HumanEval",
    "aqua_rat": "M1 math — AQuA-RAT",
    "gsm8k": "M1 math — GSM8K",
    "mmlu": "M1 knowledge — MMLU",
    "math500": "M1/M3 math — MATH-500",
    "aime_2024": "M1/M3 math — AIME 2024",
    "bigcodebench": "M1 code — BigCodeBench",
    "mmlu_pro": "M1 knowledge — MMLU-Pro",
    "gpqa": "M1/M3 knowledge — GPQA",
    "arc": "M1 knowledge — ARC",
    "truthfulqa": "M1 knowledge — TruthfulQA",
    "hellaswag": "M1 commonsense — HellaSwag",
    "ifeval": "M1 instruction — IFEval",
    "gci_bench": "M1 topics×20 — Glint GCI_Bench (GPL-3.0)",
    "supra_prompt_routing": "M1 — SupraLabs Prompt-Routing (MIT)",
    "query_complexity": "M1 — llm-query-complexity-benchmark (Apache-2.0)",
    "gatewaybench": "M1 — GatewayBench-v1 full (MIT)",
    "no_robots": "M1 — HuggingFaceH4/no_robots",
    "pubmedqa": "M1 medical — PubMedQA",
    "finqa": "M1 finance — flare-finqa",
    "hendrycks_math": "M1 math — hendrycks_math levels",
    "legalbench": "M1 legal — LegalBench tasks",
    "medmcqa": "M1 medical — MedMCQA subjects",
    "routerbench": "M2 — Martian RouterBench model scores",
    "embedllm": "M2 — EmbedLLM prompts",
    "sprout": "M2 — SPROUT model_scores",
    "mix_instruct": "M2 — MixInstruct",
    "llmrouterbench": "M2 — LLMRouterBench prompt index",
    "routereval": "M2 — RouterEval thin prompts",
    "router_v1": "M2 — tensuai/router-v1 (selected_model; weak domain)",
}


def _clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _row(**kw) -> dict[str, str]:
    lab = _clean(kw.get("domain_label"))
    return {
        "id": kw["id"] if "id" in kw else f"{kw['source']}:{kw['split']}:{kw['idx']}",
        "source": kw["source"],
        "split": str(kw.get("split", "train")),
        "prompt": _clean(kw.get("prompt")),
        "domain_label": lab,
        "difficulty": _clean(kw.get("difficulty")),
        "provenance_label": lab,
        "reference": _clean(kw.get("reference")),
        "domain": _clean(kw.get("domain")) or lab,
        "metadata_json": json.dumps(kw.get("metadata") or {}, ensure_ascii=False),
        "milestone": str(kw.get("milestone", "1")),
    }


def _save(df: pd.DataFrame, path: Path) -> int:
    df = df[[c for c in COLS if c in df.columns]]
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLS]
    df = df[df["prompt"].astype(str).str.strip() != ""]
    Dataset.from_pandas(df, preserve_index=False).to_parquet(str(path))
    return len(df)


def _ensure_milestone(df: pd.DataFrame, m: str) -> pd.DataFrame:
    df = df.copy()
    df["milestone"] = m
    if "domain_label" not in df.columns and "provenance_label" in df.columns:
        df["domain_label"] = df["provenance_label"]
    if "difficulty" not in df.columns:
        df["difficulty"] = ""
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]


# -------------------- Hub fetchers (M1) --------------------

def fetch_gatewaybench() -> pd.DataFrame:
    raw = load_dataset("ModaLabs/GatewayBench-v1", "full", split="train")
    rows = []
    for i, ex in enumerate(raw):
        meta = ex.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        rows.append(_row(
            source="gatewaybench", split="train", idx=i, milestone="1",
            prompt=ex["user_prompt"],
            domain_label=meta.get("domain") or "general",
            difficulty=str(int(meta.get("difficulty") or 3)),
            reference=ex.get("reference_answer") or "",
            domain=meta.get("domain") or "",
            metadata={"task_type": ex.get("task_type")},
        ))
    return pd.DataFrame(rows)


def fetch_no_robots() -> pd.DataFrame:
    raw = load_dataset("HuggingFaceH4/no_robots")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            cat = _clean(ex.get("category")) or "Generation"
            rows.append(_row(
                source="no_robots", split=split, idx=i, milestone="1",
                prompt=ex["prompt"], domain_label=cat,
                difficulty=NO_ROBOTS_DIFF.get(cat, "2"), domain=cat,
            ))
    return pd.DataFrame(rows)


def fetch_pubmedqa() -> pd.DataFrame:
    raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    for i, ex in enumerate(raw):
        rows.append(_row(
            source="pubmedqa", split="train", idx=i, milestone="1",
            prompt=ex["question"], domain_label="medical", difficulty="4",
            reference=_clean(ex.get("final_decision")), domain="pubmedqa",
        ))
    return pd.DataFrame(rows)


def fetch_finqa() -> pd.DataFrame:
    raw = load_dataset("ChanceFocus/flare-finqa")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            prompt = _clean(ex.get("query")) or _clean(ex.get("text"))
            rows.append(_row(
                source="finqa", split=split, idx=i, milestone="1",
                prompt=prompt, domain_label="finance", difficulty="4",
                reference=_clean(ex.get("answer")), domain="finqa",
            ))
    return pd.DataFrame(rows)


def fetch_hendrycks_math() -> pd.DataFrame:
    rows = []
    for cfg in get_dataset_config_names("EleutherAI/hendrycks_math"):
        raw = load_dataset("EleutherAI/hendrycks_math", cfg)
        for split, ds in raw.items():
            for i, ex in enumerate(ds):
                level = _clean(ex.get("level"))
                m = re.search(r"(\d+)", level)
                diff = str(min(5, max(1, int(m.group(1))))) if m else "3"
                rows.append(_row(
                    source="hendrycks_math", split=split, idx=len(rows), milestone="1",
                    prompt=ex["problem"],
                    domain_label=_clean(ex.get("type")) or cfg,
                    difficulty=diff, reference=_clean(ex.get("solution")), domain=cfg,
                    metadata={"level_raw": level},
                ))
    return pd.DataFrame(rows)


def fetch_legalbench() -> pd.DataFrame:
    rows = []
    for cfg in get_dataset_config_names("nguha/legalbench"):
        try:
            raw = load_dataset("nguha/legalbench", cfg)
        except Exception as e:
            print(f"  skip legal {cfg}: {e}")
            continue
        for split, ds in raw.items():
            for i, ex in enumerate(ds):
                text = _clean(ex.get("text") or ex.get("question"))
                if not text:
                    continue
                rows.append(_row(
                    source="legalbench", split=split, idx=len(rows), milestone="1",
                    prompt=text, domain_label=cfg, difficulty="3",
                    reference=_clean(ex.get("answer")), domain="legal",
                    metadata={"legalbench_config": cfg},
                ))
    return pd.DataFrame(rows)


def fetch_medmcqa() -> pd.DataFrame:
    raw = load_dataset("openlifescienceai/medmcqa")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            subj = _clean(ex.get("subject_name")) or "medical"
            q = _clean(ex.get("question"))
            opts = "\n".join(
                f"{L}. {_clean(ex.get(k))}"
                for L, k in zip("ABCD", ("opa", "opb", "opc", "opd"))
                if _clean(ex.get(k))
            )
            prompt = f"{q}\n\n{opts}" if opts else q
            try:
                ref = "ABCD"[int(ex.get("cop"))]
            except Exception:
                ref = _clean(ex.get("cop"))
            rows.append(_row(
                source="medmcqa", split=split, idx=i, milestone="1",
                prompt=prompt, domain_label=subj, difficulty="3",
                reference=ref, domain=_clean(ex.get("topic_name")) or subj,
            ))
    return pd.DataFrame(rows)


# -------------------- Hub fetchers (M2) --------------------

def fetch_router_v1() -> pd.DataFrame:
    """tensuai/router-v1 — prompt + selected_model; difficulty from reasoningEffort if any."""
    raw = load_dataset("tensuai/router-v1", split="train")
    rows = []
    for i, ex in enumerate(raw):
        sel = ex.get("selection") or {}
        if isinstance(sel, str):
            try:
                sel = json.loads(sel)
            except Exception:
                sel = {}
        effort = sel.get("reasoningEffort")
        if effort is None:
            diff = "3"
        else:
            try:
                diff = str(min(5, max(1, int(float(effort)))))
            except Exception:
                diff = "3"
        # no native domain — tag as routing/mixed; user remeshes later
        rows.append(_row(
            source="router_v1", split="train", idx=i, milestone="2",
            prompt=ex["prompt"],
            domain_label="routing",
            difficulty=diff,
            reference="",
            domain="router_v1",
            metadata={"selected_model": sel.get("selected_model"), "selection": sel},
        ))
    return pd.DataFrame(rows)


# Heavy sets (legalbench ~all configs, medmcqa ~180k) are opt-in via --heavy
M1_FETCHERS_LIGHT: list[tuple[str, Callable[[], pd.DataFrame]]] = [
    ("gatewaybench", fetch_gatewaybench),
    ("no_robots", fetch_no_robots),
    ("pubmedqa", fetch_pubmedqa),
    ("finqa", fetch_finqa),
    ("hendrycks_math", fetch_hendrycks_math),
]
M1_FETCHERS_HEAVY: list[tuple[str, Callable[[], pd.DataFrame]]] = [
    ("legalbench", fetch_legalbench),
    ("medmcqa", fetch_medmcqa),
]
M1_FETCHERS = M1_FETCHERS_LIGHT

M2_FETCHERS: list[tuple[str, Callable[[], pd.DataFrame]]] = [
    ("router_v1", fetch_router_v1),
]


def _merge_folder(folder: Path) -> dict[str, int]:
    parts, counts = [], {}
    for pq in sorted(folder.glob("*.parquet")):
        if pq.name == "all.parquet":
            continue
        ds = Dataset.from_parquet(str(pq))
        counts[pq.stem] = len(ds)
        parts.append(ds)
        print(f"  [{folder.name}] {pq.stem}: {len(ds)}")
    if parts:
        merged = concatenate_datasets(parts).shuffle(seed=271828182)
        merged.to_parquet(str(folder / "all.parquet"))
        print(f"  [{folder.name}] all.parquet: {len(merged)}")
    return counts


def write_readme(out: Path, c1: dict[str, int], c2: dict[str, int], c3: dict[str, int]) -> None:
    def cfg(name: str, path: str) -> list[str]:
        return [
            f"  - config_name: {name}",
            "    data_files:",
            "      - split: train",
            f"        path: {path}",
        ]

    yaml = [
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
        "  - config_name: milestone3",
        "    data_files:",
        "      - split: train",
        "        path: milestone3/all.parquet",
    ]
    for src in c1:
        yaml.extend(cfg(f"m1_{src}", f"milestone1/{src}.parquet"))
    for src in c2:
        yaml.extend(cfg(f"m2_{src}", f"milestone2/{src}.parquet"))
    for src in c3:
        yaml.extend(cfg(f"m3_{src}", f"milestone3/{src}.parquet"))

    def table(title: str, counts: dict[str, int]) -> str:
        lines = [f"### {title}", "", "| source | rows |", "| --- | ---: |"]
        for k, v in counts.items():
            lines.append(f"| `{k}` | {v:,} |")
        lines.append(f"| **subtotal** | **{sum(counts.values()):,}** |")
        return "\n".join(lines)

    src_lines = ["| source | folder | notes |", "| --- | --- | --- |"]
    for src, n in c1.items():
        src_lines.append(f"| `{src}` | `milestone1/` | {NOTES.get(src, src)} |")
    for src in c2:
        src_lines.append(f"| `{src}` | `milestone2/` | {NOTES.get(src, src)} |")
    for src in c3:
        src_lines.append(f"| `{src}` | `milestone3/` | {NOTES.get(src, src)} (live API suite) |")

    total = sum(c1.values()) + sum(c2.values()) + sum(c3.values())
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
  - milestone-3
size_categories:
  - 100K<n<1M
license: other
{chr(10).join(yaml)}
---

# router-bench

Open corpus for [Gittensor-TinyRouter](https://github.com/James-CUDA/Gittensor-TinyRouter)
by [James-Cuda](https://huggingface.co/James-Cuda).

## Three milestones

| Folder | Product goal | Predict / run | Needs |
| --- | --- | --- | --- |
| **`milestone1/`** | Prompt **triage** | **domain** + **difficulty** | CPU/GPU classifier; **no** API |
| **`milestone2/`** | **Model↔prompt scoring** / difficulty routing | which model (from scores or difficulty) | GPU optional; **no** live API |
| **`milestone3/`** | Full **TinyRouter** | **3 models × 3 roles** (Thinker/Worker/Verifier) | GPU + `OPENROUTER_API_KEY` |

You control how many domain labels and difficulty levels to keep — remesh on load.

## Milestone 1 — domain + difficulty

Gold: `domain_label`, `difficulty` (1–5 or remappable).  
Sources: clean provenance sets + Glint GCI topics + Supra / GatewayBench / query-complexity / pro domains.

## Milestone 2 — model–prompt scoring

Router corpora with **other pools' scores**, prompt indexes, or difficulty-for-cascade labels.  
Use to train/eval “pick a model” **without** calling OpenRouter. Build your own A/B/C cache later if needed.

## Milestone 3 — live 3×3 API

Held-out prompts with references for the real pool:

`qwen3.5-35b-a3b` · `gemini-3.1-flash-lite` · `deepseek-v4-flash`  
× roles Thinker / Worker / Verifier.

```bash
source ~/.config/trinity/secrets.env
CUDA_VISIBLE_DEVICES=5 python -m trinity.eval --benchmark ...  # or your live harness
```

## Sources

{chr(10).join(src_lines)}

## Counts

{table("milestone1/ (domain + difficulty)", c1)}

{table("milestone2/ (model–prompt scoring)", c2)}

{table("milestone3/ (live 3×3 API suite)", c3)}

| | rows |
| --- | ---: |
| **grand total (file subtotals)** | **{total:,}** |

## Schema

| column | meaning |
| --- | --- |
| `prompt` | user / problem text |
| `domain_label` | domain gold (M1; remesh freely) |
| `difficulty` | difficulty gold (remesh freely) |
| `reference` | answer when available |
| `milestone` | `"1"` / `"2"` / `"3"` |
| `metadata_json` | extras (model scores, etc.) |

## Load

```python
from datasets import load_dataset

m1 = load_dataset("James-Cuda/router-bench", "milestone1", split="train")
m2 = load_dataset("James-Cuda/router-bench", "milestone2", split="train")
m3 = load_dataset("James-Cuda/router-bench", "milestone3", split="train")

print(m1[0]["domain_label"], m1[0]["difficulty"])
```

## License

Upstream licenses apply per `source` (MIT / Apache-2.0 / GPL-3.0 for GCI / etc.).
"""
    (out / "README.md").write_text(text, encoding="utf-8")
    print(f"[readme] wrote {out / 'README.md'}")


def run(out: Path, *, skip_fetch: bool = False) -> None:
    m1, m2, m3 = out / "milestone1", out / "milestone2", out / "milestone3"
    for d in (m1, m2, m3):
        d.mkdir(parents=True, exist_ok=True)

    # Normalize milestone tags on existing M1/M2 files
    for pq in m1.glob("*.parquet"):
        if pq.name == "all.parquet":
            continue
        df = _ensure_milestone(pd.read_parquet(pq), "1")
        _save(df, pq)
    for pq in m2.glob("*.parquet"):
        if pq.name == "all.parquet":
            continue
        df = _ensure_milestone(pd.read_parquet(pq), "2")
        _save(df, pq)

    if not skip_fetch:
        for name, fn in M1_FETCHERS:
            dest = m1 / f"{name}.parquet"
            if dest.exists():
                print(f"[m1] resume {name}")
                continue
            print(f"[m1] fetch {name} ...")
            try:
                df = _ensure_milestone(fn(), "1")
                n = _save(df, dest)
                print(f"[m1] {name}: {n}")
            except Exception as e:
                print(f"[m1] FAIL {name}: {type(e).__name__}: {e}")

        for name, fn in M2_FETCHERS:
            dest = m2 / f"{name}.parquet"
            if dest.exists():
                print(f"[m2] resume {name}")
                continue
            print(f"[m2] fetch {name} ...")
            try:
                df = _ensure_milestone(fn(), "2")
                n = _save(df, dest)
                print(f"[m2] {name}: {n}")
            except Exception as e:
                print(f"[m2] FAIL {name}: {type(e).__name__}: {e}")

    # Build M3 live suite from M1 sources (copy with milestone=3)
    for src in M3_LIVE_SOURCES:
        src_pq = m1 / f"{src}.parquet"
        if not src_pq.exists():
            print(f"[m3] missing source {src}")
            continue
        df = _ensure_milestone(pd.read_parquet(src_pq), "3")
        # gpqa: prefer diamond if tagged
        if src == "gpqa" and "metadata_json" in df.columns:
            def is_diamond(s):
                try:
                    return "diamond" in json.loads(s).get("config", "").lower()
                except Exception:
                    return True
            mask = df["metadata_json"].map(is_diamond)
            if mask.any():
                df = df[mask]
        n = _save(df, m3 / f"{src}.parquet")
        print(f"[m3] {src}: {n}")

    c1 = _merge_folder(m1)
    c2 = _merge_folder(m2)
    c3 = _merge_folder(m3)
    (out / "counts_milestone1.json").write_text(json.dumps(c1, indent=2))
    (out / "counts_milestone2.json").write_text(json.dumps(c2, indent=2))
    (out / "counts_milestone3.json").write_text(json.dumps(c3, indent=2))
    write_readme(out, c1, c2, c3)


def push(out: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=str(out / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Three milestones: triage / scoring / live 3x3 API",
    )
    for name in ("counts_milestone1.json", "counts_milestone2.json", "counts_milestone3.json"):
        p = out / name
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p), path_in_repo=name, repo_id=repo_id,
                repo_type="dataset", commit_message=f"Update {name}",
            )
    for folder in ("milestone1", "milestone2", "milestone3"):
        api.upload_folder(
            folder_path=str(out / folder),
            path_in_repo=folder,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {folder}/",
        )
    print(f"[push] → https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/router-bench"))
    ap.add_argument("--repo-id", default="James-Cuda/router-bench")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument(
        "--heavy",
        action="store_true",
        help="Also fetch legalbench + medmcqa (slow / large)",
    )
    args = ap.parse_args()
    global M1_FETCHERS
    if args.heavy:
        M1_FETCHERS = M1_FETCHERS_LIGHT + M1_FETCHERS_HEAVY
    run(args.out, skip_fetch=args.skip_fetch)
    if args.push:
        push(args.out, args.repo_id)


if __name__ == "__main__":
    main()
