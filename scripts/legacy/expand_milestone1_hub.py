#!/usr/bin/env python3
"""Fetch open Hub datasets into milestone1/ (prompt + domain_label + difficulty).

Skips corpora that lack both domain and difficulty (or a clean map).
You can later remesh domain counts / difficulty levels as you like.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, create_repo

COLS = [
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

NO_ROBOTS_DIFF = {
    "Coding": "4",
    "Closed QA": "3",
    "Open QA": "2",
    "Classify": "2",
    "Extract": "2",
    "Summarize": "2",
    "Rewrite": "2",
    "Generation": "2",
    "Brainstorm": "1",
    "Chat": "1",
}

COMPLEXITY_DIFF = {"LOW": "2", "MEDIUM": "3", "HIGH": "5"}


def _clean(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _row(
    *,
    source: str,
    split: str,
    idx: int,
    prompt: str,
    domain_label: str,
    difficulty: str,
    reference: str = "",
    domain: str = "",
    metadata: dict | None = None,
) -> dict[str, str]:
    lab = _clean(domain_label)
    return {
        "id": f"{source}:{split}:{idx}",
        "source": source,
        "split": split,
        "prompt": _clean(prompt),
        "domain_label": lab,
        "difficulty": _clean(difficulty),
        "provenance_label": lab,
        "reference": _clean(reference),
        "domain": _clean(domain) or lab,
        "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        "milestone": "1",
    }


def _df(rows: list[dict[str, str]]) -> pd.DataFrame:
    rows = [r for r in rows if r["prompt"] and r["domain_label"] and r["difficulty"]]
    return pd.DataFrame(rows)[COLS] if rows else pd.DataFrame(columns=COLS)


def load_supra() -> pd.DataFrame:
    raw = load_dataset("SupraLabs/Prompt-Routing-Dataset", split="train")
    rows = []
    for i, ex in enumerate(raw):
        rows.append(
            _row(
                source="supra_prompt_routing",
                split="train",
                idx=i,
                prompt=ex["prompt"],
                domain_label=ex["primary_domain"],
                difficulty=str(int(ex["complexity_score"])),
                reference=ex.get("full_answer") or "",
                domain=ex["primary_domain"],
                metadata={
                    "coding_task": bool(ex.get("coding_task")),
                    "math_task": bool(ex.get("math_task")),
                    "routing_choice": ex.get("routing_choice"),
                },
            )
        )
    return _df(rows)


def load_query_complexity() -> pd.DataFrame:
    raw = load_dataset("anasnassar/llm-query-complexity-benchmark")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            gt = _clean(ex.get("ground_truth")).upper()
            rows.append(
                _row(
                    source="query_complexity",
                    split=str(split),
                    idx=i,
                    prompt=ex["text"],
                    domain_label=ex["domain"],
                    difficulty=COMPLEXITY_DIFF.get(gt, "3"),
                    domain=ex.get("subject") or ex["domain"],
                    metadata={"upstream_source": ex.get("source"), "ground_truth": gt},
                )
            )
    return _df(rows)


def load_gatewaybench() -> pd.DataFrame:
    raw = load_dataset("ModaLabs/GatewayBench-v1", "full", split="train")
    rows = []
    for i, ex in enumerate(raw):
        meta = ex.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        rows.append(
            _row(
                source="gatewaybench",
                split="train",
                idx=i,
                prompt=ex["user_prompt"],
                domain_label=meta.get("domain") or ex.get("task_type") or "general",
                difficulty=str(int(meta.get("difficulty") or 3)),
                reference=ex.get("reference_answer") or "",
                domain=meta.get("domain") or "",
                metadata={"task_type": ex.get("task_type"), "gw_id": ex.get("id")},
            )
        )
    return _df(rows)


def load_no_robots() -> pd.DataFrame:
    raw = load_dataset("HuggingFaceH4/no_robots")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            cat = _clean(ex.get("category")) or "Generation"
            rows.append(
                _row(
                    source="no_robots",
                    split=str(split),
                    idx=i,
                    prompt=ex["prompt"],
                    domain_label=cat,
                    difficulty=NO_ROBOTS_DIFF.get(cat, "2"),
                    domain=cat,
                    metadata={"prompt_id": ex.get("prompt_id")},
                )
            )
    return _df(rows)


def load_pubmedqa() -> pd.DataFrame:
    raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    for i, ex in enumerate(raw):
        rows.append(
            _row(
                source="pubmedqa",
                split="train",
                idx=i,
                prompt=ex["question"],
                domain_label="medical",
                difficulty="4",
                reference=_clean(ex.get("final_decision")),
                domain="pubmedqa",
                metadata={"pubid": ex.get("pubid")},
            )
        )
    return _df(rows)


def load_medmcqa(*, max_rows: int | None = None) -> pd.DataFrame:
    raw = load_dataset("openlifescienceai/medmcqa")
    rows = []
    n = 0
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            if max_rows is not None and n >= max_rows:
                break
            subj = _clean(ex.get("subject_name")) or "medical"
            q = _clean(ex.get("question"))
            opts = "\n".join(
                f"{L}. {_clean(ex.get(k))}"
                for L, k in zip("ABCD", ("opa", "opb", "opc", "opd"))
                if _clean(ex.get(k))
            )
            prompt = f"{q}\n\n{opts}" if opts else q
            ans_i = ex.get("cop")
            ref = ""
            try:
                ref = "ABCD"[int(ans_i)]
            except Exception:
                ref = _clean(ans_i)
            rows.append(
                _row(
                    source="medmcqa",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    domain_label=subj,
                    difficulty="3",
                    reference=ref,
                    domain=_clean(ex.get("topic_name")) or subj,
                )
            )
            n += 1
        if max_rows is not None and n >= max_rows:
            break
    return _df(rows)


def load_finqa() -> pd.DataFrame:
    raw = load_dataset("ChanceFocus/flare-finqa")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            # prefer query; fall back to text
            prompt = _clean(ex.get("query")) or _clean(ex.get("text"))
            rows.append(
                _row(
                    source="finqa",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    domain_label="finance",
                    difficulty="4",
                    reference=_clean(ex.get("answer")),
                    domain="finqa",
                    metadata={"flare_id": ex.get("id")},
                )
            )
    return _df(rows)


def load_hendrycks_math() -> pd.DataFrame:
    rows = []
    for cfg in get_dataset_config_names("EleutherAI/hendrycks_math"):
        raw = load_dataset("EleutherAI/hendrycks_math", cfg)
        for split, ds in raw.items():
            for i, ex in enumerate(ds):
                level = _clean(ex.get("level"))
                m = re.search(r"(\d+)", level)
                diff = m.group(1) if m else "3"
                diff = str(min(5, max(1, int(diff))))
                rows.append(
                    _row(
                        source="hendrycks_math",
                        split=str(split),
                        idx=len(rows),
                        prompt=ex["problem"],
                        domain_label=_clean(ex.get("type")) or cfg,
                        difficulty=diff,
                        reference=_clean(ex.get("solution")),
                        domain=cfg,
                        metadata={"level_raw": level},
                    )
                )
    return _df(rows)


def load_legalbench(*, max_rows: int | None = None) -> pd.DataFrame:
    rows = []
    cfgs = get_dataset_config_names("nguha/legalbench")
    for cfg in cfgs:
        try:
            raw = load_dataset("nguha/legalbench", cfg)
        except Exception as e:
            print(f"[m1] legalbench skip {cfg}: {e}")
            continue
        for split, ds in raw.items():
            for i, ex in enumerate(ds):
                if max_rows is not None and len(rows) >= max_rows:
                    return _df(rows)
                text = _clean(ex.get("text") or ex.get("question") or ex.get("input"))
                if not text:
                    continue
                rows.append(
                    _row(
                        source="legalbench",
                        split=str(split),
                        idx=len(rows),
                        prompt=text,
                        domain_label=cfg,  # fine legal task as domain; remesh later
                        difficulty="3",
                        reference=_clean(ex.get("answer")),
                        domain="legal",
                        metadata={"legalbench_config": cfg},
                    )
                )
    return _df(rows)


LOADERS: list[tuple[str, Any]] = [
    ("supra_prompt_routing", load_supra),
    ("query_complexity", load_query_complexity),
    ("gatewaybench", load_gatewaybench),
    ("no_robots", load_no_robots),
    ("pubmedqa", load_pubmedqa),
    ("finqa", load_finqa),
    ("hendrycks_math", load_hendrycks_math),
    ("legalbench", lambda: load_legalbench(max_rows=None)),
    ("medmcqa", lambda: load_medmcqa(max_rows=None)),
]


def expand(out_dir: Path, *, only: set[str] | None = None) -> dict[str, int]:
    m1 = out_dir / "milestone1"
    m1.mkdir(parents=True, exist_ok=True)
    new_counts: dict[str, int] = {}
    for name, fn in LOADERS:
        if only and name not in only:
            continue
        dest = m1 / f"{name}.parquet"
        print(f"[m1] loading {name} ...")
        try:
            df = fn()
        except Exception as e:
            print(f"[m1] FAIL {name}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            print(f"[m1] empty {name}")
            continue
        Dataset.from_pandas(df, preserve_index=False).to_parquet(str(dest))
        new_counts[name] = len(df)
        print(
            f"[m1] {name}: {len(df)}  "
            f"domains={df['domain_label'].nunique()}  "
            f"diff={df['difficulty'].value_counts().to_dict()}"
        )
    return new_counts


def rebuild_all(out_dir: Path) -> dict[str, int]:
    m1 = out_dir / "milestone1"
    parts: list[Dataset] = []
    counts: dict[str, int] = {}
    for pq in sorted(m1.glob("*.parquet")):
        if pq.name == "all.parquet":
            continue
        ds = Dataset.from_parquet(str(pq))
        counts[pq.stem] = len(ds)
        parts.append(ds)
        print(f"[m1] keep {pq.stem}: {len(ds)}")
    if not parts:
        raise SystemExit("no milestone1 parts")
    merged = concatenate_datasets(parts).shuffle(seed=271828182)
    merged.to_parquet(str(m1 / "all.parquet"))
    print(f"[m1] all.parquet: {len(merged)}")
    (out_dir / "counts_milestone1.json").write_text(json.dumps(counts, indent=2))
    return counts


def push_m1(out_dir: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reorganize_router_bench_milestones import write_readme

    c1 = json.loads((out_dir / "counts_milestone1.json").read_text())
    c2 = {}
    c2p = out_dir / "counts_milestone2.json"
    if c2p.exists():
        c2 = json.loads(c2p.read_text())
    # ensure notes exist for new sources
    from reorganize_router_bench_milestones import DOMAIN_NOTES

    DOMAIN_NOTES.update(
        {
            "supra_prompt_routing": "mixed — SupraLabs/Prompt-Routing-Dataset (MIT)",
            "query_complexity": "mixed — anasnassar/llm-query-complexity-benchmark (Apache-2.0)",
            "gatewaybench": "mixed — ModaLabs/GatewayBench-v1 full (MIT)",
            "no_robots": "instruction — HuggingFaceH4/no_robots categories",
            "pubmedqa": "medical — qiaojin/PubMedQA",
            "finqa": "finance — ChanceFocus/flare-finqa",
            "hendrycks_math": "math — EleutherAI/hendrycks_math (level 1–5)",
            "legalbench": "legal — nguha/legalbench (task=domain_label)",
            "medmcqa": "medical — openlifescienceai/medmcqa (subject=domain_label)",
            "gci_bench": DOMAIN_NOTES.get("gci_bench", "topic — Glint GCI_Bench"),
        }
    )
    write_readme(out_dir, c1, c2)
    api.upload_file(
        path_or_fileobj=str(out_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Expand milestone1 with open Hub triage corpora",
    )
    api.upload_file(
        path_or_fileobj=str(out_dir / "counts_milestone1.json"),
        path_in_repo="counts_milestone1.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update milestone1 counts",
    )
    api.upload_folder(
        folder_path=str(out_dir / "milestone1"),
        path_in_repo="milestone1",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add expanded milestone1 Hub corpora",
    )
    print(f"[push] → https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/router-bench"))
    ap.add_argument("--repo-id", default="James-Cuda/router-bench")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    expand(args.out, only=only)
    rebuild_all(args.out)
    if args.push:
        push_m1(args.out, args.repo_id)


if __name__ == "__main__":
    main()
