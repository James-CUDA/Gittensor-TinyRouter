#!/usr/bin/env python3
"""Build data2/ expansion for James-Cuda/router-bench and push to the Hub.

Adds Priority-1 router corpora + Priority-2 hard-task sets under data2/,
keeps existing data/ untouched, and refreshes README configs.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download


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
    rows = [r for r in rows if r["prompt"]]
    return Dataset.from_list(rows)


def _label_from_name(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("code", "humaneval", "mbpp", "bigcode", "livecode")):
        return "code"
    if any(x in n for x in ("math", "gsm", "aime", "aqua")):
        return "math"
    if any(x in n for x in ("mmlu", "gpqa", "arc", "knowledge", "truthful")):
        return "knowledge"
    if any(x in n for x in ("hellaswag", "winogrande", "commonsense")):
        return "commonsense"
    if any(x in n for x in ("ifeval", "instruct", "chat", "arena")):
        return "instruction"
    return "other"


# -------------------- Priority-1 loaders --------------------


def load_embedllm() -> Dataset:
    """Unique prompts from EmbedLLM (dedupe by prompt_id)."""
    paths = {
        "train": hf_hub_download("RZ412/EmbedLLM", "train.csv", repo_type="dataset"),
        "validation": hf_hub_download("RZ412/EmbedLLM", "val.csv", repo_type="dataset"),
        "test": hf_hub_download("RZ412/EmbedLLM", "test.csv", repo_type="dataset"),
    }
    rows = []
    seen: set[int] = set()
    for split, path in paths.items():
        df = pd.read_csv(path)
        # one row per prompt_id (first occurrence keeps category)
        for _, ex in df.drop_duplicates("prompt_id").iterrows():
            pid = int(ex["prompt_id"])
            if pid in seen:
                continue
            seen.add(pid)
            cat = _clean(ex.get("category"))
            rows.append(
                _row(
                    source="embedllm",
                    split=split,
                    idx=pid,
                    prompt=_clean(ex.get("prompt")),
                    provenance_label=_label_from_name(cat),
                    domain=cat,
                    metadata={"prompt_id": pid, "category_id": int(ex.get("category_id", -1))},
                )
            )
    return _from_rows(rows)


def _sprout_scores(ex: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k, v in ex.items():
        if k in ("key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"):
            continue
        try:
            if isinstance(v, dict):
                jr = v.get("judge_response")
                if isinstance(jr, str):
                    jr = json.loads(jr)
                if isinstance(jr, dict) and "correctness_score" in jr:
                    scores[k] = float(jr["correctness_score"])
            elif isinstance(v, str) and "correctness_score" in v:
                # nested JSON-as-string
                try:
                    obj = json.loads(v)
                    jr = obj.get("judge_response", obj)
                    if isinstance(jr, str):
                        jr = json.loads(jr)
                    if isinstance(jr, dict) and "correctness_score" in jr:
                        scores[k] = float(jr["correctness_score"])
                except Exception:
                    pass
        except Exception:
            continue
    return scores


def load_sprout() -> Dataset:
    raw = load_dataset("CARROT-LLM-Routing/SPROUT")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            rows.append(
                _row(
                    source="sprout",
                    split=str(split),
                    idx=i,
                    prompt=_clean(ex.get("prompt")),
                    provenance_label=_label_from_name(_clean(ex.get("dataset"))),
                    reference=_clean(ex.get("golden_answer")),
                    domain=_clean(ex.get("dataset")),
                    metadata={
                        "key": ex.get("key"),
                        "dataset_idx": ex.get("dataset_idx"),
                        "model_scores": _sprout_scores(ex),
                    },
                )
            )
    return _from_rows(rows)


def load_mix_instruct() -> Dataset:
    files = {
        "train": "train_data_prepared.jsonl",
        "validation": "val_data_prepared.jsonl",
        "test": "test_data_prepared.jsonl",
    }
    rows = []
    for split, fname in files.items():
        path = hf_hub_download("llm-blender/mix-instruct", fname, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                instr = _clean(ex.get("instruction"))
                inp = _clean(ex.get("input"))
                prompt = f"{instr}\n\n{inp}".strip() if instr and inp else (instr or inp)
                cands = ex.get("candidates") or []
                models = []
                if isinstance(cands, list):
                    for c in cands[:20]:
                        if isinstance(c, dict) and c.get("model"):
                            models.append(c["model"])
                rows.append(
                    _row(
                        source="mix_instruct",
                        split=split,
                        idx=i,
                        prompt=prompt,
                        provenance_label="instruction",
                        reference=_clean(ex.get("output")),
                        domain=_clean((ex.get("id") or "").split("/")[0]),
                        metadata={
                            "orig_id": ex.get("id"),
                            "candidate_models": models,
                            "n_candidates": len(cands) if isinstance(cands, list) else 0,
                        },
                    )
                )
    return _from_rows(rows)


def load_llmrouterbench() -> Dataset:
    """Extract unique prompts from NPULH/LLMRouterBench release tarball (~1.2GB).

    Each JSON is a *model result* file with top-level ``records`` list. Prompts live
    in ``records[i].prompt`` / ``origin_query``. We keep one file per ``dataset_name``
    (first seen) to avoid re-reading the same questions across 30+ models.
    """
    print("[data2] downloading LLMRouterBench tar (large) ...")
    tar_path = hf_hub_download(
        "NPULH/LLMRouterBench", "bench-release.tar.gz", repo_type="dataset"
    )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    seen_datasets: set[str] = set()
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")]
        print(f"[data2] LLMRouterBench json files: {len(members)}")
        for mi, m in enumerate(members):
            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                obj = json.load(f)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            parts = Path(m.name).parts
            fallback = parts[-3] if len(parts) >= 3 else ""
            dset = _clean(obj.get("dataset_name") or fallback)
            if dset and dset in seen_datasets:
                continue
            records = obj.get("records")
            if not isinstance(records, list) or not records:
                continue
            if dset:
                seen_datasets.add(dset)
            split = _clean(obj.get("split") or "test")
            for ex in records:
                if not isinstance(ex, dict):
                    continue
                prompt = _clean(
                    ex.get("prompt")
                    or ex.get("origin_query")
                    or ex.get("question")
                    or ex.get("input")
                    or ""
                )
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                rows.append(
                    _row(
                        source="llmrouterbench",
                        split=split or "test",
                        idx=len(rows),
                        prompt=prompt,
                        provenance_label=_label_from_name(dset or m.name),
                        reference=_clean(
                            ex.get("answer")
                            or ex.get("target")
                            or ex.get("reference")
                            or ex.get("gold")
                            or ""
                        ),
                        domain=dset or "llmrouterbench",
                        metadata={
                            "index": ex.get("index"),
                            "tar_member": m.name,
                            "model_name_sample": obj.get("model_name"),
                        },
                    )
                )
            if (mi + 1) % 50 == 0:
                print(
                    f"[data2] scanned {mi+1}/{len(members)} files, "
                    f"datasets={len(seen_datasets)}, prompts={len(rows)}"
                )
    print(
        f"[data2] LLMRouterBench unique prompts: {len(rows)} "
        f"across {len(seen_datasets)} datasets"
    )
    return _from_rows(rows)


# -------------------- Priority-2 loaders --------------------


def load_mmlu_pro() -> Dataset:
    raw = load_dataset("TIGER-Lab/MMLU-Pro")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            opts = ex.get("options") or []
            if isinstance(opts, list):
                labeled = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(opts))
            else:
                labeled = _clean(opts)
            q = _clean(ex.get("question"))
            prompt = f"{q}\n\n{labeled}".strip()
            ans = ex.get("answer")
            if isinstance(ans, int):
                ref = chr(65 + ans) if 0 <= ans < 26 else str(ans)
            else:
                ref = _clean(ans)
            rows.append(
                _row(
                    source="mmlu_pro",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    provenance_label="knowledge",
                    reference=ref,
                    domain=_clean(ex.get("category") or ex.get("src")),
                    metadata={"question_id": ex.get("question_id")},
                )
            )
    return _from_rows(rows)


def load_aime() -> Dataset:
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _row(
                source="aime_2024",
                split="train",
                idx=i,
                prompt=_clean(ex.get("problem") or ex.get("question")),
                provenance_label="math",
                reference=_clean(ex.get("answer") or ex.get("solution")),
                domain="aime",
                metadata={"id": ex.get("id")},
            )
        )
    return _from_rows(rows)


def load_bigcodebench() -> Dataset:
    # try common configs
    try:
        raw = load_dataset("bigcode/bigcodebench", "default")
    except Exception:
        raw = load_dataset("bigcode/bigcodebench")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            prompt = _clean(
                ex.get("instruct_prompt")
                or ex.get("complete_prompt")
                or ex.get("prompt")
                or ex.get("question")
            )
            rows.append(
                _row(
                    source="bigcodebench",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    provenance_label="code",
                    reference=_clean(ex.get("canonical_solution") or ex.get("solution") or ""),
                    domain="code",
                    metadata={"task_id": ex.get("task_id") or ex.get("id")},
                )
            )
    return _from_rows(rows)


def load_ifeval() -> Dataset:
    ds = load_dataset("google/IFEval", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _row(
                source="ifeval",
                split="train",
                idx=i,
                prompt=_clean(ex.get("prompt")),
                provenance_label="instruction",
                reference="",
                domain="ifeval",
                metadata={
                    "key": ex.get("key"),
                    "instruction_id_list": ex.get("instruction_id_list"),
                },
            )
        )
    return _from_rows(rows)


def load_hellaswag() -> Dataset:
    raw = load_dataset("Rowan/hellaswag")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            endings = ex.get("endings") or []
            ctx = _clean(ex.get("ctx") or ex.get("ctx_a"))
            labeled = "\n".join(f"{chr(65+j)}. {_clean(e)}" for j, e in enumerate(endings))
            prompt = f"{ctx}\n\n{labeled}".strip()
            label = ex.get("label")
            ref = ""
            try:
                ref = chr(65 + int(label))
            except Exception:
                ref = _clean(label)
            rows.append(
                _row(
                    source="hellaswag",
                    split=str(split),
                    idx=i,
                    prompt=prompt,
                    provenance_label="commonsense",
                    reference=ref,
                    domain="hellaswag",
                )
            )
    return _from_rows(rows)


def load_arc() -> Dataset:
    rows = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        raw = load_dataset("allenai/ai2_arc", cfg)
        for split, ds in raw.items():
            for i, ex in enumerate(ds):
                choices = ex.get("choices") or {}
                texts = choices.get("text") if isinstance(choices, dict) else None
                labels = choices.get("label") if isinstance(choices, dict) else None
                if texts and labels:
                    labeled = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
                else:
                    labeled = ""
                prompt = f"{_clean(ex.get('question'))}\n\n{labeled}".strip()
                rows.append(
                    _row(
                        source="arc",
                        split=f"{cfg}:{split}",
                        idx=i,
                        prompt=prompt,
                        provenance_label="knowledge",
                        reference=_clean(ex.get("answerKey")),
                        domain=cfg,
                        metadata={"id": ex.get("id")},
                    )
                )
    return _from_rows(rows)


def load_truthfulqa() -> Dataset:
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rows = []
    for i, ex in enumerate(ds):
        rows.append(
            _row(
                source="truthfulqa",
                split="validation",
                idx=i,
                prompt=_clean(ex.get("question")),
                provenance_label="knowledge",
                reference=_clean(
                    ex.get("best_answer")
                    or (
                        (ex.get("correct_answers") or [""])[0]
                        if isinstance(ex.get("correct_answers"), list)
                        else ""
                    )
                ),
                domain=_clean(ex.get("category")),
                metadata={"type": ex.get("type")},
            )
        )
    return _from_rows(rows)


def load_gpqa() -> Dataset:
    """Gated dataset — requires HF_TOKEN with access granted."""
    # try common configs
    for cfg in ("gpqa_diamond", "gpqa_main", "gpqa_extended", None):
        try:
            if cfg:
                raw = load_dataset("Idavidrein/gpqa", cfg)
            else:
                raw = load_dataset("Idavidrein/gpqa")
            break
        except Exception as e:
            last = e
            raw = None
    if raw is None:
        raise RuntimeError(f"GPQA load failed (gated?): {last}")
    rows = []
    for split, ds in raw.items():
        for i, ex in enumerate(ds):
            # schema varies
            q = _clean(
                ex.get("Question")
                or ex.get("question")
                or ex.get("Problem")
            )
            rows.append(
                _row(
                    source="gpqa",
                    split=str(split),
                    idx=i,
                    prompt=q,
                    provenance_label="knowledge",
                    reference=_clean(
                        ex.get("Correct Answer")
                        or ex.get("answer")
                        or ex.get("Correct_Answer")
                        or ""
                    ),
                    domain=_clean(ex.get("Subdomain") or ex.get("High-level domain") or "gpqa"),
                )
            )
    return _from_rows(rows)


def load_routereval() -> Dataset:
    """Thin unique-prompt index from linggm/RouterEval (one detail zip per subset).

    Full RouterEval is 50k+ model-detail zips. We download the *smallest* zip under
    each ``full_data/<subset>/`` and extract ``example`` strings from the harness
    pickle — enough for Milestone-1 prompt routing without vendoring scores.
    """
    import pickle
    import zipfile

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    repo_id = "linggm/RouterEval"
    # list subset directories under full_data/
    tree = list(
        api.list_repo_tree(
            repo_id,
            path_in_repo="full_data",
            repo_type="dataset",
            recursive=False,
        )
    )
    subsets = sorted(
        p.path.split("/", 1)[1]
        for p in tree
        if getattr(p, "path", None) and str(p.path).startswith("full_data/")
        and "/" not in str(p.path)[len("full_data/") :]
    )
    if not subsets:
        # fallback known list if tree API shape differs
        subsets = [
            "arc_challenge",
            "bbh",
            "gpqa",
            "gsm8k",
            "hellaswag",
            "ifeval",
            "math_lv_5",
            "mmlu",
            "mmlu_pro",
            "musr",
            "truthfulqa",
            "winogrande",
        ]

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for subset in subsets:
        prefix = f"full_data/{subset}"
        try:
            files = list(
                api.list_repo_tree(
                    repo_id,
                    path_in_repo=prefix,
                    repo_type="dataset",
                    recursive=False,
                )
            )
        except Exception as e:
            print(f"[data2] routereval: list {subset} failed: {e}")
            continue
        zips = [
            f
            for f in files
            if str(getattr(f, "path", "")).endswith(".zip")
        ]
        if not zips:
            print(f"[data2] routereval: no zips in {subset}")
            continue
        # Prefer mid-size zips: tiniest shards are often empty/corrupt (~100–200 B).
        zips.sort(key=lambda f: getattr(f, "size", None) or 10**18)
        # try up to 8 candidates from the lower half of the size distribution
        candidates = [z for z in zips if (getattr(z, "size", None) or 0) >= 50_000][:8]
        if not candidates:
            candidates = zips[:8]

        added_here = 0
        for chosen in candidates:
            rel = chosen.path
            print(
                f"[data2] routereval: try {subset} ← {rel} "
                f"({getattr(chosen, 'size', '?')} B)"
            )
            try:
                local = hf_hub_download(
                    repo_id, rel, repo_type="dataset", token=token
                )
            except Exception as e:
                print(f"[data2] routereval: download failed {rel}: {e}")
                continue
            try:
                with zipfile.ZipFile(local) as zf:
                    names = zf.namelist()
                    if not names:
                        continue
                    obj = pickle.loads(zf.read(names[0]))
            except Exception as e:
                print(f"[data2] routereval: pickle failed {rel}: {e}")
                continue
            if not isinstance(obj, dict):
                continue
            batch: list[dict[str, str]] = []
            for harness_key, block in obj.items():
                if not isinstance(block, dict):
                    continue
                examples = block.get("example") or []
                for i, ex in enumerate(examples):
                    # example is often [question, answer] or a bare string
                    if isinstance(ex, (list, tuple)) and ex:
                        prompt = _clean(ex[0])
                        ref = _clean(ex[1]) if len(ex) > 1 else ""
                    else:
                        prompt = _clean(ex)
                        ref = ""
                    if not prompt or prompt in seen:
                        continue
                    seen.add(prompt)
                    batch.append(
                        _row(
                            source="routereval",
                            split=subset,
                            idx=len(rows) + len(batch),
                            prompt=prompt,
                            provenance_label=_label_from_name(subset),
                            reference=ref,
                            domain=subset,
                            metadata={
                                "harness_key": str(harness_key),
                                "detail_zip": rel,
                                "example_index": i,
                            },
                        )
                    )
            if batch:
                rows.extend(batch)
                added_here = len(batch)
                print(f"[data2] routereval: {subset} +{added_here} prompts")
                break
        if added_here == 0:
            print(f"[data2] routereval: WARNING no prompts for {subset}")
    return _from_rows(rows)


LOADERS = (
    ("embedllm", load_embedllm),
    ("sprout", load_sprout),
    ("mix_instruct", load_mix_instruct),
    ("llmrouterbench", load_llmrouterbench),
    ("mmlu_pro", load_mmlu_pro),
    ("aime_2024", load_aime),
    ("bigcodebench", load_bigcodebench),
    ("ifeval", load_ifeval),
    ("hellaswag", load_hellaswag),
    ("arc", load_arc),
    ("truthfulqa", load_truthfulqa),
    ("gpqa", load_gpqa),
    ("routereval", load_routereval),
)


NOTES = {
    "embedllm": ("mixed", "EmbedLLM unique prompts (RZ412/EmbedLLM)"),
    "sprout": ("mixed", "CARROT SPROUT multi-model scored prompts"),
    "mix_instruct": ("instruction", "LLM-Blender MixInstruct"),
    "llmrouterbench": ("mixed", "LLMRouterBench unique prompts from release tarball"),
    "mmlu_pro": ("knowledge", "TIGER-Lab/MMLU-Pro"),
    "aime_2024": ("math", "HuggingFaceH4/aime_2024"),
    "bigcodebench": ("code", "bigcode/bigcodebench"),
    "ifeval": ("instruction", "google/IFEval"),
    "hellaswag": ("commonsense", "Rowan/hellaswag"),
    "arc": ("knowledge", "allenai/ai2_arc Challenge+Easy"),
    "truthfulqa": ("knowledge", "truthfulqa/truthful_qa generation"),
    "gpqa": ("knowledge", "Idavidrein/gpqa (gated)"),
    "routereval": (
        "mixed",
        "RouterEval thin prompt index (1 detail zip / subset; linggm/RouterEval)",
    ),
}


def build_data2(out_dir: Path, skip: set[str], *, resume: bool = True) -> dict[str, int]:
    data2 = out_dir / "data2"
    data2.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    parts: list[Dataset] = []
    for name, fn in LOADERS:
        dest = data2 / f"{name}.parquet"
        if name in skip:
            print(f"[data2] skip {name}")
            if resume and dest.exists():
                ds = Dataset.from_parquet(str(dest))
                counts[name] = len(ds)
                parts.append(ds)
                print(f"[data2]   kept existing {name}: {len(ds)} rows")
            continue
        if resume and dest.exists():
            ds = Dataset.from_parquet(str(dest))
            print(f"[data2] resume {name}: {len(ds)} rows (existing parquet)")
            counts[name] = len(ds)
            parts.append(ds)
            continue
        print(f"[data2] loading {name} ...")
        try:
            ds = fn()
        except Exception as e:
            print(f"[data2] WARNING: failed {name}: {type(e).__name__}: {e}")
            continue
        print(f"[data2]   {name}: {len(ds)} rows")
        if len(ds) == 0:
            print(f"[data2] WARNING: empty {name}, skipping file")
            continue
        ds.to_parquet(str(dest))
        counts[name] = len(ds)
        parts.append(ds)
    if parts:
        merged = concatenate_datasets(parts).shuffle(seed=271828182)
        merged.to_parquet(str(data2 / "all.parquet"))
        print(f"[data2] wrote data2/all.parquet ({len(merged)} rows)")
    with open(out_dir / "counts_data2.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    return counts


def merge_readme(out_dir: Path, counts2: dict[str, int]) -> None:
    """Rewrite README with data/ + data2/ configs and fixed tables."""
    counts1 = {}
    c1 = out_dir / "counts.json"
    if c1.exists():
        counts1 = json.loads(c1.read_text())

    def cfg_block(folder: str, name: str, fname: str) -> list[str]:
        return [
            f"  - config_name: {name}",
            "    data_files:",
            "      - split: train",
            f"        path: {folder}/{fname}",
        ]

    yaml_lines = [
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        "        path: data/all.parquet",
        "  - config_name: data2",
        "    data_files:",
        "      - split: train",
        "        path: data2/all.parquet",
    ]
    for src in counts1:
        yaml_lines.extend(cfg_block("data", src, f"{src}.parquet"))
    for src in counts2:
        yaml_lines.extend(cfg_block("data2", f"data2_{src}", f"{src}.parquet"))

    def table_sources(counts: dict[str, int], folder: str) -> str:
        lines = [
            "| source | folder | provenance_label | notes |",
            "| --- | --- | --- | --- |",
        ]
        for src, n in counts.items():
            lab, note = NOTES.get(src, ("mixed", src))
            # data1 notes from original
            if folder == "data":
                note = src
                lab = "see v1"
            lines.append(f"| `{src}` | `{folder}/` | {lab} | {note} |")
        return "\n".join(lines)

    def table_counts(title: str, counts: dict[str, int]) -> str:
        lines = [
            f"### {title}",
            "",
            "| source | rows |",
            "| --- | ---: |",
        ]
        for k, v in counts.items():
            lines.append(f"| `{k}` | {v:,} |")
        lines.append(f"| **subtotal** | **{sum(counts.values()):,}** |")
        return "\n".join(lines)

    # richer data1 source table
    data1_notes = {
        "dolly-15k": ("instruction", "Databricks Dolly"),
        "mbpp": ("code", "MBPP"),
        "humaneval": ("code", "OpenAI HumanEval"),
        "aqua_rat": ("math", "AQuA-RAT"),
        "gsm8k": ("math", "GSM8K"),
        "mmlu": ("knowledge", "CAIS MMLU"),
        "math500": ("math", "MATH-500"),
        "routerbench": ("mixed", "Martian RouterBench"),
    }
    src_lines = [
        "| source | folder | provenance_label | notes |",
        "| --- | --- | --- | --- |",
    ]
    for src in counts1:
        lab, note = data1_notes.get(src, ("mixed", src))
        src_lines.append(f"| `{src}` | `data/` | {lab} | {note} |")
    for src in counts2:
        lab, note = NOTES.get(src, ("mixed", src))
        src_lines.append(f"| `{src}` | `data2/` | {lab} | {note} |")

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
size_categories:
  - 100K<n<1M
license: other
{chr(10).join(yaml_lines)}
---

# router-bench

Unified **open LLM-router evaluation corpus** for TinyRouter Milestone 1
(no live API required for triage / cached-routing experiments).

Built by [James-Cuda](https://huggingface.co/James-Cuda) for
[Gittensor-TinyRouter](https://github.com/James-CUDA/Gittensor-TinyRouter).

- `data/` — original Milestone-1 merge
- `data2/` — expansion: more router benches + harder open tasks

> Note: `data2_routereval` is a **thin unique-prompt index** (one detail zip per
> [RouterEval](https://huggingface.co/datasets/linggm/RouterEval) subset). Full
> score matrices stay upstream.

## Sources

{chr(10).join(src_lines)}

## Counts

{table_counts("data/ (v1)", counts1)}

{table_counts("data2/ (expansion)", counts2)}

| | rows |
| --- | ---: |
| **grand total (v1 + data2 unique files)** | **{total:,}** |

## Schema

Unified prompt rows for open LLM-router evaluation (Milestone 1).

| column | meaning |
| --- | --- |
| `id` | unique row id (`source:split:local_index`) |
| `source` | origin dataset name |
| `split` | train / validation / test / ... |
| `prompt` | user / problem text |
| `provenance_label` | coarse label (`code` / `math` / `knowledge` / ...) |
| `reference` | gold answer when available |
| `domain` | finer tag (subject / eval name) |
| `metadata_json` | source-specific JSON |

## Load

```python
from datasets import load_dataset

# Original merge
v1 = load_dataset("James-Cuda/router-bench", split="train")

# data2 merge
v2 = load_dataset("James-Cuda/router-bench", "data2", split="train")

# One source from data2
sprout = load_dataset("James-Cuda/router-bench", "data2_sprout", split="train")
mmlu_pro = load_dataset("James-Cuda/router-bench", "data2_mmlu_pro", split="train")
```

## License

Upstream licenses apply per `source`. This repo redistributes prompts/labels needed
for router evaluation; respect each source license for redistribution.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")
    print(f"[data2] wrote README ({total} combined row refs)")


def push_data2(out_dir: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    # upload README + counts_data2 + data2/
    api.upload_file(
        path_or_fileobj=str(out_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update README for data2 expansion",
    )
    if (out_dir / "counts_data2.json").exists():
        api.upload_file(
            path_or_fileobj=str(out_dir / "counts_data2.json"),
            path_in_repo="counts_data2.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add data2 counts",
        )
    api.upload_folder(
        folder_path=str(out_dir / "data2"),
        path_in_repo="data2",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add data2 router-bench expansion corpora",
    )
    print(f"[data2] pushed → https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/router-bench"))
    ap.add_argument("--repo-id", default="James-Cuda/router-bench")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip LLMRouterBench 1.2GB tarball",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Rebuild even if data2/<source>.parquet already exists",
    )
    args = ap.parse_args()
    skip = set(args.skip)
    if args.skip_heavy:
        skip.add("llmrouterbench")
    counts2 = build_data2(args.out, skip, resume=not args.no_resume)
    merge_readme(args.out, counts2)
    if args.push:
        push_data2(args.out, args.repo_id)


if __name__ == "__main__":
    main()
