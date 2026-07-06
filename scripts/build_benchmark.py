#!/usr/bin/env python3
"""Build the hidden benchmark for the TinyRouter accuracy competition.

Creates the benchmark files that pr_eval.py loads. This is a ONE-TIME
maintainer operation — after building, the benchmark is stored encrypted
outside the repo and must never be modified or revealed.

Usage:
    source ~/.config/trinity/secrets.env
    export BENCHMARK_PASSWORD=<strong-password>
    python scripts/build_benchmark.py --benchmark math500

Output:
    $TINYROUTER_BENCHMARK_DIR/math500/
        eval.json     # 150 questions with cached model answers
        audit.json    # 50 questions with cached model answers
        live.json     # 20 questions (no cached answers — live API eval)

The seed is FIXED and committed — it determines the exact question set forever.
Changing the seed after the competition starts would invalidate all scores.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

# ---- SEALED SEED — never change after first build ----
_BENCHMARK_SEED: int = 271828182  # first 9 digits of e — arbitrary but fixed forever


def _derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2-SHA256 key derivation."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)


def _encrypt_json(data: dict, password: str) -> str:
    """AES-256-GCM encrypt a JSON-serializable dict. Returns base64 string."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        print("ERROR: cryptography package required. pip install cryptography")
        sys.exit(1)

    plain = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    salt = secrets.token_bytes(16)
    key = _derive_key(password, salt)
    nonce = secrets.token_bytes(12)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plain, None)
    combined = salt + nonce + ct
    return base64.b64encode(combined).decode("ascii")


def _load_tasks(benchmark: str, count: int) -> List[Any]:
    """Load tasks from HF datasets with the sealed seed."""
    from trinity.orchestration.dataset import load_tasks
    import random as _random

    rng = _random.Random(_BENCHMARK_SEED)
    tasks = load_tasks(benchmark, "train", max_items=count * 3, seed=_BENCHMARK_SEED)
    rng.shuffle(tasks)
    return tasks[:count]


async def _cache_answers(items: List[Dict], pool, pool_models: List[str]) -> None:
    """Call each model ONCE per question (temp=0) and cache the answer."""
    import httpx

    async with httpx.AsyncClient() as client:
        for item in items:
            for model_name in pool_models:
                if item["model_answers"].get(model_name):
                    continue  # already cached
                try:
                    res = await pool.chat(
                        model_name,
                        [{"role": "user", "content": item["question_text"]}],
                        max_tokens=4096, temperature=0.0, top_p=1.0,
                        client=client,
                    )
                    item["model_answers"][model_name] = res.text
                except Exception as exc:
                    print(f"  [warn] {item['question_id']} / {model_name}: {exc}")
                    item["model_answers"][model_name] = ""


def _task_to_item(task: Any) -> Dict:
    """Convert a trinity Task to a benchmark item dict."""
    benchmark = getattr(task, "benchmark", "math500") or "math500"
    b_lower = benchmark.lower().strip()
    if b_lower in ("math500", "math", "aime", "aime2025"):
        task_type = "math"
    elif b_lower in ("mmlu", "gpqa", "gpqa-diamond", "gpqa_diamond"):
        task_type = "knowledge"
    else:
        task_type = "code"

    return {
        "question_id": getattr(task, "task_id", f"q_{hash(task.prompt) % 100000}"),
        "question_text": getattr(task, "prompt", ""),
        "task_type": task_type,
        "benchmark": benchmark,
        "correct_answer": getattr(task, "answer", None),
        "model_answers": {},
    }


async def build_benchmark(benchmark: str, output_dir: str, password: str) -> str:
    """Build a hidden benchmark and save encrypted files.

    Returns the benchmark content hash (for the audit trail).
    """
    bench_dir = Path(output_dir) / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building hidden benchmark for: {benchmark}")
    print(f"  Seed: {_BENCHMARK_SEED} (SEALED — never change)")
    print(f"  Output: {bench_dir}")
    print()

    # Determine task counts based on benchmark type
    eval_count = 150
    audit_count = 50
    live_count = 20
    total_needed = eval_count + audit_count + live_count

    # Load tasks
    print(f"Loading {total_needed}+ tasks from {benchmark}...")
    tasks = _load_tasks(benchmark, total_needed + 50)  # 50 extra margin
    print(f"  Loaded {len(tasks)} tasks")

    # Split: first eval_size are eval, next audit_size are audit, next live_size are live
    eval_tasks = tasks[:eval_count]
    audit_tasks = tasks[eval_count:eval_count + audit_count]
    live_tasks = tasks[eval_count + audit_count:eval_count + audit_count + live_count]

    eval_items = [_task_to_item(t) for t in eval_tasks]
    audit_items = [_task_to_item(t) for t in audit_tasks]
    live_items_raw = [_task_to_item(t) for t in live_tasks]

    # Live items don't need cached answers (they get live API calls)
    # But we need to strip model_answers expectation
    live_items = [
        {k: v for k, v in item.items() if k != "model_answers"}
        for item in live_items_raw
    ]

    print(f"  Eval:  {len(eval_items)} questions")
    print(f"  Audit: {len(audit_items)} questions")
    print(f"  Live:  {len(live_items)} questions")

    # Cache answers for eval and audit (NOT live)
    from trinity.llm.fireworks_client import FireworksPool
    pool = FireworksPool(str(_REPO / "configs" / "models.yaml"))
    pool_models = list(pool.models.keys())

    all_cacheable = eval_items + audit_items
    total_calls = len(all_cacheable) * len(pool_models)
    est_cost = total_calls * 0.003
    print(f"\nPre-computing cached answers: {len(all_cacheable)} questions × "
          f"{len(pool_models)} models = {total_calls} API calls")
    print(f"Estimated cost: ~${est_cost:.2f}")

    await _cache_answers(all_cacheable, pool, pool_models)
    print("  Caching complete.")

    # Save encrypted files
    print(f"\nSaving encrypted benchmark files...")

    eval_path = bench_dir / "eval.json"
    audit_path = bench_dir / "audit.json"
    live_path = bench_dir / "live.json"

    eval_path.write_text(_encrypt_json(
        {"seed": _BENCHMARK_SEED, "count": len(eval_items), "items": eval_items},
        password,
    ))
    audit_path.write_text(_encrypt_json(
        {"seed": _BENCHMARK_SEED, "count": len(audit_items), "items": audit_items},
        password,
    ))
    live_path.write_text(_encrypt_json(
        {"seed": _BENCHMARK_SEED, "count": len(live_items), "items": live_items},
        password,
    ))

    # Compute content hash (over the unencrypted data, for audit)
    h = hashlib.sha256()
    for item in sorted(eval_items + audit_items + live_items, key=lambda x: x["question_id"]):
        h.update(item["question_text"].encode("utf-8"))
    content_hash = h.hexdigest()

    # Write unencrypted hash file (public — miners can verify benchmark hasn't changed)
    (bench_dir / "hash.txt").write_text(f"{content_hash}\n")

    # Write metadata
    meta = {
        "benchmark": benchmark,
        "seed": _BENCHMARK_SEED,
        "eval_count": len(eval_items),
        "audit_count": len(audit_items),
        "live_count": len(live_items),
        "content_hash": content_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pool_models": pool_models,
    }
    (bench_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Benchmark hash: {content_hash}")
    print(f"  Files saved to: {bench_dir}/")
    print(f"  - eval.json  ({eval_path.stat().st_size} bytes, encrypted)")
    print(f"  - audit.json ({audit_path.stat().st_size} bytes, encrypted)")
    print(f"  - live.json  ({live_path.stat().st_size} bytes, encrypted)")
    print(f"  - hash.txt   (public — commit this to the repo)")
    print(f"  - meta.json  (public — commit this to the repo)")

    return content_hash


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the hidden benchmark for the TinyRouter accuracy competition"
    )
    ap.add_argument("--benchmark", required=True,
                    help="Benchmark name (math500 or mmlu)")
    ap.add_argument("--output-dir", default=None, dest="output_dir",
                    help="Output directory (default: $TINYROUTER_BENCHMARK_DIR or "
                         "../tinyrouter-benchmark/)")
    args = ap.parse_args()

    output_dir = args.output_dir or os.environ.get(
        "TINYROUTER_BENCHMARK_DIR",
        str(_REPO.parent / "tinyrouter-benchmark"),
    )

    password = os.environ.get("BENCHMARK_PASSWORD")
    if not password:
        print("ERROR: BENCHMARK_PASSWORD environment variable is not set.")
        print("  export BENCHMARK_PASSWORD=<strong-password>")
        print("  This password encrypts the benchmark. Store it securely — if lost,")
        print("  the benchmark cannot be decrypted and must be rebuilt.")
        sys.exit(1)

    print("=" * 60)
    print("  TinyRouter — Hidden Benchmark Builder")
    print("=" * 60)
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Output:    {output_dir}")
    print(f"  Seed:      {_BENCHMARK_SEED} (SEALED)")
    print()

    content_hash = asyncio.run(build_benchmark(args.benchmark, output_dir, password))

    print(f"\nDone. Benchmark hash: {content_hash}")
    print(f"Add this hash to the repo's benchmark_hashes.txt to prove the benchmark")
    print(f"has not been modified since creation.")


if __name__ == "__main__":
    main()
