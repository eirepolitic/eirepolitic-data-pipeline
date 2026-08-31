from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from process.oireachtas_speech_issue_classifier import (
    DEFAULT_MODEL,
    ENRICHMENT_CSV_KEY,
    LEGACY_CLASSIFIED_KEY,
    SILVER_SPEECHES_KEY,
    build_classifier_prompt,
    canonicalize_label,
    classification_schema,
    make_s3_client,
    prepare_classification_plan,
    read_s3_csv,
)

# Current standard API prices for GPT-5.6 Luna as of 2026-08-31.
INPUT_USD_PER_M = 0.20
CACHED_INPUT_USD_PER_M = 0.02
OUTPUT_USD_PER_M = 1.20

_thread_local = threading.local()


def client() -> OpenAI:
    value = getattr(_thread_local, "client", None)
    if value is None:
        value = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _thread_local.client = value
    return value


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": min(cached_tokens, input_tokens),
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def add_usage(total: dict[str, int], extra: dict[str, int]) -> None:
    for key in total:
        total[key] += int(extra.get(key, 0))


def estimated_cost(usage: dict[str, int]) -> dict[str, float]:
    cached = usage["cached_input_tokens"]
    uncached = max(0, usage["input_tokens"] - cached)
    input_cost = uncached / 1_000_000 * INPUT_USD_PER_M
    cached_cost = cached / 1_000_000 * CACHED_INPUT_USD_PER_M
    output_cost = usage["output_tokens"] / 1_000_000 * OUTPUT_USD_PER_M
    return {
        "uncached_input_usd": round(input_cost, 6),
        "cached_input_usd": round(cached_cost, 6),
        "output_usd": round(output_cost, 6),
        "total_usd": round(input_cost + cached_cost + output_cost, 6),
    }


def classify_one(row: dict[str, Any], *, model: str, max_retries: int = 4) -> dict[str, Any]:
    started = time.perf_counter()
    aggregate = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    attempts = 0
    last_error = ""

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            response = client().responses.create(
                model=model,
                input=build_classifier_prompt(str(row["speech_text"])),
                reasoning={"effort": "low"},
                text={
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": "speech_issue_classification",
                        "strict": True,
                        "schema": classification_schema(),
                    },
                },
                max_output_tokens=128,
                store=False,
            )
            add_usage(aggregate, usage_dict(response))
            payload = json.loads(str(response.output_text or "").strip())
            label = canonicalize_label(payload.get("issue_label"))
            if not label:
                raise ValueError(f"invalid issue label: {payload!r}")
            elapsed = time.perf_counter() - started
            return {
                "speech_id": row["speech_id"],
                "debate_date": row["debate_date"],
                "speaker_name": row["speaker_name"],
                "word_count": int(row["word_count"]),
                "issue_label": label,
                "speech_excerpt": str(row["speech_text"])[:500],
                "status": "success",
                "attempts": attempts,
                "latency_seconds": round(elapsed, 3),
                **aggregate,
                "estimated_cost_usd": estimated_cost(aggregate)["total_usd"],
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if attempt < max_retries:
                time.sleep(min(2.0 * attempt, 6.0))

    elapsed = time.perf_counter() - started
    return {
        "speech_id": row["speech_id"],
        "debate_date": row["debate_date"],
        "speaker_name": row["speaker_name"],
        "word_count": int(row["word_count"]),
        "issue_label": "",
        "speech_excerpt": str(row["speech_text"])[:500],
        "status": "failed",
        "attempts": attempts,
        "latency_seconds": round(elapsed, 3),
        "error": last_error,
        **aggregate,
        "estimated_cost_usd": estimated_cost(aggregate)["total_usd"],
    }


def evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    if len(frame) <= count:
        return frame.copy()
    if count == 1:
        return frame.iloc[[len(frame) // 2]].copy()
    indices = [round(i * (len(frame) - 1) / (count - 1)) for i in range(count)]
    return frame.iloc[indices].copy()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-writing 1000-speech Luna cost/reliability benchmark")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--cutoff", default="2026-02-26")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_bulk_cost_sample.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    existing = read_s3_csv(s3, bucket=args.bucket, key=ENRICHMENT_CSV_KEY, optional=True)
    legacy = read_s3_csv(s3, bucket=args.bucket, key=LEGACY_CLASSIFIED_KEY, optional=True)
    plan = prepare_classification_plan(silver, existing=existing, legacy=legacy)

    pending = plan.rows[plan.rows["classification_status"] == "pending"].copy()
    pending["debate_date_parsed"] = pd.to_datetime(pending["debate_date"], errors="coerce")
    cutoff = pd.Timestamp(args.cutoff)
    recent = pending[pending["debate_date_parsed"] > cutoff].copy()
    recent = recent.sort_values(["debate_date_parsed", "speech_id"], kind="stable")
    sample = evenly_spaced(recent, args.count)
    records = sample.to_dict(orient="records")

    wall_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(classify_one, row, model=args.model) for row in records]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    results.sort(key=lambda row: (row["debate_date"], row["speech_id"]))

    successes = [row for row in results if row["status"] == "success"]
    failures = [row for row in results if row["status"] != "success"]
    usage = {
        "input_tokens": sum(int(row["input_tokens"]) for row in results),
        "cached_input_tokens": sum(int(row["cached_input_tokens"]) for row in results),
        "output_tokens": sum(int(row["output_tokens"]) for row in results),
        "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in results),
    }
    cost = estimated_cost(usage)
    category_counts = Counter(row["issue_label"] for row in successes)
    latencies = [float(row["latency_seconds"]) for row in results]
    attempts = Counter(int(row["attempts"]) for row in results)

    sample_count = max(1, len(results))
    cost_per_1000 = cost["total_usd"] / sample_count * 1000
    recent_remaining = int(len(recent))
    all_pending = int(len(pending))

    report = {
        "mode": "bulk_cost_reliability_sample",
        "model": args.model,
        "reasoning_effort": "low",
        "cutoff_exclusive": args.cutoff,
        "concurrency": args.concurrency,
        "writes_performed": False,
        "pending_total_all_dates": all_pending,
        "pending_recent_after_cutoff": recent_remaining,
        "sample_requested": args.count,
        "sample_attempted": len(results),
        "sample_succeeded": len(successes),
        "sample_failed": len(failures),
        "success_pct": round(len(successes) / sample_count * 100, 2),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_speeches_per_minute": round(sample_count / max(wall_seconds, 0.001) * 60, 2),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "attempt_counts": {str(k): v for k, v in sorted(attempts.items())},
        "usage": usage,
        "pricing_usd_per_million_tokens": {
            "uncached_input": INPUT_USD_PER_M,
            "cached_input": CACHED_INPUT_USD_PER_M,
            "output": OUTPUT_USD_PER_M,
        },
        "estimated_sample_cost_usd": cost,
        "estimated_cost_per_1000_speeches_usd": round(cost_per_1000, 4),
        "estimated_remaining_recent_cost_usd_at_sample_rate": round(cost_per_1000 / 1000 * recent_remaining, 2),
        "estimated_all_pending_cost_usd_at_sample_rate": round(cost_per_1000 / 1000 * all_pending, 2),
        "average_tokens_per_speech": {
            "input": round(usage["input_tokens"] / sample_count, 1),
            "cached_input": round(usage["cached_input_tokens"] / sample_count, 1),
            "output": round(usage["output_tokens"] / sample_count, 1),
            "reasoning": round(usage["reasoning_tokens"] / sample_count, 1),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "none_count": int(category_counts.get("NONE", 0)),
        "none_pct": round(category_counts.get("NONE", 0) / max(1, len(successes)) * 100, 2),
        "failures": failures,
        "results": successes,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results"}}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
