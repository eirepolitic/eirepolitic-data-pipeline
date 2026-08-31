from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

from process.oireachtas_speech_issue_classifier import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VERBOSITY,
    ENRICHMENT_CSV_KEY,
    LEGACY_CLASSIFIED_KEY,
    SILVER_SPEECHES_KEY,
    classify_with_openai,
    make_s3_client,
    prepare_classification_plan,
    read_s3_csv,
)


def stratified_sample(frame: pd.DataFrame, *, per_label: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, group in frame.groupby("issue_label", sort=True):
        group = group.sort_values(["debate_date", "speech_id"], kind="stable").reset_index(drop=True)
        if len(group) <= per_label:
            chosen = group
        elif per_label == 1:
            chosen = group.iloc[[len(group) // 2]]
        else:
            indices = [round(i * (len(group) - 1) / (per_label - 1)) for i in range(per_label)]
            chosen = group.iloc[indices]
        parts.append(chosen)
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Luna against safely migrated historical speech labels")
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT))
    parser.add_argument("--verbosity", default=os.getenv("OPENAI_VERBOSITY", DEFAULT_VERBOSITY))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_luna_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    existing = read_s3_csv(s3, bucket=args.bucket, key=ENRICHMENT_CSV_KEY, optional=True)
    legacy = read_s3_csv(s3, bucket=args.bucket, key=LEGACY_CLASSIFIED_KEY)
    plan = prepare_classification_plan(silver, existing=existing, legacy=legacy)

    trusted = plan.rows[
        plan.rows["issue_label_source"].isin({"legacy_migration_exact", "legacy_migration_date_hash_unique"})
    ].copy()
    trusted = trusted[trusted["issue_label"].astype(str).str.strip() != ""].copy()
    sample = stratified_sample(trusted, per_label=args.per_label)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for _, row in sample.iterrows():
        try:
            predicted = classify_with_openai(
                client,
                str(row["speech_text"]),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                verbosity=args.verbosity,
            )
            expected = str(row["issue_label"])
            exact_match = predicted == expected
            confusion[expected][predicted] += 1
            results.append(
                {
                    "speech_id": row["speech_id"],
                    "debate_date": row["debate_date"],
                    "speaker_name": row["speaker_name"],
                    "word_count": int(row["word_count"]),
                    "expected_legacy_label": expected,
                    "luna_label": predicted,
                    "exact_match": exact_match,
                    "speech_excerpt": str(row["speech_text"])[:500],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "speech_id": row["speech_id"],
                    "expected_legacy_label": row["issue_label"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )

    succeeded = len(results)
    matched = sum(1 for row in results if row["exact_match"])
    per_label_stats: dict[str, dict[str, object]] = {}
    for expected, predicted_counts in sorted(confusion.items()):
        total = sum(predicted_counts.values())
        correct = predicted_counts.get(expected, 0)
        per_label_stats[expected] = {
            "attempted": total,
            "exact_matches": correct,
            "exact_agreement_pct": round(correct / total * 100, 1) if total else 0.0,
            "luna_labels": dict(predicted_counts),
        }

    report = {
        "mode": "luna_historical_benchmark",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
        "per_label_requested": args.per_label,
        "trusted_legacy_pool_rows": int(len(trusted)),
        "sample_attempted": int(len(sample)),
        "sample_succeeded": succeeded,
        "sample_failed": len(failures),
        "exact_matches": matched,
        "exact_agreement_pct": round(matched / succeeded * 100, 1) if succeeded else 0.0,
        "writes_performed": False,
        "per_label": per_label_stats,
        "results": results,
        "failures": failures,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
