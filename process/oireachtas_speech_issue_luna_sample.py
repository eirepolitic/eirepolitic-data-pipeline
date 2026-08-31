from __future__ import annotations

import argparse
import json
import os
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


def select_evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.iloc[0:0].copy()
    if len(frame) <= count:
        return frame.copy()
    indices = [round(i * (len(frame) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    return frame.iloc[indices].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-writing Luna quality sample on post-legacy speeches")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--cutoff", default="2026-02-26")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT))
    parser.add_argument("--verbosity", default=os.getenv("OPENAI_VERBOSITY", DEFAULT_VERBOSITY))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    parser.add_argument("--report-path", default="speech_issue_luna_sample.json")
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
    recent = recent.sort_values(["debate_date_parsed", "speech_id"], kind="stable").reset_index(drop=True)
    sample = select_evenly_spaced(recent, args.count)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []
    failures = []
    for _, row in sample.iterrows():
        try:
            label = classify_with_openai(
                client,
                str(row["speech_text"]),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                verbosity=args.verbosity,
            )
            results.append(
                {
                    "speech_id": row["speech_id"],
                    "debate_date": row["debate_date"],
                    "speaker_name": row["speaker_name"],
                    "word_count": int(row["word_count"]),
                    "issue_label": label,
                    "speech_excerpt": str(row["speech_text"])[:700],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "speech_id": row["speech_id"],
                    "debate_date": row["debate_date"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )

    report = {
        "mode": "luna_recent_sample",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
        "cutoff_exclusive": args.cutoff,
        "pending_total_all_dates": int(len(pending)),
        "pending_recent_after_cutoff": int(len(recent)),
        "sample_requested": int(args.count),
        "sample_attempted": int(len(sample)),
        "sample_succeeded": int(len(results)),
        "sample_failed": int(len(failures)),
        "writes_performed": False,
        "results": results,
        "failures": failures,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
