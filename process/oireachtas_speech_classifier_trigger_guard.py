from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

from process.oireachtas_speech_issue_classifier import (
    PRODUCTION_CLASSIFICATION_POINTER_KEY,
    read_current_enrichment,
    read_source_context,
    reconcile_results_to_source,
    select_delta,
)
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client


def inspect_automatic_trigger(
    *,
    s3: Any,
    bucket: str,
    expected_source_batch_id: str,
) -> dict[str, Any]:
    expected = str(expected_source_batch_id or "").strip()
    if not expected:
        raise ValueError("expected_source_batch_id is required")

    active_batch_id, active_source_key, speeches = read_source_context(
        s3,
        bucket=bucket,
    )
    if active_batch_id != expected:
        raise RuntimeError(
            "The promoted source batch is no longer active: "
            f"expected={expected}, active={active_batch_id}"
        )

    current_pointer, existing = read_current_enrichment(s3, bucket=bucket)
    delta = select_delta(speeches, existing)
    reconciled = reconcile_results_to_source(existing, existing.iloc[0:0], speeches)
    removed_or_changed_existing_rows = int(len(existing) - len(reconciled))
    classified_source_batch_id = str(
        (current_pointer or {}).get("source_batch_id") or ""
    ).strip()
    source_batch_changed = classified_source_batch_id != active_batch_id
    should_prepare = bool(
        source_batch_changed
        or len(delta) > 0
        or removed_or_changed_existing_rows > 0
    )

    return {
        "expected_source_batch_id": expected,
        "active_source_batch_id": active_batch_id,
        "active_source_key": active_source_key,
        "classification_pointer_key": PRODUCTION_CLASSIFICATION_POINTER_KEY,
        "current_classification_run_id": str(
            (current_pointer or {}).get("run_id") or ""
        ),
        "classified_source_batch_id": classified_source_batch_id,
        "source_batch_changed": source_batch_changed,
        "source_rows": int(len(speeches)),
        "current_classification_rows": int(len(existing)),
        "delta_rows": int(len(delta)),
        "removed_or_changed_existing_rows": removed_or_changed_existing_rows,
        "should_prepare": should_prepare,
        "reason": (
            "source_batch_changed"
            if source_batch_changed
            else "speech_delta_present"
            if len(delta) > 0
            else "source_reconciliation_required"
            if removed_or_changed_existing_rows > 0
            else "no_classification_work"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guard automatic Oireachtas speech-classification candidate creation"
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET", DEFAULT_BUCKET),
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", DEFAULT_REGION),
    )
    parser.add_argument("--expected-source-batch-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = inspect_automatic_trigger(
        s3=make_s3_client(region_name=args.region),
        bucket=args.bucket,
        expected_source_batch_id=args.expected_source_batch_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
