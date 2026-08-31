from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from extract.oireachtas.batch import current_batch_id
from extract.oireachtas.io_s3 import make_s3_client
from process.oireachtas_speech_issue_classifier import (
    COMPAT_CSV_KEY,
    ENRICHMENT_CSV_KEY,
    ISSUE_CATEGORY_SET,
    SILVER_SPEECHES_KEY,
    read_s3_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a complete speech issue enrichment candidate")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION", "ca-central-1"))
    p.add_argument("--report-path", default="speech_issue_candidate_validation.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not current_batch_id():
        raise RuntimeError("OIREACHTAS_BATCH_ID is required")
    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    enrichment = read_s3_csv(s3, bucket=args.bucket, key=ENRICHMENT_CSV_KEY)
    compat = read_s3_csv(s3, bucket=args.bucket, key=COMPAT_CSV_KEY)

    silver_dates = pd.to_datetime(silver["debate_date"], errors="coerce")
    enrichment_dates = pd.to_datetime(enrichment["debate_date"], errors="coerce")
    invalid_labels = sorted(set(enrichment["issue_label"]) - ISSUE_CATEGORY_SET)
    pending = int(enrichment["classification_status"].isin(["pending", "failed"]).sum())
    blank_labels = int(enrichment["issue_label"].astype(str).str.strip().eq("").sum())
    duplicate_ids = int(enrichment["speech_id"].duplicated().sum())
    speech_id_match = set(enrichment["speech_id"]) == set(silver["speech_id"])
    compat_labels_blank = int(compat["PoliticalIssues"].astype(str).str.strip().eq("").sum())

    checks = {
        "row_count_match": len(silver) == len(enrichment) == len(compat),
        "speech_id_unique": duplicate_ids == 0,
        "speech_id_set_match": speech_id_match,
        "zero_pending_or_failed": pending == 0,
        "zero_blank_labels": blank_labels == 0,
        "zero_invalid_labels": len(invalid_labels) == 0,
        "compat_zero_blank_labels": compat_labels_blank == 0,
        "date_coverage_match": (
            silver_dates.notna().any()
            and enrichment_dates.notna().any()
            and silver_dates.max() == enrichment_dates.max()
            and silver_dates.min() == enrichment_dates.min()
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    source_counts = enrichment["issue_label_source"].fillna("").value_counts().to_dict()
    model_counts = enrichment["model_name"].fillna("").value_counts().to_dict()
    label_counts = enrichment["issue_label"].fillna("").value_counts().to_dict()

    report = {
        "status": status,
        "batch_id": current_batch_id(),
        "silver_rows": int(len(silver)),
        "enrichment_rows": int(len(enrichment)),
        "compat_rows": int(len(compat)),
        "silver_min_date": silver_dates.min().date().isoformat() if silver_dates.notna().any() else None,
        "silver_max_date": silver_dates.max().date().isoformat() if silver_dates.notna().any() else None,
        "enrichment_min_date": enrichment_dates.min().date().isoformat() if enrichment_dates.notna().any() else None,
        "enrichment_max_date": enrichment_dates.max().date().isoformat() if enrichment_dates.notna().any() else None,
        "pending_or_failed_rows": pending,
        "blank_label_rows": blank_labels,
        "compat_blank_label_rows": compat_labels_blank,
        "duplicate_speech_ids": duplicate_ids,
        "invalid_labels": invalid_labels,
        "checks": checks,
        "issue_label_source_counts": source_counts,
        "model_name_counts": model_counts,
        "issue_label_counts": label_counts,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
