#!/usr/bin/env python3
"""Audit promoted speech issue labels before issue metrics are commissioned."""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd

from extract.oireachtas.batch import PRODUCTION_POINTER_KEY, read_json_required, resolve_production_key
from political_metrics.issue_audit import audit_issue_classification

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_ISSUE_AUDIT_DIR", "artifacts/political-metrics-issue-audit"))
PERIOD_START = os.getenv("POLITICAL_METRICS_ISSUE_PERIOD_START", "2026-07-01")
PERIOD_END = os.getenv("POLITICAL_METRICS_ISSUE_PERIOD_END", "2026-07-31")

SPEECH_KEY = "processed/oireachtas_unified/latest/csv/silver_speeches.csv"
LABEL_KEY = "processed/oireachtas_unified/latest/csv/enrichment_speech_issue_labels.csv"


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False), resolved


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_counts(values: dict[str, int], limit: int = 10) -> str:
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{name}: {count:,}" for name, count in items)


def _markdown(report: dict) -> str:
    overall = report["overall"]
    period = report["period"]
    lines = [
        "# Political issue classification readiness",
        "",
        f"**Overall promoted classifier ready: {'YES' if overall['ready'] else 'NO'}**",
        f"**{PERIOD_START} to {PERIOD_END} ready: {'YES' if period['ready'] else 'NO'}**",
        "",
        "## Overall promoted data",
        "",
        f"- Speech rows in scope: **{overall.get('scope_rows', 0):,}**",
        f"- Policy-labelled speeches: **{overall.get('policy_labelled_rows', 0):,}** ({_pct(overall.get('policy_label_rate', 0.0))})",
        f"- `NONE` speeches: **{overall.get('none_rows', 0):,}**",
        f"- Missing label rows: **{overall.get('missing_label_rows', 0):,}**",
        f"- Speech-text hash mismatches: **{overall.get('hash_mismatch_rows', 0):,}**",
        f"- Non-final classification rows: **{overall.get('non_final_status_rows', 0):,}**",
        "",
        "## July 2026",
        "",
        f"- Speech rows: **{period.get('scope_rows', 0):,}**",
        f"- Policy-labelled speeches: **{period.get('policy_labelled_rows', 0):,}** ({_pct(period.get('policy_label_rate', 0.0))})",
        f"- `NONE` speeches: **{period.get('none_rows', 0):,}**",
        f"- Missing label rows: **{period.get('missing_label_rows', 0):,}**",
        f"- Speech-text hash mismatches: **{period.get('hash_mismatch_rows', 0):,}**",
        f"- Non-final classification rows: **{period.get('non_final_status_rows', 0):,}**",
        "",
        "## How labels were produced",
        "",
        f"Overall: {_format_counts(overall.get('issue_label_source_counts', {}))}",
        "",
        f"July: {_format_counts(period.get('issue_label_source_counts', {}))}",
        "",
        "## Model provenance",
        "",
        f"Overall: {_format_counts(overall.get('model_name_counts', {}))}",
        "",
        f"July: {_format_counts(period.get('model_name_counts', {}))}",
        "",
        "## Interpretation",
        "",
        "A passing gate means every canonical speech in scope has one final approved issue label tied to the same speech-text hash. It does not by itself prove that every classification is substantively correct; taxonomy and classifier quality still require ongoing review.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    pointer = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)
    speeches, speech_key = _read_csv(s3, SPEECH_KEY)
    labels, label_key = _read_csv(s3, LABEL_KEY)

    overall = audit_issue_classification(speeches, labels)
    period = audit_issue_classification(
        speeches,
        labels,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )

    report = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_batch_id": str(pointer.get("batch_id") or pointer.get("mode") or "unknown"),
        "speech_source_key": speech_key,
        "label_source_key": label_key,
        "overall": overall,
        "period_scope": {"start": PERIOD_START, "end": PERIOD_END},
        "period": period,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    return 0 if overall.get("ready") and period.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
