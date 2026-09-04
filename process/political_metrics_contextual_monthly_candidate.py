#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd

from extract.oireachtas.batch import batch_key_for_production_key, validate_batch_id
from political_metrics.candidate_publish import publish_dataset_to_candidate
from political_metrics.contextual_monthly_results import (
    audit_monthly_contextual_vote_results,
    build_monthly_contextual_vote_results,
)
from political_metrics.materialize import get_dataset_contract, load_materialization_contract
from political_metrics.periods import MetricPeriod

CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"
SOURCE_KEYS = {
    "daily_context_vote_components": "processed/oireachtas_unified/latest/metrics/daily/daily_context_vote_components/csv/daily_context_vote_components.csv",
    "context_division_party_vote_components": "processed/oireachtas_unified/latest/metrics/event/context_division_party_vote_components/csv/context_division_party_vote_components.csv",
    "monthly_metric_results": "processed/oireachtas_unified/latest/metrics/completed_month/monthly_metric_results/csv/monthly_metric_results.csv",
}

STRING_COLUMNS = [
    "metric_id", "period_type", "period_start", "period_end", "grain", "entity_id", "entity_name",
    "dimension_name", "dimension_value", "output_unit", "reliability_status", "public_use_status",
    "warning_code", "source_batch_id", "calculated_at_utc",
]
NUMERIC_COLUMNS = ["value", "numerator", "denominator"]
INTEGER_COLUMNS = ["metric_version", "contract_version"]


def _read_candidate_csv(s3, *, bucket: str, batch_id: str, logical_key: str) -> pd.DataFrame:
    key = batch_key_for_production_key(logical_key, batch_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])


def _periods_from_monthly(frame: pd.DataFrame) -> list[MetricPeriod]:
    pairs = frame[["period_start", "period_end"]].drop_duplicates().sort_values(["period_start", "period_end"])
    periods: list[MetricPeriod] = []
    for row in pairs.itertuples(index=False):
        start = date.fromisoformat(str(row.period_start))
        end = date.fromisoformat(str(row.period_end))
        periods.append(MetricPeriod(start, end, start.strftime("%Y-%m"), "month"))
    return periods


def _normalize_monthly_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in STRING_COLUMNS:
        result[col] = result[col].fillna("").astype(str)
    for col in NUMERIC_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    for col in INTEGER_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="raise").astype("int64")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Append completed-month division-context voting rows to one candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--report-path", default="contextual_monthly_candidate_report.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    s3 = boto3.client("s3", region_name=args.region)
    frames = {
        name: _read_candidate_csv(s3, bucket=args.bucket, batch_id=batch_id, logical_key=key)
        for name, key in SOURCE_KEYS.items()
    }
    contract = load_materialization_contract(CONTRACT_PATH)
    contract_version = int(contract["contract_version"])
    periods = _periods_from_monthly(frames["monthly_metric_results"])

    contextual_frames: list[pd.DataFrame] = []
    for period in periods:
        contextual_frames.append(build_monthly_contextual_vote_results(
            daily_context_vote_components=frames["daily_context_vote_components"],
            context_division_party_vote_components=frames["context_division_party_vote_components"],
            period=period,
            source_batch_id=batch_id,
            contract_version=contract_version,
        ))
    contextual = pd.concat(contextual_frames, ignore_index=True) if contextual_frames else pd.DataFrame()
    audit = audit_monthly_contextual_vote_results(results=contextual, periods=periods, source_batch_id=batch_id)
    if not audit.get("ready"):
        raise RuntimeError(f"contextual monthly result audit failed: {audit}")

    existing = frames["monthly_metric_results"].copy()
    keep = ~(
        existing["metric_id"].isin(["member_vote_participation_pct", "party_vote_cohesion_pct"])
        & existing["dimension_name"].eq("division_context")
    )
    combined = pd.concat([existing.loc[keep], contextual], ignore_index=True)

    now = datetime.now(timezone.utc).isoformat()
    combined["source_batch_id"] = batch_id
    combined["calculated_at_utc"] = now
    combined["contract_version"] = contract_version
    combined = _normalize_monthly_types(combined)

    key = ["metric_id","metric_version","period_start","period_end","grain","entity_id","dimension_name","dimension_value"]
    duplicate = int(combined.duplicated(key).sum())
    if duplicate:
        raise RuntimeError(f"monthly_metric_results would contain {duplicate} duplicate primary-key rows")

    result = publish_dataset_to_candidate(
        s3,
        bucket=args.bucket,
        batch_id=batch_id,
        frame=combined,
        dataset=get_dataset_contract(contract, "monthly_metric_results"),
        contract_version=contract_version,
        source_batch_id=batch_id,
    )
    report = {
        "batch_id": batch_id,
        "generated_at_utc": now,
        "period_count": len(periods),
        "contextual_row_count": int(len(contextual)),
        "combined_monthly_row_count": int(len(combined)),
        "audit": audit,
        "published": {"entry_name": result["entry_name"], "objects": result["objects"]},
        "production_pointer_changed": False,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
