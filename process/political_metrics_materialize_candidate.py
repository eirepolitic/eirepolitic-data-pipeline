#!/usr/bin/env python3
"""Build political metrics from a candidate batch and publish back into that candidate only."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3
import pandas as pd

from extract.oireachtas.batch import batch_key_for_production_key, validate_batch_id
from political_metrics.candidate_publish import publish_dataset_to_candidate
from political_metrics.foundations import (
    build_daily_activity_components,
    build_daily_issue_activity,
    build_daily_question_dimensions,
    build_division_party_vote_components,
)
from political_metrics.issue_audit import audit_issue_classification
from political_metrics.materialize import get_dataset_contract, load_materialization_contract
from political_metrics.monthly_results import build_monthly_results
from political_metrics.periods import MetricPeriod
from political_metrics.question_context import (
    build_oral_question_exchange_participants,
    build_oral_question_sections,
    build_speech_question_context,
)
from political_metrics.sources import canonical_speeches
from process.political_metrics_materialization_commission import TABLE_KEYS as COMMISSION_TABLE_KEYS

DUBLIN = ZoneInfo("Europe/Dublin")
CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"
TABLE_KEYS = dict(COMMISSION_TABLE_KEYS)
TABLE_KEYS["sections"] = "processed/oireachtas_unified/latest/csv/silver_debate_sections.csv"
TABLE_KEYS["offices"] = "processed/oireachtas_unified/latest/csv/silver_member_offices.csv"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build and publish political metrics into one immutable candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--report-path", default="political_metrics_candidate_report.json")
    return p


def _read_candidate_csv(s3, *, bucket: str, batch_id: str, logical_key: str) -> pd.DataFrame:
    batch_key = batch_key_for_production_key(logical_key, batch_id)
    obj = s3.get_object(Bucket=bucket, Key=batch_key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])


def _source_date_range(frames: dict[str, pd.DataFrame]) -> MetricPeriod:
    candidates: list[pd.Series] = []
    for table, col in [
        ("speeches", "debate_date"),
        ("debates", "debate_date"),
        ("divisions", "division_date"),
        ("votes", "division_date"),
        ("questions", "question_date"),
    ]:
        frame = frames[table]
        if col in frame.columns and not frame.empty:
            values = pd.to_datetime(frame[col], errors="coerce").dropna()
            if not values.empty:
                candidates.append(values)
    if not candidates:
        raise ValueError("candidate contains no metric source dates")
    combined = pd.concat(candidates, ignore_index=True)
    start = combined.min().date()
    end = combined.max().date()
    return MetricPeriod(start, end, f"{start.isoformat()}_{end.isoformat()}", "candidate_history")


def _completed_month_periods(start: date, *, today: date) -> list[MetricPeriod]:
    first_this_month = today.replace(day=1)
    last_completed = first_this_month - timedelta(days=1)
    cursor = start.replace(day=1)
    periods: list[MetricPeriod] = []
    while cursor <= last_completed:
        end = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
        if end <= last_completed:
            periods.append(MetricPeriod(cursor, end, f"{cursor.year:04d}-{cursor.month:02d}", "month"))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return periods


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    s3 = boto3.client("s3", region_name=args.region)
    contract = load_materialization_contract(CONTRACT_PATH)
    contract_version = int(contract["contract_version"])

    frames: dict[str, pd.DataFrame] = {}
    for table, logical_key in TABLE_KEYS.items():
        frames[table] = _read_candidate_csv(s3, bucket=args.bucket, batch_id=batch_id, logical_key=logical_key)
    frames["speeches"] = canonical_speeches(frames["speeches"])

    classifier_gate = audit_issue_classification(frames["speeches"], frames["labels"])
    if not classifier_gate.get("ready"):
        raise RuntimeError(f"candidate issue-classification gate failed: {classifier_gate}")

    history_period = _source_date_range(frames)
    today = datetime.now(DUBLIN).date()
    completed_months = _completed_month_periods(history_period.start, today=today)
    if not completed_months:
        raise RuntimeError("candidate has no completed calendar months available for metric results")

    datasets = {
        "daily_activity_components": build_daily_activity_components(
            speeches=frames["speeches"], labels=frames["labels"], memberships=frames["memberships"],
            member_parties=frames["parties"], member_constituencies=frames["constituencies"],
            debate_records=frames["debates"], divisions=frames["divisions"], member_votes=frames["votes"],
            questions=frames["questions"], period=history_period, source_batch_id=batch_id, contract_version=contract_version,
        ),
        "daily_issue_activity": build_daily_issue_activity(
            speeches=frames["speeches"], labels=frames["labels"], memberships=frames["memberships"],
            member_parties=frames["parties"], member_constituencies=frames["constituencies"],
            period=history_period, source_batch_id=batch_id, contract_version=contract_version,
        ),
        "division_party_vote_components": build_division_party_vote_components(
            frames["votes"], frames["parties"], period=history_period, source_batch_id=batch_id, contract_version=contract_version,
        ),
        "daily_question_dimensions": build_daily_question_dimensions(
            questions=frames["questions"], memberships=frames["memberships"], member_parties=frames["parties"],
            member_constituencies=frames["constituencies"], period=history_period, source_batch_id=batch_id,
            contract_version=contract_version,
        ),
        "oral_question_sections": build_oral_question_sections(
            questions=frames["questions"], speeches=frames["speeches"], debate_sections=frames["sections"],
            member_offices=frames["offices"], source_batch_id=batch_id, contract_version=contract_version,
        ),
        "oral_question_exchange_participants": build_oral_question_exchange_participants(
            questions=frames["questions"], speeches=frames["speeches"], member_offices=frames["offices"],
            source_batch_id=batch_id, contract_version=contract_version,
        ),
        "speech_question_context": build_speech_question_context(
            speeches=frames["speeches"], questions=frames["questions"], source_batch_id=batch_id,
            contract_version=contract_version,
        ),
    }

    monthly_frames: list[pd.DataFrame] = []
    for period in completed_months:
        monthly_frames.append(build_monthly_results(frames=frames, period=period, source_batch_id=batch_id, contract_version=contract_version))
    datasets["monthly_metric_results"] = pd.concat(monthly_frames, ignore_index=True)

    published = {}
    for name, frame in datasets.items():
        published[name] = publish_dataset_to_candidate(
            s3, bucket=args.bucket, batch_id=batch_id, frame=frame,
            dataset=get_dataset_contract(contract, name), contract_version=contract_version, source_batch_id=batch_id,
        )

    report = {
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_period": {"start": history_period.start.isoformat(), "end": history_period.end.isoformat()},
        "completed_months": [period.label for period in completed_months],
        "classifier_gate": classifier_gate,
        "question_context_policy": {
            "question_classifier_run": False,
            "written_questions_are_standalone_records": True,
            "oral_questions_anchor_debate_sections": True,
            "speech_context_values": ["oral_question_exchange", "other"],
            "exchange_participant_roles": ["ministerial", "chair", "ordinary_member", "collective_or_unidentified"],
            "question_taker_attribution_materialized": False,
        },
        "datasets": {
            name: {"row_count": result["row_count"], "entry_name": result["entry_name"], "objects": result["objects"]}
            for name, result in published.items()
        },
        "production_pointer_changed": False,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
