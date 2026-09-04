#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from extract.oireachtas.batch import batch_key_for_production_key, validate_batch_id
from political_metrics.candidate_publish import publish_dataset_to_candidate
from political_metrics.contextual_votes import (
    audit_context_vote_reconciliation,
    build_context_division_party_vote_components,
    build_daily_context_vote_components,
)
from political_metrics.materialize import get_dataset_contract, load_materialization_contract
from political_metrics.periods import MetricPeriod

CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"
SOURCE_KEYS = {
    "divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
    "member_votes": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
    "memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
    "division_context": "processed/oireachtas_unified/latest/metrics/event/division_context/csv/division_context.csv",
    "daily_activity_components": "processed/oireachtas_unified/latest/metrics/daily/daily_activity_components/csv/daily_activity_components.csv",
    "division_party_vote_components": "processed/oireachtas_unified/latest/metrics/event/division_party_vote_components/csv/division_party_vote_components.csv",
}


def _read_candidate_csv(s3, *, bucket: str, batch_id: str, logical_key: str) -> pd.DataFrame:
    key = batch_key_for_production_key(logical_key, batch_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])


def _period(divisions: pd.DataFrame, member_votes: pd.DataFrame) -> MetricPeriod:
    dates = pd.concat([
        pd.to_datetime(divisions["division_date"], errors="coerce"),
        pd.to_datetime(member_votes["division_date"], errors="coerce"),
    ]).dropna()
    if dates.empty:
        raise ValueError("candidate contains no division dates")
    start = dates.min().date()
    end = dates.max().date()
    return MetricPeriod(start, end, f"{start.isoformat()}_{end.isoformat()}", "candidate_history")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build context-aware additive voting foundations in one candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--report-path", default="contextual_votes_candidate_report.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    batch_id = validate_batch_id(args.batch_id)
    s3 = boto3.client("s3", region_name=args.region)
    frames = {
        name: _read_candidate_csv(s3, bucket=args.bucket, batch_id=batch_id, logical_key=key)
        for name, key in SOURCE_KEYS.items()
    }
    period = _period(frames["divisions"], frames["member_votes"])
    contract = load_materialization_contract(CONTRACT_PATH)
    contract_version = int(contract["contract_version"])

    daily = build_daily_context_vote_components(
        divisions=frames["divisions"],
        member_votes=frames["member_votes"],
        memberships=frames["memberships"],
        member_parties=frames["parties"],
        member_constituencies=frames["constituencies"],
        division_context=frames["division_context"],
        period=period,
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    party = build_context_division_party_vote_components(
        member_votes=frames["member_votes"],
        member_parties=frames["parties"],
        division_context=frames["division_context"],
        period=period,
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    audit = audit_context_vote_reconciliation(
        daily_context_vote_components=daily,
        context_division_party_vote_components=party,
        daily_activity_components=frames["daily_activity_components"],
        division_party_vote_components=frames["division_party_vote_components"],
    )
    if not audit.get("ready"):
        raise RuntimeError(f"context-aware vote reconciliation failed: {audit}")

    published = {}
    for name, frame in [
        ("daily_context_vote_components", daily),
        ("context_division_party_vote_components", party),
    ]:
        result = publish_dataset_to_candidate(
            s3,
            bucket=args.bucket,
            batch_id=batch_id,
            frame=frame,
            dataset=get_dataset_contract(contract, name),
            contract_version=contract_version,
            source_batch_id=batch_id,
        )
        published[name] = {"row_count": int(len(frame)), "entry_name": result["entry_name"], "objects": result["objects"]}

    report = {
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start": period.start.isoformat(), "end": period.end.isoformat()},
        "audit": audit,
        "datasets": published,
        "production_pointer_changed": False,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
