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
from political_metrics.division_context import audit_division_context, build_division_context
from political_metrics.materialize import get_dataset_contract, load_materialization_contract

CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"
SOURCE_KEYS = {
    "divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
    "member_votes": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
    "speech_context": "processed/oireachtas_unified/latest/metrics/event/speech_context/csv/speech_context.csv",
    "bill_debate_sections": "processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv",
}


def _read_candidate_csv(s3, *, bucket: str, batch_id: str, logical_key: str) -> pd.DataFrame:
    key = batch_key_for_production_key(logical_key, batch_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build deterministic division context inside one candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--report-path", default="division_context_candidate_report.json")
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
    context = build_division_context(
        divisions=frames["divisions"],
        speech_context=frames["speech_context"],
        bill_debate_sections=frames["bill_debate_sections"],
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    audit = audit_division_context(
        division_context=context,
        divisions=frames["divisions"],
        member_votes=frames["member_votes"],
        bill_debate_sections=frames["bill_debate_sections"],
    )
    if not audit.get("ready"):
        raise RuntimeError(f"division_context audit failed: {audit}")

    result = publish_dataset_to_candidate(
        s3,
        bucket=args.bucket,
        batch_id=batch_id,
        frame=context,
        dataset=get_dataset_contract(contract, "division_context"),
        contract_version=contract_version,
        source_batch_id=batch_id,
    )
    report = {
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "division_context",
        "row_count": int(len(context)),
        "audit": audit,
        "published": {"entry_name": result["entry_name"], "objects": result["objects"]},
        "production_pointer_changed": False,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
