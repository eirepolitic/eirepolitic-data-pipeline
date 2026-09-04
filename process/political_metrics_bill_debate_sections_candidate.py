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
from political_metrics.legislation_context import audit_bill_debate_sections, build_bill_debate_sections
from political_metrics.materialize import get_dataset_contract, load_materialization_contract

CONTRACT_PATH = REPO_ROOT / "configs/political_metrics/materialization.yml"
SOURCE_KEYS = {
    "bill_debates": "processed/oireachtas_unified/latest/csv/silver_bill_debates.csv",
    "sections": "processed/oireachtas_unified/latest/csv/silver_debate_sections.csv",
    "speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
}


def _read_candidate_csv(s3, *, bucket: str, batch_id: str, logical_key: str) -> pd.DataFrame:
    key = batch_key_for_production_key(logical_key, batch_id)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False, na_values=[""])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build certified Bill-to-debate-section links inside one candidate batch.")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--report-path", default="bill_debate_sections_candidate_report.json")
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
    bridge = build_bill_debate_sections(
        bill_debates=frames["bill_debates"],
        debate_sections=frames["sections"],
        source_batch_id=batch_id,
        contract_version=contract_version,
    )
    audit = audit_bill_debate_sections(bridge=bridge, speeches=frames["speeches"], divisions=frames["divisions"])
    if not audit.get("ready"):
        raise RuntimeError(f"bill_debate_sections audit failed: {audit}")

    result = publish_dataset_to_candidate(
        s3,
        bucket=args.bucket,
        batch_id=batch_id,
        frame=bridge,
        dataset=get_dataset_contract(contract, "bill_debate_sections"),
        contract_version=contract_version,
        source_batch_id=batch_id,
    )
    report = {
        "batch_id": batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "bill_debate_sections",
        "row_count": int(len(bridge)),
        "audit": audit,
        "published": {
            "entry_name": result["entry_name"],
            "objects": result["objects"],
        },
        "production_pointer_changed": False,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
