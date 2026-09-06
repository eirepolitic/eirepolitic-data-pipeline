#!/usr/bin/env python3
"""Build a reusable, read-only Bill content snapshot for editorial/social use.

Reads the currently published Oireachtas silver/metric tables from S3 and writes
local CSV/JSON/Markdown artifacts only. It does not update any production pointer
or S3 object.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from political_metrics.bill_content_snapshot import (
    BATCH_SIZE_DEFAULT,
    audit_bill_content_snapshot,
    build_bill_content_snapshot,
)

TABLE_KEYS = {
    "bills": "processed/oireachtas_unified/latest/csv/silver_bills.csv",
    "stages": "processed/oireachtas_unified/latest/csv/silver_bill_stages.csv",
    "sponsors": "processed/oireachtas_unified/latest/csv/silver_bill_sponsors.csv",
    "bridge": "processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv",
    "speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "divisions": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
    "member_votes": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build read-only Bill content snapshot artifacts.")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--output-dir", default="artifacts/bill-content-snapshot")
    return p


def _read_csv(s3, *, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False)


def _series_plan_markdown(snapshot: pd.DataFrame, audit: dict, generated_at: str) -> str:
    lines = [
        "# Bill content snapshot — proposed Instagram series",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This is a deterministic editorial planning artifact. It does not contain AI-generated Bill summaries.",
        "",
        "## Snapshot summary",
        "",
        f"- Bills: **{audit['bill_count']}**",
        f"- Proposed carousel batches at six Bills each: **{audit['series_batch_count']}**",
        f"- Bills with certified linked debate: **{audit['bills_with_certified_debate']}**",
        f"- Bills with certified recorded vote evidence: **{audit['bills_with_recorded_vote_evidence']}**",
        "",
        "## Proposed batches",
        "",
    ]
    for batch_id, group in snapshot.groupby("series_batch_id", sort=False):
        label = group.iloc[0]["series_bucket_label"]
        batch_no = int(group.iloc[0]["series_batch_no"])
        batch_count = int(group.iloc[0]["series_batch_count"])
        suffix = f" — batch {batch_no}/{batch_count}" if batch_count > 1 else ""
        lines.extend([f"### {label}{suffix}", ""])
        for row in group.itertuples(index=False):
            title = row.short_title or row.title
            sponsor = row.primary_sponsor_name or row.primary_sponsor_role_name or "Sponsor needs editorial review"
            house = row.house_badge or "House not recorded"
            vote = (
                f"latest linked division {row.latest_division_ta} Tá / {row.latest_division_nil} Níl / {row.latest_division_abstain} abstain"
                if int(row.certified_division_count) > 0 else "no certified linked division — do not infer support/opposition from speeches"
            )
            debate = f"{row.certified_speech_count} certified linked transcript interventions"
            lines.append(
                f"- **{title}** — {row.current_stage_name or row.status}; {house}; introduced by/source sponsor: {sponsor}; {debate}; {vote}."
            )
        lines.append("")
    lines.extend([
        "## Card policy",
        "",
        "Each Bill card may use source-backed title, current stage/House, sponsor, certified linked debate count and certified recorded division data. Plain-English Bill summaries and pro/con argument summaries remain editorial fields until separately sourced and reviewed. A speaker appearing in a debate must never be treated as a supporter or detractor without vote/explicit-position evidence.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    generated_at = datetime.now(timezone.utc).isoformat()
    s3 = boto3.client("s3", region_name=args.region)
    frames = {name: _read_csv(s3, bucket=args.bucket, key=key) for name, key in TABLE_KEYS.items()}
    snapshot = build_bill_content_snapshot(
        bills=frames["bills"],
        stages=frames["stages"],
        sponsors=frames["sponsors"],
        bill_debate_sections=frames["bridge"],
        speeches=frames["speeches"],
        divisions=frames["divisions"],
        member_votes=frames["member_votes"],
        batch_size=args.batch_size,
        generated_at_utc=generated_at,
    )
    audit = audit_bill_content_snapshot(snapshot, batch_size=args.batch_size)
    if not audit.get("ready"):
        raise RuntimeError(f"Bill content snapshot audit failed: {audit}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output_dir / "bill_content_snapshot.csv", index=False)
    manifest = {
        "generated_at_utc": generated_at,
        "batch_size": args.batch_size,
        "source_keys": TABLE_KEYS,
        "production_changed": False,
        "classifier_calls": 0,
        "audit": audit,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "series_plan.md").write_text(_series_plan_markdown(snapshot, audit, generated_at), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("\nSERIES_BATCHES")
    print(snapshot.groupby(["series_bucket_label", "series_batch_id"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
