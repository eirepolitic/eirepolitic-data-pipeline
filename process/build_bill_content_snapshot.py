#!/usr/bin/env python3
"""Build reusable, read-only Bill tracker artifacts for editorial/social use.

Reads current production through the active immutable-batch pointer. Writes local
artifacts only; no S3 object or production pointer is changed.
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

from extract.oireachtas.batch import resolve_production_key
from political_metrics.bill_content_snapshot import (
    BATCH_SIZE_DEFAULT,
    audit_bill_content_snapshot,
    build_bill_content_snapshot,
)
from political_metrics.bill_editorial_series import (
    audit_editorial_bill_series,
    build_editorial_bill_series,
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
    p = argparse.ArgumentParser(description="Build read-only Bill tracker artifacts.")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--previous-snapshot", default="", help="Optional previous bill_content_snapshot.csv for delta mode")
    p.add_argument("--output-dir", default="artifacts/bill-content-snapshot")
    return p


def _read_csv(s3, *, bucket: str, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved_key = resolve_production_key(s3, bucket=bucket, production_key=logical_key)
    try:
        obj = s3.get_object(Bucket=bucket, Key=resolved_key)
    except Exception as exc:
        raise FileNotFoundError(
            f"Unable to read production source logical={logical_key!r} resolved={resolved_key!r}: {exc}"
        ) from exc
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False), resolved_key


def _series_plan_markdown(series: pd.DataFrame, audit: dict, generated_at: str) -> str:
    lines = [
        "# Bill tracker — proposed Instagram series",
        "",
        f"Generated: `{generated_at}`",
        "",
        "Deterministic editorial planning artifact. No AI-generated Bill or debate summaries are included.",
        "",
        "## Edition summary",
        "",
        f"- Bills selected: **{audit['bill_count']}**",
        f"- Proposed six-Bill carousel batches: **{audit['batch_count']}**",
        f"- Scope: **{series.iloc[0]['editorial_scope_mode'] if not series.empty else 'no qualifying changes'}**",
        "",
        "## Proposed batches",
        "",
    ]
    if series.empty:
        lines.append("No Current or Enacted Bills qualified for this edition.")
        return "\n".join(lines) + "\n"

    for _batch_id, group in series.groupby("editorial_batch_id", sort=False):
        label = group.iloc[0]["editorial_bucket_label"]
        batch_no = int(group.iloc[0]["editorial_batch_no"])
        batch_count = int(group.iloc[0]["editorial_batch_count"])
        suffix = f" — batch {batch_no}/{batch_count}" if batch_count > 1 else ""
        lines.extend([f"### {label}{suffix}", ""])
        for row in group.itertuples(index=False):
            title = row.short_title or row.title
            sponsor = row.primary_sponsor_name or row.primary_sponsor_role_name or "Sponsor needs editorial review"
            house = row.house_badge or "House not recorded"
            vote = (
                f"latest linked division {row.latest_division_ta} Tá / {row.latest_division_nil} Níl / {row.latest_division_abstain} abstain"
                if int(row.certified_division_count) > 0 else "no certified linked division — do not infer support/opposition"
            )
            lines.append(
                f"- **{title}** — {row.current_stage_name or row.status}; {house}; sponsor: {sponsor}; "
                f"{row.certified_speech_count} certified linked transcript interventions; {vote}; change={row.change_type}."
            )
        lines.append("")
    lines.extend([
        "## Card policy",
        "",
        "Use only source-backed title, stage/House, sponsor, certified debate links and certified division data directly. Plain-English Bill summaries and pro/con argument summaries require a separate sourced editorial step. Never treat debate participation as support or opposition without explicit position or vote evidence.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.lookback_days < 1:
        raise ValueError("--lookback-days must be >= 1")
    generated_at = datetime.now(timezone.utc).isoformat()
    as_of_date = datetime.now(timezone.utc).date()
    s3 = boto3.client("s3", region_name=args.region)

    frames: dict[str, pd.DataFrame] = {}
    resolved_keys: dict[str, str] = {}
    for name, logical_key in TABLE_KEYS.items():
        frames[name], resolved_keys[name] = _read_csv(s3, bucket=args.bucket, logical_key=logical_key)

    snapshot = build_bill_content_snapshot(
        bills=frames["bills"], stages=frames["stages"], sponsors=frames["sponsors"],
        bill_debate_sections=frames["bridge"], speeches=frames["speeches"], divisions=frames["divisions"],
        member_votes=frames["member_votes"], batch_size=args.batch_size, generated_at_utc=generated_at,
    )
    snapshot_audit = audit_bill_content_snapshot(snapshot, batch_size=args.batch_size)
    if not snapshot_audit.get("ready"):
        raise RuntimeError(f"Bill content snapshot audit failed: {snapshot_audit}")

    previous = None
    if args.previous_snapshot:
        previous = pd.read_csv(args.previous_snapshot, dtype=str, keep_default_na=False)
    series = build_editorial_bill_series(
        snapshot,
        batch_size=args.batch_size,
        as_of_date=as_of_date,
        lookback_days=args.lookback_days,
        previous_snapshot=previous,
    )
    series_audit = audit_editorial_bill_series(series, batch_size=args.batch_size)
    if not series_audit.get("ready"):
        raise RuntimeError(f"Bill editorial-series audit failed: {series_audit}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output_dir / "bill_content_snapshot.csv", index=False)
    series.to_csv(output_dir / "bill_series_candidates.csv", index=False)
    manifest = {
        "generated_at_utc": generated_at,
        "as_of_date": as_of_date.isoformat(),
        "batch_size": args.batch_size,
        "lookback_days": args.lookback_days,
        "scope_mode": "snapshot_delta" if previous is not None else "baseline_recent",
        "logical_source_keys": TABLE_KEYS,
        "resolved_source_keys": resolved_keys,
        "production_changed": False,
        "classifier_calls": 0,
        "snapshot_audit": snapshot_audit,
        "series_audit": series_audit,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "series_plan.md").write_text(_series_plan_markdown(series, series_audit, generated_at), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not series.empty:
        print("\nEDITORIAL_BATCHES")
        print(series.groupby(["editorial_bucket_label", "editorial_batch_id"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
