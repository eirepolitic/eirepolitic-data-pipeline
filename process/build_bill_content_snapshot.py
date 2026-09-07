#!/usr/bin/env python3
"""Build reusable Bill tracker artifacts for editorial/social use.

Reads current production through the active immutable-batch pointer. By default
it is read-only. Optional editorial-state persistence writes only under a
separate caller-supplied S3 prefix and never changes production pointers.
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
    p = argparse.ArgumentParser(description="Build Bill tracker editorial artifacts.")
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET", "eirepolitic-data"))
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ca-central-1")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--previous-snapshot", default="", help="Optional local previous bill_content_snapshot.csv")
    p.add_argument(
        "--state-prefix",
        default="",
        help="Optional S3 editorial-state prefix. If supplied, latest prior snapshot is read and the new audited snapshot is persisted.",
    )
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


def _read_s3_csv_if_exists(s3, *, bucket: str, key: str) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str, keep_default_na=False)


def _safe_state_prefix(value: str) -> str:
    prefix = str(value or "").strip().strip("/")
    if not prefix:
        return ""
    if ".." in prefix.split("/"):
        raise ValueError("state prefix may not contain '..'")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./"
    if any(ch not in allowed for ch in prefix):
        raise ValueError("state prefix contains unsupported characters")
    return prefix


def _persist_state(
    s3,
    *,
    bucket: str,
    prefix: str,
    snapshot: pd.DataFrame,
    manifest: dict,
    as_of_date: str,
) -> dict[str, str]:
    prefix = _safe_state_prefix(prefix)
    if not prefix:
        return {}
    csv_bytes = snapshot.to_csv(index=False).encode("utf-8")
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    dated_root = f"{prefix}/snapshot_date={as_of_date}"
    keys = {
        "dated_snapshot": f"{dated_root}/bill_content_snapshot.csv",
        "dated_manifest": f"{dated_root}/manifest.json",
        "latest_snapshot": f"{prefix}/latest/bill_content_snapshot.csv",
        "latest_manifest": f"{prefix}/latest/manifest.json",
    }
    s3.put_object(Bucket=bucket, Key=keys["dated_snapshot"], Body=csv_bytes, ContentType="text/csv")
    s3.put_object(Bucket=bucket, Key=keys["dated_manifest"], Body=manifest_bytes, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=keys["latest_snapshot"], Body=csv_bytes, ContentType="text/csv")
    s3.put_object(Bucket=bucket, Key=keys["latest_manifest"], Body=manifest_bytes, ContentType="application/json")
    return keys


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
        "Use only source-backed title, stage/House, sponsor, certified debate links and proposition-specific certified division data directly. Plain-English Bill summaries and pro/con argument summaries require a separate sourced editorial step. Never treat debate participation as support or opposition without explicit position or vote evidence.",
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

    state_prefix = _safe_state_prefix(args.state_prefix)
    previous = None
    previous_source = "none"
    if args.previous_snapshot:
        previous = pd.read_csv(args.previous_snapshot, dtype=str, keep_default_na=False)
        previous_source = f"local:{args.previous_snapshot}"
    elif state_prefix:
        state_key = f"{state_prefix}/latest/bill_content_snapshot.csv"
        previous = _read_s3_csv_if_exists(s3, bucket=args.bucket, key=state_key)
        previous_source = f"s3:{state_key}" if previous is not None else "none_first_persisted_edition"

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
        "previous_snapshot_source": previous_source,
        "logical_source_keys": TABLE_KEYS,
        "resolved_source_keys": resolved_keys,
        "production_changed": False,
        "classifier_calls": 0,
        "snapshot_audit": snapshot_audit,
        "series_audit": series_audit,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "series_plan.md").write_text(_series_plan_markdown(series, series_audit, generated_at), encoding="utf-8")

    persisted_keys: dict[str, str] = {}
    if state_prefix:
        persisted_keys = _persist_state(
            s3,
            bucket=args.bucket,
            prefix=state_prefix,
            snapshot=snapshot,
            manifest=manifest,
            as_of_date=as_of_date.isoformat(),
        )
    manifest["editorial_state_persisted"] = bool(persisted_keys)
    manifest["editorial_state_keys"] = persisted_keys
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not series.empty:
        print("\nEDITORIAL_BATCHES")
        print(series.groupby(["editorial_bucket_label", "editorial_batch_id"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
