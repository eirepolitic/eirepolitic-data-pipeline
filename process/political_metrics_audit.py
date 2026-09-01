#!/usr/bin/env python3
"""Read-only audit of political-metric prerequisites in the promoted Oireachtas batch."""

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
from political_metrics.audit import history_coverage, speech_count_reconciliation, speech_temporal_attribution_audit
from political_metrics.sources import canonical_speeches

BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
OUT_DIR = Path(os.getenv("POLITICAL_METRICS_AUDIT_DIR", "artifacts/political-metrics-audit"))

TABLE_KEYS = {
    "silver_speeches": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
    "silver_member_memberships": "processed/oireachtas_unified/latest/csv/silver_member_memberships.csv",
    "silver_member_parties": "processed/oireachtas_unified/latest/csv/silver_member_parties.csv",
    "silver_member_constituencies": "processed/oireachtas_unified/latest/csv/silver_member_constituencies.csv",
}


def _read_csv(s3, logical_key: str) -> tuple[pd.DataFrame, str]:
    resolved_key = resolve_production_key(s3, bucket=BUCKET, production_key=logical_key)
    obj = s3.get_object(Bucket=BUCKET, Key=resolved_key)
    body = obj["Body"].read()
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False, na_values=[""]), resolved_key


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, str | None]:
    values = pd.to_datetime(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "min": values.min().date().isoformat() if values.notna().any() else None,
        "max": values.max().date().isoformat() if values.notna().any() else None,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_markdown(report: dict) -> str:
    attribution = report["speech_temporal_attribution"]
    reconciliation = report["speech_reconciliation"]
    certification = report["certification"]
    histories = report["history_coverage"]

    status_text = "READY" if certification["ready_for_historical_public_speech_metrics"] else "NOT YET READY"
    lines = [
        "# Political metrics historical audit",
        "",
        f"**Result: {status_text}**",
        "",
        f"Promoted batch: `{report['production_batch_id']}`",
        f"Speech dates in the promoted data: **{report['speech_date_range']['min']} to {report['speech_date_range']['max']}**",
        "",
        "## What this means",
        "",
    ]
    if certification["ready_for_historical_public_speech_metrics"]:
        lines.append(
            "All member-attributable speeches in this promoted batch can be assigned to a single historical party and constituency, and the history tables contain no overlapping date ranges detected by this audit."
        )
    else:
        lines.append(
            "The promoted data still contains gaps or ambiguities that could make historical party or constituency speech statistics misleading. Public historical metrics should remain uncertified until the listed checks pass."
        )

    lines.extend([
        "",
        "## Attribution coverage",
        "",
        f"- Historical party assignment: **{_pct(attribution['party_attribution_coverage'])}** ({attribution['party_unmatched_rows']} unmatched member-attributable speech rows)",
        f"- Historical constituency assignment: **{_pct(attribution['constituency_attribution_coverage'])}** ({attribution['constituency_unmatched_rows']} unmatched member-attributable speech rows)",
        f"- Speech-to-member attribution overall: **{_pct(reconciliation['member_attribution_coverage'])}**",
        "",
        "The final figure is not expected to prove attendance or party membership; it simply shows how many canonical speech records identify a member. Historical party/constituency coverage is assessed only for speeches that identify a member.",
        "",
        "## History available",
        "",
    ])

    for key in ["silver_member_memberships", "silver_member_parties", "silver_member_constituencies"]:
        item = histories[key]
        max_text = item["max_end"] or "open-ended/current"
        lines.append(
            f"- **{key}**: {item['row_count']} rows across {item['entity_count']} members; starts as early as **{item['min_start']}**; latest recorded end **{max_text}**; {len(item['validation_errors'])} validation error(s)."
        )

    lines.extend([
        "",
        "## Certification checks",
        "",
    ])
    for check, passed in certification["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {check.replace('_', ' ')}")

    lines.extend([
        "",
        "## Important interpretation",
        "",
        "Passing this audit means the promoted batch is structurally safe for the first historical speech measures. It does **not** certify issue classification, voting, attendance, questions, government/opposition status, or older periods that are not present in the promoted data.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    pointer = read_json_required(s3, bucket=BUCKET, key=PRODUCTION_POINTER_KEY)
    batch_id = str(pointer.get("batch_id") or pointer.get("mode") or "unknown")

    frames: dict[str, pd.DataFrame] = {}
    resolved_keys: dict[str, str] = {}
    for table, logical_key in TABLE_KEYS.items():
        frames[table], resolved_keys[table] = _read_csv(s3, logical_key)

    speeches = canonical_speeches(frames["silver_speeches"])
    memberships = frames["silver_member_memberships"]
    parties = frames["silver_member_parties"]
    constituencies = frames["silver_member_constituencies"]

    histories = {
        "silver_member_memberships": history_coverage(
            memberships,
            dataset="silver_member_memberships",
            entity_col="member_code",
            start_col="membership_start",
            end_col="membership_end",
            detail_columns=["membership_id", "house_no", "chamber"],
        ).as_dict(),
        "silver_member_parties": history_coverage(
            parties,
            dataset="silver_member_parties",
            entity_col="member_code",
            start_col="party_start",
            end_col="party_end",
            detail_columns=["party_uri", "party_name", "membership_id"],
        ).as_dict(),
        "silver_member_constituencies": history_coverage(
            constituencies,
            dataset="silver_member_constituencies",
            entity_col="member_code",
            start_col="represent_start",
            end_col="represent_end",
            detail_columns=["constituency_uri", "constituency_name", "membership_id"],
        ).as_dict(),
    }

    attribution = speech_temporal_attribution_audit(speeches, parties, constituencies)
    reconciliation = speech_count_reconciliation(speeches)

    checks = {
        "membership_history_has_no_overlaps_or_invalid_ranges": not histories["silver_member_memberships"]["validation_errors"],
        "party_history_has_no_overlaps_or_invalid_ranges": not histories["silver_member_parties"]["validation_errors"],
        "constituency_history_has_no_overlaps_or_invalid_ranges": not histories["silver_member_constituencies"]["validation_errors"],
        "all_member_attributable_speeches_have_historical_party": attribution["party_attribution_coverage"] == 1.0,
        "all_member_attributable_speeches_have_historical_constituency": attribution["constituency_attribution_coverage"] == 1.0,
        "speech_counts_reconcile": bool(reconciliation["reconciles"]),
    }

    report = {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": BUCKET,
        "production_pointer": pointer,
        "production_batch_id": batch_id,
        "resolved_source_keys": resolved_keys,
        "speech_date_range": _date_range(speeches, "debate_date"),
        "history_coverage": histories,
        "speech_reconciliation": reconciliation,
        "speech_temporal_attribution": attribution,
        "certification": {
            "ready_for_historical_public_speech_metrics": all(checks.values()),
            "checks": checks,
            "scope": "speech metrics requiring period-correct member, party and constituency context in this promoted batch",
        },
    }

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(_build_markdown(report), encoding="utf-8")

    print(_build_markdown(report))
    print(f"\nAudit artifacts written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
