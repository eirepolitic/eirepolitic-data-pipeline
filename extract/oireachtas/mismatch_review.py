"""Review remaining member-code mismatches between legacy and unified compat outputs."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_bytes, make_s3_client, put_dataframe_csv, put_json, put_text
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, write_review_bundle

TABLE_NAME = "member_code_mismatch_review"

LEGACY_ROSTER_KEY = "raw/members/oireachtas_members_34th_dail.csv"
COMPAT_ROSTER_KEY = "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
LEGACY_PROFILE_KEY = "processed/members/member_profile_metrics_2025.csv"
TRIAL_PROFILE_KEY = "processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv"

DATASETS = [
    {
        "dataset_name": "roster",
        "legacy_key": LEGACY_ROSTER_KEY,
        "unified_key": COMPAT_ROSTER_KEY,
        "unified_label": "compat",
    },
    {
        "dataset_name": "member_profile_metrics",
        "legacy_key": LEGACY_PROFILE_KEY,
        "unified_key": TRIAL_PROFILE_KEY,
        "unified_label": "trial",
    },
]


@dataclass(frozen=True)
class MismatchReviewResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]


def build_mismatch_review(*, s3: Any, bucket: str, review_root: Path, sample_rows: int = 50) -> MismatchReviewResult:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_date = started_at[:10]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for config in DATASETS:
        legacy_df = _read_csv(s3, bucket=bucket, key=config["legacy_key"])
        unified_df = _read_csv(s3, bucket=bucket, key=config["unified_key"])
        legacy_lookup = _member_lookup(legacy_df)
        unified_lookup = _member_lookup(unified_df)
        legacy_keys = set(legacy_lookup)
        unified_keys = set(unified_lookup)
        matched = legacy_keys & unified_keys
        legacy_only = legacy_keys - unified_keys
        unified_only = unified_keys - legacy_keys
        summaries.append(
            {
                "dataset_name": config["dataset_name"],
                "legacy_rows": int(len(legacy_df)),
                "unified_rows": int(len(unified_df)),
                "legacy_member_count": int(len(legacy_keys)),
                "unified_member_count": int(len(unified_keys)),
                "matched_member_count": int(len(matched)),
                "legacy_only_count": int(len(legacy_only)),
                "unified_only_count": int(len(unified_only)),
            }
        )
        rows.extend(
            _detail_rows(
                dataset_name=config["dataset_name"],
                side="legacy_only",
                keys=legacy_only,
                primary_lookup=legacy_lookup,
                secondary_lookup=unified_lookup,
                primary_key_label="legacy_key",
                secondary_key_label="unified_key",
            )
        )
        rows.extend(
            _detail_rows(
                dataset_name=config["dataset_name"],
                side=f"{config['unified_label']}_only",
                keys=unified_only,
                primary_lookup=unified_lookup,
                secondary_lookup=legacy_lookup,
                primary_key_label="unified_key",
                secondary_key_label="legacy_key",
            )
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(columns=["review_id", "dataset_name", "side", "member_code", "full_name", "party", "constituency", "source_hint", "other_side_present"])
    detail_df = detail_df.sort_values(["dataset_name", "side", "member_code"]).reset_index(drop=True)
    dq = _dq(detail_df, summaries)
    schema = {"table": TABLE_NAME, "primary_key": ["review_id"], "columns": list(detail_df.columns), "row_count": int(len(detail_df))}
    report_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/report.md"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"
    manifest_key = f"processed/oireachtas_unified/compat/manifests/{TABLE_NAME}/run_id={run_id}.json"
    manifest = {
        "table": TABLE_NAME,
        "mode": "review",
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "output_rows": int(len(detail_df)),
        "summary": summaries,
        "primary_key": ["review_id"],
        "primary_key_unique": bool(not detail_df["review_id"].duplicated().any()) if "review_id" in detail_df else True,
        "dq_status": dq["dq_status"],
        "s3_keys": {
            "manifest": manifest_key,
            "review_sample": review_sample_key,
            "review_schema": review_schema_key,
            "review_manifest": review_manifest_key,
            "review_report": report_key,
        },
    }
    report = _markdown_report(detail_df, manifest)
    put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
    put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=detail_df.head(sample_rows))
    put_json(s3, bucket=bucket, key=review_schema_key, payload=schema)
    put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    put_text(s3, bucket=bucket, key=report_key, text=report)
    write_review_bundle(table=TABLE_NAME, manifest=manifest, schema=schema, dq=dq, sample_rows=detail_df.head(sample_rows).to_dict(orient="records"), root=review_root)
    (review_root / "review" / TABLE_NAME / "latest" / "report.md").write_text(report, encoding="utf-8")
    return MismatchReviewResult(rows=detail_df.to_dict(orient="records"), manifest=manifest, schema=schema, dq=dq)


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _member_lookup(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    if "member_code" not in df.columns:
        return {}
    lookup: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        member_code = str(row.get("member_code", "")).strip()
        if not member_code:
            continue
        lookup[member_code] = {
            "member_code": member_code,
            "full_name": _coalesce(row, "full_name", "member_name", "name"),
            "party": _coalesce(row, "party", "party_name", "latest_party_name"),
            "constituency": _coalesce(row, "constituency", "constituency_name", "latest_constituency_name"),
            "source_hint": _coalesce(row, "source", "snapshot_date", "house_no"),
        }
    return lookup


def _coalesce(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row:
            value = str(row.get(column, "")).strip()
            if value:
                return value
    return ""


def _detail_rows(
    *,
    dataset_name: str,
    side: str,
    keys: set[str],
    primary_lookup: dict[str, dict[str, str]],
    secondary_lookup: dict[str, dict[str, str]],
    primary_key_label: str,
    secondary_key_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for member_code in sorted(keys):
        primary = primary_lookup.get(member_code, {})
        rows.append(
            {
                "review_id": f"{dataset_name}:{side}:{member_code}",
                "dataset_name": dataset_name,
                "side": side,
                "member_code": member_code,
                "full_name": primary.get("full_name", ""),
                "party": primary.get("party", ""),
                "constituency": primary.get("constituency", ""),
                "source_hint": primary.get("source_hint", ""),
                primary_key_label: "present",
                secondary_key_label: "missing",
                "other_side_present": member_code in secondary_lookup,
            }
        )
    return rows


def _dq(df: pd.DataFrame, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = int(len(df))
    pk_unique = bool("review_id" in df.columns and not df["review_id"].duplicated().any()) if row_count else True
    # This is a review table; mismatches are informational, not a DQ failure.
    status = "pass" if pk_unique else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": ["review_id"],
        "primary_key_unique": pk_unique,
        "summary": summaries,
        "checks": [
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "review_generated", "status": "pass", "metric_value": row_count},
        ],
    }


def _markdown_report(df: pd.DataFrame, manifest: dict[str, Any]) -> str:
    summary_df = pd.DataFrame(manifest.get("summary", []))
    lines = [
        "# Member-code mismatch review",
        "",
        f"Run ID: `{manifest['run_id']}`",
        "",
        "This is a review-only report. Remaining mismatches are not treated as a pipeline failure by themselves.",
        "",
        "## Summary",
        "",
        _simple_markdown_table(summary_df),
        "",
        "## Detail",
        "",
        _simple_markdown_table(df),
        "",
    ]
    return "\n".join(lines)


def _simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in df.fillna("").astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record.get(column, "").replace("|", "\\|")[:300] for column in columns) + " |")
    return "\n".join(rows)


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    sample_rows = int(os.getenv("SAMPLE_ROWS", "50") or "50")
    s3 = make_s3_client(region_name=region)
    result = build_mismatch_review(s3=s3, bucket=bucket, review_root=review_root, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "rows": len(result.rows), "dq_status": result.dq.get("dq_status"), "summary": result.manifest.get("summary"), "run_id": result.manifest.get("run_id")}, indent=2, sort_keys=True))
    return 0 if result.dq.get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
