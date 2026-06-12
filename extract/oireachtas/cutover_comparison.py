"""Review-only comparison report between legacy and unified Oireachtas outputs."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_bytes, make_s3_client, object_exists, put_dataframe_csv, put_json, put_text
from .normalize import stable_hash, utc_now_iso
from .review import REVIEW_ROOT, write_review_bundle

TABLE_NAME = "cutover_comparison_report"

LEGACY_UNIFIED_PAIRS = [
    {
        "comparison_name": "members_current_roster",
        "legacy_key": "raw/members/oireachtas_members_34th_dail.csv",
        "unified_key": "processed/oireachtas_unified/latest/csv/silver_members.csv",
        "legacy_join_column": "member_code",
        "unified_join_column": "member_code",
    },
    {
        "comparison_name": "speeches",
        "legacy_key": "raw/debates/debate_speeches_extracted.csv",
        "unified_key": "processed/oireachtas_unified/latest/csv/silver_speeches.csv",
        "legacy_join_column": "speech_id",
        "unified_join_column": "speech_id",
    },
    {
        "comparison_name": "vote_divisions",
        "legacy_key": "processed/votes/dail_vote_divisions.csv",
        "unified_key": "processed/oireachtas_unified/latest/csv/silver_divisions.csv",
        "legacy_join_column": "division_id",
        "unified_join_column": "division_id",
    },
    {
        "comparison_name": "member_votes",
        "legacy_key": "processed/votes/dail_vote_member_records.csv",
        "unified_key": "processed/oireachtas_unified/latest/csv/silver_member_votes.csv",
        "legacy_join_column": "member_vote_id",
        "unified_join_column": "member_vote_id",
    },
    {
        "comparison_name": "member_profile_metrics_yearly",
        "legacy_key": "processed/members/member_profile_metrics_2025.csv",
        "unified_key": "processed/oireachtas_unified/latest/csv/gold_member_activity_yearly.csv",
        "legacy_join_column": "member_code",
        "unified_join_column": "member_code",
    },
]


@dataclass(frozen=True)
class ComparisonResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    dq: dict[str, Any]


def build_cutover_comparison_report(
    *,
    s3: Any,
    bucket: str,
    review_root: Path,
    sample_rows: int,
) -> ComparisonResult:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_date = started_at[:10]

    rows = [_compare_pair(s3, bucket=bucket, pair=pair) for pair in LEGACY_UNIFIED_PAIRS]
    df = pd.DataFrame(rows)
    csv_key = f"processed/oireachtas_unified/reports/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    manifest_key = f"processed/oireachtas_unified/reports/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/manifest.json"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"

    dq = _dq(df)
    schema = {"table": TABLE_NAME, "primary_key": ["comparison_name"], "columns": list(df.columns), "row_count": int(len(df))}
    manifest = {
        "table": TABLE_NAME,
        "mode": "compare",
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "output_rows": int(len(df)),
        "primary_key": ["comparison_name"],
        "primary_key_unique": bool(not df["comparison_name"].duplicated().any()) if "comparison_name" in df else False,
        "dq_status": dq["dq_status"],
        "s3_keys": {
            "csv": csv_key,
            "latest_csv": latest_csv_key,
            "manifest": manifest_key,
            "review_sample": review_sample_key,
            "review_schema": review_schema_key,
            "review_manifest": review_manifest_key,
        },
    }

    put_dataframe_csv(s3, bucket=bucket, key=csv_key, df=df)
    put_dataframe_csv(s3, bucket=bucket, key=latest_csv_key, df=df)
    put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
    put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=df.head(sample_rows))
    put_json(s3, bucket=bucket, key=review_schema_key, payload=schema)
    put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    put_text(s3, bucket=bucket, key=f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/report.md", text=_markdown_report(df, manifest))
    write_review_bundle(table=TABLE_NAME, manifest=manifest, schema=schema, dq=dq, sample_rows=df.head(sample_rows).to_dict(orient="records"), root=review_root)
    (review_root / "review" / TABLE_NAME / "latest" / "report.md").write_text(_markdown_report(df, manifest), encoding="utf-8")

    return ComparisonResult(rows=df.to_dict(orient="records"), manifest=manifest, dq=dq)


def _compare_pair(s3: Any, *, bucket: str, pair: dict[str, str]) -> dict[str, Any]:
    legacy_key = pair["legacy_key"]
    unified_key = pair["unified_key"]
    legacy_exists = object_exists(s3, bucket=bucket, key=legacy_key)
    unified_exists = object_exists(s3, bucket=bucket, key=unified_key)
    legacy_df = _read_csv(s3, bucket=bucket, key=legacy_key) if legacy_exists else pd.DataFrame()
    unified_df = _read_csv(s3, bucket=bucket, key=unified_key) if unified_exists else pd.DataFrame()
    legacy_join = pair["legacy_join_column"]
    unified_join = pair["unified_join_column"]
    legacy_key_coverage = _coverage(legacy_df, legacy_join)
    unified_key_coverage = _coverage(unified_df, unified_join)
    matched = ""
    legacy_only = ""
    unified_only = ""
    if legacy_join in legacy_df.columns and unified_join in unified_df.columns:
        legacy_values = set(legacy_df[legacy_join].fillna("").astype(str).str.strip()) - {""}
        unified_values = set(unified_df[unified_join].fillna("").astype(str).str.strip()) - {""}
        matched = len(legacy_values & unified_values)
        legacy_only = len(legacy_values - unified_values)
        unified_only = len(unified_values - legacy_values)
    status = "pass" if unified_exists and len(unified_df) > 0 else "fail"
    if not legacy_exists:
        status = "warn"
    return {
        "comparison_name": pair["comparison_name"],
        "status": status,
        "legacy_key": legacy_key,
        "unified_key": unified_key,
        "legacy_exists": str(legacy_exists).lower(),
        "unified_exists": str(unified_exists).lower(),
        "legacy_rows": int(len(legacy_df)),
        "unified_rows": int(len(unified_df)),
        "legacy_columns": int(len(legacy_df.columns)),
        "unified_columns": int(len(unified_df.columns)),
        "legacy_join_column": legacy_join,
        "unified_join_column": unified_join,
        "legacy_join_coverage_pct": legacy_key_coverage,
        "unified_join_coverage_pct": unified_key_coverage,
        "matched_key_count": matched,
        "legacy_only_key_count": legacy_only,
        "unified_only_key_count": unified_only,
        "comparison_id": f"cmp:{stable_hash([pair['comparison_name'], legacy_key, unified_key], length=24)}",
    }


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _coverage(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return ""
    populated = df[column].fillna("").astype(str).str.strip().ne("").sum()
    return f"{(populated / max(1, len(df)) * 100):.2f}"


def _dq(df: pd.DataFrame) -> dict[str, Any]:
    missing_columns = sorted(set(["comparison_name", "status", "legacy_key", "unified_key", "unified_exists", "unified_rows"]) - set(df.columns))
    row_count = int(len(df))
    pk_unique = bool("comparison_name" in df.columns and not df["comparison_name"].duplicated().any())
    unified_outputs_present = bool("unified_exists" in df.columns and df["unified_exists"].eq("true").all())
    status = "pass" if row_count > 0 and not missing_columns and pk_unique and unified_outputs_present else "warn"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": ["comparison_name"],
        "primary_key_unique": pk_unique,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "unified_outputs_present", "status": "pass" if unified_outputs_present else "warn"},
        ],
    }


def _markdown_report(df: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        "# Oireachtas cutover comparison report",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Snapshot date: `{manifest['snapshot_date']}`",
        "",
        "This report is review-only. It does not change downstream consumers or disable legacy pipelines.",
        "",
        df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10"))
    s3 = make_s3_client(region_name=region)
    result = build_cutover_comparison_report(s3=s3, bucket=bucket, review_root=review_root, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "rows": len(result.rows), "dq_status": result.dq.get("dq_status"), "run_id": result.manifest.get("run_id")}, indent=2, sort_keys=True))
    return 0 if result.dq.get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
