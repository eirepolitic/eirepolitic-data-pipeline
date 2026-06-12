"""Review report for side-by-side member profile metrics trial output."""

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

TABLE_NAME = "member_profile_metrics_trial"
LEGACY_KEY = "processed/members/member_profile_metrics_2025.csv"
TRIAL_KEY = "processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv"
TRIAL_PARQUET_KEY = "processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet"


@dataclass(frozen=True)
class TrialReportResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]


def build_member_profile_trial_report(*, s3: Any, bucket: str, review_root: Path, sample_rows: int = 10) -> TrialReportResult:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_date = started_at[:10]
    legacy_df = _read_csv(s3, bucket=bucket, key=LEGACY_KEY)
    trial_df = _read_csv(s3, bucket=bucket, key=TRIAL_KEY)
    rows = _summary_rows(legacy_df, trial_df)
    df = pd.DataFrame(rows)
    dq = _dq(df)
    schema = {"table": TABLE_NAME, "primary_key": ["check_name"], "columns": list(df.columns), "row_count": int(len(df))}
    report_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/report.md"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"
    manifest_key = f"processed/oireachtas_unified/compat/manifests/{TABLE_NAME}/run_id={run_id}.json"
    manifest = {
        "table": TABLE_NAME,
        "mode": "trial",
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "output_rows": int(len(df)),
        "primary_key": ["check_name"],
        "primary_key_unique": bool(not df["check_name"].duplicated().any()),
        "dq_status": dq["dq_status"],
        "legacy_key": LEGACY_KEY,
        "trial_key": TRIAL_KEY,
        "trial_parquet_key": TRIAL_PARQUET_KEY,
        "s3_keys": {
            "manifest": manifest_key,
            "review_sample": review_sample_key,
            "review_schema": review_schema_key,
            "review_manifest": review_manifest_key,
            "review_report": report_key,
        },
    }
    report = _markdown_report(df, manifest)
    put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
    put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=df.head(sample_rows))
    put_json(s3, bucket=bucket, key=review_schema_key, payload=schema)
    put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    put_text(s3, bucket=bucket, key=report_key, text=report)
    write_review_bundle(table=TABLE_NAME, manifest=manifest, schema=schema, dq=dq, sample_rows=df.head(sample_rows).to_dict(orient="records"), root=review_root)
    (review_root / "review" / TABLE_NAME / "latest" / "report.md").write_text(report, encoding="utf-8")
    return TrialReportResult(rows=df.to_dict(orient="records"), manifest=manifest, schema=schema, dq=dq)


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _summary_rows(legacy_df: pd.DataFrame, trial_df: pd.DataFrame) -> list[dict[str, Any]]:
    legacy_members = _member_set(legacy_df)
    trial_members = _member_set(trial_df)
    common_columns = sorted(set(legacy_df.columns) & set(trial_df.columns))
    return [
        {"check_name": "legacy_rows", "status": "info", "legacy_value": len(legacy_df), "trial_value": "", "message": LEGACY_KEY},
        {"check_name": "trial_rows", "status": "pass" if len(trial_df) > 0 else "fail", "legacy_value": "", "trial_value": len(trial_df), "message": TRIAL_KEY},
        {"check_name": "legacy_member_count", "status": "info", "legacy_value": len(legacy_members), "trial_value": "", "message": "distinct legacy member_code"},
        {"check_name": "trial_member_count", "status": "pass" if len(trial_members) > 0 else "fail", "legacy_value": "", "trial_value": len(trial_members), "message": "distinct trial member_code"},
        {"check_name": "matched_member_count", "status": "pass" if len(legacy_members & trial_members) > 0 else "warn", "legacy_value": len(legacy_members), "trial_value": len(legacy_members & trial_members), "message": "legacy/trial member_code overlap"},
        {"check_name": "trial_only_member_count", "status": "info", "legacy_value": "", "trial_value": len(trial_members - legacy_members), "message": "member_code only in trial"},
        {"check_name": "legacy_only_member_count", "status": "info", "legacy_value": len(legacy_members - trial_members), "trial_value": "", "message": "member_code only in legacy"},
        {"check_name": "common_column_count", "status": "pass" if common_columns else "warn", "legacy_value": len(legacy_df.columns), "trial_value": len(common_columns), "message": ",".join(common_columns)},
    ]


def _member_set(df: pd.DataFrame) -> set[str]:
    if "member_code" not in df.columns:
        return set()
    return set(df["member_code"].fillna("").astype(str).str.strip()) - {""}


def _dq(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    pk_unique = bool("check_name" in df.columns and not df["check_name"].duplicated().any())
    failing = df[df["status"].eq("fail")]["check_name"].tolist() if "status" in df.columns else ["missing_status"]
    dq_status = "pass" if row_count > 0 and pk_unique and not failing else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": dq_status,
        "row_count": row_count,
        "primary_key": ["check_name"],
        "primary_key_unique": pk_unique,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "no_failed_checks", "status": "pass" if not failing else "fail", "failing_checks": failing},
        ],
    }


def _markdown_report(df: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        "# Member profile metrics side-by-side trial",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Legacy key: `{LEGACY_KEY}`",
        f"Trial key: `{TRIAL_KEY}`",
        f"Trial parquet key: `{TRIAL_PARQUET_KEY}`",
        "",
        "The trial output is non-destructive and does not replace legacy member profile metrics.",
        "",
        _simple_markdown_table(df),
        "",
    ]
    return "\n".join(lines)


def _simple_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in df.fillna("").astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record.get(column, "").replace("|", "\\|")[:400] for column in columns) + " |")
    return "\n".join(rows)


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10") or "10")
    s3 = make_s3_client(region_name=region)
    result = build_member_profile_trial_report(s3=s3, bucket=bucket, review_root=review_root, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "rows": len(result.rows), "dq_status": result.dq.get("dq_status"), "run_id": result.manifest.get("run_id")}, indent=2, sort_keys=True))
    return 0 if result.dq.get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
