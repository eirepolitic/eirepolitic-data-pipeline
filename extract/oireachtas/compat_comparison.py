"""Compare legacy downstream inputs to unified compatibility adapter outputs."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import comparison_status, load_contract_config
from .io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_bytes, make_s3_client, put_dataframe_csv, put_json, put_text
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, write_review_bundle

TABLE_NAME = "compat_adapter_comparison"

COMPARISONS = [
    {
        "comparison_name": "members_roster_compat",
        "legacy_key": "raw/members/oireachtas_members_34th_dail.csv",
        "compat_key": "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv",
        "legacy_join_column": "member_code",
        "compat_join_column": "member_code",
        "comparison_scope": "all_rows",
    },
    {
        "comparison_name": "member_votes_compat",
        "legacy_key": "processed/votes/dail_vote_member_records.csv",
        "compat_key": "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv",
        "legacy_join_column": "memberCode",
        "compat_join_column": "memberCode",
        "comparison_scope": "legacy_date_range",
        "legacy_date_column": "date",
        "compat_date_column": "date",
    },
]


@dataclass(frozen=True)
class CompatComparisonResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]


def build_compat_comparison(*, s3: Any, bucket: str, review_root: Path, sample_rows: int = 10) -> CompatComparisonResult:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_date = started_at[:10]
    _, thresholds = load_contract_config()
    rows = [_compare_one(s3=s3, bucket=bucket, config=config, thresholds=thresholds) for config in COMPARISONS]
    df = pd.DataFrame(rows)
    dq = _dq(df)
    schema = {"table": TABLE_NAME, "primary_key": ["comparison_name"], "columns": list(df.columns), "row_count": int(len(df))}
    report_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/report.md"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"
    manifest_key = f"processed/oireachtas_unified/compat/manifests/{TABLE_NAME}/run_id={run_id}.json"
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
        "primary_key_unique": bool(not df["comparison_name"].duplicated().any()),
        "dq_status": dq["dq_status"],
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
    write_review_bundle(
        table=TABLE_NAME,
        manifest=manifest,
        schema=schema,
        dq=dq,
        sample_rows=df.head(sample_rows).to_dict(orient="records"),
        root=review_root,
        sample_limit=sample_rows,
    )
    return CompatComparisonResult(rows=df.to_dict(orient="records"), manifest=manifest, schema=schema, dq=dq)


def _compare_one(*, s3: Any, bucket: str, config: dict[str, str], thresholds: dict[str, Any]) -> dict[str, Any]:
    legacy_full = _read_csv(s3, bucket=bucket, key=config["legacy_key"])
    compat_full = _read_csv(s3, bucket=bucket, key=config["compat_key"])
    legacy_df, compat_df, scope_meta = _scope_comparison_frames(legacy_full, compat_full, config)
    legacy_col = config["legacy_join_column"]
    compat_col = config["compat_join_column"]
    legacy_keys = _keys(legacy_df, legacy_col)
    compat_keys = _keys(compat_df, compat_col)
    matched = legacy_keys & compat_keys
    row: dict[str, Any] = {
        "comparison_name": config["comparison_name"],
        "legacy_key": config["legacy_key"],
        "compat_key": config["compat_key"],
        "comparison_scope": scope_meta["comparison_scope"],
        "comparison_date_start": scope_meta["comparison_date_start"],
        "comparison_date_end": scope_meta["comparison_date_end"],
        "legacy_total_rows": int(len(legacy_full)),
        "compat_total_rows": int(len(compat_full)),
        "legacy_rows": int(len(legacy_df)),
        "compat_rows": int(len(compat_df)),
        "compat_rows_before_comparison_window": scope_meta["compat_rows_before_comparison_window"],
        "compat_rows_after_comparison_window": scope_meta["compat_rows_after_comparison_window"],
        "legacy_columns": int(len(legacy_df.columns)),
        "compat_columns": int(len(compat_df.columns)),
        "legacy_join_column": legacy_col,
        "compat_join_column": compat_col,
        "legacy_join_coverage_pct": _coverage_number(legacy_df, legacy_col),
        "compat_join_coverage_pct": _coverage_number(compat_df, compat_col),
        "matched_key_count": int(len(matched)),
        "legacy_only_key_count": int(len(legacy_keys - compat_keys)),
        "compat_only_key_count": int(len(compat_keys - legacy_keys)),
    }
    threshold = thresholds.get(config["comparison_name"])
    if threshold is None:
        row["status"] = "fail"
        row["failure_reasons"] = "missing comparison threshold"
        row["row_delta_pct"] = round(abs(len(compat_df) - len(legacy_df)) / max(len(legacy_df), 1) * 100.0, 2)
        return row
    status, errors = comparison_status(row, threshold)
    row["status"] = status
    row["failure_reasons"] = "; ".join(errors)
    row["row_delta_pct"] = round(abs(len(compat_df) - len(legacy_df)) / max(len(legacy_df), 1) * 100.0, 2)
    row["max_legacy_only_keys"] = threshold.max_legacy_only_keys
    row["max_compat_only_keys"] = threshold.max_compat_only_keys
    row["max_row_delta_pct"] = threshold.max_row_delta_pct
    row["minimum_compat_join_coverage_pct"] = threshold.minimum_compat_join_coverage_pct
    return row


def _scope_comparison_frames(
    legacy_df: pd.DataFrame,
    compat_df: pd.DataFrame,
    config: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scope = str(config.get("comparison_scope") or "all_rows")
    metadata: dict[str, Any] = {
        "comparison_scope": scope,
        "comparison_date_start": "",
        "comparison_date_end": "",
        "compat_rows_before_comparison_window": 0,
        "compat_rows_after_comparison_window": 0,
    }
    if scope == "all_rows":
        return legacy_df, compat_df, metadata
    if scope != "legacy_date_range":
        raise ValueError(f"Unsupported comparison scope {scope!r} for {config.get('comparison_name')}")

    legacy_date_col = str(config.get("legacy_date_column") or "")
    compat_date_col = str(config.get("compat_date_column") or "")
    if legacy_date_col not in legacy_df.columns:
        raise ValueError(f"Legacy date column {legacy_date_col!r} is missing for {config.get('comparison_name')}")
    if compat_date_col not in compat_df.columns:
        raise ValueError(f"Compat date column {compat_date_col!r} is missing for {config.get('comparison_name')}")

    legacy_dates = pd.to_datetime(legacy_df[legacy_date_col], errors="coerce")
    valid_legacy = legacy_dates.notna()
    if not valid_legacy.any():
        raise ValueError(f"Legacy comparison has no valid dates for {config.get('comparison_name')}")
    start = legacy_dates.loc[valid_legacy].min()
    end = legacy_dates.loc[valid_legacy].max()
    compat_dates = pd.to_datetime(compat_df[compat_date_col], errors="coerce")

    legacy_mask = valid_legacy & legacy_dates.between(start, end, inclusive="both")
    compat_mask = compat_dates.notna() & compat_dates.between(start, end, inclusive="both")
    metadata.update(
        {
            "comparison_date_start": start.date().isoformat(),
            "comparison_date_end": end.date().isoformat(),
            "compat_rows_before_comparison_window": int((compat_dates.notna() & (compat_dates < start)).sum()),
            "compat_rows_after_comparison_window": int((compat_dates.notna() & (compat_dates > end)).sum()),
        }
    )
    return legacy_df.loc[legacy_mask].copy(), compat_df.loc[compat_mask].copy(), metadata


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _keys(df: pd.DataFrame, column: str) -> set[str]:
    if column not in df.columns:
        return set()
    return set(df[column].fillna("").astype(str).str.strip()) - {""}


def _coverage_number(df: pd.DataFrame, column: str) -> float:
    if len(df) == 0 or column not in df.columns:
        return 0.0
    covered = df[column].fillna("").astype(str).str.strip().ne("").sum()
    return round(covered / len(df) * 100.0, 2)


def _dq(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    pk_unique = bool("comparison_name" in df.columns and not df["comparison_name"].duplicated().any())
    failing = df[df["status"].eq("fail")]["comparison_name"].tolist() if "status" in df.columns else ["missing_status"]
    reasons = (
        df.loc[df["status"].eq("fail"), ["comparison_name", "failure_reasons"]].to_dict(orient="records")
        if {"status", "comparison_name", "failure_reasons"}.issubset(df.columns)
        else []
    )
    dq_status = "pass" if row_count > 0 and pk_unique and not failing else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": dq_status,
        "row_count": row_count,
        "primary_key": ["comparison_name"],
        "primary_key_unique": pk_unique,
        "failing_comparisons": reasons,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "no_failed_comparisons", "status": "pass" if not failing else "fail", "failing_comparisons": reasons},
        ],
    }


def _markdown_report(df: pd.DataFrame, manifest: dict[str, Any]) -> str:
    return "\n".join([
        "# Compatibility adapter comparison",
        "",
        f"Run ID: `{manifest['run_id']}`",
        "",
        "Strict configured thresholds are applied to missing keys, row divergence, and join coverage. Date-bounded legacy snapshots are compared only over their own observed date range; newer unified history is reported separately and is not treated as divergence.",
        "",
        _simple_markdown_table(df),
        "",
    ])


def _simple_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in df.fillna("").astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record.get(column, "").replace("|", "\\|")[:300] for column in columns) + " |")
    return "\n".join(rows)


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10") or "10")
    s3 = make_s3_client(region_name=region)
    result = build_compat_comparison(s3=s3, bucket=bucket, review_root=review_root, sample_rows=sample_rows)
    print(json.dumps({
        "table": TABLE_NAME,
        "rows": result.rows,
        "dq": result.dq,
        "dq_status": result.dq.get("dq_status"),
        "run_id": result.manifest.get("run_id"),
    }, indent=2, sort_keys=True))
    return 0 if result.dq.get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
