"""Builder for the `control_data_quality_results` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .client import OireachtasClient
from .io_s3 import get_json, put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import stable_hash, utc_now_iso
from .schemas import TableSchema

TABLE_NAME = "control_data_quality_results"
MANIFEST_PREFIX = "processed/oireachtas_unified/manifests/"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_control_data_quality_results(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
    chamber: str | None = None,
    house_no: str | None = None,
) -> TableBuildResult:
    """Build run-level DQ result rows from manifest metadata in S3."""
    del client, chamber, house_no
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]

    manifest_keys = _list_manifest_keys(s3, bucket=bucket, prefix=MANIFEST_PREFIX)
    rows: list[dict[str, str]] = []
    read_errors: list[str] = []
    for key in manifest_keys:
        try:
            payload = get_json(s3, bucket=bucket, key=key)
            rows.extend(_manifest_checks(payload, manifest_key=key, created_at_utc=started_at))
        except Exception as exc:
            read_errors.append(f"{key}: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows, columns=schema.columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["dq_result_id"], keep="last")
        df = df.sort_values(["created_at_utc", "run_id", "check_name"], ascending=[False, False, True]).copy()

    csv_key = f"processed/oireachtas_unified/control_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/control/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    latest_parquet_key = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
    manifest_key = f"processed/oireachtas_unified/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"

    dq = _dq_results(df, schema)
    write_errors: list[str] = []
    schema_payload = {"table": TABLE_NAME, "primary_key": schema.primary_key, "columns": schema.columns, "row_count": int(len(df))}
    s3_keys = {
        "csv": csv_key,
        "parquet": parquet_key,
        "latest_csv": latest_csv_key,
        "latest_parquet": latest_parquet_key,
        "manifest": manifest_key,
        "review_sample": review_sample_key,
        "review_schema": review_schema_key,
        "review_manifest": review_manifest_key,
    }
    manifest = {
        "table": TABLE_NAME,
        "mode": mode,
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "input_prefix": MANIFEST_PREFIX,
        "manifest_objects_found": len(manifest_keys),
        "read_error_count": len(read_errors),
        "read_errors": read_errors[:25],
        "candidate_dq_rows_before_limit": len(rows),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "write_errors": write_errors,
        "s3_keys": s3_keys,
    }

    try:
        put_dataframe_csv(s3, bucket=bucket, key=csv_key, df=df)
        if not df.empty:
            put_dataframe_parquet(s3, bucket=bucket, key=parquet_key, df=df)
        put_dataframe_csv(s3, bucket=bucket, key=latest_csv_key, df=df)
        if not df.empty:
            put_dataframe_parquet(s3, bucket=bucket, key=latest_parquet_key, df=df)
        put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
        sample_df = df.head(10)
        put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=sample_df)
        put_json(s3, bucket=bucket, key=review_schema_key, payload=schema_payload)
        put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    except Exception as exc:
        write_errors.append(f"{type(exc).__name__}: {exc}")
        dq["dq_status"] = "fail"
        dq["checks"].append({"check_name": "s3_write", "status": "fail", "message": write_errors[-1]})
        manifest["status"] = "failed"
        manifest["dq_status"] = "fail"
        manifest["write_errors"] = write_errors

    sample_df = df.head(10)
    return TableBuildResult(table=TABLE_NAME, rows=sample_df.to_dict(orient="records"), manifest=manifest, schema=schema_payload, dq=dq, s3_keys=s3_keys)


def _list_manifest_keys(s3: Any, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(".json") and "/run_id=" in key:
                keys.append(key)
    return sorted(keys)


def _manifest_checks(payload: dict[str, Any], *, manifest_key: str, created_at_utc: str) -> list[dict[str, str]]:
    run_id = str(payload.get("run_id") or _run_id_from_key(manifest_key))
    table_name = str(payload.get("table") or "")
    output_rows = payload.get("output_rows") if payload.get("output_rows") is not None else payload.get("row_count", "")
    primary_key_unique = payload.get("primary_key_unique")
    dq_status = str(payload.get("dq_status") or "")
    status = str(payload.get("status") or "")
    checks = [
        ("manifest_status_success", "pass" if status == "success" else "fail", "", "success", f"manifest status={status}"),
        ("dq_status_pass", "pass" if dq_status in ("pass", "") else "fail", "", "pass", f"dq_status={dq_status}"),
        ("output_rows_gt_zero", "pass" if _positive_number(output_rows) else "fail", str(output_rows), ">0", "output rows from manifest"),
    ]
    if primary_key_unique not in (None, ""):
        checks.append(("primary_key_unique", "pass" if bool(primary_key_unique) else "fail", str(primary_key_unique).lower(), "true", "primary key uniqueness from manifest"))
    return [
        {
            "dq_result_id": f"dq:{stable_hash([run_id, table_name, name, manifest_key], length=24)}",
            "run_id": run_id,
            "table_name": table_name,
            "check_name": name,
            "status": check_status,
            "metric_value": metric_value,
            "threshold": threshold,
            "message": message,
            "created_at_utc": created_at_utc,
        }
        for name, check_status, metric_value, threshold, message in checks
    ]


def _run_id_from_key(key: str) -> str:
    leaf = key.rsplit("/", 1)[-1]
    return leaf.removeprefix("run_id=").removesuffix(".json")


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except Exception:
        return False


def _nonblank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna("").astype(str).str.strip() != ""


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key[0]
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        pk_non_null = pk_unique = run_ok = table_ok = check_ok = status_ok = False
    else:
        pk_non_null = bool(_nonblank(df[pk]).all())
        pk_unique = bool(not df[pk].duplicated().any())
        run_ok = bool(_nonblank(df.get("run_id")).all())
        table_ok = bool(_nonblank(df.get("table_name")).all())
        check_ok = bool(_nonblank(df.get("check_name")).all())
        status_ok = bool(_nonblank(df.get("status")).all())
    status = "pass" if all([row_count > 0, not missing_columns, pk_non_null, pk_unique, run_ok, table_ok, check_ok, status_ok]) else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": pk_unique,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if pk_non_null else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "run_id_populated", "status": "pass" if run_ok else "fail"},
            {"check_name": "table_name_populated", "status": "pass" if table_ok else "fail"},
            {"check_name": "check_name_populated", "status": "pass" if check_ok else "fail"},
            {"check_name": "status_populated", "status": "pass" if status_ok else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
