"""Builder for the `control_pipeline_runs` table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .client import OireachtasClient
from .io_s3 import get_json, put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import utc_now_iso
from .schemas import DEFAULT_TABLES_CONFIG, TableSchema, load_table_registry

TABLE_NAME = "control_pipeline_runs"
MANIFEST_PREFIX = "processed/oireachtas_unified/manifests/"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_control_pipeline_runs(
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
    """Build run-level audit rows from table manifest JSON objects in S3."""
    del client, chamber, house_no
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]

    cadence_by_table = _cadence_by_table()
    manifest_keys = _list_manifest_keys(s3, bucket=bucket, prefix=MANIFEST_PREFIX)
    rows = []
    read_errors: list[str] = []
    for key in manifest_keys:
        try:
            payload = get_json(s3, bucket=bucket, key=key)
            rows.append(_manifest_to_row(payload, key=key, cadence_by_table=cadence_by_table))
        except Exception as exc:
            read_errors.append(f"{key}: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows, columns=schema.columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["run_id"], keep="last")
        df = df.sort_values(["started_at_utc", "table_name", "run_id"], ascending=[False, True, True]).head(max(1, limit)).copy()

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
        "manifest_objects_read": len(rows),
        "read_error_count": len(read_errors),
        "read_errors": read_errors[:25],
        "output_rows": int(len(df)),
        "table_name_values": sorted(df["table_name"].dropna().astype(str).unique().tolist()) if not df.empty else [],
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


def _cadence_by_table() -> dict[str, str]:
    try:
        registry = load_table_registry(Path(DEFAULT_TABLES_CONFIG))
        return {name: table.cadence for name, table in registry.items()}
    except Exception:
        return {}


def _manifest_to_row(payload: dict[str, Any], *, key: str, cadence_by_table: dict[str, str]) -> dict[str, str]:
    table_name = str(payload.get("table") or "")
    row = {
        "run_id": str(payload.get("run_id") or _run_id_from_key(key)),
        "workflow_run_id": str(payload.get("workflow_run_id") or ""),
        "table_name": table_name,
        "mode": str(payload.get("mode") or ""),
        "cadence": str(payload.get("cadence") or cadence_by_table.get(table_name, "")),
        "started_at_utc": str(payload.get("started_at_utc") or payload.get("created_at_utc") or ""),
        "finished_at_utc": str(payload.get("finished_at_utc") or payload.get("verified_at_utc") or ""),
        "status": str(payload.get("status") or ""),
        "input_params_json": _input_params_json(payload),
        "raw_rows": str(_first_present(payload, ["raw_rows", "raw_member_rows", "raw_api_rows", "raw_legislation_rows", "raw_speech_rows"], default="")),
        "output_rows": str(payload.get("output_rows") if payload.get("output_rows") is not None else payload.get("row_count", "")),
        "error_message": _error_message(payload),
        "manifest_s3_key": key,
    }
    return row


def _run_id_from_key(key: str) -> str:
    leaf = key.rsplit("/", 1)[-1]
    return leaf.removeprefix("run_id=").removesuffix(".json")


def _first_present(payload: dict[str, Any], names: list[str], *, default: Any) -> Any:
    for name in names:
        if payload.get(name) is not None:
            return payload.get(name)
    input_rows = payload.get("input_rows")
    if isinstance(input_rows, dict) and input_rows:
        return sum(int(value or 0) for value in input_rows.values() if str(value or "").isdigit())
    return default


def _input_params_json(payload: dict[str, Any]) -> str:
    params = {
        "endpoint_url": payload.get("endpoint_url"),
        "input_keys": payload.get("input_keys"),
        "input_prefix": payload.get("input_prefix"),
        "snapshot_date": payload.get("snapshot_date"),
    }
    compact = {key: value for key, value in params.items() if value not in (None, "", {}, [])}
    return json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _error_message(payload: dict[str, Any]) -> str:
    if payload.get("error_message"):
        return str(payload.get("error_message"))
    errors = payload.get("write_errors") or payload.get("read_errors") or []
    if isinstance(errors, list):
        return "; ".join(str(item) for item in errors[:3])
    return str(errors or "")


def _nonblank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna("").astype(str).str.strip() != ""


def _numeric_or_blank(series: pd.Series | None) -> bool:
    if series is None:
        return False
    values = series.fillna("").astype(str).str.strip()
    nonblank = values[values != ""]
    if nonblank.empty:
        return True
    return bool(pd.to_numeric(nonblank, errors="coerce").notna().all())


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key[0]
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        pk_non_null = pk_unique = table_ok = status_ok = rows_numeric = False
    else:
        pk_non_null = bool(_nonblank(df[pk]).all())
        pk_unique = bool(not df[pk].duplicated().any())
        table_ok = bool(_nonblank(df.get("table_name")).all())
        status_ok = bool(_nonblank(df.get("status")).all())
        rows_numeric = _numeric_or_blank(df.get("raw_rows")) and _numeric_or_blank(df.get("output_rows"))
    status = "pass" if all([row_count > 0, not missing_columns, pk_non_null, pk_unique, table_ok, status_ok, rows_numeric]) else "fail"
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
            {"check_name": "table_name_populated", "status": "pass" if table_ok else "fail"},
            {"check_name": "status_populated", "status": "pass" if status_ok else "fail"},
            {"check_name": "row_count_fields_numeric_or_blank", "status": "pass" if rows_numeric else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
