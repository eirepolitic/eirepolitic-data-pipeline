"""Builder for the `control_table_manifests` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .client import OireachtasClient
from .io_s3 import get_json, put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import stable_hash, utc_now_iso
from .schemas import DEFAULT_TABLES_CONFIG, TableSchema, load_table_registry

TABLE_NAME = "control_table_manifests"
MANIFEST_PREFIX = "processed/oireachtas_unified/manifests/"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_control_table_manifests(
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
    """Build one latest-manifest pointer row per table from S3 manifest objects."""
    del client, chamber, house_no
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]

    registry = _registry_by_table()
    manifest_keys = _list_manifest_keys(s3, bucket=bucket, prefix=MANIFEST_PREFIX)
    payloads: list[tuple[str, dict[str, Any]]] = []
    read_errors: list[str] = []
    for key in manifest_keys:
        try:
            payloads.append((key, get_json(s3, bucket=bucket, key=key)))
        except Exception as exc:
            read_errors.append(f"{key}: {type(exc).__name__}: {exc}")

    latest_by_table = _latest_by_table(payloads)
    rows = [_manifest_to_row(payload, key=key, registry=registry, updated_at_utc=started_at) for key, payload in latest_by_table.values()]
    df = pd.DataFrame(rows, columns=schema.columns)
    if not df.empty:
        df = df.sort_values(["table_name"]).copy()

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
        "manifest_objects_read": len(payloads),
        "read_error_count": len(read_errors),
        "read_errors": read_errors[:25],
        "latest_table_rows_before_limit": len(rows),
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


def _registry_by_table() -> dict[str, TableSchema]:
    try:
        return load_table_registry(Path(DEFAULT_TABLES_CONFIG))
    except Exception:
        return {}


def _latest_by_table(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, tuple[str, dict[str, Any]]]:
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for key, payload in payloads:
        table_name = str(payload.get("table") or "")
        if not table_name:
            continue
        current = latest.get(table_name)
        if current is None or _sort_value(payload, key) > _sort_value(current[1], current[0]):
            latest[table_name] = (key, payload)
    return latest


def _sort_value(payload: dict[str, Any], key: str) -> tuple[str, str]:
    timestamp = str(payload.get("started_at_utc") or payload.get("finished_at_utc") or payload.get("created_at_utc") or "")
    return timestamp, key


def _manifest_to_row(payload: dict[str, Any], *, key: str, registry: dict[str, TableSchema], updated_at_utc: str) -> dict[str, str]:
    table_name = str(payload.get("table") or "")
    table_schema = registry.get(table_name)
    columns = table_schema.columns if table_schema else []
    primary_key = table_schema.primary_key if table_schema else payload.get("primary_key", [])
    s3_keys = payload.get("s3_keys") if isinstance(payload.get("s3_keys"), dict) else {}
    return {
        "table_name": table_name,
        "latest_run_id": str(payload.get("run_id") or _run_id_from_key(key)),
        "latest_snapshot_date": str(payload.get("snapshot_date") or ""),
        "latest_parquet_key": str(s3_keys.get("latest_parquet") or s3_keys.get("parquet") or ""),
        "latest_csv_key": str(s3_keys.get("latest_csv") or s3_keys.get("csv") or ""),
        "row_count": str(payload.get("output_rows") if payload.get("output_rows") is not None else payload.get("row_count", "")),
        "column_count": str(len(columns)) if columns else "",
        "schema_hash": stable_hash([table_name, ",".join(primary_key), ",".join(columns)], length=24),
        "primary_key_unique": str(payload.get("primary_key_unique") if payload.get("primary_key_unique") is not None else "").lower(),
        "dq_status": str(payload.get("dq_status") or ""),
        "updated_at_utc": updated_at_utc,
    }


def _run_id_from_key(key: str) -> str:
    leaf = key.rsplit("/", 1)[-1]
    return leaf.removeprefix("run_id=").removesuffix(".json")


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
        pk_non_null = pk_unique = latest_run_ok = dq_status_ok = counts_numeric = False
    else:
        pk_non_null = bool(_nonblank(df[pk]).all())
        pk_unique = bool(not df[pk].duplicated().any())
        latest_run_ok = bool(_nonblank(df.get("latest_run_id")).all())
        dq_status_ok = bool(_nonblank(df.get("dq_status")).all())
        counts_numeric = _numeric_or_blank(df.get("row_count")) and _numeric_or_blank(df.get("column_count"))
    status = "pass" if all([row_count > 0, not missing_columns, pk_non_null, pk_unique, latest_run_ok, dq_status_ok, counts_numeric]) else "fail"
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
            {"check_name": "latest_run_id_populated", "status": "pass" if latest_run_ok else "fail"},
            {"check_name": "dq_status_populated", "status": "pass" if dq_status_ok else "fail"},
            {"check_name": "row_column_counts_numeric_or_blank", "status": "pass" if counts_numeric else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
