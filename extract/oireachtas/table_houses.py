"""Builder for the `silver_houses` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import is_current_range, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_houses"


@dataclass(frozen=True)
class TableBuildResult:
    """Result metadata returned by table builders."""

    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_houses(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
) -> TableBuildResult:
    """Fetch `/houses`, normalize, and write silver_houses outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/houses"
    params = {"limit": max(1, min(limit, 200))}

    summary = client.get_json_summary(endpoint, params=params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch {endpoint}: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected /houses results type: {type(results).__name__}")

    rows = [_normalise_house_row(item, snapshot_date=snapshot_date, endpoint=endpoint) for item in results]
    rows = _dedupe_rows(rows, primary_key="house_uri")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/houses/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
    csv_key = f"processed/oireachtas_unified/silver_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/silver/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    latest_parquet_key = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
    manifest_key = f"processed/oireachtas_unified/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"

    dq = _dq_results(df, schema)
    schema_payload = {
        "table": TABLE_NAME,
        "primary_key": schema.primary_key,
        "columns": schema.columns,
        "row_count": int(len(df)),
    }
    manifest = {
        "table": TABLE_NAME,
        "mode": mode,
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "endpoint": endpoint,
        "params": dict(summary.params),
        "url": summary.url,
        "status_code": summary.status_code,
        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "s3_keys": {
            "raw_json": raw_key,
            "csv": csv_key,
            "parquet": parquet_key,
            "latest_csv": latest_csv_key,
            "latest_parquet": latest_parquet_key,
            "manifest": manifest_key,
            "review_sample": review_sample_key,
            "review_schema": review_schema_key,
            "review_manifest": review_manifest_key,
        },
    }

    put_json(s3, bucket=bucket, key=raw_key, payload=payload)
    put_dataframe_csv(s3, bucket=bucket, key=csv_key, df=df)
    put_dataframe_parquet(s3, bucket=bucket, key=parquet_key, df=df)
    put_dataframe_csv(s3, bucket=bucket, key=latest_csv_key, df=df)
    put_dataframe_parquet(s3, bucket=bucket, key=latest_parquet_key, df=df)
    put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)

    sample_df = df.head(10)
    put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=sample_df)
    put_json(s3, bucket=bucket, key=review_schema_key, payload=schema_payload)
    put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)

    return TableBuildResult(
        table=TABLE_NAME,
        rows=df.head(10).to_dict(orient="records"),
        manifest=manifest,
        schema=schema_payload,
        dq=dq,
        s3_keys=manifest["s3_keys"],
    )


def _normalise_house_row(item: Mapping[str, Any], *, snapshot_date: str, endpoint: str) -> dict[str, Any]:
    house = dict((item or {}).get("house") or {})
    date_range = dict(house.get("dateRange") or {})
    start = parse_iso_date(date_range.get("start"))
    end = parse_iso_date(date_range.get("end"))
    house_uri = house.get("uri") or f"generated:house:{stable_hash([house.get('houseCode'), house.get('houseNo'), house.get('showAs')])}"

    return {
        "house_uri": str(house_uri),
        "house_no": _text_or_none(house.get("houseNo")),
        "house_code": _text_or_none(house.get("houseCode") or house.get("chamberCode")),
        "chamber": _text_or_none(house.get("chamberType") or house.get("chamberCode") or house.get("houseType")),
        "show_as": _text_or_none(house.get("showAs")),
        "date_start": start,
        "date_end": end,
        "is_current": is_current_range(start, end),
        "source_endpoint": endpoint,
        "snapshot_date": snapshot_date,
        "source_hash": stable_record_hash(item),
    }


def _dedupe_rows(rows: list[dict[str, Any]], *, primary_key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get(primary_key) or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key[0]
    required_columns = set(schema.columns)
    missing_columns = sorted(required_columns - set(df.columns))
    row_count = int(len(df))
    non_null_pk = bool(row_count and df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
    unique_pk = bool(row_count and not df[pk].duplicated().any())
    status = "pass" if row_count > 0 and not missing_columns and non_null_pk and unique_pk else "fail"

    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count, "threshold": 1},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail", "primary_key": pk},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail", "primary_key": pk},
        ],
    }


def _run_id(table: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{table}_{stamp}"


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
