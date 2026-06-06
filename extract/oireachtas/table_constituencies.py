"""Builder for the `silver_constituencies` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import is_current_range, normalize_name, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_constituencies"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_constituencies(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
) -> TableBuildResult:
    """Fetch `/constituencies`, normalize, and write silver_constituencies outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/constituencies"
    params = {"limit": max(1, min(limit, 200))}

    summary = client.get_json_summary(endpoint, params=params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch {endpoint}: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected /constituencies results type: {type(results).__name__}")

    rows: list[dict[str, Any]] = []
    for item in results:
        rows.extend(_normalise_constituency_record(record, house, snapshot_date=snapshot_date, endpoint=endpoint) for record, house in _iter_constituency_records(item))

    rows = _dedupe_rows(rows, primary_key="constituency_uri")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/constituencies/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
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
        rows=sample_df.to_dict(orient="records"),
        manifest=manifest,
        schema=schema_payload,
        dq=dq,
        s3_keys=manifest["s3_keys"],
    )


def _iter_constituency_records(item: Any) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Yield constituency records with the best available parent house context.

    The `/constituencies` endpoint returns wrappers that can contain a house and
    nested constituency arrays. This method supports direct `constituency`
    wrappers, `constituencies` arrays under either result or house, and a small
    recursive fallback for schema drift.
    """
    if not isinstance(item, Mapping):
        return []

    emitted: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    root_house = _extract_house({}, item)

    direct = item.get("constituency")
    if isinstance(direct, Mapping):
        emitted.append((direct, _extract_house(direct, item) or root_house))

    for parent in (item, root_house):
        if not isinstance(parent, Mapping):
            continue
        nested = parent.get("constituencies")
        if isinstance(nested, list):
            for entry in nested:
                if isinstance(entry, Mapping):
                    record = entry.get("constituency") if isinstance(entry.get("constituency"), Mapping) else entry
                    house = _extract_house(record, entry) or root_house
                    emitted.append((record, house))

    if emitted:
        return emitted

    # Last-resort recursive fallback: locate mappings under keys named
    # `constituencies` anywhere in the wrapper.
    return list(_recursive_constituencies(item, root_house))


def _recursive_constituencies(value: Any, house_context: Mapping[str, Any]) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        return
    local_house = _extract_house({}, value) or house_context
    for key, child in value.items():
        if key == "constituencies" and isinstance(child, list):
            for entry in child:
                if isinstance(entry, Mapping):
                    record = entry.get("constituency") if isinstance(entry.get("constituency"), Mapping) else entry
                    house = _extract_house(record, entry) or local_house
                    yield record, house
        elif isinstance(child, Mapping):
            yield from _recursive_constituencies(child, local_house)


def _normalise_constituency_record(record: Mapping[str, Any], house: Mapping[str, Any], *, snapshot_date: str, endpoint: str) -> dict[str, Any]:
    date_range = dict(record.get("dateRange") or record.get("date_range") or {})
    start = parse_iso_date(date_range.get("start") or record.get("dateStart") or record.get("startDate"))
    end = parse_iso_date(date_range.get("end") or record.get("dateEnd") or record.get("endDate"))
    show_as = _first_text(record, "showAs", "show_as", "name", "constituencyName")
    name = _first_text(record, "name", "constituencyName", "showAs", "show_as")
    code = _first_text(record, "constituencyCode", "code", "id")
    uri = _first_text(record, "uri", "constituencyUri")
    house_uri = _first_text(house, "uri", "houseUri")
    house_no = _first_text(house, "houseNo", "house_no")
    chamber = _first_text(house, "houseCode", "chamberCode", "chamber", "houseType")

    if not uri:
        uri = f"generated:constituency:{stable_hash([name, show_as, house_uri, house_no, chamber, start, end])}"

    return {
        "constituency_uri": uri,
        "constituency_code": code,
        "constituency_name": name or normalize_name(show_as),
        "show_as": show_as or name,
        "house_uri": house_uri,
        "house_no": house_no,
        "chamber": chamber,
        "date_start": start,
        "date_end": end,
        "is_current": is_current_range(start, end),
        "source_endpoint": endpoint,
        "snapshot_date": snapshot_date,
        "source_hash": stable_record_hash(record),
    }


def _extract_house(record: Mapping[str, Any], item: Mapping[str, Any]) -> Mapping[str, Any]:
    for candidate in (
        record.get("house") if isinstance(record, Mapping) else None,
        record.get("houseRecord") if isinstance(record, Mapping) else None,
        record.get("houseInfo") if isinstance(record, Mapping) else None,
        item.get("house") if isinstance(item, Mapping) else None,
        item.get("houseRecord") if isinstance(item, Mapping) else None,
        item.get("houseInfo") if isinstance(item, Mapping) else None,
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


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
    if row_count == 0 or pk not in df.columns:
        non_null_pk = False
        unique_pk = False
        name_populated = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        name_populated = bool(df["constituency_name"].notna().any() and (df["constituency_name"].astype(str).str.strip() != "").any())
    status = "pass" if row_count > 0 and not missing_columns and non_null_pk and unique_pk and name_populated else "fail"

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
            {"check_name": "constituency_name_populated", "status": "pass" if name_populated else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{table}_{stamp}"
