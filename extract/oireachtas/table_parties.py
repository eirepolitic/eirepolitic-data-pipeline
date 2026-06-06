"""Builder for the `silver_parties` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import is_current_range, normalize_name, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_parties"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_parties(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
) -> TableBuildResult:
    """Fetch `/parties`, normalize, and write silver_parties outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/parties"
    params = {"limit": max(1, min(limit, 200))}

    summary = client.get_json_summary(endpoint, params=params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch {endpoint}: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected /parties results type: {type(results).__name__}")

    rows: list[dict[str, Any]] = []
    for item in results:
        rows.extend(_normalise_party_record(record, snapshot_date=snapshot_date, endpoint=endpoint) for record in _iter_party_records(item))

    rows = _dedupe_rows(rows, primary_key="party_uri")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/parties/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
    csv_key = f"processed/oireachtas_unified/silver_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/silver/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    latest_parquet_key = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
    manifest_key = f"processed/oireachtas_unified/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"

    dq = _dq_results(df, schema)
    schema_payload = {"table": TABLE_NAME, "primary_key": schema.primary_key, "columns": schema.columns, "row_count": int(len(df))}
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
        "raw_result_sample": results[:2],
        "raw_result_key_paths": sorted(_key_paths(results[0], max_depth=5)) if results and isinstance(results[0], Mapping) else [],
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


def _iter_party_records(item: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(item, Mapping):
        return []

    emitted: list[Mapping[str, Any]] = []
    for key in ("party", "partyDetails", "organisation", "organization"):
        value = item.get(key)
        if isinstance(value, Mapping):
            emitted.append(value)

    for nested_key in ("parties", "partyList"):
        nested = item.get(nested_key)
        if isinstance(nested, list):
            for entry in nested:
                if isinstance(entry, Mapping):
                    emitted.append(_unwrap_party(entry))

    if emitted:
        return emitted

    # If the result itself has party-like fields, use it directly.
    if any(key in item for key in ("partyCode", "showAs", "uri", "name")):
        return [item]

    return list(_recursive_parties(item))


def _recursive_parties(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if key in {"party", "partyDetails", "organisation", "organization"} and isinstance(child, Mapping):
            yield child
        elif key in {"parties", "partyList"} and isinstance(child, list):
            for entry in child:
                if isinstance(entry, Mapping):
                    yield _unwrap_party(entry)
        elif isinstance(child, Mapping):
            yield from _recursive_parties(child)


def _unwrap_party(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("party", "partyDetails", "organisation", "organization"):
        value = entry.get(key)
        if isinstance(value, Mapping):
            return value
    return entry


def _normalise_party_record(record: Mapping[str, Any], *, snapshot_date: str, endpoint: str) -> dict[str, Any]:
    date_range = dict(record.get("dateRange") or record.get("date_range") or {})
    start = parse_iso_date(date_range.get("start") or record.get("dateStart") or record.get("startDate"))
    end = parse_iso_date(date_range.get("end") or record.get("dateEnd") or record.get("endDate"))
    show_as = _first_text(record, "showAs", "show_as", "name", "partyName", "label")
    name = _first_text(record, "partyName", "name", "showAs", "show_as", "label")
    code = _first_text(record, "partyCode", "code", "id", "representCode")
    uri = _first_text(record, "uri", "partyUri")

    if not uri:
        uri = f"generated:party:{stable_hash([code, name, show_as, start, end])}"

    return {
        "party_uri": uri,
        "party_code": code,
        "party_name": name or normalize_name(show_as),
        "show_as": show_as or name,
        "date_start": start,
        "date_end": end,
        "is_current": is_current_range(start, end),
        "source_endpoint": endpoint,
        "snapshot_date": snapshot_date,
        "source_hash": stable_record_hash(record),
    }


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
        name_populated = bool(df["party_name"].notna().any() and (df["party_name"].astype(str).str.strip() != "").any())
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
            {"check_name": "party_name_populated", "status": "pass" if name_populated else "fail"},
        ],
    }


def _key_paths(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 5) -> set[str]:
    if depth >= max_depth:
        return set()
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(child_path)
            paths.update(_key_paths(child, prefix=child_path, depth=depth + 1, max_depth=max_depth))
    elif isinstance(value, list):
        list_path = f"{prefix}[]" if prefix else "[]"
        paths.add(list_path)
        if value:
            paths.update(_key_paths(value[0], prefix=list_path, depth=depth + 1, max_depth=max_depth))
    return paths


def _run_id(table: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{table}_{stamp}"
