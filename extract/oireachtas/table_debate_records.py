"""Builder for the `silver_debate_records` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_debate_records"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_debate_records(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
    chamber: str | None = None,
    house_no: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> TableBuildResult:
    """Fetch `/debates`, normalize debate-level metadata, and write outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/debates"
    date_start = date_start or "2025-01-01"
    date_end = date_end or "2025-01-31"
    params: dict[str, Any] = {
        "chamber_id": f"/ie/oireachtas/house/{chamber or 'dail'}/{house_no or '34'}",
        "lang": "en",
        "date_start": date_start,
        "date_end": date_end,
        "limit": max(1, min(limit, 200)),
    }

    summary = client.get_json_summary(endpoint, params=params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch {endpoint}: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected /debates results type: {type(results).__name__}")

    rows = [
        _normalise_debate_row(item, snapshot_date=snapshot_date)
        for item in results
        if isinstance(item, Mapping)
    ]
    rows = _dedupe_rows(rows, primary_key="debate_id")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/debates/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
    csv_key = f"processed/oireachtas_unified/silver_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/silver/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
    latest_csv_key = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
    latest_parquet_key = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
    manifest_key = f"processed/oireachtas_unified/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"

    dq = _dq_results(df, schema)
    write_errors: list[str] = []
    schema_payload = {
        "table": TABLE_NAME,
        "primary_key": schema.primary_key,
        "columns": schema.columns,
        "row_count": int(len(df)),
    }
    s3_keys = {
        "raw_json": raw_key,
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
        "endpoint": endpoint,
        "params": dict(summary.params),
        "url": summary.url,
        "status_code": summary.status_code,
        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "source_file_id_method": "T09 stable hash of entity type, debate URI, format type, format URI, format URL",
        "raw_result_sample": results[:2],
        "raw_result_key_paths": sorted(_key_paths(results[0], max_depth=7)) if results and isinstance(results[0], Mapping) else [],
        "write_errors": write_errors,
        "s3_keys": s3_keys,
    }

    try:
        put_json(s3, bucket=bucket, key=raw_key, payload=payload)
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
    return TableBuildResult(
        table=TABLE_NAME,
        rows=sample_df.to_dict(orient="records"),
        manifest=manifest,
        schema=schema_payload,
        dq=dq,
        s3_keys=s3_keys,
    )


def _normalise_debate_row(item: Mapping[str, Any], *, snapshot_date: str) -> dict[str, Any]:
    record = item.get("debateRecord")
    if not isinstance(record, Mapping):
        record = item

    debate_uri = _first_text(record, "uri", "debateUri")
    debate_id = debate_uri or f"generated:debate:{stable_hash(record, length=20)}"
    context_date = parse_iso_date(item.get("contextDate"))
    debate_date = parse_iso_date(record.get("date")) or context_date
    chamber_record = _first_mapping(record, "chamber")
    house_record = _first_mapping(record, "house")
    house_uri = _first_text(house_record, "uri") or _first_text(chamber_record, "uri")
    house_no = _first_text(house_record, "houseNo")
    house_code = _first_text(house_record, "houseCode", "chamberCode")
    chamber = house_code or _first_text(chamber_record, "showAs") or _first_text(house_record, "showAs")
    show_as = _first_text(record, "showAs", "title") or _build_show_as(chamber, debate_date)

    formats = _first_mapping(record, "formats")
    pdf = _format_mapping(formats, "pdf")
    xml = _format_mapping(formats, "xml")
    source_pdf_uri = _first_text(pdf, "uri", "url", "href")
    source_xml_uri = _first_text(xml, "uri", "url", "href")
    source_pdf_url = source_pdf_uri if source_pdf_uri and source_pdf_uri.startswith("http") else _first_text(pdf, "url", "href")
    source_xml_url = source_xml_uri if source_xml_uri and source_xml_uri.startswith("http") else _first_text(xml, "url", "href")

    return {
        "debate_id": debate_id,
        "debate_uri": debate_uri,
        "context_date": context_date,
        "debate_date": debate_date,
        "chamber": chamber,
        "house_uri": house_uri,
        "house_no": house_no,
        "house_code": house_code,
        "show_as": show_as,
        "source_xml_uri": source_xml_uri,
        "source_xml_url": source_xml_url,
        "source_pdf_uri": source_pdf_uri,
        "source_pdf_url": source_pdf_url,
        "source_file_id_xml": _source_file_id(debate_id, "xml", source_xml_uri, source_xml_url),
        "source_file_id_pdf": _source_file_id(debate_id, "pdf", source_pdf_uri, source_pdf_url),
        "api_result_hash": stable_record_hash(item),
        "snapshot_date": snapshot_date,
    }


def _source_file_id(entity_id: str, format_type: str, format_uri: str | None, format_url: str | None) -> str | None:
    if not (format_uri or format_url):
        return None
    return f"source_file:{stable_hash(['debate', entity_id, format_type, format_uri, format_url], length=24)}"


def _format_mapping(formats: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = formats.get(key)
    return value if isinstance(value, Mapping) else {}


def _first_mapping(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _build_show_as(chamber: str | None, debate_date: str | None) -> str | None:
    parts = [part for part in (chamber, debate_date) if part]
    return " — ".join(parts) if parts else None


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
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        non_null_pk = unique_pk = uri_ok = date_ok = house_ok = xml_ok = pdf_ok = file_ids_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        uri_ok = bool(df["debate_uri"].notna().all() and (df["debate_uri"].astype(str).str.strip() != "").all())
        date_ok = bool(df["debate_date"].notna().all() and (df["debate_date"].astype(str).str.strip() != "").all())
        house_ok = bool(df["house_uri"].notna().all() and (df["house_uri"].astype(str).str.strip() != "").all())
        xml_ok = bool(df["source_xml_uri"].notna().all() and (df["source_xml_uri"].astype(str).str.strip() != "").all())
        pdf_ok = bool(df["source_pdf_uri"].notna().all() and (df["source_pdf_uri"].astype(str).str.strip() != "").all())
        file_ids_ok = bool(
            df["source_file_id_xml"].notna().all()
            and df["source_file_id_pdf"].notna().all()
            and (df["source_file_id_xml"].astype(str).str.strip() != "").all()
            and (df["source_file_id_pdf"].astype(str).str.strip() != "").all()
        )

    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, uri_ok, date_ok, house_ok, xml_ok, pdf_ok, file_ids_ok]) else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail"},
            {"check_name": "debate_uri_populated", "status": "pass" if uri_ok else "fail"},
            {"check_name": "debate_date_populated", "status": "pass" if date_ok else "fail"},
            {"check_name": "house_uri_populated", "status": "pass" if house_ok else "fail"},
            {"check_name": "source_xml_uri_populated", "status": "pass" if xml_ok else "fail"},
            {"check_name": "source_pdf_uri_populated", "status": "pass" if pdf_ok else "fail"},
            {"check_name": "source_file_ids_populated", "status": "pass" if file_ids_ok else "fail"},
        ],
    }


def _key_paths(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 7) -> set[str]:
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
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
