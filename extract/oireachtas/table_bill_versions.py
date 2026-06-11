"""Builder for the `silver_bill_versions` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlparse

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import normalize_format_url, parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema

TABLE_NAME = "silver_bill_versions"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_bill_versions(
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
    """Fetch `/legislation`, normalize one row per `bill.versions[].version`, and write outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/legislation"
    date_start = date_start or "2025-01-01"
    date_end = date_end or "2025-01-31"
    params: dict[str, Any] = {
        "chamber": chamber or "dail",
        "house_no": house_no or "34",
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
        raise RuntimeError(f"Unexpected /legislation results type: {type(results).__name__}")

    rows: list[dict[str, Any]] = []
    raw_version_samples: list[dict[str, Any]] = []
    bills_with_versions = 0
    skipped_missing_version_wrappers = 0
    for item in results:
        if not isinstance(item, Mapping):
            continue
        bill = item.get("bill") if isinstance(item.get("bill"), Mapping) else item
        bill_id = _first_text(bill, "uri", "billUri") or _first_text(bill, "billId", "id") or f"generated:bill:{stable_hash([stable_record_hash(item)], length=24)}"
        versions = bill.get("versions") or []
        if not isinstance(versions, list):
            versions = []
        if versions:
            bills_with_versions += 1
        for version_index, version_item in enumerate(versions):
            if not isinstance(version_item, Mapping):
                skipped_missing_version_wrappers += 1
                continue
            version = version_item.get("version") if isinstance(version_item.get("version"), Mapping) else version_item
            if not isinstance(version, Mapping):
                skipped_missing_version_wrappers += 1
                continue
            if len(raw_version_samples) < 10:
                raw_version_samples.append(dict(version))
            rows.append(_normalise_version_row(bill_id=bill_id, version=version, version_index=version_index, snapshot_date=snapshot_date))

    rows = _dedupe_rows(rows, primary_key="bill_version_id")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/legislation/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
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
    schema_payload = {"table": TABLE_NAME, "primary_key": schema.primary_key, "columns": schema.columns, "row_count": int(len(df))}
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
    first_result = results[0] if results and isinstance(results[0], Mapping) else {}
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
        "bills_with_versions": bills_with_versions,
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "format_rows_pdf": int(df["source_file_id_pdf"].notna().sum()) if not df.empty else 0,
        "format_rows_xml": int(df["source_file_id_xml"].notna().sum()) if not df.empty else 0,
        "skipped_missing_version_wrappers": skipped_missing_version_wrappers,
        "raw_result_key_paths": sorted(_key_paths(first_result, max_depth=8)),
        "raw_version_key_paths": sorted(_key_paths(raw_version_samples[0], max_depth=6)) if raw_version_samples else [],
        "raw_version_samples": raw_version_samples,
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
    return TableBuildResult(table=TABLE_NAME, rows=sample_df.to_dict(orient="records"), manifest=manifest, schema=schema_payload, dq=dq, s3_keys=s3_keys)


def _normalise_version_row(*, bill_id: str, version: Mapping[str, Any], version_index: int, snapshot_date: str) -> dict[str, Any]:
    version_uri = _first_text(version, "uri", "versionUri")
    version_label = _first_text(version, "showAs", "label", "title")
    version_date = parse_iso_date(_first_text(version, "date", "datetime"))
    doc_type = _first_text(version, "docType")
    lang = _first_text(version, "lang")
    version_identity = version_uri or stable_hash([bill_id, version_index, version_label, version_date, doc_type, lang], length=24)
    bill_version_id = f"bill_version:{stable_hash([bill_id, version_identity], length=24)}"

    formats = version.get("formats") if isinstance(version.get("formats"), Mapping) else {}
    pdf_uri = _format_uri(formats, "pdf")
    xml_uri = _format_uri(formats, "xml")
    pdf_url = normalize_format_url(pdf_uri) if pdf_uri else None
    xml_url = normalize_format_url(xml_uri) if xml_uri else None

    source_file_id_pdf = _source_file_id(bill_id=bill_id, format_type="pdf", format_uri=pdf_uri, format_url=pdf_url) if pdf_uri or pdf_url else None
    source_file_id_xml = _source_file_id(bill_id=bill_id, format_type="xml", format_uri=xml_uri, format_url=xml_url) if xml_uri or xml_url else None
    return {
        "bill_version_id": bill_version_id,
        "bill_id": bill_id,
        "version_label": version_label,
        "version_date": version_date,
        "format_pdf_uri": pdf_uri,
        "format_pdf_url": pdf_url,
        "format_xml_uri": xml_uri,
        "format_xml_url": xml_url,
        "source_file_id_pdf": source_file_id_pdf,
        "source_file_id_xml": source_file_id_xml,
        "s3_pdf_key": _target_s3_key(source_entity_type="legislation", source_entity_id=bill_id, source_file_id=source_file_id_pdf, format_type="pdf", format_url=pdf_url or pdf_uri) if source_file_id_pdf else None,
        "s3_xml_key": _target_s3_key(source_entity_type="legislation", source_entity_id=bill_id, source_file_id=source_file_id_xml, format_type="xml", format_url=xml_url or xml_uri) if source_file_id_xml else None,
        "snapshot_date": snapshot_date,
    }


def _format_uri(formats: Mapping[str, Any], key: str) -> str | None:
    value = formats.get(key)
    if isinstance(value, Mapping):
        return _first_text(value, "uri", "url", "href", "downloadUrl")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _source_file_id(*, bill_id: str, format_type: str, format_uri: str | None, format_url: str | None) -> str:
    return f"source_file:{stable_hash(['legislation', bill_id, format_type, format_uri, format_url], length=24)}"


def _target_s3_key(*, source_entity_type: str, source_entity_id: str, source_file_id: str | None, format_type: str | None, format_url: str | None) -> str | None:
    if not source_file_id:
        return None
    ext = _extension(format_type, format_url)
    entity_slug = _safe_slug(source_entity_id)[-120:] or "entity"
    safe_id = _safe_slug(source_file_id)
    return str(PurePosixPath("raw/oireachtas_unified/source_files", source_entity_type, entity_slug, f"{safe_id}.{ext}"))


def _safe_slug(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip().replace("https://", "").replace("http://", "")
    safe = []
    for char in text:
        safe.append(char if char.isalnum() or char in {"-", "_", "."} else "-")
    return "".join(safe).strip("-")


def _extension(format_type: str | None, url: str | None) -> str:
    inferred = _infer_format_type(url)
    fmt = (format_type or inferred or "bin").lower().strip(".")
    if fmt in {"xml", "pdf", "json", "html", "txt", "csv"}:
        return fmt
    if inferred:
        return inferred
    return "bin"


def _infer_format_type(url: str | None) -> str | None:
    if not url:
        return None
    path = urlparse(str(url)).path.lower()
    for ext in ("pdf", "xml", "json", "html", "txt", "csv"):
        if path.endswith(f".{ext}"):
            return ext
    return None


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, (dict, list)):
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
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        non_null_pk = unique_pk = bill_id_ok = label_ok = date_ok = format_ok = source_id_ok = s3_key_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        bill_id_ok = bool(df["bill_id"].notna().all() and (df["bill_id"].astype(str).str.strip() != "").all())
        label_ok = bool(df["version_label"].notna().all() and (df["version_label"].astype(str).str.strip() != "").all())
        date_ok = bool(df["version_date"].notna().all() and (df["version_date"].astype(str).str.strip() != "").all())
        format_ok = bool(((df["format_pdf_uri"].notna()) | (df["format_xml_uri"].notna()) | (df["format_pdf_url"].notna()) | (df["format_xml_url"].notna())).all())
        source_id_ok = bool(((df["source_file_id_pdf"].notna()) | (df["source_file_id_xml"].notna())).all())
        s3_key_ok = bool(((df["s3_pdf_key"].notna()) | (df["s3_xml_key"].notna())).all())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, bill_id_ok, label_ok, date_ok, format_ok, source_id_ok, s3_key_ok]) else "fail"
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
            {"check_name": "bill_id_populated", "status": "pass" if bill_id_ok else "fail"},
            {"check_name": "version_label_populated", "status": "pass" if label_ok else "fail"},
            {"check_name": "version_date_populated", "status": "pass" if date_ok else "fail"},
            {"check_name": "at_least_one_source_format", "status": "pass" if format_ok else "fail"},
            {"check_name": "source_file_ids_populated_when_format_present", "status": "pass" if source_id_ok else "fail"},
            {"check_name": "source_s3_keys_populated_when_format_present", "status": "pass" if s3_key_ok else "fail"},
        ],
    }


def _key_paths(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 8) -> set[str]:
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
