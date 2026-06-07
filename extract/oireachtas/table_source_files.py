"""Builder for the `silver_source_files` table.

This table is a metadata inventory of source XML/PDF/document references exposed by
Oireachtas API payload `formats` fields. It intentionally does not download files
in this packet; it creates stable file IDs and target S3 keys for later download
workers and downstream table joins.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import stable_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_source_files"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_source_files(
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
    """Discover format/file references from debates, questions, and legislation."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    date_start = date_start or "2025-01-01"
    date_end = date_end or "2025-01-31"
    safe_limit = max(1, min(limit, 25))

    endpoint_specs = [
        {
            "entity_type": "debate",
            "endpoint": "/debates",
            "params": {
                "chamber_id": f"/ie/oireachtas/house/{chamber or 'dail'}/{house_no or '34'}",
                "lang": "en",
                "date_start": date_start,
                "date_end": date_end,
                "limit": safe_limit,
            },
        },
        {
            "entity_type": "question",
            "endpoint": "/questions",
            "params": {
                "chamber": chamber or "dail",
                "house_no": house_no or "34",
                "date_start": date_start,
                "date_end": date_end,
                "limit": safe_limit,
            },
        },
        {
            "entity_type": "legislation",
            "endpoint": "/legislation",
            "params": {
                "chamber": chamber or "dail",
                "house_no": house_no or "34",
                "date_start": date_start,
                "date_end": date_end,
                "limit": safe_limit,
            },
        },
    ]

    rows: list[dict[str, Any]] = []
    endpoint_summaries: list[dict[str, Any]] = []
    raw_format_samples: list[dict[str, Any]] = []
    skipped_null_formats = 0
    raw_payloads: dict[str, Any] = {}

    for spec in endpoint_specs:
        summary = client.get_json_summary(spec["endpoint"], params=spec["params"])
        payload = dict(summary.payload or {})
        raw_payloads[str(spec["entity_type"])] = payload
        results = payload.get("results") or []
        if not isinstance(results, list):
            results = []

        endpoint_rows: list[dict[str, Any]] = []
        endpoint_skipped = 0
        for item_index, item in enumerate(results):
            if not isinstance(item, Mapping):
                continue
            source_entity_id = _entity_id(item, entity_type=str(spec["entity_type"]), item_index=item_index)
            for format_record in _iter_format_records(item):
                if len(raw_format_samples) < 10:
                    raw_format_samples.append(dict(format_record.raw))
                if not (format_record.format_uri or format_record.format_url):
                    skipped_null_formats += 1
                    endpoint_skipped += 1
                    continue
                row = _normalise_format_row(
                    format_record=format_record,
                    source_entity_type=str(spec["entity_type"]),
                    source_entity_id=source_entity_id,
                    snapshot_date=snapshot_date,
                )
                endpoint_rows.append(row)

        rows.extend(endpoint_rows)
        endpoint_summaries.append(
            {
                "entity_type": spec["entity_type"],
                "endpoint": spec["endpoint"],
                "params": dict(summary.params),
                "url": summary.url,
                "status_code": summary.status_code,
                "ok": bool(summary.ok),
                "raw_rows": len(results),
                "format_rows": len(endpoint_rows),
                "skipped_null_format_rows": endpoint_skipped,
                "raw_result_key_paths": sorted(_key_paths(results[0], max_depth=7)) if results and isinstance(results[0], Mapping) else [],
                "error": summary.error,
            }
        )

    rows = _dedupe_rows(rows, primary_key="source_file_id")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/source_files/snapshot_date={snapshot_date}/run_id={run_id}/payloads.json"
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
    manifest = {
        "table": TABLE_NAME,
        "mode": mode,
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "date_start": date_start,
        "date_end": date_end,
        "metadata_only": True,
        "download_status": "not_downloaded",
        "skipped_null_format_rows": skipped_null_formats,
        "endpoint_summaries": endpoint_summaries,
        "raw_format_samples": raw_format_samples,
        "raw_format_key_paths": sorted(_key_paths(raw_format_samples[0], max_depth=7)) if raw_format_samples else [],
        "write_errors": write_errors,
        "s3_keys": s3_keys,
    }

    try:
        put_json(s3, bucket=bucket, key=raw_key, payload=raw_payloads)
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


@dataclass(frozen=True)
class FormatRecord:
    format_type: str | None
    format_uri: str | None
    format_url: str | None
    raw: Mapping[str, Any]


def _iter_format_records(value: Any) -> Iterable[FormatRecord]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"formats", "format"}:
                yield from _format_records_from_container(child)
            else:
                yield from _iter_format_records(child)
    elif isinstance(value, list):
        for entry in value:
            yield from _iter_format_records(entry)


def _format_records_from_container(container: Any) -> Iterable[FormatRecord]:
    if isinstance(container, list):
        for entry in container:
            yield from _format_records_from_container(entry)
        return
    if not isinstance(container, Mapping):
        return

    # Shape 1: {"pdf": {"uri": ...}, "xml": {"uri": ...}, "writtens_pdf": null}
    emitted = False
    has_format_like_key = any(_looks_like_format_key(str(key)) for key in container.keys())
    for key, child in container.items():
        if isinstance(child, Mapping) and (_first_text(child, "uri", "url", "href", "downloadUrl") or _looks_like_format_key(str(key))):
            record = _make_format_record(child, fallback_type=str(key))
            if record.format_uri or record.format_url:
                emitted = True
                yield record
        elif isinstance(child, list) and _looks_like_format_key(str(key)):
            for entry in child:
                if isinstance(entry, Mapping):
                    record = _make_format_record(entry, fallback_type=str(key))
                    if record.format_uri or record.format_url:
                        emitted = True
                        yield record
    if emitted or has_format_like_key:
        return

    # Shape 2: {"formatType": "pdf", "uri": ..., "url": ...}
    record = _make_format_record(container, fallback_type=None)
    if record.format_uri or record.format_url:
        yield record


def _make_format_record(raw: Mapping[str, Any], *, fallback_type: str | None) -> FormatRecord:
    format_type = _first_text(raw, "formatType", "type", "mediaType", "name", "label") or fallback_type
    format_uri = _first_text(raw, "uri", "formatUri")
    format_url = _first_text(raw, "url", "href", "downloadUrl", "formatUrl")
    if not format_url and format_uri and format_uri.startswith("http"):
        format_url = format_uri
    if not format_uri and format_url and format_url.startswith("http"):
        format_uri = format_url
    return FormatRecord(format_type=_normalise_format_type(format_type, format_uri or format_url), format_uri=format_uri, format_url=format_url, raw=raw)


def _normalise_format_row(*, format_record: FormatRecord, source_entity_type: str, source_entity_id: str, snapshot_date: str) -> dict[str, Any]:
    format_type = format_record.format_type or _infer_format_type(format_record.format_uri or format_record.format_url) or "unknown"
    source_file_id = f"source_file:{stable_hash([source_entity_type, source_entity_id, format_type, format_record.format_uri, format_record.format_url], length=24)}"
    format_url = format_record.format_url
    s3_key = _target_s3_key(source_entity_type=source_entity_type, source_entity_id=source_entity_id, source_file_id=source_file_id, format_type=format_type, format_url=format_url or format_record.format_uri)
    return {
        "source_file_id": source_file_id,
        "source_entity_type": source_entity_type,
        "source_entity_id": source_entity_id,
        "format_type": format_type,
        "format_uri": format_record.format_uri,
        "format_url": format_url,
        "s3_key": s3_key,
        "content_type": _content_type(format_type, format_url or format_record.format_uri),
        "download_status": "not_downloaded",
        "downloaded_at_utc": None,
        "byte_size": None,
        "etag_or_hash": None,
        "snapshot_date": snapshot_date,
    }


def _entity_id(item: Mapping[str, Any], *, entity_type: str, item_index: int) -> str:
    for value in _walk_mappings(item):
        for key in ("uri", f"{entity_type}Uri", "debateUri", "questionUri", "billUri", "id", f"{entity_type}Id"):
            text = _first_text(value, key)
            if text:
                return text
    return f"generated:{entity_type}:{item_index}:{stable_hash(item, length=16)}"


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for entry in value:
            yield from _walk_mappings(entry)


def _target_s3_key(*, source_entity_type: str, source_entity_id: str, source_file_id: str, format_type: str | None, format_url: str | None) -> str:
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


def _content_type(format_type: str | None, url: str | None) -> str | None:
    ext = _extension(format_type, url)
    return {"xml": "application/xml", "pdf": "application/pdf", "json": "application/json", "html": "text/html", "txt": "text/plain", "csv": "text/csv"}.get(ext)


def _normalise_format_type(value: str | None, url: str | None) -> str | None:
    if value:
        text = str(value).strip().lower().replace("application/", "")
        if "pdf" in text:
            return "pdf"
        if "xml" in text:
            return "xml"
        if "json" in text:
            return "json"
        if "html" in text:
            return "html"
        return text[:50]
    return _infer_format_type(url)


def _infer_format_type(url: str | None) -> str | None:
    if not url:
        return None
    path = urlparse(str(url)).path.lower()
    for ext in ("pdf", "xml", "json", "html", "txt", "csv"):
        if path.endswith(f".{ext}"):
            return ext
    return None


def _looks_like_format_key(key: str) -> bool:
    return key.lower() in {"pdf", "xml", "json", "html", "txt", "csv", "akn", "docx", "writtens_pdf"}


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
        non_null_pk = unique_pk = entity_ok = type_ok = locator_ok = s3_ok = status_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        entity_ok = bool(df["source_entity_id"].notna().all() and (df["source_entity_id"].astype(str).str.strip() != "").all())
        type_ok = bool(df["format_type"].notna().all() and (df["format_type"].astype(str).str.strip() != "").all())
        locator_ok = bool(((df["format_uri"].notna()) | (df["format_url"].notna())).all())
        s3_ok = bool(df["s3_key"].notna().all() and (df["s3_key"].astype(str).str.strip() != "").all())
        status_ok = bool((df["download_status"] == "not_downloaded").all())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, entity_ok, type_ok, locator_ok, s3_ok, status_ok]) else "fail"
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
            {"check_name": "source_entity_id_populated", "status": "pass" if entity_ok else "fail"},
            {"check_name": "format_type_populated", "status": "pass" if type_ok else "fail"},
            {"check_name": "format_locator_populated", "status": "pass" if locator_ok else "fail"},
            {"check_name": "s3_key_populated", "status": "pass" if s3_ok else "fail"},
            {"check_name": "download_status_metadata_only", "status": "pass" if status_ok else "fail"},
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
