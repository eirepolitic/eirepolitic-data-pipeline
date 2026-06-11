"""Builder for the `silver_bill_related_docs` table."""

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

TABLE_NAME = "silver_bill_related_docs"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_bill_related_docs(
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
    """Fetch `/legislation`, normalize bill related documents, and write outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = schema.endpoint or "/legislation"
    params: dict[str, Any] = {
        "chamber": chamber or "dail",
        "house_no": house_no or "34",
        "date_start": date_start or "2025-01-01",
        "date_end": date_end or "2025-01-31",
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
    raw_doc_count = 0
    bills_with_related_docs: set[str] = set()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        item_rows = _normalise_related_doc_rows(item, snapshot_date=snapshot_date)
        raw_doc_count += _raw_related_doc_count(item)
        if item_rows:
            bills_with_related_docs.add(str(item_rows[0].get("bill_id") or ""))
        rows.extend(item_rows)

    rows = _dedupe_rows(rows, primary_key="related_doc_id")
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
        "raw_related_doc_rows": raw_doc_count,
        "output_rows": int(len(df)),
        "bills_with_related_docs": len([bill_id for bill_id in bills_with_related_docs if bill_id]),
        "pdf_source_rows": int(df["format_pdf_uri"].notna().sum()) if not df.empty else 0,
        "xml_source_rows": int(df["format_xml_uri"].notna().sum()) if not df.empty else 0,
        "doc_type_values": sorted(df["doc_type"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "language_values": sorted(df["language"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "source_file_ids_deterministic": dq["source_file_ids_deterministic"],
        "dq_status": dq["dq_status"],
        "raw_result_key_paths": sorted(_key_paths(first_result, max_depth=8)),
        "raw_result_structure": _compact_structure(first_result, max_depth=6),
        "nested_collection_summary": _collection_summary(first_result),
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


def _normalise_related_doc_rows(item: Mapping[str, Any], *, snapshot_date: str) -> list[dict[str, Any]]:
    bill = item.get("bill") if isinstance(item.get("bill"), Mapping) else item
    bill_uri = _first_text(bill, "uri", "billUri")
    bill_id = bill_uri or _first_text(bill, "billId", "id") or f"generated:bill:{stable_record_hash(bill, length=24)}"
    related_docs = bill.get("relatedDocs") if isinstance(bill.get("relatedDocs"), list) else []

    rows: list[dict[str, Any]] = []
    for doc_index, doc_wrapper in enumerate(related_docs):
        if not isinstance(doc_wrapper, Mapping):
            continue
        doc = doc_wrapper.get("relatedDoc") if isinstance(doc_wrapper.get("relatedDoc"), Mapping) else doc_wrapper
        if not isinstance(doc, Mapping):
            continue

        doc_uri = _first_text(doc, "uri", "relatedDocUri", "docUri")
        doc_label = _first_text(doc, "showAs", "title", "label")
        doc_date = parse_iso_date(doc.get("date"))
        doc_type = _first_text(doc, "docType", "type")
        language = _first_text(doc, "lang", "language")
        related_doc_id = doc_uri or f"generated:bill_related_doc:{stable_hash([bill_id, doc_label, doc_date, doc_type, language, doc_index], length=24)}"

        formats = doc.get("formats") if isinstance(doc.get("formats"), Mapping) else {}
        pdf_uri, pdf_url_for_hash, pdf_url = _format_locator(formats, "pdf")
        xml_uri, xml_url_for_hash, xml_url = _format_locator(formats, "xml")

        source_file_id_pdf = _source_file_id(bill_id, "pdf", pdf_uri, pdf_url_for_hash)
        source_file_id_xml = _source_file_id(bill_id, "xml", xml_uri, xml_url_for_hash)
        s3_pdf_key = _target_s3_key("legislation", bill_id, source_file_id_pdf, "pdf", pdf_url_for_hash or pdf_uri)
        s3_xml_key = _target_s3_key("legislation", bill_id, source_file_id_xml, "xml", xml_url_for_hash or xml_uri)

        rows.append(
            {
                "related_doc_id": related_doc_id,
                "bill_id": bill_id,
                "related_doc_label": doc_label,
                "related_doc_date": doc_date,
                "doc_type": doc_type,
                "language": language,
                "format_pdf_uri": pdf_uri,
                "format_pdf_url": pdf_url,
                "format_xml_uri": xml_uri,
                "format_xml_url": xml_url,
                "source_file_id_pdf": source_file_id_pdf,
                "source_file_id_xml": source_file_id_xml,
                "s3_pdf_key": s3_pdf_key,
                "s3_xml_key": s3_xml_key,
                "snapshot_date": snapshot_date,
            }
        )
    return rows


def _format_locator(formats: Mapping[str, Any], format_type: str) -> tuple[str | None, str | None, str | None]:
    raw = formats.get(format_type) if isinstance(formats.get(format_type), Mapping) else {}
    if not isinstance(raw, Mapping):
        return None, None, None
    format_uri = _first_text(raw, "uri", "formatUri")
    raw_url = _first_text(raw, "url", "href", "downloadUrl", "formatUrl")
    url_for_hash = raw_url
    if not url_for_hash and format_uri and format_uri.startswith("http"):
        url_for_hash = format_uri
    if not format_uri and url_for_hash and url_for_hash.startswith("http"):
        format_uri = url_for_hash
    output_url = normalize_format_url(url_for_hash or format_uri)
    return format_uri, url_for_hash, output_url


def _source_file_id(bill_id: str, format_type: str, format_uri: str | None, format_url: str | None) -> str | None:
    if not (format_uri or format_url):
        return None
    return f"source_file:{stable_hash(['legislation', bill_id, format_type, format_uri, format_url], length=24)}"


def _target_s3_key(source_entity_type: str, source_entity_id: str, source_file_id: str | None, format_type: str | None, format_url: str | None) -> str | None:
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


def _raw_related_doc_count(item: Mapping[str, Any]) -> int:
    bill = item.get("bill") if isinstance(item.get("bill"), Mapping) else item
    related_docs = bill.get("relatedDocs") if isinstance(bill.get("relatedDocs"), list) else []
    return len(related_docs)


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
        non_null_pk = unique_pk = bill_ok = label_ok = date_ok = doc_type_ok = source_ok = source_id_ok = s3_key_ok = deterministic_ids = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        bill_ok = bool(df["bill_id"].notna().all() and (df["bill_id"].astype(str).str.strip() != "").all())
        label_ok = bool(df["related_doc_label"].notna().all() and (df["related_doc_label"].astype(str).str.strip() != "").all())
        date_ok = bool(df["related_doc_date"].notna().all() and (df["related_doc_date"].astype(str).str.strip() != "").all())
        doc_type_ok = bool(df["doc_type"].notna().all() and (df["doc_type"].astype(str).str.strip() != "").all())
        source_ok = bool((df["format_pdf_uri"].notna() | df["format_pdf_url"].notna() | df["format_xml_uri"].notna() | df["format_xml_url"].notna()).all())
        source_id_ok = bool((df["source_file_id_pdf"].notna() | df["source_file_id_xml"].notna()).all())
        s3_key_ok = bool((df["s3_pdf_key"].notna() | df["s3_xml_key"].notna()).all())
        deterministic_ids = _source_file_ids_match(df)
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, bill_ok, label_ok, date_ok, doc_type_ok, source_ok, source_id_ok, s3_key_ok, deterministic_ids]) else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "source_file_ids_deterministic": deterministic_ids,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail"},
            {"check_name": "bill_id_populated", "status": "pass" if bill_ok else "fail"},
            {"check_name": "related_doc_label_populated", "status": "pass" if label_ok else "fail"},
            {"check_name": "related_doc_date_populated", "status": "pass" if date_ok else "fail"},
            {"check_name": "doc_type_populated", "status": "pass" if doc_type_ok else "fail"},
            {"check_name": "at_least_one_source_format", "status": "pass" if source_ok else "fail"},
            {"check_name": "source_file_id_populated_when_source_format_present", "status": "pass" if source_id_ok else "fail"},
            {"check_name": "s3_key_populated_when_source_format_present", "status": "pass" if s3_key_ok else "fail"},
            {"check_name": "source_file_ids_t09_pattern", "status": "pass" if deterministic_ids else "fail"},
        ],
    }


def _source_file_ids_match(df: pd.DataFrame) -> bool:
    for _, row in df.iterrows():
        bill_id = _none_if_blank(row.get("bill_id"))
        for format_type, id_col, uri_col, url_col in (("pdf", "source_file_id_pdf", "format_pdf_uri", "format_pdf_url"), ("xml", "source_file_id_xml", "format_xml_uri", "format_xml_url")):
            actual = _none_if_blank(row.get(id_col))
            uri = _none_if_blank(row.get(uri_col))
            output_url = _none_if_blank(row.get(url_col))
            url_for_hash = _t09_url_for_hash(uri=uri, output_url=output_url)
            expected = _source_file_id(str(bill_id), format_type, uri, url_for_hash) if bill_id and (uri or url_for_hash) else None
            if actual != expected:
                return False
    return True


def _t09_url_for_hash(*, uri: str | None, output_url: str | None) -> str | None:
    if uri and uri.startswith("http"):
        return uri
    if output_url and not uri:
        return output_url
    return None


def _none_if_blank(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _compact_structure(value: Any, *, depth: int = 0, max_depth: int = 6) -> Any:
    if depth >= max_depth:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        return {str(key): _compact_structure(child, depth=depth + 1, max_depth=max_depth) for key, child in value.items()}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "first": _compact_structure(value[0], depth=depth + 1, max_depth=max_depth) if value else None}
    if value is None:
        return None
    text = str(value)
    return text[:200] + ("..." if len(text) > 200 else "")


def _collection_summary(value: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, list):
                first_type = type(child[0]).__name__ if child else None
                first_keys = sorted(child[0].keys()) if child and isinstance(child[0], Mapping) else []
                rows.append({"path": path, "length": len(child), "first_type": first_type, "first_keys": first_keys})
            rows.extend(_collection_summary(child, prefix=path))
    elif isinstance(value, list):
        for child in value[:1]:
            rows.extend(_collection_summary(child, prefix=f"{prefix}[]"))
    return rows[:75]


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
