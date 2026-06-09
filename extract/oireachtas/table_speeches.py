"""Builder for the `silver_speeches` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Mapping

import pandas as pd
import requests

from .client import OireachtasClient
from .io_s3 import put_bytes, put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import normalize_format_url, safe_text, stable_hash, utc_now_iso
from .schemas import TableSchema
from .xml_debates import ParsedSpeech, parse_debate_xml


TABLE_NAME = "silver_speeches"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_speeches(
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
    """Fetch debate metadata/XML, parse speeches, and write silver outputs."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    endpoint = "/debates"
    date_start = date_start or "2025-01-01"
    date_end = date_end or "2025-01-31"
    params: dict[str, Any] = {
        "chamber_id": f"/ie/oireachtas/house/{chamber or 'dail'}/{house_no or '34'}",
        "lang": "en",
        "date_start": date_start,
        "date_end": date_end,
        "limit": max(1, min(limit, 10)),
    }

    summary = client.get_json_summary(endpoint, params=params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch {endpoint}: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected /debates results type: {type(results).__name__}")

    rows: list[dict[str, Any]] = []
    xml_diagnostics: list[dict[str, Any]] = []
    xml_downloads: list[dict[str, Any]] = []
    http = requests.Session()
    http.headers.update({"User-Agent": "eirepolitic-data-pipeline/1.0"})

    for item in results:
        if not isinstance(item, Mapping):
            continue
        record = item.get("debateRecord") if isinstance(item.get("debateRecord"), Mapping) else item
        debate_id = _first_text(record, "uri", "debateUri")
        debate_date = _first_text(record, "date") or _first_text(item, "contextDate")
        formats = record.get("formats") if isinstance(record.get("formats"), Mapping) else {}
        xml_format = formats.get("xml") if isinstance(formats.get("xml"), Mapping) else {}
        xml_uri = _first_text(xml_format, "uri", "url", "href")
        xml_url = normalize_format_url(xml_uri)
        if not debate_id or not xml_url:
            xml_downloads.append({"debate_id": debate_id, "xml_url": xml_url, "status": "skipped_missing_identity_or_url"})
            continue

        source_file_id = _source_file_id(debate_id, xml_uri, xml_url)
        xml_source_key = _xml_source_key(debate_id, source_file_id)
        try:
            xml_bytes = _download_xml(http, xml_url)
            put_bytes(s3, bucket=bucket, key=xml_source_key, body=xml_bytes, content_type="application/xml")
            parsed, diagnostics = parse_debate_xml(xml_bytes=xml_bytes, debate_id=debate_id, debate_date=debate_date)
            rows.extend(
                _normalise_speech_row(
                    speech,
                    source_file_id=source_file_id,
                    xml_source_key=xml_source_key,
                    snapshot_date=snapshot_date,
                )
                for speech in parsed
            )
            xml_downloads.append(
                {
                    "debate_id": debate_id,
                    "xml_url": xml_url,
                    "status": "success",
                    "byte_size": len(xml_bytes),
                    "sha256": sha256(xml_bytes).hexdigest(),
                    "speech_rows": len(parsed),
                    "source_file_id": source_file_id,
                    "xml_source_key": xml_source_key,
                }
            )
            xml_diagnostics.append({"debate_id": debate_id, **diagnostics})
        except Exception as exc:
            xml_downloads.append(
                {
                    "debate_id": debate_id,
                    "xml_url": xml_url,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    rows = _dedupe_rows(rows, primary_key="speech_id")
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
        "endpoint": endpoint,
        "params": dict(summary.params),
        "url": summary.url,
        "status_code": summary.status_code,
        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "xml_downloads": xml_downloads,
        "xml_diagnostics": xml_diagnostics,
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


def _download_xml(session: requests.Session, url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            content = response.content
            if not content or b"<" not in content[:500]:
                raise ValueError("Response does not look like XML")
            return content
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
    raise RuntimeError(f"XML download failed: {last_error}")


def _normalise_speech_row(speech: ParsedSpeech, *, source_file_id: str, xml_source_key: str, snapshot_date: str) -> dict[str, Any]:
    member_code = _member_code_from_ref(speech.speaker_ref)
    match_method = "speaker_ref_member_code" if member_code else None
    confidence = 1.0 if member_code else None
    text_hash = sha256(speech.speech_text.encode("utf-8")).hexdigest()[:24]
    return {
        "speech_id": speech.speech_id,
        "debate_id": speech.debate_id,
        "debate_section_id": speech.debate_section_id,
        "debate_date": speech.debate_date,
        "speech_order": speech.speech_order,
        "speaker_ref": speech.speaker_ref,
        "speaker_name": speech.speaker_name,
        "speaker_member_code": member_code,
        "speaker_match_method": match_method,
        "speaker_match_confidence": confidence,
        "speech_text": speech.speech_text,
        "speech_text_hash": text_hash,
        "word_count": len(speech.speech_text.split()),
        "char_count": len(speech.speech_text),
        "language": speech.language,
        "source_file_id": source_file_id,
        "xml_source_key": xml_source_key,
        "snapshot_date": snapshot_date,
    }


def _member_code_from_ref(value: str | None) -> str | None:
    text = safe_text(value)
    if not text:
        return None
    text = text.lstrip("#")
    for marker in ("member/", "member-"):
        if marker in text:
            candidate = text.split(marker, 1)[1].split("/", 1)[0].strip()
            return candidate or None
    return None


def _source_file_id(debate_id: str, xml_uri: str | None, xml_url: str | None) -> str:
    return f"source_file:{stable_hash(['debate', debate_id, 'xml', xml_uri, xml_url], length=24)}"


def _xml_source_key(debate_id: str, source_file_id: str) -> str:
    entity_slug = _safe_slug(debate_id)[-120:] or "debate"
    file_slug = _safe_slug(source_file_id)
    return str(PurePosixPath("raw/oireachtas_unified/source_files", "debate", entity_slug, f"{file_slug}.xml"))


def _safe_slug(value: str) -> str:
    text = value.strip().replace("https://", "").replace("http://", "")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in text).strip("-")


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
        non_null_pk = unique_pk = debate_ok = section_any = order_ok = text_ok = counts_ok = source_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        debate_ok = bool(df["debate_id"].notna().all() and (df["debate_id"].astype(str).str.strip() != "").all())
        section_any = bool(df["debate_section_id"].notna().any())
        order_ok = bool(df["speech_order"].notna().all() and not df.duplicated(subset=["debate_id", "speech_order"]).any())
        text_ok = bool(df["speech_text"].notna().all() and (df["speech_text"].astype(str).str.strip() != "").all())
        counts_ok = bool((df["word_count"].fillna(0) > 0).all() and (df["char_count"].fillna(0) > 0).all())
        source_ok = bool(df["source_file_id"].notna().all() and df["xml_source_key"].notna().all())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, debate_ok, section_any, order_ok, text_ok, counts_ok, source_ok]) else "fail"
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
            {"check_name": "debate_id_populated", "status": "pass" if debate_ok else "fail"},
            {"check_name": "debate_section_id_any_populated", "status": "pass" if section_any else "fail"},
            {"check_name": "speech_order_unique_per_debate", "status": "pass" if order_ok else "fail"},
            {"check_name": "speech_text_populated", "status": "pass" if text_ok else "fail"},
            {"check_name": "word_and_char_counts_positive", "status": "pass" if counts_ok else "fail"},
            {"check_name": "source_file_fields_populated", "status": "pass" if source_ok else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
