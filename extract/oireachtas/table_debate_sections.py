"""Builder for the `silver_debate_sections` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import stable_hash, utc_now_iso
from .schemas import TableSchema


TABLE_NAME = "silver_debate_sections"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_debate_sections(
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
    """Fetch `/debates`, explode debate sections, and write outputs."""
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

    rows: list[dict[str, Any]] = []
    section_counts: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        debate_record = item.get("debateRecord") if isinstance(item.get("debateRecord"), Mapping) else item
        debate_id = _first_text(debate_record, "uri", "debateUri") or f"generated:debate:{stable_hash(debate_record, length=20)}"
        section_items = debate_record.get("debateSections") if isinstance(debate_record, Mapping) else []
        if not isinstance(section_items, list):
            section_items = []
        section_counts.append({"debate_id": debate_id, "section_count": len(section_items)})
        for index, section_item in enumerate(section_items, start=1):
            if not isinstance(section_item, Mapping):
                continue
            section = section_item.get("debateSection")
            if not isinstance(section, Mapping):
                section = section_item
            rows.append(_normalise_section_row(section, debate_id=debate_id, section_order=index, snapshot_date=snapshot_date))

    rows = _dedupe_rows(rows, primary_key="debate_section_id")
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
        "section_counts": section_counts,
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
    return TableBuildResult(table=TABLE_NAME, rows=sample_df.to_dict(orient="records"), manifest=manifest, schema=schema_payload, dq=dq, s3_keys=s3_keys)


def _normalise_section_row(section: Mapping[str, Any], *, debate_id: str, section_order: int, snapshot_date: str) -> dict[str, Any]:
    section_uri = _first_text(section, "uri", "sectionUri")
    section_eid = _first_text(section, "debateSectionId", "sectionId", "eid")
    debate_section_id = section_uri or f"generated:debate_section:{stable_hash([debate_id, section_eid, section_order], length=20)}"
    show_as = _first_text(section, "showAs", "heading", "title")
    heading = _first_text(section, "heading", "title", "showAs")
    parent_section_id = _parent_section_id(section.get("parentDebateSection"))

    return {
        "debate_section_id": debate_section_id,
        "debate_id": debate_id,
        "section_eid": section_eid,
        "section_uri": section_uri,
        "section_order": section_order,
        "heading": heading,
        "show_as": show_as,
        "parent_section_id": parent_section_id,
        "snapshot_date": snapshot_date,
    }


def _parent_section_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return _first_text(value, "uri", "debateSectionId", "sectionId", "eid")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        non_null_pk = unique_pk = debate_ok = uri_ok = order_ok = show_as_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        debate_ok = bool(df["debate_id"].notna().all() and (df["debate_id"].astype(str).str.strip() != "").all())
        uri_ok = bool(df["section_uri"].notna().all() and (df["section_uri"].astype(str).str.strip() != "").all())
        order_ok = bool(df["section_order"].notna().all() and (df["section_order"].astype(str).str.strip() != "").all())
        show_as_ok = bool(df["show_as"].notna().any() and (df["show_as"].astype(str).str.strip() != "").any())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, debate_ok, uri_ok, order_ok, show_as_ok]) else "fail"
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
            {"check_name": "section_uri_populated", "status": "pass" if uri_ok else "fail"},
            {"check_name": "section_order_populated", "status": "pass" if order_ok else "fail"},
            {"check_name": "show_as_any_populated", "status": "pass" if show_as_ok else "fail"},
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
