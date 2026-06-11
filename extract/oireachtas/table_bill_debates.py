"""Builder for the `silver_bill_debates` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema

TABLE_NAME = "silver_bill_debates"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_bill_debates(
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
    """Fetch `/legislation`, normalize bill debate references, and write outputs."""
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
    raw_debate_count = 0
    bills_with_debates: set[str] = set()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        item_rows = _normalise_debate_rows(item, snapshot_date=snapshot_date)
        raw_debate_count += _raw_debate_count(item)
        if item_rows:
            bills_with_debates.add(str(item_rows[0].get("bill_id") or ""))
        rows.extend(item_rows)

    rows = _dedupe_rows(rows, primary_key="bill_debate_id")
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
        "raw_debate_rows": raw_debate_count,
        "output_rows": int(len(df)),
        "bills_with_debates": len([bill_id for bill_id in bills_with_debates if bill_id]),
        "debate_section_rows": int(df["debate_section_id"].notna().sum()) if not df.empty else 0,
        "chamber_name_values": sorted(df["chamber_name"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
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


def _normalise_debate_rows(item: Mapping[str, Any], *, snapshot_date: str) -> list[dict[str, Any]]:
    bill = item.get("bill") if isinstance(item.get("bill"), Mapping) else item
    bill_uri = _first_text(bill, "uri", "billUri")
    bill_id = bill_uri or _first_text(bill, "billId", "id") or f"generated:bill:{stable_record_hash(bill, length=24)}"
    debates = bill.get("debates") if isinstance(bill.get("debates"), list) else []

    rows: list[dict[str, Any]] = []
    for debate_index, debate in enumerate(debates):
        if not isinstance(debate, Mapping):
            continue
        debate_uri = _first_text(debate, "uri", "debateUri")
        debate_date = parse_iso_date(debate.get("date"))
        debate_show_as = _first_text(debate, "showAs", "title", "label")
        debate_section_id = _first_text(debate, "debateSectionId", "sectionId")
        chamber = debate.get("chamber") if isinstance(debate.get("chamber"), Mapping) else {}
        chamber_uri = _first_text(chamber, "uri")
        chamber_name = _first_text(chamber, "showAs", "name")
        debate_order = str(debate_index + 1)
        debate_id = debate_uri or f"generated:debate:{stable_hash([bill_id, debate_show_as, debate_date, debate_section_id, debate_order], length=24)}"
        bill_debate_id = f"bill_debate:{stable_hash([bill_id, debate_uri, debate_section_id, debate_order], length=24)}"
        rows.append(
            {
                "bill_debate_id": bill_debate_id,
                "bill_id": bill_id,
                "debate_id": debate_id,
                "debate_uri": debate_uri,
                "debate_date": debate_date,
                "debate_show_as": debate_show_as,
                "debate_section_id": debate_section_id,
                "chamber_uri": chamber_uri,
                "chamber_name": chamber_name,
                "debate_order": debate_order,
                "snapshot_date": snapshot_date,
            }
        )
    return rows


def _raw_debate_count(item: Mapping[str, Any]) -> int:
    bill = item.get("bill") if isinstance(item.get("bill"), Mapping) else item
    debates = bill.get("debates") if isinstance(bill.get("debates"), list) else []
    return len(debates)


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
        non_null_pk = unique_pk = bill_ok = debate_id_ok = uri_ok = date_ok = show_as_ok = section_ok = chamber_ok = order_ok = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        bill_ok = bool(df["bill_id"].notna().all() and (df["bill_id"].astype(str).str.strip() != "").all())
        debate_id_ok = bool(df["debate_id"].notna().all() and (df["debate_id"].astype(str).str.strip() != "").all())
        uri_ok = bool(df["debate_uri"].notna().all() and (df["debate_uri"].astype(str).str.strip() != "").all())
        date_ok = bool(df["debate_date"].notna().all() and (df["debate_date"].astype(str).str.strip() != "").all())
        show_as_ok = bool(df["debate_show_as"].notna().all() and (df["debate_show_as"].astype(str).str.strip() != "").all())
        section_ok = bool(df["debate_section_id"].notna().all() and (df["debate_section_id"].astype(str).str.strip() != "").all())
        chamber_ok = bool(df["chamber_uri"].notna().all() and df["chamber_name"].notna().all() and (df["chamber_uri"].astype(str).str.strip() != "").all() and (df["chamber_name"].astype(str).str.strip() != "").all())
        order_ok = bool(df["debate_order"].notna().all() and (df["debate_order"].astype(str).str.strip() != "").all())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, bill_ok, debate_id_ok, uri_ok, date_ok, show_as_ok, section_ok, chamber_ok, order_ok]) else "fail"
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
            {"check_name": "bill_id_populated", "status": "pass" if bill_ok else "fail"},
            {"check_name": "debate_id_populated", "status": "pass" if debate_id_ok else "fail"},
            {"check_name": "debate_uri_populated", "status": "pass" if uri_ok else "fail"},
            {"check_name": "debate_date_populated", "status": "pass" if date_ok else "fail"},
            {"check_name": "debate_show_as_populated", "status": "pass" if show_as_ok else "fail"},
            {"check_name": "debate_section_id_populated", "status": "pass" if section_ok else "fail"},
            {"check_name": "chamber_populated", "status": "pass" if chamber_ok else "fail"},
            {"check_name": "debate_order_populated", "status": "pass" if order_ok else "fail"},
        ],
    }


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
