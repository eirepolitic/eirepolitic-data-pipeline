"""Builder for the `silver_divisions` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient, ResponseSummary
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import parse_iso_date, stable_hash, stable_record_hash, utc_now_iso
from .schemas import TableSchema

TABLE_NAME = "silver_divisions"
CANONICAL_ENDPOINT = "/divisions"
FALLBACK_ENDPOINT = "/votes"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_divisions(
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
    """Fetch division events from canonical `/divisions` with `/votes` fallback."""
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]
    date_start = date_start or "2025-01-01"
    date_end = date_end or "2025-01-31"
    params: dict[str, Any] = {
        "chamber_id": f"/ie/oireachtas/house/{chamber or 'dail'}/{house_no or '34'}",
        "date_start": date_start,
        "date_end": date_end,
        "limit": max(1, min(limit, 200)),
    }

    summary, endpoint_used, fallback_used = _fetch_divisions(client, params)
    if not summary.ok or not summary.payload:
        raise RuntimeError(f"Failed to fetch divisions: {summary.error or summary.status_code}")

    payload = dict(summary.payload)
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError(f"Unexpected division results type: {type(results).__name__}")

    rows = [
        _normalise_division_row(item, snapshot_date=snapshot_date)
        for item in results
        if isinstance(item, Mapping)
    ]
    rows = _dedupe_rows(rows, primary_key="division_id")
    df = pd.DataFrame(rows, columns=schema.columns)

    raw_key = f"raw/oireachtas_unified/api/divisions/snapshot_date={snapshot_date}/run_id={run_id}/page-00000.json"
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
        "canonical_endpoint": CANONICAL_ENDPOINT,
        "fallback_endpoint": FALLBACK_ENDPOINT,
        "endpoint_used": endpoint_used,
        "fallback_used": fallback_used,
        "params": dict(summary.params),
        "url": summary.url,
        "status_code": summary.status_code,
        "raw_rows": len(results),
        "output_rows": int(len(df)),
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "raw_result_samples": results[:3],
        "raw_result_key_paths": sorted(_key_paths(results[0], max_depth=8)) if results and isinstance(results[0], Mapping) else [],
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


def _fetch_divisions(client: OireachtasClient, params: Mapping[str, Any]) -> tuple[ResponseSummary, str, bool]:
    canonical = client.get_json_summary(CANONICAL_ENDPOINT, params=params)
    if canonical.ok and canonical.payload is not None:
        return canonical, CANONICAL_ENDPOINT, False
    fallback = client.get_json_summary(FALLBACK_ENDPOINT, params=params)
    return fallback, FALLBACK_ENDPOINT, True


def _normalise_division_row(item: Mapping[str, Any], *, snapshot_date: str) -> dict[str, Any]:
    record = _record(item)
    division_uri = _first_text(record, "uri", "divisionUri", "voteUri")
    vote_id = _first_text(record, "voteId", "divisionId", "id", "eId")
    division_id = division_uri or vote_id or f"generated:division:{stable_hash(record, length=24)}"

    context_date = parse_iso_date(item.get("contextDate"))
    division_date = (
        parse_iso_date(record.get("date"))
        or parse_iso_date(record.get("voteDate"))
        or parse_iso_date(record.get("divisionDate"))
        or context_date
    )

    house = _first_mapping(record, "house")
    chamber_record = _first_mapping(record, "chamber")
    house_uri = _first_text(house, "uri") or _first_text(chamber_record, "uri") or _deep_first_text(record, "houseUri")
    house_no = _first_text(house, "houseNo", "number") or _deep_first_text(record, "houseNo")
    chamber = (
        _first_text(house, "houseCode", "chamberCode", "showAs")
        or _first_text(chamber_record, "houseCode", "chamberCode", "showAs")
        or _deep_first_text(record, "chamberCode")
    )

    debate = _first_mapping(record, "debate", "debateRecord")
    debate_section = _first_mapping(record, "debateSection")
    debate_uri = _first_text(debate, "uri", "debateUri") or _deep_first_text(record, "debateUri")
    debate_section_uri = _first_text(debate_section, "uri", "sectionUri") or _deep_first_text(record, "debateSectionUri")
    debate_show_as = (
        _first_text(debate_section, "showAs", "heading", "title")
        or _first_text(debate, "showAs", "title")
        or _deep_first_text(record, "debateShowAs")
    )

    subject = (
        _first_text(record, "subject", "showAs", "title", "motion", "question")
        or _deep_first_text(record, "subject")
        or _deep_first_text(record, "showAs")
    )
    outcome = (
        _first_text(record, "outcome", "result", "decision", "voteResult")
        or _deep_first_text(record, "outcome")
        or _deep_first_text(record, "result")
    )
    committee_code = (
        _first_text(record, "committeeCode")
        or _deep_first_text(record, "committeeCode")
        or _deep_first_text(record, "committeeId")
    )

    return {
        "division_id": division_id,
        "vote_id": vote_id,
        "division_date": division_date,
        "chamber": chamber,
        "house_uri": house_uri,
        "house_no": house_no,
        "committee_code": committee_code,
        "subject": subject,
        "outcome": outcome,
        "debate_id": debate_uri,
        "debate_section_id": debate_section_uri,
        "debate_show_as": debate_show_as,
        "api_result_hash": stable_record_hash(item),
        "snapshot_date": snapshot_date,
    }


def _record(item: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("division", "vote", "divisionRecord", "voteRecord"):
        value = item.get(key)
        if isinstance(value, Mapping):
            return value
    return item


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


def _deep_first_text(value: Any, target_key: str) -> str | None:
    if isinstance(value, Mapping):
        direct = value.get(target_key)
        if direct is not None and not isinstance(direct, (dict, list)):
            text = str(direct).strip()
            if text:
                return text
        for child in value.values():
            found = _deep_first_text(child, target_key)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _deep_first_text(child, target_key)
            if found:
                return found
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
        non_null_pk = unique_pk = date_ok = house_ok = subject_ok = outcome_any = False
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        date_ok = bool(df["division_date"].notna().all() and (df["division_date"].astype(str).str.strip() != "").all())
        house_ok = bool(df["house_uri"].notna().all() and (df["house_uri"].astype(str).str.strip() != "").all())
        subject_ok = bool(df["subject"].notna().all() and (df["subject"].astype(str).str.strip() != "").all())
        outcome_any = bool(df["outcome"].notna().any() and (df["outcome"].astype(str).str.strip() != "").any())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, date_ok, house_ok, subject_ok, outcome_any]) else "fail"
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
            {"check_name": "division_date_populated", "status": "pass" if date_ok else "fail"},
            {"check_name": "house_uri_populated", "status": "pass" if house_ok else "fail"},
            {"check_name": "subject_populated", "status": "pass" if subject_ok else "fail"},
            {"check_name": "outcome_any_populated", "status": "pass" if outcome_any else "fail"},
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
