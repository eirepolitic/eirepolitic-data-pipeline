"""Builder for the `silver_division_tallies` table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from .client import OireachtasClient
from .io_s3 import put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import stable_hash, utc_now_iso
from .schemas import TableSchema
from .table_divisions import CANONICAL_ENDPOINT, FALLBACK_ENDPOINT, _fetch_divisions, _record

TABLE_NAME = "silver_division_tallies"

VOTE_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "taVotes": ("ta", "yes"),
    "nilVotes": ("nil", "no"),
    "staonVotes": ("staon", "abstain"),
}


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_silver_division_tallies(
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
    """Fetch divisions and emit one row per division/tally category."""
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

    rows: list[dict[str, Any]] = []
    source_diagnostics: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        record = _record(item)
        division_id = _division_id(record)
        tallies = record.get("tallies") if isinstance(record.get("tallies"), Mapping) else {}
        category_rows = _normalise_tallies(
            tallies,
            division_id=division_id,
            snapshot_date=snapshot_date,
        )
        rows.extend(category_rows)
        source_diagnostics.append(
            {
                "division_id": division_id,
                "source_categories": sorted(str(key) for key in tallies.keys()),
                "output_categories": [row["vote_code"] for row in category_rows],
                "category_count": len(category_rows),
            }
        )

    rows = _dedupe_rows(rows, primary_key="division_tally_id")
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
        "division_count": int(df["division_id"].nunique()) if not df.empty else 0,
        "vote_codes": sorted(df["vote_code"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "source_diagnostics": source_diagnostics,
        "tally_member_mismatches": _tally_member_mismatches(rows),
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


def _normalise_tallies(
    tallies: Mapping[str, Any],
    *,
    division_id: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_key, value in tallies.items():
        if not isinstance(value, Mapping):
            continue
        vote_code, vote_label = VOTE_CATEGORY_MAP.get(
            str(source_key),
            (_generic_vote_code(str(source_key)), _generic_vote_label(str(source_key))),
        )
        members = value.get("members")
        members_list = members if isinstance(members, list) else []
        api_tally = _to_non_negative_int(value.get("tally"))
        member_count = api_tally if api_tally is not None else len(members_list)
        rows.append(
            {
                "division_tally_id": f"division_tally:{stable_hash([division_id, vote_code], length=24)}",
                "division_id": division_id,
                "vote_code": vote_code,
                "vote_label": vote_label,
                "show_as": _text(value.get("showAs")) or vote_label,
                "member_count": member_count,
                "snapshot_date": snapshot_date,
                "_api_tally": api_tally,
                "_members_length": len(members_list),
                "_source_key": str(source_key),
            }
        )
    return rows


def _division_id(record: Mapping[str, Any]) -> str:
    for key in ("uri", "divisionUri", "voteUri", "voteId", "divisionId", "id", "eId"):
        value = _text(record.get(key))
        if value:
            return value
    return f"generated:division:{stable_hash(record, length=24)}"


def _generic_vote_code(source_key: str) -> str:
    value = source_key.strip()
    if value.lower().endswith("votes"):
        value = value[:-5]
    output: list[str] = []
    for char in value:
        if char.isupper() and output:
            output.append("_")
        output.append(char.lower())
    return "".join(output).strip("_") or "unknown"


def _generic_vote_label(source_key: str) -> str:
    code = _generic_vote_code(source_key)
    return code.replace("_", " ")


def _to_non_negative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _tally_member_mismatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        api_tally = row.get("_api_tally")
        members_length = row.get("_members_length")
        if api_tally is not None and members_length is not None and api_tally != members_length:
            mismatches.append(
                {
                    "division_id": row.get("division_id"),
                    "vote_code": row.get("vote_code"),
                    "api_tally": api_tally,
                    "members_length": members_length,
                }
            )
    return mismatches


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
        non_null_pk = unique_pk = division_ok = code_ok = label_ok = show_as_ok = count_ok = mismatch_ok = categories_ok = False
        division_count = 0
    else:
        non_null_pk = bool(df[pk].notna().all() and (df[pk].astype(str).str.strip() != "").all())
        unique_pk = bool(not df[pk].duplicated().any())
        division_ok = bool(df["division_id"].notna().all() and (df["division_id"].astype(str).str.strip() != "").all())
        code_ok = bool(df["vote_code"].notna().all() and (df["vote_code"].astype(str).str.strip() != "").all())
        label_ok = bool(df["vote_label"].notna().all() and (df["vote_label"].astype(str).str.strip() != "").all())
        show_as_ok = bool(df["show_as"].notna().all() and (df["show_as"].astype(str).str.strip() != "").all())
        numeric_counts = pd.to_numeric(df["member_count"], errors="coerce")
        count_ok = bool(numeric_counts.notna().all() and (numeric_counts >= 0).all())
        api_tallies = pd.to_numeric(df["_api_tally"], errors="coerce")
        member_lengths = pd.to_numeric(df["_members_length"], errors="coerce")
        comparable = api_tallies.notna() & member_lengths.notna()
        mismatch_ok = bool((api_tallies[comparable] == member_lengths[comparable]).all())
        required = {"ta", "nil", "staon"}
        division_count = int(df["division_id"].nunique())
        categories_ok = all(required.issubset(set(group["vote_code"].astype(str))) for _, group in df.groupby("division_id"))
    status = "pass" if all([
        row_count > 0,
        not missing_columns,
        non_null_pk,
        unique_pk,
        division_ok,
        code_ok,
        label_ok,
        show_as_ok,
        count_ok,
        mismatch_ok,
        categories_ok,
    ]) else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "division_count": division_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": unique_pk,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if non_null_pk else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if unique_pk else "fail"},
            {"check_name": "division_id_populated", "status": "pass" if division_ok else "fail"},
            {"check_name": "vote_code_populated", "status": "pass" if code_ok else "fail"},
            {"check_name": "vote_label_populated", "status": "pass" if label_ok else "fail"},
            {"check_name": "show_as_populated", "status": "pass" if show_as_ok else "fail"},
            {"check_name": "member_count_non_negative", "status": "pass" if count_ok else "fail"},
            {"check_name": "api_tally_equals_members_length", "status": "pass" if mismatch_ok else "fail"},
            {"check_name": "standard_categories_per_division", "status": "pass" if categories_ok else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
