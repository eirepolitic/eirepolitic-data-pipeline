"""Builder for the `gold_content_fact_pool` table."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .client import OireachtasClient
from .io_s3 import get_bytes, put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import stable_hash, utc_now_iso
from .schemas import TableSchema

TABLE_NAME = "gold_content_fact_pool"
LATEST_CSV_PREFIX = "processed/oireachtas_unified/latest/csv"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_gold_content_fact_pool(
    *,
    client: OireachtasClient,
    s3: Any,
    bucket: str,
    schema: TableSchema,
    limit: int,
    mode: str,
    chamber: str | None = None,
    house_no: str | None = None,
) -> TableBuildResult:
    """Build deterministic content fact candidates from latest gold marts."""
    del client, chamber, house_no
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]

    input_keys = {
        "member_yearly": f"{LATEST_CSV_PREFIX}/gold_member_activity_yearly.csv",
        "member_monthly": f"{LATEST_CSV_PREFIX}/gold_member_activity_monthly.csv",
        "constituency_yearly": f"{LATEST_CSV_PREFIX}/gold_constituency_activity_yearly.csv",
        "current_members": f"{LATEST_CSV_PREFIX}/gold_current_members.csv",
    }
    inputs = {name: _read_latest_csv(s3, bucket=bucket, key=key) for name, key in input_keys.items()}

    rows: list[dict[str, Any]] = []
    rows.extend(_member_yearly_facts(inputs["member_yearly"], inputs["current_members"], snapshot_date=snapshot_date))
    rows.extend(_member_monthly_facts(inputs["member_monthly"], inputs["current_members"], snapshot_date=snapshot_date))
    rows.extend(_constituency_yearly_facts(inputs["constituency_yearly"], snapshot_date=snapshot_date))

    df = pd.DataFrame(rows, columns=schema.columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["fact_id"], keep="first")
        df = df.sort_values(["period_start", "fact_type", "metric_value", "entity_id"], ascending=[False, True, False, True])
        df = df.head(max(1, limit)).copy()

    csv_key = f"processed/oireachtas_unified/gold_csv/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/{TABLE_NAME}.csv"
    parquet_key = f"processed/oireachtas_unified/gold/{TABLE_NAME}/snapshot_date={snapshot_date}/run_id={run_id}/part-00000.parquet"
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
        "input_keys": input_keys,
        "input_rows": {name: int(len(df_in)) for name, df_in in inputs.items()},
        "candidate_rows_before_limit": len(rows),
        "output_rows": int(len(df)),
        "fact_type_values": sorted(df["fact_type"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "primary_key": schema.primary_key,
        "primary_key_unique": dq["primary_key_unique"],
        "dq_status": dq["dq_status"],
        "write_errors": write_errors,
        "s3_keys": s3_keys,
    }

    try:
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


def _read_latest_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _member_name_lookup(current_members: pd.DataFrame) -> dict[str, str]:
    if current_members.empty or not {"member_code", "full_name"}.issubset(current_members.columns):
        return {}
    return dict(zip(current_members["member_code"].astype(str), current_members["full_name"].astype(str)))


def _member_yearly_facts(member_yearly: pd.DataFrame, current_members: pd.DataFrame, *, snapshot_date: str) -> list[dict[str, Any]]:
    if member_yearly.empty:
        return []
    names = _member_name_lookup(current_members)
    facts: list[dict[str, Any]] = []
    for _, row in member_yearly.iterrows():
        member_code = str(row.get("member_code", "")).strip()
        year = str(row.get("year", "")).strip()
        if not member_code or not year:
            continue
        member_name = names.get(member_code, member_code)
        facts.append(_fact("member_speech_yearly", "member", member_code, year, f"{member_name} made {int_float(row.get('speech_count'))} speeches in {year}.", "speech_count", row.get("speech_count"), "gold_member_activity_yearly", f"{member_code}|{year}", snapshot_date))
        facts.append(_fact("member_vote_participation_yearly", "member", member_code, year, f"{member_name} voted in {int_float(row.get('vote_participation_pct'))}% of recorded divisions in {year}.", "vote_participation_pct", row.get("vote_participation_pct"), "gold_member_activity_yearly", f"{member_code}|{year}", snapshot_date))
    return facts


def _member_monthly_facts(member_monthly: pd.DataFrame, current_members: pd.DataFrame, *, snapshot_date: str) -> list[dict[str, Any]]:
    if member_monthly.empty:
        return []
    names = _member_name_lookup(current_members)
    facts: list[dict[str, Any]] = []
    for _, row in member_monthly.iterrows():
        member_code = str(row.get("member_code", "")).strip()
        year_month = str(row.get("year_month", "")).strip()
        if not member_code or not year_month:
            continue
        member_name = names.get(member_code, member_code)
        facts.append(_fact("member_speech_monthly", "member", member_code, year_month, f"{member_name} made {int_float(row.get('speech_count'))} speeches in {year_month}.", "speech_count", row.get("speech_count"), "gold_member_activity_monthly", f"{member_code}|{year_month}", snapshot_date))
        facts.append(_fact("member_votes_monthly", "member", member_code, year_month, f"{member_name} cast {int_float(row.get('votes_cast_count'))} recorded votes in {year_month}.", "votes_cast_count", row.get("votes_cast_count"), "gold_member_activity_monthly", f"{member_code}|{year_month}", snapshot_date))
    return facts


def _constituency_yearly_facts(constituency_yearly: pd.DataFrame, *, snapshot_date: str) -> list[dict[str, Any]]:
    if constituency_yearly.empty:
        return []
    facts: list[dict[str, Any]] = []
    for _, row in constituency_yearly.iterrows():
        constituency = str(row.get("constituency_name", "")).strip()
        year = str(row.get("year", "")).strip()
        if not constituency or not year:
            continue
        facts.append(_fact("constituency_speech_yearly", "constituency", constituency, year, f"Members associated with {constituency} made {int_float(row.get('speech_count'))} speeches in {year}.", "speech_count", row.get("speech_count"), "gold_constituency_activity_yearly", f"{constituency}|{year}", snapshot_date))
        facts.append(_fact("constituency_votes_yearly", "constituency", constituency, year, f"Members associated with {constituency} cast {int_float(row.get('votes_cast_count'))} recorded votes in {year}.", "votes_cast_count", row.get("votes_cast_count"), "gold_constituency_activity_yearly", f"{constituency}|{year}", snapshot_date))
    return facts


def _fact(fact_type: str, entity_type: str, entity_id: str, period: str, headline: str, metric_name: str, metric_value: Any, source_table: str, source_key: str, snapshot_date: str) -> dict[str, Any]:
    period_start, period_end = _period_bounds(period)
    fact_id = f"fact:{stable_hash([fact_type, entity_type, entity_id, period, metric_name, source_key], length=24)}"
    return {
        "fact_id": fact_id,
        "fact_type": fact_type,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "period_start": period_start,
        "period_end": period_end,
        "headline": headline,
        "metric_name": metric_name,
        "metric_value": str(metric_value or "0"),
        "source_table": source_table,
        "source_key": source_key,
        "snapshot_date": snapshot_date,
    }


def _period_bounds(period: str) -> tuple[str, str]:
    if len(period) == 7 and period[4] == "-":
        start = f"{period}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        return start, end
    if len(period) == 4:
        return f"{period}-01-01", f"{period}-12-31"
    return period, period


def int_float(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    if float(number).is_integer():
        return str(int(number))
    return str(round(float(number), 2))


def _nonblank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna("").astype(str).str.strip() != ""


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key[0]
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        pk_non_null = pk_unique = required_populated = metric_numeric = False
    else:
        pk_non_null = bool(_nonblank(df[pk]).all())
        pk_unique = bool(not df[pk].duplicated().any())
        required_populated = bool(all(_nonblank(df[col]).all() for col in ["fact_type", "entity_type", "entity_id", "period_start", "period_end", "headline", "metric_name", "source_table", "source_key", "snapshot_date"]))
        metric_numeric = bool(pd.to_numeric(df["metric_value"], errors="coerce").notna().all())
    status = "pass" if all([row_count > 0, not missing_columns, pk_non_null, pk_unique, required_populated, metric_numeric]) else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": schema.primary_key,
        "primary_key_unique": pk_unique,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if pk_non_null else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "required_content_fields_populated", "status": "pass" if required_populated else "fail"},
            {"check_name": "metric_value_numeric", "status": "pass" if metric_numeric else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
