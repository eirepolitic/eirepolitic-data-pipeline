"""Builder for the `gold_member_activity_monthly` table."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .client import OireachtasClient
from .io_s3 import get_bytes, put_dataframe_csv, put_dataframe_parquet, put_json
from .normalize import utc_now_iso
from .schemas import TableSchema

TABLE_NAME = "gold_member_activity_monthly"
LATEST_CSV_PREFIX = "processed/oireachtas_unified/latest/csv"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_gold_member_activity_monthly(
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
    """Read latest silver speech/vote tables and build monthly member metrics."""
    del client, chamber, house_no
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]

    input_keys = {
        "current_members": f"{LATEST_CSV_PREFIX}/gold_current_members.csv",
        "speeches": f"{LATEST_CSV_PREFIX}/silver_speeches.csv",
        "member_votes": f"{LATEST_CSV_PREFIX}/silver_member_votes.csv",
    }
    inputs = {name: _read_latest_csv(s3, bucket=bucket, key=key) for name, key in input_keys.items()}

    current_members = inputs["current_members"]
    speeches = inputs["speeches"]
    member_votes = inputs["member_votes"]

    speech_metrics = _speech_metrics(speeches)
    vote_metrics = _vote_metrics(member_votes)
    member_months = _member_month_grid(current_members, speech_metrics, vote_metrics)

    metrics = member_months.merge(speech_metrics, on=["member_code", "year_month"], how="left")
    metrics = metrics.merge(vote_metrics, on=["member_code", "year_month"], how="left")
    for column in ["speech_count", "debate_day_count", "votes_cast_count"]:
        metrics[column] = pd.to_numeric(metrics.get(column), errors="coerce").fillna(0).astype(int)
    metrics["snapshot_date"] = snapshot_date

    df = metrics.reindex(columns=schema.columns).sort_values(["year_month", "speech_count", "votes_cast_count", "member_code"], ascending=[True, False, False, True]).copy()
    df = _dedupe_rows(df, primary_keys=schema.primary_key)

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
        "speech_metric_rows": int(len(speech_metrics)),
        "vote_metric_rows": int(len(vote_metrics)),
        "member_month_rows_before_limit": int(len(metrics)),
        "output_rows": int(len(df)),
        "year_month_values": sorted(df["year_month"].dropna().astype(str).unique().tolist()) if not df.empty else [],
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


def _speech_metrics(speeches: pd.DataFrame) -> pd.DataFrame:
    if speeches.empty or "speaker_member_code" not in speeches.columns:
        return pd.DataFrame(columns=["member_code", "year_month", "speech_count", "debate_day_count"])
    working = speeches.copy()
    working["member_code"] = working["speaker_member_code"].fillna("").astype(str).str.strip()
    working["debate_date_parsed"] = pd.to_datetime(working.get("debate_date"), errors="coerce")
    working = working[(working["member_code"] != "") & working["debate_date_parsed"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["member_code", "year_month", "speech_count", "debate_day_count"])
    working["year_month"] = working["debate_date_parsed"].dt.strftime("%Y-%m")
    return (
        working.groupby(["member_code", "year_month"], as_index=False)
        .agg(speech_count=("speech_id", "count"), debate_day_count=("debate_date_parsed", lambda values: values.dt.date.nunique()))
    )


def _vote_metrics(member_votes: pd.DataFrame) -> pd.DataFrame:
    if member_votes.empty or "member_code" not in member_votes.columns:
        return pd.DataFrame(columns=["member_code", "year_month", "votes_cast_count"])
    working = member_votes.copy()
    working["member_code"] = working["member_code"].fillna("").astype(str).str.strip()
    working["division_date_parsed"] = pd.to_datetime(working.get("division_date"), errors="coerce")
    working = working[(working["member_code"] != "") & working["division_date_parsed"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["member_code", "year_month", "votes_cast_count"])
    working["year_month"] = working["division_date_parsed"].dt.strftime("%Y-%m")
    return working.groupby(["member_code", "year_month"], as_index=False).agg(votes_cast_count=("member_vote_id", "count"))


def _member_month_grid(current_members: pd.DataFrame, speech_metrics: pd.DataFrame, vote_metrics: pd.DataFrame) -> pd.DataFrame:
    member_codes = set()
    months = set()
    if not current_members.empty and "member_code" in current_members.columns:
        member_codes.update(current_members["member_code"].dropna().astype(str).str.strip())
    for metrics in (speech_metrics, vote_metrics):
        if not metrics.empty:
            member_codes.update(metrics["member_code"].dropna().astype(str).str.strip())
            months.update(metrics["year_month"].dropna().astype(str).str.strip())
    member_codes.discard("")
    months.discard("")
    if not months:
        months = {datetime.now(timezone.utc).strftime("%Y-%m")}
    return pd.DataFrame([{"member_code": member_code, "year_month": month} for month in sorted(months) for member_code in sorted(member_codes)])


def _dedupe_rows(df: pd.DataFrame, *, primary_keys: list[str]) -> pd.DataFrame:
    if df.empty or not set(primary_keys).issubset(df.columns):
        return df
    return df.drop_duplicates(subset=primary_keys, keep="first")


def _nonblank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna("").astype(str).str.strip() != ""


def _numeric_populated(df: pd.DataFrame, columns: list[str]) -> bool:
    for column in columns:
        if column not in df.columns:
            return False
        if pd.to_numeric(df[column], errors="coerce").isna().any():
            return False
    return True


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or not set(pk).issubset(df.columns):
        pk_non_null = pk_unique = member_ok = month_ok = numeric_ok = False
    else:
        pk_non_null = bool(all(_nonblank(df[col]).all() for col in pk))
        pk_unique = bool(not df.duplicated(subset=pk).any())
        member_ok = bool(_nonblank(df.get("member_code")).all())
        month_ok = bool(_nonblank(df.get("year_month")).all())
        numeric_ok = _numeric_populated(df, ["speech_count", "debate_day_count", "votes_cast_count"])
    status = "pass" if all([row_count > 0, not missing_columns, pk_non_null, pk_unique, member_ok, month_ok, numeric_ok]) else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": pk,
        "primary_key_unique": pk_unique,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "required_columns_present", "status": "pass" if not missing_columns else "fail", "missing_columns": missing_columns},
            {"check_name": "primary_key_non_null", "status": "pass" if pk_non_null else "fail"},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "member_code_populated", "status": "pass" if member_ok else "fail"},
            {"check_name": "year_month_populated", "status": "pass" if month_ok else "fail"},
            {"check_name": "numeric_metrics_populated", "status": "pass" if numeric_ok else "fail"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
