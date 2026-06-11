"""Builder for the `gold_current_members` table."""

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

TABLE_NAME = "gold_current_members"
SILVER_LATEST_PREFIX = "processed/oireachtas_unified/latest/csv"


@dataclass(frozen=True)
class TableBuildResult:
    table: str
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]
    s3_keys: dict[str, str]


def build_gold_current_members(
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
    """Read latest silver member tables from S3 and build a current roster mart."""
    del client, chamber, house_no
    started_at = utc_now_iso()
    run_id = _run_id(TABLE_NAME)
    snapshot_date = started_at[:10]

    input_keys = {
        "members": f"{SILVER_LATEST_PREFIX}/silver_members.csv",
        "memberships": f"{SILVER_LATEST_PREFIX}/silver_member_memberships.csv",
        "parties": f"{SILVER_LATEST_PREFIX}/silver_member_parties.csv",
        "constituencies": f"{SILVER_LATEST_PREFIX}/silver_member_constituencies.csv",
        "offices": f"{SILVER_LATEST_PREFIX}/silver_member_offices.csv",
    }
    inputs = {name: _read_latest_csv(s3, bucket=bucket, key=key) for name, key in input_keys.items()}

    members = inputs["members"]
    memberships = inputs["memberships"]
    parties = inputs["parties"]
    constituencies = inputs["constituencies"]
    offices = inputs["offices"]

    current_memberships = _select_current_or_latest(
        memberships,
        group_key="member_code",
        current_col="is_current",
        start_col="membership_start",
        end_col="membership_end",
    )
    current_parties = _select_current_or_latest(
        parties,
        group_key="member_code",
        current_col="is_current",
        start_col="party_start",
        end_col="party_end",
    )
    current_constituencies = _select_current_or_latest(
        constituencies,
        group_key="member_code",
        current_col="is_current",
        start_col="represent_start",
        end_col="represent_end",
    )
    current_offices = _aggregate_current_offices(offices)

    roster = members.copy()
    if not current_memberships.empty:
        roster = roster.merge(
            current_memberships[["member_code", "house_no", "membership_id"]],
            on="member_code",
            how="left",
            suffixes=("", "_membership"),
        )
    else:
        roster["house_no"] = None
        roster["membership_id"] = None

    if not current_parties.empty:
        roster = roster.merge(current_parties[["member_code", "party_name"]], on="member_code", how="left", suffixes=("", "_party"))
    else:
        roster["party_name"] = None

    if not current_constituencies.empty:
        roster = roster.merge(
            current_constituencies[["member_code", "constituency_name"]],
            on="member_code",
            how="left",
            suffixes=("", "_constituency"),
        )
    else:
        roster["constituency_name"] = None

    if not current_offices.empty:
        roster = roster.merge(current_offices[["member_code", "office_name"]], on="member_code", how="left")
    else:
        roster["office_name"] = None

    roster["party_name"] = _coalesce_series(roster.get("party_name"), roster.get("latest_party_name"))
    roster["constituency_name"] = _coalesce_series(roster.get("constituency_name"), roster.get("latest_constituency_name"))
    roster["house_no"] = _coalesce_series(roster.get("house_no"), roster.get("latest_house_no"))
    roster["office_name"] = roster.get("office_name").fillna("") if "office_name" in roster else ""
    roster["snapshot_date"] = snapshot_date

    current_member_mask = _truthy(roster.get("is_current_member"))
    if current_member_mask.any():
        roster = roster[current_member_mask].copy()
    elif not current_memberships.empty:
        roster = roster[roster["member_code"].isin(set(current_memberships["member_code"].dropna().astype(str)))].copy()

    df = roster.reindex(columns=schema.columns).head(max(1, limit)).copy()
    df = _dedupe_rows(df, primary_key="member_code")

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
        "current_membership_rows": int(len(current_memberships)),
        "current_party_rows": int(len(current_parties)),
        "current_constituency_rows": int(len(current_constituencies)),
        "current_office_member_rows": int(len(current_offices)),
        "output_rows": int(len(df)),
        "party_populated_rows": int(_nonblank(df.get("party_name")).sum()) if not df.empty else 0,
        "constituency_populated_rows": int(_nonblank(df.get("constituency_name")).sum()) if not df.empty else 0,
        "office_populated_rows": int(_nonblank(df.get("office_name")).sum()) if not df.empty else 0,
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


def _select_current_or_latest(df: pd.DataFrame, *, group_key: str, current_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    if df.empty or group_key not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working["__current"] = _truthy(working.get(current_col))
    working["__start"] = pd.to_datetime(working.get(start_col), errors="coerce")
    working["__end"] = pd.to_datetime(working.get(end_col), errors="coerce").fillna(pd.Timestamp.max)
    working["__rank_current"] = working["__current"].astype(int)
    working = working.sort_values([group_key, "__rank_current", "__end", "__start"], ascending=[True, False, False, False])
    return working.drop_duplicates(subset=[group_key], keep="first").drop(columns=["__current", "__start", "__end", "__rank_current"])


def _aggregate_current_offices(offices: pd.DataFrame) -> pd.DataFrame:
    if offices.empty or "member_code" not in offices.columns:
        return pd.DataFrame(columns=["member_code", "office_name"])
    selected = _select_current_or_latest(
        offices,
        group_key="member_code",
        current_col="is_current",
        start_col="office_start",
        end_col="office_end",
    )
    if selected.empty:
        return pd.DataFrame(columns=["member_code", "office_name"])
    grouped = (
        selected[selected["office_name"].astype(str).str.strip() != ""]
        .groupby("member_code", as_index=False)["office_name"]
        .agg(lambda values: "; ".join(sorted(set(str(v).strip() for v in values if str(v).strip()))))
    )
    return grouped


def _coalesce_series(primary: pd.Series | None, fallback: pd.Series | None) -> pd.Series:
    if primary is None and fallback is None:
        return pd.Series(dtype=str)
    if primary is None:
        return fallback.fillna("")
    if fallback is None:
        return primary.fillna("")
    p = primary.fillna("").astype(str)
    f = fallback.fillna("").astype(str)
    return p.where(p.str.strip() != "", f)


def _truthy(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _nonblank(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.fillna("").astype(str).str.strip() != ""


def _dedupe_rows(df: pd.DataFrame, *, primary_key: str) -> pd.DataFrame:
    if df.empty or primary_key not in df.columns:
        return df
    return df.drop_duplicates(subset=[primary_key], keep="first")


def _dq_results(df: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    pk = schema.primary_key[0]
    missing_columns = sorted(set(schema.columns) - set(df.columns))
    row_count = int(len(df))
    if row_count == 0 or pk not in df.columns:
        non_null_pk = unique_pk = name_ok = party_ok = constituency_ok = house_ok = False
    else:
        non_null_pk = bool(_nonblank(df[pk]).all())
        unique_pk = bool(not df[pk].duplicated().any())
        name_ok = bool(_nonblank(df.get("full_name")).all())
        party_ok = bool(_nonblank(df.get("party_name")).all())
        constituency_ok = bool(_nonblank(df.get("constituency_name")).all())
        house_ok = bool(_nonblank(df.get("house_no")).all())
    status = "pass" if all([row_count > 0, not missing_columns, non_null_pk, unique_pk, name_ok, party_ok, constituency_ok, house_ok]) else "fail"
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
            {"check_name": "full_name_populated", "status": "pass" if name_ok else "fail"},
            {"check_name": "party_name_populated", "status": "pass" if party_ok else "fail"},
            {"check_name": "constituency_name_populated", "status": "pass" if constituency_ok else "fail"},
            {"check_name": "house_no_populated", "status": "pass" if house_ok else "fail"},
            {"check_name": "office_name_optional", "status": "pass", "message": "office_name is optional for members without current office records"},
        ],
    }


def _run_id(table: str) -> str:
    return f"{table}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
