"""Build non-destructive downstream compatibility CSVs from unified Oireachtas outputs."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, get_bytes, make_s3_client, put_dataframe_csv, put_json
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, write_review_bundle

TABLE_NAME = "compat_downstream_adapters"

SOURCE_CURRENT_MEMBERS = "processed/oireachtas_unified/latest/csv/gold_current_members.csv"
SOURCE_MEMBER_VOTES = "processed/oireachtas_unified/latest/csv/silver_member_votes.csv"
OUTPUT_MEMBERS_COMPAT = "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
OUTPUT_MEMBER_VOTES_COMPAT = "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"


@dataclass(frozen=True)
class CompatResult:
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    schema: dict[str, Any]
    dq: dict[str, Any]


def build_downstream_compat_adapters(*, s3: Any, bucket: str, review_root: Path, sample_rows: int = 10) -> CompatResult:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_date = started_at[:10]

    summary_rows: list[dict[str, Any]] = []
    members_df = _read_csv(s3, bucket=bucket, key=SOURCE_CURRENT_MEMBERS)
    members_compat = _build_members_compat(members_df)
    put_dataframe_csv(s3, bucket=bucket, key=OUTPUT_MEMBERS_COMPAT, df=members_compat)
    summary_rows.append(_summary_row("members_roster", SOURCE_CURRENT_MEMBERS, OUTPUT_MEMBERS_COMPAT, members_df, members_compat, "member_code"))

    votes_df = _read_csv(s3, bucket=bucket, key=SOURCE_MEMBER_VOTES)
    votes_compat = _build_member_votes_compat(votes_df)
    put_dataframe_csv(s3, bucket=bucket, key=OUTPUT_MEMBER_VOTES_COMPAT, df=votes_compat)
    summary_rows.append(_summary_row("member_votes", SOURCE_MEMBER_VOTES, OUTPUT_MEMBER_VOTES_COMPAT, votes_df, votes_compat, "memberCode"))

    summary_df = pd.DataFrame(summary_rows)
    dq = _dq(summary_df)
    schema = {"table": TABLE_NAME, "primary_key": ["adapter_name"], "columns": list(summary_df.columns), "row_count": int(len(summary_df))}
    manifest_key = f"processed/oireachtas_unified/compat/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"
    manifest = {
        "table": TABLE_NAME,
        "mode": "compat",
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "snapshot_date": snapshot_date,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "output_rows": int(len(summary_df)),
        "primary_key": ["adapter_name"],
        "primary_key_unique": bool(not summary_df["adapter_name"].duplicated().any()),
        "dq_status": dq["dq_status"],
        "s3_keys": {
            "members_compat_csv": OUTPUT_MEMBERS_COMPAT,
            "member_votes_compat_csv": OUTPUT_MEMBER_VOTES_COMPAT,
            "manifest": manifest_key,
            "review_sample": review_sample_key,
            "review_schema": review_schema_key,
            "review_manifest": review_manifest_key,
        },
    }

    put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
    put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=summary_df.head(sample_rows))
    put_json(s3, bucket=bucket, key=review_schema_key, payload=schema)
    put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    write_review_bundle(table=TABLE_NAME, manifest=manifest, schema=schema, dq=dq, sample_rows=summary_df.head(sample_rows).to_dict(orient="records"), root=review_root)
    return CompatResult(rows=summary_df.to_dict(orient="records"), manifest=manifest, schema=schema, dq=dq)


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _build_members_compat(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["member_code"] = _col(df, "member_code")
    output["full_name"] = _col(df, "full_name")
    output["constituency"] = _col(df, "constituency_name")
    output["party"] = _col(df, "party_name")
    output["house_no"] = _col(df, "house_no")
    output["source"] = "oireachtas_unified"
    output["snapshot_date"] = _col(df, "snapshot_date")
    return output.sort_values(by=["full_name", "member_code"], kind="stable")


def _build_member_votes_compat(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["memberCode"] = _col(df, "member_code")
    output["member_name"] = _col(df, "member_name")
    output["unique_vote_id"] = _col(df, "division_id").where(_col(df, "division_id") != "", _col(df, "vote_id"))
    output["date"] = _col(df, "division_date")
    output["vote"] = _col(df, "vote_label")
    output["party"] = _col(df, "party_name_at_vote")
    output["constituency"] = _col(df, "constituency_name_at_vote")
    output["source"] = "oireachtas_unified"
    output["snapshot_date"] = _col(df, "snapshot_date")
    return output.sort_values(by=["date", "unique_vote_id", "memberCode"], kind="stable")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype="object")


def _summary_row(adapter_name: str, source_key: str, output_key: str, source_df: pd.DataFrame, output_df: pd.DataFrame, pk_column: str) -> dict[str, Any]:
    pk_populated = bool(pk_column in output_df.columns and output_df[pk_column].fillna("").astype(str).str.strip().ne("").all()) if len(output_df) else False
    return {
        "adapter_name": adapter_name,
        "status": "pass" if len(output_df) > 0 and pk_populated else "fail",
        "source_key": source_key,
        "output_key": output_key,
        "source_rows": int(len(source_df)),
        "output_rows": int(len(output_df)),
        "source_columns": int(len(source_df.columns)),
        "output_columns": int(len(output_df.columns)),
        "primary_key_column": pk_column,
        "primary_key_populated": str(pk_populated).lower(),
    }


def _dq(df: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(df))
    pk_unique = bool("adapter_name" in df.columns and not df["adapter_name"].duplicated().any())
    all_pass = bool("status" in df.columns and df["status"].eq("pass").all())
    status = "pass" if row_count > 0 and pk_unique and all_pass else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "primary_key": ["adapter_name"],
        "primary_key_unique": pk_unique,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "primary_key_unique", "status": "pass" if pk_unique else "fail"},
            {"check_name": "all_adapters_pass", "status": "pass" if all_pass else "fail"},
        ],
    }


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10") or "10")
    s3 = make_s3_client(region_name=region)
    result = build_downstream_compat_adapters(s3=s3, bucket=bucket, review_root=review_root, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "rows": len(result.rows), "dq_status": result.dq.get("dq_status"), "run_id": result.manifest.get("run_id")}, indent=2, sort_keys=True))
    return 0 if result.dq.get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
