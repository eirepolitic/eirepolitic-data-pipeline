"""Build side-by-side member photo URL enrichment outputs.

This module does not scrape new pages and does not overwrite existing legacy
photo URL keys. It reshapes the current legacy photo URL CSV into a unified
enrichment table plus a legacy-compatible adapter.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io_s3 import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    get_bytes,
    make_s3_client,
    object_exists,
    put_dataframe_csv,
    put_dataframe_parquet,
    put_json,
    put_text,
)
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, write_review_bundle

TABLE_NAME = "enrichment_member_photo_urls"
SOURCE_KEY_CANDIDATES = [
    "processed/members/member_photos/members_photo_urls.csv",
    "processed/members/members_photo_urls.csv",
]
TRIAL_CSV_KEY = "processed/oireachtas_unified/enrichment/media/member_photo_urls/member_photo_urls_trial.csv"
TRIAL_PARQUET_KEY = "processed/oireachtas_unified/enrichment/media/member_photo_urls/parquets/member_photo_urls_trial.parquet"
COMPAT_CSV_KEY = "processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv"
COMPAT_PARQUET_KEY = "processed/oireachtas_unified/compat/media/parquets/members_photo_urls_compat.parquet"


def build_enrichment_member_photo_urls(*, s3: Any, bucket: str, review_root: Path, row_limit: int = 0, sample_rows: int = 10) -> dict[str, Any]:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    source_key, source_df = _read_first_available(s3, bucket=bucket, keys=SOURCE_KEY_CANDIDATES)
    source_rows = int(len(source_df))
    if row_limit and row_limit > 0:
        source_df = source_df.head(row_limit).copy()

    trial_df = _build_trial_df(source_df, run_id=run_id, source_key=source_key)
    compat_df = _build_compat_df(trial_df)
    dq = _dq(trial_df, source_rows=source_rows, row_limit=row_limit)

    put_dataframe_csv(s3, bucket=bucket, key=TRIAL_CSV_KEY, df=trial_df)
    put_dataframe_parquet(s3, bucket=bucket, key=TRIAL_PARQUET_KEY, df=trial_df)
    put_dataframe_csv(s3, bucket=bucket, key=COMPAT_CSV_KEY, df=compat_df)
    put_dataframe_parquet(s3, bucket=bucket, key=COMPAT_PARQUET_KEY, df=compat_df)

    manifest_key = f"processed/oireachtas_unified/enrichment/manifests/{TABLE_NAME}/run_id={run_id}.json"
    review_sample_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/sample.csv"
    review_schema_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/schema.json"
    review_manifest_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/manifest.json"
    review_dq_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/dq.json"
    review_report_key = f"processed/oireachtas_unified/review/{TABLE_NAME}/latest/report.md"

    schema = {
        "table": TABLE_NAME,
        "primary_key": ["record_id"],
        "columns": list(trial_df.columns),
        "row_count": int(len(trial_df)),
    }
    manifest = {
        "table": TABLE_NAME,
        "mode": "enrichment_trial",
        "status": "success" if dq["dq_status"] != "fail" else "failed",
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "source_key": source_key,
        "source_rows": source_rows,
        "row_limit": int(row_limit or 0),
        "output_rows": int(len(trial_df)),
        "compat_rows": int(len(compat_df)),
        "primary_key": ["record_id"],
        "primary_key_unique": bool(not trial_df["record_id"].duplicated().any()) if len(trial_df) else False,
        "dq_status": dq["dq_status"],
        "photo_url_populated_count": dq["photo_url_populated_count"],
        "photo_url_missing_count": dq["photo_url_missing_count"],
        "s3_keys": {
            "trial_csv": TRIAL_CSV_KEY,
            "trial_parquet": TRIAL_PARQUET_KEY,
            "compat_csv": COMPAT_CSV_KEY,
            "compat_parquet": COMPAT_PARQUET_KEY,
            "manifest": manifest_key,
            "review_sample": review_sample_key,
            "review_schema": review_schema_key,
            "review_manifest": review_manifest_key,
            "review_dq": review_dq_key,
            "review_report": review_report_key,
        },
    }

    sample = trial_df.head(sample_rows)
    report = _report_markdown(manifest, dq)
    put_json(s3, bucket=bucket, key=manifest_key, payload=manifest)
    put_dataframe_csv(s3, bucket=bucket, key=review_sample_key, df=sample)
    put_json(s3, bucket=bucket, key=review_schema_key, payload=schema)
    put_json(s3, bucket=bucket, key=review_manifest_key, payload=manifest)
    put_json(s3, bucket=bucket, key=review_dq_key, payload=dq)
    put_text(s3, bucket=bucket, key=review_report_key, text=report)

    out_dir = write_review_bundle(
        table=TABLE_NAME,
        manifest=manifest,
        schema=schema,
        dq=dq,
        sample_rows=sample.to_dict(orient="records"),
        root=review_root,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return {"manifest": manifest, "dq": dq, "schema": schema, "rows": trial_df.to_dict(orient="records")}


def _read_first_available(s3: Any, *, bucket: str, keys: list[str]) -> tuple[str, pd.DataFrame]:
    for key in keys:
        if object_exists(s3, bucket=bucket, key=key):
            body = get_bytes(s3, bucket=bucket, key=key)
            return key, pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)
    raise RuntimeError(f"No source photo URL CSV found. Checked: {', '.join(keys)}")


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype="object")


def _build_trial_df(df: pd.DataFrame, *, run_id: str, source_key: str) -> pd.DataFrame:
    member_code = _col(df, "member_code", "memberCode")
    full_name = _col(df, "full_name", "member_name", "name")
    photo_url = _col(df, "photo_url", "image_url", "url")
    source_url = _col(df, "source_url", "profile_url", "uri")

    output = pd.DataFrame()
    output["member_code"] = member_code
    output["full_name"] = full_name
    output["photo_url"] = photo_url
    output["source_url"] = source_url
    output["record_id"] = [f"member_photo:{_stable_id(code, name, idx)}" for idx, (code, name) in enumerate(zip(member_code, full_name), start=1)]
    output["source_key"] = source_key
    output["source_system"] = "legacy_member_photo_url_output"
    output["source_hash"] = [_stable_hash([code, name, url]) for code, name, url in zip(member_code, full_name, photo_url)]
    output["retrieved_at_utc"] = ""
    output["review_status"] = "unreviewed"
    output["run_id"] = run_id
    return output[[
        "record_id",
        "member_code",
        "full_name",
        "photo_url",
        "source_url",
        "source_key",
        "source_system",
        "source_hash",
        "retrieved_at_utc",
        "review_status",
        "run_id",
    ]].sort_values(by=["full_name", "member_code", "record_id"], kind="stable")


def _build_compat_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    compat = pd.DataFrame()
    compat["member_code"] = trial_df["member_code"]
    compat["full_name"] = trial_df["full_name"]
    compat["photo_url"] = trial_df["photo_url"]
    return compat.sort_values(by=["full_name", "member_code"], kind="stable")


def _stable_id(member_code: str, full_name: str, idx: int) -> str:
    base = str(member_code or "").strip() or str(full_name or "").strip() or str(idx)
    return _stable_hash([base])[:24]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def _dq(df: pd.DataFrame, *, source_rows: int, row_limit: int) -> dict[str, Any]:
    row_count = int(len(df))
    record_id_unique = bool(row_count and not df["record_id"].duplicated().any())
    member_code_populated = bool(row_count and df["member_code"].fillna("").astype(str).str.strip().ne("").all())
    photo_populated = df["photo_url"].fillna("").astype(str).str.strip().ne("") if row_count else pd.Series([], dtype=bool)
    photo_url_populated_count = int(photo_populated.sum()) if row_count else 0
    photo_url_missing_count = int(row_count - photo_url_populated_count)
    expected_rows = min(source_rows, row_limit) if row_limit and row_limit > 0 else source_rows
    row_count_expected = row_count == expected_rows
    status = "pass" if row_count > 0 and record_id_unique and member_code_populated and row_count_expected else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "source_rows": int(source_rows),
        "row_limit": int(row_limit or 0),
        "expected_rows": int(expected_rows),
        "primary_key": ["record_id"],
        "primary_key_unique": record_id_unique,
        "member_code_populated": member_code_populated,
        "photo_url_populated_count": photo_url_populated_count,
        "photo_url_missing_count": photo_url_missing_count,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "record_id_unique", "status": "pass" if record_id_unique else "fail"},
            {"check_name": "member_code_populated", "status": "pass" if member_code_populated else "fail"},
            {"check_name": "photo_url_coverage", "status": "pass", "metric_value": photo_url_populated_count, "missing_count": photo_url_missing_count, "severity": "info"},
            {"check_name": "row_count_expected", "status": "pass" if row_count_expected else "fail", "metric_value": row_count},
        ],
    }


def _report_markdown(manifest: dict[str, Any], dq: dict[str, Any]) -> str:
    return "\n".join([
        "# Enrichment member photo URLs trial",
        "",
        f"- Status: `{manifest['status']}`",
        f"- DQ status: `{dq['dq_status']}`",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Source key: `{manifest['source_key']}`",
        f"- Source rows: `{manifest['source_rows']}`",
        f"- Row limit: `{manifest['row_limit']}`",
        f"- Trial rows: `{manifest['output_rows']}`",
        f"- Compat rows: `{manifest['compat_rows']}`",
        f"- Photo URLs populated: `{dq['photo_url_populated_count']}`",
        f"- Photo URLs missing: `{dq['photo_url_missing_count']}`",
        "",
        "## Outputs",
        "",
        f"- Trial CSV: `{TRIAL_CSV_KEY}`",
        f"- Trial parquet: `{TRIAL_PARQUET_KEY}`",
        f"- Compat CSV: `{COMPAT_CSV_KEY}`",
        f"- Compat parquet: `{COMPAT_PARQUET_KEY}`",
        "",
        "This trial does not overwrite legacy photo URL keys.",
        "",
    ])


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    row_limit = int(os.getenv("ROW_LIMIT", "0") or "0")
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10") or "10")
    s3 = make_s3_client(region_name=region)
    result = build_enrichment_member_photo_urls(s3=s3, bucket=bucket, review_root=review_root, row_limit=row_limit, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "dq_status": result["dq"].get("dq_status"), "run_id": result["manifest"].get("run_id"), "rows": result["manifest"].get("output_rows")}, indent=2, sort_keys=True))
    return 0 if result["dq"].get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
