"""Build side-by-side constituency image enrichment outputs.

This module does not create, download, or overwrite image files. It reshapes the
existing legacy constituency image index into a unified enrichment table plus a
legacy-compatible adapter.
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

TABLE_NAME = "enrichment_constituency_images"
SOURCE_KEY = "processed/constituencies/constituency_images.csv"
TRIAL_CSV_KEY = "processed/oireachtas_unified/enrichment/media/constituency_images/constituency_images_trial.csv"
TRIAL_PARQUET_KEY = "processed/oireachtas_unified/enrichment/media/constituency_images/parquets/constituency_images_trial.parquet"
COMPAT_CSV_KEY = "processed/oireachtas_unified/compat/media/constituency_images_compat.csv"
COMPAT_PARQUET_KEY = "processed/oireachtas_unified/compat/media/parquets/constituency_images_compat.parquet"


def build_enrichment_constituency_images(*, s3: Any, bucket: str, review_root: Path, row_limit: int = 0, sample_rows: int = 10) -> dict[str, Any]:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    source_df = _read_csv(s3, bucket=bucket, key=SOURCE_KEY)
    source_rows = int(len(source_df))
    if row_limit and row_limit > 0:
        source_df = source_df.head(row_limit).copy()

    trial_df = _build_trial_df(source_df, run_id=run_id, source_key=SOURCE_KEY)
    compat_df = _build_compat_df(trial_df)
    dq = _dq(trial_df, source_rows=source_rows, row_limit=row_limit)

    put_dataframe_csv(s3, bucket=bucket, key=TRIAL_CSV_KEY, df=trial_df)
    if not trial_df.empty:
        put_dataframe_parquet(s3, bucket=bucket, key=TRIAL_PARQUET_KEY, df=trial_df)
    put_dataframe_csv(s3, bucket=bucket, key=COMPAT_CSV_KEY, df=compat_df)
    if not compat_df.empty:
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
        "source_key": SOURCE_KEY,
        "source_rows": source_rows,
        "row_limit": int(row_limit or 0),
        "output_rows": int(len(trial_df)),
        "compat_rows": int(len(compat_df)),
        "primary_key": ["record_id"],
        "primary_key_unique": bool(not trial_df["record_id"].duplicated().any()) if len(trial_df) else False,
        "dq_status": dq["dq_status"],
        "image_locator_populated_count": dq["image_locator_populated_count"],
        "image_locator_missing_count": dq["image_locator_missing_count"],
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


def _read_csv(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    if not object_exists(s3, bucket=bucket, key=key):
        raise RuntimeError(f"Source constituency image index not found: {key}")
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype="object")


def _build_trial_df(df: pd.DataFrame, *, run_id: str, source_key: str) -> pd.DataFrame:
    filename = _col(df, "filename", "file_name")
    image_key = _col(df, "s3_key", "image_key", "key")
    image_url = _col(df, "url", "image_url")
    constituency = _col(df, "constituency", "constituency_name")
    inferred_constituency = [c or _constituency_from_filename(f) for c, f in zip(constituency, filename)]

    output = pd.DataFrame()
    output["record_id"] = [f"constituency_image:{_stable_hash([key, url, fname])}" for key, url, fname in zip(image_key, image_url, filename)]
    output["constituency"] = inferred_constituency
    output["filename"] = filename
    output["image_key"] = image_key
    output["image_url"] = image_url
    output["media_type"] = [f"image/{_extension(fname) or 'unknown'}" for fname in filename]
    output["source_key"] = source_key
    output["source_system"] = "legacy_constituency_image_index"
    output["source_hash"] = [_stable_hash([fname, key, url]) for fname, key, url in zip(filename, image_key, image_url)]
    output["retrieved_at_utc"] = ""
    output["review_status"] = "unreviewed"
    output["run_id"] = run_id
    return output.sort_values(by=["constituency", "filename", "record_id"], kind="stable")


def _build_compat_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    compat = pd.DataFrame()
    compat["filename"] = trial_df["filename"]
    compat["s3_key"] = trial_df["image_key"]
    compat["url"] = trial_df["image_url"]
    return compat.sort_values(by=["filename", "s3_key"], kind="stable")


def _constituency_from_filename(filename: str) -> str:
    stem = str(filename or "").rsplit(".", 1)[0]
    for suffix in ("_cover", "-cover", "_image", "-image"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    text = stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def _extension(filename: str) -> str:
    if "." not in str(filename or ""):
        return ""
    ext = str(filename).rsplit(".", 1)[-1].lower()
    if ext == "jpg":
        return "jpeg"
    return ext


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def _dq(df: pd.DataFrame, *, source_rows: int, row_limit: int) -> dict[str, Any]:
    row_count = int(len(df))
    record_id_unique = bool(row_count and not df["record_id"].duplicated().any())
    constituency_populated = bool(row_count and df["constituency"].fillna("").astype(str).str.strip().ne("").all())
    image_key_populated = df["image_key"].fillna("").astype(str).str.strip().ne("") if row_count else pd.Series([], dtype=bool)
    image_url_populated = df["image_url"].fillna("").astype(str).str.strip().ne("") if row_count else pd.Series([], dtype=bool)
    image_locator_populated = image_key_populated | image_url_populated if row_count else pd.Series([], dtype=bool)
    image_locator_populated_count = int(image_locator_populated.sum()) if row_count else 0
    image_locator_missing_count = int(row_count - image_locator_populated_count)
    expected_rows = min(source_rows, row_limit) if row_limit and row_limit > 0 else source_rows
    row_count_expected = row_count == expected_rows
    status = "pass" if row_count > 0 and record_id_unique and constituency_populated and image_locator_missing_count == 0 and row_count_expected else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "source_rows": int(source_rows),
        "row_limit": int(row_limit or 0),
        "expected_rows": int(expected_rows),
        "primary_key": ["record_id"],
        "primary_key_unique": record_id_unique,
        "constituency_populated": constituency_populated,
        "image_key_populated_count": int(image_key_populated.sum()) if row_count else 0,
        "image_url_populated_count": int(image_url_populated.sum()) if row_count else 0,
        "image_locator_populated_count": image_locator_populated_count,
        "image_locator_missing_count": image_locator_missing_count,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "record_id_unique", "status": "pass" if record_id_unique else "fail"},
            {"check_name": "constituency_populated", "status": "pass" if constituency_populated else "fail"},
            {"check_name": "image_locator_populated", "status": "pass" if image_locator_missing_count == 0 else "fail", "metric_value": image_locator_populated_count, "missing_count": image_locator_missing_count},
            {"check_name": "row_count_expected", "status": "pass" if row_count_expected else "fail", "metric_value": row_count},
        ],
    }


def _report_markdown(manifest: dict[str, Any], dq: dict[str, Any]) -> str:
    return "\n".join([
        "# Enrichment constituency images trial",
        "",
        f"- Status: `{manifest['status']}`",
        f"- DQ status: `{dq['dq_status']}`",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Source key: `{manifest['source_key']}`",
        f"- Source rows: `{manifest['source_rows']}`",
        f"- Row limit: `{manifest['row_limit']}`",
        f"- Trial rows: `{manifest['output_rows']}`",
        f"- Compat rows: `{manifest['compat_rows']}`",
        f"- Image locators populated: `{dq['image_locator_populated_count']}`",
        f"- Image locators missing: `{dq['image_locator_missing_count']}`",
        "",
        "## Outputs",
        "",
        f"- Trial CSV: `{TRIAL_CSV_KEY}`",
        f"- Trial parquet: `{TRIAL_PARQUET_KEY}`",
        f"- Compat CSV: `{COMPAT_CSV_KEY}`",
        f"- Compat parquet: `{COMPAT_PARQUET_KEY}`",
        "",
        "This trial does not overwrite legacy constituency image keys.",
        "",
    ])


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    row_limit = int(os.getenv("ROW_LIMIT", "0") or "0")
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10") or "10")
    s3 = make_s3_client(region_name=region)
    result = build_enrichment_constituency_images(s3=s3, bucket=bucket, review_root=review_root, row_limit=row_limit, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "dq_status": result["dq"].get("dq_status"), "run_id": result["manifest"].get("run_id"), "rows": result["manifest"].get("output_rows")}, indent=2, sort_keys=True))
    return 0 if result["dq"].get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
