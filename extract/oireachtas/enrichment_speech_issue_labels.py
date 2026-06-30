"""Build side-by-side speech issue label enrichment outputs.

This module does not call OpenAI and does not overwrite the existing legacy
classified debate output. It reshapes the current classified debate CSV into a
unified enrichment table plus a legacy-compatible classified debate adapter.
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
    put_dataframe_csv,
    put_dataframe_parquet,
    put_json,
    put_text,
)
from .normalize import utc_now_iso
from .review import REVIEW_ROOT, table_review_dir, write_review_bundle

TABLE_NAME = "enrichment_speech_issue_labels"
LEGACY_CLASSIFIED_KEY = "processed/debates/debate_speeches_classified.csv"
TRIAL_CSV_KEY = "processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv"
TRIAL_PARQUET_KEY = "processed/oireachtas_unified/enrichment/speech_issue_labels/parquets/speech_issue_labels_2025_trial.parquet"
COMPAT_CSV_KEY = "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
COMPAT_PARQUET_KEY = "processed/oireachtas_unified/compat/debates/parquets/debate_speeches_classified_compat.parquet"

ISSUE_CATEGORIES = {
    "Macroeconomics",
    "Civil Rights, Minority Issues and Civil Liberties",
    "Health",
    "Agriculture",
    "Labor, Employment and Immigration",
    "Education",
    "Environment",
    "Energy",
    "Transportation",
    "Law/Crime and Family Issues",
    "Social Welfare",
    "Housing and Community Development",
    "Banking/Finance and Domestic Commerce",
    "Defense",
    "Space, Science, and Technology",
    "Foreign Trade",
    "International Affairs and Foreign Aid",
    "Government Operations",
    "Public Lands and Water Management",
    "State and Local Government Administration",
    "Culture and Arts",
    "Sports and Recreation",
    "Other/Miscellaneous",
    "Domestic Terrorism",
    "NONE",
}


def build_enrichment_speech_issue_labels(*, s3: Any, bucket: str, review_root: Path, row_limit: int = 50, sample_rows: int = 10) -> dict[str, Any]:
    started_at = utc_now_iso()
    run_id = f"{TABLE_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    legacy_df = _read_csv(s3, bucket=bucket, key=LEGACY_CLASSIFIED_KEY)
    source_rows = int(len(legacy_df))
    if row_limit and row_limit > 0:
        legacy_df = legacy_df.head(row_limit).copy()

    trial_df = _build_trial_df(legacy_df, run_id=run_id, source_key=LEGACY_CLASSIFIED_KEY)
    compat_df = _build_compat_df(legacy_df, trial_df)
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
        "source_key": LEGACY_CLASSIFIED_KEY,
        "source_rows": source_rows,
        "row_limit": int(row_limit or 0),
        "output_rows": int(len(trial_df)),
        "compat_rows": int(len(compat_df)),
        "primary_key": ["record_id"],
        "primary_key_unique": bool(not trial_df["record_id"].duplicated().any()) if len(trial_df) else False,
        "dq_status": dq["dq_status"],
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
    body = get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype="object")


def _build_trial_df(df: pd.DataFrame, *, run_id: str, source_key: str) -> pd.DataFrame:
    output = pd.DataFrame()
    speech_text = _col(df, "Speech Text", "speech_text")
    output["speech_id"] = _col(df, "speech_id")
    missing_id = output["speech_id"].str.strip().eq("")
    if missing_id.any():
        output.loc[missing_id, "speech_id"] = df[missing_id].apply(_make_speech_id, axis=1)

    output["record_id"] = [f"{sid}:{idx}" for idx, sid in enumerate(output["speech_id"].fillna("").astype(str), start=1)]
    output["member_code"] = _col(df, "member_code", "memberCode")
    output["speaker_name"] = _col(df, "Speaker Name", "speaker_name", "member_name")
    output["debate_date"] = _col(df, "Debate Date", "date", "debate_date")
    output["speech_order"] = _col(df, "Speech Order", "speech_order")
    output["source_speech_text_hash"] = speech_text.map(_text_hash)
    output["issue_label"] = _col(df, "PoliticalIssues", "political_issues", "issue_label").map(_clean_label)
    output["issue_label_source"] = "legacy_classified_debate_output"
    output["model_name"] = "legacy_unknown"
    output["classification_status"] = output["issue_label"].map(lambda value: "unclassified" if value == "" else "classified")
    output["review_status"] = "unreviewed"
    output["classified_at_utc"] = ""
    output["source_key"] = source_key
    output["run_id"] = run_id
    return output.sort_values(by=["debate_date", "speech_order", "speaker_name", "record_id"], kind="stable")


def _build_compat_df(source_df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    compat = source_df.copy()
    if "speech_id" not in compat.columns:
        compat["speech_id"] = trial_df["speech_id"].values
    compat["PoliticalIssues"] = trial_df["issue_label"].values
    if "Speaker Name" not in compat.columns:
        compat["Speaker Name"] = trial_df["speaker_name"].values
    if "Debate Date" not in compat.columns:
        compat["Debate Date"] = trial_df["debate_date"].values
    return compat


def _make_speech_id(row: pd.Series) -> str:
    parts = [
        str(row.get("Debate Date", row.get("date", ""))).strip(),
        str(row.get("Speaker Name", row.get("speaker_name", ""))).strip(),
        str(row.get("Speech Order", row.get("speech_order", ""))).strip(),
        str(row.get("Speech Text", row.get("speech_text", ""))).strip(),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:24]


def _text_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()[:24]


def _clean_label(value: str) -> str:
    label = str(value or "").strip()
    for category in ISSUE_CATEGORIES:
        if category.lower() == label.lower():
            return category
    return label


def _dq(df: pd.DataFrame, *, source_rows: int, row_limit: int) -> dict[str, Any]:
    row_count = int(len(df))
    record_id_unique = bool(row_count and not df["record_id"].duplicated().any())
    speech_id_populated = bool(row_count and df["speech_id"].fillna("").astype(str).str.strip().ne("").all())
    invalid_labels = int((~df["issue_label"].isin(ISSUE_CATEGORIES | {""})).sum()) if row_count else 0
    expected_rows = min(source_rows, row_limit) if row_limit and row_limit > 0 else source_rows
    row_count_expected = row_count == expected_rows
    status = "pass" if row_count > 0 and record_id_unique and speech_id_populated and invalid_labels == 0 and row_count_expected else "fail"
    return {
        "table": TABLE_NAME,
        "dq_status": status,
        "row_count": row_count,
        "source_rows": int(source_rows),
        "row_limit": int(row_limit or 0),
        "expected_rows": int(expected_rows),
        "primary_key": ["record_id"],
        "primary_key_unique": record_id_unique,
        "speech_id_populated": speech_id_populated,
        "invalid_issue_label_count": invalid_labels,
        "checks": [
            {"check_name": "row_count_gt_zero", "status": "pass" if row_count > 0 else "fail", "metric_value": row_count},
            {"check_name": "record_id_unique", "status": "pass" if record_id_unique else "fail"},
            {"check_name": "speech_id_populated", "status": "pass" if speech_id_populated else "fail"},
            {"check_name": "approved_issue_labels", "status": "pass" if invalid_labels == 0 else "fail", "metric_value": invalid_labels},
            {"check_name": "row_count_expected", "status": "pass" if row_count_expected else "fail", "metric_value": row_count},
        ],
    }


def _report_markdown(manifest: dict[str, Any], dq: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Enrichment speech issue labels trial",
            "",
            f"- Status: `{manifest['status']}`",
            f"- DQ status: `{dq['dq_status']}`",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Source key: `{manifest['source_key']}`",
            f"- Source rows: `{manifest['source_rows']}`",
            f"- Row limit: `{manifest['row_limit']}`",
            f"- Trial rows: `{manifest['output_rows']}`",
            f"- Compat rows: `{manifest['compat_rows']}`",
            "",
            "## Outputs",
            "",
            f"- Trial CSV: `{TRIAL_CSV_KEY}`",
            f"- Trial parquet: `{TRIAL_PARQUET_KEY}`",
            f"- Compat CSV: `{COMPAT_CSV_KEY}`",
            f"- Compat parquet: `{COMPAT_PARQUET_KEY}`",
            "",
            "This trial does not overwrite `processed/debates/debate_speeches_classified.csv`.",
            "",
        ]
    )


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    review_root = Path(os.getenv("REVIEW_OUTPUT_ROOT", str(REVIEW_ROOT)))
    row_limit = int(os.getenv("ROW_LIMIT", "50") or "50")
    sample_rows = int(os.getenv("SAMPLE_ROWS", "10") or "10")
    s3 = make_s3_client(region_name=region)
    result = build_enrichment_speech_issue_labels(s3=s3, bucket=bucket, review_root=review_root, row_limit=row_limit, sample_rows=sample_rows)
    print(json.dumps({"table": TABLE_NAME, "dq_status": result["dq"].get("dq_status"), "run_id": result["manifest"].get("run_id"), "rows": result["manifest"].get("output_rows")}, indent=2, sort_keys=True))
    return 0 if result["dq"].get("dq_status") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
