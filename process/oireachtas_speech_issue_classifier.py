from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

from extract.oireachtas.io_s3 import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    get_bytes,
    make_s3_client,
    put_dataframe_csv,
    put_dataframe_parquet,
    put_json,
)

SILVER_SPEECHES_KEY = "processed/oireachtas_unified/latest/csv/silver_speeches.csv"
LEGACY_CLASSIFIED_KEY = "processed/debates/debate_speeches_classified.csv"
ENRICHMENT_CSV_KEY = "processed/oireachtas_unified/enrichment/speech_issue_labels/latest/speech_issue_labels.csv"
ENRICHMENT_PARQUET_KEY = "processed/oireachtas_unified/enrichment/speech_issue_labels/latest/speech_issue_labels.parquet"
COMPAT_CSV_KEY = "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
COMPAT_PARQUET_KEY = "processed/oireachtas_unified/compat/debates/parquets/debate_speeches_classified_compat.parquet"
MANIFEST_KEY = "processed/oireachtas_unified/enrichment/speech_issue_labels/latest/manifest.json"

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_VERBOSITY = "low"
MIN_CLASSIFY_WORDS = 20

ISSUE_CATEGORIES = [
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
]
ISSUE_CATEGORY_SET = set(ISSUE_CATEGORIES)


@dataclass(frozen=True)
class ClassificationPlan:
    rows: pd.DataFrame
    stats: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()[:24]


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_name(value: Any) -> str:
    return " ".join(normalize_text(value).lower().split())


def normalize_date(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def normalize_order(value: Any) -> str:
    text = normalize_text(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def canonicalize_label(value: Any) -> str:
    text = normalize_text(value)
    for category in ISSUE_CATEGORIES:
        if category.lower() == text.lower():
            return category
    return ""


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)


def read_s3_csv(s3: Any, *, bucket: str, key: str, optional: bool = False) -> pd.DataFrame:
    try:
        return _read_csv_bytes(get_bytes(s3, bucket=bucket, key=key))
    except Exception:
        if optional:
            return pd.DataFrame()
        raise


def build_legacy_lookup(legacy: pd.DataFrame) -> tuple[dict[tuple[str, str, str, str], str], dict[str, int]]:
    if legacy.empty:
        return {}, {"legacy_rows": 0, "legacy_valid_labels": 0, "legacy_ambiguous_keys": 0}

    working = pd.DataFrame(index=legacy.index)
    speech_text = _col(legacy, "Speech Text", "speech_text")
    working["text_hash"] = speech_text.map(text_hash)
    working["debate_date"] = _col(legacy, "Debate Date", "debate_date", "date").map(normalize_date)
    working["speech_order"] = _col(legacy, "Speech Order", "speech_order").map(normalize_order)
    working["speaker_name"] = _col(legacy, "Speaker Name", "speaker_name", "member_name").map(normalize_name)
    working["issue_label"] = _col(legacy, "PoliticalIssues", "political_issues", "issue_label").map(canonicalize_label)
    working = working[working["issue_label"] != ""].copy()
    working["key"] = list(zip(working["debate_date"], working["speech_order"], working["speaker_name"], working["text_hash"]))

    lookup: dict[tuple[str, str, str, str], str] = {}
    ambiguous = 0
    for key, group in working.groupby("key", sort=False):
        labels = sorted(set(group["issue_label"].tolist()))
        if len(labels) == 1:
            lookup[key] = labels[0]
        else:
            ambiguous += 1
    return lookup, {
        "legacy_rows": int(len(legacy)),
        "legacy_valid_labels": int(len(working)),
        "legacy_ambiguous_keys": int(ambiguous),
    }


def build_existing_lookup(existing: pd.DataFrame) -> dict[str, tuple[str, str, str, str]]:
    lookup: dict[str, tuple[str, str, str, str]] = {}
    if existing.empty or "speech_id" not in existing.columns:
        return lookup
    for _, row in existing.iterrows():
        speech_id = normalize_text(row.get("speech_id"))
        label = canonicalize_label(row.get("issue_label"))
        source_hash = normalize_text(row.get("source_speech_text_hash"))
        status = normalize_text(row.get("classification_status"))
        model_name = normalize_text(row.get("model_name"))
        source = normalize_text(row.get("issue_label_source"))
        if speech_id and label and source_hash:
            lookup[speech_id] = (source_hash, label, status or "classified", model_name or source)
    return lookup


def prepare_classification_plan(
    silver: pd.DataFrame,
    *,
    existing: pd.DataFrame | None = None,
    legacy: pd.DataFrame | None = None,
    min_words: int = MIN_CLASSIFY_WORDS,
) -> ClassificationPlan:
    existing = existing if existing is not None else pd.DataFrame()
    legacy = legacy if legacy is not None else pd.DataFrame()

    required = {"speech_id", "speech_text", "speech_text_hash", "debate_date", "speech_order", "speaker_name"}
    missing = sorted(required - set(silver.columns))
    if missing:
        raise ValueError(f"silver_speeches missing required columns: {missing}")
    if silver["speech_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("silver_speeches contains blank speech_id values")
    if silver["speech_id"].duplicated().any():
        raise ValueError("silver_speeches contains duplicate speech_id values")

    existing_lookup = build_existing_lookup(existing)
    legacy_lookup, legacy_stats = build_legacy_lookup(legacy)
    now = utc_now_iso()

    rows: list[dict[str, Any]] = []
    stats = {
        "silver_rows": int(len(silver)),
        "reused_existing": 0,
        "migrated_legacy": 0,
        "short_text_none": 0,
        "pending_model": 0,
        "existing_hash_mismatch": 0,
        **legacy_stats,
    }

    for _, source in silver.iterrows():
        speech_id = normalize_text(source.get("speech_id"))
        speech_text = normalize_text(source.get("speech_text"))
        source_hash = normalize_text(source.get("speech_text_hash")) or text_hash(speech_text)
        debate_date = normalize_date(source.get("debate_date"))
        speech_order = normalize_order(source.get("speech_order"))
        speaker_name = normalize_text(source.get("speaker_name"))
        speaker_member_code = normalize_text(source.get("speaker_member_code"))
        word_count = int(pd.to_numeric(source.get("word_count"), errors="coerce") or 0)
        if word_count <= 0 and speech_text:
            word_count = len(speech_text.split())

        issue_label = ""
        issue_label_source = ""
        model_name = ""
        classification_status = "pending"
        classified_at_utc = ""

        prior = existing_lookup.get(speech_id)
        if prior:
            prior_hash, prior_label, prior_status, prior_model = prior
            if prior_hash == source_hash:
                issue_label = prior_label
                issue_label_source = "existing_unified_enrichment"
                model_name = prior_model
                classification_status = prior_status if prior_status in {"classified", "none", "skipped_short_text"} else ("none" if prior_label == "NONE" else "classified")
                stats["reused_existing"] += 1
            else:
                stats["existing_hash_mismatch"] += 1

        if not issue_label:
            legacy_key = (debate_date, speech_order, normalize_name(speaker_name), source_hash)
            legacy_label = legacy_lookup.get(legacy_key, "")
            if legacy_label:
                issue_label = legacy_label
                issue_label_source = "legacy_migration_exact"
                model_name = "legacy_unknown"
                classification_status = "none" if legacy_label == "NONE" else "classified"
                stats["migrated_legacy"] += 1

        if not issue_label and word_count < min_words:
            issue_label = "NONE"
            issue_label_source = "rule_short_text"
            model_name = "rule"
            classification_status = "skipped_short_text"
            classified_at_utc = now
            stats["short_text_none"] += 1

        if not issue_label:
            stats["pending_model"] += 1

        rows.append(
            {
                "speech_id": speech_id,
                "member_code": speaker_member_code,
                "speaker_name": speaker_name,
                "debate_date": debate_date,
                "speech_order": speech_order,
                "source_speech_text_hash": source_hash,
                "issue_label": issue_label,
                "issue_label_source": issue_label_source,
                "model_name": model_name,
                "classification_status": classification_status,
                "review_status": "unreviewed",
                "classified_at_utc": classified_at_utc,
                "speech_text": speech_text,
                "word_count": word_count,
            }
        )

    result = pd.DataFrame(rows)
    stats["planned_rows"] = int(len(result))
    stats["classified_or_none_before_model"] = int((result["issue_label"] != "").sum())
    return ClassificationPlan(rows=result, stats=stats)


def classification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"issue_label": {"type": "string", "enum": ISSUE_CATEGORIES}},
        "required": ["issue_label"],
        "additionalProperties": False,
    }


def build_classifier_prompt(speech_text: str) -> list[dict[str, str]]:
    categories = "\n".join(f"- {category}" for category in ISSUE_CATEGORIES)
    return [
        {
            "role": "system",
            "content": (
                "You classify Irish parliamentary speeches by their single core policy topic. "
                "Choose exactly one allowed issue label. Use NONE when the speech has no sufficiently clear core policy topic. "
                "Do not infer party positions, intent, or sentiment."
            ),
        },
        {
            "role": "user",
            "content": f"Allowed issue labels:\n{categories}\n\nSpeech:\n{speech_text}",
        },
    ]


def classify_with_openai(
    client: OpenAI,
    speech_text: str,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    verbosity: str = DEFAULT_VERBOSITY,
    max_retries: int = 4,
    retry_backoff_seconds: float = 2.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=build_classifier_prompt(speech_text),
                reasoning={"effort": reasoning_effort},
                text={
                    "verbosity": verbosity,
                    "format": {
                        "type": "json_schema",
                        "name": "speech_issue_classification",
                        "strict": True,
                        "schema": classification_schema(),
                    },
                },
                max_output_tokens=128,
                store=False,
            )
            payload = json.loads(str(response.output_text or "").strip())
            label = canonicalize_label(payload.get("issue_label"))
            if not label:
                raise ValueError(f"Model returned invalid issue label: {payload!r}")
            return label
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * attempt)
    raise RuntimeError(f"OpenAI classification failed after {max_retries} attempts: {last_error}")


def execute_model_classification(
    plan: ClassificationPlan,
    *,
    classify_fn: Callable[[str], str],
    model_name: str,
    max_rows: int = 0,
    delay_seconds: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = plan.rows.copy()
    pending = output.index[output["classification_status"] == "pending"].tolist()
    if max_rows > 0:
        pending = pending[:max_rows]

    succeeded = 0
    failed = 0
    for idx in pending:
        try:
            label = canonicalize_label(classify_fn(str(output.at[idx, "speech_text"])))
            if not label:
                raise ValueError("classifier returned a label outside the approved taxonomy")
            output.at[idx, "issue_label"] = label
            output.at[idx, "issue_label_source"] = "openai_model"
            output.at[idx, "model_name"] = model_name
            output.at[idx, "classification_status"] = "none" if label == "NONE" else "classified"
            output.at[idx, "classified_at_utc"] = utc_now_iso()
            succeeded += 1
        except Exception as exc:
            output.at[idx, "classification_status"] = "failed"
            output.at[idx, "issue_label_source"] = f"classification_error:{type(exc).__name__}"
            failed += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    stats = {
        **plan.stats,
        "model_attempted": int(len(pending)),
        "model_succeeded": int(succeeded),
        "model_failed": int(failed),
        "model_remaining_pending": int((output["classification_status"] == "pending").sum()),
    }
    return output, stats


def build_compat_output(silver: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    labels = enrichment[["speech_id", "issue_label", "classification_status"]].copy()
    merged = silver.merge(labels, on="speech_id", how="left", validate="one_to_one")
    compat = pd.DataFrame(
        {
            "speech_id": merged["speech_id"],
            "member_code": _col(merged, "speaker_member_code"),
            "Speaker Name": _col(merged, "speaker_name"),
            "Debate Date": _col(merged, "debate_date"),
            "Speech Order": _col(merged, "speech_order"),
            "Speech Text": _col(merged, "speech_text"),
            "PoliticalIssues": _col(merged, "issue_label"),
            "classification_status": _col(merged, "classification_status"),
        }
    )
    return compat


def validate_enrichment(silver: pd.DataFrame, enrichment: pd.DataFrame) -> dict[str, Any]:
    row_count_match = len(silver) == len(enrichment)
    speech_id_unique = bool(not enrichment["speech_id"].duplicated().any()) if len(enrichment) else False
    speech_id_populated = bool(enrichment["speech_id"].astype(str).str.strip().ne("").all()) if len(enrichment) else False
    invalid_labels = int((~enrichment["issue_label"].isin(ISSUE_CATEGORY_SET | {""})).sum()) if len(enrichment) else 0
    invalid_statuses = int((~enrichment["classification_status"].isin({"classified", "none", "skipped_short_text", "pending", "failed"})).sum()) if len(enrichment) else 0
    classified_missing_model = int(
        ((enrichment["issue_label_source"] == "openai_model") & (enrichment["model_name"].astype(str).str.strip() == "")).sum()
    ) if len(enrichment) else 0
    status = "pass" if all([len(enrichment) > 0, row_count_match, speech_id_unique, speech_id_populated, invalid_labels == 0, invalid_statuses == 0, classified_missing_model == 0]) else "fail"
    return {
        "dq_status": status,
        "silver_rows": int(len(silver)),
        "enrichment_rows": int(len(enrichment)),
        "row_count_match": bool(row_count_match),
        "speech_id_unique": bool(speech_id_unique),
        "speech_id_populated": bool(speech_id_populated),
        "invalid_issue_label_count": invalid_labels,
        "invalid_status_count": invalid_statuses,
        "model_classified_missing_model_name": classified_missing_model,
        "pending_rows": int((enrichment["classification_status"] == "pending").sum()),
        "failed_rows": int((enrichment["classification_status"] == "failed").sum()),
    }


def write_outputs(
    s3: Any,
    *,
    bucket: str,
    silver: pd.DataFrame,
    enrichment: pd.DataFrame,
    stats: dict[str, Any],
    model: str,
    mode: str,
) -> dict[str, Any]:
    compat = build_compat_output(silver, enrichment)
    dq = validate_enrichment(silver, enrichment)
    manifest = {
        "table": "enrichment_speech_issue_labels",
        "run_id": f"speech_issue_labels_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at_utc": utc_now_iso(),
        "mode": mode,
        "model": model,
        "status": "success" if dq["dq_status"] == "pass" else "failed",
        "stats": stats,
        "dq": dq,
        "source_keys": {
            "silver_speeches": SILVER_SPEECHES_KEY,
            "legacy_classified": LEGACY_CLASSIFIED_KEY,
        },
        "output_keys": {
            "enrichment_csv": ENRICHMENT_CSV_KEY,
            "enrichment_parquet": ENRICHMENT_PARQUET_KEY,
            "compat_csv": COMPAT_CSV_KEY,
            "compat_parquet": COMPAT_PARQUET_KEY,
            "manifest": MANIFEST_KEY,
        },
    }
    put_dataframe_csv(s3, bucket=bucket, key=ENRICHMENT_CSV_KEY, df=enrichment)
    put_dataframe_parquet(s3, bucket=bucket, key=ENRICHMENT_PARQUET_KEY, df=enrichment)
    put_dataframe_csv(s3, bucket=bucket, key=COMPAT_CSV_KEY, df=compat)
    put_dataframe_parquet(s3, bucket=bucket, key=COMPAT_PARQUET_KEY, df=compat)
    put_json(s3, bucket=bucket, key=MANIFEST_KEY, payload=manifest)
    return manifest


def readiness_report(silver: pd.DataFrame, existing: pd.DataFrame, legacy: pd.DataFrame) -> dict[str, Any]:
    plan = prepare_classification_plan(silver, existing=existing, legacy=legacy)
    rows = plan.rows
    dates = pd.to_datetime(silver["debate_date"], errors="coerce")
    classified_dates = pd.to_datetime(rows.loc[rows["issue_label"] != "", "debate_date"], errors="coerce")
    pending_dates = pd.to_datetime(rows.loc[rows["classification_status"] == "pending", "debate_date"], errors="coerce")
    return {
        **plan.stats,
        "silver_min_date": dates.min().date().isoformat() if dates.notna().any() else None,
        "silver_max_date": dates.max().date().isoformat() if dates.notna().any() else None,
        "classified_min_date": classified_dates.min().date().isoformat() if classified_dates.notna().any() else None,
        "classified_max_date": classified_dates.max().date().isoformat() if classified_dates.notna().any() else None,
        "pending_min_date": pending_dates.min().date().isoformat() if pending_dates.notna().any() else None,
        "pending_max_date": pending_dates.max().date().isoformat() if pending_dates.notna().any() else None,
        "ready_for_live_test": bool(plan.stats["pending_model"] > 0 and plan.stats["migrated_legacy"] > 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Oireachtas speech issue classifier")
    parser.add_argument("--mode", choices=["readiness", "dry-run", "classify"], default="readiness")
    parser.add_argument("--max-model-rows", type=int, default=0)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT))
    parser.add_argument("--verbosity", default=os.getenv("OPENAI_VERBOSITY", DEFAULT_VERBOSITY))
    parser.add_argument("--delay-seconds", type=float, default=float(os.getenv("CLASSIFIER_DELAY_SECONDS", "0.1")))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--report-path", default="speech_issue_classifier_report.json")
    parser.add_argument("--write", action="store_true", help="Write enrichment/compat outputs. Required for production use.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    s3 = make_s3_client(region_name=args.region)
    silver = read_s3_csv(s3, bucket=args.bucket, key=SILVER_SPEECHES_KEY)
    existing = read_s3_csv(s3, bucket=args.bucket, key=ENRICHMENT_CSV_KEY, optional=True)
    legacy = read_s3_csv(s3, bucket=args.bucket, key=LEGACY_CLASSIFIED_KEY, optional=True)

    if args.mode == "readiness":
        report = readiness_report(silver, existing, legacy)
        Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    plan = prepare_classification_plan(silver, existing=existing, legacy=legacy)
    output = plan.rows
    stats = dict(plan.stats)
    if args.mode == "classify":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required in classify mode")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        classify_fn = lambda text: classify_with_openai(
            client,
            text,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
        )
        output, stats = execute_model_classification(
            plan,
            classify_fn=classify_fn,
            model_name=args.model,
            max_rows=args.max_model_rows,
            delay_seconds=args.delay_seconds,
        )

    dq = validate_enrichment(silver, output)
    report = {"mode": args.mode, "model": args.model, "stats": stats, "dq": dq, "write_requested": bool(args.write)}
    if args.write:
        if args.mode == "dry-run":
            raise RuntimeError("--write is not allowed in dry-run mode")
        report["manifest"] = write_outputs(s3, bucket=args.bucket, silver=silver, enrichment=output, stats=stats, model=args.model, mode=args.mode)
    Path(args.report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if dq["dq_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
