from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

from extract.oireachtas.batch import PRODUCTION_POINTER_KEY, read_json_required, resolve_production_key
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client

TABLE_NAME = "enrichment_speech_issue_labels"
SOURCE_LOGICAL_KEY = "processed/oireachtas_unified/latest/parquet/silver_speeches.parquet"
LATEST_CSV_KEY = f"processed/oireachtas_unified/latest/csv/{TABLE_NAME}.csv"
LATEST_PARQUET_KEY = f"processed/oireachtas_unified/latest/parquet/{TABLE_NAME}.parquet"
RUN_ROOT = f"processed/oireachtas_unified/enrichment/{TABLE_NAME}/runs"

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

OUTPUT_COLUMNS = [
    "speech_id",
    "speech_text_hash",
    "issue_label",
    "classification_status",
    "model_name",
    "prompt_version",
    "taxonomy_version",
    "classified_at_utc",
    "input_tokens",
    "output_tokens",
    "source_batch_id",
    "openai_response_id",
    "review_status",
]

PROMPT_VERSION = "speech-issue-v2.0"
TAXONOMY_VERSION = "legacy-25-v1"


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    response_id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonicalize_label(value: str | None) -> str | None:
    text = str(value or "").strip()
    for category in ISSUE_CATEGORIES:
        if text.casefold() == category.casefold():
            return category
    return None


def build_prompt(speech_text: str) -> str:
    allowed = "\n".join(f"- {value}" for value in ISSUE_CATEGORIES)
    return (
        "Classify this Irish parliamentary speech into exactly one political issue category.\n"
        "Use NONE when the text is procedural, too short, unclear, or lacks one dominant political topic.\n"
        "Return only one category name from the allowed list.\n\n"
        f"Allowed categories:\n{allowed}\n\nSpeech:\n{speech_text.strip()}"
    )


def extract_response_text(response: Any) -> str:
    direct = str(getattr(response, "output_text", "") or "").strip()
    if direct:
        return direct
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def classify_with_openai(client: OpenAI, *, model: str, speech_text: str) -> ClassificationResult:
    response = client.responses.create(
        model=model,
        input=build_prompt(speech_text),
        max_output_tokens=64,
    )
    raw = extract_response_text(response)
    label = canonicalize_label(raw)
    if label is None:
        raise ValueError(f"Model returned an unapproved category: {raw!r}")
    usage = getattr(response, "usage", None)
    return ClassificationResult(
        label=label,
        response_id=str(getattr(response, "id", "") or ""),
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def select_delta(speeches: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    required = {"speech_id", "speech_text_hash", "speech_text"}
    missing = sorted(required - set(speeches.columns))
    if missing:
        raise ValueError(f"silver_speeches is missing required columns: {missing}")

    current = speeches.copy()
    current["speech_id"] = current["speech_id"].fillna("").astype(str).str.strip()
    current["speech_text_hash"] = current["speech_text_hash"].fillna("").astype(str).str.strip()
    current = current[(current["speech_id"] != "") & (current["speech_text_hash"] != "")]
    current = current.drop_duplicates(subset=["speech_id"], keep="last")

    if existing.empty or not {"speech_id", "speech_text_hash", "classification_status"}.issubset(existing.columns):
        return current.reset_index(drop=True)

    prior = existing.copy()
    prior = prior[prior["classification_status"].eq("classified")]
    prior = prior.drop_duplicates(subset=["speech_id"], keep="last")
    classified_hash = dict(zip(prior["speech_id"].astype(str), prior["speech_text_hash"].astype(str)))
    mask = current.apply(lambda row: classified_hash.get(str(row["speech_id"])) != str(row["speech_text_hash"]), axis=1)
    return current.loc[mask].reset_index(drop=True)


def merge_results(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, new_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in OUTPUT_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    combined = combined[OUTPUT_COLUMNS]
    combined = combined.drop_duplicates(subset=["speech_id"], keep="last")
    return combined.sort_values("speech_id", kind="stable").reset_index(drop=True)


def run_classifier(
    *,
    s3: Any,
    bucket: str,
    model: str,
    max_rows: int,
    publish_latest: bool,
    classifier: Callable[[str], ClassificationResult],
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    pointer = read_json_required(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    source_batch_id = str(pointer.get("batch_id") or "")
    source_key = resolve_production_key(s3, bucket=bucket, production_key=SOURCE_LOGICAL_KEY)
    speeches = _read_parquet(s3, bucket=bucket, key=source_key)
    existing = _read_parquet_if_exists(s3, bucket=bucket, key=LATEST_PARQUET_KEY)
    delta = select_delta(speeches, existing)
    if max_rows > 0:
        delta = delta.head(max_rows).copy()

    classified_at = utc_now_iso()
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in delta.to_dict(orient="records"):
        speech_id = str(row["speech_id"])
        speech_text = str(row.get("speech_text") or "").strip()
        try:
            result = ClassificationResult(label="NONE") if len(speech_text.split()) < 20 else classifier(speech_text)
            output_rows.append(
                {
                    "speech_id": speech_id,
                    "speech_text_hash": str(row["speech_text_hash"]),
                    "issue_label": result.label,
                    "classification_status": "classified",
                    "model_name": model,
                    "prompt_version": PROMPT_VERSION,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "classified_at_utc": classified_at,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "source_batch_id": source_batch_id,
                    "openai_response_id": result.response_id,
                    "review_status": "unreviewed",
                }
            )
        except Exception as exc:
            failures.append({"speech_id": speech_id, "error": f"{type(exc).__name__}: {exc}"})
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    new_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    merged = merge_results(existing, new_df)
    invalid = sorted(set(new_df.get("issue_label", pd.Series(dtype=str)).dropna()) - set(ISSUE_CATEGORIES))
    duplicate_ids = int(merged["speech_id"].duplicated().sum()) if not merged.empty else 0
    dq_status = "pass" if not failures and not invalid and duplicate_ids == 0 else "fail"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = f"{RUN_ROOT}/run_id={run_id}"
    candidate_csv = f"{run_prefix}/{TABLE_NAME}.csv"
    candidate_parquet = f"{run_prefix}/{TABLE_NAME}.parquet"
    manifest_key = f"{run_prefix}/manifest.json"
    _write_dataframe(s3, bucket=bucket, csv_key=candidate_csv, parquet_key=candidate_parquet, df=merged)

    if publish_latest:
        if dq_status != "pass":
            raise RuntimeError("Refusing to publish failed classification output")
        _write_dataframe(s3, bucket=bucket, csv_key=LATEST_CSV_KEY, parquet_key=LATEST_PARQUET_KEY, df=merged)

    manifest = {
        "table": TABLE_NAME,
        "run_id": run_id,
        "status": "success" if dq_status == "pass" else "failed",
        "dq_status": dq_status,
        "source_batch_id": source_batch_id,
        "source_key": source_key,
        "model_name": model,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "source_rows": int(len(speeches)),
        "existing_rows": int(len(existing)),
        "delta_rows_selected": int(len(delta)),
        "classified_rows": int(len(new_df)),
        "failure_count": len(failures),
        "failures": failures,
        "output_rows": int(len(merged)),
        "published_latest": bool(publish_latest),
        "candidate_csv": candidate_csv,
        "candidate_parquet": candidate_parquet,
    }
    s3.put_object(Bucket=bucket, Key=manifest_key, Body=(json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), ContentType="application/json")
    return manifest


def _read_parquet(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def _read_parquet_if_exists(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    try:
        return _read_parquet(s3, bucket=bucket, key=key)
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _write_dataframe(s3: Any, *, bucket: str, csv_key: str, parquet_key: str, df: pd.DataFrame) -> None:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=csv_key, Body=csv_buffer.getvalue().encode("utf-8"), ContentType="text/csv")
    s3.put_object(Bucket=bucket, Key=parquet_key, Body=parquet_buffer.getvalue(), ContentType="application/x-parquet")


def main() -> int:
    bucket = os.getenv("S3_BUCKET", DEFAULT_BUCKET)
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    max_rows = int(os.getenv("MAX_ROWS", "25") or "25")
    publish_latest = os.getenv("PUBLISH_LATEST", "false").strip().lower() == "true"
    delay_seconds = float(os.getenv("DELAY_SECONDS", "0.2") or "0.2")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    s3 = make_s3_client(region_name=region)
    manifest = run_classifier(
        s3=s3,
        bucket=bucket,
        model=model,
        max_rows=max_rows,
        publish_latest=publish_latest,
        classifier=lambda text: classify_with_openai(client, model=model, speech_text=text),
        delay_seconds=delay_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["dq_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
