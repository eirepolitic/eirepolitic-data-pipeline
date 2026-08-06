from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from botocore.exceptions import ClientError
from openai import OpenAI

from extract.oireachtas.batch import (
    PRODUCTION_POINTER_KEY,
    read_json_required,
    resolve_production_key,
)
from extract.oireachtas.io_s3 import DEFAULT_BUCKET, DEFAULT_REGION, make_s3_client

TABLE_NAME = "enrichment_speech_issue_labels"
SOURCE_LOGICAL_KEY = "processed/oireachtas_unified/latest/parquet/silver_speeches.parquet"
ENRICHMENT_ROOT = f"processed/oireachtas_unified/enrichment/{TABLE_NAME}"
RUN_ROOT = f"{ENRICHMENT_ROOT}/runs"
PRODUCTION_CLASSIFICATION_POINTER_KEY = f"{ENRICHMENT_ROOT}/pointers/production.json"
PREVIOUS_CLASSIFICATION_POINTER_KEY = f"{ENRICHMENT_ROOT}/pointers/previous.json"

PROMPT_VERSION = "speech-issue-v2.1"
TAXONOMY_VERSION = "legacy-25-v1"
DEFAULT_MAX_FAILURE_RATE = 0.02
DEFAULT_SHORT_SPEECH_WORD_LIMIT = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_HOURS = 24
DEFAULT_MAX_SUBMISSION_ATTEMPTS = 3
BATCH_ENDPOINT = "/v1/responses"
BATCH_COMPLETION_WINDOW = "24h"

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
ISSUE_CATEGORY_SET = frozenset(ISSUE_CATEGORIES)

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
    "source_batch_speech_key",
    "classification_run_id",
    "openai_response_id",
    "openai_batch_id",
    "review_status",
    "classification_error",
    "attempt_count",
    "retry_eligible_after_utc",
]
SOURCE_REQUIRED_COLUMNS = {"speech_id", "speech_text_hash", "speech_text"}
PUBLISH_REQUIRED_COLUMNS = {
    "speech_id",
    "speech_text_hash",
    "classification_status",
    "model_name",
    "prompt_version",
    "taxonomy_version",
    "classified_at_utc",
    "source_batch_id",
    "classification_run_id",
    "review_status",
    "attempt_count",
}
STRING_OUTPUT_COLUMNS = [
    column
    for column in OUTPUT_COLUMNS
    if column not in {"input_tokens", "output_tokens", "attempt_count"}
]

SYSTEM_PROMPT = (
    "Classify an Irish parliamentary speech into exactly one political issue category. "
    "Choose the single dominant subject. Use NONE for procedural, unclear, extremely short, "
    "or non-substantive text without one dominant political subject."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"issue_label": {"type": "string", "enum": ISSUE_CATEGORIES}},
    "required": ["issue_label"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ParsedBatchResult:
    custom_id: str
    status: str
    label: str = ""
    response_id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str = ""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return f"speech_issue_{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}"


def run_prefix(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value or "/" in value or ".." in value:
        raise ValueError(f"Unsafe run_id: {run_id!r}")
    return f"{RUN_ROOT}/run_id={value}"


def run_manifest_key(run_id: str) -> str:
    return f"{run_prefix(run_id)}/manifest.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _is_missing_s3_error(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    response = getattr(exc, "response", {}) or {}
    error = response.get("Error", {}) or {}
    code = str(error.get("Code") or "")
    status = int((response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode") or 0)
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def _is_precondition_failure(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    response = getattr(exc, "response", {}) or {}
    error = response.get("Error", {}) or {}
    code = str(error.get("Code") or "")
    status = int((response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode") or 0)
    return code in {"PreconditionFailed", "412", "ConditionalRequestConflict"} or status in {409, 412}


def read_json_optional(s3: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return read_json_required(s3, bucket=bucket, key=key)
    except ClientError as exc:
        if _is_missing_s3_error(exc):
            return None
        raise


def read_json_optional_with_etag(
    s3: Any,
    *,
    bucket: str,
    key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_missing_s3_error(exc):
            return None, None
        raise
    payload = json.loads(response["Body"].read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at s3://{bucket}/{key}")
    etag = str(response.get("ETag") or "").strip() or None
    return payload, etag


def put_bytes_direct(
    s3: Any,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
) -> dict[str, Any]:
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)
    return {"key": key, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def put_json_direct(s3: Any, *, bucket: str, key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return put_bytes_direct(
        s3,
        bucket=bucket,
        key=key,
        payload=_json_bytes(payload),
        content_type="application/json",
    )


def put_json_conditional(
    s3: Any,
    *,
    bucket: str,
    key: str,
    payload: Mapping[str, Any],
    expected_etag: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": _json_bytes(payload),
        "ContentType": "application/json",
    }
    if expected_etag:
        kwargs["IfMatch"] = expected_etag
    else:
        kwargs["IfNoneMatch"] = "*"
    try:
        s3.put_object(**kwargs)
    except ClientError as exc:
        if _is_precondition_failure(exc):
            raise RuntimeError(
                "Classification production pointer changed concurrently; refusing publication"
            ) from exc
        raise
    body = kwargs["Body"]
    return {"key": key, "sha256": sha256_bytes(body), "size_bytes": len(body)}


def get_bytes_required(s3: Any, *, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def get_bytes_verified(
    s3: Any,
    *,
    bucket: str,
    key: str,
    expected_sha256: str | None,
) -> bytes:
    payload = get_bytes_required(s3, bucket=bucket, key=key)
    if expected_sha256 and sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"Checksum mismatch for s3://{bucket}/{key}")
    return payload


def read_parquet_required(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(get_bytes_required(s3, bucket=bucket, key=key)))


def read_csv_required(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(get_bytes_required(s3, bucket=bucket, key=key)), dtype=object)


def dataframe_artifacts(df: pd.DataFrame) -> tuple[bytes, bytes]:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    return csv_buffer.getvalue().encode("utf-8"), parquet_buffer.getvalue()


def write_dataframe_direct(
    s3: Any,
    *,
    bucket: str,
    csv_key: str,
    parquet_key: str,
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    csv_payload, parquet_payload = dataframe_artifacts(df)
    return {
        "csv": put_bytes_direct(
            s3,
            bucket=bucket,
            key=csv_key,
            payload=csv_payload,
            content_type="text/csv",
        ),
        "parquet": put_bytes_direct(
            s3,
            bucket=bucket,
            key=parquet_key,
            payload=parquet_payload,
            content_type="application/x-parquet",
        ),
    }


def normalize_source_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(SOURCE_REQUIRED_COLUMNS - set(speeches.columns))
    if missing:
        raise ValueError(f"silver_speeches is missing required columns: {missing}")
    current = speeches.copy()
    for column in ("speech_id", "speech_text_hash", "speech_text"):
        current[column] = current[column].fillna("").astype(str).str.strip()
    if current["speech_id"].eq("").any() or current["speech_text_hash"].eq("").any():
        raise ValueError("silver_speeches contains blank speech_id or speech_text_hash")
    duplicates = current.loc[current["speech_id"].duplicated(keep=False), "speech_id"].tolist()
    if duplicates:
        raise ValueError(
            f"silver_speeches contains duplicate speech_id values: {sorted(set(duplicates))[:10]}"
        )
    return current.reset_index(drop=True)


def normalize_existing(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    missing = sorted(
        {"speech_id", "speech_text_hash", "classification_status"} - set(existing.columns)
    )
    if missing:
        raise ValueError(f"Current classification table is missing required columns: {missing}")
    output = existing.copy()
    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = 0 if column == "attempt_count" else ""
    for column in STRING_OUTPUT_COLUMNS:
        output[column] = output[column].fillna("").astype(str).str.strip()
    output["attempt_count"] = (
        pd.to_numeric(output["attempt_count"], errors="coerce").fillna(0).astype(int)
    )
    for column in ("input_tokens", "output_tokens"):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    if output["speech_id"].eq("").any():
        raise ValueError("Current classification table contains blank speech_id values")
    if output["speech_id"].duplicated().any():
        raise ValueError("Current classification table contains duplicate speech_id values")
    return output[OUTPUT_COLUMNS].reset_index(drop=True)


def validate_label(value: Any) -> str:
    label = str(value or "").strip()
    if label not in ISSUE_CATEGORY_SET:
        raise ValueError(f"Invalid issue label: {label!r}")
    return label


def _parse_utc_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp: {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _retry_is_eligible(
    previous: Mapping[str, Any],
    *,
    max_retries: int,
    now: datetime,
) -> bool:
    attempt_count = int(previous.get("attempt_count") or 0)
    if attempt_count >= max_retries:
        return False
    retry_after = _parse_utc_timestamp(previous.get("retry_eligible_after_utc"))
    return retry_after is None or retry_after <= now


def select_delta(
    speeches: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    force_speech_ids: Iterable[str] = (),
    retry_failed: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    current = normalize_source_speeches(speeches)
    prior = normalize_existing(existing)
    forced = {str(value).strip() for value in force_speech_ids if str(value).strip()}
    now = (now_utc or utc_now()).astimezone(timezone.utc)
    prior_by_id = prior.set_index("speech_id", drop=False) if not prior.empty else None
    selected_rows: list[dict[str, Any]] = []

    for row in current.to_dict(orient="records"):
        speech_id = str(row["speech_id"])
        reason = ""
        prior_attempt_count = 0
        if prior_by_id is None or speech_id not in prior_by_id.index:
            reason = "new"
        else:
            previous = prior_by_id.loc[speech_id]
            changed = str(previous.get("speech_text_hash") or "") != str(
                row["speech_text_hash"]
            )
            failed = str(previous.get("classification_status") or "") != "classified"
            if speech_id in forced:
                reason = "forced"
            elif changed:
                reason = "changed_hash"
            elif failed and retry_failed and _retry_is_eligible(
                previous,
                max_retries=max_retries,
                now=now,
            ):
                reason = "retry_failed"
                prior_attempt_count = int(previous.get("attempt_count") or 0)
        if reason:
            selected = dict(row)
            selected["selection_reason"] = reason
            selected["prior_attempt_count"] = prior_attempt_count
            selected_rows.append(selected)

    extra_columns = ["selection_reason", "prior_attempt_count"]
    if not selected_rows:
        return pd.DataFrame(columns=[*current.columns, *extra_columns])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def merge_results(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    prior = normalize_existing(existing)
    incoming = normalize_existing(new_rows) if not new_rows.empty else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not incoming.empty and incoming["speech_id"].duplicated().any():
        raise ValueError("New classification rows contain duplicate speech_id values")
    frames = [frame for frame in (prior, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("speech_id", keep="last")
        .sort_values("speech_id", kind="stable")
        .reset_index(drop=True)
    )


def reconcile_results_to_source(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    speeches: pd.DataFrame,
) -> pd.DataFrame:
    source = normalize_source_speeches(speeches)
    merged = merge_results(existing, new_rows)
    if merged.empty:
        return merged
    source_hashes = dict(
        zip(source["speech_id"].astype(str), source["speech_text_hash"].astype(str))
    )
    keep = merged.apply(
        lambda row: source_hashes.get(str(row["speech_id"]))
        == str(row["speech_text_hash"]),
        axis=1,
    )
    return merged.loc[keep, OUTPUT_COLUMNS].reset_index(drop=True)


def structured_response_body(*, model: str, speech_text: str) -> dict[str, Any]:
    if not str(model or "").strip():
        raise ValueError("A model ID is required")
    return {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": speech_text},
        ],
        "max_output_tokens": 64,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "speech_issue_label",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    }


def custom_id_for_speech(speech_id: str) -> str:
    return f"speech-{hashlib.sha256(str(speech_id).encode('utf-8')).hexdigest()[:32]}"


def build_batch_requests(
    delta: pd.DataFrame,
    *,
    model: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prepared = delta.copy()
    prepared["custom_id"] = prepared["speech_id"].map(custom_id_for_speech)
    if prepared["custom_id"].duplicated().any():
        raise ValueError("Generated duplicate OpenAI custom_id values")
    requests = [
        {
            "custom_id": row["custom_id"],
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": structured_response_body(
                model=model,
                speech_text=str(row["speech_text"]),
            ),
        }
        for row in prepared.to_dict(orient="records")
    ]
    return prepared, requests


def parse_response_output_text(body: Mapping[str, Any]) -> str:
    direct = str(body.get("output_text") or "").strip()
    if direct:
        return direct
    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, Mapping):
                continue
            if content.get("type") == "refusal":
                raise ValueError("OpenAI refusal")
            if content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def parse_structured_label(body: Mapping[str, Any]) -> str:
    raw = parse_response_output_text(body)
    if not raw:
        raise ValueError("OpenAI response contained no output text")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("Structured output was not a JSON object")
    return validate_label(payload.get("issue_label"))


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def parse_batch_result_line(line: Mapping[str, Any]) -> ParsedBatchResult:
    custom_id = str(line.get("custom_id") or "").strip()
    if not custom_id:
        raise ValueError("Batch result line is missing custom_id")
    if line.get("error"):
        return ParsedBatchResult(
            custom_id,
            "failed",
            error=json.dumps(line["error"], sort_keys=True),
        )
    response = line.get("response")
    if not isinstance(response, Mapping):
        return ParsedBatchResult(custom_id, "failed", error="Missing response object")
    body = response.get("body")
    status_code = int(response.get("status_code") or 0)
    if status_code < 200 or status_code >= 300 or not isinstance(body, Mapping):
        return ParsedBatchResult(custom_id, "failed", error=f"HTTP {status_code}")
    try:
        label = parse_structured_label(body)
    except Exception as exc:
        return ParsedBatchResult(
            custom_id,
            "failed",
            response_id=str(body.get("id") or ""),
            error=f"{type(exc).__name__}: {exc}",
        )
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    return ParsedBatchResult(
        custom_id,
        "classified",
        label,
        str(body.get("id") or ""),
        _optional_int(usage.get("input_tokens")),
        _optional_int(usage.get("output_tokens")),
    )


def parse_batch_output_jsonl(payload: bytes) -> list[ParsedBatchResult]:
    results: list[ParsedBatchResult] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Batch output line {line_number} is not a JSON object")
        results.append(parse_batch_result_line(parsed))
    if len({result.custom_id for result in results}) != len(results):
        raise ValueError("Batch output contains duplicate custom_id values")
    return results


def combine_batch_results(
    output_results: Sequence[ParsedBatchResult],
    error_results: Sequence[ParsedBatchResult],
) -> list[ParsedBatchResult]:
    combined = [*output_results, *error_results]
    custom_ids = [result.custom_id for result in combined]
    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError("OpenAI output and error files contain overlapping custom_id values")
    return combined


def read_source_context(s3: Any, *, bucket: str) -> tuple[str, str, pd.DataFrame]:
    pointer = read_json_required(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    source_batch_id = str(pointer.get("batch_id") or "").strip()
    if not source_batch_id:
        raise ValueError("Active Oireachtas production pointer has no batch_id")
    source_key = resolve_production_key(
        s3,
        bucket=bucket,
        production_key=SOURCE_LOGICAL_KEY,
    )
    speeches = normalize_source_speeches(
        read_parquet_required(s3, bucket=bucket, key=source_key)
    )
    return source_batch_id, source_key, speeches


def read_source_snapshot(s3: Any, *, bucket: str, source_key: str) -> pd.DataFrame:
    return normalize_source_speeches(
        read_parquet_required(s3, bucket=bucket, key=source_key)
    )


def read_current_enrichment(
    s3: Any,
    *,
    bucket: str,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    pointer = read_json_optional(
        s3,
        bucket=bucket,
        key=PRODUCTION_CLASSIFICATION_POINTER_KEY,
    )
    if pointer is None:
        return None, pd.DataFrame(columns=OUTPUT_COLUMNS)
    key = str(pointer.get("table_parquet_key") or "").strip()
    if not key:
        raise ValueError("Classification production pointer is missing table_parquet_key")
    return pointer, normalize_existing(read_parquet_required(s3, bucket=bucket, key=key))


def read_base_enrichment(
    s3: Any,
    *,
    bucket: str,
    table_key: str,
) -> pd.DataFrame:
    if not str(table_key or "").strip():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return normalize_existing(read_parquet_required(s3, bucket=bucket, key=table_key))


def pointer_identity(pointer: Mapping[str, Any] | None) -> tuple[str, str]:
    if not pointer:
        return "", ""
    return (
        str(pointer.get("run_id") or "").strip(),
        str(pointer.get("table_parquet_key") or "").strip(),
    )


def candidate_staleness_reasons(
    *,
    manifest: Mapping[str, Any],
    active_source_batch_id: str,
    active_source_key: str,
    current_classification_pointer: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if active_source_batch_id != str(manifest.get("source_batch_id") or ""):
        reasons.append("active_source_batch_changed")
    if active_source_key != str(manifest.get("source_batch_speech_key") or ""):
        reasons.append("active_source_key_changed")
    expected_base = (
        str(manifest.get("base_classification_run_id") or ""),
        str(manifest.get("base_classification_table_key") or ""),
    )
    if pointer_identity(current_classification_pointer) != expected_base:
        reasons.append("classification_pointer_changed")
    return reasons


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _results_from_deterministic(
    rows: pd.DataFrame,
    *,
    model: str,
    source_batch_id: str,
    source_key: str,
    run_id: str,
) -> pd.DataFrame:
    classified_at = utc_now_iso()
    output: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        output.append(
            {
                "speech_id": str(row["speech_id"]),
                "speech_text_hash": str(row["speech_text_hash"]),
                "issue_label": "NONE",
                "classification_status": "classified",
                "model_name": f"{model}:deterministic-short-speech",
                "prompt_version": PROMPT_VERSION,
                "taxonomy_version": TAXONOMY_VERSION,
                "classified_at_utc": classified_at,
                "input_tokens": 0,
                "output_tokens": 0,
                "source_batch_id": source_batch_id,
                "source_batch_speech_key": source_key,
                "classification_run_id": run_id,
                "openai_response_id": "",
                "openai_batch_id": "",
                "review_status": "unreviewed",
                "classification_error": "",
                "attempt_count": int(row.get("prior_attempt_count") or 0) + 1,
                "retry_eligible_after_utc": "",
            }
        )
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)


def prepare_run(
    *,
    s3: Any,
    bucket: str,
    model: str,
    max_rows: int,
    historical_backfill: bool,
    force_speech_ids: Iterable[str] = (),
    short_speech_word_limit: int = DEFAULT_SHORT_SPEECH_WORD_LIMIT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    run_id: str | None = None,
) -> dict[str, Any]:
    if max_rows < 0:
        raise ValueError("max_rows cannot be negative")
    if historical_backfill and os.getenv(
        "OIREACHTAS_SPEECH_CLASSIFIER_BACKFILL_ENABLED",
        "false",
    ).lower() != "true":
        raise RuntimeError("Historical backfill switch is disabled")

    run_id = run_id or new_run_id()
    prefix = run_prefix(run_id)
    source_batch_id, source_key, speeches = read_source_context(s3, bucket=bucket)
    current_pointer, existing = read_current_enrichment(s3, bucket=bucket)
    forced_ids = [str(value).strip() for value in force_speech_ids if str(value).strip()]
    full_delta = select_delta(
        speeches,
        existing,
        force_speech_ids=forced_ids,
        max_retries=max_retries,
    )
    delta = full_delta.head(max_rows).copy() if max_rows > 0 else full_delta.copy()

    short_mask = delta["speech_text"].str.split().str.len() < short_speech_word_limit
    deterministic = delta.loc[short_mask].reset_index(drop=True)
    paid_delta = delta.loc[~short_mask].reset_index(drop=True)
    selection, requests = build_batch_requests(paid_delta, model=model)
    deterministic_rows = _results_from_deterministic(
        deterministic,
        model=model,
        source_batch_id=source_batch_id,
        source_key=source_key,
        run_id=run_id,
    )

    selection_csv_key = f"{prefix}/selection.csv"
    selection_parquet_key = f"{prefix}/selection.parquet"
    requests_key = f"{prefix}/openai_requests.jsonl"
    deterministic_key = f"{prefix}/deterministic_results.csv"
    selection_artifacts = write_dataframe_direct(
        s3,
        bucket=bucket,
        csv_key=selection_csv_key,
        parquet_key=selection_parquet_key,
        df=selection,
    )
    request_artifact = put_bytes_direct(
        s3,
        bucket=bucket,
        key=requests_key,
        payload=_jsonl_bytes(requests),
        content_type="application/jsonl",
    )
    deterministic_csv, _ = dataframe_artifacts(deterministic_rows)
    deterministic_artifact = put_bytes_direct(
        s3,
        bucket=bucket,
        key=deterministic_key,
        payload=deterministic_csv,
        content_type="text/csv",
    )

    reconciled_existing = reconcile_results_to_source(
        existing,
        pd.DataFrame(columns=OUTPUT_COLUMNS),
        speeches,
    )
    maintenance_needed = len(reconciled_existing) != len(existing)
    if current_pointer and str(current_pointer.get("source_batch_id") or "") != source_batch_id:
        maintenance_needed = True
    status = (
        "no_op"
        if delta.empty and not maintenance_needed
        else ("ready_to_collect" if not requests else "prepared")
    )
    base_run_id, base_table_key = pointer_identity(current_pointer)
    manifest = {
        "table": TABLE_NAME,
        "run_id": run_id,
        "status": status,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "source_batch_id": source_batch_id,
        "source_batch_speech_key": source_key,
        "base_classification_run_id": base_run_id,
        "base_classification_table_key": base_table_key,
        "model_name": model,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "historical_backfill": bool(historical_backfill),
        "source_rows": int(len(speeches)),
        "existing_rows": int(len(existing)),
        "full_delta_rows": int(len(full_delta)),
        "delta_rows_selected": int(len(delta)),
        "delta_truncated": bool(len(delta) < len(full_delta)),
        "maintenance_needed": bool(maintenance_needed),
        "deterministic_none_rows": int(len(deterministic_rows)),
        "batch_request_rows": int(len(requests)),
        "max_rows": int(max_rows),
        "max_retries": int(max_retries),
        "short_speech_word_limit": int(short_speech_word_limit),
        "selection_csv_key": selection_csv_key,
        "selection_parquet_key": selection_parquet_key,
        "requests_jsonl_key": requests_key,
        "deterministic_results_key": deterministic_key,
        "manifest_key": run_manifest_key(run_id),
        "artifact_checksums": {
            "selection_csv": selection_artifacts["csv"],
            "selection_parquet": selection_artifacts["parquet"],
            "openai_requests_jsonl": request_artifact,
            "deterministic_results_csv": deterministic_artifact,
        },
        "openai_batch_id": "",
        "openai_input_file_id": "",
        "openai_output_file_id": "",
        "openai_error_file_id": "",
        "batch_submission_attempts": 0,
        "published": False,
    }
    put_json_direct(
        s3,
        bucket=bucket,
        key=run_manifest_key(run_id),
        payload=manifest,
    )
    return manifest


def verify_models_available(client: OpenAI, models: Iterable[str]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for model in models:
        model_id = str(model).strip()
        if not model_id:
            continue
        try:
            client.models.retrieve(model_id)
            results[model_id] = True
        except Exception:
            results[model_id] = False
    return results


def submit_run(
    *,
    s3: Any,
    bucket: str,
    client: OpenAI,
    run_id: str,
    max_submission_attempts: int = DEFAULT_MAX_SUBMISSION_ATTEMPTS,
) -> dict[str, Any]:
    manifest = read_json_required(s3, bucket=bucket, key=run_manifest_key(run_id))
    if manifest.get("status") == "no_op" or manifest.get("openai_batch_id"):
        return manifest
    if manifest.get("status") != "prepared":
        raise ValueError("Run is not ready for submission")
    attempts = int(manifest.get("batch_submission_attempts") or 0)
    if attempts >= max_submission_attempts:
        raise RuntimeError("OpenAI Batch submission retry limit reached")

    model = str(manifest["model_name"])
    if not verify_models_available(client, [model]).get(model):
        raise RuntimeError(f"Model unavailable: {model}")
    request_artifact = (manifest.get("artifact_checksums") or {}).get(
        "openai_requests_jsonl",
        {},
    )
    request_bytes = get_bytes_verified(
        s3,
        bucket=bucket,
        key=str(manifest["requests_jsonl_key"]),
        expected_sha256=str(request_artifact.get("sha256") or "") or None,
    )
    upload = io.BytesIO(request_bytes)
    upload.name = f"{run_id}.jsonl"
    manifest["batch_submission_attempts"] = attempts + 1
    manifest["updated_at_utc"] = utc_now_iso()
    put_json_direct(s3, bucket=bucket, key=run_manifest_key(run_id), payload=manifest)

    input_file = client.files.create(file=upload, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={
            "pipeline": TABLE_NAME,
            "run_id": run_id,
            "source_batch_id": str(manifest["source_batch_id"]),
            "prompt_version": PROMPT_VERSION,
        },
    )
    manifest.update(
        status="submitted",
        updated_at_utc=utc_now_iso(),
        submitted_at_utc=utc_now_iso(),
        openai_input_file_id=str(input_file.id),
        openai_batch_id=str(batch.id),
        openai_batch_status=str(batch.status),
    )
    put_json_direct(s3, bucket=bucket, key=run_manifest_key(run_id), payload=manifest)
    return manifest


def refresh_batch_status(
    *,
    s3: Any,
    bucket: str,
    client: OpenAI,
    run_id: str,
) -> dict[str, Any]:
    manifest = read_json_required(s3, bucket=bucket, key=run_manifest_key(run_id))
    batch_id = str(manifest.get("openai_batch_id") or "")
    if not batch_id:
        return manifest
    batch = client.batches.retrieve(batch_id)
    manifest.update(
        openai_batch_status=str(batch.status),
        openai_output_file_id=str(getattr(batch, "output_file_id", "") or ""),
        openai_error_file_id=str(getattr(batch, "error_file_id", "") or ""),
        updated_at_utc=utc_now_iso(),
    )
    if batch.status == "completed":
        manifest["status"] = "ready_to_collect"
    elif batch.status in {"failed", "expired", "cancelled"}:
        manifest["status"] = "batch_failed"
    put_json_direct(s3, bucket=bucket, key=run_manifest_key(run_id), payload=manifest)
    return manifest


def _client_file_bytes(client: OpenAI, file_id: str) -> bytes:
    content = client.files.content(file_id)
    if hasattr(content, "read"):
        value = content.read()
    elif hasattr(content, "content"):
        value = content.content
    else:
        value = bytes(content)
    return value if isinstance(value, bytes) else bytes(value)


def materialize_batch_rows(
    selection: pd.DataFrame,
    results: Sequence[ParsedBatchResult],
    *,
    model: str,
    source_batch_id: str,
    source_key: str,
    run_id: str,
    openai_batch_id: str,
    retry_delay_hours: int = DEFAULT_RETRY_DELAY_HOURS,
) -> pd.DataFrame:
    source = {
        str(row["custom_id"]): row for row in selection.to_dict(orient="records")
    }
    result_map = {result.custom_id: result for result in results}
    unknown = sorted(set(result_map) - set(source))
    if unknown:
        raise ValueError(f"Unknown custom IDs: {unknown[:10]}")
    classified_at = utc_now_iso()
    retry_after = (
        utc_now() + timedelta(hours=retry_delay_hours)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output: list[dict[str, Any]] = []
    for custom_id, row in source.items():
        result = result_map.get(
            custom_id,
            ParsedBatchResult(
                custom_id,
                "failed",
                error="Missing result from completed OpenAI batch",
            ),
        )
        output.append(
            {
                "speech_id": str(row["speech_id"]),
                "speech_text_hash": str(row["speech_text_hash"]),
                "issue_label": result.label,
                "classification_status": result.status,
                "model_name": model,
                "prompt_version": PROMPT_VERSION,
                "taxonomy_version": TAXONOMY_VERSION,
                "classified_at_utc": classified_at,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "source_batch_id": source_batch_id,
                "source_batch_speech_key": source_key,
                "classification_run_id": run_id,
                "openai_response_id": result.response_id,
                "openai_batch_id": openai_batch_id,
                "review_status": "unreviewed",
                "classification_error": result.error,
                "attempt_count": int(row.get("prior_attempt_count") or 0) + 1,
                "retry_eligible_after_utc": retry_after if result.status == "failed" else "",
            }
        )
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)


def _check(name: str, passed: bool, metric: Any) -> dict[str, Any]:
    return {
        "check_name": name,
        "status": "pass" if passed else "fail",
        "metric_value": metric,
    }


def validate_candidate(
    candidate: pd.DataFrame,
    *,
    speeches: pd.DataFrame,
    new_rows: pd.DataFrame,
    max_failure_rate: float,
) -> dict[str, Any]:
    source = normalize_source_speeches(speeches)
    checks: list[dict[str, Any]] = []
    missing = sorted(PUBLISH_REQUIRED_COLUMNS - set(candidate.columns))
    checks.append(_check("required_columns", not missing, missing))

    duplicates = (
        int(candidate["speech_id"].duplicated().sum())
        if "speech_id" in candidate
        else len(candidate)
    )
    checks.append(_check("speech_id_unique", duplicates == 0, duplicates))

    classified = candidate["classification_status"].eq("classified")
    invalid = int(
        (classified & ~candidate["issue_label"].isin(ISSUE_CATEGORY_SET)).sum()
    )
    checks.append(_check("classified_labels_valid", invalid == 0, invalid))

    statuses = sorted(
        set(candidate["classification_status"].astype(str)) - {"classified", "failed"}
    )
    checks.append(_check("classification_status_valid", not statuses, statuses))

    source_hashes = dict(
        zip(source["speech_id"].astype(str), source["speech_text_hash"].astype(str))
    )
    mismatches = [
        str(row["speech_id"])
        for row in candidate.to_dict(orient="records")
        if source_hashes.get(str(row["speech_id"])) != str(row["speech_text_hash"])
    ]
    checks.append(_check("source_hash_matches", not mismatches, mismatches[:20]))

    candidate_ids = set(candidate["speech_id"].astype(str))
    source_ids = set(source["speech_id"].astype(str))
    missing_source_ids = sorted(source_ids - candidate_ids)
    unexpected_ids = sorted(candidate_ids - source_ids)
    checks.append(
        _check(
            "source_coverage_complete",
            not missing_source_ids and not unexpected_ids,
            {
                "missing_count": len(missing_source_ids),
                "unexpected_count": len(unexpected_ids),
                "missing_examples": missing_source_ids[:20],
                "unexpected_examples": unexpected_ids[:20],
            },
        )
    )

    blanks = sum(
        int(candidate[column].fillna("").astype(str).str.strip().eq("").sum())
        for column in PUBLISH_REQUIRED_COLUMNS
        if column in candidate
    )
    checks.append(_check("required_values_populated", blanks == 0, blanks))

    failed = (
        int(new_rows["classification_status"].eq("failed").sum())
        if len(new_rows)
        else 0
    )
    failure_rate = failed / len(new_rows) if len(new_rows) else 0.0
    checks.append(
        _check(
            "failure_rate_acceptable",
            failure_rate <= max_failure_rate,
            failure_rate,
        )
    )

    failed_names = [check["check_name"] for check in checks if check["status"] == "fail"]
    if not failed_names:
        candidate_status = "validated"
    elif failed_names == ["source_coverage_complete"]:
        candidate_status = "validated_partial"
    else:
        candidate_status = "validation_failed"
    return {
        "table": TABLE_NAME,
        "dq_status": "pass" if not failed_names else "fail",
        "candidate_status": candidate_status,
        "row_count": int(len(candidate)),
        "source_row_count": int(len(source)),
        "new_row_count": int(len(new_rows)),
        "failed_row_count": failed,
        "failure_rate": failure_rate,
        "max_failure_rate": max_failure_rate,
        "checks": checks,
    }


def build_compatibility_output(
    speeches: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    source = normalize_source_speeches(speeches)
    current = normalize_existing(labels)
    joined = source.merge(
        current[["speech_id", "issue_label", "classification_status"]],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )
    output = pd.DataFrame(
        {
            "speech_id": joined["speech_id"],
            "Debate Date": joined["debate_date"] if "debate_date" in joined else "",
            "Speaker Name": joined["speaker_name"] if "speaker_name" in joined else "",
            "Speech Order": joined["speech_order"] if "speech_order" in joined else "",
            "Speech Text": joined["speech_text"],
            "PoliticalIssues": joined["issue_label"].fillna(""),
            "classification_status": joined["classification_status"].fillna(
                "unclassified"
            ),
            "speech_text_hash": joined["speech_text_hash"],
        }
    )
    return output.sort_values(
        ["Debate Date", "Speech Order", "speech_id"],
        kind="stable",
    ).reset_index(drop=True)


def collect_run(
    *,
    s3: Any,
    bucket: str,
    client: OpenAI | None,
    run_id: str,
    max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
) -> dict[str, Any]:
    manifest = read_json_required(s3, bucket=bucket, key=run_manifest_key(run_id))
    if manifest.get("status") == "no_op":
        return manifest
    if manifest.get("status") == "submitted":
        if client is None:
            raise ValueError("OpenAI client required")
        manifest = refresh_batch_status(
            s3=s3,
            bucket=bucket,
            client=client,
            run_id=run_id,
        )
    if manifest.get("status") != "ready_to_collect":
        raise ValueError("Run is not ready to collect")

    selection = read_parquet_required(
        s3,
        bucket=bucket,
        key=str(manifest["selection_parquet_key"]),
    )
    deterministic = read_csv_required(
        s3,
        bucket=bucket,
        key=str(manifest["deterministic_results_key"]),
    )
    if deterministic.empty:
        deterministic = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        deterministic = normalize_existing(deterministic)

    prefix = run_prefix(run_id)
    collected_artifacts: dict[str, Any] = {}
    output_results: list[ParsedBatchResult] = []
    error_results: list[ParsedBatchResult] = []
    output_file_id = str(manifest.get("openai_output_file_id") or "")
    error_file_id = str(manifest.get("openai_error_file_id") or "")
    if output_file_id:
        if client is None:
            raise ValueError("OpenAI client required")
        output_payload = _client_file_bytes(client, output_file_id)
        collected_artifacts["openai_output_jsonl"] = put_bytes_direct(
            s3,
            bucket=bucket,
            key=f"{prefix}/openai_output.jsonl",
            payload=output_payload,
            content_type="application/jsonl",
        )
        output_results = parse_batch_output_jsonl(output_payload)
    if error_file_id:
        if client is None:
            raise ValueError("OpenAI client required")
        error_payload = _client_file_bytes(client, error_file_id)
        collected_artifacts["openai_errors_jsonl"] = put_bytes_direct(
            s3,
            bucket=bucket,
            key=f"{prefix}/openai_errors.jsonl",
            payload=error_payload,
            content_type="application/jsonl",
        )
        error_results = parse_batch_output_jsonl(error_payload)
    if (
        not output_file_id
        and not error_file_id
        and int(manifest.get("batch_request_rows") or 0) > 0
    ):
        raise ValueError("Completed batch has no output or error file")

    results = combine_batch_results(output_results, error_results)
    batch_rows = materialize_batch_rows(
        selection,
        results,
        model=str(manifest["model_name"]),
        source_batch_id=str(manifest["source_batch_id"]),
        source_key=str(manifest["source_batch_speech_key"]),
        run_id=run_id,
        openai_batch_id=str(manifest.get("openai_batch_id") or ""),
    )
    new_rows = pd.concat([deterministic, batch_rows], ignore_index=True)
    source_snapshot = read_source_snapshot(
        s3,
        bucket=bucket,
        source_key=str(manifest["source_batch_speech_key"]),
    )
    base = read_base_enrichment(
        s3,
        bucket=bucket,
        table_key=str(manifest.get("base_classification_table_key") or ""),
    )
    candidate = reconcile_results_to_source(base, new_rows, source_snapshot)
    dq = validate_candidate(
        candidate,
        speeches=source_snapshot,
        new_rows=new_rows,
        max_failure_rate=max_failure_rate,
    )

    active_source_batch_id, active_source_key, _ = read_source_context(s3, bucket=bucket)
    current_pointer = read_json_optional(
        s3,
        bucket=bucket,
        key=PRODUCTION_CLASSIFICATION_POINTER_KEY,
    )
    stale_reasons = candidate_staleness_reasons(
        manifest=manifest,
        active_source_batch_id=active_source_batch_id,
        active_source_key=active_source_key,
        current_classification_pointer=current_pointer,
    )

    table_csv = f"{prefix}/{TABLE_NAME}.csv"
    table_parquet = f"{prefix}/{TABLE_NAME}.parquet"
    compat_csv = f"{prefix}/debate_speeches_classified_compat.csv"
    compat_parquet = f"{prefix}/debate_speeches_classified_compat.parquet"
    dq_key = f"{prefix}/dq.json"
    candidate_artifacts = write_dataframe_direct(
        s3,
        bucket=bucket,
        csv_key=table_csv,
        parquet_key=table_parquet,
        df=candidate,
    )
    compatibility_artifacts = write_dataframe_direct(
        s3,
        bucket=bucket,
        csv_key=compat_csv,
        parquet_key=compat_parquet,
        df=build_compatibility_output(source_snapshot, candidate),
    )
    dq_artifact = put_json_direct(s3, bucket=bucket, key=dq_key, payload=dq)

    candidate_status = str(dq["candidate_status"])
    final_status = "stale_candidate" if stale_reasons else candidate_status
    manifest.update(
        status=final_status,
        candidate_validation_status=candidate_status,
        stale_reasons=stale_reasons,
        updated_at_utc=utc_now_iso(),
        collected_at_utc=utc_now_iso(),
        classified_rows=int(new_rows["classification_status"].eq("classified").sum()),
        failed_rows=int(new_rows["classification_status"].eq("failed").sum()),
        output_rows=int(len(candidate)),
        dq_status=dq["dq_status"],
        dq_key=dq_key,
        table_csv_key=table_csv,
        table_parquet_key=table_parquet,
        compat_csv_key=compat_csv,
        compat_parquet_key=compat_parquet,
    )
    artifact_checksums = dict(manifest.get("artifact_checksums") or {})
    artifact_checksums.update(collected_artifacts)
    artifact_checksums.update(
        {
            "candidate_csv": candidate_artifacts["csv"],
            "candidate_parquet": candidate_artifacts["parquet"],
            "compatibility_csv": compatibility_artifacts["csv"],
            "compatibility_parquet": compatibility_artifacts["parquet"],
            "dq_json": dq_artifact,
        }
    )
    manifest["artifact_checksums"] = artifact_checksums
    put_json_direct(s3, bucket=bucket, key=run_manifest_key(run_id), payload=manifest)
    return manifest


def publish_run(
    *,
    s3: Any,
    bucket: str,
    run_id: str,
    publish_enabled: bool,
) -> dict[str, Any]:
    if not publish_enabled:
        raise RuntimeError("Publication switch disabled")
    manifest = read_json_required(s3, bucket=bucket, key=run_manifest_key(run_id))
    if manifest.get("status") == "no_op":
        return manifest
    if manifest.get("status") != "validated" or manifest.get("dq_status") != "pass":
        raise RuntimeError("Run is not a complete validated publication candidate")
    if manifest.get("stale_reasons"):
        raise RuntimeError("Stale candidate cannot be published")

    active_source_batch_id, active_source_key, _ = read_source_context(s3, bucket=bucket)
    if (
        active_source_batch_id != str(manifest["source_batch_id"])
        or active_source_key != str(manifest["source_batch_speech_key"])
    ):
        raise RuntimeError("Refusing stale publication")

    current, current_etag = read_json_optional_with_etag(
        s3,
        bucket=bucket,
        key=PRODUCTION_CLASSIFICATION_POINTER_KEY,
    )
    expected_base = (
        str(manifest.get("base_classification_run_id") or ""),
        str(manifest.get("base_classification_table_key") or ""),
    )
    if pointer_identity(current) != expected_base:
        raise RuntimeError("Classification pointer changed since candidate preparation")

    pointer = {
        "table": TABLE_NAME,
        "run_id": run_id,
        "manifest_key": run_manifest_key(run_id),
        "table_csv_key": manifest["table_csv_key"],
        "table_parquet_key": manifest["table_parquet_key"],
        "compat_csv_key": manifest["compat_csv_key"],
        "compat_parquet_key": manifest["compat_parquet_key"],
        "source_batch_id": manifest["source_batch_id"],
        "model_name": manifest["model_name"],
        "prompt_version": manifest["prompt_version"],
        "taxonomy_version": manifest["taxonomy_version"],
        "published_at_utc": utc_now_iso(),
        "previous_run_id": (current or {}).get("run_id"),
    }
    pointer_artifact = put_json_conditional(
        s3,
        bucket=bucket,
        key=PRODUCTION_CLASSIFICATION_POINTER_KEY,
        payload=pointer,
        expected_etag=current_etag,
    )
    if current:
        put_json_direct(
            s3,
            bucket=bucket,
            key=PREVIOUS_CLASSIFICATION_POINTER_KEY,
            payload={
                **current,
                "superseded_at_utc": utc_now_iso(),
                "superseded_by_run_id": run_id,
            },
        )
    manifest.update(
        status="published",
        published=True,
        published_at_utc=utc_now_iso(),
        updated_at_utc=utc_now_iso(),
        production_pointer_key=PRODUCTION_CLASSIFICATION_POINTER_KEY,
    )
    artifact_checksums = dict(manifest.get("artifact_checksums") or {})
    artifact_checksums["production_pointer"] = pointer_artifact
    manifest["artifact_checksums"] = artifact_checksums
    put_json_direct(s3, bucket=bucket, key=run_manifest_key(run_id), payload=manifest)
    return manifest


def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    return (
        input_tokens * input_price_per_million / 1_000_000
        + output_tokens * output_price_per_million / 1_000_000
    )


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"expected_issue_label", "predicted_issue_label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = predictions.copy()
    frame["expected_issue_label"] = frame["expected_issue_label"].map(validate_label)
    invalid = ~frame["predicted_issue_label"].isin(ISSUE_CATEGORY_SET)
    total = len(frame)
    correct = int(
        (
            ~invalid
            & frame["expected_issue_label"].eq(frame["predicted_issue_label"])
        ).sum()
    )
    expected_none = frame["expected_issue_label"].eq("NONE")
    predicted_none = frame["predicted_issue_label"].eq("NONE")
    true_positive = int((expected_none & predicted_none).sum())
    per_category = {
        category: {
            "support": int(frame["expected_issue_label"].eq(category).sum()),
            "accuracy": (
                float(
                    frame.loc[
                        frame["expected_issue_label"].eq(category),
                        "predicted_issue_label",
                    ].eq(category).mean()
                )
                if frame["expected_issue_label"].eq(category).any()
                else None
            ),
        }
        for category in ISSUE_CATEGORIES
    }
    return {
        "rows": total,
        "overall_accuracy": correct / total if total else 0.0,
        "none_precision": (
            true_positive / int(predicted_none.sum()) if predicted_none.any() else 0.0
        ),
        "none_recall": (
            true_positive / int(expected_none.sum()) if expected_none.any() else 0.0
        ),
        "invalid_output_rate": float(invalid.mean()) if total else 0.0,
        "per_category": per_category,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    subs = parser.add_subparsers(dest="command", required=True)

    prepare = subs.add_parser("prepare")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--max-rows", type=int, default=int(os.getenv("MAX_ROWS", "25")))
    prepare.add_argument("--historical-backfill", action="store_true")
    prepare.add_argument("--force-speech-ids", default="")
    prepare.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
    )

    for name in ("submit", "status", "collect", "publish"):
        subs.add_parser(name).add_argument("--run-id", required=True)
    subs.choices["submit"].add_argument(
        "--max-submission-attempts",
        type=int,
        default=int(
            os.getenv(
                "MAX_SUBMISSION_ATTEMPTS",
                str(DEFAULT_MAX_SUBMISSION_ATTEMPTS),
            )
        ),
    )
    subs.choices["collect"].add_argument(
        "--max-failure-rate",
        type=float,
        default=float(os.getenv("MAX_FAILURE_RATE", str(DEFAULT_MAX_FAILURE_RATE))),
    )
    subs.choices["publish"].add_argument("--publish-enabled", action="store_true")

    verify = subs.add_parser("verify-models")
    verify.add_argument("--models", required=True)
    evaluation = subs.add_parser("evaluate-file")
    evaluation.add_argument("--predictions-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate-file":
        print(
            json.dumps(
                evaluate_predictions(pd.read_csv(args.predictions_csv)),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    s3 = make_s3_client(region_name=args.region)
    client = (
        OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if args.command in {"submit", "status", "collect", "verify-models"}
        else None
    )
    if args.command == "prepare":
        result = prepare_run(
            s3=s3,
            bucket=args.bucket,
            model=args.model,
            max_rows=args.max_rows,
            historical_backfill=args.historical_backfill,
            force_speech_ids=[
                value.strip()
                for value in args.force_speech_ids.split(",")
                if value.strip()
            ],
            max_retries=args.max_retries,
        )
    elif args.command == "submit":
        result = submit_run(
            s3=s3,
            bucket=args.bucket,
            client=client,
            run_id=args.run_id,
            max_submission_attempts=args.max_submission_attempts,
        )
    elif args.command == "status":
        result = refresh_batch_status(
            s3=s3,
            bucket=args.bucket,
            client=client,
            run_id=args.run_id,
        )
    elif args.command == "collect":
        result = collect_run(
            s3=s3,
            bucket=args.bucket,
            client=client,
            run_id=args.run_id,
            max_failure_rate=args.max_failure_rate,
        )
    elif args.command == "publish":
        result = publish_run(
            s3=s3,
            bucket=args.bucket,
            run_id=args.run_id,
            publish_enabled=(
                args.publish_enabled
                and os.getenv(
                    "OIREACHTAS_SPEECH_CLASSIFIER_PUBLISH_ENABLED",
                    "false",
                ).lower()
                == "true"
            ),
        )
    else:
        availability = verify_models_available(
            client,
            [value.strip() for value in args.models.split(",") if value.strip()],
        )
        print(
            json.dumps(
                {"models": availability, "all_available": all(availability.values())},
                indent=2,
            )
        )
        return 0 if all(availability.values()) else 2

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") not in {"batch_failed", "validation_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
