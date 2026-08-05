from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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
]
SOURCE_REQUIRED_COLUMNS = {"speech_id", "speech_text_hash", "speech_text"}
PUBLISH_REQUIRED_COLUMNS = {
    "speech_id", "speech_text_hash", "classification_status", "model_name",
    "prompt_version", "taxonomy_version", "classified_at_utc", "source_batch_id",
    "classification_run_id", "review_status",
}

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return f"speech_issue_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def run_prefix(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value or "/" in value or ".." in value:
        raise ValueError(f"Unsafe run_id: {run_id!r}")
    return f"{RUN_ROOT}/run_id={value}"


def run_manifest_key(run_id: str) -> str:
    return f"{run_prefix(run_id)}/manifest.json"


def _is_missing_s3_error(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    response = getattr(exc, "response", {}) or {}
    error = response.get("Error", {}) or {}
    code = str(error.get("Code") or "")
    status = int((response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode") or 0)
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def read_json_optional(s3: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return read_json_required(s3, bucket=bucket, key=key)
    except ClientError as exc:
        if _is_missing_s3_error(exc):
            return None
        raise


def put_json_direct(s3: Any, *, bucket: str, key: str, payload: Mapping[str, Any]) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode(), ContentType="application/json")


def get_bytes_required(s3: Any, *, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_parquet_required(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(get_bytes_required(s3, bucket=bucket, key=key)))


def read_csv_required(s3: Any, *, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(get_bytes_required(s3, bucket=bucket, key=key)), dtype=object)


def write_dataframe_direct(s3: Any, *, bucket: str, csv_key: str, parquet_key: str, df: pd.DataFrame) -> None:
    csv_buffer = io.StringIO(); df.to_csv(csv_buffer, index=False)
    parquet_buffer = io.BytesIO(); df.to_parquet(parquet_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=csv_key, Body=csv_buffer.getvalue().encode(), ContentType="text/csv")
    s3.put_object(Bucket=bucket, Key=parquet_key, Body=parquet_buffer.getvalue(), ContentType="application/x-parquet")


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
        raise ValueError(f"silver_speeches contains duplicate speech_id values: {sorted(set(duplicates))[:10]}")
    return current.reset_index(drop=True)


def normalize_existing(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    missing = sorted({"speech_id", "speech_text_hash", "classification_status"} - set(existing.columns))
    if missing:
        raise ValueError(f"Current classification table is missing required columns: {missing}")
    output = existing.copy()
    if output["speech_id"].astype(str).duplicated().any():
        raise ValueError("Current classification table contains duplicate speech_id values")
    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[OUTPUT_COLUMNS].reset_index(drop=True)


def validate_label(value: Any) -> str:
    label = str(value or "").strip()
    if label not in ISSUE_CATEGORY_SET:
        raise ValueError(f"Invalid issue label: {label!r}")
    return label


def select_delta(speeches: pd.DataFrame, existing: pd.DataFrame, *, force_speech_ids: Iterable[str] = (), retry_failed: bool = True) -> pd.DataFrame:
    current = normalize_source_speeches(speeches)
    prior = normalize_existing(existing)
    forced = {str(value).strip() for value in force_speech_ids if str(value).strip()}
    if prior.empty:
        return current
    prior_by_id = prior.set_index("speech_id", drop=False)
    selected: list[bool] = []
    for row in current.to_dict(orient="records"):
        speech_id = str(row["speech_id"])
        if speech_id in forced or speech_id not in prior_by_id.index:
            selected.append(True); continue
        previous = prior_by_id.loc[speech_id]
        changed = str(previous.get("speech_text_hash") or "") != str(row["speech_text_hash"])
        failed = str(previous.get("classification_status") or "") != "classified"
        selected.append(changed or (retry_failed and failed))
    return current.loc[selected].reset_index(drop=True)


def merge_results(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    prior = normalize_existing(existing)
    incoming = new_rows.copy()
    if not incoming.empty and incoming["speech_id"].astype(str).duplicated().any():
        raise ValueError("New classification rows contain duplicate speech_id values")
    for column in OUTPUT_COLUMNS:
        if column not in incoming.columns:
            incoming[column] = ""
    frames = [frame[OUTPUT_COLUMNS] for frame in (prior, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("speech_id", keep="last").sort_values("speech_id", kind="stable").reset_index(drop=True)


def structured_response_body(*, model: str, speech_text: str) -> dict[str, Any]:
    if not str(model or "").strip():
        raise ValueError("A model ID is required")
    return {"model": model, "input": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": speech_text}], "max_output_tokens": 64, "text": {"format": {"type": "json_schema", "name": "speech_issue_label", "strict": True, "schema": OUTPUT_SCHEMA}}}


def custom_id_for_speech(speech_id: str) -> str:
    return f"speech-{hashlib.sha256(str(speech_id).encode()).hexdigest()[:32]}"


def build_batch_requests(delta: pd.DataFrame, *, model: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prepared = delta.copy(); prepared["custom_id"] = prepared["speech_id"].map(custom_id_for_speech)
    if prepared["custom_id"].duplicated().any():
        raise ValueError("Generated duplicate OpenAI custom_id values")
    requests = [{"custom_id": row["custom_id"], "method": "POST", "url": BATCH_ENDPOINT, "body": structured_response_body(model=model, speech_text=str(row["speech_text"]))} for row in prepared.to_dict(orient="records")]
    return prepared, requests


def parse_response_output_text(body: Mapping[str, Any]) -> str:
    direct = str(body.get("output_text") or "").strip()
    if direct: return direct
    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, Mapping): continue
        for content in item.get("content", []) or []:
            if not isinstance(content, Mapping): continue
            if content.get("type") == "refusal": raise ValueError("OpenAI refusal")
            if content.get("text"): chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def parse_structured_label(body: Mapping[str, Any]) -> str:
    raw = parse_response_output_text(body)
    if not raw: raise ValueError("OpenAI response contained no output text")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping): raise ValueError("Structured output was not a JSON object")
    return validate_label(payload.get("issue_label"))


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def parse_batch_result_line(line: Mapping[str, Any]) -> ParsedBatchResult:
    custom_id = str(line.get("custom_id") or "").strip()
    if not custom_id: raise ValueError("Batch result line is missing custom_id")
    if line.get("error"): return ParsedBatchResult(custom_id, "failed", error=json.dumps(line["error"], sort_keys=True))
    response = line.get("response")
    if not isinstance(response, Mapping): return ParsedBatchResult(custom_id, "failed", error="Missing response object")
    body = response.get("body"); status_code = int(response.get("status_code") or 0)
    if status_code < 200 or status_code >= 300 or not isinstance(body, Mapping): return ParsedBatchResult(custom_id, "failed", error=f"HTTP {status_code}")
    try: label = parse_structured_label(body)
    except Exception as exc: return ParsedBatchResult(custom_id, "failed", response_id=str(body.get("id") or ""), error=f"{type(exc).__name__}: {exc}")
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    return ParsedBatchResult(custom_id, "classified", label, str(body.get("id") or ""), _optional_int(usage.get("input_tokens")), _optional_int(usage.get("output_tokens")))


def parse_batch_output_jsonl(payload: bytes) -> list[ParsedBatchResult]:
    results = [parse_batch_result_line(json.loads(line)) for line in payload.decode().splitlines() if line.strip()]
    if len({r.custom_id for r in results}) != len(results): raise ValueError("Batch output contains duplicate custom_id values")
    return results


def read_source_context(s3: Any, *, bucket: str) -> tuple[str, str, pd.DataFrame]:
    pointer = read_json_required(s3, bucket=bucket, key=PRODUCTION_POINTER_KEY)
    source_batch_id = str(pointer.get("batch_id") or "").strip()
    if not source_batch_id: raise ValueError("Active Oireachtas production pointer has no batch_id")
    source_key = resolve_production_key(s3, bucket=bucket, production_key=SOURCE_LOGICAL_KEY)
    return source_batch_id, source_key, normalize_source_speeches(read_parquet_required(s3, bucket=bucket, key=source_key))


def read_current_enrichment(s3: Any, *, bucket: str) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    pointer = read_json_optional(s3, bucket=bucket, key=PRODUCTION_CLASSIFICATION_POINTER_KEY)
    if pointer is None: return None, pd.DataFrame(columns=OUTPUT_COLUMNS)
    key = str(pointer.get("table_parquet_key") or "").strip()
    if not key: raise ValueError("Classification production pointer is missing table_parquet_key")
    return pointer, normalize_existing(read_parquet_required(s3, bucket=bucket, key=key))


def _write_jsonl_direct(s3: Any, *, bucket: str, key: str, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode(), ContentType="application/jsonl")


def _results_from_deterministic(rows: pd.DataFrame, *, model: str, source_batch_id: str, source_key: str, run_id: str) -> pd.DataFrame:
    now = utc_now_iso(); output = []
    for row in rows.to_dict(orient="records"):
        output.append({"speech_id": str(row["speech_id"]), "speech_text_hash": str(row["speech_text_hash"]), "issue_label": "NONE", "classification_status": "classified", "model_name": f"{model}:deterministic-short-speech", "prompt_version": PROMPT_VERSION, "taxonomy_version": TAXONOMY_VERSION, "classified_at_utc": now, "input_tokens": 0, "output_tokens": 0, "source_batch_id": source_batch_id, "source_batch_speech_key": source_key, "classification_run_id": run_id, "openai_response_id": "", "openai_batch_id": "", "review_status": "unreviewed", "classification_error": ""})
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)


def prepare_run(*, s3: Any, bucket: str, model: str, max_rows: int, historical_backfill: bool, force_speech_ids: Iterable[str] = (), short_speech_word_limit: int = DEFAULT_SHORT_SPEECH_WORD_LIMIT, run_id: str | None = None) -> dict[str, Any]:
    if max_rows < 0: raise ValueError("max_rows cannot be negative")
    if historical_backfill and os.getenv("OIREACHTAS_SPEECH_CLASSIFIER_BACKFILL_ENABLED", "false").lower() != "true": raise RuntimeError("Historical backfill switch is disabled")
    run_id = run_id or new_run_id(); prefix = run_prefix(run_id)
    source_batch_id, source_key, speeches = read_source_context(s3, bucket=bucket)
    current_pointer, existing = read_current_enrichment(s3, bucket=bucket)
    delta = select_delta(speeches, existing, force_speech_ids=force_speech_ids)
    if max_rows > 0: delta = delta.head(max_rows).copy()
    deterministic = delta.loc[delta["speech_text"].str.split().str.len() < short_speech_word_limit].reset_index(drop=True)
    paid_delta = delta.drop(index=delta.loc[delta["speech_text"].str.split().str.len() < short_speech_word_limit].index).reset_index(drop=True)
    selection, requests = build_batch_requests(paid_delta, model=model)
    deterministic_rows = _results_from_deterministic(deterministic, model=model, source_batch_id=source_batch_id, source_key=source_key, run_id=run_id)
    selection_csv_key=f"{prefix}/selection.csv"; selection_parquet_key=f"{prefix}/selection.parquet"; requests_key=f"{prefix}/openai_requests.jsonl"; deterministic_key=f"{prefix}/deterministic_results.csv"
    write_dataframe_direct(s3,bucket=bucket,csv_key=selection_csv_key,parquet_key=selection_parquet_key,df=selection); _write_jsonl_direct(s3,bucket=bucket,key=requests_key,rows=requests)
    buf=io.StringIO(); deterministic_rows.to_csv(buf,index=False); s3.put_object(Bucket=bucket,Key=deterministic_key,Body=buf.getvalue().encode(),ContentType="text/csv")
    status="no_op" if delta.empty else ("ready_to_collect" if not requests else "prepared")
    manifest={"table":TABLE_NAME,"run_id":run_id,"status":status,"created_at_utc":utc_now_iso(),"updated_at_utc":utc_now_iso(),"source_batch_id":source_batch_id,"source_batch_speech_key":source_key,"previous_classification_run_id":(current_pointer or {}).get("run_id"),"model_name":model,"prompt_version":PROMPT_VERSION,"taxonomy_version":TAXONOMY_VERSION,"historical_backfill":bool(historical_backfill),"source_rows":len(speeches),"existing_rows":len(existing),"delta_rows_selected":len(delta),"deterministic_none_rows":len(deterministic_rows),"batch_request_rows":len(requests),"max_rows":max_rows,"short_speech_word_limit":short_speech_word_limit,"selection_csv_key":selection_csv_key,"selection_parquet_key":selection_parquet_key,"requests_jsonl_key":requests_key,"deterministic_results_key":deterministic_key,"manifest_key":run_manifest_key(run_id),"openai_batch_id":"","openai_input_file_id":"","openai_output_file_id":"","openai_error_file_id":"","published":False}
    put_json_direct(s3,bucket=bucket,key=run_manifest_key(run_id),payload=manifest); return manifest


def verify_models_available(client: OpenAI, models: Iterable[str]) -> dict[str, bool]:
    results={}
    for model in models:
        model_id=str(model).strip()
        if not model_id: continue
        try: client.models.retrieve(model_id); results[model_id]=True
        except Exception: results[model_id]=False
    return results


def submit_run(*, s3: Any, bucket: str, client: OpenAI, run_id: str) -> dict[str, Any]:
    manifest=read_json_required(s3,bucket=bucket,key=run_manifest_key(run_id))
    if manifest.get("status")=="no_op" or manifest.get("openai_batch_id"): return manifest
    if manifest.get("status")!="prepared": raise ValueError("Run is not ready for submission")
    model=str(manifest["model_name"])
    if not verify_models_available(client,[model]).get(model): raise RuntimeError(f"Model unavailable: {model}")
    request_bytes=get_bytes_required(s3,bucket=bucket,key=manifest["requests_jsonl_key"])
    upload=io.BytesIO(request_bytes); upload.name=f"{run_id}.jsonl"
    input_file=client.files.create(file=upload,purpose="batch")
    batch=client.batches.create(input_file_id=input_file.id,endpoint=BATCH_ENDPOINT,completion_window=BATCH_COMPLETION_WINDOW,metadata={"pipeline":TABLE_NAME,"run_id":run_id,"source_batch_id":str(manifest["source_batch_id"]),"prompt_version":PROMPT_VERSION})
    manifest.update(status="submitted",updated_at_utc=utc_now_iso(),submitted_at_utc=utc_now_iso(),openai_input_file_id=str(input_file.id),openai_batch_id=str(batch.id),openai_batch_status=str(batch.status)); put_json_direct(s3,bucket=bucket,key=run_manifest_key(run_id),payload=manifest); return manifest


def refresh_batch_status(*, s3: Any, bucket: str, client: OpenAI, run_id: str) -> dict[str, Any]:
    manifest=read_json_required(s3,bucket=bucket,key=run_manifest_key(run_id)); batch_id=str(manifest.get("openai_batch_id") or "")
    if not batch_id: return manifest
    batch=client.batches.retrieve(batch_id); manifest.update(openai_batch_status=str(batch.status),openai_output_file_id=str(getattr(batch,"output_file_id","") or ""),openai_error_file_id=str(getattr(batch,"error_file_id","") or ""),updated_at_utc=utc_now_iso())
    if batch.status=="completed": manifest["status"]="ready_to_collect"
    elif batch.status in {"failed","expired","cancelled"}: manifest["status"]="batch_failed"
    put_json_direct(s3,bucket=bucket,key=run_manifest_key(run_id),payload=manifest); return manifest


def _client_file_bytes(client: OpenAI, file_id: str) -> bytes:
    content=client.files.content(file_id)
    if hasattr(content,"read"): return content.read()
    return content.content if hasattr(content,"content") else bytes(content)


def materialize_batch_rows(selection: pd.DataFrame, results: Sequence[ParsedBatchResult], *, model: str, source_batch_id: str, source_key: str, run_id: str, openai_batch_id: str) -> pd.DataFrame:
    source={str(row["custom_id"]):row for row in selection.to_dict(orient="records")}; result_map={r.custom_id:r for r in results}
    unknown=sorted(set(result_map)-set(source))
    if unknown: raise ValueError(f"Unknown custom IDs: {unknown[:10]}")
    now=utc_now_iso(); output=[]
    for custom_id,row in source.items():
        result=result_map.get(custom_id,ParsedBatchResult(custom_id,"failed",error="Missing result from completed OpenAI batch"))
        output.append({"speech_id":str(row["speech_id"]),"speech_text_hash":str(row["speech_text_hash"]),"issue_label":result.label,"classification_status":result.status,"model_name":model,"prompt_version":PROMPT_VERSION,"taxonomy_version":TAXONOMY_VERSION,"classified_at_utc":now,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,"source_batch_id":source_batch_id,"source_batch_speech_key":source_key,"classification_run_id":run_id,"openai_response_id":result.response_id,"openai_batch_id":openai_batch_id,"review_status":"unreviewed","classification_error":result.error})
    return pd.DataFrame(output,columns=OUTPUT_COLUMNS)


def _check(name: str, passed: bool, metric: Any) -> dict[str, Any]: return {"check_name":name,"status":"pass" if passed else "fail","metric_value":metric}


def validate_candidate(candidate: pd.DataFrame, *, speeches: pd.DataFrame, new_rows: pd.DataFrame, max_failure_rate: float) -> dict[str, Any]:
    checks=[]; missing=sorted(PUBLISH_REQUIRED_COLUMNS-set(candidate.columns)); checks.append(_check("required_columns",not missing,missing))
    duplicates=int(candidate["speech_id"].duplicated().sum()) if "speech_id" in candidate else len(candidate); checks.append(_check("speech_id_unique",duplicates==0,duplicates))
    classified=candidate["classification_status"].eq("classified"); invalid=int((classified & ~candidate["issue_label"].isin(ISSUE_CATEGORY_SET)).sum()); checks.append(_check("classified_labels_valid",invalid==0,invalid))
    statuses=sorted(set(candidate["classification_status"].astype(str))-{"classified","failed"}); checks.append(_check("classification_status_valid",not statuses,statuses))
    hashes=dict(zip(speeches["speech_id"].astype(str),speeches["speech_text_hash"].astype(str))); mismatches=[str(r["speech_id"]) for r in candidate.to_dict(orient="records") if hashes.get(str(r["speech_id"]))!=str(r["speech_text_hash"])]; checks.append(_check("source_hash_matches",not mismatches,mismatches[:20]))
    blanks=sum(int(candidate[c].fillna("").astype(str).str.strip().eq("").sum()) for c in PUBLISH_REQUIRED_COLUMNS if c in candidate); checks.append(_check("required_values_populated",blanks==0,blanks))
    failed=int(new_rows["classification_status"].eq("failed").sum()) if len(new_rows) else 0; rate=failed/len(new_rows) if len(new_rows) else 0.0; checks.append(_check("failure_rate_acceptable",rate<=max_failure_rate,rate))
    return {"table":TABLE_NAME,"dq_status":"pass" if all(c["status"]=="pass" for c in checks) else "fail","row_count":len(candidate),"new_row_count":len(new_rows),"failed_row_count":failed,"failure_rate":rate,"max_failure_rate":max_failure_rate,"checks":checks}


def build_compatibility_output(speeches: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    joined=normalize_source_speeches(speeches).merge(normalize_existing(labels)[["speech_id","issue_label","classification_status"]],on="speech_id",how="left",validate="one_to_one")
    output=pd.DataFrame({"speech_id":joined["speech_id"],"Debate Date":joined["debate_date"] if "debate_date" in joined else "","Speaker Name":joined["speaker_name"] if "speaker_name" in joined else "","Speech Order":joined["speech_order"] if "speech_order" in joined else "","Speech Text":joined["speech_text"],"PoliticalIssues":joined["issue_label"].fillna(""),"classification_status":joined["classification_status"].fillna("unclassified"),"speech_text_hash":joined["speech_text_hash"]})
    return output.sort_values(["Debate Date","Speech Order","speech_id"],kind="stable").reset_index(drop=True)


def collect_run(*, s3: Any, bucket: str, client: OpenAI | None, run_id: str, max_failure_rate: float=DEFAULT_MAX_FAILURE_RATE) -> dict[str, Any]:
    manifest=read_json_required(s3,bucket=bucket,key=run_manifest_key(run_id))
    if manifest.get("status")=="no_op": return manifest
    if manifest.get("status")=="submitted":
        if client is None: raise ValueError("OpenAI client required")
        manifest=refresh_batch_status(s3=s3,bucket=bucket,client=client,run_id=run_id)
    if manifest.get("status")!="ready_to_collect": raise ValueError("Run is not ready to collect")
    selection=read_parquet_required(s3,bucket=bucket,key=manifest["selection_parquet_key"]); deterministic=read_csv_required(s3,bucket=bucket,key=manifest["deterministic_results_key"])
    if deterministic.empty: deterministic=pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        for c in OUTPUT_COLUMNS:
            if c not in deterministic: deterministic[c]=""
        deterministic=deterministic[OUTPUT_COLUMNS]
    results=[]; output_file_id=str(manifest.get("openai_output_file_id") or "")
    if output_file_id:
        if client is None: raise ValueError("OpenAI client required")
        results=parse_batch_output_jsonl(_client_file_bytes(client,output_file_id))
    elif int(manifest.get("batch_request_rows") or 0)>0: raise ValueError("Completed batch has no output_file_id")
    batch_rows=materialize_batch_rows(selection,results,model=manifest["model_name"],source_batch_id=manifest["source_batch_id"],source_key=manifest["source_batch_speech_key"],run_id=run_id,openai_batch_id=str(manifest.get("openai_batch_id") or "")); new_rows=pd.concat([deterministic,batch_rows],ignore_index=True)
    _,existing=read_current_enrichment(s3,bucket=bucket); merged=merge_results(existing,new_rows)
    source_batch_id,source_key,speeches=read_source_context(s3,bucket=bucket)
    if source_batch_id!=manifest["source_batch_id"] or source_key!=manifest["source_batch_speech_key"]: raise RuntimeError("Active source batch changed")
    dq=validate_candidate(merged,speeches=speeches,new_rows=new_rows,max_failure_rate=max_failure_rate); prefix=run_prefix(run_id)
    table_csv=f"{prefix}/{TABLE_NAME}.csv"; table_parquet=f"{prefix}/{TABLE_NAME}.parquet"; compat_csv=f"{prefix}/debate_speeches_classified_compat.csv"; compat_parquet=f"{prefix}/debate_speeches_classified_compat.parquet"; dq_key=f"{prefix}/dq.json"
    write_dataframe_direct(s3,bucket=bucket,csv_key=table_csv,parquet_key=table_parquet,df=merged); write_dataframe_direct(s3,bucket=bucket,csv_key=compat_csv,parquet_key=compat_parquet,df=build_compatibility_output(speeches,merged)); put_json_direct(s3,bucket=bucket,key=dq_key,payload=dq)
    manifest.update(status="validated" if dq["dq_status"]=="pass" else "validation_failed",updated_at_utc=utc_now_iso(),collected_at_utc=utc_now_iso(),classified_rows=int(new_rows["classification_status"].eq("classified").sum()),failed_rows=int(new_rows["classification_status"].eq("failed").sum()),output_rows=len(merged),dq_status=dq["dq_status"],dq_key=dq_key,table_csv_key=table_csv,table_parquet_key=table_parquet,compat_csv_key=compat_csv,compat_parquet_key=compat_parquet); put_json_direct(s3,bucket=bucket,key=run_manifest_key(run_id),payload=manifest); return manifest


def publish_run(*, s3: Any, bucket: str, run_id: str, publish_enabled: bool) -> dict[str, Any]:
    if not publish_enabled: raise RuntimeError("Publication switch disabled")
    manifest=read_json_required(s3,bucket=bucket,key=run_manifest_key(run_id))
    if manifest.get("status")=="no_op": return manifest
    if manifest.get("status")!="validated" or manifest.get("dq_status")!="pass": raise RuntimeError("Run is not validated")
    active=read_json_required(s3,bucket=bucket,key=PRODUCTION_POINTER_KEY)
    if str(active.get("batch_id") or "")!=str(manifest["source_batch_id"]): raise RuntimeError("Refusing stale publication")
    current=read_json_optional(s3,bucket=bucket,key=PRODUCTION_CLASSIFICATION_POINTER_KEY)
    if current: put_json_direct(s3,bucket=bucket,key=PREVIOUS_CLASSIFICATION_POINTER_KEY,payload={**current,"superseded_at_utc":utc_now_iso(),"superseded_by_run_id":run_id})
    pointer={"table":TABLE_NAME,"run_id":run_id,"manifest_key":run_manifest_key(run_id),"table_csv_key":manifest["table_csv_key"],"table_parquet_key":manifest["table_parquet_key"],"compat_csv_key":manifest["compat_csv_key"],"compat_parquet_key":manifest["compat_parquet_key"],"source_batch_id":manifest["source_batch_id"],"model_name":manifest["model_name"],"prompt_version":manifest["prompt_version"],"taxonomy_version":manifest["taxonomy_version"],"published_at_utc":utc_now_iso(),"previous_run_id":(current or {}).get("run_id")}
    put_json_direct(s3,bucket=bucket,key=PRODUCTION_CLASSIFICATION_POINTER_KEY,payload=pointer); manifest.update(status="published",published=True,published_at_utc=utc_now_iso(),updated_at_utc=utc_now_iso(),production_pointer_key=PRODUCTION_CLASSIFICATION_POINTER_KEY); put_json_direct(s3,bucket=bucket,key=run_manifest_key(run_id),payload=manifest); return manifest


def estimate_cost(*, input_tokens:int, output_tokens:int, input_price_per_million:float, output_price_per_million:float)->float: return input_tokens*input_price_per_million/1_000_000+output_tokens*output_price_per_million/1_000_000


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    required={"expected_issue_label","predicted_issue_label"}; missing=sorted(required-set(predictions.columns))
    if missing: raise ValueError(f"Missing columns: {missing}")
    frame=predictions.copy(); frame["expected_issue_label"]=frame["expected_issue_label"].map(validate_label); invalid=~frame["predicted_issue_label"].isin(ISSUE_CATEGORY_SET); total=len(frame); correct=int((~invalid & frame["expected_issue_label"].eq(frame["predicted_issue_label"])).sum()); expected_none=frame["expected_issue_label"].eq("NONE"); predicted_none=frame["predicted_issue_label"].eq("NONE"); tp=int((expected_none&predicted_none).sum())
    per={c:{"support":int(frame["expected_issue_label"].eq(c).sum()),"accuracy":float(frame.loc[frame["expected_issue_label"].eq(c),"predicted_issue_label"].eq(c).mean()) if frame["expected_issue_label"].eq(c).any() else None} for c in ISSUE_CATEGORIES}
    return {"rows":total,"overall_accuracy":correct/total if total else 0.0,"none_precision":tp/int(predicted_none.sum()) if predicted_none.any() else 0.0,"none_recall":tp/int(expected_none.sum()) if expected_none.any() else 0.0,"invalid_output_rate":float(invalid.mean()) if total else 0.0,"per_category":per}


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(); parser.add_argument("--bucket",default=os.getenv("S3_BUCKET",DEFAULT_BUCKET)); parser.add_argument("--region",default=os.getenv("AWS_REGION",DEFAULT_REGION)); subs=parser.add_subparsers(dest="command",required=True)
    p=subs.add_parser("prepare"); p.add_argument("--model",required=True); p.add_argument("--max-rows",type=int,default=int(os.getenv("MAX_ROWS","25"))); p.add_argument("--historical-backfill",action="store_true"); p.add_argument("--force-speech-ids",default="")
    for name in ("submit","status","collect","publish"): subs.add_parser(name).add_argument("--run-id",required=True)
    subs.choices["collect"].add_argument("--max-failure-rate",type=float,default=float(os.getenv("MAX_FAILURE_RATE",str(DEFAULT_MAX_FAILURE_RATE)))); subs.choices["publish"].add_argument("--publish-enabled",action="store_true")
    v=subs.add_parser("verify-models"); v.add_argument("--models",required=True); e=subs.add_parser("evaluate-file"); e.add_argument("--predictions-csv",required=True); return parser


def main(argv: Sequence[str] | None=None) -> int:
    args=build_parser().parse_args(argv)
    if args.command=="evaluate-file": print(json.dumps(evaluate_predictions(pd.read_csv(args.predictions_csv)),indent=2,sort_keys=True)); return 0
    s3=make_s3_client(region_name=args.region); client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]) if args.command in {"submit","status","collect","verify-models"} else None
    if args.command=="prepare": result=prepare_run(s3=s3,bucket=args.bucket,model=args.model,max_rows=args.max_rows,historical_backfill=args.historical_backfill,force_speech_ids=[x.strip() for x in args.force_speech_ids.split(",") if x.strip()])
    elif args.command=="submit": result=submit_run(s3=s3,bucket=args.bucket,client=client,run_id=args.run_id)
    elif args.command=="status": result=refresh_batch_status(s3=s3,bucket=args.bucket,client=client,run_id=args.run_id)
    elif args.command=="collect": result=collect_run(s3=s3,bucket=args.bucket,client=client,run_id=args.run_id,max_failure_rate=args.max_failure_rate)
    elif args.command=="publish": result=publish_run(s3=s3,bucket=args.bucket,run_id=args.run_id,publish_enabled=args.publish_enabled and os.getenv("OIREACHTAS_SPEECH_CLASSIFIER_PUBLISH_ENABLED","false").lower()=="true")
    else:
        availability=verify_models_available(client,[x.strip() for x in args.models.split(",") if x.strip()]); print(json.dumps({"models":availability,"all_available":all(availability.values())},indent=2)); return 0 if all(availability.values()) else 2
    print(json.dumps(result,indent=2,sort_keys=True,default=str)); return 0 if result.get("status") not in {"batch_failed","validation_failed"} else 1


if __name__ == "__main__": raise SystemExit(main())
