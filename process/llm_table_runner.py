#!/usr/bin/env python3
"""
llm_table_runner.py

Generic, resumable table->LLM->table runner driven by a YAML config.

Reads CSV from S3, keeps selected columns, creates up to 5 prompt variables from columns,
calls OpenAI Responses API (optionally with web_search), writes output CSV + Parquet to S3.

Write modes:
- full_table: write the full dataset (input rows + kept cols + output col)
- processed_only: write only rows processed in this run (plus id + kept cols + output col)

Overwrite mode:
- overwrite_existing: true  => recompute output_col for ALL rows (or first TEST_ROWS) and overwrite values
- overwrite_existing: false => only fill missing output_col values (default resumable behavior)

Citation removal is toggleable.
"""

import io
import os
import re
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from openai import OpenAI

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import yaml  # pyyaml
except ImportError as e:
    raise RuntimeError("Missing dependency: pyyaml. Add it to requirements.txt or pip install pyyaml") from e


# ---------------- S3 HELPERS ----------------
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def s3_get_text(s3, bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8-sig", errors="replace")


def s3_put_text(s3, bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8-sig"), ContentType=content_type)


def s3_put_bytes(s3, bucket: str, key: str, b: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=b, ContentType=content_type)


def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# ---------------- UTILS ----------------
def is_missing(v) -> bool:
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, str) and v.strip().lower() in {"nan", "na", "null"}:
        return True
    return False


def is_gpt5_family(model_name: str) -> bool:
    return (model_name or "").strip().lower().startswith("gpt-5")


def _get_attr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def extract_text_from_response(resp: Any) -> str:
    text = (_get_attr(resp, "output_text", "") or "").strip()
    if text:
        return text

    out_items = _get_attr(resp, "output", []) or []
    chunks: List[str] = []
    for item in out_items:
        if _get_attr(item, "type", None) != "message":
            continue
        content_items = _get_attr(item, "content", []) or []
        for c in content_items:
            if _get_attr(c, "type", None) in ("output_text", "text"):
                t = _get_attr(c, "text", None)
                if t:
                    chunks.append(str(t))
    return "\n".join(chunks).strip()


_CIT_RE = re.compile(r"\s*\[\d+\]\s*")
def strip_inline_citations(text: str) -> str:
    t = _CIT_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", t).strip()


def clamp_to_max_words(s: str, max_words: int) -> str:
    if max_words <= 0:
        return (s or "").strip()
    parts = (s or "").strip().split()
    if len(parts) <= max_words:
        return " ".join(parts).strip()
    return " ".join(parts[:max_words]).strip()


def sha_row_id(values: List[str], length: int = 24) -> str:
    raw = "||".join(values).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:length]


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


# ---------------- CONFIG ----------------
@dataclass
class TaskConfig:
    bucket: str
    region: str
    input_key: str
    output_csv_key: str
    output_parquet_key: str

    keep_cols: List[str]
    id_col: Optional[str]
    id_hash_cols: List[str]
    var_cols: List[Optional[str]]  # up to 5
    output_col: str

    prompt_template: str

    model: str
    use_web_search: bool
    strip_citations: bool
    reasoning_effort: str
    verbosity: str
    temperature: Optional[float]
    max_output_tokens: int

    max_retries: int
    delay_between_requests: float
    autosave_interval: int
    test_rows: int

    write_mode: str  # full_table | processed_only
    overwrite_existing: bool

    # validation
    require_non_empty: bool
    max_words: int
    regex_must_match: Optional[str]


def load_config(path: str) -> TaskConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s3cfg = cfg["s3"]
    colcfg = cfg["columns"]
    llm = cfg["llm"]
    run = cfg.get("run", {})
    val = cfg.get("validation", {})
    wmode = cfg.get("write_mode", "full_table")

    # Vars (pad to 5)
    vars_list = (colcfg.get("vars") or [])[:5]
    while len(vars_list) < 5:
        vars_list.append(None)

    use_web = bool(llm.get("use_web_search", False))
    temp = llm.get("temperature", None)
    if (temp is None) and (not use_web) and (not is_gpt5_family(llm.get("model", ""))):
        temp = 0.0

    return TaskConfig(
        bucket=str(s3cfg.get("bucket", "eirepolitic-data")),
        region=str(s3cfg.get("region", os.getenv("AWS_REGION", "us-east-2"))),
        input_key=str(s3cfg["input_key"]),
        output_csv_key=str(s3cfg["output_csv_key"]),
        output_parquet_key=str(s3cfg["output_parquet_key"]),

        keep_cols=list(colcfg.get("keep") or []),
        id_col=colcfg.get("id"),
        id_hash_cols=list(colcfg.get("id_hash_cols") or []),
        var_cols=vars_list,
        output_col=str(colcfg["output_col"]),

        prompt_template=str(cfg["prompt_template"]),

        model=str(llm.get("model", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))),
        use_web_search=use_web,
        strip_citations=bool(llm.get("strip_citations", False)),
        reasoning_effort=str(llm.get("reasoning_effort", os.getenv("OPENAI_REASONING_EFFORT", "low"))),
        verbosity=str(llm.get("verbosity", os.getenv("OPENAI_VERBOSITY", "low"))),
        temperature=temp,
        max_output_tokens=int(llm.get("max_output_tokens", 0)),

        max_retries=int(run.get("max_retries", int(os.getenv("MAX_RETRIES", "5")))),
        delay_between_requests=float(run.get("delay_between_requests", float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.25")))),
        autosave_interval=int(run.get("autosave_interval", int(os.getenv("AUTOSAVE_INTERVAL", "25")))),
        test_rows=int(run.get("test_rows", int(os.getenv("TEST_ROWS", "0")))),

        write_mode=str(wmode),
        overwrite_existing=bool(run.get("overwrite_existing", False)),

        require_non_empty=bool(val.get("require_non_empty", True)),
        max_words=int(val.get("max_words", 0)),
        regex_must_match=val.get("regex_must_match", None),
    )


# ---------------- PROMPT RENDER ----------------
def render_prompt(template: str, vars_map: Dict[str, str]) -> str:
    return template.format(**vars_map)


# ---------------- OPENAI CALL ----------------
def call_openai(client: OpenAI, cfg: TaskConfig, prompt: str) -> str:
    last_err = None

    reasoning_effort = cfg.reasoning_effort
    if cfg.use_web_search and is_gpt5_family(cfg.model) and reasoning_effort == "minimal":
        reasoning_effort = "low"

    max_tokens = cfg.max_output_tokens if cfg.max_output_tokens > 0 else 320

    for attempt in range(1, cfg.max_retries + 1):
        try:
            payload: Dict[str, Any] = {
                "model": cfg.model,
                "input": prompt,
                "max_output_tokens": max_tokens,
            }

            if cfg.use_web_search:
                payload["tools"] = [{"type": "web_search"}]
                payload["tool_choice"] = "auto"

            if is_gpt5_family(cfg.model):
                payload["reasoning"] = {"effort": reasoning_effort}
                payload["text"] = {"verbosity": cfg.verbosity}
            else:
                if cfg.temperature is not None:
                    payload["temperature"] = float(cfg.temperature)

            resp = client.responses.create(**payload)
            out = extract_text_from_response(resp)

            if cfg.strip_citations:
                out = strip_inline_citations(out)

            return (out or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2.0 * attempt)

    raise RuntimeError(f"OpenAI call failed after {cfg.max_retries} attempts: {last_err}")


# ---------------- VALIDATION ----------------
def validate_output(cfg: TaskConfig, text: str) -> Tuple[bool, str, str]:
    t = (text or "").strip()

    if cfg.require_non_empty and not t:
        return False, "empty_output", t

    if cfg.max_words and cfg.max_words > 0:
        t = clamp_to_max_words(t, cfg.max_words)

    if cfg.regex_must_match:
        if not re.search(cfg.regex_must_match, t or ""):
            return False, "regex_failed", t

    return True, "ok", t


# ---------------- OUTPUT WRITER ----------------
def write_outputs(s3, cfg: TaskConfig, df_res: pd.DataFrame, processed_rows: List[Dict[str, Any]]) -> None:
    if cfg.write_mode == "processed_only":
        df_write = pd.DataFrame(processed_rows)
    else:
        df_write = df_res

    buf = io.StringIO()
    df_write.to_csv(buf, index=False, encoding="utf-8-sig")
    s3_put_text(s3, cfg.bucket, cfg.output_csv_key, buf.getvalue())

    pq_bytes = df_to_parquet_bytes(df_write)
    s3_put_bytes(s3, cfg.bucket, cfg.output_parquet_key, pq_bytes, content_type="application/x-parquet")


# ---------------- MAIN RUN ----------------
def run_task(cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    s3 = s3_client(cfg.region)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"📥 Input:  s3://{cfg.bucket}/{cfg.input_key}")
    print(f"📤 Output: s3://{cfg.bucket}/{cfg.output_csv_key}")
    print(f"📦 Parquet:s3://{cfg.bucket}/{cfg.output_parquet_key}")
    print(f"🧠 Model:  {cfg.model} | web_search={cfg.use_web_search} | overwrite_existing={cfg.overwrite_existing}\n")

    in_text = s3_get_text(s3, cfg.bucket, cfg.input_key)
    df_in = pd.read_csv(io.StringIO(in_text))

    # Keep selected cols (if empty, keep all)
    if cfg.keep_cols:
        missing = [c for c in cfg.keep_cols if c not in df_in.columns]
        if missing:
            raise RuntimeError(f"Missing keep columns in input: {missing}")
        df_base = df_in[cfg.keep_cols].copy()
    else:
        df_base = df_in.copy()

    # Ensure ID
    if cfg.id_col and cfg.id_col in df_base.columns:
        df_base[cfg.id_col] = df_base[cfg.id_col].astype(str).str.strip()
    else:
        if not cfg.id_hash_cols:
            raise RuntimeError("No id_col present and id_hash_cols not provided.")
        missing = [c for c in cfg.id_hash_cols if c not in df_base.columns]
        if missing:
            raise RuntimeError(f"Missing id_hash_cols in input: {missing}")
        df_base["_row_id"] = df_base.apply(
            lambda r: sha_row_id([str(r.get(c, "")).strip() for c in cfg.id_hash_cols]),
            axis=1,
        )
        cfg.id_col = "_row_id"

    # Load existing output if present (so we don't lose other columns you're keeping)
    if s3_exists(s3, cfg.bucket, cfg.output_csv_key):
        out_text = s3_get_text(s3, cfg.bucket, cfg.output_csv_key)
        df_out = pd.read_csv(io.StringIO(out_text))
        if cfg.id_col not in df_out.columns:
            raise RuntimeError(f"Existing output missing id column '{cfg.id_col}'.")
    else:
        df_out = pd.DataFrame(columns=list(df_base.columns) + [cfg.output_col])

    # Build existing map by id for this output_col only
    existing: Dict[str, str] = {}
    if (not df_out.empty) and (cfg.output_col in df_out.columns):
        for rid, val in zip(df_out[cfg.id_col].astype(str), df_out[cfg.output_col]):
            if rid and not is_missing(val):
                existing[rid.strip()] = str(val).strip()

    # Start df_res as base; add output_col
    df_res = df_base.copy()
    if cfg.output_col not in df_res.columns:
        df_res[cfg.output_col] = pd.NA

    # Pre-fill from existing unless overwrite is enabled
    if cfg.overwrite_existing:
        df_res[cfg.output_col] = pd.NA
    else:
        df_res[cfg.output_col] = df_res[cfg.id_col].astype(str).map(
            lambda rid: existing.get(str(rid).strip(), pd.NA)
        )

    # Determine work rows
    if cfg.overwrite_existing:
        idxs = df_res.index.tolist()
    else:
        idxs = df_res.index[df_res[cfg.output_col].apply(is_missing)].tolist()

    if cfg.test_rows and cfg.test_rows > 0:
        idxs = idxs[:cfg.test_rows]
        print(f"🧪 Test mode: {len(idxs)} rows.\n")
    else:
        print(f"📄 Rows needing output: {len(idxs)}\n")

    processed_rows: List[Dict[str, Any]] = []
    done = 0

    for n, idx in enumerate(idxs, start=1):
        row = df_res.loc[idx]

        vars_map: Dict[str, str] = {}
        for i, col in enumerate(cfg.var_cols, start=1):
            key = f"var{i}"
            if col and col in df_res.columns:
                v = row.get(col, "")
                vars_map[key] = str(v if v is not None else "").strip()
            else:
                vars_map[key] = ""

        prompt = render_prompt(cfg.prompt_template, vars_map)

        rid = str(row.get(cfg.id_col, "")).strip()
        print(f"  [{n}/{len(idxs)}] {cfg.id_col}={rid}")

        out = call_openai(client, cfg, prompt)
        ok, reason, cleaned = validate_output(cfg, out)

        if not ok:
            repair_prompt = prompt + f"\n\nFix: previous output failed '{reason}'. Return a corrected response."
            out2 = call_openai(client, cfg, repair_prompt)
            ok2, _, cleaned2 = validate_output(cfg, out2)
            cleaned = cleaned2 if ok2 else cleaned

        df_res.at[idx, cfg.output_col] = cleaned

        if cfg.write_mode == "processed_only":
            rec = {}
            for c in df_res.columns:
                rec[c] = row.get(c, None)
            rec[cfg.output_col] = cleaned
            processed_rows.append(rec)

        done += 1
        if cfg.autosave_interval > 0 and (done % cfg.autosave_interval == 0):
            print("    💾 autosave...")
            write_outputs(s3, cfg, df_res, processed_rows)

        time.sleep(cfg.delay_between_requests)

    print("\n📤 Writing final outputs...")
    write_outputs(s3, cfg, df_res, processed_rows)
    print("🎉 Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python process/llm_table_runner.py <task_config.yml>")
    run_task(sys.argv[1])
