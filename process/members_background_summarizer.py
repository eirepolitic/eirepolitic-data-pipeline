"""
members_background_summarizer.py

Reads members CSV from S3, builds a 2-col working table (member_code, full_name),
then uses OpenAI Responses API + web_search tool to generate a <=200 word neutral
background summary per member, focusing on:
- where they grew up
- what they worked as before politics
- their political history before 2025

Writes:
- s3://<bucket>/processed/members/members_summaries.csv
- s3://<bucket>/processed/members/parquets/members_summaries.parquet

Resumable:
- If output exists, only fills rows where background is missing.

Env vars:
- OPENAI_API_KEY (required)
- S3_BUCKET (default: eirepolitic-data)
- INPUT_KEY (default: raw/members/oireachtas_members_34th_dail.csv)
- OUTPUT_CSV_KEY (default: processed/members/members_summaries.csv)
- OUTPUT_PARQUET_KEY (default: processed/members/parquets/members_summaries.parquet)
- OPENAI_MODEL (default: gpt-4.1-mini)
- OPENAI_REASONING_EFFORT (default: low)  # for gpt-5*; NOTE: web_search not supported with gpt-5 + minimal
- OPENAI_VERBOSITY (default: low)         # for gpt-5*
- MAX_OUTPUT_TOKENS (default: 0 => auto)
- TEST_ROWS (default: 0 => all missing)
- AUTOSAVE_INTERVAL (default: 25)
- DELAY_BETWEEN_REQUESTS (default: 0.25)
- MAX_RETRIES (default: 5)
"""

import io
import os
import re
import time
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from openai import OpenAI

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------- CONFIG ----------------
S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
INPUT_KEY = os.getenv("INPUT_KEY", "raw/members/oireachtas_members_34th_dail.csv")

OUTPUT_CSV_KEY = os.getenv("OUTPUT_CSV_KEY", "processed/members/members_summaries.csv")
OUTPUT_PARQUET_KEY = os.getenv("OUTPUT_PARQUET_KEY", "processed/members/parquets/members_summaries.parquet")

AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").strip()  # minimal/low/medium/high/none
OPENAI_VERBOSITY = os.getenv("OPENAI_VERBOSITY", "low").strip()                # low/medium/high

DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "0"))           # 0 => auto
TEST_ROWS = int(os.getenv("TEST_ROWS", "0"))                                   # 0 => all missing
AUTOSAVE_INTERVAL = int(os.getenv("AUTOSAVE_INTERVAL", "25"))
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.25"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

OUT_COL = "background"
KEY_COL = "member_code"
NAME_COL = "full_name"

s3 = boto3.client("s3", region_name=AWS_REGION)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------- S3 HELPERS ----------------
def s3_get_text(bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8-sig", errors="replace")


def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8-sig"), ContentType=content_type)


def s3_put_bytes(bucket: str, key: str, b: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=b, ContentType=content_type)


def s3_exists(bucket: str, key: str) -> bool:
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


def max_output_tokens_for_model() -> int:
    if DEFAULT_MAX_OUTPUT_TOKENS and DEFAULT_MAX_OUTPUT_TOKENS > 0:
        return DEFAULT_MAX_OUTPUT_TOKENS
    # ~200 words typically fits comfortably under ~300 tokens
    return 320


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
    # Remove common “[1]” style markers if the model includes them anyway
    t = _CIT_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", t).strip()


# ---------------- PROMPT ----------------
def build_prompt(full_name: str) -> str:
    return f"""
Use web search to write a politically neutral, factual background summary (MAX 200 words) of the Irish politician "{full_name}".

Include, if available:
- Where they grew up (town/county/region).
- What they worked as before becoming a politician.
- Their political history before 2025 (roles, elections, notable positions held).

Rules:
- Neutral, factual tone.
- If a detail cannot be verified reliably, omit it rather than guessing.
- Do NOT include citations, links, URLs, domain names, markdown links, or parenthetical source references.
- Output plain text only.
""".strip()


# ---------------- OPENAI CALL ----------------
def run_openai_summary(prompt: str) -> str:
    last_err = None

    # Note: OpenAI docs state web_search isn’t supported with gpt-5 + minimal reasoning.
    reasoning_effort = OPENAI_REASONING_EFFORT
    if is_gpt5_family(OPENAI_MODEL) and reasoning_effort == "minimal":
        reasoning_effort = "low"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload: Dict[str, Any] = {
                "model": OPENAI_MODEL,
                "tools": [{"type": "web_search"}],
                "tool_choice": "auto",
                "input": prompt,
                "max_output_tokens": max_output_tokens_for_model(),
            }

            if is_gpt5_family(OPENAI_MODEL):
                payload["reasoning"] = {"effort": reasoning_effort}
                payload["text"] = {"verbosity": OPENAI_VERBOSITY}
            else:
                payload["temperature"] = 0

            resp = client.responses.create(**payload)
            out = extract_text_from_response(resp)
            out = strip_inline_citations(out)

            if out:
                return out

            last_err = RuntimeError("Empty model output")
            time.sleep(2.0 * attempt)
        except Exception as e:
            last_err = e
            time.sleep(2.0 * attempt)

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_err}")


# ---------------- PARQUET WRITER ----------------
def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


# ---------------- MAIN ----------------
def main() -> None:
    print(f"📥 Loading input from s3://{S3_BUCKET}/{INPUT_KEY}")
    in_text = s3_get_text(S3_BUCKET, INPUT_KEY)
    df_in = pd.read_csv(io.StringIO(in_text))

    # Reduce to required columns
    if KEY_COL not in df_in.columns or NAME_COL not in df_in.columns:
        raise RuntimeError(f"Input CSV must contain columns: '{KEY_COL}', '{NAME_COL}'")

    df_base = df_in[[KEY_COL, NAME_COL]].copy()

    # Load existing output if present
    if s3_exists(S3_BUCKET, OUTPUT_CSV_KEY):
        print(f"📥 Loading existing output from s3://{S3_BUCKET}/{OUTPUT_CSV_KEY}")
        out_text = s3_get_text(S3_BUCKET, OUTPUT_CSV_KEY)
        df_out = pd.read_csv(io.StringIO(out_text))
    else:
        print("📄 No existing output found — creating new output.")
        df_out = pd.DataFrame(columns=[KEY_COL, NAME_COL, OUT_COL])

    # Map existing summaries by member_code
    existing: Dict[str, str] = {}
    if not df_out.empty and KEY_COL in df_out.columns and OUT_COL in df_out.columns:
        for code, val in zip(df_out[KEY_COL].astype(str), df_out[OUT_COL]):
            if code and not is_missing(val):
                existing[code.strip()] = str(val).strip()

    # Build result frame (preserve base ordering)
    df_res = df_base.copy()
    df_res[OUT_COL] = df_res[KEY_COL].astype(str).map(lambda c: existing.get(c.strip(), pd.NA))

    # Find missing rows
    idxs = df_res.index[df_res[OUT_COL].apply(is_missing)].tolist()
    if TEST_ROWS and TEST_ROWS > 0:
        idxs = idxs[:TEST_ROWS]
        print(f"🧪 Test mode: processing first {len(idxs)} missing rows.")
    else:
        print(f"📄 Rows needing summaries: {len(idxs)}")

    processed = 0
    for n, idx in enumerate(idxs, start=1):
        full_name = str(df_res.at[idx, NAME_COL] or "").strip()
        code = str(df_res.at[idx, KEY_COL] or "").strip()
        print(f"\n👤 {n}/{len(idxs)} member_code={code} name='{full_name}'")

        if not full_name:
            df_res.at[idx, OUT_COL] = pd.NA
            print("⚠️ Missing full_name; skipping.")
            continue

        prompt = build_prompt(full_name)
        summary = run_openai_summary(prompt)
        df_res.at[idx, OUT_COL] = summary
        print("✅ Summary written.")

        processed += 1
        if processed % AUTOSAVE_INTERVAL == 0:
            print("💾 Autosaving CSV + Parquet to S3...")
            buf = io.StringIO()
            df_res.to_csv(buf, index=False, encoding="utf-8-sig")
            s3_put_text(S3_BUCKET, OUTPUT_CSV_KEY, buf.getvalue())

            pq_bytes = dataframe_to_parquet_bytes(df_res)
            s3_put_bytes(S3_BUCKET, OUTPUT_PARQUET_KEY, pq_bytes, content_type="application/x-parquet")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n📤 Writing final CSV + Parquet to S3...")
    buf = io.StringIO()
    df_res.to_csv(buf, index=False, encoding="utf-8-sig")
    s3_put_text(S3_BUCKET, OUTPUT_CSV_KEY, buf.getvalue())

    pq_bytes = dataframe_to_parquet_bytes(df_res)
    s3_put_bytes(S3_BUCKET, OUTPUT_PARQUET_KEY, pq_bytes, content_type="application/x-parquet")

    print("🎉 Done.")
    print(f"   CSV:    s3://{S3_BUCKET}/{OUTPUT_CSV_KEY}")
    print(f"   Parquet:s3://{S3_BUCKET}/{OUTPUT_PARQUET_KEY}")


if __name__ == "__main__":
    main()
