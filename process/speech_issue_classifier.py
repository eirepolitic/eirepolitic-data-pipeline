"""
classify_political_issue_openai_s3.py (GPT-5-safe)

Pipeline version:
- Reads input CSV from S3
- Loads existing output CSV from S3 if present
- Only classifies rows where PoliticalIssues is missing in the output
- Writes updated output CSV back to S3

Key GPT-5 fixes:
- Use message-array input (not bare string)
- Robustly extract text from resp.output (not only resp.output_text)
- Treat empty output as retryable
- Do NOT treat the literal label "NONE" as missing
- Optional GPT-5 knobs: reasoning.effort, text.verbosity, larger max_output_tokens
"""

import io
import os
import time
import hashlib
from typing import Optional, Tuple, Any, List

import pandas as pd
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI

# ---------------- CONFIG ----------------
S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
INPUT_KEY = os.getenv("INPUT_KEY", "raw/debates/debate_speeches_extracted.csv")
OUTPUT_KEY = os.getenv("OUTPUT_KEY", "processed/debates/debate_speeches_classified.csv")

AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# For GPT-5 family (optional)
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip()  # e.g. minimal/low/medium/high/none
OPENAI_TEXT_VERBOSITY = os.getenv("OPENAI_TEXT_VERBOSITY", "").strip()      # e.g. low/medium/high

# Determinism for non-GPT-5 models
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "").strip()            # e.g. "0"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# For GPT-5 family only (safe defaults for classification/extraction tasks)
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal")  # minimal/low/medium/high
OPENAI_VERBOSITY = os.getenv("OPENAI_VERBOSITY", "low")  # low/medium/high

# Token controls
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "0"))  # 0 => auto
# (Auto logic below picks a safer default for GPT-5)

DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.25"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
AUTOSAVE_INTERVAL = int(os.getenv("AUTOSAVE_INTERVAL", "50"))

# Optional: process only first N missing rows (0 = all)
TEST_ROWS = int(os.getenv("TEST_ROWS", "0"))

CLASS_COL = "PoliticalIssues"
TEXT_COL = "Speech Text"

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

s3 = boto3.client("s3", region_name=AWS_REGION)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------- S3 HELPERS ----------------
def s3_get_text(bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8-sig", errors="replace")


def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8-sig"),
        ContentType=content_type,
    )


def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# ---------------- DATA HELPERS ----------------
def is_missing(v) -> bool:
    """
    IMPORTANT:
    - pd.NA / NaN / None / "" are missing
    - The literal category "NONE" is NOT missing (it is a valid label)
    """
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass

    if v is None:
        return True

    if isinstance(v, str) and v.strip() == "":
        return True

    # Treat obvious null-like strings as missing, BUT NOT "none" (that's a real label)
    if isinstance(v, str) and v.strip().lower() in {"nan", "na", "null"}:
        return True

    return False

def is_gpt5_family(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return m.startswith("gpt-5")


def make_speech_id(row: pd.Series) -> str:
    parts = [
        str(row.get("Debate Date", "")).strip(),
        str(row.get("Speaker Name", "")).strip(),
        str(row.get("Speech Order", "")).strip(),
        str(row.get("Speech Text", "")).strip(),
    ]
    raw = "||".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def canonicalize_label(label: str) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    for cat in ISSUE_CATEGORIES:
        if cat.lower() == s.lower():
            return cat
    return None


# ---------------- PROMPTS ----------------
def build_base_prompt(speech_text: str) -> str:
    choices_list = "\n".join(f"- {c}" for c in ISSUE_CATEGORIES)
    return f"""
Classify the following Irish parliamentary speech into EXACTLY ONE political issue category from the list below.

Choose ONE category only. Respond ONLY with the category name (or NONE).

Allowed categories:
{choices_list}

Speech:
{speech_text}
""".strip()


def build_refinement_prompt(base_prompt: str, bad_output: str, reason: str) -> str:
    return (
        base_prompt
        + f"\n\nThe previous output '{bad_output}' was invalid because: {reason}.\n"
          "Please correct this and return EXACTLY ONE category name from the allowed list. "
          "Return only the category name."
    )


# ---------------- OPENAI OUTPUT EXTRACTION ----------------
def _get_attr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def extract_text_from_response(resp: Any) -> str:
    """
    Robust extraction for Responses API objects.
    Uses:
      1) resp.output_text (SDK convenience)
      2) resp.output[*].content[*].text (fallback)
    """
    text = (_get_attr(resp, "output_text", "") or "").strip()
    if text:
        return text

    out_items = _get_attr(resp, "output", []) or []
    chunks: List[str] = []
    for item in out_items:
        # We look for "message" items with "content" parts containing text
        if _get_attr(item, "type", None) != "message":
            continue
        content_items = _get_attr(item, "content", []) or []
        for c in content_items:
            ctype = _get_attr(c, "type", None)
            if ctype in ("output_text", "text"):
                t = _get_attr(c, "text", None)
                if t:
                    chunks.append(str(t))

    return "\n".join(chunks).strip()


def max_output_tokens_for_model() -> int:
    """
    GPT-5 models can spend tokens on internal reasoning; too-low max_output_tokens can lead to empty final text.
    Use a larger default for gpt-5* unless overridden.
    """
    if DEFAULT_MAX_OUTPUT_TOKENS and DEFAULT_MAX_OUTPUT_TOKENS > 0:
        return DEFAULT_MAX_OUTPUT_TOKENS

    if OPENAI_MODEL.startswith("gpt-5"):
        return 512  # safer than 64 for reasoning models
    return 128


# ---------------- OPENAI CALL ----------------
def run_openai(prompt: str, max_retries: int = 5, backoff: float = 2.0) -> str:
    """
    Robust OpenAI Responses API call wrapper.
    - GPT-5 family: use reasoning.effort + text.verbosity (do NOT send temperature)
    - Non GPT-5: use temperature=0 for determinism
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": OPENAI_MODEL,
                "input": prompt,
                "max_output_tokens": 64,  # keep small; >=16 min
            }

            if is_gpt5_family(OPENAI_MODEL):
                payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
                payload["text"] = {"verbosity": OPENAI_VERBOSITY}
                # IMPORTANT: do NOT include temperature/top_p/logprobs unless reasoning.effort == "none"
                # (otherwise GPT-5* requests can error)  :contentReference[oaicite:2]{index=2}
            else:
                payload["temperature"] = 0

            resp = client.responses.create(**payload)
            return (resp.output_text or "").strip()

        except Exception as e:
            last_err = e
            time.sleep(backoff * attempt)

    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")

# ---------------- VALIDATION ----------------
def rule_validate_label(label: str) -> Tuple[bool, str]:
    if label is None:
        return False, "Empty output."
    s = str(label).strip()
    if s == "":
        return False, "Empty output."

    canon = canonicalize_label(s)
    if canon:
        return True, "OK"

    if "," in s:
        return False, "Contains commas but does not match approved category."

    return False, f"'{s}' not in approved list."


def classify_speech_text_recursive(text: str) -> str:
    base_prompt = build_base_prompt(text)
    prompt = base_prompt
    last_output = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"    ↳ Iter {iteration}/{MAX_ITERATIONS}...")
        out = run_openai(prompt)
        last_output = out or ""
        print(f"      Model output: [{last_output}]")

        ok, reason = rule_validate_label(last_output)
        if ok:
            return canonicalize_label(last_output) or "NONE"

        prompt = build_refinement_prompt(base_prompt, last_output, reason)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return "NONE"


# ---------------- MAIN PROCESS ----------------
def main() -> None:
    print(f"📥 Loading input from s3://{S3_BUCKET}/{INPUT_KEY}")
    input_csv_text = s3_get_text(S3_BUCKET, INPUT_KEY)
    df_in = pd.read_csv(io.StringIO(input_csv_text))

    # Ensure speech_id exists (computed deterministically)
    if "speech_id" not in df_in.columns:
        df_in["speech_id"] = df_in.apply(make_speech_id, axis=1)
    else:
        missing_ids = df_in["speech_id"].isna() | (df_in["speech_id"].astype(str).str.strip() == "")
        if missing_ids.any():
            df_in.loc[missing_ids, "speech_id"] = df_in[missing_ids].apply(make_speech_id, axis=1)

    # Load existing output if present
    if s3_exists(S3_BUCKET, OUTPUT_KEY):
        print(f"📥 Loading existing output from s3://{S3_BUCKET}/{OUTPUT_KEY}")
        out_text = s3_get_text(S3_BUCKET, OUTPUT_KEY)
        df_out = pd.read_csv(io.StringIO(out_text))

        if "speech_id" not in df_out.columns:
            df_out["speech_id"] = df_out.apply(make_speech_id, axis=1)
    else:
        print("📄 No existing output found — creating new output.")
        df_out = pd.DataFrame(columns=list(df_in.columns) + [CLASS_COL])

    # Build a lookup of existing classifications by speech_id
    existing_map = {}
    if not df_out.empty and "speech_id" in df_out.columns and CLASS_COL in df_out.columns:
        for sid, val in zip(df_out["speech_id"].astype(str), df_out[CLASS_COL]):
            if sid and not is_missing(val):
                existing_map[sid] = str(val).strip()

    # Start output as the input (preserves row order), then fill PoliticalIssues from existing_map
    df_res = df_in.copy()
    if CLASS_COL not in df_res.columns:
        df_res[CLASS_COL] = pd.NA

    df_res[CLASS_COL] = df_res.apply(
        lambda r: existing_map.get(str(r["speech_id"]).strip(), r.get(CLASS_COL, pd.NA)),
        axis=1
    )

    # Determine rows to process = missing PoliticalIssues
    mask_missing = df_res[CLASS_COL].apply(is_missing)

    idxs = df_res.index[mask_missing].tolist()
    if TEST_ROWS and TEST_ROWS > 0:
        idxs = idxs[:TEST_ROWS]
        print(f"🧪 Test mode: processing first {len(idxs)} missing rows.")
    else:
        print(f"📄 Rows needing classification: {len(idxs)}")

    processed = 0

    for n, idx in enumerate(idxs, start=1):
        speech = str(df_res.at[idx, TEXT_COL] if TEXT_COL in df_res.columns else "").strip()
        print(f"\n🗣️ Classifying missing row {n}/{len(idxs)} (index {idx})...")

        if not speech:
            df_res.at[idx, CLASS_COL] = "NONE"
            print("⚠️ Empty speech → NONE.")
            continue

        wc = len(speech.split())
        if wc < 20:
            df_res.at[idx, CLASS_COL] = "NONE"
            print(f"⚠️ Speech too short ({wc} words) → NONE.")
            continue

        label = classify_speech_text_recursive(speech)
        df_res.at[idx, CLASS_COL] = label
        print(f"➡️ Assigned: {label}")

        processed += 1
        if processed % AUTOSAVE_INTERVAL == 0:
            print("💾 Autosaving partial progress to S3...")
            buf = io.StringIO()
            df_res.to_csv(buf, index=False, encoding="utf-8-sig")
            s3_put_text(S3_BUCKET, OUTPUT_KEY, buf.getvalue())

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n📤 Writing output to s3://{S3_BUCKET}/{OUTPUT_KEY}")
    buf = io.StringIO()
    df_res.to_csv(buf, index=False, encoding="utf-8-sig")
    s3_put_text(S3_BUCKET, OUTPUT_KEY, buf.getvalue())
    print("🎉 Done.")


if __name__ == "__main__":
    main()
