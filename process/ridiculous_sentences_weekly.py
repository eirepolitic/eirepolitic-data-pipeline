import io
import os
import re
import json
import time
import hashlib
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Tuple

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from openai import OpenAI

import pyarrow as pa
import pyarrow.parquet as pq


S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
INPUT_KEY = os.getenv("INPUT_KEY", "raw/debates/debate_speeches_extracted.csv")
OUTPUT_KEY = os.getenv("OUTPUT_KEY", "processed/debates/ridiculous_sentences_weekly.csv")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal").strip()
OPENAI_VERBOSITY = os.getenv("OPENAI_VERBOSITY", "low").strip()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))

RUN_MODE = os.getenv("RUN_MODE", "year").strip().lower()  # year | previous_week
RUN_YEAR = int(os.getenv("RUN_YEAR", "2026"))
RUN_WEEK_ID = os.getenv("RUN_WEEK_ID", "").strip()  # optional exact week override, e.g. 202601
AS_OF_DATE = os.getenv("AS_OF_DATE", "").strip()  # optional YYYY-MM-DD override for previous_week

TOP_N_PER_WEEK = max(1, int(os.getenv("TOP_N_PER_WEEK", "10")))
MAX_SENTENCE_WORDS = max(1, int(os.getenv("MAX_SENTENCE_WORDS", "50")))
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "20")))

DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.25"))
AUTOSAVE_INTERVAL = max(0, int(os.getenv("AUTOSAVE_INTERVAL", "5")))
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "5")))
TEST_ROWS = max(0, int(os.getenv("TEST_ROWS", "0")))

PARQUET_KEY = os.getenv("PARQUET_KEY", "processed/debates/parquets/ridiculous_sentences_weekly.parquet")

DATE_COL = "Debate Date"
SPEAKER_COL = "Speaker Name"
TEXT_COL = "Speech Text"

s3 = boto3.client("s3", region_name=AWS_REGION)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=(?:["“‘(\[])?[A-Z0-9])')


def s3_get_text(bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8-sig", errors="replace")


def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8-sig"), ContentType=content_type)


def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


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

    output_items = _get_attr(resp, "output", []) or []
    chunks: List[str] = []
    for item in output_items:
        if _get_attr(item, "type", None) != "message":
            continue
        for content in (_get_attr(item, "content", []) or []):
            if _get_attr(content, "type", None) in ("output_text", "text"):
                value = _get_attr(content, "text", None)
                if value:
                    chunks.append(str(value))
    return "\n".join(chunks).strip()


def call_openai(prompt: str, max_output_tokens: int) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload: Dict[str, Any] = {
                "model": OPENAI_MODEL,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
            }
            if is_gpt5_family(OPENAI_MODEL):
                payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
                payload["text"] = {"verbosity": OPENAI_VERBOSITY}
            else:
                payload["temperature"] = 0

            resp = client.responses.create(**payload)
            return extract_text_from_response(resp).strip()
        except Exception as e:
            last_err = e
            time.sleep(2.0 * attempt)

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_err}")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_sentence_for_dedupe(text: str) -> str:
    text = normalize_ws(text).lower()
    return text.strip(' "\'“”‘’()[]')


def candidate_id(week_id: str, speaker_name: str, sentence: str) -> str:
    raw = "||".join([week_id, normalize_ws(speaker_name), normalize_ws(sentence)]).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def split_sentences(text: str) -> List[str]:
    cleaned = normalize_ws(text)
    if not cleaned:
        return []

    parts = SENTENCE_SPLIT_RE.split(cleaned)
    sentences: List[str] = []
    for part in parts:
        piece = normalize_ws(part)
        if not piece:
            continue

        if count_words(piece) > MAX_SENTENCE_WORDS and ";" in piece:
            for sub in [normalize_ws(x) for x in piece.split(";")]:
                if sub:
                    sentences.append(sub)
        else:
            sentences.append(piece)

    return sentences


def first_monday_of_year(year: int) -> date:
    d = date(year, 1, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def week_start_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_year_and_number(d: date) -> Tuple[int, int]:
    ws = week_start_monday(d)
    year = d.year
    if ws < first_monday_of_year(year):
        year -= 1
    first_monday = first_monday_of_year(year)
    week_num = ((ws - first_monday).days // 7) + 1
    return year, week_num


def make_week_id(d: date) -> str:
    year, week_num = week_year_and_number(d)
    return f"{year}{week_num:02d}"


def previous_completed_week_id(today: date) -> str:
    if today.weekday() == 6:
        target_date = today
    else:
        target_date = today - timedelta(days=today.weekday() + 1)
    return make_week_id(target_date)


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for debate_date_raw, speaker_name, speech_text in df[[DATE_COL, SPEAKER_COL, TEXT_COL]].itertuples(index=False, name=None):
        if pd.isna(speech_text) or pd.isna(debate_date_raw):
            continue

        debate_dt = pd.to_datetime(str(debate_date_raw), errors="coerce")
        if pd.isna(debate_dt):
            continue

        debate_date = debate_dt.date()
        week_id = make_week_id(debate_date)

        for sentence in split_sentences(str(speech_text)):
            wc = count_words(sentence)
            if wc <= 0 or wc > MAX_SENTENCE_WORDS:
                continue
            if not re.search(r"[A-Za-z]", sentence):
                continue

            records.append({
                "week_id": week_id,
                "debate_date": debate_date.isoformat(),
                "speaker_name": normalize_ws("" if pd.isna(speaker_name) else str(speaker_name)),
                "sentence": normalize_ws(sentence),
                "sentence_norm": normalize_sentence_for_dedupe(sentence),
                "word_count": wc,
            })

    if not records:
        return pd.DataFrame(columns=["week_id", "debate_date", "speaker_name", "sentence", "sentence_norm", "word_count", "candidate_id"])

    cdf = pd.DataFrame(records)
    cdf = cdf.drop_duplicates(subset=["week_id", "speaker_name", "sentence_norm"], keep="first").reset_index(drop=True)
    cdf["candidate_id"] = cdf.apply(lambda r: candidate_id(r["week_id"], r["speaker_name"], r["sentence"]), axis=1)
    return cdf


def filter_target_scope(cdf: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if cdf.empty:
        return cdf, []

    if RUN_WEEK_ID:
        scoped = cdf[cdf["week_id"].astype(str) == RUN_WEEK_ID].copy()
        return scoped, [RUN_WEEK_ID] if not scoped.empty else [RUN_WEEK_ID]

    if RUN_MODE == "year":
        target_prefix = f"{RUN_YEAR}"
        scoped = cdf[cdf["week_id"].astype(str).str.startswith(target_prefix)].copy()
        target_weeks = sorted(scoped["week_id"].astype(str).unique().tolist())
        return scoped, target_weeks

    if RUN_MODE == "previous_week":
        as_of = datetime.strptime(AS_OF_DATE, "%Y-%m-%d").date() if AS_OF_DATE else datetime.utcnow().date()
        target_week = previous_completed_week_id(as_of)
        scoped = cdf[cdf["week_id"].astype(str) == target_week].copy()
        return scoped, [target_week]

    raise RuntimeError("RUN_MODE must be 'year' or 'previous_week'.")


def apply_test_limit(cdf: pd.DataFrame) -> pd.DataFrame:
    if TEST_ROWS > 0:
        return cdf.head(TEST_ROWS).copy()
    return cdf


def extract_json_payload(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model output.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    left = raw.find("[")
    right = raw.rfind("]")
    if left != -1 and right != -1 and right > left:
        return json.loads(raw[left:right + 1])

    left = raw.find("{")
    right = raw.rfind("}")
    if left != -1 and right != -1 and right > left:
        return json.loads(raw[left:right + 1])

    raise ValueError("Could not locate JSON payload in model output.")


def build_scoring_prompt(batch: List[Dict[str, Any]]) -> str:
    payload = [{"candidate_id": row["candidate_id"], "sentence": row["sentence"]} for row in batch]
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    return f"""
You are ranking short sentences from Irish parliamentary debate transcripts.

Score each sentence from 1 to 100 for how ridiculous, funny, inappropriate, least parliamentary, or strikingly absurd it sounds in parliamentary context.

Scoring guidance:
- 1 to 20: ordinary, procedural, or unremarkable parliamentary language
- 21 to 40: mildly pointed or slightly amusing
- 41 to 60: notably sharp, odd, or funny
- 61 to 80: highly ridiculous, inappropriate, or very funny
- 81 to 100: exceptional standout line that is unusually ridiculous, absurd, or clearly least-parliamentary

Important rules:
- Judge the sentence only by its wording in parliamentary context.
- Do not infer a score from topic alone.
- Score every candidate independently.
- Be willing to give low scores to normal sentences.
- Use the full 1-100 range when justified.
- Return every candidate_id exactly once.

Return ONLY valid JSON.
The JSON must be an array of objects in this exact shape:
[
  {{"candidate_id": "abc", "score": 73}}
]

Candidates:
{payload_json}
""".strip()


def parse_scores(text: str, expected_ids: List[str]) -> Dict[str, int]:
    data = extract_json_payload(text)
    items = data["scores"] if isinstance(data, dict) and "scores" in data else data
    if not isinstance(items, list):
        raise ValueError("JSON payload is not a list.")

    scores: Dict[str, int] = {}
    expected_set = set(expected_ids)
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip()
        score = item.get("score", None)
        if not cid:
            continue
        try:
            score_int = int(score)
        except Exception:
            continue
        scores[cid] = max(1, min(100, score_int))

    missing = [cid for cid in expected_ids if cid not in scores]
    extras = [cid for cid in scores if cid not in expected_set]
    if missing or extras:
        raise ValueError(f"Invalid score payload. Missing={missing[:5]} Extras={extras[:5]}")

    return scores


def score_batch(batch_df: pd.DataFrame) -> Dict[str, int]:
    batch_records = batch_df[["candidate_id", "sentence"]].to_dict(orient="records")
    expected_ids = [row["candidate_id"] for row in batch_records]

    prompt = build_scoring_prompt(batch_records)
    max_tokens = MAX_OUTPUT_TOKENS if MAX_OUTPUT_TOKENS > 0 else max(400, len(batch_records) * 30)

    last_err = None
    repair_prompt = prompt
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = call_openai(repair_prompt, max_output_tokens=max_tokens)
            return parse_scores(out, expected_ids)
        except Exception as e:
            last_err = e
            repair_prompt = prompt + f"\n\nThe previous output was invalid because: {e}\nReturn only corrected valid JSON."
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to score batch after {MAX_RETRIES} attempts: {last_err}")


def score_week(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        week_df = week_df.copy()
        week_df["score"] = pd.Series(dtype="int64")
        return week_df

    result = week_df.copy().reset_index(drop=True)
    result["score"] = pd.NA
    week_id = str(result.at[0, "week_id"])

    print(f"🧠 Scoring week {week_id} with {len(result)} candidate sentences...")

    for start in range(0, len(result), BATCH_SIZE):
        batch = result.iloc[start:start + BATCH_SIZE].copy()
        scores = score_batch(batch[["candidate_id", "sentence"]])

        for idx, row in batch.iterrows():
            result.at[idx, "score"] = scores[row["candidate_id"]]

        time.sleep(DELAY_BETWEEN_REQUESTS)

    result["score"] = result["score"].astype(int)
    return result


def select_top_rows(cdf: pd.DataFrame) -> pd.DataFrame:
    if cdf.empty:
        return pd.DataFrame(columns=["week_id", "sentence", "speaker_name", "score"])

    ranked = cdf.sort_values(
        ["week_id", "score", "speaker_name", "sentence"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    top = (
        ranked.groupby("week_id", sort=True, as_index=False, group_keys=False)
        .head(TOP_N_PER_WEEK)
        .copy()
    )

    return top[["week_id", "sentence", "speaker_name", "score"]].reset_index(drop=True)


def write_outputs(df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    s3_put_text(S3_BUCKET, OUTPUT_KEY, buf.getvalue(), content_type="text/csv")

    table = pa.Table.from_pandas(df, preserve_index=False)
    out_buf = io.BytesIO()
    pq.write_table(table, out_buf, compression="snappy")
    s3_put_bytes(S3_BUCKET, PARQUET_KEY, out_buf.getvalue(), content_type="application/x-parquet")


def merge_with_existing(df_new: pd.DataFrame, target_weeks: List[str], replace_full_target_scope: bool) -> pd.DataFrame:
    if s3_exists(S3_BUCKET, OUTPUT_KEY):
        existing_text = s3_get_text(S3_BUCKET, OUTPUT_KEY)
        df_existing = pd.read_csv(io.StringIO(existing_text))
    else:
        df_existing = pd.DataFrame(columns=["week_id", "sentence", "speaker_name", "score"])

    if df_existing.empty:
        combined = df_new.copy()
    else:
        existing = df_existing.copy()
        if RUN_WEEK_ID:
            existing = existing[existing["week_id"].astype(str) != RUN_WEEK_ID].copy()
        elif RUN_MODE == "year" and replace_full_target_scope:
            existing = existing[~existing["week_id"].astype(str).str.startswith(str(RUN_YEAR))].copy()
        else:
            existing = existing[~existing["week_id"].astype(str).isin(target_weeks)].copy()
        combined = pd.concat([existing, df_new], ignore_index=True)

    combined = combined[["week_id", "sentence", "speaker_name", "score"]].copy()
    return combined.sort_values(
        ["week_id", "score", "speaker_name", "sentence"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def main() -> None:
    print(f"📥 Loading input from s3://{S3_BUCKET}/{INPUT_KEY}")
    input_text = s3_get_text(S3_BUCKET, INPUT_KEY)
    df = pd.read_csv(io.StringIO(input_text))

    required = [DATE_COL, SPEAKER_COL, TEXT_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Input is missing required columns: {missing}")

    candidates = build_candidates(df)
    print(f"🧾 Built {len(candidates)} sentence candidates after sentence filtering and dedupe.")

    scoped, target_weeks = filter_target_scope(candidates)
    if RUN_WEEK_ID:
        print(f"📆 Target exact week: {RUN_WEEK_ID}")
    elif RUN_MODE == "year":
        print(f"📆 Target year: {RUN_YEAR} | weeks found: {len(target_weeks)}")
    else:
        target_label = target_weeks[0] if target_weeks else "(none)"
        print(f"📆 Target previous completed week: {target_label}")

    target_candidates = apply_test_limit(scoped)
    print(f"🎯 Sending {len(target_candidates)} candidate sentences to the LLM scorer.")

    scored_frames: List[pd.DataFrame] = []
    processed_weeks: List[str] = []

    for i, week_id in enumerate(sorted(target_candidates["week_id"].astype(str).unique().tolist()), start=1):
        week_df = target_candidates[target_candidates["week_id"].astype(str) == week_id].copy()
        scored_week = score_week(week_df)
        scored_frames.append(scored_week)
        processed_weeks.append(week_id)

        if AUTOSAVE_INTERVAL > 0 and i % AUTOSAVE_INTERVAL == 0:
            partial_scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else target_candidates.iloc[0:0].copy()
            partial_top = select_top_rows(partial_scored)
            partial_combined = merge_with_existing(partial_top, processed_weeks, replace_full_target_scope=False)
            print(f"💾 Autosaving after {i} week(s)...")
            write_outputs(partial_combined)

    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else target_candidates.iloc[0:0].copy()
    top_rows = select_top_rows(scored)
    print(f"🏁 Selected {len(top_rows)} final rows.")

    combined = merge_with_existing(top_rows, target_weeks, replace_full_target_scope=True)
    print(f"📤 Writing CSV to s3://{S3_BUCKET}/{OUTPUT_KEY}")
    print(f"📤 Writing Parquet to s3://{S3_BUCKET}/{PARQUET_KEY}")
    write_outputs(combined)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
