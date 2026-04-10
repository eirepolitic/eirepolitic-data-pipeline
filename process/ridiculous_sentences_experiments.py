import io
import os
import re
import json
import time
import hashlib
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import boto3
import pandas as pd
from openai import OpenAI

import pyarrow as pa
import pyarrow.parquet as pq


S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
INPUT_KEY = os.getenv("INPUT_KEY", "raw/debates/debate_speeches_extracted.csv")
OUTPUT_KEY = os.getenv("OUTPUT_KEY", "processed/debates/ridiculous_sentences_experiments.csv")
OUTPUT_PARQUET_KEY = os.getenv("OUTPUT_PARQUET_KEY", "processed/debates/parquets/ridiculous_sentences_experiments.parquet")
SUMMARY_KEY = os.getenv("SUMMARY_KEY", "processed/debates/ridiculous_sentences_experiments_summary.csv")
SUMMARY_PARQUET_KEY = os.getenv("SUMMARY_PARQUET_KEY", "processed/debates/parquets/ridiculous_sentences_experiments_summary.parquet")
PROMPTS_CONFIG = os.getenv("PROMPTS_CONFIG", "prompts/ridiculous_sentences_variants.json")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal").strip()
OPENAI_VERBOSITY = os.getenv("OPENAI_VERBOSITY", "low").strip()
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))

TARGET_WEEK_IDS = [x.strip() for x in os.getenv("TARGET_WEEK_IDS", "202602,202603").split(",") if x.strip()]
APPROACH_FILTER = os.getenv("APPROACH_FILTER", "all").strip().lower()
VARIANT_FILTER = {x.strip() for x in os.getenv("VARIANT_FILTER", "").split(",") if x.strip()}

TOP_N_PER_WEEK = max(1, int(os.getenv("TOP_N_PER_WEEK", "10")))
MAX_SENTENCE_WORDS = max(1, int(os.getenv("MAX_SENTENCE_WORDS", "60")))
MAX_EXTRACT_WORDS = max(1, int(os.getenv("MAX_EXTRACT_WORDS", "60")))
MAX_QUOTES_PER_SPEECH = max(1, int(os.getenv("MAX_QUOTES_PER_SPEECH", "3")))
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "10")))
TEST_SPEECHES_PER_WEEK = max(0, int(os.getenv("TEST_SPEECHES_PER_WEEK", "20")))

DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.25"))
AUTOSAVE_VARIANT_INTERVAL = max(0, int(os.getenv("AUTOSAVE_VARIANT_INTERVAL", "1")))
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "5")))

DATE_COL = "Debate Date"
SPEAKER_COL = "Speaker Name"
TEXT_COL = "Speech Text"
ORDER_COL = "Speech Order"
SECTION_COL = "Debate Section Name"

WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=(?:["“‘(\[])?[A-Z0-9])')

s3 = boto3.client("s3", region_name=AWS_REGION)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def s3_get_text(bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8-sig", errors="replace")


def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8-sig"), ContentType=content_type)


def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


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
        for response_content in (_get_attr(item, "content", []) or []):
            if _get_attr(response_content, "type", None) in ("output_text", "text"):
                value = _get_attr(response_content, "text", None)
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


def normalize_text_for_match(text: str) -> str:
    text = normalize_ws(text).lower()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_quote_for_dedupe(text: str) -> str:
    return normalize_text_for_match(text).strip(' "\'()[]')


def speech_id_from_row(row: pd.Series) -> str:
    parts = [
        str(row.get(DATE_COL, "")).strip(),
        str(row.get(SPEAKER_COL, "")).strip(),
        str(row.get(ORDER_COL, "")).strip(),
        str(row.get(TEXT_COL, "")).strip(),
    ]
    raw = "||".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def candidate_id(variant_id: str, week_id: str, speaker_name: str, quote: str) -> str:
    raw = "||".join([variant_id, week_id, normalize_ws(speaker_name), normalize_ws(quote)]).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


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


def load_prompt_variants(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        variants = json.load(f)
    if not isinstance(variants, list):
        raise RuntimeError("Prompt config must be a JSON list.")
    return variants


def filter_variants(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant.get("variant_id", "")).strip()
        approach = str(variant.get("approach", "")).strip().lower()
        if not variant_id or not approach:
            continue
        if APPROACH_FILTER != "all" and approach != APPROACH_FILTER:
            continue
        if VARIANT_FILTER and variant_id not in VARIANT_FILTER:
            continue
        out.append(variant)
    return out


def load_input_speeches() -> pd.DataFrame:
    input_text = s3_get_text(S3_BUCKET, INPUT_KEY)
    df = pd.read_csv(io.StringIO(input_text))

    required = [DATE_COL, SPEAKER_COL, TEXT_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Input is missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df[df[DATE_COL].notna()].copy()
    df["week_id"] = df[DATE_COL].dt.date.apply(make_week_id)
    df = df[df["week_id"].astype(str).isin(TARGET_WEEK_IDS)].copy()

    if ORDER_COL not in df.columns:
        df[ORDER_COL] = range(1, len(df) + 1)
    if SECTION_COL not in df.columns:
        df[SECTION_COL] = ""

    df["speech_id"] = df.apply(speech_id_from_row, axis=1)
    df["speaker_name"] = df[SPEAKER_COL].fillna("").astype(str).map(normalize_ws)
    df["speech_text"] = df[TEXT_COL].fillna("").astype(str).map(normalize_ws)
    df["debate_date"] = df[DATE_COL].dt.strftime("%Y-%m-%d")
    df["section_name"] = df[SECTION_COL].fillna("").astype(str).map(normalize_ws)

    df = df.sort_values(["week_id", DATE_COL, ORDER_COL], ascending=[True, True, True]).reset_index(drop=True)

    if TEST_SPEECHES_PER_WEEK > 0:
        frames: List[pd.DataFrame] = []
        for _, group in df.groupby("week_id", sort=True):
            frames.append(group.head(TEST_SPEECHES_PER_WEEK).copy())
        df = pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()

    return df[["speech_id", "week_id", "debate_date", "speaker_name", "speech_text", "section_name"]].copy()


def build_sentence_candidates(df_speeches: pd.DataFrame, variant: Dict[str, Any]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    variant_id = str(variant["variant_id"])
    family = str(variant.get("prompt_family", ""))

    for speech_id, week_id, debate_date, speaker_name, speech_text, section_name in df_speeches.itertuples(index=False, name=None):
        for sentence in split_sentences(speech_text):
            wc = count_words(sentence)
            if wc <= 0 or wc > MAX_SENTENCE_WORDS:
                continue
            if not re.search(r"[A-Za-z]", sentence):
                continue
            records.append({
                "variant_id": variant_id,
                "prompt_family": family,
                "approach": "sentence_score",
                "speech_id": speech_id,
                "week_id": week_id,
                "debate_date": debate_date,
                "speaker_name": speaker_name,
                "quote": normalize_ws(sentence),
                "quote_norm": normalize_quote_for_dedupe(sentence),
                "section_name": section_name,
                "word_count": wc,
            })

    cdf = pd.DataFrame(records)
    if cdf.empty:
        return pd.DataFrame(columns=["variant_id", "prompt_family", "approach", "speech_id", "week_id", "debate_date", "speaker_name", "quote", "quote_norm", "section_name", "word_count", "candidate_id"])

    cdf = cdf.drop_duplicates(subset=["variant_id", "week_id", "speaker_name", "quote_norm"], keep="first").reset_index(drop=True)
    cdf["candidate_id"] = cdf.apply(lambda r: candidate_id(r["variant_id"], r["week_id"], r["speaker_name"], r["quote"]), axis=1)
    return cdf


def build_extraction_prompt(variant: Dict[str, Any], speech_row: pd.Series) -> str:
    extraction_instructions = str(variant.get("extraction_instructions", "")).strip()
    return f"""
You are reviewing an Irish parliamentary speech.

Your job is to extract only direct quotes from the speech text that genuinely fit the requested tone.

Hard rules:
- Return between 0 and {MAX_QUOTES_PER_SPEECH} quotes.
- Return NO quotes if none truly fit the criteria.
- Do NOT return the merely most argumentative, most partisan, or most negative line unless it is also genuinely funny, ridiculous, bizarre, insulting, unusually harsh, or clearly unparliamentary in wording.
- Each quote must be copied verbatim from the speech text, not paraphrased.
- Each quote must be a contiguous span from the speech text.
- Each quote must be {MAX_EXTRACT_WORDS} words or fewer.
- Prefer quotes that stand on their own and feel memorable.
- Do not return procedural or ordinary political language.

Variant-specific instructions:
{extraction_instructions}

Return ONLY valid JSON in this exact format:
[
  {{"quote": "..."}}
]

Speaker: {speech_row['speaker_name']}
Week: {speech_row['week_id']}
Date: {speech_row['debate_date']}
Section: {speech_row['section_name']}

Speech text:
{speech_row['speech_text']}
""".strip()


def parse_extracted_quotes(text: str, speech_text: str) -> List[str]:
    payload = extract_json_payload(text)
    items = payload["quotes"] if isinstance(payload, dict) and "quotes" in payload else payload
    if not isinstance(items, list):
        raise ValueError("Extraction payload is not a list.")

    source_norm = normalize_text_for_match(speech_text)
    out: List[str] = []
    seen = set()

    for item in items:
        if isinstance(item, dict):
            quote = str(item.get("quote", "")).strip()
        else:
            quote = str(item).strip()
        quote = normalize_ws(quote)
        if not quote:
            continue
        if count_words(quote) <= 0 or count_words(quote) > MAX_EXTRACT_WORDS:
            continue
        quote_norm = normalize_text_for_match(quote)
        if not quote_norm or quote_norm not in source_norm:
            continue
        dedupe_key = normalize_quote_for_dedupe(quote)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(quote)
        if len(out) >= MAX_QUOTES_PER_SPEECH:
            break

    return out


def extract_candidates_from_speeches(df_speeches: pd.DataFrame, variant: Dict[str, Any]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    variant_id = str(variant["variant_id"])
    family = str(variant.get("prompt_family", ""))

    for idx, speech_row in df_speeches.iterrows():
        print(f"    ↳ Extracting [{idx + 1}/{len(df_speeches)}] speech_id={speech_row['speech_id']}")
        prompt = build_extraction_prompt(variant, speech_row)
        max_tokens = MAX_OUTPUT_TOKENS if MAX_OUTPUT_TOKENS > 0 else 800

        last_err = None
        repair_prompt = prompt
        quotes: List[str] = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out = call_openai(repair_prompt, max_output_tokens=max_tokens)
                quotes = parse_extracted_quotes(out, speech_row["speech_text"])
                last_err = None
                break
            except Exception as e:
                last_err = e
                repair_prompt = prompt + f"\n\nThe previous output was invalid because: {e}\nReturn corrected valid JSON only."
                time.sleep(1.5 * attempt)

        if last_err is not None and not quotes:
            print(f"      ⚠️ Extraction failed after retries: {last_err}")

        for quote in quotes:
            records.append({
                "variant_id": variant_id,
                "prompt_family": family,
                "approach": "extract_then_score",
                "speech_id": speech_row["speech_id"],
                "week_id": speech_row["week_id"],
                "debate_date": speech_row["debate_date"],
                "speaker_name": speech_row["speaker_name"],
                "quote": quote,
                "quote_norm": normalize_quote_for_dedupe(quote),
                "section_name": speech_row["section_name"],
                "word_count": count_words(quote),
            })

        time.sleep(DELAY_BETWEEN_REQUESTS)

    cdf = pd.DataFrame(records)
    if cdf.empty:
        return pd.DataFrame(columns=["variant_id", "prompt_family", "approach", "speech_id", "week_id", "debate_date", "speaker_name", "quote", "quote_norm", "section_name", "word_count", "candidate_id"])

    cdf = cdf.drop_duplicates(subset=["variant_id", "week_id", "speaker_name", "quote_norm"], keep="first").reset_index(drop=True)
    cdf["candidate_id"] = cdf.apply(lambda r: candidate_id(r["variant_id"], r["week_id"], r["speaker_name"], r["quote"]), axis=1)
    return cdf


def build_scoring_prompt(variant: Dict[str, Any], batch: List[Dict[str, Any]]) -> str:
    scoring_instructions = str(variant.get("scoring_instructions", "")).strip()
    payload = [{"candidate_id": row["candidate_id"], "quote": row["quote"]} for row in batch]
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    return f"""
You are scoring short quotes from Irish parliamentary debates.

Your job is to score how well each quote fits this broad idea: funny, ridiculous, bizarre, insulting, unusually harsh, or clearly unparliamentary.

Hard rules:
- Most quotes should score low unless they are genuinely standout.
- Ordinary partisan criticism, normal debate point-scoring, and routine rhetorical attacks should usually score low.
- Use high scores sparingly.
- Judge the wording itself, not whether you agree with the politics.

Variant-specific instructions:
{scoring_instructions}

Return ONLY valid JSON in this exact format:
[
  {{"candidate_id": "abc", "score": 73}}
]

Candidates:
{payload_json}
""".strip()


def parse_scores(text: str, expected_ids: List[str], allow_partial: bool = False) -> Dict[str, int]:
    data = extract_json_payload(text)
    items = data["scores"] if isinstance(data, dict) and "scores" in data else data
    if not isinstance(items, list):
        raise ValueError("Score payload is not a list.")

    expected = set(expected_ids)
    scores: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("candidate_id", "")).strip()
        if not cid or cid not in expected:
            continue
        try:
            score = int(item.get("score", 0))
        except Exception:
            continue
        scores[cid] = max(1, min(100, score))

    missing = [cid for cid in expected_ids if cid not in scores]
    if allow_partial:
        if not scores:
            raise ValueError("No valid scores returned for the requested candidate IDs.")
        if missing:
            print(f"    ↳ Partial score response: resolved={len(scores)}/{len(expected_ids)}; missing={len(missing)}")
        return scores

    extras = [cid for cid in scores if cid not in expected]
    if missing or extras:
        raise ValueError(f"Invalid score payload. Missing={missing[:5]} Extras={extras[:5]}")
    return scores


def request_scores_for_records(variant: Dict[str, Any], batch_records: List[Dict[str, Any]]) -> Dict[str, int]:
    expected_ids = [row["candidate_id"] for row in batch_records]
    prompt = build_scoring_prompt(variant, batch_records)
    max_tokens = MAX_OUTPUT_TOKENS if MAX_OUTPUT_TOKENS > 0 else max(400, len(batch_records) * 30)

    last_err = None
    repair_prompt = prompt
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = call_openai(repair_prompt, max_output_tokens=max_tokens)
            return parse_scores(out, expected_ids, allow_partial=True)
        except Exception as e:
            last_err = e
            repair_prompt = prompt + f"\n\nThe previous output was invalid because: {e}\nReturn corrected valid JSON only."
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to score requested records after {MAX_RETRIES} attempts: {last_err}")


def score_records_with_fallback(variant: Dict[str, Any], batch_records: List[Dict[str, Any]]) -> Dict[str, int]:
    if not batch_records:
        return {}

    resolved: Dict[str, int] = {}
    queue: List[List[Dict[str, Any]]] = [batch_records]
    expected_ids = [row["candidate_id"] for row in batch_records]

    while queue:
        chunk = queue.pop(0)
        chunk = [row for row in chunk if row["candidate_id"] not in resolved]
        if not chunk:
            continue

        try:
            chunk_scores = request_scores_for_records(variant, chunk)
        except Exception as e:
            if len(chunk) == 1:
                raise RuntimeError(f"Failed to score single candidate {chunk[0]['candidate_id']}: {e}")
            midpoint = max(1, len(chunk) // 2)
            print(f"    ↳ Score fallback split: chunk={len(chunk)} due to {e}")
            queue.insert(0, chunk[midpoint:])
            queue.insert(0, chunk[:midpoint])
            continue

        resolved.update(chunk_scores)
        missing_rows = [row for row in chunk if row["candidate_id"] not in chunk_scores]
        if missing_rows:
            if len(missing_rows) == len(chunk):
                if len(chunk) == 1:
                    raise RuntimeError(f"Model returned no usable score for candidate {chunk[0]['candidate_id']}.")
                midpoint = max(1, len(chunk) // 2)
                print(f"    ↳ No progress on chunk={len(chunk)}; splitting for retries")
                queue.insert(0, chunk[midpoint:])
                queue.insert(0, chunk[:midpoint])
            else:
                print(f"    ↳ Retrying missing subset of size {len(missing_rows)}")
                queue.insert(0, missing_rows)

    missing_ids = [cid for cid in expected_ids if cid not in resolved]
    if missing_ids:
        raise RuntimeError(f"Unresolved candidate IDs after fallback scoring: {missing_ids[:10]}")

    return {cid: resolved[cid] for cid in expected_ids}


def score_candidates(cdf: pd.DataFrame, variant: Dict[str, Any]) -> pd.DataFrame:
    if cdf.empty:
        cdf = cdf.copy()
        cdf["score"] = pd.Series(dtype="int64")
        return cdf

    result = cdf.copy().reset_index(drop=True)
    result["score"] = pd.NA

    for week_id, group_idx in result.groupby("week_id", sort=True).groups.items():
        idxs = list(group_idx)
        print(f"  🧠 Scoring variant={variant['variant_id']} week={week_id} candidates={len(idxs)}")
        for start in range(0, len(idxs), BATCH_SIZE):
            batch_idxs = idxs[start:start + BATCH_SIZE]
            batch_df = result.loc[batch_idxs, ["candidate_id", "quote"]].copy()
            batch_records = batch_df.to_dict(orient="records")
            scores = score_records_with_fallback(variant, batch_records)

            for idx in batch_idxs:
                cid = result.at[idx, "candidate_id"]
                result.at[idx, "score"] = scores[cid]

            time.sleep(DELAY_BETWEEN_REQUESTS)

    result["score"] = result["score"].astype(int)
    return result


def select_top_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["variant_id", "prompt_family", "approach", "week_id", "debate_date", "speaker_name", "quote", "score", "speech_id", "section_name", "word_count", "week_rank"])

    ranked = scored.sort_values(
        ["variant_id", "week_id", "score", "speaker_name", "quote"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    ranked["week_rank"] = ranked.groupby(["variant_id", "week_id"]).cumcount() + 1
    top = ranked[ranked["week_rank"] <= TOP_N_PER_WEEK].copy()
    return top[["variant_id", "prompt_family", "approach", "week_id", "debate_date", "speaker_name", "quote", "score", "speech_id", "section_name", "word_count", "week_rank"]].reset_index(drop=True)


def build_summary(top_rows: pd.DataFrame, all_scored: pd.DataFrame) -> pd.DataFrame:
    if all_scored.empty:
        return pd.DataFrame(columns=["variant_id", "prompt_family", "approach", "week_id", "scored_candidates", "top_rows", "max_score", "avg_top_score"])

    rows: List[Dict[str, Any]] = []
    grouped = all_scored.groupby(["variant_id", "prompt_family", "approach", "week_id"], sort=True)
    for (variant_id, prompt_family, approach, week_id), g in grouped:
        top_g = top_rows[(top_rows["variant_id"] == variant_id) & (top_rows["week_id"] == week_id)].copy()
        rows.append({
            "variant_id": variant_id,
            "prompt_family": prompt_family,
            "approach": approach,
            "week_id": week_id,
            "scored_candidates": int(len(g)),
            "top_rows": int(len(top_g)),
            "max_score": int(g["score"].max()) if len(g) else None,
            "avg_top_score": float(top_g["score"].mean()) if len(top_g) else None,
        })
    return pd.DataFrame(rows).sort_values(["variant_id", "week_id"]).reset_index(drop=True)


def write_dataframe_outputs(df: pd.DataFrame, csv_key: str, parquet_key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    s3_put_text(S3_BUCKET, csv_key, buf.getvalue(), content_type="text/csv")

    table = pa.Table.from_pandas(df, preserve_index=False)
    out_buf = io.BytesIO()
    pq.write_table(table, out_buf, compression="snappy")
    s3_put_bytes(S3_BUCKET, parquet_key, out_buf.getvalue(), content_type="application/x-parquet")


def run_variant(df_speeches: pd.DataFrame, variant: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    variant_id = str(variant["variant_id"])
    approach = str(variant["approach"])
    print(f"\n🚀 Running variant={variant_id} approach={approach}")

    if approach == "sentence_score":
        candidates = build_sentence_candidates(df_speeches, variant)
    elif approach == "extract_then_score":
        candidates = extract_candidates_from_speeches(df_speeches, variant)
    else:
        raise RuntimeError(f"Unknown approach: {approach}")

    print(f"  🧾 Candidates built: {len(candidates)}")
    scored = score_candidates(candidates, variant)
    print(f"  ✅ Candidates scored: {len(scored)}")
    top_rows = select_top_rows(scored)
    print(f"  🏁 Top rows retained: {len(top_rows)}")
    return scored, top_rows


def main() -> None:
    print(f"📥 Loading debate speeches from s3://{S3_BUCKET}/{INPUT_KEY}")
    df_speeches = load_input_speeches()
    print(f"📆 Loaded {len(df_speeches)} speeches across target weeks: {', '.join(TARGET_WEEK_IDS)}")

    variants = filter_variants(load_prompt_variants(PROMPTS_CONFIG))
    if not variants:
        raise RuntimeError("No prompt variants selected.")
    print(f"🧪 Selected {len(variants)} prompt variants.")

    all_scored_frames: List[pd.DataFrame] = []
    all_top_frames: List[pd.DataFrame] = []

    for i, variant in enumerate(variants, start=1):
        scored, top_rows = run_variant(df_speeches, variant)
        if not scored.empty:
            all_scored_frames.append(scored)
        if not top_rows.empty:
            all_top_frames.append(top_rows)

        if AUTOSAVE_VARIANT_INTERVAL > 0 and i % AUTOSAVE_VARIANT_INTERVAL == 0:
            scored_df = pd.concat(all_scored_frames, ignore_index=True) if all_scored_frames else pd.DataFrame()
            top_df = pd.concat(all_top_frames, ignore_index=True) if all_top_frames else pd.DataFrame()
            summary_df = build_summary(top_df, scored_df)
            print(f"💾 Autosaving after {i} variant(s)...")
            write_dataframe_outputs(top_df, OUTPUT_KEY, OUTPUT_PARQUET_KEY)
            write_dataframe_outputs(summary_df, SUMMARY_KEY, SUMMARY_PARQUET_KEY)

    scored_df = pd.concat(all_scored_frames, ignore_index=True) if all_scored_frames else pd.DataFrame()
    top_df = pd.concat(all_top_frames, ignore_index=True) if all_top_frames else pd.DataFrame()
    summary_df = build_summary(top_df, scored_df)

    print(f"📤 Writing top-row experiment output to s3://{S3_BUCKET}/{OUTPUT_KEY}")
    print(f"📤 Writing summary output to s3://{S3_BUCKET}/{SUMMARY_KEY}")
    write_dataframe_outputs(top_df, OUTPUT_KEY, OUTPUT_PARQUET_KEY)
    write_dataframe_outputs(summary_df, SUMMARY_KEY, SUMMARY_PARQUET_KEY)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
