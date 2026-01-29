import io
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
import boto3
from typing import List, Dict

# Fixed locations per your pipeline
BUCKET = "eirepolitic-data"
XML_PREFIX = "raw/debates/xml/"
OUTPUT_KEY = "raw/debates/debate_speeches_extracted.csv"

AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

# Akoma Ntoso namespace
NS = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0/CSD13"}


def list_s3_keys(prefix: str) -> List[str]:
    """List all object keys in BUCKET under prefix."""
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):
                continue
            if k.lower().endswith(".xml"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(keys)


def get_object_bytes(key: str) -> bytes:
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return resp["Body"].read()


def debate_date_from_key(key: str) -> str:
    """
    Files are named like: YYYY-MM-DD__<debate_id>.xml
    We keep debate date as YYYY-MM-DD.
    """
    fname = key.split("/")[-1]
    base = re.sub(r"\.xml$", "", fname, flags=re.IGNORECASE)
    if "__" in base:
        return base.split("__", 1)[0]
    return base


def extract_speeches_from_xml_bytes(xml_bytes: bytes, debate_date: str) -> List[Dict]:
    """Parse one Akoma Ntoso XML debate file bytes and return list of speech records."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        print(f"⚠️ Parse error for debate_date={debate_date}, skipping.")
        return []

    records = []
    speech_counter = 1

    for section in root.findall(".//akn:debateSection", NS):
        section_id = section.get("eId", "")

        heading_tag = section.find("./akn:heading", NS)
        if heading_tag is not None:
            heading_text = "".join(heading_tag.itertext()).strip()
        else:
            heading_text = section.get("name", "")

        for speech in section.findall(".//akn:speech", NS):
            speaker_ref = speech.get("by", "")
            speaker_name = ""

            if speaker_ref.startswith("#"):
                ref_xpath = f".//akn:TLCPerson[@eId='{speaker_ref[1:]}']"
                person = root.find(ref_xpath, NS)
                if person is not None:
                    speaker_name = person.get("showAs", "") or ""

            if not speaker_name:
                from_tag = speech.find(".//akn:from", NS)
                if from_tag is not None and from_tag.text:
                    speaker_name = from_tag.text.strip()

            paras = [p.text.strip() for p in speech.findall(".//akn:p", NS) if p.text]
            speech_text = " ".join(paras).strip()

            if speech_text:
                records.append({
                    "Debate Date": debate_date,
                    "Debate Section": section_id,
                    "Debate Section Name": heading_text,
                    "Speaker Name": speaker_name,
                    "Speech Text": speech_text,
                    "Speech Order": speech_counter
                })
                speech_counter += 1

    return records


def write_df_to_s3_csv(df: pd.DataFrame, key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buf.getvalue().encode("utf-8-sig"),
        ContentType="text/csv"
    )


def _normalize_for_dedupe(s: pd.Series) -> pd.Series:
    """
    Normalize text for deduping:
    - fill nulls
    - strip
    - collapse all whitespace to single spaces
    """
    s = s.fillna("").astype(str)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def main():
    print(f"📥 Listing XML files: s3://{BUCKET}/{XML_PREFIX}")
    keys = list_s3_keys(XML_PREFIX)
    print(f"📄 Found {len(keys)} XML files.")

    all_records: List[Dict] = []

    for i, key in enumerate(keys, start=1):
        debate_date = debate_date_from_key(key)
        print(f"  [{i}/{len(keys)}] Processing {key} (Debate Date={debate_date})")

        xml_bytes = get_object_bytes(key)
        recs = extract_speeches_from_xml_bytes(xml_bytes, debate_date)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)

    if df.empty:
        print("✅ Extracted 0 speeches total.")
        print(f"📤 Writing CSV: s3://{BUCKET}/{OUTPUT_KEY}")
        write_df_to_s3_csv(df, OUTPUT_KEY)
        print("🎉 Done.")
        return

    # Stable ordering before dedupe so "keep first" is deterministic
    df = df.sort_values(["Debate Date", "Speech Order"], ascending=[True, True]).reset_index(drop=True)

    before = len(df)

    # Create normalized helper columns used ONLY for dedupe keys
    df["_dedupe_date"] = _normalize_for_dedupe(df["Debate Date"])
    df["_dedupe_speaker"] = _normalize_for_dedupe(df["Speaker Name"])
    df["_dedupe_text"] = _normalize_for_dedupe(df["Speech Text"])

    # Drop duplicates on the requested combination
    df = df.drop_duplicates(
        subset=["_dedupe_date", "_dedupe_speaker", "_dedupe_text"],
        keep="first"
    ).reset_index(drop=True)

    # Remove helper columns
    df = df.drop(columns=["_dedupe_date", "_dedupe_speaker", "_dedupe_text"])

    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"🧹 Deduped {removed} rows (kept {after} unique speeches).")
    else:
        print("🧹 No duplicates found under (Debate Date, Speaker Name, Speech Text).")

    print(f"✅ Extracted {len(df)} speeches total (post-dedupe).")
    print(f"📤 Writing CSV: s3://{BUCKET}/{OUTPUT_KEY}")
    write_df_to_s3_csv(df, OUTPUT_KEY)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
