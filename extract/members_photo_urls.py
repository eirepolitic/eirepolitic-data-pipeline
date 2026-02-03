"""
members_photo_urls.py

Builds a member table (member_code, full_name, photo_url) by scraping the
Oireachtas public member profile page for the profile photo URL.

Input:
  s3://<bucket>/raw/members/oireachtas_members_34th_dail.csv

Outputs:
  s3://<bucket>/processed/members/members_photo_urls.csv
  s3://<bucket>/processed/members/parquets/members_photo_urls.parquet

Resumable:
  - If output CSV exists, only fills rows where photo_url is missing.

Env vars:
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
- S3_BUCKET (default: eirepolitic-data)
- INPUT_KEY (default: raw/members/oireachtas_members_34th_dail.csv)
- OUTPUT_CSV_KEY (default: processed/members/members_photo_urls.csv)
- OUTPUT_PARQUET_KEY (default: processed/members/parquets/members_photo_urls.parquet)
- REQUEST_TIMEOUT (default: 10)
- DELAY_BETWEEN_REQUESTS (default: 0.2)
- AUTOSAVE_INTERVAL (default: 50)
- TEST_ROWS (default: 0)
"""

import io
import os
import time
from typing import Dict, Optional
from urllib.parse import urljoin

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup

import pyarrow as pa
import pyarrow.parquet as pq


S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")
INPUT_KEY = os.getenv("INPUT_KEY", "raw/members/oireachtas_members_34th_dail.csv")
OUTPUT_CSV_KEY = os.getenv("OUTPUT_CSV_KEY", "processed/members/members_photo_urls.csv")
OUTPUT_PARQUET_KEY = os.getenv("OUTPUT_PARQUET_KEY", "processed/members/parquets/members_photo_urls.parquet")

AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "0.2"))
AUTOSAVE_INTERVAL = int(os.getenv("AUTOSAVE_INTERVAL", "50"))
TEST_ROWS = int(os.getenv("TEST_ROWS", "0"))

KEY_COL = "member_code"
NAME_COL = "full_name"
URI_COL = "uri"
OUT_COL = "photo_url"

s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------- S3 HELPERS ----------------
def s3_get_text(bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8-sig", errors="replace")


def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8-sig"), ContentType=content_type)


def s3_put_bytes(bucket: str, key: str, b: bytes, content_type: str = "application/x-parquet") -> None:
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


def is_missing(v) -> bool:
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return v is None or (isinstance(v, str) and v.strip() == "")


# ---------------- SCRAPING ----------------
def to_public_profile_url(member_uri: str) -> Optional[str]:
    if not member_uri or not isinstance(member_uri, str):
        return None

    # Convert data.oireachtas.ie member URI -> oireachtas.ie public page
    if "data.oireachtas.ie" in member_uri and "/ie/oireachtas/member/id/" in member_uri:
        return (
            member_uri.replace(
                "https://data.oireachtas.ie/ie/oireachtas/member/id/",
                "https://www.oireachtas.ie/en/members/member/",
            ).rstrip("/")
            + "/"
        )

    # If it’s already a public URL, just normalize trailing slash
    if member_uri.startswith("http://") or member_uri.startswith("https://"):
        return member_uri.rstrip("/") + "/"

    return None


def scrape_photo_url(profile_url: str) -> Optional[str]:
    r = requests.get(profile_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Current structure (your confirmed selector)
    img_tag = soup.select_one("img.c-member-about__img")

    # Fallback patterns
    if not img_tag:
        img_tag = (
            soup.select_one("img.member-profile-photo")
            or soup.select_one("div.member-image img")
            or soup.find("img", src=lambda s: s and "/media/members/photo/" in s)
        )

    if not img_tag or not img_tag.get("src"):
        return None

    return urljoin(profile_url, img_tag["src"])


# ---------------- PARQUET ----------------
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

    for col in (KEY_COL, NAME_COL, URI_COL):
        if col not in df_in.columns:
            raise RuntimeError(f"Input CSV missing required column: {col}")

    df_base = df_in[[KEY_COL, NAME_COL, URI_COL]].copy()

    # Load existing output (resumable)
    if s3_exists(S3_BUCKET, OUTPUT_CSV_KEY):
        print(f"📥 Loading existing output from s3://{S3_BUCKET}/{OUTPUT_CSV_KEY}")
        out_text = s3_get_text(S3_BUCKET, OUTPUT_CSV_KEY)
        df_out = pd.read_csv(io.StringIO(out_text))
    else:
        print("📄 No existing output found — creating new output.")
        df_out = pd.DataFrame(columns=[KEY_COL, NAME_COL, "photo_url"])

    existing_map: Dict[str, str] = {}
    if not df_out.empty and KEY_COL in df_out.columns and OUT_COL in df_out.columns:
        for code, val in zip(df_out[KEY_COL].astype(str), df_out[OUT_COL]):
            if code and not is_missing(val):
                existing_map[code.strip()] = str(val).strip()

    # Start result with base ordering
    df_res = df_base[[KEY_COL, NAME_COL]].copy()
    df_res[OUT_COL] = df_base[KEY_COL].astype(str).map(lambda c: existing_map.get(c.strip(), pd.NA))

    # Indices needing scrape
    idxs = df_res.index[df_res[OUT_COL].apply(is_missing)].tolist()
    if TEST_ROWS and TEST_ROWS > 0:
        idxs = idxs[:TEST_ROWS]
        print(f"🧪 Test mode: processing first {len(idxs)} missing rows.")
    else:
        print(f"📄 Rows needing photo_url: {len(idxs)}")

    processed = 0
    failures = 0

    for n, idx in enumerate(idxs, start=1):
        code = str(df_base.at[idx, KEY_COL] or "").strip()
        name = str(df_base.at[idx, NAME_COL] or "").strip()
        uri = df_base.at[idx, URI_COL]

        profile_url = to_public_profile_url(str(uri) if uri is not None else "")
        print(f"\n🖼️ {n}/{len(idxs)} member_code={code} name='{name}'")

        if not profile_url:
            failures += 1
            df_res.at[idx, OUT_COL] = pd.NA
            print("⚠️ No valid profile URL.")
            continue

        try:
            photo_url = scrape_photo_url(profile_url)
            if not photo_url:
                failures += 1
                df_res.at[idx, OUT_COL] = pd.NA
                print("❌ No image found on page.")
            else:
                df_res.at[idx, OUT_COL] = photo_url
                print(f"✅ {photo_url}")
        except Exception as e:
            failures += 1
            df_res.at[idx, OUT_COL] = pd.NA
            print(f"⚠️ Error: {e}")

        processed += 1
        if processed % AUTOSAVE_INTERVAL == 0:
            print("💾 Autosaving CSV + Parquet to S3...")
            buf = io.StringIO()
            df_res.to_csv(buf, index=False, encoding="utf-8-sig")
            s3_put_text(S3_BUCKET, OUTPUT_CSV_KEY, buf.getvalue())
            s3_put_bytes(S3_BUCKET, OUTPUT_PARQUET_KEY, dataframe_to_parquet_bytes(df_res))

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("\n📤 Writing final CSV + Parquet to S3...")
    buf = io.StringIO()
    df_res.to_csv(buf, index=False, encoding="utf-8-sig")
    s3_put_text(S3_BUCKET, OUTPUT_CSV_KEY, buf.getvalue())
    s3_put_bytes(S3_BUCKET, OUTPUT_PARQUET_KEY, dataframe_to_parquet_bytes(df_res))

    print("🎉 Done.")
    print(f"   Failures/no-image: {failures}")
    print(f"   CSV:     s3://{S3_BUCKET}/{OUTPUT_CSV_KEY}")
    print(f"   Parquet: s3://{S3_BUCKET}/{OUTPUT_PARQUET_KEY}")


if __name__ == "__main__":
    main()
