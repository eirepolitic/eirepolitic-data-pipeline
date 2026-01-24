import os
import json
import time
from zoneinfo import ZoneInfo

import requests
import boto3

# ---------------- CONFIG ----------------
BASE_URL = "https://api.oireachtas.ie/v1/debates"

# Irish context: GMT (not strictly needed anymore since we don't timestamp folders,
# but kept in case you later timestamp within files/metadata)
TZ = ZoneInfo("GMT")

# Filters
CHAMBER_ID = "/ie/oireachtas/house/dail/34"
LANG = "en"

# "Ignore pagination" => single request only
SKIP = 0
LIMIT = 500

# Cloud storage (bucket is eirepolitic-data)
S3_BUCKET = os.environ.get("S3_BUCKET", "eirepolitic-data")
AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")

# Save directly into raw/debates/ (no timestamp folder)
S3_PREFIX = "raw/debates"

# Optional: throttle between uploads
UPLOAD_SLEEP = float(os.environ.get("UPLOAD_SLEEP", "0.0"))

s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------- HELPERS ----------------
def safe_get(url, params, retries=5, backoff=2.0, timeout=60):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            if r.status_code >= 500:
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(backoff * attempt)
    raise last_exc or RuntimeError("Request failed")


def extract_debate_id(item: dict) -> str:
    """
    Creates a stable filename from debateRecord.uri (preferred),
    fallback to a time-based id (rare).
    """
    rec = (item or {}).get("debateRecord", {}) or {}
    uri = (rec.get("uri") or "").strip()
    if uri:
        tail = uri.rstrip("/").split("/")[-1]
        tail = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tail)
        if tail:
            return tail
    return f"debate_{int(time.time() * 1000)}"


def s3_put_json(bucket: str, key: str, obj: dict):
    body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


# ---------------- MAIN ----------------
def download_debates_one_page_and_store_individual():
    params = {
        "chamber_id": CHAMBER_ID,
        "lang": LANG,
        "skip": SKIP,
        "limit": LIMIT,
    }

    print(f"🔄 Requesting debates (single page only) chamber_id={CHAMBER_ID} limit={LIMIT} skip={SKIP}")
    r = safe_get(BASE_URL, params=params)
    data = r.json()
    results = data.get("results", []) or []

    print(f"📥 Retrieved {len(results)} results (NOTE: pagination disabled; may not be complete if > {LIMIT}).")
    if not results:
        print("✅ No results returned. Done.")
        return

    uploaded = 0
    for item in results:
        debate_id = extract_debate_id(item)
        key = f"{S3_PREFIX}/{debate_id}.json"  # direct in debates folder
        s3_put_json(S3_BUCKET, key, item)
        uploaded += 1

        if UPLOAD_SLEEP > 0:
            time.sleep(UPLOAD_SLEEP)

    print(f"✅ Uploaded {uploaded} debate JSON files to s3://{S3_BUCKET}/{S3_PREFIX}/")
    print("ℹ️ Files are written with stable names, so re-runs will overwrite matching debate_id files.")


if __name__ == "__main__":
    download_debates_one_page_and_store_individual()
