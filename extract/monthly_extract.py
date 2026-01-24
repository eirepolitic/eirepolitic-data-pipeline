"""
monthly_extract.py

Fetch ALL debates for Dáil 34 (no date filtering), download raw XML, upload to S3.

Uploads to:
  s3://eirepolitic-data/raw/debates/xml/

Naming:
  YYYY-MM-DD__<debate_id>.xml

Overwrites are allowed on reruns.

Expected env vars (GitHub Actions):
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION

Optional env vars:
- CHAMBER_ID (default: /ie/oireachtas/house/dail/34)
- LANG (default: en)
- API_LIMIT (default: 200)
- API_SLEEP (default: 0.2)
- DOWNLOAD_TIMEOUT (default: 30)
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import requests
import boto3

API_BASE = "https://api.oireachtas.ie/v1"
DATA_BASE = "https://data.oireachtas.ie"

# Fixed bucket + prefix per your request
S3_BUCKET = "eirepolitic-data"
S3_PREFIX = "raw/debates/xml"

AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")
s3 = boto3.client("s3", region_name=AWS_REGION)

CHAMBER_ID = os.environ.get("CHAMBER_ID", "/ie/oireachtas/house/dail/34")
LANG = os.environ.get("LANG", "en")

API_LIMIT = int(os.environ.get("API_LIMIT", "200"))
API_SLEEP = float(os.environ.get("API_SLEEP", "0.2"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "30"))


def safe_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 30,
    retries: int = 5,
    backoff: float = 2.0
) -> requests.Response:
    """
    GET with retry/backoff and basic 429 handling.
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            return r
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("safe_get unreachable")


def extract_debate_fields(item: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract:
    - debate_date (YYYY-MM-DD)
    - xml_uri (relative or absolute)
    - debate_id (best-effort stable identifier)
    """
    debate = (item or {}).get("debateRecord", {}) or {}
    debate_date = debate.get("date") or item.get("contextDate")

    formats = debate.get("formats", {}) or {}
    xml_obj = formats.get("xml", {}) or {}
    xml_uri = xml_obj.get("uri")

    debate_id = debate.get("debateId")
    if not debate_id:
        uri = debate.get("uri", "") or ""
        debate_id = uri.rstrip("/").split("/")[-1] if uri else None

    if not debate_date or not xml_uri:
        return None, None, None

    return str(debate_date).strip(), str(xml_uri).strip(), str(debate_id).strip() if debate_id else None


def normalize_xml_url(xml_uri: str) -> str:
    """
    API often returns relative URIs for data.oireachtas.ie.
    """
    if xml_uri.startswith("http://") or xml_uri.startswith("https://"):
        return xml_uri
    return f"{DATA_BASE}{xml_uri}"


def safe_filename_part(s: str) -> str:
    """
    Keep filenames S3-friendly and predictable.
    """
    s = (s or "").strip()
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s) or "unknown"


def build_s3_key(debate_date: str, debate_id: Optional[str]) -> str:
    """
    Key format:
      raw/debates/xml/YYYY-MM-DD__<debate_id>.xml
    """
    safe_date = safe_filename_part(debate_date)
    safe_id = safe_filename_part(debate_id or "unknown")
    return f"{S3_PREFIX}/{safe_date}__{safe_id}.xml"


def upload_xml_to_s3(key: str, xml_bytes: bytes) -> None:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=xml_bytes,
        ContentType="application/xml",
    )


def fetch_all_debates() -> List[Dict]:
    """
    Page through /debates for the chamber. No date filters.
    """
    results: List[Dict] = []
    skip = 0

    while True:
        params = {
            "chamber_id": CHAMBER_ID,
            "lang": LANG,
            "skip": skip,
            "limit": API_LIMIT,
        }
        r = safe_get(f"{API_BASE}/debates", params=params, timeout=DOWNLOAD_TIMEOUT)
        payload = r.json()
        page = payload.get("results", []) or []

        if not page:
            break

        results.extend(page)
        skip += API_LIMIT
        time.sleep(API_SLEEP)

    return results


def main():
    print("🚀 Extracting ALL debates (no date filtering)")
    print(f"🏛️ Chamber: {CHAMBER_ID} | Lang: {LANG}")
    print(f"🪣 Upload target: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print("🧾 Naming: YYYY-MM-DD__<debate_id>.xml (overwrites allowed)\n")

    debates = fetch_all_debates()
    print(f"📥 Debate records returned: {len(debates)}")

    uploaded = 0
    missing_xml = 0
    failed = 0

    for i, item in enumerate(debates, start=1):
        debate_date, xml_uri, debate_id = extract_debate_fields(item)
        if not debate_date or not xml_uri:
            missing_xml += 1
            continue

        xml_url = normalize_xml_url(xml_uri)
        key = build_s3_key(debate_date, debate_id)

        print(f"  [{i}/{len(debates)}] {debate_date} | id={debate_id or 'unknown'} -> {key}")

        try:
            r = safe_get(xml_url, timeout=DOWNLOAD_TIMEOUT)
            upload_xml_to_s3(key, r.content)
            uploaded += 1
            time.sleep(API_SLEEP)
        except Exception as e:
            failed += 1
            print(f"    ❌ Failed for {debate_date} (id={debate_id or 'unknown'}): {e}")

    print("\n✅ Done.")
    print(f"   Uploaded: {uploaded}")
    print(f"   Missing XML link: {missing_xml}")
    print(f"   Failed downloads/uploads: {failed}")


if __name__ == "__main__":
    main()
