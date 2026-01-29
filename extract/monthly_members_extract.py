"""
monthly_members_extract.py

Fetch ALL members for Dáil 34 (no date filtering), build a member-level table,
and upload a single CSV to S3.

Writes to:
  s3://eirepolitic-data/raw/members/oireachtas_members_34th_dail.csv

Overwrites are allowed on reruns.

Expected env vars (GitHub Actions):
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION

Optional env vars:
- API_URL (default: https://api.oireachtas.ie/v1/members?chamber=dail&house_no=34&limit=500)
- API_TIMEOUT (default: 30)
- API_SLEEP (default: 0.2)
"""

import os
import time
import io
from typing import Dict, List, Optional

import requests
import pandas as pd
import boto3

# Fixed bucket + prefix per your request
S3_BUCKET = "eirepolitic-data"
S3_KEY = "raw/members/oireachtas_members_34th_dail.csv"

AWS_REGION = os.environ.get("AWS_REGION", "ca-central-1")
s3 = boto3.client("s3", region_name=AWS_REGION)

API_URL = os.environ.get(
    "API_URL",
    "https://api.oireachtas.ie/v1/members?chamber=dail&house_no=34&limit=500",
)
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "30"))
API_SLEEP = float(os.environ.get("API_SLEEP", "0.2"))


def safe_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 30,
    retries: int = 5,
    backoff: float = 2.0,
) -> requests.Response:
    """GET with retry/backoff and basic 429 handling."""
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


def pick_dail34_membership(member: Dict) -> Dict:
    """
    Prefer explicit Dáil 34 membership.
    If not found, fall back to most recent membership by start date.
    """
    memberships = member.get("memberships", []) or []

    # 1) Try to find explicit Dáil 34 membership
    for m in memberships:
        mm = (m or {}).get("membership", {}) or {}
        house = (mm.get("house", {}) or {})
        if house.get("houseCode") == "dail" and str(house.get("houseNo")) == "34":
            return mm

    # 2) Fallback: most recent by dateRange.start
    if memberships:
        def start_date(m_obj: Dict) -> str:
            try:
                return str(m_obj["membership"]["dateRange"]["start"])
            except Exception:
                return ""

        m_sorted = sorted(memberships, key=start_date, reverse=True)
        return (m_sorted[0] or {}).get("membership", {}) or {}

    return {}


def build_members_table(payload: Dict) -> pd.DataFrame:
    rows: List[Dict] = []

    for result in (payload.get("results", []) or []):
        member = (result or {}).get("member", {}) or {}

        dail34 = pick_dail34_membership(member)
        represents = dail34.get("represents", []) or []
        parties = dail34.get("parties", []) or []

        constituency = None
        if represents:
            constituency = (((represents[0] or {}).get("represent", {}) or {}).get("showAs"))

        party = None
        if parties:
            party = ((((parties[0] or {}).get("party", {}) or {}).get("showAs")))

        rows.append(
            {
                "full_name": member.get("fullName", "") or "",
                "first_name": member.get("firstName", "") or "",
                "last_name": member.get("lastName", "") or "",
                "constituency": constituency,
                "party": party,
                "gender": member.get("gender", "") or "",
                "member_code": member.get("memberCode", "") or "",
                "uri": member.get("uri", "") or "",
            }
        )

    return pd.DataFrame(rows)


def upload_csv_to_s3(df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_KEY,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )


def main():
    print("🚀 Extracting members for Dáil 34")
    print(f"🌐 API: {API_URL}")
    print(f"🪣 Upload target: s3://{S3_BUCKET}/{S3_KEY}")
    print("🧾 Output: single CSV overwrite\n")

    r = safe_get(API_URL, timeout=API_TIMEOUT)
    payload = r.json()

    df = build_members_table(payload)

    print(f"📥 Members returned: {len(df)}")
    if len(df) == 0:
        raise RuntimeError("No members returned from API (df is empty).")

    upload_csv_to_s3(df)
    time.sleep(API_SLEEP)

    print("✅ Done.")
    print(f"   Wrote: s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == "__main__":
    main()
