"""
debate_speeches_csv_to_parquet.py

Reads the classified CSV from S3 and writes a Parquet file to S3 with:
- cleaned column names: lowercase, underscores, alphanumeric only
- stable output location (default): processed/debates/parquets/<base>.parquet

Env vars (optional):
- S3_BUCKET   (default: eirepolitic-data)
- CSV_KEY     (default: processed/debates/debate_speeches_classified.csv)
- PARQUET_KEY (default: processed/debates/parquets/debate_speeches_classified.parquet)
- FORCE_STRING (default: "1")  # keep everything as string to avoid type surprises
"""

import io
import os
import re
import sys
from typing import Dict

import boto3
import pandas as pd
from botocore.exceptions import ClientError


def _default_parquet_key(csv_key: str) -> str:
    base_name = os.path.basename(csv_key)
    if base_name.lower().endswith(".csv"):
        base_name = base_name[:-4] + ".parquet"
    else:
        base_name = base_name + ".parquet"
    return f"processed/debates/parquets/{base_name}"


def _clean_column_name(name: str) -> str:
    """
    Lowercase, convert whitespace/hyphens to underscores, remove non-alphanum/_,
    collapse multiple underscores, strip underscores.
    """
    s = name.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)          # spaces/hyphens -> _
    s = re.sub(r"[^a-z0-9_]", "", s)        # drop accents/punct (keeps ascii)
    s = re.sub(r"_+", "_", s).strip("_")    # collapse/trim
    return s or "col"


def _dedupe_columns(cols) -> Dict[str, str]:
    """
    Make cleaned column names unique by suffixing _2, _3, ...
    Returns mapping {original: cleaned_unique}
    """
    seen = {}
    mapping = {}
    for c in cols:
        base = _clean_column_name(str(c))
        candidate = base
        i = 2
        while candidate in seen:
            candidate = f"{base}_{i}"
            i += 1
        seen[candidate] = True
        mapping[c] = candidate
    return mapping


def main() -> int:
    bucket = os.getenv("S3_BUCKET", "eirepolitic-data")
    csv_key = os.getenv("CSV_KEY", "processed/debates/debate_speeches_classified.csv")
    parquet_key = os.getenv("PARQUET_KEY", _default_parquet_key(csv_key))

    force_string = os.getenv("FORCE_STRING", "1") != "0"

    s3 = boto3.client("s3")

    print(f"📥 Loading CSV from s3://{bucket}/{csv_key}")
    try:
        obj = s3.get_object(Bucket=bucket, Key=csv_key)
        csv_bytes = obj["Body"].read()
    except ClientError as e:
        print(f"❌ Failed to read CSV from S3: {e}")
        return 1

    read_kwargs = dict(keep_default_na=False)
    if force_string:
        read_kwargs["dtype"] = str

    df = pd.read_csv(io.BytesIO(csv_bytes), **read_kwargs)

    print(f"🧾 Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Clean column names
    mapping = _dedupe_columns(list(df.columns))
    df = df.rename(columns=mapping)

    print("🏷️ Cleaned columns:")
    for old, new in mapping.items():
        if str(old) != new:
            print(f"  - {old} -> {new}")
        else:
            print(f"  - {old}")

    # Write Parquet to memory (requires pyarrow)
    out_buf = io.BytesIO()
    try:
        df.to_parquet(out_buf, index=False, engine="pyarrow")
    except Exception as e:
        print("❌ Failed to write Parquet. Ensure 'pyarrow' is installed.")
        print(f"   Error: {e}")
        return 1

    out_buf.seek(0)

    print(f"📤 Writing Parquet to s3://{bucket}/{parquet_key}")
    try:
        s3.put_object(Bucket=bucket, Key=parquet_key, Body=out_buf.getvalue())
    except ClientError as e:
        print(f"❌ Failed to write Parquet to S3: {e}")
        return 1

    print("🎉 Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
