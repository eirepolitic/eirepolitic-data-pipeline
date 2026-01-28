"""
debate_speeches_csv_to_parquet.py

Pipeline module:
- Reads the classified CSV from S3
- Writes a Parquet version to the same S3 folder (same base name)
  e.g. processed/debates/debate_speeches_classified.csv
    -> processed/debates/debate_speeches_classified.parquet

Env vars (optional):
- S3_BUCKET   (default: eirepolitic-data)
- CSV_KEY     (default: processed/debates/debate_speeches_classified.csv)
- PARQUET_KEY (default: same as CSV_KEY but .parquet)
"""

import io
import os
import sys
import pandas as pd
import boto3
from botocore.exceptions import ClientError


def _default_parquet_key(csv_key: str) -> str:
    if csv_key.lower().endswith(".csv"):
        return csv_key[:-4] + ".parquet"
    return csv_key + ".parquet"


def main() -> int:
    bucket = os.getenv("S3_BUCKET", "eirepolitic-data")
    csv_key = os.getenv("CSV_KEY", "processed/debates/debate_speeches_classified.csv")
    parquet_key = os.getenv("PARQUET_KEY", _default_parquet_key(csv_key))

    s3 = boto3.client("s3")

    print(f"📥 Loading CSV from s3://{bucket}/{csv_key}")
    try:
        obj = s3.get_object(Bucket=bucket, Key=csv_key)
        csv_bytes = obj["Body"].read()
    except ClientError as e:
        print(f"❌ Failed to read CSV from S3: {e}")
        return 1

    # Read CSV robustly for typical pipeline outputs
    # If your CSV uses utf-8-sig, this will still work.
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)

    print(f"🧾 Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Write Parquet to an in-memory buffer (requires pyarrow or fastparquet installed)
    out_buf = io.BytesIO()
    try:
        df.to_parquet(out_buf, index=False, engine="pyarrow")
    except Exception as e:
        print("❌ Failed to write parquet. Ensure 'pyarrow' is installed.")
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
