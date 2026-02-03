"""
delete_s3_column.py

Deletes a specified column from BOTH a CSV and Parquet file in S3 (same dataset).

Inputs (env vars):
- AWS_REGION (default: ca-central-1)
- S3_BUCKET (default: eirepolitic-data)

- CSV_KEY (required)     e.g. processed/members/members_summaries.csv
- PARQUET_KEY (required) e.g. processed/members/parquets/members_summaries.parquet
- COLUMN (required)      e.g. background

Behavior controls:
- STRICT (default: "0")
    "1" => error if COLUMN not found in either file
    "0" => no-op for the file(s) where COLUMN not present
"""

import io
import os
import sys

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")

CSV_KEY = os.getenv("CSV_KEY", "").strip()
PARQUET_KEY = os.getenv("PARQUET_KEY", "").strip()
COLUMN = os.getenv("COLUMN", "").strip()

STRICT = os.getenv("STRICT", "0").strip() == "1"

s3 = boto3.client("s3", region_name=AWS_REGION)


def require(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")


def s3_get_bytes(bucket: str, key: str) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def drop_column_csv(csv_bytes: bytes, column: str) -> tuple[bytes, bool]:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if column not in df.columns:
        return csv_bytes, False
    df = df.drop(columns=[column])
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig"), True


def drop_column_parquet(parquet_bytes: bytes, column: str) -> tuple[bytes, bool]:
    table = pq.read_table(io.BytesIO(parquet_bytes))
    if column not in table.schema.names:
        return parquet_bytes, False
    table2 = table.drop([column])
    out = io.BytesIO()
    pq.write_table(table2, out, compression="snappy")
    return out.getvalue(), True


def main() -> None:
    require("CSV_KEY", CSV_KEY)
    require("PARQUET_KEY", PARQUET_KEY)
    require("COLUMN", COLUMN)

    print(f"🪣 Bucket: {S3_BUCKET}")
    print(f"📄 CSV:     s3://{S3_BUCKET}/{CSV_KEY}")
    print(f"🧱 Parquet: s3://{S3_BUCKET}/{PARQUET_KEY}")
    print(f"🧹 Column to delete: {COLUMN}")
    print(f"⚙️ STRICT: {STRICT}\n")

    # ---- CSV ----
    csv_bytes = s3_get_bytes(S3_BUCKET, CSV_KEY)
    new_csv_bytes, csv_dropped = drop_column_csv(csv_bytes, COLUMN)

    # ---- Parquet ----
    pq_bytes = s3_get_bytes(S3_BUCKET, PARQUET_KEY)
    new_pq_bytes, pq_dropped = drop_column_parquet(pq_bytes, COLUMN)

    if STRICT and (not csv_dropped or not pq_dropped):
        missing = []
        if not csv_dropped:
            missing.append("CSV")
        if not pq_dropped:
            missing.append("PARQUET")
        raise RuntimeError(f"Column '{COLUMN}' not found in: {', '.join(missing)} (STRICT=1)")

    # Always write back (keeps behavior consistent + avoids guessing)
    s3_put_bytes(S3_BUCKET, CSV_KEY, new_csv_bytes, content_type="text/csv")
    s3_put_bytes(S3_BUCKET, PARQUET_KEY, new_pq_bytes, content_type="application/x-parquet")

    print("✅ Done.")
    print(f"   CSV column removed: {csv_dropped}")
    print(f"   Parquet column removed: {pq_dropped}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise
