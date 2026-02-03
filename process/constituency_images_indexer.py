#!/usr/bin/env python3
"""
constituency_images_indexer.py

Lists public images in:
  s3://eirepolitic-data/processed/constituencies/images/

Writes an index table (filename, s3_key, url) to:
  CSV:    s3://eirepolitic-data/processed/constituencies/constituency_images.csv
  Parquet:s3://eirepolitic-data/processed/constituencies/parquets/constituency_images.parquet

Env vars:
- AWS_REGION (default: ca-central-1)
- S3_BUCKET (default: eirepolitic-data)
- SOURCE_PREFIX (default: processed/constituencies/images/)
- OUTPUT_CSV_KEY (default: processed/constituencies/constituency_images.csv)
- OUTPUT_PARQUET_KEY (default: processed/constituencies/parquets/constituency_images.parquet)
"""

import io
import os
from urllib.parse import quote

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
S3_BUCKET = os.getenv("S3_BUCKET", "eirepolitic-data")

SOURCE_PREFIX = os.getenv("SOURCE_PREFIX", "processed/constituencies/images/").lstrip("/")
OUTPUT_CSV_KEY = os.getenv("OUTPUT_CSV_KEY", "processed/constituencies/constituency_images.csv").lstrip("/")
OUTPUT_PARQUET_KEY = os.getenv("OUTPUT_PARQUET_KEY", "processed/constituencies/parquets/constituency_images.parquet").lstrip("/")

# "all images in folder" => filter to typical image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".svg"}

s3 = boto3.client("s3", region_name=AWS_REGION)


def is_image_key(key: str) -> bool:
    k = (key or "").lower()
    for ext in IMAGE_EXTS:
        if k.endswith(ext):
            return True
    return False


def public_s3_url(bucket: str, region: str, key: str) -> str:
    # Encode key safely for URLs (spaces etc.)
    return f"https://{bucket}.s3.{region}.amazonaws.com/{quote(key)}"


def list_s3_keys(bucket: str, prefix: str):
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []) or []:
            yield obj["Key"]

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8-sig"), ContentType="text/csv")


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    out = io.BytesIO()
    pq.write_table(table, out, compression="snappy")
    s3.put_object(Bucket=bucket, Key=key, Body=out.getvalue(), ContentType="application/x-parquet")


def main() -> None:
    print(f"📥 Listing images from s3://{S3_BUCKET}/{SOURCE_PREFIX}")
    keys = [k for k in list_s3_keys(S3_BUCKET, SOURCE_PREFIX) if is_image_key(k)]

    print(f"🧾 Found {len(keys)} image objects")
    rows = []
    for k in keys:
        filename = k.split("/")[-1]
        rows.append(
            {
                "filename": filename,
                "s3_key": k,
                "url": public_s3_url(S3_BUCKET, AWS_REGION, k),
            }
        )

    df = pd.DataFrame(rows).sort_values(["filename"], kind="stable").reset_index(drop=True)

    print(f"📤 Writing CSV  -> s3://{S3_BUCKET}/{OUTPUT_CSV_KEY}")
    write_csv_to_s3(df, S3_BUCKET, OUTPUT_CSV_KEY)

    print(f"📤 Writing Parquet -> s3://{S3_BUCKET}/{OUTPUT_PARQUET_KEY}")
    write_parquet_to_s3(df, S3_BUCKET, OUTPUT_PARQUET_KEY)

    print("✅ Done.")


if __name__ == "__main__":
    main()
