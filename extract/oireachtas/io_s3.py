"""S3 IO helpers for the unified Oireachtas pipeline."""

from __future__ import annotations

import io
import json
from typing import Any, Mapping, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import stable_json_dumps


DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_REGION = "ca-central-1"


def make_s3_client(*, region_name: str = DEFAULT_REGION) -> Any:
    """Create a boto3 S3 client."""
    return boto3.client("s3", region_name=region_name)


def put_bytes(
    s3: Any,
    *,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    """Write bytes to S3."""
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def get_bytes(s3: Any, *, bucket: str, key: str) -> bytes:
    """Read bytes from S3."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def put_json(
    s3: Any,
    *,
    bucket: str,
    key: str,
    payload: Mapping[str, Any],
) -> None:
    """Write deterministic UTF-8 JSON to S3."""
    body = stable_json_dumps(payload).encode("utf-8")
    put_bytes(s3, bucket=bucket, key=key, body=body, content_type="application/json")


def get_json(s3: Any, *, bucket: str, key: str) -> Any:
    """Read JSON from S3."""
    return json.loads(get_bytes(s3, bucket=bucket, key=key).decode("utf-8"))


def put_text(
    s3: Any,
    *,
    bucket: str,
    key: str,
    text: str,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    """Write UTF-8 text to S3."""
    put_bytes(s3, bucket=bucket, key=key, body=text.encode("utf-8"), content_type=content_type)


def put_dataframe_csv(
    s3: Any,
    *,
    bucket: str,
    key: str,
    df: pd.DataFrame,
    include_bom: bool = False,
) -> None:
    """Write a DataFrame as CSV to S3."""
    encoding = "utf-8-sig" if include_bom else "utf-8"
    body = df.to_csv(index=False).encode(encoding)
    put_bytes(s3, bucket=bucket, key=key, body=body, content_type="text/csv")


def put_dataframe_parquet(
    s3: Any,
    *,
    bucket: str,
    key: str,
    df: pd.DataFrame,
    compression: str = "snappy",
) -> None:
    """Write a DataFrame as Parquet to S3 using pyarrow."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression=compression)
    put_bytes(s3, bucket=bucket, key=key, body=buffer.getvalue(), content_type="application/x-parquet")


def object_exists(s3: Any, *, bucket: str, key: str) -> bool:
    """Return whether an object exists in S3."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False
