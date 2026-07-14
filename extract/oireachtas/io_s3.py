"""S3 IO helpers for the unified Oireachtas pipeline."""

from __future__ import annotations

import io
import json
import os
from typing import Any, Mapping

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import stable_json_dumps


DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_REGION = "ca-central-1"
PRODUCTION_PREFIXES = (
    "processed/oireachtas_unified/latest/",
    "processed/oireachtas_unified/compat/",
)


def make_s3_client(*, region_name: str = DEFAULT_REGION) -> Any:
    """Create a boto3 S3 client."""
    return boto3.client("s3", region_name=region_name)


def production_publishing_enabled() -> bool:
    """Return whether writes to mutable unified production keys are enabled.

    Publishing is deliberately default-deny. Both the repository-level switch and
    the per-run latest flag must be enabled before mutable production objects can
    be changed.
    """
    repo_switch = os.getenv("OIREACHTAS_PUBLISH_ENABLED", "false").strip().lower()
    run_switch = os.getenv("OIREACHTAS_PUBLISH_LATEST", "false").strip().lower()
    truthy = {"1", "true", "yes", "on"}
    return repo_switch in truthy and run_switch in truthy


def latest_publishing_enabled() -> bool:
    """Backward-compatible alias for the production publishing guard."""
    return production_publishing_enabled()


def is_unified_production_key(key: str) -> bool:
    """Return whether an S3 key is a mutable unified production output."""
    return key.startswith(PRODUCTION_PREFIXES)


def is_unified_latest_key(key: str) -> bool:
    """Backward-compatible predicate for callers and tests."""
    return key.startswith(PRODUCTION_PREFIXES[0])


def put_bytes(
    s3: Any,
    *,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    """Write bytes to S3, suppressing mutable production writes unless enabled."""
    if is_unified_production_key(key) and not production_publishing_enabled():
        return
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
