"""S3 IO helpers for the unified Oireachtas pipeline."""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Mapping

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .merge import merge_for_policy
from .normalize import stable_json_dumps
from .schemas import get_table_schema
from .write_policies import get_write_policy


DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_REGION = "ca-central-1"
PRODUCTION_PREFIXES = (
    "processed/oireachtas_unified/latest/",
    "processed/oireachtas_unified/compat/",
)
_LATEST_TABLE_PATTERN = re.compile(r"^processed/oireachtas_unified/latest/(?:csv|parquet)/([^/.]+)\.(?:csv|parquet)$")


def make_s3_client(*, region_name: str = DEFAULT_REGION) -> Any:
    return boto3.client("s3", region_name=region_name)


def production_publishing_enabled() -> bool:
    repo_switch = os.getenv("OIREACHTAS_PUBLISH_ENABLED", "false").strip().lower()
    run_switch = os.getenv("OIREACHTAS_PUBLISH_LATEST", "false").strip().lower()
    truthy = {"1", "true", "yes", "on"}
    return repo_switch in truthy and run_switch in truthy


def latest_publishing_enabled() -> bool:
    return production_publishing_enabled()


def is_unified_production_key(key: str) -> bool:
    return key.startswith(PRODUCTION_PREFIXES)


def is_unified_latest_key(key: str) -> bool:
    return key.startswith(PRODUCTION_PREFIXES[0])


def put_bytes(s3: Any, *, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    if is_unified_production_key(key) and not production_publishing_enabled():
        return
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def get_bytes(s3: Any, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def put_json(s3: Any, *, bucket: str, key: str, payload: Mapping[str, Any]) -> None:
    put_bytes(s3, bucket=bucket, key=key, body=stable_json_dumps(payload).encode("utf-8"), content_type="application/json")


def get_json(s3: Any, *, bucket: str, key: str) -> Any:
    return json.loads(get_bytes(s3, bucket=bucket, key=key).decode("utf-8"))


def put_text(s3: Any, *, bucket: str, key: str, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
    put_bytes(s3, bucket=bucket, key=key, body=text.encode("utf-8"), content_type=content_type)


def put_dataframe_csv(s3: Any, *, bucket: str, key: str, df: pd.DataFrame, include_bom: bool = False) -> None:
    prepared = _prepare_latest_dataframe(s3, bucket=bucket, key=key, incoming=df, file_format="csv")
    encoding = "utf-8-sig" if include_bom else "utf-8"
    put_bytes(s3, bucket=bucket, key=key, body=prepared.to_csv(index=False).encode(encoding), content_type="text/csv")


def put_dataframe_parquet(s3: Any, *, bucket: str, key: str, df: pd.DataFrame, compression: str = "snappy") -> None:
    prepared = _prepare_latest_dataframe(s3, bucket=bucket, key=key, incoming=df, file_format="parquet")
    table = pa.Table.from_pandas(prepared, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression=compression)
    put_bytes(s3, bucket=bucket, key=key, body=buffer.getvalue(), content_type="application/x-parquet")


def _prepare_latest_dataframe(s3: Any, *, bucket: str, key: str, incoming: pd.DataFrame, file_format: str) -> pd.DataFrame:
    """Merge mutable latest data according to the table registry.

    Immutable run-scoped objects are returned unchanged. The publishing guard is
    checked before any read, so validation runs never depend on production state.
    """
    table_name = _latest_table_name(key)
    if not table_name or not production_publishing_enabled():
        return incoming.copy()
    policy = get_write_policy(table_name)
    schema = get_table_schema(table_name)
    if policy.write_strategy in {"snapshot_replace", "rebuild"}:
        return incoming.copy()
    existing = _read_existing_dataframe(s3, bucket=bucket, key=key, file_format=file_format)
    return merge_for_policy(existing, incoming, primary_key=schema.primary_key, policy=policy)


def _read_existing_dataframe(s3: Any, *, bucket: str, key: str, file_format: str) -> pd.DataFrame:
    try:
        body = get_bytes(s3, bucket=bucket, key=key)
    except Exception:
        return pd.DataFrame()
    if not body:
        return pd.DataFrame()
    if file_format == "csv":
        return pd.read_csv(io.BytesIO(body), dtype=object)
    return pq.read_table(io.BytesIO(body)).to_pandas()


def _latest_table_name(key: str) -> str | None:
    match = _LATEST_TABLE_PATTERN.match(key)
    return match.group(1) if match else None


def object_exists(s3: Any, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False
