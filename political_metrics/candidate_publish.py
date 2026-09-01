from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pandas as pd

from extract.oireachtas.batch import batch_key_for_production_key, record_batch_table, validate_batch_id
from political_metrics.materialize import DatasetContract, validate_materialized_frame


def logical_metric_key(dataset: DatasetContract, fmt: str) -> str:
    if fmt not in {"csv", "parquet"}:
        raise ValueError(f"unsupported metric format: {fmt}")
    return (
        "processed/oireachtas_unified/latest/metrics/"
        f"{dataset.cadence}/{dataset.name}/{fmt}/{dataset.name}.{fmt}"
    )


def _serialize(frame: pd.DataFrame, fmt: str) -> tuple[bytes, str]:
    if fmt == "csv":
        return frame.to_csv(index=False).encode("utf-8"), "text/csv"
    if fmt == "parquet":
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue(), "application/octet-stream"
    raise ValueError(f"unsupported metric format: {fmt}")


def publish_dataset_to_candidate(
    s3,
    *,
    bucket: str,
    batch_id: str,
    frame: pd.DataFrame,
    dataset: DatasetContract,
    contract_version: int,
    source_batch_id: str,
) -> dict:
    """Write one validated metric dataset into an immutable candidate batch.

    This function does not write `latest` objects and cannot update production or
    previous pointers. Promotion remains the responsibility of the existing batch
    control workflow after the assembled batch passes validation.
    """
    batch_id = validate_batch_id(batch_id)
    if source_batch_id != batch_id:
        raise ValueError(
            f"metric source_batch_id must equal candidate batch_id; source={source_batch_id!r}, candidate={batch_id!r}"
        )

    errors = validate_materialized_frame(
        frame,
        dataset,
        expected_source_batch_id=batch_id,
    )
    if errors:
        raise ValueError(f"{dataset.name} validation failed: {errors}")

    ordered = frame[dataset.columns].copy()
    logical_keys: list[str] = []
    objects: list[dict] = []
    for fmt in dataset.formats:
        logical_key = logical_metric_key(dataset, fmt)
        batch_key = batch_key_for_production_key(logical_key, batch_id)
        body, content_type = _serialize(ordered, fmt)
        s3.put_object(Bucket=bucket, Key=batch_key, Body=body, ContentType=content_type)
        logical_keys.append(logical_key)
        objects.append({"format": fmt, "logical_key": logical_key, "batch_key": batch_key, "bytes": len(body)})

    generated_at = datetime.now(timezone.utc).isoformat()
    entry_name = f"political_metrics_{dataset.name}"
    manifest = {
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "dataset": dataset.name,
        "output_rows": int(len(ordered)),
        "source_batch_id": batch_id,
        "contract_version": contract_version,
        "generated_at_utc": generated_at,
        "objects": objects,
    }
    schema = {
        "primary_key": dataset.primary_key,
        "columns": dataset.columns,
    }
    dq = {
        "dq_status": "pass",
        "checks": {
            "materialization_contract": "pass",
            "source_batch_matches_candidate": "pass",
            "primary_key_unique": "pass",
        },
    }
    entry = record_batch_table(
        s3,
        bucket=bucket,
        batch_id=batch_id,
        table=entry_name,
        manifest=manifest,
        schema=schema,
        dq=dq,
        candidate_keys=logical_keys,
    )
    return {
        "entry_name": entry_name,
        "dataset": dataset.name,
        "row_count": int(len(ordered)),
        "objects": objects,
        "batch_entry": entry,
    }
