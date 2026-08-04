from __future__ import annotations

import csv
import io
import os
from datetime import datetime
from typing import Any, Callable

import boto3
from botocore.exceptions import ClientError

from instagram.renderer.constants import DEFAULT_BUCKET, DEFAULT_REGION
from instagram.visuals.s3_resolver import BATCH_ROOT, batch_key_for_logical_key

RecordBuilder = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


def _read_csv_body(body: Any) -> list[dict[str, Any]]:
    text = body.read().decode("utf-8-sig", errors="replace")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def _current_batch_id(source_manifest: dict[str, Any]) -> str | None:
    for source_name in ("members", "speeches"):
        meta = source_manifest.get(source_name)
        if not isinstance(meta, dict):
            continue
        resolution = meta.get("resolution")
        if isinstance(resolution, dict) and resolution.get("batch_id"):
            return str(resolution["batch_id"])
    return None


def annotate_current_records(
    records: list[dict[str, Any]],
    source_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    batch_id = _current_batch_id(source_manifest)
    members = source_manifest.get("members") if isinstance(source_manifest.get("members"), dict) else {}
    speeches = source_manifest.get("speeches") if isinstance(source_manifest.get("speeches"), dict) else {}
    for record in records:
        record.setdefault("data_origin", "current_real")
        record.setdefault("source_batch_id", batch_id)
        record.setdefault("source_member_key", members.get("resolved_key") or members.get("key"))
        record.setdefault("source_speech_key", speeches.get("resolved_key") or speeches.get("key"))
    return records


def _list_batch_ids(client: Any, *, bucket: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    batch_ids: list[str] = []
    prefix = f"{BATCH_ROOT}/"
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            value = str(item.get("Prefix") or "")
            if not value.startswith(prefix):
                continue
            batch_id = value[len(prefix) :].strip("/")
            if batch_id:
                batch_ids.append(batch_id)
    return sorted(set(batch_ids))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def load_historical_joined_records(
    *,
    data_source: str,
    project: dict[str, Any],
    current_source_manifest: dict[str, Any],
    member_logical_key: str,
    speech_logical_key: str,
    build_records: RecordBuilder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = project.get("validation", {}).get("historical_search", {})
    enabled = bool(policy.get("enabled", False)) if isinstance(policy, dict) else False
    max_batches = int(policy.get("max_batches", 12)) if isinstance(policy, dict) else 12
    required = bool(policy.get("required", True)) if isinstance(policy, dict) else True

    manifest: dict[str, Any] = {
        "enabled": enabled,
        "required": required,
        "max_batches": max_batches,
        "search_order": ["current_real", "historical_real", "synthetic_contract_edge", "waived"],
        "current_batch_id": _current_batch_id(current_source_manifest),
        "attempted_batches": [],
        "loaded_batches": [],
        "record_count": 0,
    }
    if not enabled:
        manifest.update({"status": "disabled", "reason": "Project historical_search.enabled is false."})
        return [], manifest
    if data_source != "s3":
        manifest.update({"status": "skipped", "reason": "Historical search is only available for S3 validation."})
        return [], manifest

    bucket = str(policy.get("bucket") or os.getenv("INSTAGRAM_VISUAL_S3_BUCKET") or DEFAULT_BUCKET)
    region = str(policy.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION)
    manifest.update({"bucket": bucket, "region": region})
    client = boto3.client("s3", region_name=region)

    try:
        batch_ids = _list_batch_ids(client, bucket=bucket)
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        if required:
            raise RuntimeError(f"Historical batch discovery failed: {exc}") from exc
        return [], manifest

    current_batch_id = manifest.get("current_batch_id")
    candidates: list[dict[str, Any]] = []
    for batch_id in batch_ids:
        if current_batch_id and batch_id == current_batch_id:
            continue
        member_key = batch_key_for_logical_key(member_logical_key, batch_id)
        speech_key = batch_key_for_logical_key(speech_logical_key, batch_id)
        attempt: dict[str, Any] = {
            "batch_id": batch_id,
            "member_key": member_key,
            "speech_key": speech_key,
        }
        try:
            member_head = client.head_object(Bucket=bucket, Key=member_key)
            speech_head = client.head_object(Bucket=bucket, Key=speech_key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "unknown")
            attempt.update({"status": "skipped_missing_source", "error_code": code})
            manifest["attempted_batches"].append(attempt)
            continue
        member_modified = member_head.get("LastModified")
        speech_modified = speech_head.get("LastModified")
        timestamps = [value for value in (member_modified, speech_modified) if value is not None]
        newest = max(timestamps) if timestamps else None
        attempt.update({
            "status": "available",
            "member_last_modified": _iso(member_modified),
            "speech_last_modified": _iso(speech_modified),
            "sort_timestamp": _iso(newest),
        })
        manifest["attempted_batches"].append(attempt)
        candidates.append({**attempt, "sort_value": newest})

    candidates.sort(
        key=lambda row: (
            row.get("sort_value") is not None,
            row.get("sort_value") or datetime.min,
            row["batch_id"],
        ),
        reverse=True,
    )

    historical_records: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[:max_batches], start=1):
        batch_id = candidate["batch_id"]
        try:
            member_obj = client.get_object(Bucket=bucket, Key=candidate["member_key"])
            speech_obj = client.get_object(Bucket=bucket, Key=candidate["speech_key"])
            members = _read_csv_body(member_obj["Body"])
            speeches = _read_csv_body(speech_obj["Body"])
            records, join_manifest = build_records(members, speeches)
        except Exception as exc:
            manifest["loaded_batches"].append({
                "batch_id": batch_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            if required:
                raise RuntimeError(f"Historical batch {batch_id} could not be loaded: {exc}") from exc
            continue

        for record in records:
            record.update({
                "data_origin": "historical_real",
                "source_batch_id": batch_id,
                "source_member_key": candidate["member_key"],
                "source_speech_key": candidate["speech_key"],
                "source_last_modified": candidate.get("sort_timestamp"),
                "historical_batch_rank": rank,
            })
        historical_records.extend(records)
        manifest["loaded_batches"].append({
            "batch_id": batch_id,
            "status": "loaded",
            "rank": rank,
            "record_count": len(records),
            "member_row_count": len(members),
            "speech_row_count": len(speeches),
            "member_key": candidate["member_key"],
            "speech_key": candidate["speech_key"],
            "last_modified": candidate.get("sort_timestamp"),
            "join_manifest": join_manifest,
        })

    manifest["record_count"] = len(historical_records)
    manifest["status"] = "completed"
    manifest["available_batch_count"] = len(candidates)
    manifest["loaded_batch_count"] = len([row for row in manifest["loaded_batches"] if row["status"] == "loaded"])
    return historical_records, manifest
