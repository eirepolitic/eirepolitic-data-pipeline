#!/usr/bin/env python3
"""Shared party identity and asset registry utilities.

This module is intentionally consumer-neutral. Instagram and future outputs should
resolve party assets through this registry instead of embedding party-specific paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import boto3

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs/reference/party_assets_v1.csv"
DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_ASSET_PREFIX = "processed/reference/party_assets/v1/assets/"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
VALID_STATUSES = {"approved", "approved_fallback", "source_identified_pending_ingest", "pending_review"}
GENERATED_SOURCE_TYPES = {"eirepolitic_generated_standin"}


@dataclass(frozen=True)
class PartyAsset:
    party_key: str
    party_name: str
    party_aliases: tuple[str, ...]
    logo_s3_uri: str
    source_url: str
    source_type: str
    retrieval_date: str
    licence_usage_note: str
    asset_status: str
    fallback_type: str


def canonical_party_key(value: str) -> str:
    value = (value or "").strip()
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower().replace("%", "")
    return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")


def _aliases(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in (raw or "").split(";") if item.strip())


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> list[PartyAsset]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            PartyAsset(
                party_key=(row.get("party_key") or "").strip(),
                party_name=(row.get("party_name") or "").strip(),
                party_aliases=_aliases(row.get("party_aliases") or ""),
                logo_s3_uri=(row.get("logo_s3_uri") or "").strip(),
                source_url=(row.get("source_url") or "").strip(),
                source_type=(row.get("source_type") or "").strip(),
                retrieval_date=(row.get("retrieval_date") or "").strip(),
                licence_usage_note=(row.get("licence_usage_note") or "").strip(),
                asset_status=(row.get("asset_status") or "").strip(),
                fallback_type=(row.get("fallback_type") or "").strip(),
            )
            for row in reader
        ]


def build_alias_index(rows: Iterable[PartyAsset]) -> dict[str, PartyAsset]:
    index: dict[str, PartyAsset] = {}
    for row in rows:
        values = {row.party_name, row.party_key, *row.party_aliases}
        for value in values:
            if not value:
                continue
            normalized = canonical_party_key(value)
            existing = index.get(normalized)
            if existing and existing.party_key != row.party_key:
                raise ValueError(f"Party alias collision: {value!r} -> {existing.party_key} / {row.party_key}")
            index[normalized] = row
    return index


def resolve_party(value: str, rows: Iterable[PartyAsset] | None = None) -> PartyAsset | None:
    registry = list(rows) if rows is not None else load_registry()
    return build_alias_index(registry).get(canonical_party_key(value))


def validate_registry(rows: Iterable[PartyAsset]) -> list[str]:
    rows = list(rows)
    errors: list[str] = []
    seen_keys: set[str] = set()

    for row in rows:
        if not row.party_key:
            errors.append("missing party_key")
            continue
        if row.party_key in seen_keys:
            errors.append(f"duplicate party_key: {row.party_key}")
        seen_keys.add(row.party_key)

        if row.asset_status not in VALID_STATUSES:
            errors.append(f"{row.party_key}: invalid asset_status {row.asset_status!r}")
        if not row.party_name:
            errors.append(f"{row.party_key}: missing party_name")
        if not row.retrieval_date:
            errors.append(f"{row.party_key}: missing retrieval_date")
        if not row.licence_usage_note:
            errors.append(f"{row.party_key}: missing licence_usage_note")

        if row.asset_status == "approved_fallback":
            if not row.fallback_type:
                errors.append(f"{row.party_key}: approved_fallback requires fallback_type")
            continue

        if not row.logo_s3_uri.startswith("s3://"):
            errors.append(f"{row.party_key}: non-fallback row requires logo_s3_uri")
        if not row.source_type:
            errors.append(f"{row.party_key}: non-fallback row requires source_type")
        elif row.source_type in GENERATED_SOURCE_TYPES:
            if row.source_url:
                errors.append(f"{row.party_key}: generated stand-in must not claim an external source_url")
            if not row.fallback_type:
                errors.append(f"{row.party_key}: generated stand-in requires fallback_type")
        elif not row.source_url.startswith("https://"):
            errors.append(f"{row.party_key}: external-source row requires HTTPS source_url")

    try:
        build_alias_index(rows)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def list_all_s3_keys(bucket: str, client=None) -> Iterable[str]:
    client = client or boto3.client("s3", region_name=os.getenv("AWS_REGION", "ca-central-1"))
    token = None
    while True:
        kwargs = {"Bucket": bucket, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []) or []:
            yield obj["Key"]
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")


def audit_s3(rows: Iterable[PartyAsset], bucket: str = DEFAULT_BUCKET, client=None) -> dict:
    rows = list(rows)
    all_keys = list(list_all_s3_keys(bucket, client=client))
    image_keys = [key for key in all_keys if Path(key.lower()).suffix in IMAGE_EXTENSIONS]
    party_tokens = {
        row.party_key: {canonical_party_key(row.party_name), row.party_key, *(canonical_party_key(a) for a in row.party_aliases)}
        for row in rows
    }
    candidates: dict[str, list[str]] = {row.party_key: [] for row in rows}
    for key in image_keys:
        key_norm = canonical_party_key(key)
        for party_key, tokens in party_tokens.items():
            if any(token and token in key_norm for token in tokens):
                candidates[party_key].append(key)

    expected: dict[str, dict] = {}
    key_set = set(all_keys)
    for row in rows:
        if row.logo_s3_uri:
            expected_bucket, expected_key = parse_s3_uri(row.logo_s3_uri)
            expected[row.party_key] = {
                "uri": row.logo_s3_uri,
                "bucket_matches": expected_bucket == bucket,
                "exists": expected_bucket == bucket and expected_key in key_set,
            }
        else:
            expected[row.party_key] = {"uri": "", "bucket_matches": True, "exists": False}
    return {
        "bucket": bucket,
        "object_count": len(all_keys),
        "image_object_count": len(image_keys),
        "expected_assets": expected,
        "existing_candidates": candidates,
    }


def _write_report(report: dict, output_path: str | None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or audit the shared party asset registry")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--audit-s3", action="store_true")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = load_registry(args.registry)
    errors = validate_registry(rows)
    report = {
        "registry": str(args.registry),
        "party_count": len(rows),
        "validation_errors": errors,
        "parties": [
            {"party_key": row.party_key, "party_name": row.party_name, "asset_status": row.asset_status, "logo_s3_uri": row.logo_s3_uri}
            for row in rows
        ],
    }
    audit_failed = False
    if args.audit_s3:
        try:
            report["s3_audit"] = audit_s3(rows, bucket=args.bucket)
        except Exception as exc:
            audit_failed = True
            report["s3_audit_error"] = {"type": type(exc).__name__, "message": str(exc)}
    _write_report(report, args.output)
    return 1 if errors or audit_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
