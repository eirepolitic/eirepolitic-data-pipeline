"""Production ingestion for the Irish Polling Indicator datasets.

The source repository publishes two CSVs:
- data_polls.csv: individual polls, values in percentage points (0-100)
- data_pollingindicator.csv: modelled estimates and intervals, values as proportions (0-1)

Publication is deliberately separated from validation. Running the module without
``--publish`` fetches and validates the current upstream source but never touches S3.
Publishing is additionally gated by ``IPI_REUSE_CONFIRMED=true`` because the
upstream repository does not currently expose a clearly identifiable open licence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

UPSTREAM_OWNER = "Irish-Polling-Indicator"
UPSTREAM_REPO = "ipi-data"
UPSTREAM_REPO_URL = f"https://github.com/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"
UPSTREAM_API_URL = f"https://api.github.com/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"
SOURCE_FILES = ("data_polls.csv", "data_pollingindicator.csv")
SOURCE_CITATION = "Irish Polling Indicator (IPI); cite the source authors/repository when using the data."
SOURCE_LICENSE_STATUS = "unconfirmed"

DEFAULT_BUCKET = "eirepolitic-data"
DEFAULT_REGION = "ca-central-1"
DEFAULT_RAW_PREFIX = "raw/polling/irish_polling_indicator"
DEFAULT_PROCESSED_PREFIX = "processed/polling/irish_polling_indicator"

RAW_POLL_COLUMNS = (
    "date",
    "date_start",
    "date_end",
    "date_middle",
    "pollster",
    "sample_size",
    "FF",
    "FG",
    "SF",
    "LAB",
    "GP",
    "PD",
    "WP",
    "DL",
    "SPBP",
    "RENUA",
    "SD",
    "AU",
    "II",
    "IND_OTH_IT",
    "PREV_INDOTH_II",
    "PREV_II",
    "OTH_IND",
)
RAW_POLL_PARTY_COLUMNS = RAW_POLL_COLUMNS[6:]
RAW_DATE_COLUMNS = ("date", "date_start", "date_end", "date_middle")
RAW_CORE_COLUMNS = ("date", "date_start", "date_end", "date_middle", "pollster", "sample_size")
RAW_COMPOSITE_KEY = ("date", "date_start", "date_end", "pollster", "sample_size")

INDICATOR_PARTIES = ("FF", "FG", "LAB", "PD", "WP", "OTH", "GP", "DL", "SF", "SD", "SPBP", "AU", "II")
INDICATOR_COLUMNS = (
    "date",
    "cycle",
    *tuple(column for party in INDICATOR_PARTIES for column in (party, f"{party}_lo", f"{party}_hi")),
)

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class UpstreamSnapshot:
    branch: str
    commit_sha: str
    retrieved_at: str
    files: Mapping[str, bytes]
    urls: Mapping[str, str]


@dataclass(frozen=True)
class PreparedIngestion:
    snapshot: UpstreamSnapshot
    polls: pd.DataFrame
    polling_indicator: pd.DataFrame
    artifacts: Mapping[str, bytes]
    manifest: Mapping[str, Any]


class ValidationError(RuntimeError):
    """Raised when a source change makes publication unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "EirePolitic-data-pipeline/ipi-ingestion",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def safe_get(
    url: str,
    *,
    timeout: int = 45,
    retries: int = 4,
    backoff: float = 2.0,
    headers: Mapping[str, str] | None = None,
    session: Any = requests,
) -> Any:
    """HTTP GET with bounded retry/backoff and 429 handling."""
    merged_headers = {"User-Agent": "EirePolitic-data-pipeline/ipi-ingestion"}
    if headers:
        merged_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, headers=merged_headers)
            if response.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:  # requests exposes several exception subclasses
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError(f"GET failed for {url}: {last_error}")


def resolve_upstream_snapshot(*, ref: str | None = None, session: Any = requests) -> UpstreamSnapshot:
    """Resolve an upstream ref to an immutable commit and fetch both CSVs."""
    repo_response = safe_get(UPSTREAM_API_URL, headers=_github_headers(), session=session)
    repo_metadata = repo_response.json()
    branch = (ref or repo_metadata.get("default_branch") or "main").strip()

    commit_url = f"{UPSTREAM_API_URL}/commits/{quote(branch, safe='')}"
    commit_response = safe_get(commit_url, headers=_github_headers(), session=session)
    commit_sha = str(commit_response.json().get("sha") or "").strip()
    if len(commit_sha) != 40:
        raise ValidationError(f"Could not resolve a 40-character upstream commit SHA for ref {branch!r}")

    files: dict[str, bytes] = {}
    urls: dict[str, str] = {}
    for filename in SOURCE_FILES:
        url = f"{RAW_BASE_URL}/{commit_sha}/{filename}"
        response = safe_get(url, session=session)
        files[filename] = response.content
        urls[filename] = url
        if not response.content:
            raise ValidationError(f"Upstream file is empty: {filename}")

    return UpstreamSnapshot(
        branch=branch,
        commit_sha=commit_sha,
        retrieved_at=_utc_now(),
        files=files,
        urls=urls,
    )


def _read_csv(body: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=True)


def _require_exact_columns(frame: pd.DataFrame, expected: Iterable[str], *, source_name: str) -> None:
    actual = tuple(str(column) for column in frame.columns)
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        raise ValidationError(
            f"{source_name} schema changed. Expected {list(expected_tuple)!r}; got {list(actual)!r}"
        )


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _strict_dates(frame: pd.DataFrame, columns: Iterable[str], *, source_name: str) -> dict[str, pd.Series]:
    parsed: dict[str, pd.Series] = {}
    for column in columns:
        missing = _missing_mask(frame[column])
        if missing.any():
            raise ValidationError(f"{source_name}.{column} contains {int(missing.sum())} missing values")
        values = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="coerce")
        invalid = values.isna()
        if invalid.any():
            bad = frame.loc[invalid, column].head(5).tolist()
            raise ValidationError(f"{source_name}.{column} contains invalid YYYY-MM-DD values: {bad!r}")
        parsed[column] = values
    return parsed


def _mark_flags(flags: list[set[str]], mask: pd.Series, flag: str) -> None:
    for position, is_match in enumerate(mask.fillna(False).tolist()):
        if bool(is_match):
            flags[position].add(flag)


def validate_and_normalize_polls(
    body: bytes,
    *,
    commit_sha: str,
    branch: str,
    source_url: str,
    retrieved_at: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv(body)
    _require_exact_columns(frame, RAW_POLL_COLUMNS, source_name="data_polls.csv")
    if frame.empty:
        raise ValidationError("data_polls.csv contains no rows")

    for column in RAW_CORE_COLUMNS:
        missing = _missing_mask(frame[column])
        if missing.any():
            raise ValidationError(f"data_polls.csv.{column} contains {int(missing.sum())} missing values")

    parsed_dates = _strict_dates(frame, RAW_DATE_COLUMNS, source_name="data_polls.csv")

    sample_size = pd.to_numeric(frame["sample_size"], errors="coerce")
    if sample_size.isna().any():
        raise ValidationError("data_polls.csv.sample_size contains non-numeric values")
    if (sample_size <= 0).any():
        raise ValidationError("data_polls.csv.sample_size contains non-positive values")
    if ((sample_size % 1) != 0).any():
        raise ValidationError("data_polls.csv.sample_size contains non-integer values")

    numeric_parties: dict[str, pd.Series] = {}
    negative_party_cells = 0
    for column in RAW_POLL_PARTY_COLUMNS:
        raw = frame[column]
        nonblank = ~_missing_mask(raw)
        numeric = pd.to_numeric(raw, errors="coerce")
        bad_numeric = nonblank & numeric.isna()
        if bad_numeric.any():
            bad = raw.loc[bad_numeric].head(5).tolist()
            raise ValidationError(f"data_polls.csv.{column} contains non-numeric poll values: {bad!r}")
        if (numeric.dropna() > 100).any():
            raise ValidationError(f"data_polls.csv.{column} contains values above 100")
        if (numeric.dropna() < -1).any():
            raise ValidationError(f"data_polls.csv.{column} contains values below the known historical floor of -1")
        negative_party_cells += int((numeric < 0).fillna(False).sum())
        numeric_parties[column] = numeric.astype("Float64")

    flags: list[set[str]] = [set() for _ in range(len(frame))]
    exact_duplicates = frame.duplicated(keep=False)
    composite_duplicates = frame.duplicated(subset=list(RAW_COMPOSITE_KEY), keep=False)
    start_after_end = parsed_dates["date_start"] > parsed_dates["date_end"]
    middle_outside = (parsed_dates["date_middle"] < parsed_dates["date_start"]) | (
        parsed_dates["date_middle"] > parsed_dates["date_end"]
    )
    end_after_publication = parsed_dates["date_end"] > parsed_dates["date"]

    _mark_flags(flags, exact_duplicates, "exact_duplicate_source_row")
    _mark_flags(flags, composite_duplicates, "duplicate_poll_composite_key")
    _mark_flags(flags, start_after_end, "fieldwork_start_after_end")
    _mark_flags(flags, middle_outside, "fieldwork_middle_outside_range")
    _mark_flags(flags, end_after_publication, "fieldwork_end_after_publication")

    for column, numeric in numeric_parties.items():
        _mark_flags(flags, numeric < 0, f"negative_source_value:{column}")

    normalized = frame.copy()
    for column, parsed in parsed_dates.items():
        normalized[column] = parsed.dt.strftime("%Y-%m-%d")
    normalized["sample_size"] = sample_size.astype("Int64")
    for column, numeric in numeric_parties.items():
        normalized[column] = numeric

    normalized.insert(0, "source_row_number", pd.Series(range(2, len(normalized) + 2), dtype="Int64"))
    normalized["value_unit"] = "percentage_points"
    normalized["quality_flags"] = [";".join(sorted(row_flags)) for row_flags in flags]
    normalized["source_commit_sha"] = commit_sha
    normalized["source_branch"] = branch
    normalized["source_file"] = "data_polls.csv"
    normalized["source_url"] = source_url
    normalized["source_retrieved_at"] = retrieved_at

    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "exact_duplicate_extra_rows": int(len(frame) - len(frame.drop_duplicates())),
        "exact_duplicate_rows": int(exact_duplicates.sum()),
        "composite_duplicate_extra_rows": int(len(frame) - len(frame.drop_duplicates(subset=list(RAW_COMPOSITE_KEY)))),
        "composite_duplicate_rows": int(composite_duplicates.sum()),
        "fieldwork_start_after_end_rows": int(start_after_end.sum()),
        "fieldwork_middle_outside_range_rows": int(middle_outside.sum()),
        "fieldwork_end_after_publication_rows": int(end_after_publication.sum()),
        "negative_party_cells": int(negative_party_cells),
        "quality_flagged_rows": int(sum(bool(row_flags) for row_flags in flags)),
        "unit": "percentage_points",
    }
    return normalized, summary


def validate_and_normalize_indicator(
    body: bytes,
    *,
    commit_sha: str,
    branch: str,
    source_url: str,
    retrieved_at: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv(body)
    _require_exact_columns(frame, INDICATOR_COLUMNS, source_name="data_pollingindicator.csv")
    if frame.empty:
        raise ValidationError("data_pollingindicator.csv contains no rows")

    if _missing_mask(frame["cycle"]).any():
        raise ValidationError("data_pollingindicator.csv.cycle contains missing values")
    parsed_date = _strict_dates(frame, ("date",), source_name="data_pollingindicator.csv")["date"]

    duplicate_key = frame.duplicated(subset=["date", "cycle"], keep=False)
    if duplicate_key.any():
        sample = frame.loc[duplicate_key, ["date", "cycle"]].head(10).to_dict("records")
        raise ValidationError(f"data_pollingindicator.csv has duplicate (date, cycle) keys: {sample!r}")

    normalized = frame.copy()
    normalized["date"] = parsed_date.dt.strftime("%Y-%m-%d")
    numeric_columns: dict[str, pd.Series] = {}
    interval_rows = 0

    for party in INDICATOR_PARTIES:
        columns = (party, f"{party}_lo", f"{party}_hi")
        values = []
        for column in columns:
            raw = frame[column]
            nonblank = ~_missing_mask(raw)
            numeric = pd.to_numeric(raw, errors="coerce")
            bad_numeric = nonblank & numeric.isna()
            if bad_numeric.any():
                bad = raw.loc[bad_numeric].head(5).tolist()
                raise ValidationError(f"data_pollingindicator.csv.{column} contains non-numeric values: {bad!r}")
            numeric_columns[column] = numeric.astype("Float64")
            values.append(numeric)

        estimate, lower, upper = values
        present_count = pd.concat([estimate.notna(), lower.notna(), upper.notna()], axis=1).sum(axis=1)
        partial = ~present_count.isin([0, 3])
        if partial.any():
            raise ValidationError(f"data_pollingindicator.csv has partial estimate intervals for {party}")

        present = present_count.eq(3)
        invalid = present & (
            (lower < 0)
            | (estimate < 0)
            | (upper < 0)
            | (lower > estimate)
            | (estimate > upper)
            | (upper > 1)
        )
        if invalid.any():
            sample = frame.loc[invalid, ["date", "cycle", *columns]].head(10).to_dict("records")
            raise ValidationError(f"data_pollingindicator.csv has invalid {party} intervals: {sample!r}")
        interval_rows += int(present.sum())

    for column, numeric in numeric_columns.items():
        normalized[column] = numeric

    duplicate_calendar_date = frame["date"].duplicated(keep=False)
    flags: list[set[str]] = [set() for _ in range(len(frame))]
    _mark_flags(flags, duplicate_calendar_date, "cycle_boundary_duplicate_calendar_date")

    normalized.insert(0, "source_row_number", pd.Series(range(2, len(normalized) + 2), dtype="Int64"))
    normalized["value_unit"] = "proportion"
    normalized["quality_flags"] = [";".join(sorted(row_flags)) for row_flags in flags]
    normalized["source_commit_sha"] = commit_sha
    normalized["source_branch"] = branch
    normalized["source_file"] = "data_pollingindicator.csv"
    normalized["source_url"] = source_url
    normalized["source_retrieved_at"] = retrieved_at

    duplicate_dates = sorted(frame.loc[duplicate_calendar_date, "date"].drop_duplicates().tolist())
    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "unique_date_cycle_key": True,
        "duplicate_calendar_dates": duplicate_dates,
        "duplicate_calendar_date_count": int(len(duplicate_dates)),
        "validated_non_null_party_intervals": int(interval_rows),
        "quality_flagged_rows": int(duplicate_calendar_date.sum()),
        "unit": "proportion",
    }
    return normalized, summary


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    return buffer.getvalue()


def prepare_ingestion(*, ref: str | None = None, session: Any = requests) -> PreparedIngestion:
    snapshot = resolve_upstream_snapshot(ref=ref, session=session)
    polls, polls_summary = validate_and_normalize_polls(
        snapshot.files["data_polls.csv"],
        commit_sha=snapshot.commit_sha,
        branch=snapshot.branch,
        source_url=snapshot.urls["data_polls.csv"],
        retrieved_at=snapshot.retrieved_at,
    )
    indicator, indicator_summary = validate_and_normalize_indicator(
        snapshot.files["data_pollingindicator.csv"],
        commit_sha=snapshot.commit_sha,
        branch=snapshot.branch,
        source_url=snapshot.urls["data_pollingindicator.csv"],
        retrieved_at=snapshot.retrieved_at,
    )

    artifacts = {
        "polls_csv": _csv_bytes(polls),
        "polls_parquet": _parquet_bytes(polls),
        "indicator_csv": _csv_bytes(indicator),
        "indicator_parquet": _parquet_bytes(indicator),
    }

    manifest: dict[str, Any] = {
        "dataset": "irish_polling_indicator",
        "source_repository": UPSTREAM_REPO_URL,
        "source_branch_or_ref": snapshot.branch,
        "source_commit_sha": snapshot.commit_sha,
        "source_retrieved_at": snapshot.retrieved_at,
        "source_license_status": SOURCE_LICENSE_STATUS,
        "source_citation": SOURCE_CITATION,
        "source_files": {
            filename: {
                "url": snapshot.urls[filename],
                "sha256": _sha256(snapshot.files[filename]),
                "bytes": len(snapshot.files[filename]),
            }
            for filename in SOURCE_FILES
        },
        "tables": {
            "polls": {
                **polls_summary,
                "csv_sha256": _sha256(artifacts["polls_csv"]),
                "parquet_sha256": _sha256(artifacts["polls_parquet"]),
            },
            "polling_indicator": {
                **indicator_summary,
                "csv_sha256": _sha256(artifacts["indicator_csv"]),
                "parquet_sha256": _sha256(artifacts["indicator_parquet"]),
            },
        },
    }
    return PreparedIngestion(
        snapshot=snapshot,
        polls=polls,
        polling_indicator=indicator,
        artifacts=artifacts,
        manifest=manifest,
    )


def _put(s3: Any, *, bucket: str, key: str, body: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def publish_ingestion(
    prepared: PreparedIngestion,
    *,
    s3: Any,
    bucket: str = DEFAULT_BUCKET,
    raw_prefix: str = DEFAULT_RAW_PREFIX,
    processed_prefix: str = DEFAULT_PROCESSED_PREFIX,
) -> dict[str, str]:
    """Publish immutable commit artifacts first, then stable latest paths, manifest last."""
    sha = prepared.snapshot.commit_sha
    raw_commit_prefix = f"{raw_prefix}/by_commit/{sha}"
    processed_commit_prefix = f"{processed_prefix}/by_commit/{sha}"
    latest_prefix = f"{processed_prefix}/latest"

    keys = {
        "raw_polls": f"{raw_commit_prefix}/data_polls.csv",
        "raw_indicator": f"{raw_commit_prefix}/data_pollingindicator.csv",
        "commit_polls_csv": f"{processed_commit_prefix}/csv/polls.csv",
        "commit_polls_parquet": f"{processed_commit_prefix}/parquet/polls.parquet",
        "commit_indicator_csv": f"{processed_commit_prefix}/csv/polling_indicator.csv",
        "commit_indicator_parquet": f"{processed_commit_prefix}/parquet/polling_indicator.parquet",
        "commit_manifest": f"{processed_commit_prefix}/manifest.json",
        "latest_polls_csv": f"{latest_prefix}/csv/polls.csv",
        "latest_polls_parquet": f"{latest_prefix}/parquet/polls.parquet",
        "latest_indicator_csv": f"{latest_prefix}/csv/polling_indicator.csv",
        "latest_indicator_parquet": f"{latest_prefix}/parquet/polling_indicator.parquet",
        "latest_manifest": f"{latest_prefix}/manifest.json",
    }

    manifest = dict(prepared.manifest)
    manifest["s3"] = {
        "bucket": bucket,
        "keys": keys,
        "publication_order": "immutable artifacts, stable latest data, latest manifest last",
    }
    manifest_body = _manifest_bytes(manifest)

    _put(s3, bucket=bucket, key=keys["raw_polls"], body=prepared.snapshot.files["data_polls.csv"], content_type="text/csv")
    _put(s3, bucket=bucket, key=keys["raw_indicator"], body=prepared.snapshot.files["data_pollingindicator.csv"], content_type="text/csv")

    artifact_targets = (
        ("commit_polls_csv", "polls_csv", "text/csv"),
        ("commit_polls_parquet", "polls_parquet", "application/x-parquet"),
        ("commit_indicator_csv", "indicator_csv", "text/csv"),
        ("commit_indicator_parquet", "indicator_parquet", "application/x-parquet"),
    )
    for key_name, artifact_name, content_type in artifact_targets:
        _put(s3, bucket=bucket, key=keys[key_name], body=prepared.artifacts[artifact_name], content_type=content_type)
    _put(s3, bucket=bucket, key=keys["commit_manifest"], body=manifest_body, content_type="application/json")

    latest_targets = (
        ("latest_polls_csv", "polls_csv", "text/csv"),
        ("latest_polls_parquet", "polls_parquet", "application/x-parquet"),
        ("latest_indicator_csv", "indicator_csv", "text/csv"),
        ("latest_indicator_parquet", "indicator_parquet", "application/x-parquet"),
    )
    for key_name, artifact_name, content_type in latest_targets:
        _put(s3, bucket=bucket, key=keys[key_name], body=prepared.artifacts[artifact_name], content_type=content_type)

    # This pointer is intentionally the final write. Consumers can use it as the publication marker.
    _put(s3, bucket=bucket, key=keys["latest_manifest"], body=manifest_body, content_type="application/json")
    return keys


def _reuse_confirmed() -> bool:
    return os.getenv("IPI_REUSE_CONFIRMED", "false").strip().lower() in _TRUTHY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and optionally publish Irish Polling Indicator data")
    parser.add_argument("--publish", action="store_true", help="Publish validated artifacts to S3")
    parser.add_argument("--ref", default=os.getenv("IPI_UPSTREAM_REF") or None, help="Optional upstream branch/tag/SHA")
    parser.add_argument("--bucket", default=os.getenv("POLLING_S3_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prepared = prepare_ingestion(ref=args.ref)

    print(json.dumps(prepared.manifest, indent=2, sort_keys=True, ensure_ascii=False))
    if not args.publish:
        print("Validation succeeded. No S3 writes requested.")
        return 0

    if not _reuse_confirmed():
        raise RuntimeError(
            "IPI publication is blocked because reuse rights are not confirmed. "
            "Set repository variable IPI_REUSE_CONFIRMED=true only after the source licence/permission is documented."
        )

    s3 = boto3.client("s3", region_name=args.region)
    keys = publish_ingestion(prepared, s3=s3, bucket=args.bucket)
    print(f"Published source commit {prepared.snapshot.commit_sha} to s3://{args.bucket}/")
    print(json.dumps(keys, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
