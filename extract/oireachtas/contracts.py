"""Downstream schema, freshness, and batch-consistency contracts."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .io_s3 import get_bytes, resolve_read_key


DEFAULT_CONTRACTS_PATH = Path("configs/oireachtas/downstream_contracts.yml")


@dataclass(frozen=True)
class DatasetContract:
    name: str
    logical_key: str
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    minimum_rows: int
    maximum_age_days: int


@dataclass(frozen=True)
class ComparisonThreshold:
    name: str
    max_legacy_only_keys: int
    max_compat_only_keys: int
    max_row_delta_pct: float
    minimum_compat_join_coverage_pct: float


def load_contract_config(path: Path | str = DEFAULT_CONTRACTS_PATH) -> tuple[dict[str, DatasetContract], dict[str, ComparisonThreshold]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    datasets: dict[str, DatasetContract] = {}
    for name, raw in (payload.get("datasets") or {}).items():
        datasets[name] = DatasetContract(
            name=name,
            logical_key=str(raw["logical_key"]),
            required_columns=tuple(str(value) for value in raw.get("required_columns") or []),
            primary_key=tuple(str(value) for value in raw.get("primary_key") or []),
            minimum_rows=int(raw.get("minimum_rows", 1)),
            maximum_age_days=int(raw.get("maximum_age_days", 30)),
        )
    thresholds: dict[str, ComparisonThreshold] = {}
    for name, raw in (payload.get("comparison_thresholds") or {}).items():
        thresholds[name] = ComparisonThreshold(
            name=name,
            max_legacy_only_keys=int(raw.get("max_legacy_only_keys", 0)),
            max_compat_only_keys=int(raw.get("max_compat_only_keys", 0)),
            max_row_delta_pct=float(raw.get("max_row_delta_pct", 0.0)),
            minimum_compat_join_coverage_pct=float(raw.get("minimum_compat_join_coverage_pct", 100.0)),
        )
    return datasets, thresholds


def validate_dataset_contract(
    s3: Any,
    *,
    bucket: str,
    contract: DatasetContract,
    as_of: date | None = None,
) -> dict[str, Any]:
    as_of = as_of or datetime.now(timezone.utc).date()
    resolved_key = resolve_read_key(s3, bucket=bucket, key=contract.logical_key)
    try:
        body = get_bytes(s3, bucket=bucket, key=contract.logical_key)
        df = pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)
        head = s3.head_object(Bucket=bucket, Key=resolved_key)
    except Exception as exc:
        return {
            "dataset": contract.name,
            "status": "fail",
            "logical_key": contract.logical_key,
            "resolved_key": resolved_key,
            "error": f"{type(exc).__name__}: {exc}",
        }

    missing_columns = sorted(set(contract.required_columns) - set(df.columns))
    row_count = int(len(df))
    duplicate_count = 0
    blank_pk_rows = 0
    if contract.primary_key and not missing_columns:
        duplicate_count = int(df.duplicated(subset=list(contract.primary_key), keep=False).sum())
        blank_mask = pd.Series(False, index=df.index)
        for column in contract.primary_key:
            blank_mask = blank_mask | df[column].fillna("").astype(str).str.strip().eq("")
        blank_pk_rows = int(blank_mask.sum())

    last_modified = head.get("LastModified")
    if last_modified is not None:
        modified_date = last_modified.astimezone(timezone.utc).date()
        age_days = (as_of - modified_date).days
    else:
        modified_date = None
        age_days = None
    fresh = age_days is not None and age_days <= contract.maximum_age_days
    errors: list[str] = []
    if row_count < contract.minimum_rows:
        errors.append(f"row_count {row_count} below minimum {contract.minimum_rows}")
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
    if duplicate_count:
        errors.append(f"duplicate primary-key rows: {duplicate_count}")
    if blank_pk_rows:
        errors.append(f"blank primary-key rows: {blank_pk_rows}")
    if not fresh:
        errors.append(f"dataset age {age_days!r} exceeds maximum {contract.maximum_age_days}")

    return {
        "dataset": contract.name,
        "status": "fail" if errors else "pass",
        "logical_key": contract.logical_key,
        "resolved_key": resolved_key,
        "row_count": row_count,
        "columns": list(df.columns),
        "missing_columns": missing_columns,
        "duplicate_primary_key_rows": duplicate_count,
        "blank_primary_key_rows": blank_pk_rows,
        "last_modified_utc": last_modified.astimezone(timezone.utc).isoformat() if last_modified else None,
        "age_days": age_days,
        "maximum_age_days": contract.maximum_age_days,
        "errors": errors,
    }


def validate_contract_set(
    s3: Any,
    *,
    bucket: str,
    names: list[str] | None = None,
    contracts_path: Path | str = DEFAULT_CONTRACTS_PATH,
    as_of: date | None = None,
) -> dict[str, Any]:
    contracts, _ = load_contract_config(contracts_path)
    selected = names or sorted(contracts)
    unknown = sorted(set(selected) - set(contracts))
    if unknown:
        raise KeyError(f"Unknown downstream contracts: {unknown}")
    results = [validate_dataset_contract(s3, bucket=bucket, contract=contracts[name], as_of=as_of) for name in selected]
    return {
        "status": "pass" if results and all(result["status"] == "pass" for result in results) else "fail",
        "dataset_count": len(results),
        "results": results,
    }


def comparison_status(row: Mapping[str, Any], threshold: ComparisonThreshold) -> tuple[str, list[str]]:
    errors: list[str] = []
    legacy_only = int(row.get("legacy_only_key_count") or 0)
    compat_only = int(row.get("compat_only_key_count") or 0)
    legacy_rows = int(row.get("legacy_rows") or 0)
    compat_rows = int(row.get("compat_rows") or 0)
    coverage = float(row.get("compat_join_coverage_pct") or 0.0)
    denominator = max(legacy_rows, 1)
    row_delta_pct = abs(compat_rows - legacy_rows) / denominator * 100.0
    if compat_rows <= 0:
        errors.append("compat output is empty")
    if legacy_only > threshold.max_legacy_only_keys:
        errors.append(f"legacy-only keys {legacy_only} exceed {threshold.max_legacy_only_keys}")
    if compat_only > threshold.max_compat_only_keys:
        errors.append(f"compat-only keys {compat_only} exceed {threshold.max_compat_only_keys}")
    if row_delta_pct > threshold.max_row_delta_pct:
        errors.append(f"row delta {row_delta_pct:.2f}% exceeds {threshold.max_row_delta_pct:.2f}%")
    if coverage < threshold.minimum_compat_join_coverage_pct:
        errors.append(f"compat join coverage {coverage:.2f}% below {threshold.minimum_compat_join_coverage_pct:.2f}%")
    return ("fail" if errors else "pass", errors)
