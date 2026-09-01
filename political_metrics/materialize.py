from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class DatasetContract:
    name: str
    columns: list[str]
    primary_key: list[str]
    formats: list[str]
    cadence: str


def load_materialization_contract(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("materialization contract must be a mapping")
    return payload


def get_dataset_contract(contract: dict, dataset_name: str) -> DatasetContract:
    for section in ("foundation_datasets", "result_datasets"):
        datasets = contract.get(section) or {}
        if dataset_name in datasets:
            cfg = datasets[dataset_name]
            return DatasetContract(
                name=dataset_name,
                columns=list(cfg["columns"]),
                primary_key=list(cfg["primary_key"]),
                formats=list(cfg.get("formats", ["csv", "parquet"])),
                cadence=str(cfg["cadence"]),
            )
    raise KeyError(f"dataset not found in materialization contract: {dataset_name}")


def validate_materialized_frame(
    frame: pd.DataFrame,
    dataset: DatasetContract,
    *,
    expected_source_batch_id: str | None = None,
) -> list[str]:
    errors: list[str] = []
    missing = [col for col in dataset.columns if col not in frame.columns]
    extra = [col for col in frame.columns if col not in dataset.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    if missing:
        return errors

    if frame.duplicated(subset=dataset.primary_key).any():
        count = int(frame.duplicated(subset=dataset.primary_key).sum())
        errors.append(f"{count} duplicate rows for primary key {dataset.primary_key}")

    for col in dataset.primary_key:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            errors.append(f"primary-key column {col} contains blank/null values")

    if "source_batch_id" in frame.columns:
        batch_values = set(frame["source_batch_id"].dropna().astype(str))
        if len(batch_values) > 1:
            errors.append(f"multiple source_batch_id values present: {sorted(batch_values)}")
        if expected_source_batch_id is not None and batch_values != {expected_source_batch_id} and not frame.empty:
            errors.append(
                f"source_batch_id mismatch: expected {expected_source_batch_id!r}, found {sorted(batch_values)}"
            )

    if "value" in frame.columns:
        value_numeric = pd.to_numeric(frame["value"], errors="coerce")
        invalid = frame["value"].notna() & value_numeric.isna()
        if invalid.any():
            errors.append(f"{int(invalid.sum())} metric value rows are non-numeric")

    if {"period_start", "period_end"}.issubset(frame.columns):
        starts = pd.to_datetime(frame["period_start"], errors="coerce")
        ends = pd.to_datetime(frame["period_end"], errors="coerce")
        invalid = starts.isna() | ends.isna() | (ends < starts)
        if invalid.any():
            errors.append(f"{int(invalid.sum())} rows have invalid period_start/period_end")

    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_materialized_dataset(
    frame: pd.DataFrame,
    *,
    dataset: DatasetContract,
    output_root: str | Path,
    source_batch_id: str,
    contract_version: int,
) -> dict:
    """Write validated local commissioning artifacts for one materialized dataset.

    This function intentionally performs no S3 operations. It is safe to use in
    commissioning workflows before production batch publication is enabled.
    """
    errors = validate_materialized_frame(
        frame,
        dataset,
        expected_source_batch_id=source_batch_id,
    )
    if errors:
        raise ValueError(f"{dataset.name} validation failed: {errors}")

    root = Path(output_root) / dataset.cadence / dataset.name
    root.mkdir(parents=True, exist_ok=True)

    ordered = frame[dataset.columns].copy()
    files: list[dict] = []
    if "csv" in dataset.formats:
        path = root / f"{dataset.name}.csv"
        ordered.to_csv(path, index=False)
        files.append({"format": "csv", "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    if "parquet" in dataset.formats:
        path = root / f"{dataset.name}.parquet"
        ordered.to_parquet(path, index=False)
        files.append({"format": "parquet", "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})

    manifest = {
        "dataset": dataset.name,
        "cadence": dataset.cadence,
        "row_count": int(len(ordered)),
        "primary_key": dataset.primary_key,
        "source_batch_id": source_batch_id,
        "contract_version": contract_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
