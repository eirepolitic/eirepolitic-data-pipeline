from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


RESULT_COLUMNS = [
    "metric_id",
    "metric_version",
    "period_type",
    "period_start",
    "period_end",
    "grain",
    "entity_id",
    "entity_name",
    "dimension_name",
    "dimension_value",
    "value",
    "numerator",
    "denominator",
    "output_unit",
    "reliability_status",
    "public_use_status",
    "warning_code",
    "source_batch_id",
    "calculated_at_utc",
    "contract_version",
]


def metric_result_row(
    *,
    metric_id: str,
    metric_version: int,
    period_start: str,
    period_end: str,
    grain: str,
    entity_id: str,
    entity_name: str,
    value: float | int | None,
    numerator: float | int | None,
    denominator: float | int | None,
    output_unit: str,
    source_batch_id: str,
    contract_version: int,
    dimension_name: str = "none",
    dimension_value: str = "none",
    reliability_status: str = "not_applicable",
    public_use_status: str = "suitable_with_context",
    warning_code: str = "none",
    calculated_at_utc: str | None = None,
) -> dict:
    if not metric_id.strip():
        raise ValueError("metric_id is required")
    if not grain.strip():
        raise ValueError("grain is required")
    if not entity_id.strip():
        raise ValueError("entity_id is required")
    if not dimension_name.strip() or not dimension_value.strip():
        raise ValueError("dimension_name and dimension_value must be explicit; use 'none' when not applicable")

    return {
        "metric_id": metric_id,
        "metric_version": int(metric_version),
        "period_type": "calendar_month",
        "period_start": period_start,
        "period_end": period_end,
        "grain": grain,
        "entity_id": entity_id,
        "entity_name": entity_name or entity_id,
        "dimension_name": dimension_name,
        "dimension_value": dimension_value,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "output_unit": output_unit,
        "reliability_status": reliability_status,
        "public_use_status": public_use_status,
        "warning_code": warning_code,
        "source_batch_id": source_batch_id,
        "calculated_at_utc": calculated_at_utc or datetime.now(timezone.utc).isoformat(),
        "contract_version": int(contract_version),
    }


def metric_results_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    missing = [col for col in RESULT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"metric result rows missing columns: {missing}")
    return frame[RESULT_COLUMNS].copy()


def append_metric_rows(
    rows: list[dict],
    frame: pd.DataFrame,
    *,
    metric_id: str,
    metric_version: int,
    value_col: str,
    numerator_col: str | None,
    denominator_col: str | None,
    grain: str,
    entity_id_col: str,
    entity_name_col: str,
    period_start: str,
    period_end: str,
    output_unit: str,
    source_batch_id: str,
    contract_version: int,
    dimension_name: str = "none",
    dimension_value_col: str | None = None,
    reliability_col: str | None = None,
    public_use_col: str | None = None,
    warning_col: str | None = None,
) -> None:
    """Append one metric row per dataframe row using a shared consumer schema."""
    for record in frame.to_dict(orient="records"):
        dimension_value = "none" if dimension_value_col is None else str(record.get(dimension_value_col) or "none")
        rows.append(
            metric_result_row(
                metric_id=metric_id,
                metric_version=metric_version,
                period_start=period_start,
                period_end=period_end,
                grain=grain,
                entity_id=str(record.get(entity_id_col) or ""),
                entity_name=str(record.get(entity_name_col) or record.get(entity_id_col) or ""),
                dimension_name=dimension_name,
                dimension_value=dimension_value,
                value=record.get(value_col),
                numerator=record.get(numerator_col) if numerator_col else None,
                denominator=record.get(denominator_col) if denominator_col else None,
                output_unit=output_unit,
                source_batch_id=source_batch_id,
                contract_version=contract_version,
                reliability_status=str(record.get(reliability_col) or "not_applicable") if reliability_col else "not_applicable",
                public_use_status=str(record.get(public_use_col) or "suitable_with_context") if public_use_col else "suitable_with_context",
                warning_code=str(record.get(warning_col) or "none") if warning_col else "none",
            )
        )
