"""Deterministic merge and integrity helpers for unified Oireachtas tables."""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

import pandas as pd

from .normalize import parse_iso_date
from .write_policies import ForeignKeyPolicy, WritePolicy


def merge_for_policy(existing: pd.DataFrame, incoming: pd.DataFrame, *, primary_key: list[str], policy: WritePolicy) -> pd.DataFrame:
    """Apply the configured table write strategy."""
    if policy.write_strategy in {"snapshot_replace", "rebuild"}:
        return incoming.reset_index(drop=True).copy()
    if not primary_key:
        raise ValueError(f"Table {policy.table} requires a primary key for {policy.write_strategy}")
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    missing = [column for column in primary_key if column not in combined.columns]
    if missing:
        raise ValueError(f"Missing primary-key columns for {policy.table}: {missing}")
    merged = combined.drop_duplicates(subset=primary_key, keep="last")
    if policy.business_key_columns:
        missing_business = [column for column in policy.business_key_columns if column not in merged.columns]
        if missing_business:
            raise ValueError(f"Missing business-key columns for {policy.table}: {missing_business}")
        merged = merged.drop_duplicates(subset=list(policy.business_key_columns), keep="last")
    return merged.reset_index(drop=True)


def temporal_integrity(df: pd.DataFrame, *, policy: WritePolicy, as_of: date | None = None) -> dict[str, object]:
    """Validate temporal bounds and future current-state flags."""
    as_of = as_of or date.today()
    start_col = policy.valid_from_column
    end_col = policy.valid_to_column
    current_col = policy.current_column
    errors: list[str] = []
    if not start_col or start_col not in df.columns:
        return {"status": "pass", "errors": [], "future_current_rows": 0, "invalid_ranges": 0}

    starts = [_normalized_iso_date(value) for value in df[start_col].tolist()]
    ends = (
        [_normalized_iso_date(value) for value in df[end_col].tolist()]
        if end_col and end_col in df.columns
        else [None] * len(df)
    )
    invalid_ranges = sum(
        1
        for start, end in zip(starts, ends)
        if start is not None and end is not None and start > end
    )

    future_current_rows = 0
    if current_col and current_col in df.columns:
        current_mask = df[current_col].fillna(False).astype(bool).tolist()
        future_current_rows = sum(
            1
            for is_current, start in zip(current_mask, starts)
            if is_current and start is not None and start > as_of.isoformat()
        )

    if invalid_ranges:
        errors.append(f"{invalid_ranges} rows have valid_from after valid_to")
    if future_current_rows:
        errors.append(f"{future_current_rows} future rows are marked current")
    return {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "future_current_rows": future_current_rows,
        "invalid_ranges": invalid_ranges,
    }


def foreign_key_integrity(child: pd.DataFrame, parent: pd.DataFrame, *, policy: ForeignKeyPolicy) -> dict[str, object]:
    """Return orphan counts for one configured foreign key."""
    missing_child = [column for column in policy.columns if column not in child.columns]
    missing_parent = [column for column in policy.referenced_columns if column not in parent.columns]
    if missing_child or missing_parent:
        return {
            "status": "fail",
            "orphan_count": None,
            "missing_child_columns": missing_child,
            "missing_parent_columns": missing_parent,
        }
    left = child[list(policy.columns)].copy()
    if policy.nullable:
        left = left.dropna(how="any")
    right = parent[list(policy.referenced_columns)].drop_duplicates().copy()
    right.columns = list(policy.columns)
    merged = left.merge(right.assign(_matched=True), how="left", on=list(policy.columns))
    orphan_count = int(merged["_matched"].isna().sum())
    return {"status": "pass" if orphan_count == 0 else "fail", "orphan_count": orphan_count}


def overlap_count(df: pd.DataFrame, *, entity_columns: Iterable[str], start_column: str, end_column: str) -> int:
    """Count overlapping date ranges within each entity."""
    entity_columns = list(entity_columns)
    if df.empty:
        return 0
    count = 0
    group_key: str | list[str] = entity_columns[0] if len(entity_columns) == 1 else entity_columns
    for _, group in df.groupby(group_key, dropna=False):
        ranges: list[tuple[str, str]] = []
        for _, row in group.iterrows():
            start = _normalized_iso_date(row.get(start_column))
            end = _normalized_iso_date(row.get(end_column)) or "9999-12-31"
            if start:
                ranges.append((start, end))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] <= previous[1]:
                count += 1
    return count


def _normalized_iso_date(value: Any) -> str | None:
    """Return an ISO date or None for pandas/JSON missing values."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return parse_iso_date(value)
