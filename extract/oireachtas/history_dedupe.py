"""Business-key deduplication helpers for Oireachtas member-history tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DedupeResult:
    rows: list[dict[str, Any]]
    duplicate_rows_removed: int
    conflicting_keys: list[dict[str, Any]]


def dedupe_history_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    business_key: Sequence[str],
    compared_fields: Sequence[str],
) -> DedupeResult:
    """Deduplicate exact business rows and retain evidence of conflicting duplicates.

    Rows sharing a business key are considered duplicates. Exact duplicates across the
    compared fields are collapsed. When compared values disagree, the first row is
    retained and the differing key/values are returned in ``conflicting_keys`` so the
    caller can fail data quality checks rather than silently choosing a value.
    """
    kept: dict[tuple[str, ...], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    input_count = 0

    for source_row in rows:
        input_count += 1
        row = dict(source_row)
        key = tuple(_normalise(row.get(column)) for column in business_key)
        existing = kept.get(key)
        if existing is None:
            kept[key] = row
            continue

        differences = {
            column: {
                "kept": _normalise(existing.get(column)),
                "duplicate": _normalise(row.get(column)),
            }
            for column in compared_fields
            if _normalise(existing.get(column)) != _normalise(row.get(column))
        }
        if differences:
            conflicts.append(
                {
                    "business_key": dict(zip(business_key, key)),
                    "differences": differences,
                }
            )

    output = list(kept.values())
    return DedupeResult(
        rows=output,
        duplicate_rows_removed=input_count - len(output),
        conflicting_keys=conflicts,
    )


def business_key_unique(rows: Iterable[Mapping[str, Any]], *, business_key: Sequence[str]) -> bool:
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(_normalise(row.get(column)) for column in business_key)
        if key in seen:
            return False
        seen.add(key)
    return True


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
