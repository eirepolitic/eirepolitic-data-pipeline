from __future__ import annotations

import pandas as pd


def validate_temporal_history(
    history: pd.DataFrame,
    *,
    entity_col: str,
    start_col: str,
    end_col: str,
    end_boundary: str = "exclusive",
) -> list[str]:
    """Return validation errors for malformed or overlapping history intervals.

    Oireachtas political-history ranges default to start-inclusive/end-exclusive
    semantics, so one row ending on the exact date the next row starts is a valid
    transition rather than an overlap.
    """
    if end_boundary not in {"exclusive", "inclusive"}:
        raise ValueError(f"unsupported end boundary: {end_boundary}")

    errors: list[str] = []
    if history.empty:
        return errors

    data = history[[entity_col, start_col, end_col]].copy()
    data[start_col] = pd.to_datetime(data[start_col], errors="coerce").dt.normalize()
    data[end_col] = pd.to_datetime(data[end_col], errors="coerce").dt.normalize()

    invalid = data[data[start_col].isna()]
    if not invalid.empty:
        errors.append(f"{len(invalid)} rows have missing/invalid {start_col}")

    if end_boundary == "exclusive":
        reversed_rows = data[data[end_col].notna() & (data[end_col] <= data[start_col])]
    else:
        reversed_rows = data[data[end_col].notna() & (data[end_col] < data[start_col])]
    if not reversed_rows.empty:
        errors.append(f"{len(reversed_rows)} rows have invalid {start_col}/{end_col} ranges")

    for entity, group in data.dropna(subset=[start_col]).groupby(entity_col, dropna=False):
        ordered = group.sort_values(start_col)
        previous_end = None
        for row in ordered.itertuples(index=False):
            start = getattr(row, start_col)
            end = getattr(row, end_col)
            if previous_end is None:
                previous_end = end
                continue
            if pd.isna(previous_end):
                errors.append(f"{entity_col}={entity!r} has an open-ended interval followed by another interval")
                break
            overlaps = start < previous_end if end_boundary == "exclusive" else start <= previous_end
            if overlaps:
                errors.append(f"{entity_col}={entity!r} has overlapping intervals")
                break
            previous_end = end

    return errors


def temporal_join_coverage(joined: pd.DataFrame, history_value_col: str) -> float:
    """Share of event rows with a successfully attributed historical dimension."""
    if joined.empty:
        return 1.0
    return float(joined[history_value_col].notna().mean())


def validate_share_total(values: pd.Series, *, tolerance: float = 1e-9) -> bool:
    """Validate that a complete set of shares sums to one within tolerance."""
    if values.empty:
        return True
    return abs(float(values.fillna(0).sum()) - 1.0) <= tolerance
