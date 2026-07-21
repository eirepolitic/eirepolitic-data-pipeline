from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any


def _display_rows(record: dict[str, Any], rows_field: str, max_items: int) -> list[dict[str, Any]]:
    return [deepcopy(row) for row in record.get(rows_field, [])[:max_items]]


def _complexity(record: dict[str, Any], *, label_field: str, rows_field: str, total_field: str, max_items: int) -> int:
    rows = _display_rows(record, rows_field, max_items)
    longest_row_label = max((len(str(row.get("label", ""))) for row in rows), default=0)
    return len(str(record.get(label_field, ""))) + len(rows) * 12 + longest_row_label + min(int(record.get(total_field, 0)), 100)


def _real_scenario(
    record: dict[str, Any],
    *,
    scenario: str,
    reason: str,
    key_field: str,
    label_field: str,
    rows_field: str,
    total_field: str,
    max_items: int,
) -> dict[str, Any]:
    rows = _display_rows(record, rows_field, max_items)
    selected = deepcopy(record)
    selected[rows_field] = rows
    selected["issue_count"] = len(rows)
    selected[total_field] = sum(int(row.get("value", 0)) for row in rows)
    selected.update({
        "scenario": scenario,
        "synthetic": False,
        "no_publication": True,
        "data_origin": "current_real",
        "selection_reason": reason,
        "source_item_key": str(record.get(key_field, "")),
        "source_item_label": str(record.get(label_field, "")),
    })
    return selected


def select_real_category_value_scenarios(
    records: list[dict[str, Any]],
    *,
    key_field: str,
    label_field: str,
    rows_field: str = "issue_rows",
    total_field: str = "speech_count",
    max_items: int = 7,
) -> dict[str, dict[str, Any]]:
    """Select factual minimum, maximum, and representative records for category/value visuals.

    Minimum and maximum are based first on the number of categories that will actually
    be displayed, then total value and label length. Synthetic data is deliberately not
    created here; it belongs to a separate contract-edge fallback layer.
    """
    if not records:
        raise ValueError("Cannot select validation scenarios from an empty record set")

    def displayed_count(record: dict[str, Any]) -> int:
        return len(record.get(rows_field, [])[:max_items])

    minimum = min(
        records,
        key=lambda row: (
            displayed_count(row),
            int(row.get(total_field, 0)),
            len(str(row.get(label_field, ""))),
            str(row.get(label_field, "")),
        ),
    )
    maximum = max(
        records,
        key=lambda row: (
            displayed_count(row),
            int(row.get(total_field, 0)),
            len(str(row.get(label_field, ""))),
            str(row.get(label_field, "")),
        ),
    )

    complexity_values = sorted(
        _complexity(
            row,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        )
        for row in records
    )
    target = median(complexity_values)
    representative = min(
        records,
        key=lambda row: (
            abs(
                _complexity(
                    row,
                    label_field=label_field,
                    rows_field=rows_field,
                    total_field=total_field,
                    max_items=max_items,
                )
                - target
            ),
            str(row.get(label_field, "")),
        ),
    )

    return {
        "minimum": _real_scenario(
            minimum,
            scenario="minimum",
            reason="Current real record with the fewest displayed categories; ties use the smallest total value and shortest item label.",
            key_field=key_field,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        ),
        "maximum": _real_scenario(
            maximum,
            scenario="maximum",
            reason="Current real record with the most displayed categories; ties use the largest total value and longest item label.",
            key_field=key_field,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        ),
        "real_example": _real_scenario(
            representative,
            scenario="real_example",
            reason="Current real record nearest the median combined layout and visual complexity.",
            key_field=key_field,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        ),
    }
