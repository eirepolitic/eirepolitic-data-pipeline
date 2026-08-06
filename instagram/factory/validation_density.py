from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any

from .validation_scenarios import select_real_category_value_scenarios

SPARSE_ALLOWED = {"item_count_min", "all_equal", "ties", "zeros"}
DENSE_PREFERRED = {
    "labels_short",
    "labels_long",
    "values_small",
    "values_large",
    "values_tight",
    "values_wide",
    "single_outlier",
    "real_example",
}


def _count(record: dict[str, Any], rows_field: str, max_items: int) -> int:
    return min(len(record.get(rows_field) or []), max_items)


def _waiver(scenario: str, minimum: int) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "waived": True,
        "waiver_reason": (
            f"No qualifying real record contains at least {minimum} displayed categories. "
            "Current and configured historical production data must be checked before this waiver is accepted."
        ),
        "synthetic": False,
        "no_publication": True,
        "data_origin": "waived_no_real_case",
        "density_requirement": minimum,
    }


def _representative_dense_record(
    records: list[dict[str, Any]],
    *,
    rows_field: str,
    total_field: str,
    label_field: str,
    max_items: int,
) -> dict[str, Any]:
    complexities = []
    for record in records:
        rows = (record.get(rows_field) or [])[:max_items]
        longest = max((len(str(row.get("label", ""))) for row in rows), default=0)
        complexity = len(str(record.get(label_field, ""))) + len(rows) * 12 + longest + min(int(record.get(total_field, 0)), 100)
        complexities.append((record, complexity))
    target = median(value for _, value in complexities)
    return min(complexities, key=lambda item: (abs(item[1] - target), str(item[0].get(label_field, ""))))[0]


def select_density_aware_category_value_scenarios(
    records: list[dict[str, Any]],
    *,
    key_field: str,
    label_field: str,
    rows_field: str = "issue_rows",
    total_field: str = "speech_count",
    max_items: int = 7,
    dense_min_items: int = 5,
    maximum_min_items: int = 6,
) -> dict[str, dict[str, Any]]:
    """Apply density requirements on top of the existing real-data-first selector."""
    base = select_real_category_value_scenarios(
        records,
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )

    dense_records = [record for record in records if _count(record, rows_field, max_items) >= dense_min_items]
    maximum_records = [record for record in records if _count(record, rows_field, max_items) >= maximum_min_items]

    if maximum_records:
        dense_max = select_real_category_value_scenarios(
            maximum_records,
            key_field=key_field,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        )
        base["item_count_max"] = dense_max["item_count_max"]
    else:
        base["item_count_max"] = _waiver("item_count_max", maximum_min_items)

    if dense_records:
        dense = select_real_category_value_scenarios(
            dense_records,
            key_field=key_field,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        )
        for scenario in DENSE_PREFERRED:
            if dense.get(scenario, {}).get("waived") is not True:
                base[scenario] = dense[scenario]

        representative = _representative_dense_record(
            dense_records,
            rows_field=rows_field,
            total_field=total_field,
            label_field=label_field,
            max_items=max_items,
        )
        rep = select_real_category_value_scenarios(
            [representative],
            key_field=key_field,
            label_field=label_field,
            rows_field=rows_field,
            total_field=total_field,
            max_items=max_items,
        )["real_example"]
        rep["selection_reason"] = (
            f"Representative real record selected from candidates with at least {dense_min_items} displayed categories."
        )
        base["real_example"] = rep
    else:
        for scenario in DENSE_PREFERRED:
            base[scenario] = _waiver(scenario, dense_min_items)

    base["minimum"] = {**deepcopy(base["item_count_min"]), "scenario": "minimum"}
    base["maximum"] = {**deepcopy(base["item_count_max"]), "scenario": "maximum"}
    return base
