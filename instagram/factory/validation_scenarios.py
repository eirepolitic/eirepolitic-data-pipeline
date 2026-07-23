from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any, Callable

HORIZONTAL_BAR_REQUIRED_SCENARIOS = [
    "item_count_min",
    "item_count_max",
    "labels_short",
    "labels_long",
    "values_small",
    "values_large",
    "values_tight",
    "values_wide",
    "single_outlier",
    "all_equal",
    "ties",
    "zeros",
    "real_example",
]


def _display_rows(record: dict[str, Any], rows_field: str, max_items: int) -> list[dict[str, Any]]:
    return [deepcopy(row) for row in record.get(rows_field, [])[:max_items]]


def _values(record: dict[str, Any], rows_field: str, max_items: int) -> list[float]:
    output: list[float] = []
    for row in _display_rows(record, rows_field, max_items):
        try:
            output.append(float(row.get("value", 0)))
        except (TypeError, ValueError):
            continue
    return output


def _labels(record: dict[str, Any], rows_field: str, max_items: int) -> list[str]:
    return [str(row.get("label", "")) for row in _display_rows(record, rows_field, max_items)]


def _scenario_metrics(record: dict[str, Any], rows_field: str, max_items: int) -> dict[str, Any]:
    values = _values(record, rows_field, max_items)
    labels = _labels(record, rows_field, max_items)
    positive = [value for value in values if value > 0]
    ordered = sorted(values, reverse=True)
    return {
        "displayed_item_count": len(values),
        "shortest_label_length": min((len(label) for label in labels), default=0),
        "longest_label_length": max((len(label) for label in labels), default=0),
        "minimum_value": min(values, default=None),
        "maximum_value": max(values, default=None),
        "relative_spread": (
            (max(values) - min(values)) / max(values)
            if values and max(values) > 0
            else None
        ),
        "positive_max_to_min_ratio": (
            max(positive) / min(positive)
            if len(positive) >= 2 and min(positive) > 0
            else None
        ),
        "top_to_second_ratio": (
            ordered[0] / ordered[1]
            if len(ordered) >= 2 and ordered[1] > 0
            else None
        ),
        "has_ties": len(values) != len(set(values)),
        "all_equal": len(values) >= 2 and len(set(values)) == 1,
        "has_zero": any(value == 0 for value in values),
    }


def _complexity(record: dict[str, Any], *, label_field: str, rows_field: str, total_field: str, max_items: int) -> int:
    metrics = _scenario_metrics(record, rows_field, max_items)
    return (
        len(str(record.get(label_field, "")))
        + int(metrics["displayed_item_count"]) * 12
        + int(metrics["longest_label_length"])
        + min(int(record.get(total_field, 0)), 100)
    )


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
    selected[total_field] = sum(int(float(row.get("value", 0))) for row in rows)
    selected.update({
        "scenario": scenario,
        "synthetic": False,
        "no_publication": True,
        "data_origin": "current_real",
        "selection_reason": reason,
        "source_item_key": str(record.get(key_field, "")),
        "source_item_label": str(record.get(label_field, "")),
        "scenario_metrics": _scenario_metrics(record, rows_field, max_items),
    })
    return selected


def _waiver(scenario: str, reason: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "waived": True,
        "waiver_reason": reason,
        "synthetic": False,
        "no_publication": True,
        "data_origin": "waived_no_real_case",
    }


def _select(
    records: list[dict[str, Any]],
    *,
    scenario: str,
    reason: str,
    key: Callable[[dict[str, Any]], Any],
    key_field: str,
    label_field: str,
    rows_field: str,
    total_field: str,
    max_items: int,
    candidates: Callable[[dict[str, Any]], bool] | None = None,
    maximum: bool = False,
    waiver_reason: str | None = None,
) -> dict[str, Any]:
    eligible = [record for record in records if candidates is None or candidates(record)]
    if not eligible:
        return _waiver(scenario, waiver_reason or f"No qualifying current real record exists for {scenario}.")
    selected = (max if maximum else min)(eligible, key=key)
    return _real_scenario(
        selected,
        scenario=scenario,
        reason=reason,
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )


def select_real_horizontal_bar_scenarios(
    records: list[dict[str, Any]],
    *,
    key_field: str,
    label_field: str,
    rows_field: str = "issue_rows",
    total_field: str = "speech_count",
    max_items: int = 7,
    outlier_ratio: float = 3.0,
) -> dict[str, dict[str, Any]]:
    """Build a real-data-first validation matrix for a horizontal bar visual."""
    if not records:
        raise ValueError("Cannot select validation scenarios from an empty record set")

    def metrics(record: dict[str, Any]) -> dict[str, Any]:
        return _scenario_metrics(record, rows_field, max_items)

    scenarios: dict[str, dict[str, Any]] = {}
    scenarios["item_count_min"] = _select(
        records,
        scenario="item_count_min",
        reason="Current real record with the fewest categories actually displayed.",
        key=lambda row: (metrics(row)["displayed_item_count"], int(row.get(total_field, 0)), str(row.get(label_field, ""))),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )
    scenarios["item_count_max"] = _select(
        records,
        scenario="item_count_max",
        reason="Current real record with the most categories actually displayed.",
        key=lambda row: (metrics(row)["displayed_item_count"], int(row.get(total_field, 0)), len(str(row.get(label_field, "")))),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        maximum=True,
    )
    scenarios["labels_short"] = _select(
        records,
        scenario="labels_short",
        reason="Current real record minimizing the longest displayed category label.",
        key=lambda row: (metrics(row)["longest_label_length"], sum(len(label) for label in _labels(row, rows_field, max_items))),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )
    scenarios["labels_long"] = _select(
        records,
        scenario="labels_long",
        reason="Current real record maximizing the longest displayed category label.",
        key=lambda row: (metrics(row)["longest_label_length"], sum(len(label) for label in _labels(row, rows_field, max_items))),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        maximum=True,
    )
    scenarios["values_small"] = _select(
        records,
        scenario="values_small",
        reason="Current real record with the smallest displayed maximum value.",
        key=lambda row: (metrics(row)["maximum_value"] or 0, int(row.get(total_field, 0))),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )
    scenarios["values_large"] = _select(
        records,
        scenario="values_large",
        reason="Current real record with the largest displayed maximum value.",
        key=lambda row: (metrics(row)["maximum_value"] or 0, int(row.get(total_field, 0))),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        maximum=True,
    )
    scenarios["values_tight"] = _select(
        records,
        scenario="values_tight",
        reason="Current real record with the smallest positive relative spread between displayed values.",
        key=lambda row: metrics(row)["relative_spread"],
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        candidates=lambda row: (
            metrics(row)["displayed_item_count"] >= 2
            and metrics(row)["relative_spread"] is not None
            and metrics(row)["relative_spread"] > 0
        ),
        waiver_reason="No current real record contains at least two distinct positive displayed values for a tight-range test.",
    )
    scenarios["values_wide"] = _select(
        records,
        scenario="values_wide",
        reason="Current real record with the largest positive max-to-min value ratio.",
        key=lambda row: metrics(row)["positive_max_to_min_ratio"],
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        candidates=lambda row: metrics(row)["positive_max_to_min_ratio"] is not None,
        maximum=True,
        waiver_reason="No current real record contains at least two positive displayed values for a wide-range test.",
    )
    scenarios["single_outlier"] = _select(
        records,
        scenario="single_outlier",
        reason=f"Current real record with the largest top-to-second value ratio, meeting the {outlier_ratio:g}x outlier threshold.",
        key=lambda row: metrics(row)["top_to_second_ratio"],
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        candidates=lambda row: (
            metrics(row)["displayed_item_count"] >= 3
            and metrics(row)["top_to_second_ratio"] is not None
            and metrics(row)["top_to_second_ratio"] >= outlier_ratio
        ),
        maximum=True,
        waiver_reason=f"No current real record has at least three bars and a top value at least {outlier_ratio:g}x the second value.",
    )
    scenarios["all_equal"] = _select(
        records,
        scenario="all_equal",
        reason="Current real record in which all displayed values are equal.",
        key=lambda row: (metrics(row)["displayed_item_count"], metrics(row)["maximum_value"] or 0),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        candidates=lambda row: metrics(row)["all_equal"],
        maximum=True,
        waiver_reason="No current real record has at least two displayed categories with all values equal.",
    )
    scenarios["ties"] = _select(
        records,
        scenario="ties",
        reason="Current real record containing tied displayed values.",
        key=lambda row: (metrics(row)["displayed_item_count"], metrics(row)["maximum_value"] or 0),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        candidates=lambda row: metrics(row)["has_ties"],
        maximum=True,
        waiver_reason="No current real record contains tied displayed values.",
    )
    scenarios["zeros"] = _select(
        records,
        scenario="zeros",
        reason="Current real record containing a displayed zero value.",
        key=lambda row: (metrics(row)["displayed_item_count"], metrics(row)["maximum_value"] or 0),
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
        candidates=lambda row: metrics(row)["has_zero"],
        maximum=True,
        waiver_reason="No current real record contains a displayed zero value; count metrics currently omit zero-count categories.",
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
    scenarios["real_example"] = _real_scenario(
        representative,
        scenario="real_example",
        reason="Current real record nearest the median combined layout and visual complexity.",
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )

    scenarios["minimum"] = {**deepcopy(scenarios["item_count_min"]), "scenario": "minimum"}
    scenarios["maximum"] = {**deepcopy(scenarios["item_count_max"]), "scenario": "maximum"}
    return scenarios


def select_real_category_value_scenarios(
    records: list[dict[str, Any]],
    *,
    key_field: str,
    label_field: str,
    rows_field: str = "issue_rows",
    total_field: str = "speech_count",
    max_items: int = 7,
) -> dict[str, dict[str, Any]]:
    """Backward-compatible alias for the horizontal-bar real-data matrix."""
    return select_real_horizontal_bar_scenarios(
        records,
        key_field=key_field,
        label_field=label_field,
        rows_field=rows_field,
        total_field=total_field,
        max_items=max_items,
    )
