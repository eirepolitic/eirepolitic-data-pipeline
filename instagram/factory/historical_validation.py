from __future__ import annotations

from copy import deepcopy
from typing import Any


def merge_historical_scenarios(
    current: dict[str, dict[str, Any]],
    historical: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Replace only current waivers with qualifying historical real scenarios."""
    merged = deepcopy(current)
    replacements: list[dict[str, Any]] = []
    retained_current: list[str] = []
    retained_waivers: list[str] = []

    for scenario, current_value in current.items():
        if scenario in {"minimum", "maximum"}:
            continue
        if current_value.get("waived") is not True:
            retained_current.append(scenario)
            continue
        candidate = historical.get(scenario)
        if not candidate or candidate.get("waived") is True:
            retained_waivers.append(scenario)
            continue

        replacement = deepcopy(candidate)
        replacement["scenario"] = scenario
        replacement["data_origin"] = "historical_real"
        replacement["selection_reason"] = (
            "No qualifying current production record existed. "
            + str(replacement.get("selection_reason") or "Selected from historical production data.")
              .replace("Current real record", "Historical real record")
        )
        replacement["historical_fallback"] = True
        replacement["search_stages_attempted"] = ["current_real", "historical_real"]
        merged[scenario] = replacement
        replacements.append({
            "scenario": scenario,
            "source_batch_id": replacement.get("source_batch_id"),
            "source_item_key": replacement.get("source_item_key"),
            "source_item_label": replacement.get("source_item_label"),
        })

    if "item_count_min" in merged:
        merged["minimum"] = {**deepcopy(merged["item_count_min"]), "scenario": "minimum"}
    if "item_count_max" in merged:
        merged["maximum"] = {**deepcopy(merged["item_count_max"]), "scenario": "maximum"}

    for scenario in retained_waivers:
        waiver = merged[scenario]
        waiver["search_stages_attempted"] = ["current_real", "historical_real"]
        original = str(waiver.get("waiver_reason") or "No qualifying real record exists.")
        waiver["waiver_reason"] = (
            original.replace("No current real record", "No current or searched historical real record")
            if "No current real record" in original
            else f"Current and searched historical production data were checked. {original}"
        )

    report = {
        "replacement_count": len(replacements),
        "replacements": replacements,
        "retained_current_scenarios": sorted(retained_current),
        "retained_waivers": sorted(retained_waivers),
    }
    return merged, report
