from __future__ import annotations

from unittest import TestCase

from instagram.factory.adapters import ADAPTERS
from instagram.factory.historical_validation import merge_historical_scenarios
from instagram.factory.validation_contact_sheet import CARD_HEIGHT, WAIVER_CARD_HEIGHT, _layout_grid, _row_heights
from instagram.factory.validation_scenarios import (
    HORIZONTAL_BAR_REQUIRED_SCENARIOS,
    ITEM_COUNT_MAX_MIN,
    ITEM_COUNT_MIN_MAX,
    LABELS_LONG_MIN_LENGTH,
    LABELS_SHORT_MAX_LENGTH,
    REAL_EXAMPLE_MIN_ITEMS,
    VALUES_TIGHT_MAX_RELATIVE_SPREAD,
    VALUES_WIDE_MIN_POSITIVE_RATIO,
    select_real_category_value_scenarios,
)


def _record(key: str, label: str, labels: list[str], values: list[int]) -> dict:
    rows = [{"label": item_label, "value": value} for item_label, value in zip(labels, values)]
    return {
        "item_key": key,
        "item_label": label,
        "issue_rows": rows,
        "issue_count": len(rows),
        "speech_count": sum(values),
    }


def _select(records: list[dict]) -> dict[str, dict]:
    return select_real_category_value_scenarios(records, key_field="item_key", label_field="item_label", max_items=7)


class ValidationScenarioSemanticContractTest(TestCase):
    def test_threshold_qualified_scenarios_or_waivers(self) -> None:
        records = [
            _record("sparse", "Sparse", ["Tax", "Jobs", "Health"], [17, 16, 15]),
            _record("dense", "Dense", ["Tax", "Jobs", "Health", "Housing", "Energy", "Justice", "Education"], [100, 80, 50, 30, 20, 10, 5]),
            _record("long", "Long", ["A category label deliberately longer than thirty five chars", "Jobs", "Tax", "Health", "Energy", "Justice"], [40, 35, 30, 25, 20, 15]),
        ]
        scenarios = _select(records)

        self.assertLessEqual(scenarios["item_count_min"]["scenario_metrics"]["displayed_item_count"], ITEM_COUNT_MIN_MAX)
        self.assertGreaterEqual(scenarios["item_count_max"]["scenario_metrics"]["displayed_item_count"], ITEM_COUNT_MAX_MIN)
        self.assertLessEqual(scenarios["labels_short"]["scenario_metrics"]["longest_label_length"], LABELS_SHORT_MAX_LENGTH)
        self.assertGreaterEqual(scenarios["labels_long"]["scenario_metrics"]["longest_label_length"], LABELS_LONG_MIN_LENGTH)
        self.assertLessEqual(scenarios["values_tight"]["scenario_metrics"]["relative_spread"], VALUES_TIGHT_MAX_RELATIVE_SPREAD)
        self.assertGreaterEqual(scenarios["values_wide"]["scenario_metrics"]["positive_max_to_min_ratio"], VALUES_WIDE_MIN_POSITIVE_RATIO)
        self.assertGreaterEqual(scenarios["real_example"]["scenario_metrics"]["displayed_item_count"], REAL_EXAMPLE_MIN_ITEMS)

    def test_values_tight_accepts_zero_relative_spread(self) -> None:
        scenarios = _select([
            _record("equal", "Equal", ["Tax", "Jobs", "Health", "Housing", "Energy"], [5, 5, 5, 5, 5]),
        ])
        self.assertFalse(scenarios["values_tight"].get("waived", False))
        self.assertEqual(scenarios["values_tight"]["scenario_metrics"]["relative_spread"], 0.0)

    def test_non_qualifying_best_available_records_are_waived(self) -> None:
        records = [
            _record("seven", "Seven", ["Long category label number one", "Long category label number two", "Health", "Housing", "Energy", "Justice", "Education"], [100, 90, 80, 70, 60, 55, 51]),
        ]
        scenarios = _select(records)
        self.assertTrue(scenarios["item_count_min"]["waived"])
        self.assertTrue(scenarios["labels_short"]["waived"])
        self.assertTrue(scenarios["values_tight"]["waived"])
        self.assertTrue(scenarios["values_wide"]["waived"])

    def test_invalid_current_candidate_does_not_block_qualifying_history(self) -> None:
        current = _select([
            _record("current", "Current", ["Tax", "Jobs", "Health", "Housing", "Energy", "Justice", "Education"], [100, 95, 90, 85, 80, 75, 70]),
        ])
        historical = _select([
            _record("historical", "Historical", ["Tax", "Jobs", "Health", "Housing", "Energy", "Justice", "Education"], [100, 70, 40, 20, 10, 5, 2]),
        ])
        historical["values_wide"]["data_origin"] = "historical_real"
        historical["values_wide"]["source_batch_id"] = "historical-batch"

        self.assertTrue(current["values_wide"]["waived"])
        merged, report = merge_historical_scenarios(current, historical)
        selected = merged["values_wide"]
        self.assertFalse(selected.get("waived", False))
        self.assertEqual(selected["data_origin"], "historical_real")
        self.assertGreaterEqual(selected["scenario_metrics"]["positive_max_to_min_ratio"], VALUES_WIDE_MIN_POSITIVE_RATIO)
        self.assertEqual(report["replacement_count"], 1)

    def test_no_current_or_historical_qualifier_remains_explicit_waiver(self) -> None:
        current = _select([_record("current", "Current", ["Tax", "Jobs", "Health", "Housing", "Energy", "Justice", "Education"], [100, 90, 80, 70, 60, 55, 51])])
        historical = _select([_record("history", "History", ["Tax", "Jobs", "Health", "Housing", "Energy", "Justice", "Education"], [120, 110, 100, 90, 80, 70, 65])])
        merged, _ = merge_historical_scenarios(current, historical)
        selected = merged["values_wide"]
        self.assertTrue(selected["waived"])
        self.assertEqual(selected["search_stages_attempted"], ["current_real", "historical_real"])

    def test_exact_semantic_conditions_remain_exact(self) -> None:
        scenarios = _select([
            _record("ties", "Ties", ["Tax", "Jobs", "Health", "Housing", "Energy"], [10, 10, 8, 7, 6]),
            _record("equal", "Equal", ["Tax", "Jobs", "Health", "Housing", "Energy"], [5, 5, 5, 5, 5]),
            _record("zero", "Zero", ["Tax", "Jobs", "Health", "Housing", "Energy"], [10, 8, 5, 2, 0]),
        ])
        self.assertTrue(scenarios["ties"]["scenario_metrics"]["has_ties"])
        self.assertTrue(scenarios["all_equal"]["scenario_metrics"]["all_equal"])
        self.assertTrue(scenarios["zeros"]["scenario_metrics"]["has_zero"])

    def test_party_and_constituency_adapters_share_semantic_contract(self) -> None:
        rows = [
            {"label": "Tax", "value": 100},
            {"label": "Jobs", "value": 70},
            {"label": "Health", "value": 40},
            {"label": "Housing", "value": 20},
            {"label": "Energy", "value": 10},
            {"label": "Justice", "value": 5},
            {"label": "Education", "value": 2},
        ]
        party_record = {"party": "Example Party", "party_key": "example-party", "issue_rows": rows, "issue_count": 7, "speech_count": 247}
        constituency_record = {"constituency": "Example", "constituency_key": "example", "issue_rows": rows, "issue_count": 7, "speech_count": 247}
        party = ADAPTERS["party_issue_profile_v1"].build_scenarios([party_record], {})
        constituency = ADAPTERS["constituency_issue_profile_v1"].build_scenarios([constituency_record], {})
        for scenario in HORIZONTAL_BAR_REQUIRED_SCENARIOS:
            self.assertEqual(bool(party[scenario].get("waived")), bool(constituency[scenario].get("waived")), scenario)
            if not party[scenario].get("waived"):
                self.assertEqual(party[scenario]["scenario_metrics"], constituency[scenario]["scenario_metrics"], scenario)

    def test_contact_sheet_uses_compact_waiver_rows_without_shrinking_visual_cards(self) -> None:
        entries = [
            {"kind": "visual", "scenarios": ["real_example"]},
            {"kind": "visual", "scenarios": ["item_count_max"]},
            {"kind": "visual", "scenarios": ["labels_long"]},
            {"kind": "waiver", "scenarios": ["zeros"]},
            {"kind": "waiver", "scenarios": ["all_equal"]},
            {"kind": "waiver", "scenarios": ["single_outlier"]},
        ]
        placements = _layout_grid(entries)
        heights = _row_heights(placements)
        self.assertEqual(CARD_HEIGHT, 1180)
        self.assertGreaterEqual(WAIVER_CARD_HEIGHT, 300)
        self.assertLessEqual(WAIVER_CARD_HEIGHT, 400)
        self.assertIn(CARD_HEIGHT, heights.values())
        self.assertIn(WAIVER_CARD_HEIGHT, heights.values())

        kinds_by_row: dict[int, set[str]] = {}
        for row, _, _, entry in placements:
            kinds_by_row.setdefault(row, set()).add(entry["kind"])
        self.assertTrue(all(len(kinds) == 1 for kinds in kinds_by_row.values()))

        waiver_placements = [placement for placement in placements if placement[3]["kind"] == "waiver"]
        self.assertEqual(waiver_placements[0][0], waiver_placements[1][0])
        self.assertEqual(waiver_placements[-1][2], 2)
