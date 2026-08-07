from __future__ import annotations

from unittest import TestCase

from instagram.factory.historical_validation import merge_historical_scenarios


class HistoricalFallbackTest(TestCase):
    def test_current_real_wins_over_historical(self) -> None:
        current = {
            "values_wide": {
                "scenario": "values_wide",
                "data_origin": "current_real",
                "source_item_label": "Current Party",
                "selection_reason": "Current real record selected.",
                "synthetic": False,
            }
        }
        historical = {
            "values_wide": {
                "scenario": "values_wide",
                "data_origin": "historical_real",
                "source_item_label": "Historical Party",
                "source_batch_id": "batch-old",
                "selection_reason": "Historical real record selected.",
                "synthetic": False,
            }
        }
        merged, report = merge_historical_scenarios(current, historical)
        selected = merged["values_wide"]
        self.assertEqual(selected["source_item_label"], "Current Party")
        self.assertEqual(selected["data_origin"], "current_real")
        self.assertEqual(report["replacement_count"], 0)
        self.assertIn("values_wide", report["retained_current_scenarios"])

    def test_historical_real_fills_missing_current_scenario(self) -> None:
        current = {
            "single_outlier": {
                "scenario": "single_outlier",
                "waived": True,
                "waiver_reason": "No current real record qualifies.",
                "data_origin": "waived_no_real_case",
                "synthetic": False,
            }
        }
        historical = {
            "single_outlier": {
                "scenario": "single_outlier",
                "data_origin": "historical_real",
                "source_item_label": "Historical Party",
                "source_item_key": "historical-party",
                "source_batch_id": "batch-old",
                "selection_reason": "Current real record with a large outlier.",
                "synthetic": False,
            }
        }
        merged, report = merge_historical_scenarios(current, historical)
        selected = merged["single_outlier"]
        self.assertFalse(selected.get("waived", False))
        self.assertEqual(selected["data_origin"], "historical_real")
        self.assertEqual(selected["source_batch_id"], "batch-old")
        self.assertTrue(selected["historical_fallback"])
        self.assertEqual(selected["search_stages_attempted"], ["current_real", "historical_real"])
        self.assertIn("No qualifying current production record existed", selected["selection_reason"])
        self.assertEqual(report["replacement_count"], 1)

    def test_waiver_records_current_and_historical_search(self) -> None:
        current = {
            "zeros": {
                "scenario": "zeros",
                "waived": True,
                "waiver_reason": "No current real record contains a displayed zero value.",
                "data_origin": "waived_no_real_case",
                "synthetic": False,
            }
        }
        historical = {
            "zeros": {
                "scenario": "zeros",
                "waived": True,
                "waiver_reason": "No historical zero-value category exists.",
                "data_origin": "waived_no_real_case",
                "synthetic": False,
            }
        }
        merged, report = merge_historical_scenarios(current, historical)
        selected = merged["zeros"]
        self.assertTrue(selected["waived"])
        self.assertEqual(selected["search_stages_attempted"], ["current_real", "historical_real"])
        self.assertIn("current or searched historical real record", selected["waiver_reason"])
        self.assertEqual(report["replacement_count"], 0)
        self.assertIn("zeros", report["retained_waivers"])
