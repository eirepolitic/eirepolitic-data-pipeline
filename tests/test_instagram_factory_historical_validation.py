from __future__ import annotations

from unittest import TestCase

from instagram.factory.historical_validation import merge_historical_scenarios


class HistoricalValidationMergeTest(TestCase):
    def test_replaces_only_current_waivers(self) -> None:
        current = {
            "values_tight": {
                "scenario": "values_tight",
                "data_origin": "current_real",
                "source_item_key": "current-party",
            },
            "single_outlier": {
                "scenario": "single_outlier",
                "waived": True,
                "data_origin": "waived_no_real_case",
                "waiver_reason": "No current real record qualifies.",
            },
            "zeros": {
                "scenario": "zeros",
                "waived": True,
                "data_origin": "waived_no_real_case",
                "waiver_reason": "No current real record contains zero values.",
            },
            "item_count_min": {
                "scenario": "item_count_min",
                "data_origin": "current_real",
                "source_item_key": "current-min",
            },
            "item_count_max": {
                "scenario": "item_count_max",
                "data_origin": "current_real",
                "source_item_key": "current-max",
            },
        }
        historical = {
            "values_tight": {
                "scenario": "values_tight",
                "data_origin": "current_real",
                "source_item_key": "historical-tight",
                "source_batch_id": "batch-old",
            },
            "single_outlier": {
                "scenario": "single_outlier",
                "data_origin": "current_real",
                "source_item_key": "historical-outlier",
                "source_item_label": "Historic Party",
                "source_batch_id": "batch-old",
                "selection_reason": "Current real record with a qualifying outlier.",
            },
            "zeros": {
                "scenario": "zeros",
                "waived": True,
                "data_origin": "waived_no_real_case",
                "waiver_reason": "No qualifying historical record.",
            },
        }

        merged, report = merge_historical_scenarios(current, historical)

        self.assertEqual(merged["values_tight"]["source_item_key"], "current-party")
        self.assertEqual(merged["single_outlier"]["data_origin"], "historical_real")
        self.assertEqual(merged["single_outlier"]["source_batch_id"], "batch-old")
        self.assertTrue(merged["single_outlier"]["historical_fallback"])
        self.assertEqual(
            merged["single_outlier"]["search_stages_attempted"],
            ["current_real", "historical_real"],
        )
        self.assertTrue(merged["zeros"]["waived"])
        self.assertIn("historical", merged["zeros"]["waiver_reason"].lower())
        self.assertEqual(report["replacement_count"], 1)
        self.assertEqual(report["retained_waivers"], ["zeros"])
        self.assertEqual(merged["minimum"]["source_item_key"], "current-min")
        self.assertEqual(merged["maximum"]["source_item_key"], "current-max")
