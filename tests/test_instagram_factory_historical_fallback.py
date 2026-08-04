from __future__ import annotations

from unittest import TestCase

from instagram.factory.generic_tests import _combine_scenarios


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
                "data_origin": "current_real",
                "source_item_label": "Historical Party",
                "source_batch_id": "batch-old",
                "selection_reason": "Current real record selected.",
                "synthetic": False,
            }
        }
        combined = _combine_scenarios(
            required_scenarios=["values_wide"],
            current=current,
            historical=historical,
            historical_manifest={"status": "completed", "loaded_batch_count": 3},
        )
        selected = combined["values_wide"]
        self.assertEqual(selected["source_item_label"], "Current Party")
        self.assertEqual(selected["data_origin"], "current_real")
        self.assertEqual(selected["search_stages"][0]["status"], "matched")
        self.assertEqual(selected["search_stages"][1]["status"], "not_needed")

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
                "data_origin": "current_real",
                "source_item_label": "Historical Party",
                "source_batch_id": "batch-old",
                "historical_batch_rank": 2,
                "selection_reason": "Current real record with a large outlier.",
                "synthetic": False,
            }
        }
        combined = _combine_scenarios(
            required_scenarios=["single_outlier"],
            current=current,
            historical=historical,
            historical_manifest={"status": "completed", "loaded_batch_count": 4},
        )
        selected = combined["single_outlier"]
        self.assertFalse(selected.get("waived", False))
        self.assertEqual(selected["data_origin"], "historical_real")
        self.assertEqual(selected["source_batch_id"], "batch-old")
        self.assertIn("Historical real", selected["selection_reason"])
        self.assertEqual(selected["search_stages"][0]["status"], "no_qualifying_case")
        self.assertEqual(selected["search_stages"][1]["status"], "matched")

    def test_waiver_records_current_and_historical_search(self) -> None:
        current = {
            "zeros": {
                "scenario": "zeros",
                "waived": True,
                "waiver_reason": "No current real zero-value category exists.",
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
        combined = _combine_scenarios(
            required_scenarios=["zeros"],
            current=current,
            historical=historical,
            historical_manifest={"status": "completed", "loaded_batch_count": 5},
        )
        selected = combined["zeros"]
        self.assertTrue(selected["waived"])
        self.assertIn("5 loaded batch(es)", selected["waiver_reason"])
        self.assertEqual([stage["stage"] for stage in selected["search_stages"]], [
            "current_real",
            "historical_real",
            "synthetic_contract_edge",
            "waived",
        ])
