from __future__ import annotations

import unittest

from process.oireachtas_refresh_inputs import _order_control_tables_last


class RefreshOrderTests(unittest.TestCase):
    def test_control_tables_are_moved_to_safe_tail_order(self) -> None:
        requested = [
            "silver_members",
            "control_table_manifests",
            "silver_questions",
            "control_data_quality_results",
            "control_pipeline_runs",
        ]

        ordered = _order_control_tables_last(requested)

        self.assertEqual(
            ordered,
            [
                "silver_members",
                "silver_questions",
                "control_pipeline_runs",
                "control_data_quality_results",
                "control_table_manifests",
            ],
        )

    def test_non_control_order_is_preserved(self) -> None:
        requested = ["silver_questions", "silver_members", "silver_bills"]

        self.assertEqual(_order_control_tables_last(requested), requested)


if __name__ == "__main__":
    unittest.main()
