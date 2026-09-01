import unittest
from datetime import date

from process.political_metrics_materialize_candidate import _completed_month_periods


class CandidateMaterializationTests(unittest.TestCase):
    def test_completed_months_stop_before_current_month(self):
        periods = _completed_month_periods(date(2026, 6, 15), today=date(2026, 8, 31))
        self.assertEqual([period.label for period in periods], ["2026-06", "2026-07"])
        self.assertEqual(periods[-1].end, date(2026, 7, 31))

    def test_start_month_is_included_when_completed(self):
        periods = _completed_month_periods(date(2024, 12, 18), today=date(2025, 2, 1))
        self.assertEqual([period.label for period in periods], ["2024-12", "2025-01"])

    def test_no_completed_months_when_data_starts_current_month(self):
        periods = _completed_month_periods(date(2026, 8, 1), today=date(2026, 8, 31))
        self.assertEqual(periods, [])


if __name__ == "__main__":
    unittest.main()
