import argparse
import unittest

from process.oireachtas_refresh_inputs import normalize


class RefreshPolicyTests(unittest.TestCase):
    def _args(self, refresh_type: str, tables: str = ""):
        return argparse.Namespace(
            refresh_type=refresh_type,
            mode="",
            tables=tables,
            chamber="dail",
            house_no="34",
            date_start="",
            date_end="",
            page_size="",
            sample_rows="10",
            as_of_date="2026-08-31",
            github_output="",
        )

    def test_weekly_default_changes_speeches(self):
        result = normalize(self._args("weekly"))
        self.assertEqual(result["changes_speeches"], "true")

    def test_monthly_default_does_not_change_speeches(self):
        result = normalize(self._args("monthly"))
        self.assertEqual(result["changes_speeches"], "false")

    def test_yearly_default_does_not_change_speeches(self):
        result = normalize(self._args("yearly"))
        self.assertEqual(result["changes_speeches"], "false")

    def test_custom_refresh_detects_speech_table(self):
        result = normalize(self._args("monthly", tables="silver_speeches"))
        self.assertEqual(result["changes_speeches"], "true")


if __name__ == "__main__":
    unittest.main()
