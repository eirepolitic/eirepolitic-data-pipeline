from __future__ import annotations

import argparse
import unittest
from datetime import date
from pathlib import Path

from process.oireachtas_refresh_inputs import normalize


class RefreshInputTests(unittest.TestCase):
    def args(self, **overrides: str) -> argparse.Namespace:
        values = {
            "refresh_type": "weekly",
            "mode": "",
            "tables": "silver_members,silver_member_votes",
            "chamber": "dail",
            "house_no": "34",
            "date_start": "2026-06-01",
            "date_end": "2026-07-01",
            "page_size": "100",
            "sample_rows": "10",
            "as_of_date": "2026-07-14",
            "github_output": "",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_manual_and_scheduled_equivalent_inputs_normalize_identically(self) -> None:
        manual = normalize(self.args(date_start="2026-06-09", date_end="2026-07-14"))
        scheduled = normalize(self.args(date_start="", date_end=""))
        self.assertEqual(manual, scheduled)

    def test_weekly_default_window_is_35_days(self) -> None:
        payload = normalize(self.args(date_start="", date_end=""))
        self.assertEqual(payload["date_start"], "2026-06-09")
        self.assertEqual(payload["date_end"], "2026-07-14")

    def test_monthly_default_window_includes_seven_day_overlap(self) -> None:
        payload = normalize(
            self.args(
                refresh_type="monthly",
                mode="incremental",
                date_start="",
                date_end="",
                page_size="200",
            )
        )
        self.assertEqual(payload["date_start"], "2026-05-25")
        self.assertEqual(payload["date_end"], "2026-06-30")

    def test_rejects_unknown_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown tables"):
            normalize(self.args(tables="silver_members,not_a_table"))

    def test_rejects_duplicate_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate tables"):
            normalize(self.args(tables="silver_members,silver_members"))

    def test_rejects_page_size_above_api_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 200"):
            normalize(self.args(page_size="500"))

    def test_rejects_reverse_date_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be after"):
            normalize(self.args(date_start="2026-07-02", date_end="2026-07-01"))


class WorkflowArchitectureTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return Path(".github/workflows", name).read_text(encoding="utf-8")

    def test_only_orchestrator_has_oireachtas_refresh_schedule(self) -> None:
        weekly = self.read("oireachtas_weekly_refresh.yml")
        monthly = self.read("oireachtas_monthly_refresh.yml")
        yearly = self.read("oireachtas_yearly_refresh.yml")
        orchestrator = self.read("oireachtas_refresh_validation_orchestrator.yml")
        self.assertNotIn("schedule:", weekly)
        self.assertNotIn("schedule:", monthly)
        self.assertNotIn("schedule:", yearly)
        self.assertEqual(orchestrator.count("schedule:"), 1)

    def test_orchestrator_does_not_poll_or_dispatch_with_gh(self) -> None:
        orchestrator = self.read("oireachtas_refresh_validation_orchestrator.yml")
        self.assertNotIn("gh workflow run", orchestrator)
        self.assertNotIn("gh run list", orchestrator)
        self.assertNotIn("headSha", orchestrator)
        self.assertIn("uses: ./.github/workflows/oireachtas_refresh_reusable.yml", orchestrator)

    def test_all_manual_refreshes_use_shared_reusable_workflow(self) -> None:
        for name in (
            "oireachtas_weekly_refresh.yml",
            "oireachtas_monthly_refresh.yml",
            "oireachtas_yearly_refresh.yml",
        ):
            text = self.read(name)
            self.assertIn("uses: ./.github/workflows/oireachtas_refresh_reusable.yml", text)
            self.assertIn("batch_id:", text)
            self.assertEqual(text.count("permissions:\n  contents: read"), 1)

    def test_reusable_refresh_uses_shared_concurrency_and_unique_artifacts(self) -> None:
        text = self.read("oireachtas_refresh_reusable.yml")
        self.assertIn("group: oireachtas-production-refresh", text)
        self.assertIn("${{ github.run_id }}-${{ github.run_attempt }}", text)
        self.assertIn("set -euo pipefail", text)


if __name__ == "__main__":
    unittest.main()
