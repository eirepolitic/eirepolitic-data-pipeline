from __future__ import annotations

import unittest

import pandas as pd

from extract.oireachtas.contracts import comparison_status, load_contract_config
from extract.oireachtas.downstream_compat import SOURCE_CURRENT_MEMBERS, _build_members_compat


class RosterCompatibilityTests(unittest.TestCase):
    def test_roster_adapter_uses_complete_silver_members(self) -> None:
        self.assertEqual(
            SOURCE_CURRENT_MEMBERS,
            "processed/oireachtas_unified/latest/csv/silver_members.csv",
        )

    def test_roster_adapter_preserves_all_rows_and_silver_fallbacks(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "member_code": f"m{index}",
                    "full_name": f"Member {index}",
                    "latest_constituency_name": "Example",
                    "latest_party_name": "Party",
                    "latest_house_no": "34",
                    "snapshot_date": "2026-07-16",
                }
                for index in range(176)
            ]
        )
        output = _build_members_compat(source)
        self.assertEqual(len(output), 176)
        self.assertEqual(set(output["member_code"]), set(source["member_code"]))
        self.assertTrue(output["constituency"].eq("Example").all())
        self.assertTrue(output["party"].eq("Party").all())
        self.assertTrue(output["house_no"].eq("34").all())


class VoteCompatibilityThresholdTests(unittest.TestCase):
    def test_incremental_vote_volume_difference_does_not_fail_member_coverage_gate(self) -> None:
        _, thresholds = load_contract_config()
        status, reasons = comparison_status(
            {
                "legacy_rows": 30968,
                "compat_rows": 8973,
                "legacy_only_key_count": 2,
                "compat_only_key_count": 2,
                "compat_join_coverage_pct": 100.0,
            },
            thresholds["member_votes_compat"],
        )
        self.assertEqual(status, "pass", reasons)

    def test_vote_member_drift_above_two_still_fails(self) -> None:
        _, thresholds = load_contract_config()
        status, reasons = comparison_status(
            {
                "legacy_rows": 30968,
                "compat_rows": 8973,
                "legacy_only_key_count": 3,
                "compat_only_key_count": 2,
                "compat_join_coverage_pct": 100.0,
            },
            thresholds["member_votes_compat"],
        )
        self.assertEqual(status, "fail")
        self.assertTrue(any("legacy-only keys 3" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
