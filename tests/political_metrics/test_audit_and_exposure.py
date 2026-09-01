import unittest

import pandas as pd

from political_metrics.audit import history_coverage, speech_count_reconciliation, speech_temporal_attribution_audit
from political_metrics.eligibility import constituency_debate_day_exposure, party_debate_day_exposure


class GroupExposureTests(unittest.TestCase):
    def test_party_switch_splits_exposure(self):
        history = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "party_uri": ["party:A", "party:B"],
            "party_start": ["2026-01-01", "2026-01-16"],
            "party_end": ["2026-01-15", None],
        })
        debate_days = ["2026-01-10", "2026-01-20"]
        result = party_debate_day_exposure(history, debate_days).set_index("party_uri")
        self.assertEqual(result.loc["party:A", "member_debate_days"], 1)
        self.assertEqual(result.loc["party:B", "member_debate_days"], 1)
        self.assertAlmostEqual(result.loc["party:A", "active_member_equivalent"], 0.5)
        self.assertAlmostEqual(result.loc["party:B", "active_member_equivalent"], 0.5)

    def test_constituency_exposure_counts_distinct_member_days(self):
        history = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "constituency_uri": ["constituency:C", "constituency:C"],
            "represent_start": ["2026-01-01", "2026-01-01"],
            "represent_end": [None, None],
        })
        debate_days = ["2026-01-10", "2026-01-20"]
        result = constituency_debate_day_exposure(history, debate_days).set_index("constituency_uri")
        self.assertEqual(result.loc["constituency:C", "member_debate_days"], 4)
        self.assertEqual(result.loc["constituency:C", "active_member_equivalent"], 2.0)
        self.assertEqual(result.loc["constituency:C", "active_member_count"], 2)


class AuditTests(unittest.TestCase):
    def test_history_coverage_reports_range(self):
        history = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "party_start": ["2025-01-01", "2026-01-01"],
            "party_end": ["2025-12-31", None],
        })
        result = history_coverage(
            history,
            dataset="silver_member_parties",
            entity_col="member_code",
            start_col="party_start",
            end_col="party_end",
        )
        self.assertEqual(result.row_count, 2)
        self.assertEqual(result.entity_count, 2)
        self.assertEqual(result.min_start, "2025-01-01")
        self.assertEqual(result.max_end, "2025-12-31")
        self.assertEqual(result.open_ended_rows, 1)

    def test_speech_reconciliation(self):
        speeches = pd.DataFrame({
            "speech_id": ["s1", "s2", "s3"],
            "member_code": ["m1", None, "m2"],
        })
        result = speech_count_reconciliation(speeches)
        self.assertEqual(result["national_distinct_speeches"], 3)
        self.assertEqual(result["attributable_member_speeches"], 2)
        self.assertEqual(result["unattributed_speeches"], 1)
        self.assertTrue(result["reconciles"])

    def test_temporal_attribution_coverage(self):
        speeches = pd.DataFrame({
            "speech_id": ["s1", "s2"],
            "member_code": ["m1", "m2"],
            "debate_date": ["2026-01-10", "2026-01-10"],
        })
        parties = pd.DataFrame({
            "member_code": ["m1"],
            "party_start": ["2026-01-01"],
            "party_end": [None],
            "party_uri": ["party:A"],
            "party_name": ["Party A"],
        })
        constituencies = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "represent_start": ["2026-01-01", "2026-01-01"],
            "represent_end": [None, None],
            "constituency_uri": ["constituency:C1", "constituency:C2"],
            "constituency_name": ["C1", "C2"],
        })
        result = speech_temporal_attribution_audit(speeches, parties, constituencies)
        self.assertEqual(result["party_attribution_coverage"], 0.5)
        self.assertEqual(result["constituency_attribution_coverage"], 1.0)
        self.assertEqual(result["party_unmatched_rows"], 1)


if __name__ == "__main__":
    unittest.main()
