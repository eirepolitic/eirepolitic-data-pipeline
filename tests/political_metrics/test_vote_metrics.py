import unittest

import pandas as pd

from political_metrics.calculators.votes import (
    constituency_vote_participation,
    eligible_division_pairs,
    member_vote_participation,
    party_vote_metrics,
    vote_unity_reliability,
)


class VoteMetricTests(unittest.TestCase):
    def test_division_eligibility_respects_membership_dates(self):
        memberships = pd.DataFrame({
            "member_code": ["m1"],
            "membership_start": ["2026-07-10"],
            "membership_end": [None],
        })
        divisions = pd.DataFrame({
            "division_id": ["d1", "d2", "d3"],
            "division_date": ["2026-07-01", "2026-07-10", "2026-07-20"],
        })
        pairs = eligible_division_pairs(memberships, divisions)
        self.assertEqual(pairs["division_id"].tolist(), ["d2", "d3"])

    def test_member_participation_uses_eligible_divisions(self):
        eligible = pd.DataFrame({
            "member_code": ["m1", "m1", "m2"],
            "division_id": ["d1", "d2", "d2"],
            "division_date": ["2026-07-01", "2026-07-02", "2026-07-02"],
        })
        votes = pd.DataFrame({
            "member_code": ["m1"],
            "division_id": ["d1"],
            "division_date": ["2026-07-01"],
            "vote_code": ["ta"],
        })
        result = member_vote_participation(votes, eligible).set_index("member_code")
        self.assertAlmostEqual(float(result.loc["m1", "vote_participation_pct"]), 0.5)
        self.assertEqual(float(result.loc["m2", "vote_participation_pct"]), 0.0)

    def test_party_cohesion_counts_modal_recorded_vote(self):
        eligible = pd.DataFrame({
            "member_code": ["m1", "m2", "m1", "m2"],
            "division_id": ["d1", "d1", "d2", "d2"],
            "division_date": ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02"],
        })
        parties = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "party_start": ["2026-01-01", "2026-01-01"],
            "party_end": [None, None],
            "party_uri": ["party:A", "party:A"],
            "party_name": ["Party A", "Party A"],
        })
        votes = pd.DataFrame({
            "member_code": ["m1", "m2", "m1", "m2"],
            "division_id": ["d1", "d1", "d2", "d2"],
            "division_date": ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02"],
            "vote_code": ["ta", "ta", "ta", "nil"],
        })
        result = party_vote_metrics(votes, eligible, parties).set_index("party_uri")
        self.assertEqual(result.loc["party:A", "qualifying_unity_divisions"], 2)
        self.assertEqual(result.loc["party:A", "unity_votes_aligned"], 3)
        self.assertEqual(result.loc["party:A", "unity_votes_total"], 4)
        self.assertAlmostEqual(float(result.loc["party:A", "vote_cohesion_pct"]), 0.75)
        self.assertAlmostEqual(float(result.loc["party:A", "vote_participation_pct"]), 1.0)
        self.assertEqual(result.loc["party:A", "unity_reliability_status"], "insufficient_for_comparison")
        self.assertFalse(bool(result.loc["party:A", "unity_public_safe"]))

    def test_single_party_voter_does_not_create_unity_evidence(self):
        eligible = pd.DataFrame({
            "member_code": ["m1"],
            "division_id": ["d1"],
            "division_date": ["2026-07-01"],
        })
        parties = pd.DataFrame({
            "member_code": ["m1"],
            "party_start": ["2026-01-01"],
            "party_end": [None],
            "party_uri": ["party:A"],
            "party_name": ["Party A"],
        })
        votes = pd.DataFrame({
            "member_code": ["m1"],
            "division_id": ["d1"],
            "division_date": ["2026-07-01"],
            "vote_code": ["ta"],
        })
        result = party_vote_metrics(votes, eligible, parties).iloc[0]
        self.assertEqual(result["qualifying_unity_divisions"], 0)
        self.assertTrue(pd.isna(result["vote_cohesion_pct"]))
        self.assertEqual(result["unity_reliability_status"], "insufficient_for_comparison")
        self.assertFalse(bool(result["unity_public_safe"]))

    def test_unity_reliability_thresholds(self):
        self.assertEqual(vote_unity_reliability(4), "insufficient_for_comparison")
        self.assertEqual(vote_unity_reliability(5), "caution")
        self.assertEqual(vote_unity_reliability(9), "caution")
        self.assertEqual(vote_unity_reliability(10), "reliable")

    def test_constituency_participation_is_period_correct(self):
        eligible = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "division_id": ["d1", "d2"],
            "division_date": ["2026-07-01", "2026-07-20"],
        })
        constituencies = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "represent_start": ["2026-01-01", "2026-07-10"],
            "represent_end": ["2026-07-10", None],
            "constituency_uri": ["const:C1", "const:C2"],
            "constituency_name": ["C1", "C2"],
        })
        votes = pd.DataFrame({
            "member_code": ["m1"],
            "division_id": ["d2"],
            "division_date": ["2026-07-20"],
            "vote_code": ["ta"],
        })
        result = constituency_vote_participation(votes, eligible, constituencies).set_index("constituency_uri")
        self.assertEqual(float(result.loc["const:C1", "vote_participation_pct"]), 0.0)
        self.assertEqual(float(result.loc["const:C2", "vote_participation_pct"]), 1.0)


if __name__ == "__main__":
    unittest.main()
