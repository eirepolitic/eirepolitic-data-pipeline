import unittest

import pandas as pd

from political_metrics.calculators.issues import grouped_issue_metrics, national_issue_metrics, party_issue_comparisons


class IssueMetricTests(unittest.TestCase):
    def test_national_issue_share_excludes_none(self):
        data = pd.DataFrame({
            "speech_id": ["s1", "s2", "s3"],
            "issue_label": ["Health", "Health", "NONE"],
        })
        result = national_issue_metrics(data).set_index("issue_label")
        self.assertEqual(result.loc["Health", "issue_speech_count"], 2)
        self.assertEqual(result.loc["Health", "policy_speech_count"], 2)
        self.assertEqual(float(result.loc["Health", "issue_share"]), 1.0)

    def test_group_issue_metrics_include_zero_issue_shares(self):
        data = pd.DataFrame({
            "speech_id": ["s1", "s2"],
            "party_uri": ["A", "B"],
            "issue_label": ["Health", "Education"],
        })
        result = grouped_issue_metrics(data, group_col="party_uri")
        health = result[(result["party_uri"] == "B") & (result["issue_label"] == "Health")].iloc[0]
        self.assertEqual(health["issue_speech_count"], 0)
        self.assertEqual(health["policy_speech_count"], 1)
        self.assertEqual(float(health["issue_share"]), 0.0)

    def test_party_comparison_uses_unweighted_eligible_party_average(self):
        party = pd.DataFrame({
            "party_uri": ["A", "B", "Independent"],
            "issue_label": ["Health", "Health", "Health"],
            "issue_speech_count": [10, 2, 10],
            "policy_speech_count": [20, 20, 20],
            "issue_share": [0.5, 0.1, 0.5],
        })
        national = pd.DataFrame({
            "issue_label": ["Health"],
            "issue_speech_count": [22],
            "policy_speech_count": [60],
            "issue_share": [22 / 60],
        })
        result = party_issue_comparisons(
            party,
            national,
            excluded_average_party_ids={"Independent"},
            baseline_min_policy_speeches=20,
        ).set_index("party_uri")
        self.assertAlmostEqual(float(result.loc["A", "average_party_issue_share"]), 0.3)
        self.assertAlmostEqual(float(result.loc["A", "share_vs_average_party_pp"]), 20.0)
        self.assertAlmostEqual(float(result.loc["B", "share_vs_average_party_pp"]), -20.0)

    def test_small_party_is_not_public_safe_for_comparison(self):
        party = pd.DataFrame({
            "party_uri": ["A"],
            "issue_label": ["Health"],
            "issue_speech_count": [2],
            "policy_speech_count": [9],
            "issue_share": [2 / 9],
        })
        national = pd.DataFrame({
            "issue_label": ["Health"],
            "issue_speech_count": [20],
            "policy_speech_count": [100],
            "issue_share": [0.2],
        })
        result = party_issue_comparisons(party, national)
        self.assertEqual(result.loc[0, "reliability_status"], "insufficient_for_comparison")
        self.assertFalse(bool(result.loc[0, "comparison_public_safe"]))


if __name__ == "__main__":
    unittest.main()
