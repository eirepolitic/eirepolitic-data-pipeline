import unittest

import pandas as pd

from political_metrics.issue_audit import audit_issue_classification


class IssueAuditTests(unittest.TestCase):
    def _speeches(self):
        return pd.DataFrame({
            "speech_id": ["s1", "s2"],
            "speech_text_hash": ["h1", "h2"],
            "debate_date": ["2026-07-01", "2026-07-02"],
        })

    def _labels(self):
        return pd.DataFrame({
            "speech_id": ["s1", "s2"],
            "source_speech_text_hash": ["h1", "h2"],
            "issue_label": ["Health", "NONE"],
            "issue_label_source": ["openai_model", "rule_short_text"],
            "model_name": ["model-x", "rule"],
            "classification_status": ["classified", "skipped_short_text"],
            "classified_at_utc": ["2026-07-03T00:00:00Z", "2026-07-03T00:00:00Z"],
        })

    def test_complete_final_labels_are_ready(self):
        result = audit_issue_classification(self._speeches(), self._labels())
        self.assertTrue(result["ready"])
        self.assertEqual(result["policy_labelled_rows"], 1)
        self.assertEqual(result["none_rows"], 1)

    def test_missing_label_row_fails(self):
        result = audit_issue_classification(self._speeches(), self._labels().iloc[:1].copy())
        self.assertFalse(result["ready"])
        self.assertEqual(result["missing_label_rows"], 1)

    def test_hash_mismatch_fails(self):
        labels = self._labels()
        labels.loc[0, "source_speech_text_hash"] = "stale"
        result = audit_issue_classification(self._speeches(), labels)
        self.assertFalse(result["ready"])
        self.assertEqual(result["hash_mismatch_rows"], 1)

    def test_pending_classification_fails(self):
        labels = self._labels()
        labels.loc[0, "classification_status"] = "pending"
        result = audit_issue_classification(self._speeches(), labels)
        self.assertFalse(result["ready"])
        self.assertEqual(result["non_final_status_rows"], 1)

    def test_period_scope_only_checks_requested_speeches(self):
        labels = self._labels()
        labels.loc[1, "source_speech_text_hash"] = "stale"
        result = audit_issue_classification(
            self._speeches(), labels, period_start="2026-07-01", period_end="2026-07-01"
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["scope_rows"], 1)


if __name__ == "__main__":
    unittest.main()
