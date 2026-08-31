from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class SpeechClassifierWiringTests(unittest.TestCase):
    def test_classifier_workflow_uses_unified_classifier_and_not_legacy_raw_input(self) -> None:
        text = Path(".github/workflows/speech_issue_classifier.yml").read_text(encoding="utf-8")
        self.assertIn("process/oireachtas_speech_issue_classifier.py", text)
        self.assertIn("gpt-5.6-luna", text)
        self.assertIn("openai>=3.6.0,<4", text)
        self.assertNotIn("raw/debates/debate_speeches_extracted.csv", text)
        self.assertNotIn("process/speech_issue_classifier.py", text)

    def test_legacy_enrichment_workflow_is_readiness_only(self) -> None:
        text = Path(".github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml").read_text(encoding="utf-8")
        self.assertIn("--mode readiness", text)
        self.assertIn("process/oireachtas_speech_issue_classifier.py", text)
        self.assertNotIn("extract.oireachtas.enrichment_speech_issue_labels", text)

    def test_refresh_hook_exists_but_is_disabled_by_default(self) -> None:
        text = Path(".github/workflows/oireachtas_refresh_reusable.yml").read_text(encoding="utf-8")
        self.assertIn("classify_speeches: {required: false, type: boolean, default: false}", text)
        self.assertIn("speech_classifier_model", text)
        self.assertIn("process/oireachtas_speech_issue_classifier.py", text)
        self.assertIn("openai>=3.6.0,<4", text)
        self.assertIn("--required-table enrichment_speech_issue_labels", text)

    def test_enrichment_is_registered_as_rebuild_table(self) -> None:
        tables = yaml.safe_load(Path("configs/oireachtas/tables.yml").read_text(encoding="utf-8"))["tables"]
        policies = yaml.safe_load(Path("configs/oireachtas/write_policies.yml").read_text(encoding="utf-8"))["tables"]
        table = tables["enrichment_speech_issue_labels"]
        self.assertEqual(table["primary_key"], ["speech_id"])
        self.assertEqual(table["status"], "in_progress")
        self.assertEqual(policies["enrichment_speech_issue_labels"]["write_strategy"], "rebuild")

    def test_repository_wide_openai_requirement_is_not_major_bumped(self) -> None:
        text = Path("requirements.txt").read_text(encoding="utf-8")
        self.assertIn("openai>=1.99.2", text)
        self.assertNotIn("openai>=3.6.0", text)


if __name__ == "__main__":
    unittest.main()
