from __future__ import annotations

import unittest

import pandas as pd

from process.oireachtas_speech_issue_classifier import (
    ISSUE_CATEGORIES,
    build_compat_output,
    build_legacy_lookup,
    classification_schema,
    execute_model_classification,
    prepare_classification_plan,
    text_hash,
    validate_enrichment,
)


def silver_row(
    speech_id: str,
    text: str,
    *,
    date: str = "2026-02-26",
    order: str = "1",
    speaker: str = "Jane Doe",
    member_code: str = "m1",
) -> dict[str, object]:
    return {
        "speech_id": speech_id,
        "debate_date": date,
        "speech_order": order,
        "speaker_name": speaker,
        "speaker_member_code": member_code,
        "speech_text": text,
        "speech_text_hash": text_hash(text),
        "word_count": len(text.split()),
    }


class MigrationTests(unittest.TestCase):
    def test_exact_legacy_match_is_migrated_to_unified_speech_id(self) -> None:
        text = "Housing supply and planning reform are central to increasing the number of homes available across the country today."
        silver = pd.DataFrame([silver_row("speech:new-id", text, order="7")])
        legacy = pd.DataFrame([
            {
                "Debate Date": "2026-02-26",
                "Speech Order": "7",
                "Speaker Name": "Jane Doe",
                "Speech Text": text,
                "PoliticalIssues": "Housing and Community Development",
            }
        ])

        plan = prepare_classification_plan(silver, legacy=legacy)

        self.assertEqual(plan.rows.iloc[0]["speech_id"], "speech:new-id")
        self.assertEqual(plan.rows.iloc[0]["issue_label"], "Housing and Community Development")
        self.assertEqual(plan.rows.iloc[0]["issue_label_source"], "legacy_migration_exact")
        self.assertEqual(plan.stats["migrated_legacy"], 1)
        self.assertEqual(plan.stats["pending_model"], 0)

    def test_existing_unified_label_reused_only_when_text_hash_matches(self) -> None:
        original = "This sufficiently long speech discusses public transport services and rail capacity across several regions in detail today."
        changed = original + " Additional material changes the stored source text."
        silver = pd.DataFrame([silver_row("speech:1", changed)])
        existing = pd.DataFrame([
            {
                "speech_id": "speech:1",
                "source_speech_text_hash": text_hash(original),
                "issue_label": "Transportation",
                "classification_status": "classified",
                "model_name": "gpt-5.6-luna",
                "issue_label_source": "openai_model",
            }
        ])

        plan = prepare_classification_plan(silver, existing=existing)

        self.assertEqual(plan.rows.iloc[0]["classification_status"], "pending")
        self.assertEqual(plan.rows.iloc[0]["issue_label"], "")
        self.assertEqual(plan.stats["existing_hash_mismatch"], 1)
        self.assertEqual(plan.stats["pending_model"], 1)

    def test_existing_unified_label_with_matching_hash_is_reused(self) -> None:
        text = "This sufficiently long speech discusses health service capacity, waiting lists, staffing and patient access across the system today."
        silver = pd.DataFrame([silver_row("speech:1", text)])
        existing = pd.DataFrame([
            {
                "speech_id": "speech:1",
                "source_speech_text_hash": text_hash(text),
                "issue_label": "Health",
                "classification_status": "classified",
                "model_name": "gpt-5.6-luna",
                "issue_label_source": "openai_model",
            }
        ])

        plan = prepare_classification_plan(silver, existing=existing)

        self.assertEqual(plan.rows.iloc[0]["issue_label"], "Health")
        self.assertEqual(plan.rows.iloc[0]["issue_label_source"], "existing_unified_enrichment")
        self.assertEqual(plan.stats["reused_existing"], 1)
        self.assertEqual(plan.stats["pending_model"], 0)

    def test_short_speech_becomes_none_without_model_call(self) -> None:
        text = "I thank the Minister for that answer."
        silver = pd.DataFrame([silver_row("speech:short", text)])

        plan = prepare_classification_plan(silver)

        self.assertEqual(plan.rows.iloc[0]["issue_label"], "NONE")
        self.assertEqual(plan.rows.iloc[0]["classification_status"], "skipped_short_text")
        self.assertEqual(plan.rows.iloc[0]["model_name"], "rule")
        self.assertEqual(plan.stats["short_text_none"], 1)

    def test_ambiguous_legacy_key_is_not_reused(self) -> None:
        text = "This sufficiently long speech covers a clearly identifiable policy topic with enough words to require classification by the model later."
        legacy = pd.DataFrame([
            {"Debate Date": "2026-02-26", "Speech Order": "1", "Speaker Name": "Jane Doe", "Speech Text": text, "PoliticalIssues": "Health"},
            {"Debate Date": "2026-02-26", "Speech Order": "1", "Speaker Name": "Jane Doe", "Speech Text": text, "PoliticalIssues": "Education"},
        ])
        lookup, stats = build_legacy_lookup(legacy)
        self.assertEqual(lookup, {})
        self.assertEqual(stats["legacy_ambiguous_keys"], 1)


class ClassificationTests(unittest.TestCase):
    def test_structured_output_schema_is_closed_enum(self) -> None:
        schema = classification_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["issue_label"]["enum"], ISSUE_CATEGORIES)

    def test_model_execution_updates_only_requested_pending_rows(self) -> None:
        text1 = "This is a sufficiently long speech about schools, teachers, pupils and education funding across the country for the coming year."
        text2 = "This is a sufficiently long speech about buses, trains, transport capacity and public services across the country for the coming year."
        silver = pd.DataFrame([silver_row("speech:1", text1), silver_row("speech:2", text2, order="2")])
        plan = prepare_classification_plan(silver)

        output, stats = execute_model_classification(
            plan,
            classify_fn=lambda _: "Education",
            model_name="gpt-5.6-luna",
            max_rows=1,
        )

        self.assertEqual(output.iloc[0]["issue_label"], "Education")
        self.assertEqual(output.iloc[0]["model_name"], "gpt-5.6-luna")
        self.assertEqual(output.iloc[1]["classification_status"], "pending")
        self.assertEqual(stats["model_attempted"], 1)
        self.assertEqual(stats["model_remaining_pending"], 1)

    def test_compat_output_preserves_unified_member_code(self) -> None:
        text = "This is a sufficiently long speech about schools, teachers, pupils and education funding across the country for the coming year."
        silver = pd.DataFrame([silver_row("speech:1", text, member_code="member-123")])
        plan = prepare_classification_plan(silver)
        enriched, _ = execute_model_classification(plan, classify_fn=lambda _: "Education", model_name="gpt-5.6-luna")

        compat = build_compat_output(silver, enriched)

        self.assertEqual(compat.iloc[0]["member_code"], "member-123")
        self.assertEqual(compat.iloc[0]["PoliticalIssues"], "Education")
        self.assertEqual(compat.iloc[0]["speech_id"], "speech:1")

    def test_dq_allows_pending_rows_for_dry_run_inventory(self) -> None:
        text = "This is a sufficiently long speech with a policy subject that remains pending until an explicit live classification test is requested."
        silver = pd.DataFrame([silver_row("speech:1", text)])
        plan = prepare_classification_plan(silver)
        dq = validate_enrichment(silver, plan.rows)
        self.assertEqual(dq["dq_status"], "pass")
        self.assertEqual(dq["pending_rows"], 1)


if __name__ == "__main__":
    unittest.main()
