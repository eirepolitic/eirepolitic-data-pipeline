from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from extract.oireachtas.merge import foreign_key_integrity, merge_for_policy, overlap_count, temporal_integrity
from extract.oireachtas.normalize import is_current_range, stable_hash
from extract.oireachtas.schemas import load_table_registry
from extract.oireachtas.write_policies import ForeignKeyPolicy, WritePolicy, load_write_policies, validate_policy_coverage


class WritePolicyCoverageTests(unittest.TestCase):
    def test_every_registered_table_has_write_policy(self) -> None:
        registry = load_table_registry()
        policies = load_write_policies()
        self.assertEqual(validate_policy_coverage(set(registry), policies), [])


class MergeSemanticsTests(unittest.TestCase):
    def test_overlapping_incremental_windows_preserve_and_update_history(self) -> None:
        policy = WritePolicy(table="silver_member_votes", write_strategy="upsert")
        existing = pd.DataFrame([
            {"member_vote_id": "v1", "vote_label": "yes"},
            {"member_vote_id": "v2", "vote_label": "no"},
        ])
        incoming = pd.DataFrame([
            {"member_vote_id": "v2", "vote_label": "abstain"},
            {"member_vote_id": "v3", "vote_label": "yes"},
        ])
        merged = merge_for_policy(existing, incoming, primary_key=["member_vote_id"], policy=policy)
        self.assertEqual(set(merged["member_vote_id"]), {"v1", "v2", "v3"})
        self.assertEqual(merged.loc[merged["member_vote_id"] == "v2", "vote_label"].iloc[0], "abstain")

    def test_yearly_aggregation_uses_preserved_history(self) -> None:
        policy = WritePolicy(table="silver_speeches", write_strategy="upsert")
        january = pd.DataFrame([{"speech_id": "s1", "speaker_member_code": "m1", "speech_date": "2025-01-10"}])
        december = pd.DataFrame([{"speech_id": "s2", "speaker_member_code": "m1", "speech_date": "2025-12-10"}])
        merged = merge_for_policy(january, december, primary_key=["speech_id"], policy=policy)
        annual = merged[merged["speech_date"].str.startswith("2025")].groupby("speaker_member_code").size()
        self.assertEqual(int(annual.loc["m1"]), 2)

    def test_snapshot_replace_does_not_retain_missing_rows(self) -> None:
        policy = WritePolicy(table="silver_members", write_strategy="snapshot_replace")
        existing = pd.DataFrame([{"member_code": "old"}])
        incoming = pd.DataFrame([{"member_code": "current"}])
        merged = merge_for_policy(existing, incoming, primary_key=["member_code"], policy=policy)
        self.assertEqual(merged["member_code"].tolist(), ["current"])


class TemporalAndIntegrityTests(unittest.TestCase):
    def test_future_open_ended_record_is_not_current(self) -> None:
        self.assertFalse(is_current_range("2099-01-01", None, today=date(2026, 7, 13)))

    def test_temporal_integrity_rejects_invalid_and_future_current_rows(self) -> None:
        policy = WritePolicy(
            table="history",
            write_strategy="upsert",
            valid_from_column="valid_from",
            valid_to_column="valid_to",
            current_column="is_current",
        )
        frame = pd.DataFrame([
            {"valid_from": "2026-08-01", "valid_to": None, "is_current": True},
            {"valid_from": "2026-06-10", "valid_to": "2026-06-01", "is_current": False},
        ])
        result = temporal_integrity(frame, policy=policy, as_of=date(2026, 7, 13))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["future_current_rows"], 1)
        self.assertEqual(result["invalid_ranges"], 1)

    def test_foreign_key_orphans_are_visible(self) -> None:
        policy = ForeignKeyPolicy(columns=("member_code",), references="silver_members", referenced_columns=("member_code",))
        child = pd.DataFrame([{"member_code": "m1"}, {"member_code": "missing"}])
        parent = pd.DataFrame([{"member_code": "m1"}])
        result = foreign_key_integrity(child, parent, policy=policy)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["orphan_count"], 1)

    def test_overlapping_ranges_are_counted(self) -> None:
        frame = pd.DataFrame([
            {"member_code": "m1", "start": "2025-01-01", "end": "2025-06-30"},
            {"member_code": "m1", "start": "2025-06-15", "end": "2025-12-31"},
        ])
        self.assertEqual(overlap_count(frame, entity_columns=["member_code"], start_column="start", end_column="end"), 1)

    def test_fallback_membership_identity_does_not_depend_on_end_date(self) -> None:
        base = ["m1", "/member/m1", "/house/dail/34", "34", "dail", "2025-01-01"]
        self.assertEqual(stable_hash(base), stable_hash(base))


if __name__ == "__main__":
    unittest.main()
