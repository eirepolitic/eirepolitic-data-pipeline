from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from extract.oireachtas.merge import foreign_key_integrity, merge_for_policy, overlap_count, temporal_integrity
from extract.oireachtas.normalize import is_current_range
from extract.oireachtas.schemas import load_table_registry
from extract.oireachtas.table_member_constituencies import _normalise_constituency_row
from extract.oireachtas.table_member_memberships import _normalise_membership_row
from extract.oireachtas.table_member_offices import _normalise_office_row
from extract.oireachtas.table_member_parties import _normalise_party_row
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


class StableHistoryIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.member = {"memberCode": "m1", "uri": "/member/m1"}
        self.open_membership = {
            "house": {"uri": "/house/dail/34", "houseNo": "34", "houseCode": "dail"},
            "dateRange": {"start": "2025-01-01"},
        }
        self.closed_membership = {
            **self.open_membership,
            "dateRange": {"start": "2025-01-01", "end": "2026-01-01"},
        }

    def test_membership_id_ignores_later_end_date(self) -> None:
        open_row = _normalise_membership_row(self.member, self.open_membership, snapshot_date="2026-01-01")
        closed_row = _normalise_membership_row(self.member, self.closed_membership, snapshot_date="2026-01-02")
        self.assertEqual(open_row["membership_id"], closed_row["membership_id"])

    def test_party_id_ignores_later_end_date(self) -> None:
        open_party = {"showAs": "Example Party", "dateRange": {"start": "2025-01-01"}}
        closed_party = {"showAs": "Example Party", "dateRange": {"start": "2025-01-01", "end": "2026-01-01"}}
        open_row = _normalise_party_row(self.member, self.open_membership, open_party, snapshot_date="2026-01-01")
        closed_row = _normalise_party_row(self.member, self.closed_membership, closed_party, snapshot_date="2026-01-02")
        self.assertEqual(open_row["member_party_id"], closed_row["member_party_id"])

    def test_constituency_id_ignores_later_end_date(self) -> None:
        open_representation = {"showAs": "Example", "dateRange": {"start": "2025-01-01"}}
        closed_representation = {"showAs": "Example", "dateRange": {"start": "2025-01-01", "end": "2026-01-01"}}
        open_row = _normalise_constituency_row(self.member, self.open_membership, open_representation, snapshot_date="2026-01-01")
        closed_row = _normalise_constituency_row(self.member, self.closed_membership, closed_representation, snapshot_date="2026-01-02")
        self.assertEqual(open_row["member_constituency_id"], closed_row["member_constituency_id"])

    def test_office_id_ignores_later_end_date(self) -> None:
        open_office = {"showAs": "Minister", "dateRange": {"start": "2025-01-01"}}
        closed_office = {"showAs": "Minister", "dateRange": {"start": "2025-01-01", "end": "2026-01-01"}}
        open_row = _normalise_office_row(self.member, self.open_membership, open_office, snapshot_date="2026-01-01")
        closed_row = _normalise_office_row(self.member, self.closed_membership, closed_office, snapshot_date="2026-01-02")
        self.assertEqual(open_row["member_office_id"], closed_row["member_office_id"])


if __name__ == "__main__":
    unittest.main()
