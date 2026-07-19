from __future__ import annotations

import unittest

import pandas as pd

from extract.oireachtas.merge import merge_for_policy
from extract.oireachtas.write_policies import WritePolicy


class BusinessKeyMergeTests(unittest.TestCase):
    def test_upsert_removes_legacy_duplicate_ids_by_business_key(self) -> None:
        existing = pd.DataFrame(
            [
                {
                    "member_party_id": "old-id-1",
                    "member_code": "M1",
                    "party_uri": "party:one",
                    "party_start": "2024-11-29",
                    "party_end": "",
                    "party_name": "Party One",
                },
                {
                    "member_party_id": "old-id-2",
                    "member_code": "M1",
                    "party_uri": "party:one",
                    "party_start": "2024-11-29",
                    "party_end": "",
                    "party_name": "Party One",
                },
                {
                    "member_party_id": "history-id",
                    "member_code": "M1",
                    "party_uri": "party:old",
                    "party_start": "2020-01-01",
                    "party_end": "2024-11-28",
                    "party_name": "Old Party",
                },
            ]
        )
        incoming = pd.DataFrame(
            [
                {
                    "member_party_id": "new-stable-id",
                    "member_code": "M1",
                    "party_uri": "party:one",
                    "party_start": "2024-11-29",
                    "party_end": "",
                    "party_name": "Party One",
                }
            ]
        )
        policy = WritePolicy(
            table="silver_member_parties",
            write_strategy="upsert",
            business_key_columns=("member_code", "party_uri", "party_start", "party_end"),
        )

        merged = merge_for_policy(
            existing,
            incoming,
            primary_key=["member_party_id"],
            policy=policy,
        )

        self.assertEqual(len(merged), 2)
        current = merged[merged["party_uri"] == "party:one"]
        self.assertEqual(len(current), 1)
        self.assertEqual(current.iloc[0]["member_party_id"], "new-stable-id")
        self.assertEqual(len(merged[merged["party_uri"] == "party:old"]), 1)

    def test_policy_without_business_key_keeps_distinct_primary_keys(self) -> None:
        existing = pd.DataFrame([{"id": "1", "value": "same"}])
        incoming = pd.DataFrame([{"id": "2", "value": "same"}])
        policy = WritePolicy(table="sample", write_strategy="upsert")

        merged = merge_for_policy(existing, incoming, primary_key=["id"], policy=policy)

        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
