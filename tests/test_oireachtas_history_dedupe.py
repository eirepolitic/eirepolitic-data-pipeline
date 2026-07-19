from __future__ import annotations

import unittest

from extract.oireachtas.history_dedupe import business_key_unique, dedupe_history_rows


class HistoryDedupeTests(unittest.TestCase):
    def test_exact_duplicate_business_rows_are_collapsed(self) -> None:
        rows = [
            {
                "member_party_id": "generated:1",
                "member_code": "M1",
                "party_uri": "party:one",
                "party_start": "2024-11-29",
                "party_end": "",
                "party_name": "Party One",
                "is_current": True,
            },
            {
                "member_party_id": "generated:2",
                "member_code": "M1",
                "party_uri": "party:one",
                "party_start": "2024-11-29",
                "party_end": "",
                "party_name": "Party One",
                "is_current": True,
            },
        ]

        result = dedupe_history_rows(
            rows,
            business_key=("member_code", "party_uri", "party_start", "party_end"),
            compared_fields=("party_name", "is_current"),
        )

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.duplicate_rows_removed, 1)
        self.assertEqual(result.conflicting_keys, [])
        self.assertTrue(
            business_key_unique(
                result.rows,
                business_key=("member_code", "party_uri", "party_start", "party_end"),
            )
        )

    def test_conflicting_duplicates_are_reported(self) -> None:
        rows = [
            {
                "member_code": "M1",
                "constituency_uri": "constituency:one",
                "represent_start": "2024-11-29",
                "represent_end": "",
                "constituency_name": "Constituency One",
                "is_current": True,
            },
            {
                "member_code": "M1",
                "constituency_uri": "constituency:one",
                "represent_start": "2024-11-29",
                "represent_end": "",
                "constituency_name": "Different Name",
                "is_current": True,
            },
        ]

        result = dedupe_history_rows(
            rows,
            business_key=(
                "member_code",
                "constituency_uri",
                "represent_start",
                "represent_end",
            ),
            compared_fields=("constituency_name", "is_current"),
        )

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.duplicate_rows_removed, 1)
        self.assertEqual(len(result.conflicting_keys), 1)
        self.assertIn("constituency_name", result.conflicting_keys[0]["differences"])


if __name__ == "__main__":
    unittest.main()
