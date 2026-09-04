from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from extract.polling.ipi import (
    INDICATOR_COLUMNS,
    PreparedIngestion,
    RAW_POLL_COLUMNS,
    UpstreamSnapshot,
    ValidationError,
    _reuse_confirmed,
    publish_ingestion,
    validate_and_normalize_indicator,
    validate_and_normalize_polls,
)


def _csv(columns, rows) -> bytes:
    frame = pd.DataFrame(rows, columns=columns)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _raw_row(**overrides):
    row = {column: "" for column in RAW_POLL_COLUMNS}
    row.update(
        {
            "date": "2026-08-02",
            "date_start": "2026-07-29",
            "date_end": "2026-07-31",
            "date_middle": "2026-07-30",
            "pollster": "Example Pollster",
            "sample_size": "1000",
            "FF": "20",
            "FG": "18.5",
            "SF": "25",
            "LAB": "5",
            "GP": "3",
            "PD": "",
            "WP": "",
            "DL": "",
            "SPBP": "4",
            "RENUA": "",
            "SD": "6",
            "AU": "2",
            "II": "3",
            "IND_OTH_IT": "",
            "PREV_INDOTH_II": "",
            "PREV_II": "",
            "OTH_IND": "13.5",
        }
    )
    row.update(overrides)
    return row


def _indicator_row(date="2026-07-31", cycle="2024", **overrides):
    row = {column: "" for column in INDICATOR_COLUMNS}
    row["date"] = date
    row["cycle"] = cycle
    for column in INDICATOR_COLUMNS[2:]:
        if column.endswith("_lo"):
            row[column] = "0.10"
        elif column.endswith("_hi"):
            row[column] = "0.30"
        else:
            row[column] = "0.20"
    row.update(overrides)
    return row


class PollValidationTests(unittest.TestCase):
    def test_raw_poll_duplicate_and_negative_value_are_flagged_not_corrected(self) -> None:
        row = _raw_row(OTH_IND="-1")
        body = _csv(RAW_POLL_COLUMNS, [row, row])
        normalized, summary = validate_and_normalize_polls(
            body,
            commit_sha="a" * 40,
            branch="main",
            source_url="https://example.test/data_polls.csv",
            retrieved_at="2026-09-03T00:00:00+00:00",
        )
        self.assertEqual(summary["exact_duplicate_extra_rows"], 1)
        self.assertEqual(summary["negative_party_cells"], 2)
        self.assertEqual(float(normalized.loc[0, "OTH_IND"]), -1.0)
        self.assertIn("exact_duplicate_source_row", normalized.loc[0, "quality_flags"])
        self.assertIn("negative_source_value:OTH_IND", normalized.loc[0, "quality_flags"])
        self.assertEqual(set(normalized["value_unit"]), {"percentage_points"})

    def test_raw_poll_fieldwork_anomalies_are_visible(self) -> None:
        body = _csv(
            RAW_POLL_COLUMNS,
            [
                _raw_row(
                    date="2025-06-11",
                    date_start="2026-06-07",
                    date_end="2025-06-08",
                    date_middle="2025-12-07",
                )
            ],
        )
        normalized, summary = validate_and_normalize_polls(
            body,
            commit_sha="b" * 40,
            branch="main",
            source_url="https://example.test/data_polls.csv",
            retrieved_at="2026-09-03T00:00:00+00:00",
        )
        flags = normalized.loc[0, "quality_flags"]
        self.assertIn("fieldwork_start_after_end", flags)
        self.assertIn("fieldwork_middle_outside_range", flags)
        self.assertEqual(summary["fieldwork_start_after_end_rows"], 1)

    def test_raw_poll_value_above_100_blocks_publication(self) -> None:
        body = _csv(RAW_POLL_COLUMNS, [_raw_row(FF="101")])
        with self.assertRaises(ValidationError):
            validate_and_normalize_polls(
                body,
                commit_sha="c" * 40,
                branch="main",
                source_url="https://example.test/data_polls.csv",
                retrieved_at="2026-09-03T00:00:00+00:00",
            )

    def test_schema_change_blocks_publication(self) -> None:
        columns = list(RAW_POLL_COLUMNS[:-1])
        body = _csv(columns, [{column: _raw_row().get(column, "") for column in columns}])
        with self.assertRaises(ValidationError):
            validate_and_normalize_polls(
                body,
                commit_sha="d" * 40,
                branch="main",
                source_url="https://example.test/data_polls.csv",
                retrieved_at="2026-09-03T00:00:00+00:00",
            )


class IndicatorValidationTests(unittest.TestCase):
    def test_cycle_boundary_duplicate_calendar_date_is_allowed(self) -> None:
        body = _csv(
            INDICATOR_COLUMNS,
            [
                _indicator_row(date="2016-02-26", cycle="2011"),
                _indicator_row(date="2016-02-26", cycle="2016"),
            ],
        )
        normalized, summary = validate_and_normalize_indicator(
            body,
            commit_sha="e" * 40,
            branch="main",
            source_url="https://example.test/data_pollingindicator.csv",
            retrieved_at="2026-09-03T00:00:00+00:00",
        )
        self.assertEqual(summary["duplicate_calendar_dates"], ["2016-02-26"])
        self.assertTrue(normalized["quality_flags"].str.contains("cycle_boundary_duplicate_calendar_date").all())
        self.assertEqual(set(normalized["value_unit"]), {"proportion"})

    def test_duplicate_date_cycle_key_blocks_publication(self) -> None:
        row = _indicator_row()
        body = _csv(INDICATOR_COLUMNS, [row, row])
        with self.assertRaises(ValidationError):
            validate_and_normalize_indicator(
                body,
                commit_sha="f" * 40,
                branch="main",
                source_url="https://example.test/data_pollingindicator.csv",
                retrieved_at="2026-09-03T00:00:00+00:00",
            )

    def test_invalid_interval_blocks_publication(self) -> None:
        body = _csv(INDICATOR_COLUMNS, [_indicator_row(FF="0.05", FF_lo="0.10", FF_hi="0.30")])
        with self.assertRaises(ValidationError):
            validate_and_normalize_indicator(
                body,
                commit_sha="1" * 40,
                branch="main",
                source_url="https://example.test/data_pollingindicator.csv",
                retrieved_at="2026-09-03T00:00:00+00:00",
            )


class PublicationTests(unittest.TestCase):
    def test_latest_manifest_is_written_last(self) -> None:
        class FakeS3:
            def __init__(self):
                self.calls = []

            def put_object(self, **kwargs):
                self.calls.append(kwargs)

        snapshot = UpstreamSnapshot(
            branch="main",
            commit_sha="2" * 40,
            retrieved_at="2026-09-03T00:00:00+00:00",
            files={"data_polls.csv": b"polls", "data_pollingindicator.csv": b"indicator"},
            urls={"data_polls.csv": "https://example.test/polls", "data_pollingindicator.csv": "https://example.test/indicator"},
        )
        prepared = PreparedIngestion(
            snapshot=snapshot,
            polls=pd.DataFrame(),
            polling_indicator=pd.DataFrame(),
            artifacts={
                "polls_csv": b"pcsv",
                "polls_parquet": b"ppq",
                "indicator_csv": b"icsv",
                "indicator_parquet": b"ipq",
            },
            manifest={"dataset": "irish_polling_indicator"},
        )
        s3 = FakeS3()
        keys = publish_ingestion(prepared, s3=s3, bucket="test-bucket")
        self.assertEqual(s3.calls[-1]["Key"], keys["latest_manifest"])
        self.assertEqual(len(s3.calls), 12)
        self.assertIn("/by_commit/" + ("2" * 40) + "/", keys["raw_polls"])

    def test_publish_gate_requires_explicit_reuse_confirmation(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_reuse_confirmed())
        with patch.dict(os.environ, {"IPI_REUSE_CONFIRMED": "true"}, clear=True):
            self.assertTrue(_reuse_confirmed())


if __name__ == "__main__":
    unittest.main()
