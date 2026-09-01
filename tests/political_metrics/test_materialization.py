import tempfile
import unittest
from pathlib import Path

import pandas as pd

from political_metrics.materialize import (
    DatasetContract,
    validate_materialized_frame,
    write_materialized_dataset,
)
from political_metrics.results import metric_result_row, metric_results_frame


class MaterializationTests(unittest.TestCase):
    def _dataset(self):
        return DatasetContract(
            name="monthly_metric_results",
            columns=[
                "metric_id", "metric_version", "period_type", "period_start", "period_end",
                "grain", "entity_id", "entity_name", "dimension_name", "dimension_value",
                "value", "numerator", "denominator", "output_unit", "reliability_status",
                "public_use_status", "warning_code", "source_batch_id", "calculated_at_utc",
                "contract_version",
            ],
            primary_key=[
                "metric_id", "metric_version", "period_start", "period_end", "grain",
                "entity_id", "dimension_name", "dimension_value",
            ],
            formats=["csv", "parquet"],
            cadence="completed_month",
        )

    def _frame(self):
        row = metric_result_row(
            metric_id="member_speech_count",
            metric_version=1,
            period_start="2026-07-01",
            period_end="2026-07-31",
            grain="member",
            entity_id="m1",
            entity_name="Member One",
            value=10,
            numerator=10,
            denominator=None,
            output_unit="count",
            source_batch_id="batch-1",
            contract_version=1,
        )
        return metric_results_frame([row])

    def test_valid_frame_passes(self):
        errors = validate_materialized_frame(
            self._frame(), self._dataset(), expected_source_batch_id="batch-1"
        )
        self.assertEqual(errors, [])

    def test_duplicate_primary_key_fails(self):
        frame = pd.concat([self._frame(), self._frame()], ignore_index=True)
        errors = validate_materialized_frame(frame, self._dataset())
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_source_batch_mismatch_fails(self):
        errors = validate_materialized_frame(
            self._frame(), self._dataset(), expected_source_batch_id="batch-2"
        )
        self.assertTrue(any("source_batch_id mismatch" in error for error in errors))

    def test_invalid_period_fails(self):
        frame = self._frame()
        frame.loc[0, "period_start"] = "2026-08-01"
        frame.loc[0, "period_end"] = "2026-07-31"
        errors = validate_materialized_frame(frame, self._dataset())
        self.assertTrue(any("invalid period" in error for error in errors))

    def test_dimension_must_be_explicit(self):
        with self.assertRaises(ValueError):
            metric_result_row(
                metric_id="x",
                metric_version=1,
                period_start="2026-07-01",
                period_end="2026-07-31",
                grain="member",
                entity_id="m1",
                entity_name="M1",
                value=1,
                numerator=1,
                denominator=None,
                output_unit="count",
                source_batch_id="batch-1",
                contract_version=1,
                dimension_name="",
                dimension_value="",
            )

    def test_writer_creates_csv_parquet_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_materialized_dataset(
                self._frame(),
                dataset=self._dataset(),
                output_root=tmp,
                source_batch_id="batch-1",
                contract_version=1,
            )
            self.assertEqual(manifest["row_count"], 1)
            root = Path(tmp) / "completed_month" / "monthly_metric_results"
            self.assertTrue((root / "monthly_metric_results.csv").exists())
            self.assertTrue((root / "monthly_metric_results.parquet").exists())
            self.assertTrue((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
