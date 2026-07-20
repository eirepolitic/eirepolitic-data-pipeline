from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import pandas as pd

from extract.oireachtas.table_control_table_manifests import _actual_candidate_counts


class ControlManifestCountTests(unittest.TestCase):
    def test_actual_candidate_counts_use_merged_objects(self) -> None:
        csv_bytes = pd.DataFrame([{"id": "1"}, {"id": "2"}, {"id": "3"}]).to_csv(index=False).encode("utf-8")
        parquet_buffer = io.BytesIO()
        pd.DataFrame([{"id": "1"}, {"id": "2"}, {"id": "3"}]).to_parquet(parquet_buffer, index=False)

        def fake_get_bytes(_s3, *, bucket: str, key: str) -> bytes:
            self.assertEqual(bucket, "bucket")
            return csv_bytes if key.endswith(".csv") else parquet_buffer.getvalue()

        with patch("extract.oireachtas.table_control_table_manifests.get_bytes", side_effect=fake_get_bytes):
            result = _actual_candidate_counts(
                object(),
                bucket="bucket",
                csv_key="processed/oireachtas_unified/latest/csv/sample.csv",
                parquet_key="processed/oireachtas_unified/latest/parquet/sample.parquet",
            )

        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["csv_rows"], 3)
        self.assertEqual(result["parquet_rows"], 3)

    def test_csv_parquet_mismatch_fails(self) -> None:
        csv_bytes = pd.DataFrame([{"id": "1"}, {"id": "2"}]).to_csv(index=False).encode("utf-8")
        parquet_buffer = io.BytesIO()
        pd.DataFrame([{"id": "1"}]).to_parquet(parquet_buffer, index=False)

        def fake_get_bytes(_s3, *, bucket: str, key: str) -> bytes:
            return csv_bytes if key.endswith(".csv") else parquet_buffer.getvalue()

        with patch("extract.oireachtas.table_control_table_manifests.get_bytes", side_effect=fake_get_bytes):
            with self.assertRaisesRegex(ValueError, "CSV/Parquet row mismatch"):
                _actual_candidate_counts(
                    object(),
                    bucket="bucket",
                    csv_key="sample.csv",
                    parquet_key="sample.parquet",
                )


if __name__ == "__main__":
    unittest.main()
