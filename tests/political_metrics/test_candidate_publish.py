import unittest

import pandas as pd

from extract.oireachtas.batch import PREVIOUS_POINTER_KEY, PRODUCTION_POINTER_KEY
from political_metrics.candidate_publish import publish_dataset_to_candidate
from political_metrics.materialize import DatasetContract
from political_metrics.results import metric_result_row, metric_results_frame


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = {"Body": Body, "ContentType": ContentType}
        return {}

    def head_object(self, *, Bucket, Key):
        item = self.objects[(Bucket, Key)]
        body = item["Body"]
        return {"ContentLength": len(body), "ETag": '"fake-etag"'}


class CandidatePublishTests(unittest.TestCase):
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
            formats=["csv"],
            cadence="completed_month",
        )

    def _frame(self, batch_id="batch-1"):
        return metric_results_frame([
            metric_result_row(
                metric_id="national_speech_count",
                metric_version=1,
                period_start="2026-07-01",
                period_end="2026-07-31",
                grain="national",
                entity_id="dail",
                entity_name="Dáil",
                value=100,
                numerator=100,
                denominator=None,
                output_unit="count",
                source_batch_id=batch_id,
                contract_version=1,
            )
        ])

    def test_publish_is_batch_scoped_and_records_entry(self):
        s3 = FakeS3()
        result = publish_dataset_to_candidate(
            s3,
            bucket="bucket",
            batch_id="batch-1",
            frame=self._frame(),
            dataset=self._dataset(),
            contract_version=1,
            source_batch_id="batch-1",
        )
        keys = {key for bucket, key in s3.objects if bucket == "bucket"}
        self.assertIn(
            "processed/oireachtas_unified/batches/batch-1/metrics/completed_month/"
            "monthly_metric_results/csv/monthly_metric_results.csv",
            keys,
        )
        self.assertIn(
            "processed/oireachtas_unified/batches/batch-1/entries/"
            "political_metrics_monthly_metric_results.json",
            keys,
        )
        self.assertNotIn(PRODUCTION_POINTER_KEY, keys)
        self.assertNotIn(PREVIOUS_POINTER_KEY, keys)
        self.assertEqual(result["row_count"], 1)

    def test_source_batch_must_equal_candidate_batch(self):
        with self.assertRaises(ValueError):
            publish_dataset_to_candidate(
                FakeS3(),
                bucket="bucket",
                batch_id="batch-2",
                frame=self._frame("batch-1"),
                dataset=self._dataset(),
                contract_version=1,
                source_batch_id="batch-1",
            )


if __name__ == "__main__":
    unittest.main()
