import unittest

from extract.oireachtas.batch import batch_key_for_production_key


class MetricBatchPathTests(unittest.TestCase):
    def test_daily_metric_path_maps_inside_same_batch(self):
        key = (
            "processed/oireachtas_unified/latest/metrics/daily/"
            "daily_activity_components/csv/daily_activity_components.csv"
        )
        result = batch_key_for_production_key(key, "batch-123")
        self.assertEqual(
            result,
            "processed/oireachtas_unified/batches/batch-123/metrics/daily/"
            "daily_activity_components/csv/daily_activity_components.csv",
        )

    def test_monthly_metric_path_maps_inside_same_batch(self):
        key = (
            "processed/oireachtas_unified/latest/metrics/completed_month/"
            "monthly_metric_results/parquet/monthly_metric_results.parquet"
        )
        result = batch_key_for_production_key(key, "batch-123")
        self.assertEqual(
            result,
            "processed/oireachtas_unified/batches/batch-123/metrics/completed_month/"
            "monthly_metric_results/parquet/monthly_metric_results.parquet",
        )

    def test_metric_filename_must_match_dataset(self):
        key = (
            "processed/oireachtas_unified/latest/metrics/daily/"
            "daily_activity_components/csv/wrong.csv"
        )
        with self.assertRaises(ValueError):
            batch_key_for_production_key(key, "batch-123")

    def test_metric_format_and_extension_must_match(self):
        key = (
            "processed/oireachtas_unified/latest/metrics/daily/"
            "daily_activity_components/csv/daily_activity_components.parquet"
        )
        with self.assertRaises(ValueError):
            batch_key_for_production_key(key, "batch-123")


if __name__ == "__main__":
    unittest.main()
