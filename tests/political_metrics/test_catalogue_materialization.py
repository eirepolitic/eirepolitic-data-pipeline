import unittest

from political_metrics.catalogue import load_catalogue_metric_ids
from political_metrics.monthly_results import MATERIALIZED_METRIC_IDS


class CatalogueMaterializationTests(unittest.TestCase):
    def test_every_catalogue_metric_is_supported_by_monthly_materialization(self):
        catalogue_ids = load_catalogue_metric_ids()
        self.assertEqual(catalogue_ids, MATERIALIZED_METRIC_IDS)


if __name__ == "__main__":
    unittest.main()
