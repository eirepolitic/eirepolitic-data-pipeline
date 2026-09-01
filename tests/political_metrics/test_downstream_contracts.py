import unittest
from pathlib import Path

import yaml

from extract.oireachtas.batch import batch_key_for_production_key
from political_metrics.materialize import get_dataset_contract, load_materialization_contract


REPO_ROOT = Path(__file__).resolve().parents[2]


class DownstreamContractTests(unittest.TestCase):
    def setUp(self):
        self.downstream = yaml.safe_load(
            (REPO_ROOT / "configs/political_metrics/downstream_contracts.yml").read_text(encoding="utf-8")
        )
        self.materialization = load_materialization_contract(
            REPO_ROOT / "configs/political_metrics/materialization.yml"
        )

    def test_all_documented_logical_keys_resolve_inside_batch(self):
        for dataset, formats in self.downstream["logical_keys"].items():
            for fmt, logical_key in formats.items():
                resolved = batch_key_for_production_key(logical_key, "batch-test")
                self.assertIn("processed/oireachtas_unified/batches/batch-test/metrics/", resolved)
                self.assertTrue(resolved.endswith(f"/{dataset}.{fmt}"))

    def test_monthly_consumer_primary_key_matches_materialization_contract(self):
        dataset = get_dataset_contract(self.materialization, "monthly_metric_results")
        self.assertEqual(self.downstream["primary_key"], dataset.primary_key)


if __name__ == "__main__":
    unittest.main()
