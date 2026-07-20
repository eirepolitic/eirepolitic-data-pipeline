from __future__ import annotations

from unittest import TestCase

from instagram.factory.recurring import evaluate_readiness


class InstagramFactoryPhase5Test(TestCase):
    def test_local_fixture_is_ready_without_previous_manifest(self) -> None:
        report = evaluate_readiness(data_source="local")
        self.assertTrue(report["ready"])
        self.assertEqual(report["source_batch_id"], "local-fixture")
        self.assertGreater(report["expected_item_count"], 0)

    def test_duplicate_source_batch_is_not_ready(self) -> None:
        report = evaluate_readiness(
            data_source="local",
            latest_manifest={"source_batch_id": "local-fixture"},
        )
        self.assertFalse(report["ready"])
        self.assertTrue(report["duplicate_source_batch"])
        self.assertTrue(any("already been generated" in reason for reason in report["reasons"]))

    def test_different_previous_batch_does_not_block(self) -> None:
        report = evaluate_readiness(
            data_source="local",
            latest_manifest={"source_batch_id": "older-batch"},
        )
        self.assertTrue(report["ready"])
        self.assertFalse(report["duplicate_source_batch"])
