from __future__ import annotations

import unittest

from instagram.renderer.attribution import resolve_attributions
from instagram.renderer.render import apply_source_attribution


class InstagramSourceAttributionTests(unittest.TestCase):
    def test_ipi_source_is_automatically_added_to_visible_footer(self) -> None:
        spec = {
            "post": {"slug": "polling-test"},
            "data": {"source_ids": ["irish_polling_indicator"]},
            "branding": {"footer_note": "eirepolitic.ie"},
            "slides": [],
        }

        updated = apply_source_attribution(spec)

        self.assertIn("Source: Irish Polling Indicator (IPI)", updated["branding"]["footer_note"])
        self.assertIn("eirepolitic.ie", updated["branding"]["footer_note"])
        attributions = updated["post"]["source_attributions"]
        self.assertEqual(len(attributions), 1)
        self.assertEqual(attributions[0]["source_id"], "irish_polling_indicator")
        self.assertTrue(attributions[0]["required"])

    def test_ipi_attribution_is_not_duplicated(self) -> None:
        spec = {
            "post": {"slug": "polling-test"},
            "data": {"source_ids": ["irish_polling_indicator", "irish_polling_indicator"]},
            "branding": {"footer_note": "Source: Irish Polling Indicator (IPI)"},
            "slides": [],
        }

        updated = apply_source_attribution(spec)

        self.assertEqual(updated["branding"]["footer_note"].count("Source: Irish Polling Indicator (IPI)"), 1)
        self.assertEqual(len(updated["post"]["source_attributions"]), 1)

    def test_unknown_source_is_rejected_instead_of_silently_omitted(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_attributions(["unregistered_source"])

    def test_posts_without_declared_external_sources_are_unchanged(self) -> None:
        spec = {
            "post": {"slug": "internal-test"},
            "data": {},
            "branding": {"footer_note": "eirepolitic.ie"},
            "slides": [],
        }

        updated = apply_source_attribution(spec)

        self.assertEqual(updated["branding"]["footer_note"], "eirepolitic.ie")
        self.assertEqual(updated["post"]["source_attributions"], [])


if __name__ == "__main__":
    unittest.main()
