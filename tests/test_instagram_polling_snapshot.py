from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from PIL import Image

from process.instagram_render_polling_snapshot import render_polling_snapshot


class InstagramPollingSnapshotTests(unittest.TestCase):
    output_root = Path("generated_posts/ipi_polling_snapshot_fixture")

    def tearDown(self) -> None:
        shutil.rmtree(self.output_root, ignore_errors=True)

    def test_fixture_renders_with_source_and_uncertainty(self) -> None:
        result = render_polling_snapshot(
            "instagram/campaigns/ipi_polling_snapshot_v1/render_spec_fixture.yml"
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["publish_ready"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["visible_source_footer"], "Source: Irish Polling Indicator (IPI)")
        self.assertTrue(result["source_reference_in_caption"])
        self.assertEqual(result["dimensions"], [1080, 1350])

        output = Path(result["output_file"])
        self.assertTrue(output.exists())
        with Image.open(output) as image:
            self.assertEqual(image.size, (1080, 1350))

        caption = Path(result["caption_file"]).read_text(encoding="utf-8")
        self.assertIn("Source: Irish Polling Indicator (IPI)", caption)
        self.assertIn("not the result of a single opinion poll", caption)

        context = json.loads((self.output_root / "metadata/post_context.json").read_text(encoding="utf-8"))
        self.assertEqual(context["model_date"], "2026-07-31")
        self.assertEqual(context["chart_rows"][0]["party_code"], "SF")
        self.assertEqual(context["chart_rows"][0]["value"], 25.0)
        self.assertEqual(context["chart_rows"][0]["low"], 23.0)
        self.assertEqual(context["chart_rows"][0]["high"], 27.0)
        self.assertEqual(context["source_attributions"][0]["source_id"], "irish_polling_indicator")


if __name__ == "__main__":
    unittest.main()
