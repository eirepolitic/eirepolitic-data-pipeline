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

    def test_fixture_renders_three_slide_carousel_with_source_change_trend_and_approved_style(self) -> None:
        result = render_polling_snapshot(
            "instagram/campaigns/ipi_polling_snapshot_v1/render_spec_fixture.yml"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["slide_count"], 3)
        self.assertFalse(result["publish_ready"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["visible_source_footer"], "Source: Irish Polling Indicator (IPI)")
        self.assertTrue(result["source_reference_in_caption"])
        self.assertEqual(result["dimensions"], [1080, 1350])
        self.assertEqual(result["latest_model_date"], "2026-07-31")
        self.assertEqual(result["previous_model_date"], "2026-07-30")
        self.assertEqual(result["render_style"], "approved_editorial_v1")

        self.assertEqual(len(result["slide_files"]), 3)
        for path in result["slide_files"]:
            output = Path(path)
            self.assertTrue(output.exists())
            with Image.open(output).convert("RGB") as image:
                self.assertEqual(image.size, (1080, 1350))
                self.assertEqual(image.getpixel((540, 1340)), (15, 47, 36))
                self.assertEqual(image.getpixel((540, 175)), (216, 180, 95))
                self.assertEqual(image.getpixel((22, 22)), (216, 180, 95))

        caption = Path(result["caption_file"]).read_text(encoding="utf-8")
        self.assertIn("Source: Irish Polling Indicator (IPI)", caption)
        self.assertIn("not a comparison of two individual polls", caption)
        self.assertIn("not a single opinion poll or an election forecast", caption)

        context = json.loads((self.output_root / "metadata/post_context.json").read_text(encoding="utf-8"))
        self.assertEqual(context["latest_model_date"], "2026-07-31")
        self.assertEqual(context["previous_model_date"], "2026-07-30")
        self.assertEqual(context["trend_days"], 90)
        self.assertEqual(context["render_style"], "approved_editorial_v1")
        self.assertEqual(context["visual_reference"], "workflow 33894430571 / party_issue_monthly_profile_v2")
        self.assertEqual(len(context["slides"]), 3)

        latest = context["latest_rows"]
        self.assertEqual(latest[0]["party_code"], "SF")
        self.assertEqual(latest[0]["value"], 25.0)
        self.assertEqual(latest[0]["low"], 23.0)
        self.assertEqual(latest[0]["high"], 27.0)

        changes = {row["party_code"]: row for row in context["change_rows"]}
        self.assertEqual(changes["SF"]["value"], 1.0)
        self.assertEqual(changes["FF"]["value"], 1.0)
        self.assertEqual(changes["FG"]["value"], -1.0)
        self.assertEqual(changes["SD"]["value"], 1.0)
        self.assertEqual(changes["LAB"]["value"], 0.0)

        trend = {series["party_code"]: series for series in context["trend_series"]}
        self.assertEqual(set(trend), {"SF", "FF", "FG"})
        self.assertGreaterEqual(len(trend["SF"]["points"]), 3)
        self.assertEqual(trend["SF"]["points"][-1]["value"], 25.0)
        self.assertEqual(context["source_attributions"][0]["source_id"], "irish_polling_indicator")


if __name__ == "__main__":
    unittest.main()
