import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from process.party_assets_build import CANVAS_SIZE, build_assets, normalize_logo, validate_normalized


class PartyAssetBuildTests(unittest.TestCase):
    def test_normalize_logo_creates_square_transparent_png(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            output = root / "logo.png"
            image = Image.new("RGBA", (600, 300), (0, 0, 0, 0))
            for x in range(100, 500):
                for y in range(50, 250):
                    image.putpixel((x, y), (20, 80, 120, 255))
            image.save(source)

            metadata = normalize_logo(source, output)
            self.assertTrue(output.is_file())
            self.assertEqual(validate_normalized(output), [])
            self.assertEqual(metadata["output_width"], CANVAS_SIZE)
            self.assertEqual(metadata["output_height"], CANVAS_SIZE)
            self.assertTrue(metadata["has_transparency"])
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_build_assets_handles_explicit_fallback_and_builds_contact_sheet(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "sources"
            output_root = root / "output"
            registry = root / "registry.csv"

            registry.parent.mkdir(parents=True, exist_ok=True)
            with registry.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "party_key", "party_name", "party_aliases", "logo_s3_uri", "source_url",
                    "source_type", "retrieval_date", "licence_usage_note", "asset_status", "fallback_type",
                ])
                writer.writerow([
                    "example-party", "Example Party", "", "s3://eirepolitic-data/processed/reference/party_assets/v1/assets/example-party/logo.png",
                    "https://example.com/logo", "official_party_site", "2026-08-31", "Test note",
                    "source_identified_pending_ingest", "",
                ])
                writer.writerow([
                    "independent", "Independent", "Non-Party", "", "", "none", "2026-08-31",
                    "No party logo", "approved_fallback", "no_party_logo",
                ])

            party_dir = source_root / "example-party"
            party_dir.mkdir(parents=True)
            Image.new("RGBA", (500, 500), (120, 40, 100, 255)).save(party_dir / "reviewed.png")

            manifest = build_assets(source_root, output_root, registry)
            self.assertTrue(manifest["success"])
            self.assertTrue((output_root / "assets/example-party/logo.png").is_file())
            self.assertTrue((output_root / "assets/example-party/source.png").is_file())
            self.assertTrue((output_root / "contact_sheet.png").is_file())
            self.assertTrue((output_root / "manifest.json").is_file())

            saved = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            statuses = {entry["party_key"]: entry["build_status"] for entry in saved["entries"]}
            self.assertEqual(statuses["example-party"], "built")
            self.assertEqual(statuses["independent"], "fallback")


if __name__ == "__main__":
    unittest.main()
