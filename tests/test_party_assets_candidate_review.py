import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from process.party_assets_candidate_review import build_candidate_review, select_candidates


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "image/png"):
        self.body = body
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        yield self.body


class FakeSession:
    def __init__(self, body: bytes):
        self.body = body

    def get(self, url, timeout, stream, allow_redirects):
        return FakeResponse(self.body)


def png_bytes() -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGBA", (300, 160), (50, 90, 140, 255)).save(buffer, "PNG")
    return buffer.getvalue()


class PartyAssetCandidateReviewTests(unittest.TestCase):
    def test_select_candidates_enforces_score_threshold(self):
        report = {
            "results": [
                {
                    "party_key": "example-party",
                    "candidates": [
                        {"url": "https://example.com/logo.png", "score": 10, "reasons": ["contains_logo"]},
                        {"url": "https://example.com/photo.png", "score": 2, "reasons": []},
                    ],
                }
            ]
        }
        selected = select_candidates(report)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["url"], "https://example.com/logo.png")

    def test_build_candidate_review_creates_sheet_without_registry_changes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            discovery = root / "discovery.json"
            output = root / "review"
            discovery.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "party_key": "example-party",
                                "candidates": [
                                    {
                                        "url": "https://example.com/example-logo.png",
                                        "score": 12,
                                        "reasons": ["contains_logo", "party_token:example"],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = build_candidate_review(discovery, output, session=FakeSession(png_bytes()))
            self.assertTrue(result["success"])
            self.assertEqual(result["reviewable_count"], 1)
            self.assertTrue((output / "candidate_contact_sheet.png").is_file())
            self.assertTrue((output / "candidate_review.json").is_file())
            self.assertTrue(any((output / "previews/example-party").glob("*.png")))

    def test_empty_candidate_set_still_creates_review_report(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            discovery = root / "discovery.json"
            output = root / "review"
            discovery.write_text(json.dumps({"results": []}), encoding="utf-8")
            result = build_candidate_review(discovery, output)
            self.assertTrue(result["success"])
            self.assertEqual(result["selected_count"], 0)
            self.assertTrue((output / "candidate_contact_sheet.png").is_file())


if __name__ == "__main__":
    unittest.main()
