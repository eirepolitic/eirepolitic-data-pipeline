import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from process.party_assets import PartyAsset
from process.party_assets_fetch import fetch_row, source_extension


class FakeResponse:
    def __init__(self, body=b"image-bytes", content_type="image/png", url="https://example.com/logo.png"):
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.url = url

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=65536):
        yield self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, timeout, stream, allow_redirects):
        self.calls.append((url, timeout, stream, allow_redirects))
        return self.response


def row(**overrides):
    values = {
        "party_key": "example-party",
        "party_name": "Example Party",
        "party_aliases": (),
        "logo_s3_uri": "s3://bucket/path/logo.png",
        "source_url": "https://example.com/logo.png",
        "source_type": "official_party_logo_asset",
        "retrieval_date": "2026-08-31",
        "licence_usage_note": "test",
        "asset_status": "source_identified_pending_ingest",
        "fallback_type": "",
    }
    values.update(overrides)
    return PartyAsset(**values)


class PartyAssetFetchTests(unittest.TestCase):
    def test_direct_asset_extension_requires_direct_source_type(self):
        self.assertEqual(source_extension(row()), ".png")
        self.assertIsNone(source_extension(row(source_type="official_party_site")))
        self.assertIsNone(source_extension(row(source_url="http://example.com/logo.png")))
        self.assertIsNone(source_extension(row(source_url="https://example.com/page")))

    def test_generic_page_is_unresolved_and_not_fetched(self):
        with TemporaryDirectory() as temp_dir:
            result = fetch_row(
                row(source_url="https://example.com/", source_type="official_party_site"),
                Path(temp_dir),
                session=FakeSession(FakeResponse()),
            )
            self.assertEqual(result["status"], "unresolved_source")
            self.assertEqual(list(Path(temp_dir).rglob("*")), [])

    def test_fetches_direct_asset_to_deterministic_staging_path(self):
        with TemporaryDirectory() as temp_dir:
            session = FakeSession(FakeResponse(body=b"abc"))
            result = fetch_row(row(), Path(temp_dir), session=session)
            destination = Path(temp_dir) / "example-party/source.png"
            self.assertEqual(result["status"], "fetched")
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes(), b"abc")
            self.assertEqual(len(session.calls), 1)

    def test_generated_independent_standin_is_staged_locally(self):
        with TemporaryDirectory() as temp_dir:
            result = fetch_row(
                row(
                    party_key="independent",
                    party_name="Independent",
                    source_url="",
                    source_type="eirepolitic_generated_standin",
                    logo_s3_uri="s3://bucket/assets/independent/logo.png",
                    asset_status="pending_review",
                    fallback_type="eirepolitic_neutral_standin",
                ),
                Path(temp_dir),
            )
            destination = Path(temp_dir) / "independent/source.png"
            self.assertEqual(result["status"], "generated")
            self.assertFalse(result["official_branding"])
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
