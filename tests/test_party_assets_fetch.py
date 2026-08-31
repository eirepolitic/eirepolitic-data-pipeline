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

    def test_fallback_is_never_fetched(self):
        with TemporaryDirectory() as temp_dir:
            result = fetch_row(
                row(
                    party_key="independent",
                    party_name="Independent",
                    source_url="",
                    source_type="none",
                    logo_s3_uri="",
                    asset_status="approved_fallback",
                    fallback_type="no_party_logo",
                ),
                Path(temp_dir),
            )
            self.assertEqual(result["status"], "fallback")
            self.assertEqual(result["fallback_type"], "no_party_logo")


if __name__ == "__main__":
    unittest.main()
