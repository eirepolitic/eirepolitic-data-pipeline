import unittest

from process.party_assets import PartyAsset
from process.party_assets_discover import discover_row, score_candidate


class FakeResponse:
    def __init__(self, html, url="https://example.com/"):
        self.content = html.encode("utf-8")
        self.encoding = "utf-8"
        self.url = url

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, html):
        self.response = FakeResponse(html)

    def get(self, url, timeout, allow_redirects):
        return self.response


def row(**overrides):
    values = {
        "party_key": "example-party",
        "party_name": "Example Party",
        "party_aliases": ("Example",),
        "logo_s3_uri": "s3://bucket/logo.png",
        "source_url": "https://example.com/",
        "source_type": "official_party_site",
        "retrieval_date": "2026-08-31",
        "licence_usage_note": "test",
        "asset_status": "source_identified_pending_ingest",
        "fallback_type": "",
    }
    values.update(overrides)
    return PartyAsset(**values)


class PartyAssetDiscoverTests(unittest.TestCase):
    def test_logo_candidate_scores_above_unrelated_image(self):
        party = row()
        logo_score, _ = score_candidate(
            party,
            {"alt": "Example Party logo", "title": "", "class": "site-logo"},
            "https://example.com/assets/example-logo.svg",
        )
        photo_score, _ = score_candidate(
            party,
            {"alt": "News photo", "title": "", "class": "hero"},
            "https://example.com/assets/news.jpg",
        )
        self.assertGreater(logo_score, photo_score)

    def test_discovery_ranks_supported_same_site_logo(self):
        html = '''
        <html><body>
          <img class="hero" src="/images/news.jpg" alt="News">
          <img class="site-logo" src="/assets/example-party-logo.svg" alt="Example Party logo">
          <img src="https://other.example.net/logo.svg" alt="Example Party logo">
        </body></html>
        '''
        result = discover_row(row(), session=FakeSession(html))
        self.assertEqual(result["status"], "candidates_found")
        self.assertEqual(result["candidates"][0]["url"], "https://example.com/assets/example-party-logo.svg")
        self.assertNotIn("other.example.net", " ".join(item["url"] for item in result["candidates"]))

    def test_non_official_page_row_is_not_scanned(self):
        result = discover_row(row(source_type="official_party_logo_asset"), session=FakeSession(""))
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["candidates"], [])


if __name__ == "__main__":
    unittest.main()
