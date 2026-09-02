import unittest

from process.party_assets import canonical_party_key, load_registry, resolve_party, validate_registry


class PartyAssetTests(unittest.TestCase):
    def test_party_key_contract_matches_commissioning_values(self):
        expected = {
            "100% RDR": "100-rdr",
            "Aontú": "aontu",
            "Fianna Fáil": "fianna-fail",
            "Fine Gael": "fine-gael",
            "Green Party": "green-party",
            "Independent": "independent",
            "Independent Ireland": "independent-ireland",
            "Labour Party": "labour-party",
            "People Before Profit-Solidarity": "people-before-profit-solidarity",
            "Sinn Féin": "sinn-fein",
            "Social Democrats": "social-democrats",
        }
        self.assertEqual({name: canonical_party_key(name) for name in expected}, expected)

    def test_registry_is_valid_and_complete_for_commissioning_set(self):
        rows = load_registry()
        self.assertEqual(validate_registry(rows), [])
        self.assertEqual(
            {row.party_key for row in rows},
            {
                "100-rdr", "aontu", "fianna-fail", "fine-gael", "green-party",
                "independent", "independent-ireland", "labour-party",
                "people-before-profit-solidarity", "sinn-fein", "social-democrats",
            },
        )

    def test_aliases_resolve_to_stable_keys(self):
        rows = load_registry()
        self.assertEqual(resolve_party("100% Redress Party", rows).party_key, "100-rdr")
        self.assertEqual(resolve_party("Aontu", rows).party_key, "aontu")
        self.assertEqual(resolve_party("The Green Party", rows).party_key, "green-party")
        self.assertEqual(resolve_party("Non-Party", rows).party_key, "independent")
        self.assertEqual(resolve_party("Solidarity - People Before Profit", rows).party_key, "people-before-profit-solidarity")
        self.assertEqual(resolve_party("Sinn Fein", rows).party_key, "sinn-fein")

    def test_independent_uses_explicit_generated_non_official_standin(self):
        row = resolve_party("Independent")
        self.assertIsNotNone(row)
        self.assertEqual(row.asset_status, "approved")
        self.assertEqual(row.source_type, "eirepolitic_generated_standin")
        self.assertEqual(row.fallback_type, "eirepolitic_neutral_standin")
        self.assertEqual(
            row.logo_s3_uri,
            "s3://eirepolitic-data/processed/reference/party_assets/v1/assets/independent/logo.png",
        )
        self.assertEqual(row.source_url, "")
        self.assertIn("Not official branding", row.licence_usage_note)


if __name__ == "__main__":
    unittest.main()
