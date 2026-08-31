from process.party_assets import canonical_party_key, load_registry, resolve_party, validate_registry


def test_party_key_contract_matches_commissioning_values():
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
    assert {name: canonical_party_key(name) for name in expected} == expected


def test_registry_is_valid_and_complete_for_commissioning_set():
    rows = load_registry()
    assert validate_registry(rows) == []
    assert {row.party_key for row in rows} == {
        "100-rdr",
        "aontu",
        "fianna-fail",
        "fine-gael",
        "green-party",
        "independent",
        "independent-ireland",
        "labour-party",
        "people-before-profit-solidarity",
        "sinn-fein",
        "social-democrats",
    }


def test_aliases_resolve_to_stable_keys():
    rows = load_registry()
    assert resolve_party("100% Redress Party", rows).party_key == "100-rdr"
    assert resolve_party("Aontu", rows).party_key == "aontu"
    assert resolve_party("The Green Party", rows).party_key == "green-party"
    assert resolve_party("Non-Party", rows).party_key == "independent"
    assert resolve_party("Solidarity - People Before Profit", rows).party_key == "people-before-profit-solidarity"
    assert resolve_party("Sinn Fein", rows).party_key == "sinn-fein"


def test_independent_has_explicit_non_party_fallback():
    row = resolve_party("Independent")
    assert row is not None
    assert row.asset_status == "approved_fallback"
    assert row.fallback_type == "no_party_logo"
    assert row.logo_s3_uri == ""
