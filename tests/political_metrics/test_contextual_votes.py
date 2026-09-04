from datetime import date

import pandas as pd

from political_metrics.contextual_votes import (
    audit_context_vote_reconciliation,
    build_context_division_party_vote_components,
    build_daily_context_vote_components,
)
from political_metrics.periods import MetricPeriod


def _period():
    return MetricPeriod(date(2026,1,1), date(2026,1,31), "2026-01", "month")


def _divisions():
    return pd.DataFrame([
        {"division_id":"d1","division_date":"2026-01-10","debate_section_id":"s1"},
        {"division_id":"d2","division_date":"2026-01-20","debate_section_id":"s2"},
    ])


def _division_context():
    return pd.DataFrame([
        {"division_id":"d1","division_context":"bill_or_legislation"},
        {"division_id":"d2","division_context":"motion_proceeding"},
    ])


def _memberships():
    return pd.DataFrame([
        {"member_code":"m1","membership_start":"2025-01-01","membership_end":None},
        {"member_code":"m2","membership_start":"2025-01-01","membership_end":None},
    ])


def _parties():
    return pd.DataFrame([
        {"member_code":"m1","party_start":"2025-01-01","party_end":None,"party_uri":"party:A","party_name":"A"},
        {"member_code":"m2","party_start":"2025-01-01","party_end":None,"party_uri":"party:B","party_name":"B"},
    ])


def _constituencies():
    return pd.DataFrame([
        {"member_code":"m1","represent_start":"2025-01-01","represent_end":None,"constituency_uri":"c:1","constituency_name":"C1"},
        {"member_code":"m2","represent_start":"2025-01-01","represent_end":None,"constituency_uri":"c:2","constituency_name":"C2"},
    ])


def _votes():
    return pd.DataFrame([
        {"division_id":"d1","division_date":"2026-01-10","member_code":"m1","vote_code":"ta"},
        {"division_id":"d1","division_date":"2026-01-10","member_code":"m2","vote_code":"nil"},
        {"division_id":"d2","division_date":"2026-01-20","member_code":"m1","vote_code":"ta"},
    ])


def _daily():
    return build_daily_context_vote_components(
        divisions=_divisions(), member_votes=_votes(), memberships=_memberships(),
        member_parties=_parties(), member_constituencies=_constituencies(),
        division_context=_division_context(), period=_period(), source_batch_id="b1", contract_version=1,
    )


def _party():
    return build_context_division_party_vote_components(
        member_votes=_votes(), member_parties=_parties(), division_context=_division_context(),
        period=_period(), source_batch_id="b1", contract_version=1,
    )


def test_context_daily_components_preserve_context_and_denominators():
    out = _daily()
    m1 = out[(out.grain=="member") & (out.entity_id=="m1")]
    eligible = m1[m1.component_id=="eligible_member_division_count"]
    recorded = m1[m1.component_id=="recorded_vote_count"]
    assert int(eligible.component_value.sum()) == 2
    assert int(recorded.component_value.sum()) == 2
    assert set(eligible.division_context) == {"bill_or_legislation","motion_proceeding"}


def test_context_party_components_keep_historical_party_and_context():
    out = _party()
    assert len(out) == 3
    assert set(out.division_context) == {"bill_or_legislation","motion_proceeding"}
    assert set(out.party_uri) == {"party:A","party:B"}


def test_reconciliation_matches_collapsed_unfiltered_foundations_exactly():
    daily_context = _daily()
    party_context = _party()

    existing_daily = daily_context.groupby(
        ["activity_date","grain","entity_id","component_id"], as_index=False
    )["component_value"].sum()
    existing_daily["source_batch_id"]="b1"
    existing_daily["component_version"]=1
    existing_daily["calculated_at_utc"]="now"
    existing_daily["contract_version"]=1

    existing_party = party_context.groupby(
        ["division_id","division_date","party_uri","vote_code"], as_index=False
    )["recorded_vote_count"].sum()
    existing_party["source_batch_id"]="b1"
    existing_party["component_version"]=1
    existing_party["calculated_at_utc"]="now"
    existing_party["contract_version"]=1

    audit = audit_context_vote_reconciliation(
        daily_context_vote_components=daily_context,
        context_division_party_vote_components=party_context,
        daily_activity_components=existing_daily,
        division_party_vote_components=existing_party,
    )
    assert audit["ready"] is True
    assert audit["daily_reconciliation_mismatches"] == 0
    assert audit["party_reconciliation_mismatches"] == 0


def test_missing_division_context_is_rejected():
    incomplete = _division_context().iloc[[0]].copy()
    try:
        build_daily_context_vote_components(
            divisions=_divisions(), member_votes=_votes(), memberships=_memberships(),
            member_parties=_parties(), member_constituencies=_constituencies(),
            division_context=incomplete, period=_period(), source_batch_id="b1", contract_version=1,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "does not cover every division" in str(exc)
