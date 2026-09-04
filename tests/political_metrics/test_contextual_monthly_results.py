from datetime import date

import pandas as pd

from political_metrics.contextual_monthly_results import (
    audit_monthly_contextual_vote_results,
    build_monthly_contextual_vote_results,
    member_context_reliability,
    party_context_reliability,
)
from political_metrics.periods import MetricPeriod


def _period():
    return MetricPeriod(date(2026,1,1), date(2026,1,31), "2026-01", "month")


def test_member_context_reliability_bands():
    assert member_context_reliability(25)[0] == "reliable"
    assert member_context_reliability(10)[0] == "caution"
    assert member_context_reliability(5)[0] == "caution"
    assert member_context_reliability(4)[0] == "insufficient_for_comparison"


def test_party_context_reliability_keeps_existing_thresholds():
    assert party_context_reliability(10)[0] == "reliable"
    assert party_context_reliability(5)[0] == "caution"
    assert party_context_reliability(4)[0] == "insufficient_for_comparison"


def test_build_monthly_contextual_results_use_existing_proportion_scale():
    daily = pd.DataFrame([
        {"activity_date":"2026-01-10","division_context":"bill_or_legislation","grain":"member","entity_id":"m1","component_id":"recorded_vote_count","component_value":20},
        {"activity_date":"2026-01-10","division_context":"bill_or_legislation","grain":"member","entity_id":"m1","component_id":"eligible_member_division_count","component_value":25},
    ])
    party = []
    for i in range(10):
        division_id=f"d{i}"
        party.extend([
            {"division_id":division_id,"division_date":"2026-01-15","division_context":"bill_or_legislation","party_uri":"party:A","vote_code":"ta","recorded_vote_count":2},
            {"division_id":division_id,"division_date":"2026-01-15","division_context":"bill_or_legislation","party_uri":"party:A","vote_code":"nil","recorded_vote_count":1},
        ])
    out = build_monthly_contextual_vote_results(
        daily_context_vote_components=daily,
        context_division_party_vote_components=pd.DataFrame(party),
        period=_period(),
        source_batch_id="b1",
        contract_version=1,
    )
    member = out[out.metric_id.eq("member_vote_participation_pct")].iloc[0]
    cohesion = out[out.metric_id.eq("party_vote_cohesion_pct")].iloc[0]
    assert member.value == 0.8
    assert member.output_unit == "proportion"
    assert member.reliability_status == "reliable"
    assert cohesion.value == 20 / 30
    assert cohesion.reliability_status == "reliable"
    assert cohesion.numerator == 20
    assert cohesion.denominator == 30


def test_small_member_context_sample_is_not_certified():
    daily = pd.DataFrame([
        {"activity_date":"2026-01-10","division_context":"other","grain":"member","entity_id":"m1","component_id":"recorded_vote_count","component_value":3},
        {"activity_date":"2026-01-10","division_context":"other","grain":"member","entity_id":"m1","component_id":"eligible_member_division_count","component_value":4},
    ])
    out = build_monthly_contextual_vote_results(
        daily_context_vote_components=daily,
        context_division_party_vote_components=pd.DataFrame(columns=["division_id","division_date","division_context","party_uri","vote_code","recorded_vote_count"]),
        period=_period(), source_batch_id="b1", contract_version=1,
    )
    row=out.iloc[0]
    assert row.reliability_status == "insufficient_for_comparison"
    assert row.public_use_status == "not_certified"
    assert row.warning_code == "insufficient_context_sample"


def test_audit_accepts_valid_contextual_rows_and_rejects_duplicates():
    daily = pd.DataFrame([
        {"activity_date":"2026-01-10","division_context":"other","grain":"member","entity_id":"m1","component_id":"recorded_vote_count","component_value":3},
        {"activity_date":"2026-01-10","division_context":"other","grain":"member","entity_id":"m1","component_id":"eligible_member_division_count","component_value":4},
    ])
    out = build_monthly_contextual_vote_results(
        daily_context_vote_components=daily,
        context_division_party_vote_components=pd.DataFrame(columns=["division_id","division_date","division_context","party_uri","vote_code","recorded_vote_count"]),
        period=_period(), source_batch_id="b1", contract_version=1,
    )
    audit=audit_monthly_contextual_vote_results(results=out,periods=[_period()],source_batch_id="b1")
    assert audit["ready"] is True
    bad=pd.concat([out,out],ignore_index=True)
    assert audit_monthly_contextual_vote_results(results=bad,periods=[_period()],source_batch_id="b1")["ready"] is False
