import pandas as pd

from political_metrics.division_context import audit_division_context, build_division_context


def _divisions():
    return pd.DataFrame([
        {"division_id":"d1","division_date":"2026-01-01","debate_section_id":"s1"},
        {"division_id":"d2","division_date":"2026-01-01","debate_section_id":"s2"},
        {"division_id":"d3","division_date":"2026-01-01","debate_section_id":"s3"},
        {"division_id":"d4","division_date":"2026-01-01","debate_section_id":"s4"},
    ])


def _speech_context():
    return pd.DataFrame([
        {"speech_id":"p1","debate_section_id":"s1","speech_context":"bill_or_legislation"},
        {"speech_id":"p2","debate_section_id":"s2","speech_context":"motion_proceeding"},
        {"speech_id":"p3","debate_section_id":"s3","speech_context":"procedural_business"},
        {"speech_id":"p4","debate_section_id":"s4","speech_context":"other"},
        {"speech_id":"p5","debate_section_id":"s2","speech_context":"motion_proceeding"},
    ])


def _bills():
    return pd.DataFrame([{"bill_id":"bill-a","debate_section_id":"s1"}])


def _votes():
    return pd.DataFrame([
        {"division_id":"d1","member_code":"m1","vote_code":"ta"},
        {"division_id":"d1","member_code":"m2","vote_code":"nil"},
        {"division_id":"d2","member_code":"m1","vote_code":"ta"},
        {"division_id":"d3","member_code":"m1","vote_code":"staon"},
        {"division_id":"d4","member_code":"m1","vote_code":"ta"},
    ])


def test_build_division_context_assigns_all_supported_contexts_and_bill_entity():
    out = build_division_context(
        divisions=_divisions(),
        speech_context=_speech_context(),
        bill_debate_sections=_bills(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    got = dict(zip(out.division_id, out.division_context))
    assert got == {
        "d1":"bill_or_legislation",
        "d2":"motion_proceeding",
        "d3":"procedural_business",
        "d4":"other",
    }
    bill = out.loc[out.division_id.eq("d1")].iloc[0]
    assert bill.linked_entity_type == "bill"
    assert bill.linked_entity_id == "bill-a"


def test_bill_bridge_takes_precedence_over_section_projection():
    speech_context = _speech_context().copy()
    speech_context.loc[speech_context.debate_section_id.eq("s1"), "speech_context"] = "motion_proceeding"
    out = build_division_context(
        divisions=_divisions().iloc[[0]],
        speech_context=speech_context,
        bill_debate_sections=_bills(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert out.iloc[0].division_context == "bill_or_legislation"
    assert out.iloc[0].linked_entity_id == "bill-a"


def test_multiple_non_other_speech_contexts_in_one_section_are_rejected():
    speech_context = pd.DataFrame([
        {"speech_id":"p1","debate_section_id":"s1","speech_context":"motion_proceeding"},
        {"speech_id":"p2","debate_section_id":"s1","speech_context":"procedural_business"},
    ])
    bills = pd.DataFrame(columns=["bill_id","debate_section_id"])
    try:
        build_division_context(
            divisions=_divisions().iloc[[0]],
            speech_context=speech_context,
            bill_debate_sections=bills,
            source_batch_id="batch-1",
            contract_version=1,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "multiple non-other contexts" in str(exc)


def test_audit_requires_complete_division_coverage_and_no_vote_multiplication():
    out = build_division_context(
        divisions=_divisions(),
        speech_context=_speech_context(),
        bill_debate_sections=_bills(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    audit = audit_division_context(
        division_context=out,
        divisions=_divisions(),
        member_votes=_votes(),
        bill_debate_sections=_bills(),
    )
    assert audit["ready"] is True
    assert audit["row_count"] == 4
    assert audit["member_vote_rows"] == 5
    assert audit["member_vote_rows_after_join"] == 5
