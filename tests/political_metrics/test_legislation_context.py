import json

import pandas as pd

from political_metrics.legislation_context import audit_bill_debate_sections, build_bill_debate_sections


def _sections():
    return pd.DataFrame([
        {"debate_id":"d1","debate_section_id":"s1","section_eid":"dbsect_1","heading":"Bill A: Second Stage","show_as":"Bill A: Second Stage"},
        {"debate_id":"d1","debate_section_id":"s2","section_eid":"dbsect_2","heading":"Other Business","show_as":"Other Business"},
        {"debate_id":"d2","debate_section_id":"s3","section_eid":"dbsect_1","heading":"Bill B: First Stage","show_as":"Bill B: First Stage"},
        {"debate_id":"d3","debate_section_id":"s4","section_eid":"dbsect_1","heading":"Shared Heading","show_as":"Shared Heading"},
    ])


def test_build_bill_debate_sections_collapses_duplicate_source_rows_and_scopes_eids_by_debate():
    bill_debates = pd.DataFrame([
        {"bill_debate_id":"r1","bill_id":"bill-a","debate_id":"d1","debate_date":"2026-01-01","debate_show_as":"Bill A: Second Stage","debate_section_id":"dbsect_1"},
        {"bill_debate_id":"r2","bill_id":"bill-a","debate_id":"d1","debate_date":"2026-01-01","debate_show_as":"Bill A: Second Stage","debate_section_id":"dbsect_1"},
        {"bill_debate_id":"r3","bill_id":"bill-b","debate_id":"d2","debate_date":"2026-01-02","debate_show_as":"Bill B: First Stage","debate_section_id":"dbsect_1"},
    ])
    out = build_bill_debate_sections(
        bill_debates=bill_debates,
        debate_sections=_sections(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert len(out) == 2
    a = out.loc[out.bill_id.eq("bill-a")].iloc[0]
    assert a.debate_section_id == "s1"
    assert int(a.source_bill_debate_count) == 2
    assert json.loads(a.source_bill_debate_ids_json) == ["r1", "r2"]
    b = out.loc[out.bill_id.eq("bill-b")].iloc[0]
    assert b.debate_section_id == "s3"


def test_build_bill_debate_sections_excludes_heading_conflict_and_multi_bill_section():
    bill_debates = pd.DataFrame([
        {"bill_debate_id":"bad","bill_id":"bill-x","debate_id":"d1","debate_date":"2026-01-01","debate_show_as":"Other Business","debate_section_id":"dbsect_1"},
        {"bill_debate_id":"m1","bill_id":"bill-c","debate_id":"d3","debate_date":"2026-01-03","debate_show_as":"Shared Heading","debate_section_id":"dbsect_1"},
        {"bill_debate_id":"m2","bill_id":"bill-d","debate_id":"d3","debate_date":"2026-01-03","debate_show_as":"Shared Heading","debate_section_id":"dbsect_1"},
    ])
    out = build_bill_debate_sections(
        bill_debates=bill_debates,
        debate_sections=_sections(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    assert out.empty


def test_audit_bill_debate_sections_counts_exact_section_links_without_multiplication():
    bill_debates = pd.DataFrame([
        {"bill_debate_id":"r1","bill_id":"bill-a","debate_id":"d1","debate_date":"2026-01-01","debate_show_as":"Bill A: Second Stage","debate_section_id":"dbsect_1"},
    ])
    bridge = build_bill_debate_sections(
        bill_debates=bill_debates,
        debate_sections=_sections(),
        source_batch_id="batch-1",
        contract_version=1,
    )
    speeches = pd.DataFrame([
        {"speech_id":"p1","debate_section_id":"s1"},
        {"speech_id":"p2","debate_section_id":"s1"},
        {"speech_id":"p3","debate_section_id":"s2"},
    ])
    divisions = pd.DataFrame([
        {"division_id":"v1","debate_section_id":"s1"},
        {"division_id":"v2","debate_section_id":"s2"},
    ])
    audit = audit_bill_debate_sections(bridge=bridge, speeches=speeches, divisions=divisions)
    assert audit["ready"] is True
    assert audit["linked_speeches"] == 2
    assert audit["linked_divisions"] == 1
