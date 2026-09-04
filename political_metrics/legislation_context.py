from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd


CERTIFICATION_VERSION = 1
EVIDENCE_METHOD = "debate_scoped_section_eid_plus_exact_heading_agreement"


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for col in result.columns:
        result[col] = result[col].fillna("").astype(str).str.strip()
    return result


def build_bill_debate_sections(
    *,
    bill_debates: pd.DataFrame,
    debate_sections: pd.DataFrame,
    source_batch_id: str,
    contract_version: int,
) -> pd.DataFrame:
    """Build certified Bill-to-debate-section links at one row per Bill + section.

    Certification requires the debate-scoped source section eid and a unique exact
    source heading to identify the same canonical debate section. Sections linked
    to more than one Bill are excluded. Duplicate source Bill-debate records for
    the same Bill-section pair are collapsed into provenance fields.
    """
    bd = _clean(bill_debates)
    sections = _clean(debate_sections)

    required_bill = {
        "bill_debate_id",
        "bill_id",
        "debate_id",
        "debate_date",
        "debate_show_as",
        "debate_section_id",
    }
    required_sections = {
        "debate_id",
        "debate_section_id",
        "section_eid",
        "heading",
        "show_as",
    }
    missing_bill = sorted(required_bill - set(bd.columns))
    missing_sections = sorted(required_sections - set(sections.columns))
    if missing_bill:
        raise ValueError(f"bill_debates missing required columns: {missing_bill}")
    if missing_sections:
        raise ValueError(f"debate_sections missing required columns: {missing_sections}")

    sections["section_heading"] = sections["show_as"].where(sections["show_as"].ne(""), sections["heading"])

    key_counts = sections.groupby(["debate_id", "section_eid"], dropna=False).size()
    nonunique_keys = set(key_counts[key_counts > 1].index.tolist())
    section_by_source_key: dict[tuple[str, str], tuple[str, str]] = {}
    for row in sections.itertuples(index=False):
        key = (row.debate_id, row.section_eid)
        if row.debate_id and row.section_eid and key not in nonunique_keys:
            section_by_source_key[key] = (row.debate_section_id, row.section_heading)

    heading_groups = (
        sections.loc[sections["debate_id"].ne("") & sections["section_heading"].ne("")]
        .groupby(["debate_id", "section_heading"], dropna=False)[["debate_section_id", "section_eid"]]
        .apply(lambda g: g.to_dict(orient="records"))
        .to_dict()
    )

    eligible_rows: list[dict] = []
    for row in bd.itertuples(index=False):
        source_key = (row.debate_id, row.debate_section_id)
        source_match = section_by_source_key.get(source_key)
        heading_matches = heading_groups.get((row.debate_id, row.debate_show_as), []) if row.debate_id and row.debate_show_as else []
        if not source_match or len(heading_matches) != 1:
            continue
        heading_match = heading_matches[0]
        if source_match[0] != heading_match["debate_section_id"]:
            continue
        eligible_rows.append(
            {
                "bill_debate_id": row.bill_debate_id,
                "bill_id": row.bill_id,
                "debate_section_id": source_match[0],
                "debate_id": row.debate_id,
                "debate_date": row.debate_date,
                "source_section_eid": row.debate_section_id,
                "debate_show_as": row.debate_show_as,
            }
        )

    columns = [
        "bill_id",
        "debate_section_id",
        "debate_id",
        "debate_date",
        "source_section_eid",
        "debate_show_as",
        "evidence_method",
        "source_bill_debate_count",
        "source_bill_debate_ids_json",
        "certification_version",
        "source_batch_id",
        "calculated_at_utc",
        "contract_version",
    ]
    if not eligible_rows:
        return pd.DataFrame(columns=columns)

    eligible = pd.DataFrame(eligible_rows)
    bill_counts = eligible.groupby("debate_section_id")["bill_id"].nunique()
    multi_bill_sections = set(bill_counts[bill_counts > 1].index.astype(str))
    eligible = eligible[~eligible["debate_section_id"].isin(multi_bill_sections)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    now = datetime.now(timezone.utc).isoformat()
    output_rows: list[dict] = []
    for (bill_id, debate_section_id), group in eligible.groupby(["bill_id", "debate_section_id"], sort=True):
        bill_debate_ids = sorted(set(group["bill_debate_id"].astype(str)))
        first = group.sort_values(["debate_date", "bill_debate_id"], kind="stable").iloc[0]
        output_rows.append(
            {
                "bill_id": bill_id,
                "debate_section_id": debate_section_id,
                "debate_id": first["debate_id"],
                "debate_date": first["debate_date"],
                "source_section_eid": first["source_section_eid"],
                "debate_show_as": first["debate_show_as"],
                "evidence_method": EVIDENCE_METHOD,
                "source_bill_debate_count": len(bill_debate_ids),
                "source_bill_debate_ids_json": json.dumps(bill_debate_ids, separators=(",", ":")),
                "certification_version": CERTIFICATION_VERSION,
                "source_batch_id": source_batch_id,
                "calculated_at_utc": now,
                "contract_version": contract_version,
            }
        )

    return pd.DataFrame(output_rows, columns=columns).sort_values(
        ["debate_date", "debate_section_id", "bill_id"], kind="stable"
    ).reset_index(drop=True)


def audit_bill_debate_sections(
    *,
    bridge: pd.DataFrame,
    speeches: pd.DataFrame,
    divisions: pd.DataFrame,
) -> dict:
    """Return deployment checks for a certified Bill-section bridge."""
    frame = _clean(bridge)
    speech_frame = _clean(speeches)
    division_frame = _clean(divisions)

    duplicate_pairs = int(frame.duplicated(["bill_id", "debate_section_id"]).sum()) if not frame.empty else 0
    multi_bill_sections = (
        int((frame.groupby("debate_section_id")["bill_id"].nunique() > 1).sum()) if not frame.empty else 0
    )
    speech_join = speech_frame.merge(
        frame[["bill_id", "debate_section_id"]], on="debate_section_id", how="inner", validate="many_to_one"
    ) if not frame.empty else speech_frame.iloc[0:0].copy()
    division_join = division_frame.merge(
        frame[["bill_id", "debate_section_id"]], on="debate_section_id", how="inner", validate="many_to_one"
    ) if not frame.empty else division_frame.iloc[0:0].copy()

    checks = {
        "nonempty": bool(len(frame) > 0),
        "primary_key_unique": duplicate_pairs == 0,
        "one_bill_per_section": multi_bill_sections == 0,
        "speech_join_no_multiplication": int(len(speech_join)) == int(speech_frame["debate_section_id"].isin(set(frame["debate_section_id"])).sum()),
        "division_join_no_multiplication": int(len(division_join)) == int(division_frame["debate_section_id"].isin(set(frame["debate_section_id"])).sum()),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "bridge_rows": int(len(frame)),
        "distinct_bills": int(frame["bill_id"].nunique()) if not frame.empty else 0,
        "distinct_sections": int(frame["debate_section_id"].nunique()) if not frame.empty else 0,
        "linked_speeches": int(len(speech_join)),
        "linked_divisions": int(len(division_join)),
        "duplicate_pairs": duplicate_pairs,
        "multi_bill_sections": multi_bill_sections,
    }
