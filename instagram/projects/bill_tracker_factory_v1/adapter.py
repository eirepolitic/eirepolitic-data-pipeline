from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

from instagram.factory.oireachtas_source import load_csv_tables, resolve_validated_production_batch
from instagram.factory.package import deterministic_zip
from instagram.factory.render_primitives import contact_sheet
from instagram.projects.bill_tracker_factory_v1.renderers import (
    render_cover,
    render_explainer,
    render_party_vote,
    render_procedure,
    render_process_glossary,
    render_vote_glossary,
)

PROJECT_ID = "bill_tracker_factory_v1"
_ALLOWED_PERIODS = {"post1", "post2"}


def _assert_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing rendered slide: {path}")
    with Image.open(path) as image:
        if image.size != (1080, 1350):
            raise RuntimeError(f"Unexpected dimensions for {path}: {image.size}")


def _load_content(period_spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
    period = (period_spec or "post1").strip().lower()
    if period not in _ALLOWED_PERIODS:
        raise RuntimeError(f"Bill Tracker period must be one of {sorted(_ALLOWED_PERIODS)}; got {period_spec!r}")
    path = Path("instagram/projects/bill_tracker_factory_v1/content.yml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    post = (payload.get("posts") or {}).get(period)
    if not post:
        raise RuntimeError(f"No content found for Bill Tracker period {period!r}")
    if len(post.get("bills") or []) != 3:
        raise RuntimeError(f"Bill Tracker {period} requires exactly three Bills")
    return payload["series"], post


def _col(frame: pd.DataFrame, *names: str) -> str:
    lookup = {str(c).lower(): str(c) for c in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise RuntimeError(f"Required column missing. Tried {names}; available={list(frame.columns)}")


def _as_date(value: Any) -> date | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _active(frame: pd.DataFrame, *, on: date, start_col: str, end_col: str) -> pd.DataFrame:
    starts = frame[start_col].map(_as_date)
    ends = frame[end_col].map(_as_date)
    return frame[(starts.isna() | (starts <= on)) & (ends.isna() | (ends >= on))].copy()


def _vote_bucket(value: Any) -> str:
    text = str(value or "").strip().lower().replace("á", "a").replace("í", "i")
    if text in {"ta", "yes", "aye", "for"} or text.startswith("ta "):
        return "ta"
    if text in {"nil", "no", "nay", "against"} or text.startswith("nil "):
        return "nil"
    if "staon" in text or "abst" in text:
        return "abstain"
    raise RuntimeError(f"Unrecognized Oireachtas vote label: {value!r}")


def _party_bucket(value: Any) -> str:
    text = " ".join(str(value or "").replace("–", "-").replace("—", "-").split()).strip()
    folded = text.lower()
    aliases = [
        ("fianna fáil", "Fianna Fáil"), ("fianna fail", "Fianna Fáil"),
        ("sinn féin", "Sinn Féin"), ("sinn fein", "Sinn Féin"),
        ("fine gael", "Fine Gael"),
        ("social democrats", "Social Democrats"),
        ("labour", "Labour"),
        ("independent ireland", "Independent Ireland"),
        ("people before profit", "PBP-S"), ("pbp", "PBP-S"), ("solidarity", "PBP-S"),
        ("aontú", "Aontú"), ("aontu", "Aontú"),
        ("green", "Green"),
        ("100% rdr", "100% Redress"), ("100% redress", "100% Redress"), ("redress", "100% Redress"),
        ("independent", "Independent"),
    ]
    for token, label in aliases:
        if token in folded:
            return label
    raise RuntimeError(f"Unmapped party/group name in historical party data: {text!r}")


def _find_division(
    member_votes: pd.DataFrame,
    divisions: pd.DataFrame,
    vote: dict[str, Any],
    *,
    bill_title: str,
) -> tuple[str, pd.DataFrame, dict[str, int], dict[str, Any]]:
    mv_division_col = _col(member_votes, "division_id")
    mv_date_col = _col(member_votes, "division_date", "date")
    mv_label_col = _col(member_votes, "vote_label", "vote")
    div_id_col = _col(divisions, "division_id")
    div_date_col = _col(divisions, "division_date", "date")
    debate_col = _col(divisions, "debate_show_as", "debateShowAs")
    subject_col = _col(divisions, "subject")
    outcome_col = _col(divisions, "outcome")

    target_date = date.fromisoformat(str(vote["date"]))
    day_votes = member_votes[member_votes[mv_date_col].map(_as_date) == target_date].copy()
    day_divisions = divisions[divisions[div_date_col].map(_as_date) == target_date].copy()
    title_matches = day_divisions[
        day_divisions[debate_col].fillna("").astype(str).str.startswith(bill_title, na=False)
    ].copy()
    if title_matches.empty:
        raise RuntimeError(f"No official division record found for {bill_title!r} on {target_date}")

    candidates: list[tuple[str, pd.DataFrame, dict[str, int], dict[str, Any]]] = []
    for _, division_row in title_matches.iterrows():
        division_id = str(division_row[div_id_col])
        group = day_votes[day_votes[mv_division_col].astype(str).eq(division_id)].copy()
        if group.empty:
            continue
        counts = {"ta": 0, "nil": 0, "abstain": 0}
        for label, count in group[mv_label_col].value_counts(dropna=False).items():
            counts[_vote_bucket(label)] += int(count)
        if (counts["ta"], counts["nil"], counts["abstain"]) == (
            int(vote["ta"]), int(vote["nil"]), int(vote["abstain"])
        ):
            official = {
                "division_id": division_id,
                "date": target_date.isoformat(),
                "subject": str(division_row[subject_col] or ""),
                "outcome": str(division_row[outcome_col] or ""),
                "debate_show_as": str(division_row[debate_col] or ""),
            }
            candidates.append((division_id, group, counts, official))

    if len(candidates) != 1:
        summary = [(item[0], item[2], item[3]["debate_show_as"]) for item in candidates]
        raise RuntimeError(
            f"Expected one Bill-linked official division on {target_date} matching "
            f"{vote['ta']}/{vote['nil']}/{vote['abstain']} for {bill_title!r}; found {len(candidates)}: {summary}"
        )
    return candidates[0]


def _validate_vote(vote: dict[str, Any], frames: dict[str, pd.DataFrame], *, bill_title: str) -> dict[str, Any]:
    member_votes = frames["silver_member_votes"]
    memberships = frames["silver_member_memberships"]
    parties = frames["silver_member_parties"]
    divisions = frames["silver_divisions"]
    division_id, division_votes, recorded, official = _find_division(
        member_votes, divisions, vote, bill_title=bill_title
    )
    target_date = date.fromisoformat(str(vote["date"]))

    mv_member = _col(division_votes, "member_code")
    mv_label = _col(division_votes, "vote_label", "vote")
    membership_member = _col(memberships, "member_code")
    chamber_col = _col(memberships, "chamber")
    house_col = _col(memberships, "house_no")
    ms = _col(memberships, "membership_start")
    me = _col(memberships, "membership_end")
    active_memberships = _active(memberships, on=target_date, start_col=ms, end_col=me)
    active_memberships = active_memberships[
        active_memberships[chamber_col].astype(str).str.lower().eq("dail")
        & active_memberships[house_col].astype(str).eq("34")
    ].copy()
    eligible_members = sorted(set(active_memberships[membership_member].dropna().astype(str)))
    if len(eligible_members) != int(vote["eligible"]):
        raise RuntimeError(
            f"Eligible Dáil membership mismatch on {target_date}: production={len(eligible_members)}, approved={vote['eligible']}"
        )

    recorded_members = set(division_votes[mv_member].dropna().astype(str))
    if not recorded_members.issubset(set(eligible_members)):
        extra = sorted(recorded_members - set(eligible_members))
        raise RuntimeError(f"Recorded voters outside eligible membership for {division_id}: {extra}")
    no_recorded = len(eligible_members) - len(recorded_members)
    if no_recorded != int(vote["no_recorded_vote"]):
        raise RuntimeError(
            f"No-recorded-vote mismatch for {division_id}: production={no_recorded}, approved={vote['no_recorded_vote']}"
        )

    party_member = _col(parties, "member_code")
    party_name = _col(parties, "party_name")
    ps = _col(parties, "party_start")
    pe = _col(parties, "party_end")
    active_parties = _active(parties, on=target_date, start_col=ps, end_col=pe)
    active_parties = active_parties[active_parties[party_member].astype(str).isin(eligible_members)].copy()
    overlap = active_parties.groupby(party_member).size()
    ambiguous = sorted(overlap[overlap > 1].index.astype(str))
    if ambiguous:
        raise RuntimeError(f"Overlapping/ambiguous historical party assignments on {target_date}: {ambiguous}")
    party_by_member = dict(zip(active_parties[party_member].astype(str), active_parties[party_name]))
    missing_party = sorted(set(eligible_members) - set(party_by_member))
    if missing_party:
        raise RuntimeError(f"Eligible members missing date-correct party history on {target_date}: {missing_party}")

    result: dict[str, dict[str, int]] = {}
    recorded_vote_by_member = {
        str(row[mv_member]): _vote_bucket(row[mv_label]) for _, row in division_votes.iterrows()
    }
    for member in eligible_members:
        party = _party_bucket(party_by_member[member])
        result.setdefault(
            party,
            {"eligible": 0, "ta": 0, "nil": 0, "abstain": 0, "no_recorded_vote": 0},
        )
        result[party]["eligible"] += 1
        bucket = recorded_vote_by_member.get(member, "no_recorded_vote")
        result[party][bucket] += 1

    approved_main: dict[str, dict[str, int]] = {}
    for row in vote["rows"]:
        party, eligible, ta, nil, nr = row
        approved_main[str(party)] = {
            "eligible": int(eligible), "ta": int(ta), "nil": int(nil),
            "abstain": 0, "no_recorded_vote": int(nr),
        }
    mismatches = {
        party: {"production": result.get(party), "approved": expected}
        for party, expected in approved_main.items()
        if result.get(party) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Date-correct displayed party breakdown differs from approved target for {division_id}: {mismatches}"
        )

    displayed = set(approved_main)
    smaller = {party: counts for party, counts in result.items() if party not in displayed}
    if sum(x["eligible"] for x in result.values()) != len(eligible_members):
        raise RuntimeError(f"Party eligible totals do not reconcile to membership for {division_id}")
    if sum(x["ta"] for x in result.values()) != recorded["ta"]:
        raise RuntimeError(f"Party Tá totals do not reconcile to overall result for {division_id}")
    if sum(x["nil"] for x in result.values()) != recorded["nil"]:
        raise RuntimeError(f"Party Níl totals do not reconcile to overall result for {division_id}")
    if sum(x["abstain"] for x in result.values()) != recorded["abstain"]:
        raise RuntimeError(f"Party abstention totals do not reconcile to overall result for {division_id}")
    if sum(x["no_recorded_vote"] for x in result.values()) != no_recorded:
        raise RuntimeError(f"Party no-recorded totals do not reconcile to overall result for {division_id}")

    overall = {
        "ta": recorded["ta"], "nil": recorded["nil"], "abstain": recorded["abstain"],
        "no_recorded_vote": no_recorded, "eligible": len(eligible_members),
    }
    return {
        "division_id": division_id,
        "official_proposition_record": official,
        "date": target_date.isoformat(),
        "overall": overall,
        "party_breakdown": result,
        "smaller_groups": smaller,
        "ambiguous_party_histories": [],
        "eligible_member_count": len(eligible_members),
    }


def _render_slide(path: Path, fn, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = fn(payload, path)
    _assert_image(path)
    return manifest


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    period = (period_spec or "post1").strip().lower()
    series, post = _load_content(period)

    batch = resolve_validated_production_batch()
    tables, lineage = load_csv_tables(
        batch,
        ["silver_divisions", "silver_member_votes", "silver_member_memberships", "silver_member_parties"],
    )
    validations: dict[str, Any] = {}
    for bill in post["bills"]:
        if bill.get("vote"):
            validations[str(bill["key"])] = _validate_vote(
                bill["vote"], tables, bill_title=str(bill["formal_title"])
            )
        else:
            validations[str(bill["key"])] = {
                "recorded_division": False,
                "treatment": "procedure_explainer",
            }

    root = output_root / f"period={period}"
    if root.exists():
        shutil.rmtree(root)
    slides_dir = root / "slides"
    metadata_dir = root / "metadata"
    contact_dir = root / "contact_sheets"
    for directory in (slides_dir, metadata_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slides: list[Path] = []
    render_manifests: dict[str, Any] = {}
    p = slides_dir / "01_cover.png"
    render_manifests["01_cover"] = _render_slide(p, lambda x, out: render_cover(x, series, out), post)
    slides.append(p)
    slide_no = 2
    for bill in post["bills"]:
        p = slides_dir / f"{slide_no:02d}_{bill['key']}_explainer.png"
        render_manifests[p.stem] = _render_slide(p, render_explainer, bill)
        slides.append(p)
        slide_no += 1
        if bill.get("vote"):
            p = slides_dir / f"{slide_no:02d}_{bill['key']}_vote.png"
            render_manifests[p.stem] = _render_slide(p, render_party_vote, bill["vote"])
        else:
            p = slides_dir / f"{slide_no:02d}_{bill['key']}_procedure.png"
            render_manifests[p.stem] = _render_slide(p, render_procedure, bill["procedure"])
        slides.append(p)
        slide_no += 1
    p = slides_dir / "08_process_glossary.png"
    render_manifests[p.stem] = _render_slide(p, render_process_glossary, series["glossary"])
    slides.append(p)
    p = slides_dir / "09_vote_glossary.png"
    render_manifests[p.stem] = _render_slide(p, render_vote_glossary, series["glossary"])
    slides.append(p)

    if len(slides) != 9:
        raise RuntimeError(f"Bill Tracker {period} rendered {len(slides)} slides, expected 9")
    contact_path = contact_dir / f"{period}_contact_sheet.jpg"
    labels = [Path(p).stem.replace("_", " ").title() for p in slides]
    contact_sheet(list(zip(labels, slides)), contact_path, columns=3)

    caption_path = root / "caption.txt"
    caption_path.write_text(str(post["caption_draft"]).strip() + "\n", encoding="utf-8")
    manifest = {
        "project_id": PROJECT_ID,
        "period_key": period,
        "review_state": "pending_human_review",
        "publication_enabled": False,
        "publishing_allowed": False,
        "source_batch_id": batch.batch_id,
        "source_pointer": batch.pointer,
        "source_lineage": lineage,
        "vote_validations": validations,
        "slides": [str(p) for p in slides],
        "contact_sheets": {period: str(contact_path)},
        "caption": str(caption_path),
        "render_manifests": render_manifests,
        "qa": {
            "expected_slide_count": 9,
            "actual_slide_count": len(slides),
            "dimensions": [1080, 1350],
            "source_footer_required": True,
            "publication_enabled": False,
            "publishing_allowed": False,
            "vote_revalidation_passed": True,
        },
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    package = deterministic_zip(root, root / f"bill_tracker_{period}_review.zip")
    manifest["package"] = package
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest
