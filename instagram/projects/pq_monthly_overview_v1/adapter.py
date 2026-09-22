from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

from instagram.factory.oireachtas_source import (
    load_csv_tables,
    require_completed_calendar_month,
    resolve_validated_production_batch,
)
from instagram.factory.party_asset_registry import resolve_party_asset
from instagram.factory.render_primitives import ACCENT, BG, MUTED, TEXT
from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar, horizontal_bar_grouped
from political_metrics.calculators.questions import (
    grouped_question_metrics,
    member_question_metrics,
    national_question_metrics,
    prepare_eligible_td_questions,
    question_type_distribution,
    recipient_distribution,
)
from political_metrics.commission import filter_period
from political_metrics.periods import resolve_period
from political_metrics.temporal_joins import attach_event_party, temporal_join

PROJECT_ID = "pq_monthly_overview_v1"

# First-cut party acronyms for "Name (XX)" style labels (top_askers option A /
# fewest_askers). These are the common short forms used in Irish political
# coverage, not derived from any registry — if this needs to become official,
# promote to a proper reference file (alongside configs/reference/party_assets_v1.csv)
# rather than editing this dict further.
PARTY_ACRONYM = {
    "100-rdr": "RDR",
    "aontu": "AON",
    "fianna-fail": "FF",
    "fine-gael": "FG",
    "green-party": "GP",
    "independent": "IND",
    "independent-ireland": "II",
    "labour-party": "LAB",
    "people-before-profit-solidarity": "PBPS",
    "sinn-fein": "SF",
    "social-democrats": "SD",
}

# Abstract categorical identity colors for the top_askers colored-bars-with-legend
# slide (Warren's approved visual direction, 2026-09-22, session
# 2026-09-21-monthly-questions-overview). Values are the dataviz skill's validated
# default categorical palette (dark mode), validated against this project's
# #0f2f24 chart background. These are NOT official party brand colors; they exist
# only to make each bar's party visually distinguishable, with the legend and
# value labels carrying the actual identity (never color alone).
PARTY_COLOR = {
    "fianna-fail": "#3987e5",
    "sinn-fein": "#d95926",
    "fine-gael": "#199e70",
    "independent-ireland": "#c98500",
    "social-democrats": "#d55181",
    "green-party": "#008300",
    "labour-party": "#e66767",
    "aontu": "#9085e9",
    "independent": "#3987e5",
    "people-before-profit-solidarity": "#d95926",
    "100-rdr": "#199e70",
}
PARTY_LEGEND_ORDER = [
    "fianna-fail",
    "sinn-fein",
    "fine-gael",
    "independent-ireland",
    "social-democrats",
    "green-party",
    "labour-party",
    "aontu",
    "independent",
    "people-before-profit-solidarity",
    "100-rdr",
]
FALLBACK_PARTY_COLOR = "#9c9c94"

ANSWER_STATUS_LABELS = {
    "ministerial_reply_present": "Reply received",
    "reply_not_received": "Reply not received",
    "unresolved_structure": "Unresolved (source record)",
}


def _period_label(period) -> str:
    return period.start.strftime("%B %Y")


def _display_party_name(name: str) -> str:
    return "Independents" if name == "Independent" else name


def _member_names(members: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["member_code", "full_name", "display_name", "first_name", "last_name"] if col in members.columns]
    data = members[cols].drop_duplicates("member_code").copy()
    if "full_name" in data.columns:
        data["member_name"] = data["full_name"]
    elif "display_name" in data.columns:
        data["member_name"] = data["display_name"]
    else:
        first = data["first_name"].fillna("") if "first_name" in data.columns else ""
        last = data["last_name"].fillna("") if "last_name" in data.columns else ""
        data["member_name"] = (first + " " + last).str.strip()
    return data[["member_code", "member_name"]]


def _dedupe_questions(questions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop duplicate question_id rows, keeping the first occurrence, and
    record how many were removed.

    Build note (session director/sessions/2026-09-21-monthly-questions-overview/):
    silver_questions has, at least in some periods, carried duplicate question_id
    values. The existing commissioning script
    (process/political_metrics_question_commission.py) hard-fails on this rather
    than deduping. A recurring monthly post cannot hard-fail on a source
    data-quality issue it doesn't control, so this adapter dedupes and records the
    exact counts in the run manifest instead. Warren reviewed this behavior
    2026-09-22 and asked to leave it as-is (not a big enough proportion to be a
    concern) rather than investigate further right now.
    """
    raw_count = int(len(questions))
    duplicated_mask = questions["question_id"].duplicated(keep=False)
    duplicate_id_count = int(questions.loc[duplicated_mask, "question_id"].nunique())
    deduped = questions.drop_duplicates(subset="question_id", keep="first").copy()
    removed_row_count = raw_count - int(len(deduped))
    return deduped, {
        "raw_row_count": raw_count,
        "deduped_row_count": int(len(deduped)),
        "duplicate_question_id_count": duplicate_id_count,
        "duplicate_row_count_removed": removed_row_count,
    }


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """load_csv_tables reads every column as dtype=str, so a source boolean
    column round-trips as the literal strings "True"/"False" (or NaN for a
    blank cell). bool("False") is True in Python, so a naive .astype(bool)
    would silently mark every row as True. Parse the strings explicitly
    instead, matching the same normalization used when this table was
    published (process/political_metrics_written_question_answers_candidate.py
    _normalize_sections).
    """
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _party_key(party_name: str) -> str:
    try:
        return resolve_party_asset(party_name).party_key
    except Exception:
        return "other"


def _period_end_roster(
    memberships: pd.DataFrame,
    member_parties: pd.DataFrame,
    members: pd.DataFrame,
    *,
    period,
) -> pd.DataFrame:
    """Active Dáil TDs as of the period's last day, with their Oireachtas-listed
    party as of that date, and whether their current membership interval began on
    or before the period's first day ("seated for the full period").

    Reimplemented here from the non-frozen political_metrics.temporal_joins
    primitives, mirroring the period-end snapshot pattern used by the (frozen,
    not edited) party_issue_monthly_profile_v2 adapter's
    _period_end_party_snapshot, rather than importing that frozen module.
    """
    known = memberships.drop_duplicates("member_code")[["member_code"]].copy()
    known["event_date"] = period.end.isoformat()
    active = temporal_join(
        known,
        memberships,
        event_date_col="event_date",
        entity_col="member_code",
        history_start_col="membership_start",
        history_end_col="membership_end",
        history_columns=["membership_id", "chamber", "membership_start", "membership_end"],
        allow_unmatched=True,
    )
    if "chamber" in active.columns:
        active = active[active["chamber"].fillna("").str.lower().eq("dail")].copy()
    active = active[active["membership_id"].notna()].copy()
    active = attach_event_party(active, member_parties, event_date_col="event_date")
    active["party_name"] = active["party_name"].fillna("Unknown")
    active["seated_full_period"] = pd.to_datetime(active["membership_start"], errors="coerce").dt.date <= period.start
    names = _member_names(members)
    active = active.merge(names, on="member_code", how="left")
    active["member_name"] = active["member_name"].fillna(active["member_code"])
    active["display_party_name"] = active["party_name"].map(_display_party_name)
    active["party_key"] = active["party_name"].map(_party_key)
    return active[
        ["member_code", "member_name", "party_name", "display_party_name", "party_key", "seated_full_period"]
    ].drop_duplicates("member_code")


def _variant_template(project: dict[str, Any], value_format: str) -> dict[str, Any]:
    return _chart_template(project, value_format=value_format)


def _chart_template(
    project: dict[str, Any],
    *,
    value_format: str,
    sort: str = "descending",
    max_items: int | None = None,
    width: int = 1032,
    height: int = 1210,
    min_visual_rows: int | None = None,
) -> dict[str, Any]:
    render_cfg = project.get("render") or {}
    palette = render_cfg.get("palette") or {}
    return {
        "template_id": "horizontal_bar_draft_v1",
        "params": {
            "width": width,
            "height": height,
            "max_items": int(max_items if max_items is not None else (project.get("metrics") or {}).get("max_items", 10)),
            "sort": sort,
            "value_format": value_format,
            "min_visual_rows": int(min_visual_rows if min_visual_rows is not None else render_cfg.get("min_visual_rows", 4)),
        },
        "palette": {
            "background": str(palette.get("background") or BG),
            "panel": str(palette.get("background") or BG),
            "text": str(palette.get("text") or TEXT),
            "muted": str(palette.get("muted") or MUTED),
            "accent": str(palette.get("accent") or ACCENT),
            "grid": str(palette.get("grid") or TEXT),
        },
    }


def _render_slide(
    *,
    variant_id: str,
    slide_title: str,
    body_text: str,
    rows: list[dict[str, Any]],
    renderer_module,
    render_kwargs: dict[str, Any],
    period_root: Path,
    layout: dict[str, Any],
    slide_index: int,
) -> dict[str, Any]:
    assets_dir = period_root / "assets"
    metadata_dir = period_root / "metadata"
    slides_dir = period_root / "slides"
    visual_path = assets_dir / f"{slide_index:02d}_{variant_id}-visual.png"
    visual_metadata_path = metadata_dir / f"{slide_index:02d}_{variant_id}-visual.json"
    visual_manifest_path = metadata_dir / f"{slide_index:02d}_{variant_id}-visual-manifest.json"

    visual_manifest = renderer_module.render(
        render_kwargs["template"],
        render_kwargs["sample"],
        rows,
        visual_path,
        visual_metadata_path,
        visual_manifest_path,
        render_kwargs["input_metadata"],
    )
    if visual_manifest.get("warnings"):
        raise RuntimeError(f"Visual QA warnings for {variant_id}: {visual_manifest['warnings']}")

    slide_path = slides_dir / f"{slide_index:02d}_{variant_id}.png"
    rendered = render_template(
        layout,
        {
            "slide_title": slide_title,
            "body_text": body_text,
            "main_media": str(visual_path),
            "footer_text": str(render_kwargs["sample"].get("source_note") or ""),
        },
        slide_path,
    )
    if rendered.warnings:
        raise RuntimeError(f"Outer layout warnings for {variant_id}: {rendered.warnings}")

    return {
        "id": variant_id,
        "title": slide_title,
        "path": str(slide_path.relative_to(period_root)),
        "visual_asset": str(visual_path.relative_to(period_root)),
        "readability": visual_manifest.get("readability") or {},
        "slide_path_abs": str(slide_path),
    }


def _render_text_slide(
    *,
    slide_id: str,
    slide_title: str,
    lines: list[str],
    footer_text: str,
    period_root: Path,
    layout: dict[str, Any],
    slide_index: int,
) -> dict[str, Any]:
    slides_dir = period_root / "slides"
    slide_path = slides_dir / f"{slide_index:02d}_{slide_id}.png"
    bindings: dict[str, Any] = {"slide_title": slide_title, "footer_text": footer_text}
    for i in range(6):
        bindings[f"line_{i + 1}"] = lines[i] if i < len(lines) else ""
    rendered = render_template(layout, bindings, slide_path)
    if rendered.warnings:
        raise RuntimeError(f"Text layout warnings for {slide_id}: {rendered.warnings}")
    return {
        "id": slide_id,
        "title": slide_title,
        "path": str(slide_path.relative_to(period_root)),
        "slide_path_abs": str(slide_path),
    }


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    period = resolve_period(period_spec)
    require_completed_calendar_month(period)
    period_key = period.start.strftime("%Y-%m")
    period_label = _period_label(period)

    s3 = boto3.client("s3", region_name="ca-central-1")
    batch = resolve_validated_production_batch(s3=s3)
    required_tables = [str(value) for value in ((project.get("source") or {}).get("required_tables") or [])]
    frames, source_lineage = load_csv_tables(batch, required_tables, s3=s3)

    memberships = frames["silver_member_memberships"]
    member_parties = frames["silver_member_parties"]
    member_constituencies = frames["silver_member_constituencies"]
    members_table = frames["silver_members"]

    raw_questions = frames["silver_questions"]
    period_questions = filter_period(raw_questions, "question_date", period)
    deduped_questions, dedupe_stats = _dedupe_questions(period_questions)

    eligible = prepare_eligible_td_questions(deduped_questions, memberships, member_parties, member_constituencies)
    if eligible.empty:
        raise RuntimeError(f"No Dáil-eligible questions found for {period_key}")

    # --- previous-month comparison (same batch, same tables, different date filter) ---
    prev_start = (period.start.replace(day=1) - timedelta(days=1)).replace(day=1)
    prev_period = resolve_period(prev_start.strftime("%Y-%m"))
    prev_questions = filter_period(raw_questions, "question_date", prev_period)
    prev_deduped, _prev_dedupe_stats = _dedupe_questions(prev_questions)
    prev_eligible = prepare_eligible_td_questions(prev_deduped, memberships, member_parties, member_constituencies)
    national_current = national_question_metrics(eligible)
    national_prev = national_question_metrics(prev_eligible)
    question_count_delta = national_current["question_count"] - national_prev["question_count"]

    type_dist = question_type_distribution(eligible)
    written_count = int(
        type_dist.loc[type_dist["question_type"].str.lower() == "written", "question_count"].sum()
    ) if not type_dist.empty else 0
    oral_count = int(
        type_dist.loc[type_dist["question_type"].str.lower() == "oral", "question_count"].sum()
    ) if not type_dist.empty else 0

    roster = _period_end_roster(memberships, member_parties, members_table, period=period)

    period_root = output_root / f"period={period_key}"
    if period_root.exists():
        # Re-generation is deterministic from source lineage, but stale files from a
        # previous failed attempt must never leak into the new package.
        shutil.rmtree(period_root)
    period_root.mkdir(parents=True, exist_ok=True)
    (period_root / "assets").mkdir(parents=True, exist_ok=True)
    (period_root / "metadata").mkdir(parents=True, exist_ok=True)
    (period_root / "slides").mkdir(parents=True, exist_ok=True)

    render_cfg = project.get("render") or {}
    layout_path = Path(str(render_cfg["outer_layout"]))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    text_layout_path = Path(str(render_cfg["text_layout"]))
    text_layout = json.loads(text_layout_path.read_text(encoding="utf-8"))
    source_note = f"{period_label} Dáil parliamentary questions · Houses of the Oireachtas / Eirepolitic"

    slides: list[dict[str, Any]] = []
    caveats: list[str] = []

    # ===== Slide 1: headline =====
    delta_sign = "+" if question_count_delta >= 0 else ""
    headline_lines = [
        f"{national_current['question_count']:,} parliamentary questions recorded",
        f"{written_count:,} written · {oral_count:,} oral",
        f"{national_current['asking_member_count']} TDs asked at least one question",
        f"Questions were asked on {national_current['question_day_count']} sitting day(s)",
        f"{delta_sign}{question_count_delta:,} vs {prev_period.start.strftime('%B %Y')} ({national_prev['question_count']:,} questions)",
    ]
    slide_headline = _render_text_slide(
        slide_id="headline",
        slide_title=f"Parliamentary Questions: {period_label}",
        lines=headline_lines,
        footer_text=source_note,
        period_root=period_root,
        layout=text_layout,
        slide_index=1,
    )
    slides.append(slide_headline)

    # ===== Slide 2: top_askers (Warren's approved visual direction — option B,
    # colored bars with a 3-column/2-row legend, 2026-09-22) =====
    member_counts = member_question_metrics(eligible)
    member_counts = member_counts.merge(_member_names(members_table), on="member_code", how="left")
    party_source = eligible.copy()
    party_source["party_name"] = party_source["party_name"].fillna("Unknown")
    party_lookup = (
        party_source.groupby("member_code")["party_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"party_name": "party_name_resolved"})
    )
    member_counts = member_counts.merge(party_lookup, on="member_code", how="left")
    member_counts["member_name"] = member_counts["member_name"].fillna(member_counts["member_code"])
    member_counts["party_name_resolved"] = member_counts["party_name_resolved"].fillna("Unknown")
    member_counts["display_party_name"] = member_counts["party_name_resolved"].map(_display_party_name)
    member_counts["party_key"] = member_counts["party_name_resolved"].map(_party_key)

    top_max_items = int((project.get("metrics") or {}).get("max_items", 10))
    top = member_counts.sort_values(
        ["question_count", "question_day_count", "member_name"],
        ascending=[False, False, True],
    ).head(top_max_items).copy()
    if top.empty:
        raise RuntimeError(f"No eligible TD question counts to rank for {period_key}")

    top_asker_records = [
        {
            "member_code": row.member_code,
            "member_name": row.member_name,
            "party_name": row.display_party_name,
            "party_key": row.party_key,
            "question_count": int(row.question_count),
            "question_day_count": int(row.question_day_count),
        }
        for row in top.itertuples(index=False)
    ]
    rows_colored = [
        {"label": row.member_name, "value": int(row.question_count), "group": row.party_key}
        for row in top.itertuples(index=False)
    ]
    group_legend_labels: dict[str, str] = {row.party_key: row.display_party_name for row in top.itertuples(index=False)}
    slide_top_askers = _render_slide(
        variant_id="top_askers",
        slide_title="Most questions submitted",
        body_text=f"Top {len(rows_colored)} TDs by parliamentary questions submitted, {period_label}. Colored by party — see legend.",
        rows=rows_colored,
        renderer_module=horizontal_bar_grouped,
        render_kwargs={
            "template": _variant_template(project, "integer"),
            "sample": {
                "visual_id": f"{PROJECT_ID}-top_askers-{period_key}",
                "bindings": {"label": "label", "value": "value", "group": "group"},
                "source_note": source_note,
                "empty_message": "No data available",
                "group_colors": PARTY_COLOR,
                "group_legend_labels": group_legend_labels,
                "group_legend_order": PARTY_LEGEND_ORDER,
                "group_fallback_color": FALLBACK_PARTY_COLOR,
            },
            "input_metadata": {
                "project_id": PROJECT_ID,
                "source_batch_id": batch.batch_id,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "metric_id": "top_askers",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=2,
    )
    slides.append(slide_top_askers)

    # ===== Slide 3: fewest_askers (bottom 10 eligible TDs, seated the full period) =====
    fewest_max_items = 10
    full_month_roster = roster[roster["seated_full_period"]].copy()
    fewest = full_month_roster.merge(
        member_counts[["member_code", "question_count"]], on="member_code", how="left"
    )
    fewest["question_count"] = fewest["question_count"].fillna(0).astype(int)
    fewest["acronym"] = fewest["party_key"].map(lambda key: PARTY_ACRONYM.get(key, key.upper()[:3]))
    fewest["chart_label"] = fewest["member_name"] + " (" + fewest["acronym"] + ")"
    fewest_sorted = fewest.sort_values(
        ["question_count", "member_name"], ascending=[True, True]
    ).head(fewest_max_items).copy()
    caveats.append(
        "Fewest questions submitted does not exclude office-holders or party leaders from the ranking — "
        "no reference list of current office-holders/party leaders exists yet in this pipeline. This is a "
        "data gap, not an editorial choice; Warren, let us know if you want a hand-maintained list added."
    )
    fewest_rows = [
        {"label": row.chart_label, "value": int(row.question_count)} for row in fewest_sorted.itertuples(index=False)
    ]
    fewest_records = [
        {
            "member_code": row.member_code,
            "member_name": row.member_name,
            "party_name": row.display_party_name,
            "question_count": int(row.question_count),
        }
        for row in fewest_sorted.itertuples(index=False)
    ]
    slide_fewest = _render_slide(
        variant_id="fewest_askers",
        slide_title="Fewest questions submitted",
        body_text=(
            f"The {len(fewest_rows)} TDs seated for the whole of {period_label} with the fewest recorded "
            "parliamentary questions. Does not exclude office-holders or party leaders — see methodology."
        ),
        rows=fewest_rows,
        renderer_module=horizontal_bar,
        render_kwargs={
            "template": _chart_template(project, value_format="integer", sort="ascending", max_items=fewest_max_items),
            "sample": {
                "visual_id": f"{PROJECT_ID}-fewest_askers-{period_key}",
                "bindings": {"label": "label", "value": "value"},
                "source_note": source_note,
                "empty_message": "No data available",
            },
            "input_metadata": {
                "project_id": PROJECT_ID,
                "source_batch_id": batch.batch_id,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "metric_id": "fewest_askers",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=3,
    )
    slides.append(slide_fewest)

    # ===== Slide 4: party_per_td (questions per TD, by party, period-end roster) =====
    td_counts = (
        roster.groupby(["party_name", "display_party_name"]).size().reset_index(name="td_count")
    )
    party_question_counts = grouped_question_metrics(eligible, group_col="party_name")[["party_name", "question_count"]]
    party_per_td = td_counts.merge(party_question_counts, on="party_name", how="left")
    party_per_td["question_count"] = party_per_td["question_count"].fillna(0).astype(int)
    party_per_td = party_per_td[party_per_td["td_count"] > 0].copy()
    party_per_td["rate"] = party_per_td["question_count"] / party_per_td["td_count"]
    party_per_td_sorted = party_per_td.sort_values(["rate", "question_count"], ascending=[False, False])
    party_per_td_rows = [
        {"label": row.display_party_name, "value": round(float(row.rate), 2)}
        for row in party_per_td_sorted.itertuples(index=False)
    ]
    party_per_td_records = [
        {
            "party_name": row.display_party_name,
            "td_count": int(row.td_count),
            "question_count": int(row.question_count),
            "questions_per_td": round(float(row.rate), 2),
        }
        for row in party_per_td_sorted.itertuples(index=False)
    ]
    total_tds_at_period_end = int(len(roster))
    slide_party_per_td = _render_slide(
        variant_id="party_per_td",
        slide_title="Questions per TD by party",
        body_text=(
            f"Parliamentary questions submitted per TD, by party, {period_label}. Based on {total_tds_at_period_end} "
            f"TDs serving as of {period.end.strftime('%-d %B %Y')} and their Oireachtas-listed party on that date."
        ),
        rows=party_per_td_rows,
        renderer_module=horizontal_bar,
        render_kwargs={
            "template": _chart_template(project, value_format="decimal_2", sort="descending", max_items=12),
            "sample": {
                "visual_id": f"{PROJECT_ID}-party_per_td-{period_key}",
                "bindings": {"label": "label", "value": "value"},
                "source_note": source_note,
                "empty_message": "No data available",
            },
            "input_metadata": {
                "project_id": PROJECT_ID,
                "source_batch_id": batch.batch_id,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "metric_id": "party_per_td",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=4,
    )
    slides.append(slide_party_per_td)

    # ===== Slide 5: departments (top recipients by recorded question count) =====
    recipients = recipient_distribution(eligible)
    recipients_sorted = recipients.sort_values("question_count", ascending=False).head(8).copy()
    departments_rows = [
        {"label": row.to_minister_or_department, "value": int(row.question_count)}
        for row in recipients_sorted.itertuples(index=False)
    ]
    departments_records = [
        {
            "recipient": row.to_minister_or_department,
            "question_count": int(row.question_count),
            "question_share": None if pd.isna(row.question_share) else round(float(row.question_share), 4),
        }
        for row in recipients_sorted.itertuples(index=False)
    ]
    slide_departments = _render_slide(
        variant_id="departments",
        slide_title="Departments asked most",
        body_text=(
            f"Ministers and departments that received the most recorded parliamentary questions, {period_label}. "
            "Recipient labels are taken directly from the Oireachtas source record."
        ),
        rows=departments_rows,
        renderer_module=horizontal_bar,
        render_kwargs={
            "template": _chart_template(project, value_format="integer", sort="descending", max_items=8),
            "sample": {
                "visual_id": f"{PROJECT_ID}-departments-{period_key}",
                "bindings": {"label": "label", "value": "value"},
                "source_note": source_note,
                "empty_message": "No data available",
            },
            "input_metadata": {
                "project_id": PROJECT_ID,
                "source_batch_id": batch.batch_id,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "metric_id": "departments",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=5,
    )
    slides.append(slide_departments)

    # ===== Slide 6: answers (what happened to written-question answers) =====
    answer_sections = frames["written_question_answer_sections"][
        ["debate_section_id", "answer_status", "referred_or_direct_reply"]
    ].copy()
    answer_sections["referred_or_direct_reply"] = _parse_bool_series(answer_sections["referred_or_direct_reply"])
    answer_bridge = frames["written_question_answer_bridge"][
        ["question_id", "debate_section_id"]
    ].copy()
    written_eligible_ids = set(
        eligible.loc[eligible["question_type"].fillna("").str.strip().str.lower().eq("written"), "question_id"].astype(str)
    )
    total_written = len(written_eligible_ids)
    bridge_period = answer_bridge[answer_bridge["question_id"].astype(str).isin(written_eligible_ids)].copy()
    joined = bridge_period.merge(answer_sections, on="debate_section_id", how="left")
    covered_question_ids = set(joined.loc[joined["answer_status"].notna(), "question_id"].astype(str))
    uncovered_count = total_written - len(covered_question_ids)
    status_counts = (
        joined[joined["answer_status"].notna()]
        .drop_duplicates("question_id")
        .groupby("answer_status")["question_id"]
        .nunique()
    )
    referred_count = int(
        joined.loc[joined["question_id"].astype(str).isin(covered_question_ids), "referred_or_direct_reply"]
        .fillna(False)
        .astype(bool)
        .sum()
    )
    answers_rows = [
        {"label": ANSWER_STATUS_LABELS.get(str(status), str(status)), "value": int(count)}
        for status, count in status_counts.items()
    ]
    answers_records = {
        "total_written_questions": total_written,
        "covered_by_answer_record": len(covered_question_ids),
        "uncovered_by_answer_record": uncovered_count,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "referred_or_direct_reply_count": referred_count,
    }
    if uncovered_count > 0:
        caveats.append(
            f"{uncovered_count} of {total_written} written questions this period had no matching answer record "
            "in written_question_answer_bridge/sections and are excluded from the 'What happened to the answers' slide."
        )
    answers_body = (
        f"Of {total_written} written questions asked in {period_label}, this shows what the official record "
        f"shows happened to the answer. {referred_count} were flagged in the source as referred for a direct reply."
    )
    slide_answers = _render_slide(
        variant_id="answers",
        slide_title="What happened to the answers",
        body_text=answers_body,
        rows=answers_rows,
        renderer_module=horizontal_bar,
        render_kwargs={
            "template": _chart_template(project, value_format="integer", sort="descending", max_items=6, min_visual_rows=3),
            "sample": {
                "visual_id": f"{PROJECT_ID}-answers-{period_key}",
                "bindings": {"label": "label", "value": "value"},
                "source_note": source_note,
                "empty_message": "No data available",
            },
            "input_metadata": {
                "project_id": PROJECT_ID,
                "source_batch_id": batch.batch_id,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "metric_id": "answers",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=6,
    )
    slides.append(slide_answers)

    # ===== Slide 7: methodology =====
    methodology_lines = [
        "Source: Houses of the Oireachtas open data (Dáil Éireann), via the Eirepolitic unified data pipeline.",
        f"Data batch: {batch.batch_id}",
        "A question is eligible if asked by a TD with active Dáil membership on the question date; party and "
        "constituency reflect the TD's Oireachtas-listed affiliation on that date, not their current one.",
        "'Fewest questions submitted' and 'Questions per TD by party' do not exclude office-holders or party "
        "leaders from the TD count — a data gap, not an editorial choice.",
        "Answer-status figures (written questions only) reflect the official record as published and may not "
        "cover every question asked this period; see run notes for exact coverage.",
    ]
    if dedupe_stats["duplicate_question_id_count"] > 0:
        methodology_lines.append(
            f"{dedupe_stats['duplicate_question_id_count']} duplicate question_id record(s) were removed before analysis."
        )
    slide_methodology = _render_text_slide(
        slide_id="methodology",
        slide_title="Methodology and sources",
        lines=methodology_lines,
        footer_text=source_note,
        period_root=period_root,
        layout=text_layout,
        slide_index=7,
    )
    slides.append(slide_methodology)

    run_manifest = {
        "project_id": PROJECT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"key": period_key, "start": period.start.isoformat(), "end": period.end.isoformat(), "label": period_label},
        "source_batch_id": batch.batch_id,
        "source_pointer": batch.pointer,
        "source_batch_manifest_key": batch.pointer.get("manifest_key"),
        "source_batch_status": batch.manifest.get("status"),
        "sources": source_lineage,
        "data_quality": {
            "question_dedupe": dedupe_stats,
            "note": (
                "silver_questions contained duplicate question_id rows for this period; deduped by keeping the "
                "first occurrence per question_id before eligibility and ranking. Warren reviewed 2026-09-22 and "
                "asked to leave this as-is for now."
            ),
        },
        "calculation": {
            "eligibility": "active Dáil membership on question_date (prepare_eligible_td_questions)",
            "period_end_roster": "active Dáil membership at period.end, party as of period.end (temporal_join + attach_event_party)",
            "top_askers": "distinct question_id per member_code, deduped, descending by count then question_day_count then name",
            "fewest_askers": "period-end roster filtered to membership_start <= period.start (seated full period), left-joined to question counts (0 if none), ascending",
            "party_per_td": "grouped_question_metrics by party_name / period-end TD count per party",
            "departments": "recipient_distribution grouped by to_minister_or_department, top 8 by count",
            "answers": "written_question_answer_bridge joined to written_question_answer_sections by debate_section_id, grouped by answer_status",
            "month_over_month": "current vs previous calendar month, both computed from the same source batch (see data_products.yml known_drift_fact caveat)",
        },
        "caveats": caveats,
        "visual_direction": {
            "note": "top_askers uses Warren's approved option B design (colored bars + legend), decided 2026-09-22.",
        },
        "top_askers": top_asker_records,
        "fewest_askers": fewest_records,
        "party_per_td": party_per_td_records,
        "departments": departments_records,
        "answers": answers_records,
        "national_headline": {"current": national_current, "previous": national_prev, "written_count": written_count, "oral_count": oral_count},
        "slides": slides,
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
    slide_paths_abs = [slide["slide_path_abs"] for slide in run_manifest["slides"]]
    for slide in run_manifest["slides"]:
        slide.pop("slide_path_abs", None)
    manifest_path = period_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "status": "PASS",
        "project_id": PROJECT_ID,
        "period": period_key,
        "source_batch_id": batch.batch_id,
        "slide_count": len(slides),
        "output_root": str(period_root),
        "manifest_path": str(manifest_path),
        "slides": slide_paths_abs,
        "duplicate_question_id_count": dedupe_stats["duplicate_question_id_count"],
        "duplicate_row_count_removed": dedupe_stats["duplicate_row_count_removed"],
        "caveats": caveats,
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
