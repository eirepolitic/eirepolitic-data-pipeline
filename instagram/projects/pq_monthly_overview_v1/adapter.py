from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
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
from political_metrics.calculators.questions import member_question_metrics, prepare_eligible_td_questions
from political_metrics.commission import filter_period
from political_metrics.periods import resolve_period

PROJECT_ID = "pq_monthly_overview_v1"

# First-cut party acronyms for the "Name (XX)" label option (visual-direction
# option A, session 2026-09-21-monthly-questions-overview). These are the
# common short forms used in Irish political coverage, not derived from any
# registry — if Warren picks this direction, promote to a proper reference
# file (alongside configs/reference/party_assets_v1.csv) rather than editing
# this dict further.
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

# Abstract categorical identity colors for the colored-bars-with-legend option
# (visual-direction option B). Values are the dataviz skill's validated
# default categorical palette (dark mode), validated against this project's
# #0f2f24 chart background — see the session log for the validator run.
# These are NOT official party brand colors; they exist only to make each
# bar's party visually distinguishable, with the legend and value labels
# carrying the actual identity (never color alone). The eight parties/groups
# most likely to appear in a top-10 TD chart each get a distinct slot from
# the 8-hue palette; the three smallest/rarest groups reuse the nearest slot
# — see the session log for the caveat on what happens if a future month's
# top 10 ever needs to distinguish more than 8 of these at once.
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
    July 2026 carries duplicate question_id values in silver_questions. The
    existing commissioning script (process/political_metrics_question_commission.py)
    hard-fails on this rather than deduping. A recurring monthly post cannot
    hard-fail on a source data-quality issue it doesn't control, so this
    adapter dedupes and records the exact counts in the run manifest instead.
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


def _party_key(party_name: str) -> str:
    try:
        return resolve_party_asset(party_name).party_key
    except Exception:
        return "other"


def _variant_template(project: dict[str, Any], value_format: str) -> dict[str, Any]:
    render_cfg = project.get("render") or {}
    palette = render_cfg.get("palette") or {}
    return {
        "template_id": "horizontal_bar_draft_v1",
        "params": {
            "width": 1032,
            "height": 1210,
            "max_items": int((project.get("metrics") or {}).get("max_items", 10)),
            "sort": "descending",
            "value_format": value_format,
            "min_visual_rows": int(render_cfg.get("min_visual_rows", 4)),
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


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    period = resolve_period(period_spec)
    require_completed_calendar_month(period)
    period_key = period.start.strftime("%Y-%m")
    period_label = _period_label(period)

    s3 = boto3.client("s3", region_name="ca-central-1")
    batch = resolve_validated_production_batch(s3=s3)
    required_tables = [str(value) for value in ((project.get("source") or {}).get("required_tables") or [])]
    frames, source_lineage = load_csv_tables(batch, required_tables, s3=s3)

    raw_questions = frames["silver_questions"]
    period_questions = filter_period(raw_questions, "question_date", period)
    deduped_questions, dedupe_stats = _dedupe_questions(period_questions)

    eligible = prepare_eligible_td_questions(
        deduped_questions,
        frames["silver_member_memberships"],
        frames["silver_member_parties"],
        frames["silver_member_constituencies"],
    )
    if eligible.empty:
        raise RuntimeError(f"No Dáil-eligible questions found for {period_key}")

    member = member_question_metrics(eligible)
    member = member.merge(_member_names(frames["silver_members"]), on="member_code", how="left")

    party_source = eligible.copy()
    party_source["party_name"] = party_source["party_name"].fillna("Unknown")
    party_lookup = (
        party_source.groupby("member_code")["party_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"party_name": "party_name_resolved"})
    )
    member = member.merge(party_lookup, on="member_code", how="left")
    member["member_name"] = member["member_name"].fillna(member["member_code"])
    member["party_name_resolved"] = member["party_name_resolved"].fillna("Unknown")
    member["display_party_name"] = member["party_name_resolved"].map(_display_party_name)
    member["party_key"] = member["party_name_resolved"].map(_party_key)

    max_items = int((project.get("metrics") or {}).get("max_items", 10))
    top = member.sort_values(
        ["question_count", "question_day_count", "member_name"],
        ascending=[False, False, True],
    ).head(max_items).copy()
    if top.empty:
        raise RuntimeError(f"No eligible TD question counts to rank for {period_key}")

    top["acronym"] = top["party_key"].map(lambda key: PARTY_ACRONYM.get(key, key.upper()[:3]))
    top["chart_label_acronym"] = top["member_name"] + " (" + top["acronym"] + ")"

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
    source_note = f"{period_label} Dáil parliamentary questions · Houses of the Oireachtas / Eirepolitic"

    slide_title = "Most questions submitted"

    # --- Option A: horizontal_bar.py (unmodified), party acronym in the label ---
    rows_acronym = [{"label": row.chart_label_acronym, "value": int(row.question_count)} for row in top.itertuples(index=False)]
    slide_a = _render_slide(
        variant_id="top_askers_v1_acronym",
        slide_title=slide_title,
        body_text=f"Option A — party acronym · Top {len(rows_acronym)} TDs, {period_label}",
        rows=rows_acronym,
        renderer_module=horizontal_bar,
        render_kwargs={
            "template": _variant_template(project, "integer"),
            "sample": {
                "visual_id": f"{PROJECT_ID}-top_askers_v1_acronym-{period_key}",
                "bindings": {"label": "label", "value": "value"},
                "source_note": source_note,
                "empty_message": "No data available",
            },
            "input_metadata": {
                "project_id": PROJECT_ID,
                "source_batch_id": batch.batch_id,
                "period_start": period.start.isoformat(),
                "period_end": period.end.isoformat(),
                "metric_id": "top_askers_v1_acronym",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=1,
    )

    # --- Option B: horizontal_bar_grouped.py, colored by party + legend ---
    rows_colored = [
        {"label": row.member_name, "value": int(row.question_count), "group": row.party_key}
        for row in top.itertuples(index=False)
    ]
    group_legend_labels: dict[str, str] = {}
    for row in top.itertuples(index=False):
        group_legend_labels[row.party_key] = row.display_party_name
    slide_b = _render_slide(
        variant_id="top_askers_v2_colored_legend",
        slide_title=slide_title,
        body_text=f"Option B — colored by party (see legend) · Top {len(rows_colored)} TDs, {period_label}",
        rows=rows_colored,
        renderer_module=horizontal_bar_grouped,
        render_kwargs={
            "template": _variant_template(project, "integer"),
            "sample": {
                "visual_id": f"{PROJECT_ID}-top_askers_v2_colored_legend-{period_key}",
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
                "metric_id": "top_askers_v2_colored_legend",
            },
        },
        period_root=period_root,
        layout=layout,
        slide_index=2,
    )

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
                "silver_questions contained duplicate question_id rows for this period; "
                "deduped by keeping the first occurrence per question_id before eligibility "
                "and ranking. See dedupe stats above for the exact counts."
            ),
        },
        "calculation": {
            "eligibility": "active Dáil membership on question_date (prepare_eligible_td_questions)",
            "ranking": "distinct question_id per member_code, deduped, descending by count then question_day_count then name",
            "max_items": max_items,
        },
        "visual_direction_options": {
            "note": "Two variants of the same slide, for Warren's visual-direction decision (workflows_v1.md §6.1 step 4).",
            "option_a": "top_askers_v1_acronym — horizontal_bar.py unmodified, label = 'Name (XX)' party acronym",
            "option_b": "top_askers_v2_colored_legend — new horizontal_bar_grouped.py renderer, bars colored by party with a legend",
        },
        "top_askers": top_asker_records,
        "slides": [slide_a, slide_b],
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
    slide_a_path_abs = slide_a["slide_path_abs"]
    slide_b_path_abs = slide_b["slide_path_abs"]
    for slide in run_manifest["slides"]:
        slide.pop("slide_path_abs", None)
    manifest_path = period_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "status": "PASS",
        "project_id": PROJECT_ID,
        "period": period_key,
        "source_batch_id": batch.batch_id,
        "slide_count": 2,
        "output_root": str(period_root),
        "manifest_path": str(manifest_path),
        "slides": [slide_a_path_abs, slide_b_path_abs],
        "duplicate_question_id_count": dedupe_stats["duplicate_question_id_count"],
        "duplicate_row_count_removed": dedupe_stats["duplicate_row_count_removed"],
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
