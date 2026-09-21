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
from instagram.factory.render_primitives import ACCENT, BG, MUTED, TEXT
from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar
from political_metrics.calculators.questions import member_question_metrics, prepare_eligible_td_questions
from political_metrics.commission import filter_period
from political_metrics.periods import resolve_period

PROJECT_ID = "pq_monthly_overview_v1"


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

    max_items = int((project.get("metrics") or {}).get("max_items", 10))
    top = member.sort_values(
        ["question_count", "question_day_count", "member_name"],
        ascending=[False, False, True],
    ).head(max_items).copy()
    if top.empty:
        raise RuntimeError(f"No eligible TD question counts to rank for {period_key}")

    top["chart_label"] = top["member_name"] + " (" + top["display_party_name"] + ")"
    rows = [{"label": row.chart_label, "value": int(row.question_count)} for row in top.itertuples(index=False)]
    top_asker_records = [
        {
            "member_code": row.member_code,
            "member_name": row.member_name,
            "party_name": row.display_party_name,
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

    assets_dir = period_root / "assets"
    metadata_dir = period_root / "metadata"
    slides_dir = period_root / "slides"
    assets_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    visual_path = assets_dir / "01_top_askers-visual.png"
    visual_metadata_path = metadata_dir / "01_top_askers-visual.json"
    visual_manifest_path = metadata_dir / "01_top_askers-visual-manifest.json"

    sample = {
        "visual_id": f"{PROJECT_ID}-top_askers-{period_key}",
        "bindings": {"label": "label", "value": "value"},
        "source_note": f"{period_label} Dáil parliamentary questions · Houses of the Oireachtas / Eirepolitic",
        "empty_message": "No data available",
    }
    slide_title = "Most questions submitted"
    visual_manifest = horizontal_bar.render(
        _variant_template(project, "integer"),
        sample,
        rows,
        visual_path,
        visual_metadata_path,
        visual_manifest_path,
        {
            "project_id": PROJECT_ID,
            "source_batch_id": batch.batch_id,
            "period_start": period.start.isoformat(),
            "period_end": period.end.isoformat(),
            "metric_id": "top_askers",
        },
    )
    if visual_manifest.get("warnings"):
        raise RuntimeError(f"Visual QA warnings for top_askers: {visual_manifest['warnings']}")

    render_cfg = project.get("render") or {}
    layout_path = Path(str(render_cfg["outer_layout"]))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    slide_path = slides_dir / "01_top_askers.png"
    body_text = f"Top {len(rows)} TDs by parliamentary questions submitted, {period_label}"
    footer_text = sample["source_note"]
    rendered = render_template(
        layout,
        {
            "slide_title": slide_title,
            "body_text": body_text,
            "main_media": str(visual_path),
            "footer_text": footer_text,
        },
        slide_path,
    )
    if rendered.warnings:
        raise RuntimeError(f"Outer layout warnings for top_askers: {rendered.warnings}")

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
        "slides": [
            {
                "id": "top_askers",
                "title": slide_title,
                "path": str(slide_path.relative_to(period_root)),
                "visual_asset": str(visual_path.relative_to(period_root)),
                "readability": visual_manifest.get("readability") or {},
                "rows": top_asker_records,
            }
        ],
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
    manifest_path = period_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "status": "PASS",
        "project_id": PROJECT_ID,
        "period": period_key,
        "source_batch_id": batch.batch_id,
        "slide_count": 1,
        "output_root": str(period_root),
        "manifest_path": str(manifest_path),
        "slides": [str(slide_path)],
        "duplicate_question_id_count": dedupe_stats["duplicate_question_id_count"],
        "duplicate_row_count_removed": dedupe_stats["duplicate_row_count_removed"],
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
