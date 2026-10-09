from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from instagram.projects.pq_monthly_overview_v1 import adapter as base
from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar_grouped_singleline, horizontal_bar_review

HEADLINE_LAYOUT = Path("instagram/projects/pq_monthly_overview_v1/headline_layout_v1.json")
DESCRIPTOR_LAYOUT = Path("instagram/projects/pq_monthly_overview_v1/descriptor_layout_v1.json")
_original_render_text_slide = base._render_text_slide
_original_render_slide = base._render_slide
_fewest_override: dict[str, Any] | None = None

# Review-only chart adjustments.
# Top/fewest-asker grouped charts use the approved one-line grouped renderer.
# Standard horizontal bars delegate through horizontal_bar_review, which only
# changes party_per_td and leaves all other horizontal-bar slides unchanged.
base.horizontal_bar_grouped = horizontal_bar_grouped_singleline
base.horizontal_bar = horizontal_bar_review
base.PARTY_COLOR = {
    "fianna-fail": "#4583cb",
    "sinn-fein": "#c15f36",
    "fine-gael": "#2c9570",
    "independent-ireland": "#b58218",
    "social-democrats": "#be597d",
    "green-party": "#188018",
    "labour-party": "#cc6a69",
    "aontu": "#8982ce",
    "independent": "#4583cb",
    "people-before-profit-solidarity": "#c15f36",
    "100-rdr": "#2c9570",
}


def _render_slide_review(
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
    if variant_id == "departments":
        slide_title = "Most Asked Departments"
    return _original_render_slide(
        variant_id=variant_id,
        slide_title=slide_title,
        body_text=body_text,
        rows=rows,
        renderer_module=renderer_module,
        render_kwargs=render_kwargs,
        period_root=period_root,
        layout=layout,
        slide_index=slide_index,
    )


base._render_slide = _render_slide_review


def _first_int(text: str) -> str:
    match = re.search(r"[+-]?[\d,]+", text)
    return match.group(0) if match else ""


def _prepare_fewest_override(project: dict[str, Any], period_spec: str) -> dict[str, Any]:
    """Compute the bottom-10 eligible non-office-holder TDs for the review slide."""
    period = base.resolve_period(period_spec)
    base.require_completed_calendar_month(period)
    period_key = period.start.strftime("%Y-%m")
    period_label = base._period_label(period)

    s3 = base.boto3.client("s3", region_name="ca-central-1")
    batch = base.resolve_validated_production_batch(s3=s3)
    required_tables = [str(v) for v in ((project.get("source") or {}).get("required_tables") or [])]
    frames, _ = base.load_csv_tables(batch, required_tables, s3=s3)

    period_questions = base.filter_period(frames["silver_questions"], "question_date", period)
    deduped_questions, _ = base._dedupe_questions(period_questions)
    eligible = base.prepare_eligible_td_questions(
        deduped_questions,
        frames["silver_member_memberships"],
        frames["silver_member_parties"],
        frames["silver_member_constituencies"],
    )
    if eligible.empty:
        raise RuntimeError(f"No Dáil-eligible questions found for {period_key}")

    member_counts = base.member_question_metrics(eligible)
    roster = base._period_end_roster(
        frames["silver_member_memberships"],
        frames["silver_member_parties"],
        frames["silver_members"],
        period=period,
    )
    office_holder_codes = base._office_holder_codes(frames["silver_member_offices"], period=period)
    full_month_roster = roster[roster["seated_full_period"]].copy()
    pool = full_month_roster[~full_month_roster["member_code"].isin(office_holder_codes)].copy()
    pool = pool.merge(member_counts[["member_code", "question_count"]], on="member_code", how="left")
    pool["question_count"] = pool["question_count"].fillna(0).astype(int)

    bottom = pool.sort_values(["question_count", "member_name"], ascending=[True, True]).head(10).copy()
    rows = [
        {"label": row.member_name, "value": int(row.question_count), "group": row.party_key}
        for row in bottom.itertuples(index=False)
    ]
    legend_labels = {row.party_key: row.display_party_name for row in bottom.itertuples(index=False)}
    records = [
        {
            "member_code": row.member_code,
            "member_name": row.member_name,
            "party_name": row.display_party_name,
            "party_key": row.party_key,
            "question_count": int(row.question_count),
        }
        for row in bottom.itertuples(index=False)
    ]
    zero_count = int((pool["question_count"] == 0).sum())
    total_pool = int(len(pool))

    return {
        "period": period,
        "period_key": period_key,
        "period_label": period_label,
        "source_batch_id": batch.batch_id,
        "rows": rows,
        "legend_labels": legend_labels,
        "records": records,
        "office_holders_excluded_count": int(len(full_month_roster) - len(pool)),
        "eligible_pool_after_exclusion": total_pool,
        "zero_question_count": zero_count,
        "zero_question_proportion": (zero_count / total_pool) if total_pool else 0.0,
    }


def _render_text_slide_review(
    *,
    slide_id: str,
    slide_title: str,
    lines: list[str],
    footer_text: str,
    period_root: Path,
    layout: dict[str, Any],
    slide_index: int,
) -> dict[str, Any]:
    if slide_id == "fewest_askers" and _fewest_override is not None:
        rows = list(_fewest_override["rows"])
        if not rows:
            return _original_render_text_slide(
                slide_id=slide_id,
                slide_title=slide_title,
                lines=["No eligible non-office-holder TDs seated for the full month were found."],
                footer_text=footer_text,
                period_root=period_root,
                layout=layout,
                slide_index=slide_index,
            )
        return base._render_slide(
            variant_id="fewest_askers",
            slide_title="Fewest questions submitted",
            body_text=(
                f"The {len(rows)} non-office-holder TDs seated for the whole of {_fewest_override['period_label']} "
                "with the fewest recorded parliamentary questions."
            ),
            rows=rows,
            renderer_module=horizontal_bar_grouped_singleline,
            render_kwargs={
                "template": base._chart_template(
                    _fewest_override["project"],
                    value_format="integer",
                    sort="ascending",
                    max_items=10,
                ),
                "sample": {
                    "visual_id": f"{base.PROJECT_ID}-fewest_askers-{_fewest_override['period_key']}",
                    "bindings": {"label": "label", "value": "value", "group": "group"},
                    "source_note": footer_text,
                    "empty_message": "No data available",
                    "group_colors": base.PARTY_COLOR,
                    "group_legend_labels": _fewest_override["legend_labels"],
                    "group_legend_order": base.PARTY_LEGEND_ORDER,
                    "group_fallback_color": base.FALLBACK_PARTY_COLOR,
                },
                "input_metadata": {
                    "project_id": base.PROJECT_ID,
                    "source_batch_id": _fewest_override["source_batch_id"],
                    "period_start": _fewest_override["period"].start.isoformat(),
                    "period_end": _fewest_override["period"].end.isoformat(),
                    "metric_id": "fewest_askers",
                },
            },
            period_root=period_root,
            layout=json.loads(Path(str((_fewest_override["project"].get("render") or {})["outer_layout"])).read_text(encoding="utf-8")),
            slide_index=slide_index,
        )

    if slide_id != "headline":
        return _original_render_text_slide(
            slide_id=slide_id,
            slide_title=slide_title,
            lines=lines,
            footer_text=footer_text,
            period_root=period_root,
            layout=layout,
            slide_index=slide_index,
        )

    period_label = slide_title.split(":", 1)[1].strip() if ":" in slide_title else ""
    hero_value = _first_int(lines[0]) if len(lines) > 0 else ""

    written_value = ""
    oral_value = ""
    if len(lines) > 1:
        parts = [part.strip() for part in lines[1].split("·")]
        if parts:
            written_value = _first_int(parts[0])
        if len(parts) > 1:
            oral_value = _first_int(parts[1])

    askers_value = _first_int(lines[2]) if len(lines) > 2 else ""
    days_value = _first_int(lines[3]) if len(lines) > 3 else ""

    delta_text = lines[4] if len(lines) > 4 else ""
    delta_value = _first_int(delta_text)
    previous_month = ""
    previous_total = ""
    match = re.search(r"vs\s+(.+?)\s+\(([\d,]+)\s+questions\)", delta_text)
    if match:
        previous_month = match.group(1).strip()
        previous_total = match.group(2)

    custom_layout = json.loads(HEADLINE_LAYOUT.read_text(encoding="utf-8"))
    slides_dir = period_root / "slides"
    slide_path = slides_dir / f"{slide_index:02d}_{slide_id}.png"
    bindings = {
        "eyebrow": "MONTHLY QUESTIONS OVERVIEW",
        "title": "Parliamentary Questions",
        "period": period_label,
        "hero_value": hero_value,
        "hero_label": "PARLIAMENTARY QUESTIONS",
        "written_value": written_value,
        "written_label": "Written",
        "oral_value": oral_value,
        "oral_label": "Oral",
        "askers_value": askers_value,
        "askers_label": "TDs asked questions",
        "days": f"{days_value} sitting days",
        "delta": f"{delta_value} vs {previous_month}" if previous_month else delta_text,
        "comparison_label": "Change in total questions from previous month",
        "previous_total": f"Previous month total: {previous_total} questions" if previous_total else "",
        "footer_text": footer_text,
    }
    rendered = render_template(custom_layout, bindings, slide_path)
    if rendered.warnings:
        raise RuntimeError(f"Headline layout warnings for {slide_id}: {rendered.warnings}")
    return {
        "id": slide_id,
        "title": slide_title,
        "path": str(slide_path.relative_to(period_root)),
        "slide_path_abs": str(slide_path),
    }


base._render_text_slide = _render_text_slide_review


def _insert_descriptor(period_root: Path, raw: dict[str, Any]) -> None:
    slides_dir = period_root / "slides"
    existing = sorted(slides_dir.glob("*.png"))

    for path in reversed(existing):
        match = re.match(r"(\d{2})_(.+)\.png$", path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index >= 2:
            path.rename(slides_dir / f"{index + 1:02d}_{match.group(2)}.png")

    layout = json.loads(DESCRIPTOR_LAYOUT.read_text(encoding="utf-8"))
    descriptor_path = slides_dir / "02_descriptor.png"
    bindings = {
        "eyebrow": "HOW TO READ THIS POST",
        "title": "What is a parliamentary question?",
        "intro": (
            "Parliamentary questions are one of the main ways TDs ask Government ministers "
            "for information and explanations about public matters."
        ),
        "written_heading": "WRITTEN QUESTIONS",
        "written_body": (
            "Submitted in writing to a minister. The reply is provided in writing and published "
            "in the official parliamentary record."
        ),
        "oral_heading": "ORAL QUESTIONS",
        "oral_body": (
            "Selected for oral answer in the Dáil. The minister answers in the chamber and TDs "
            "may ask supplementary questions."
        ),
        "scope_heading": "WHAT “QUESTIONS” MEANS IN THIS CAROUSEL",
        "scope_body": (
            "Dáil parliamentary questions recorded by the Houses of the Oireachtas during the month."
        ),
        "footer_text": "Source: Houses of the Oireachtas: Parliamentary Questions procedure guidance",
    }
    rendered = render_template(layout, bindings, descriptor_path)
    if rendered.warnings:
        raise RuntimeError(f"Descriptor layout warnings: {rendered.warnings}")

    raw["slides"] = [str(p) for p in sorted(slides_dir.glob("*.png"))]
    raw["slide_count"] = len(raw["slides"])

    manifest_path = Path(raw.get("manifest_path") or period_root / "run_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["slide_count"] = raw["slide_count"]
        manifest["slides"] = [str(Path(p).relative_to(period_root)) for p in raw["slides"]]
        if _fewest_override is not None:
            manifest["fewest_askers"] = _fewest_override["records"]
            manifest["fewest_askers_stats"] = {
                "office_holders_excluded_count": _fewest_override["office_holders_excluded_count"],
                "eligible_pool_after_exclusion": _fewest_override["eligible_pool_after_exclusion"],
                "zero_question_count": _fewest_override["zero_question_count"],
                "zero_question_proportion": round(_fewest_override["zero_question_proportion"], 4),
                "rendered_as": "bottom_10_chart",
            }
            if isinstance(manifest.get("calculation"), dict):
                manifest["calculation"]["fewest_askers"] = (
                    "period-end roster filtered to membership_start <= period.start (seated full period) and not an "
                    "office-holder overlapping the period (silver_member_offices); bottom 10 by question count ascending"
                )
        manifest.setdefault("review_notes", []).extend([
            "Review branch inserts a plain-English parliamentary-question descriptor slide at position 2.",
            "Top-askers review slide uses smaller one-line name labels and a muted categorical palette per Warren feedback on 2026-09-28.",
            "Fewest-askers review slide always shows the bottom 10 eligible non-office-holder TDs by recorded question count, per Warren feedback on 2026-10-03.",
            "Party-per-TD review slide displays whole-number value labels at regular weight and shortens People Before Profit-Solidarity to People Before Profit, per Warren feedback on 2026-10-08.",
            "Departments slide title changed to Most Asked Departments per Warren feedback on 2026-10-08.",
        ])
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    global _fewest_override
    _fewest_override = _prepare_fewest_override(project, period_spec)
    _fewest_override["project"] = project
    raw = base.generate(project=project, period_spec=period_spec, output_root=output_root)
    _insert_descriptor(Path(raw["output_root"]), raw)
    return raw
