from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.party_asset_registry import fetch_logo, resolve_party_asset
from instagram.factory.run_party_monthly_profile_v2 import (
    COVER_TITLE,
    LOGO_BORDER_WIDTH,
    LOGO_SCALE_OVERRIDES,
    LOGO_SIZE,
    LOGO_TOP,
    _display_party_name,
    _prepare_logo,
)
from instagram.factory.periods import resolve_monthly_period

PERIOD = "2026-07"
SOURCE_ROOT = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v1/period={PERIOD}")
OUTPUT_ROOT = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v2/period={PERIOD}")
EXPECTED_PARTY_COUNT = 11
EXPECTED_SLIDE_COUNT = 55


def _render_cover(path: Path, data: dict, s3, period) -> dict:
    party = str(data["party"])
    party_key = str(data["party_key"])
    speech_count = int(data["classified_speeches"])
    td_count = int(data["td_count"])

    image = profile._base_slide()
    profile._draw_title(image, [COVER_TITLE, period.label])
    draw = ImageDraw.Draw(image)

    asset = resolve_party_asset(party)
    if asset.party_key != party_key:
        raise RuntimeError(
            f"Registry party_key mismatch for {party!r}: source={party_key!r}, registry={asset.party_key!r}"
        )
    logo, asset_lineage = fetch_logo(s3, asset)
    logo, logo_scale, neutral_cleanup_pixels = _prepare_logo(logo, party_key)

    logo_left = (profile.W - LOGO_SIZE) // 2
    logo_right = logo_left + LOGO_SIZE - 1
    logo_bottom = LOGO_TOP + LOGO_SIZE - 1
    image.paste(logo, (logo_left, LOGO_TOP))
    draw.rectangle(
        (logo_left, LOGO_TOP, logo_right, logo_bottom),
        outline=profile.ACCENT,
        width=LOGO_BORDER_WIDTH,
    )

    number_font = profile._font(72, True)
    label_font = profile._font(25, True)
    small_font = profile._font(24)
    avg = speech_count / td_count if td_count else 0.0
    draw.text((294, 955), f"{speech_count:,}", font=number_font, fill=profile.TEXT, anchor="mm")
    draw.text((294, 1022), "CLASSIFIED SPEECHES", font=label_font, fill=profile.ACCENT, anchor="mm")
    draw.text((786, 955), f"{avg:.1f}", font=number_font, fill=profile.TEXT, anchor="mm")
    draw.text((786, 1022), "AVG SPEECHES PER TD", font=label_font, fill=profile.ACCENT, anchor="mm")
    draw.line((239, 1115, 841, 1115), fill=profile.ACCENT, width=3)
    draw.text((540, 1170), period.label.upper(), font=label_font, fill=profile.ACCENT, anchor="mm")
    draw.text((540, 1218), profile._period_dates(period), font=small_font, fill=profile.TEXT, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

    return {
        "party_asset": asset_lineage,
        "cover_title": COVER_TITLE,
        "cover_title_period": period.label,
        "display_party_name": _display_party_name(party),
        "logo_geometry": {
            "square_size": [LOGO_SIZE, LOGO_SIZE],
            "top": LOGO_TOP,
            "centered": True,
            "artwork_scale": logo_scale,
            "border": {
                "enabled": True,
                "color": profile.ACCENT,
                "width_px": LOGO_BORDER_WIDTH,
                "position": "inside_square",
            },
        },
        "logo_resampling_cleanup": {
            "enabled": party_key == "social-democrats",
            "neutral_pixels_replaced": neutral_cleanup_pixels,
            "purpose": "remove_neutral_gray_pixels_introduced_by_lanczos_downscaling",
        },
    }


def _render_chart(path: Path, party: str, period, title_lines: list[str], supporting: str, rows: list[dict], value_mode: str) -> None:
    profile._render_chart(path, party, period, title_lines, supporting, rows, value_mode)

    display_party = _display_party_name(party)
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    region_top = profile.TITLE_RULE_Y + 5
    chart_top = profile.CHART_MEDIA_Y + 105
    draw.rectangle((0, region_top, profile.W, chart_top - 1), fill=profile.BG)

    meta_text = f"{display_party.upper()} · {period.label.upper()}"
    meta_font = profile._font(22, True)
    support_font = profile._font(24)
    meta_box = draw.textbbox((0, 0), meta_text, font=meta_font, anchor="lt")
    support_box = draw.textbbox((0, 0), supporting, font=support_font, anchor="lt")
    meta_w = meta_box[2] - meta_box[0]
    meta_h = meta_box[3] - meta_box[1]
    support_w = support_box[2] - support_box[0]
    support_h = support_box[3] - support_box[1]
    line_gap = 10
    block_h = meta_h + line_gap + support_h
    block_top = region_top + ((chart_top - region_top - block_h) // 2)
    draw.text(((profile.W - meta_w) // 2, block_top), meta_text, font=meta_font, fill=profile.ACCENT, anchor="lt")
    draw.text(
        ((profile.W - support_w) // 2, block_top + meta_h + line_gap),
        supporting,
        font=support_font,
        fill=profile.TEXT,
        anchor="lt",
    )
    image.save(path)


def main() -> None:
    source_run = json.loads((SOURCE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    source_party_manifests = sorted((SOURCE_ROOT / "parties").glob("*/manifest.json"))
    if len(source_party_manifests) != EXPECTED_PARTY_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PARTY_COUNT} source party manifests; found {len(source_party_manifests)}")
    if source_run.get("qa") != {"slide_count": 55, "passed": 55, "failed": 0}:
        raise RuntimeError(f"Source July v1 batch is not 55/55 QA-passed: {source_run.get('qa')}")
    readiness = source_run.get("readiness") or {}
    if readiness.get("matched_classified_rows") != 2009 or readiness.get("unmatched_classified_rows") != 0:
        raise RuntimeError(f"Unexpected source July v1 readiness: {readiness}")

    period = resolve_monthly_period(PERIOD)
    s3 = boto3.client("s3", region_name="ca-central-1")
    party_manifests: list[dict] = []
    qa_rows: list[dict] = []
    cover_paths: list[tuple[str, Path]] = []
    raw_paths: list[tuple[str, Path]] = []
    share_paths: list[tuple[str, Path]] = []
    per_td_paths: list[tuple[str, Path]] = []
    carousel_items: list[tuple[str, list[Path]]] = []

    for source_manifest_path in source_party_manifests:
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        party = str(source["party"])
        key = str(source["party_key"])
        slides_dir = OUTPUT_ROOT / "parties" / key / "slides"
        paths = [
            slides_dir / "01_cover.png",
            slides_dir / "02_most_discussed_issues.png",
            slides_dir / "03_more_than_average.png",
            slides_dir / "04_more_per_td.png",
            slides_dir / "05_glossary.png",
        ]

        cover_lineage = _render_cover(paths[0], source, s3, period)
        _render_chart(paths[1], party, period, ["Most Discussed Issues"], "Total classified speeches", source["raw_counts"], "count")
        _render_chart(paths[2], party, period, ["Issues Discussed", "More Than Average"], "Compared with the average party", source["share_vs_average"], "share_pp")
        _render_chart(paths[3], party, period, ["Issues Discussed", "More Per TD"], "Adjusted for party size", source["per_td_vs_average"], "per_td")
        profile._render_glossary(paths[4])

        manifest = {
            **source,
            "project_id": "party_issue_monthly_profile_v2",
            "source_metrics_manifest": str(source_manifest_path),
            "source_metrics_project_id": source_run.get("project_id"),
            "party_asset_registry": "configs/reference/party_assets_v1.csv",
            **cover_lineage,
            "slides": [str(path.relative_to(OUTPUT_ROOT)) for path in paths],
            "review_state": "pending_human_review",
            "publication_enabled": False,
        }
        manifest_path = OUTPUT_ROOT / "parties" / key / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        party_manifests.append(manifest)

        for slide_no, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                ok = image.size == (profile.W, profile.H)
            qa_rows.append({
                "party": party,
                "party_key": key,
                "slide": slide_no,
                "path": str(path.relative_to(OUTPUT_ROOT)),
                "dimensions_ok": ok,
                "status": "PASS" if ok else "FAIL",
            })
        cover_paths.append((_display_party_name(party), paths[0]))
        raw_paths.append((_display_party_name(party), paths[1]))
        share_paths.append((_display_party_name(party), paths[2]))
        per_td_paths.append((_display_party_name(party), paths[3]))
        carousel_items.append((_display_party_name(party), paths))

    if len(qa_rows) != EXPECTED_SLIDE_COUNT or any(row["status"] != "PASS" for row in qa_rows):
        raise RuntimeError("Rendered-slide QA failed")

    contacts = OUTPUT_ROOT / "contact_sheets"
    profile._contact_sheet(cover_paths, contacts / "covers.jpg")
    profile._contact_sheet(raw_paths, contacts / "most_discussed_issues.jpg")
    profile._contact_sheet(share_paths, contacts / "more_than_average.jpg")
    profile._contact_sheet(per_td_paths, contacts / "more_per_td.jpg")
    profile._carousel_sheet(carousel_items, contacts / "five_slide_overview.jpg")

    import csv

    qa_path = OUTPUT_ROOT / "qa_summary.csv"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(qa_rows[0]))
        writer.writeheader()
        writer.writerows(qa_rows)

    run_manifest = {
        "project_id": "party_issue_monthly_profile_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": source_run["period"],
        "readiness": source_run["readiness"],
        "sources": source_run["sources"],
        "source_metrics_run_manifest": str(SOURCE_ROOT / "run_manifest.json"),
        "source_metrics_project_id": source_run.get("project_id"),
        "calculation": source_run["calculation"],
        "presentation_labels": source_run["presentation_labels"],
        "chart_geometry": source_run["chart_geometry"],
        "party_asset_registry": "configs/reference/party_assets_v1.csv",
        "cover_title": COVER_TITLE,
        "cover_logo_geometry": {
            "square_size": [LOGO_SIZE, LOGO_SIZE],
            "top": LOGO_TOP,
            "centered": True,
            "border": {
                "enabled": True,
                "color": profile.ACCENT,
                "width_px": LOGO_BORDER_WIDTH,
                "position": "inside_square",
            },
            "artwork_scale_overrides": LOGO_SCALE_OVERRIDES,
        },
        "party_display_aliases": {"Independent": "Independents"},
        "parties": party_manifests,
        "qa": {"slide_count": len(qa_rows), "passed": len(qa_rows), "failed": 0},
        "review_state": "pending_human_review",
        "publication_enabled": False,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "period": PERIOD, "parties": len(party_manifests), "slides": len(qa_rows), "output": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
