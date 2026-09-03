from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image, ImageChops

PERIOD = "2026-07"
ROOT = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v2/period={PERIOD}")
APPROVED_COVERS = Path("instagram/commissioning/output/party-logo-cover-v2-review")
PROJECT_PATH = Path("instagram/projects/party_issue_monthly_profile_v2/project.yml")
EXPECTED_VISUAL_SOURCE = "Variant 2 — Matplotlib final January commissioning, run #203"
EXPECTED_TITLES = {
    "02_most_discussed_issues": "Most Discussed Issues",
    "03_more_than_average": "Issues Discussed More Than Average",
    "04_more_per_td": "Issues Discussed More Than Average per TD",
}
EXPECTED_GLOSSARY_TERMS = [
    "Most Discussed Issues",
    "Issues Discussed More Than Average",
    "Issues Discussed More Than Average per TD",
    "Classified Speeches",
]
EXPECTED_KEYS = {
    "100-rdr", "aontu", "fianna-fail", "fine-gael", "green-party",
    "independent-ireland", "independent", "labour-party",
    "people-before-profit-solidarity", "sinn-fein", "social-democrats",
}
EXPECTED_SCALES = {
    "fine-gael": 1.10,
    "independent-ireland": 1.10,
    "labour-party": 1.10,
}
CANONICAL_SOCIAL = "s3://eirepolitic-data/processed/reference/party_assets/v1/assets/social-democrats/logo.png"


def check(name: str, condition: bool, detail: object = "") -> None:
    if not condition:
        message = f"{name}: {detail}"
        print(f"::error title=Party monthly profile v2 QA::{message}")
        raise AssertionError(message)
    print(f"PASS {name}")


def _images_identical(left: Path, right: Path) -> bool:
    with Image.open(left) as a, Image.open(right) as b:
        a_rgb = a.convert("RGB")
        b_rgb = b.convert("RGB")
        return a_rgb.size == b_rgb.size and ImageChops.difference(a_rgb, b_rgb).getbbox() is None


def _neutral_fringe_pixels(path: Path) -> int:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        logo_left = (1080 - 500) // 2
        interior = rgb.crop((logo_left + 6, 306, logo_left + 494, 794))
        count = 0
        for pixel in interior.getdata():
            low = min(pixel)
            high = max(pixel)
            if high <= 244 and (high - low) <= 18:
                count += 1
        return count


def main() -> None:
    project = yaml.safe_load(PROJECT_PATH.read_text(encoding="utf-8"))
    run_manifest = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf-8"))

    check("project id", run_manifest.get("project_id") == "party_issue_monthly_profile_v2", run_manifest.get("project_id"))
    check("period", run_manifest.get("period", {}).get("key") == PERIOD, run_manifest.get("period"))
    check("matched rows", run_manifest.get("readiness", {}).get("matched_classified_rows") == 2009, run_manifest.get("readiness"))
    check("zero unmatched", run_manifest.get("readiness", {}).get("unmatched_classified_rows") == 0, run_manifest.get("readiness"))
    check("55 slide QA", run_manifest.get("qa") == {"slide_count": 55, "passed": 55, "failed": 0}, run_manifest.get("qa"))
    check("publication disabled", run_manifest.get("publication_enabled") is False, run_manifest.get("publication_enabled"))
    check("review state", run_manifest.get("review_state") == "pending_human_review", run_manifest.get("review_state"))
    check("variant 2 selected", run_manifest.get("analytical_visual_variant") == 2, run_manifest.get("analytical_visual_variant"))
    check("variant 2 source", run_manifest.get("analytical_slide_visual_source") == EXPECTED_VISUAL_SOURCE, run_manifest.get("analytical_slide_visual_source"))
    check("analytical titles", run_manifest.get("analytical_titles") == EXPECTED_TITLES, run_manifest.get("analytical_titles"))
    check("glossary terms", run_manifest.get("glossary_terms") == EXPECTED_GLOSSARY_TERMS, run_manifest.get("glossary_terms"))

    chart_geometry = run_manifest.get("chart_geometry") or {}
    check("variant source run", chart_geometry.get("source_run_number") == 203, chart_geometry)
    check("variant source run id", chart_geometry.get("source_run_id") == 33448590338, chart_geometry)
    check("Matplotlib media size", chart_geometry.get("visual_media_dimensions") == [1032, 1210], chart_geometry)
    check("outer slide size", chart_geometry.get("outer_slide_dimensions") == [1080, 1350], chart_geometry)
    check("wider bar ratio", abs(float(chart_geometry.get("bar_height_ratio_for_7_rows")) - 0.62) < 1e-9, chart_geometry)
    check("short chart visual rows", chart_geometry.get("min_visual_rows") == 4, chart_geometry)

    geometry = run_manifest.get("cover_logo_geometry") or {}
    check("logo square", geometry.get("square_size") == [500, 500], geometry)
    check("logo top", geometry.get("top") == 300, geometry)
    check("gold border", geometry.get("border", {}).get("color") == "#d8b45f", geometry)
    check("border width", geometry.get("border", {}).get("width_px") == 6, geometry)
    check("scale overrides", geometry.get("artwork_scale_overrides") == EXPECTED_SCALES, geometry)

    check("project publication disabled", project["publication"]["enabled"] is False, project["publication"])
    check("project review state", project["publication"]["review_state"] == "pending_human_review", project["publication"])

    party_manifests = sorted((ROOT / "parties").glob("*/manifest.json"))
    check("11 party manifests", len(party_manifests) == 11, len(party_manifests))
    check("party keys", {p.parent.name for p in party_manifests} == EXPECTED_KEYS)

    slide_paths = sorted((ROOT / "parties").glob("*/slides/*.png"))
    check("55 slide PNGs", len(slide_paths) == 55, len(slide_paths))
    for path in slide_paths:
        with Image.open(path) as image:
            image.load()
            check(f"{path.name} format", image.format == "PNG", image.format)
            check(f"{path.name} dimensions", image.size == (1080, 1350), image.size)

    for manifest_path in party_manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = data["party_key"]
        check(f"{key} five slides", len(data.get("slides") or []) == 5, data.get("slides"))
        check(f"{key} variant 2", data.get("analytical_visual_variant") == 2, data.get("analytical_visual_variant"))
        check(f"{key} variant source", data.get("analytical_slide_visual_source") == EXPECTED_VISUAL_SOURCE, data.get("analytical_slide_visual_source"))
        check(f"{key} analytical titles", data.get("analytical_titles") == EXPECTED_TITLES, data.get("analytical_titles"))
        check(f"{key} glossary terms", data.get("glossary_terms") == EXPECTED_GLOSSARY_TERMS, data.get("glossary_terms"))
        check(f"{key} publication disabled", data.get("publication_enabled") is False, data.get("publication_enabled"))
        check(f"{key} review state", data.get("review_state") == "pending_human_review", data.get("review_state"))

        asset = data.get("party_asset") or {}
        check(f"{key} asset dimensions", asset.get("dimensions") == [1600, 1600], asset)
        check(f"{key} canonical asset", asset.get("logo_s3_uri", "").endswith(f"/{key}/logo.png"), asset)
        if key == "social-democrats":
            check("Social Democrats canonical URI", asset.get("logo_s3_uri") == CANONICAL_SOCIAL, asset)

        expected_scale = EXPECTED_SCALES.get(key, 1.0)
        check(f"{key} artwork scale", data.get("logo_geometry", {}).get("artwork_scale") == expected_scale, data.get("logo_geometry"))

        generated_cover = manifest_path.parent / "slides" / "01_cover.png"
        approved_cover = APPROVED_COVERS / f"{key}-cover.png"
        check(f"{key} approved cover match", approved_cover.exists() and _images_identical(generated_cover, approved_cover), f"generated={generated_cover}, approved={approved_cover}")

        analytical = data.get("analytical_visuals") or {}
        check(f"{key} three analytical visuals", set(analytical) == set(EXPECTED_TITLES), analytical.keys())
        for slide_key, expected_title in EXPECTED_TITLES.items():
            meta = analytical[slide_key]
            check(f"{key} {slide_key} title", meta.get("slide_title") == expected_title, meta.get("slide_title"))
            check(f"{key} {slide_key} source run", meta.get("source_run_number") == 203, meta)
            check(f"{key} {slide_key} no warnings", meta.get("warnings") == [], meta.get("warnings"))
            check(f"{key} {slide_key} visual asset exists", (ROOT / meta["visual_asset"]).exists(), meta.get("visual_asset"))
            check(f"{key} {slide_key} visual metadata exists", (ROOT / meta["visual_metadata"]).exists(), meta.get("visual_metadata"))
            check(f"{key} {slide_key} visual manifest exists", (ROOT / meta["visual_manifest"]).exists(), meta.get("visual_manifest"))

    social_cover = ROOT / "parties" / "social-democrats" / "slides" / "01_cover.png"
    check("Social Democrats rendered halo", _neutral_fringe_pixels(social_cover) == 0, _neutral_fringe_pixels(social_cover))

    with (ROOT / "qa_summary.csv").open(encoding="utf-8", newline="") as fh:
        qa_rows = list(csv.DictReader(fh))
    check("qa summary rows", len(qa_rows) == 55, len(qa_rows))
    check("qa summary all pass", all(row.get("status") == "PASS" for row in qa_rows))

    contacts = ROOT / "contact_sheets"
    for name in ("covers.jpg", "most_discussed_issues.jpg", "more_than_average.jpg", "more_per_td.jpg", "five_slide_overview.jpg"):
        path = contacts / name
        check(f"{name} exists", path.exists(), path)
        with Image.open(path) as image:
            image.load()
            check(f"{name} valid", image.size[0] > 0 and image.size[1] > 0, image.size)

    print("PASS: July 2026 v2 batch — approved covers + Variant 2 charts + revised titles/glossary")


if __name__ == "__main__":
    main()
