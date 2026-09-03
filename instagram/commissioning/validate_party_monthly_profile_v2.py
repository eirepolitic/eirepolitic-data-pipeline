from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image, ImageChops

PERIOD = "2026-07"
ROOT = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v2/period={PERIOD}")
SOURCE_V1 = Path(f"instagram/commissioning/output/party_issue_monthly_profile_v1/period={PERIOD}")
APPROVED_COVERS = Path("instagram/commissioning/output/party-logo-cover-v2-review")
PROJECT_PATH = Path("instagram/projects/party_issue_monthly_profile_v2/project.yml")
EXPECTED_VISUAL_SOURCE = "exact PNGs from party_issue_monthly_profile_v1 July 2026 successful commissioning batch"
EXPECTED_KEYS = {
    "100-rdr",
    "aontu",
    "fianna-fail",
    "fine-gael",
    "green-party",
    "independent-ireland",
    "independent",
    "labour-party",
    "people-before-profit-solidarity",
    "sinn-fein",
    "social-democrats",
}
EXPECTED_SCALES = {
    "fine-gael": 1.10,
    "independent-ireland": 1.10,
    "labour-party": 1.10,
}
BAR_CHART_SLIDES = (
    "02_most_discussed_issues.png",
    "03_more_than_average.png",
    "04_more_per_td.png",
)
CANONICAL_SOCIAL = (
    "s3://eirepolitic-data/processed/reference/party_assets/v1/assets/social-democrats/logo.png"
)


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
        interior = rgb.crop((logo_left + 6, 300 + 6, logo_left + 494, 300 + 494))
        unwanted = 0
        for pixel in interior.getdata():
            low = min(pixel)
            high = max(pixel)
            if high <= 244 and (high - low) <= 18:
                unwanted += 1
        return unwanted


def main() -> None:
    project = yaml.safe_load(PROJECT_PATH.read_text(encoding="utf-8"))
    run_manifest = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf-8"))

    check("project id", run_manifest.get("project_id") == "party_issue_monthly_profile_v2", run_manifest.get("project_id"))
    check("period", run_manifest.get("period", {}).get("key") == PERIOD, run_manifest.get("period"))
    check("period start", run_manifest.get("period", {}).get("start") == "2026-07-01", run_manifest.get("period"))
    check("period end", run_manifest.get("period", {}).get("end") == "2026-07-31", run_manifest.get("period"))
    check("party count", run_manifest.get("readiness", {}).get("party_count") == 11, run_manifest.get("readiness"))
    check("matched rows", run_manifest.get("readiness", {}).get("matched_classified_rows") == 2009, run_manifest.get("readiness"))
    check("zero unmatched", run_manifest.get("readiness", {}).get("unmatched_classified_rows") == 0, run_manifest.get("readiness"))
    check("classifier covers period", run_manifest.get("readiness", {}).get("classifier_covers_period_end") is True, run_manifest.get("readiness"))
    check("55 slide QA", run_manifest.get("qa") == {"slide_count": 55, "passed": 55, "failed": 0}, run_manifest.get("qa"))
    check("run publication disabled", run_manifest.get("publication_enabled") is False, run_manifest.get("publication_enabled"))
    check("run review state", run_manifest.get("review_state") == "pending_human_review", run_manifest.get("review_state"))
    check("registry path", run_manifest.get("party_asset_registry") == "configs/reference/party_assets_v1.csv", run_manifest.get("party_asset_registry"))
    check("cover title", run_manifest.get("cover_title") == "Party Speech Breakdown", run_manifest.get("cover_title"))
    check("analytical visual source", run_manifest.get("analytical_slide_visual_source") == EXPECTED_VISUAL_SOURCE, run_manifest.get("analytical_slide_visual_source"))

    geometry = run_manifest.get("cover_logo_geometry") or {}
    check("logo square", geometry.get("square_size") == [500, 500], geometry)
    check("logo top", geometry.get("top") == 300, geometry)
    check("gold border", geometry.get("border", {}).get("color") == "#d8b45f", geometry)
    check("border width", geometry.get("border", {}).get("width_px") == 6, geometry)
    check("border inside square", geometry.get("border", {}).get("position") == "inside_square", geometry)
    check("scale overrides", geometry.get("artwork_scale_overrides") == EXPECTED_SCALES, geometry)
    check("Independent alias", run_manifest.get("party_display_aliases") == {"Independent": "Independents"}, run_manifest.get("party_display_aliases"))

    check("project publication disabled", project["publication"]["enabled"] is False, project["publication"])
    check("project review state", project["publication"]["review_state"] == "pending_human_review", project["publication"])

    party_manifests = sorted((ROOT / "parties").glob("*/manifest.json"))
    check("11 party manifests", len(party_manifests) == 11, len(party_manifests))
    keys = {path.parent.name for path in party_manifests}
    check("party keys", keys == EXPECTED_KEYS, keys)

    slide_paths = sorted((ROOT / "parties").glob("*/slides/*.png"))
    check("55 slide PNGs", len(slide_paths) == 55, len(slide_paths))
    for path in slide_paths:
        with Image.open(path) as image:
            image.load()
            check(f"{path.relative_to(ROOT)} format", image.format == "PNG", image.format)
            check(f"{path.relative_to(ROOT)} dimensions", image.size == (1080, 1350), image.size)

    for manifest_path in party_manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = data["party_key"]
        check(f"{key} period", data.get("period") == PERIOD, data.get("period"))
        check(f"{key} slide count", len(data.get("slides") or []) == 5, data.get("slides"))
        check(f"{key} publication disabled", data.get("publication_enabled") is False, data.get("publication_enabled"))
        check(f"{key} review state", data.get("review_state") == "pending_human_review", data.get("review_state"))
        check(f"{key} registry", data.get("party_asset_registry") == "configs/reference/party_assets_v1.csv", data.get("party_asset_registry"))
        check(f"{key} cover title", data.get("cover_title") == "Party Speech Breakdown", data.get("cover_title"))
        check(f"{key} cover period", data.get("cover_title_period") == "July 2026", data.get("cover_title_period"))
        check(f"{key} analytical visual source", data.get("analytical_slide_visual_source") == EXPECTED_VISUAL_SOURCE, data.get("analytical_slide_visual_source"))
        expected_display = "Independents" if key == "independent" else data.get("party")
        check(f"{key} display name", data.get("display_party_name") == expected_display, data.get("display_party_name"))

        asset = data.get("party_asset") or {}
        check(f"{key} asset dimensions", asset.get("dimensions") == [1600, 1600], asset)
        check(f"{key} canonical asset", asset.get("logo_s3_uri", "").endswith(f"/{key}/logo.png"), asset)
        if key == "social-democrats":
            check("Social Democrats canonical URI", asset.get("logo_s3_uri") == CANONICAL_SOCIAL, asset)

        expected_scale = EXPECTED_SCALES.get(key, 1.0)
        check(f"{key} artwork scale", data.get("logo_geometry", {}).get("artwork_scale") == expected_scale, data.get("logo_geometry"))
        check(f"{key} border color", data.get("logo_geometry", {}).get("border", {}).get("color") == "#d8b45f", data.get("logo_geometry"))
        check(f"{key} border width", data.get("logo_geometry", {}).get("border", {}).get("width_px") == 6, data.get("logo_geometry"))

        generated_cover = manifest_path.parent / "slides" / "01_cover.png"
        approved_cover = APPROVED_COVERS / f"{key}-cover.png"
        check(f"{key} approved cover match", approved_cover.exists() and _images_identical(generated_cover, approved_cover), f"generated={generated_cover}, approved={approved_cover}")

        for slide_name in BAR_CHART_SLIDES:
            generated_chart = manifest_path.parent / "slides" / slide_name
            source_chart = SOURCE_V1 / "parties" / key / "slides" / slide_name
            check(f"{key} {slide_name} matches successful pre-logo visual", source_chart.exists() and _images_identical(generated_chart, source_chart), f"generated={generated_chart}, source={source_chart}")

    social_cover = ROOT / "parties" / "social-democrats" / "slides" / "01_cover.png"
    check("Social Democrats rendered halo", _neutral_fringe_pixels(social_cover) == 0, _neutral_fringe_pixels(social_cover))

    qa_path = ROOT / "qa_summary.csv"
    with qa_path.open(encoding="utf-8", newline="") as fh:
        qa_rows = list(csv.DictReader(fh))
    check("qa_summary rows", len(qa_rows) == 55, len(qa_rows))
    check("qa_summary all pass", all(row.get("status") == "PASS" for row in qa_rows), [row for row in qa_rows if row.get("status") != "PASS"])

    contacts = ROOT / "contact_sheets"
    expected_contacts = {"covers.jpg", "most_discussed_issues.jpg", "more_than_average.jpg", "more_per_td.jpg", "five_slide_overview.jpg"}
    actual_contacts = {path.name for path in contacts.glob("*.jpg")}
    check("contact sheets", expected_contacts.issubset(actual_contacts), actual_contacts)
    for name in expected_contacts:
        with Image.open(contacts / name) as image:
            image.load()
            check(f"{name} valid", image.size[0] > 0 and image.size[1] > 0, image.size)

    print("PASS: July 2026 party monthly profile v2 full batch QA — approved covers + exact pre-logo bar charts")


if __name__ == "__main__":
    main()
