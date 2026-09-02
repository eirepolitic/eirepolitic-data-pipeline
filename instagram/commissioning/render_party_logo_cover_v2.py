from __future__ import annotations

import json
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.party_asset_registry import fetch_logo, resolve_party_asset

PERIOD = "2026-07"
SOURCE_ROOT = Path(
    f"instagram/commissioning/output/party_issue_monthly_profile_v1/period={PERIOD}/parties"
)
OUTPUT_ROOT = Path("instagram/commissioning/output/party-logo-cover-v2-review")
CONTACT_SHEET = OUTPUT_ROOT / "all-party-covers-contact-sheet.png"
REVIEW_MANIFEST = OUTPUT_ROOT / "review-manifest.json"
LOGO_SIZE = 500
LOGO_TOP = 300
EXPECTED_PARTY_COUNT = 11


def _display_party_name(party: str) -> str:
    return "Independents" if party == "Independent" else party


def _render_cover(data: dict, s3) -> tuple[Path, dict]:
    source_party = str(data["party"])
    party = _display_party_name(source_party)
    party_key = str(data["party_key"])
    speech_count = int(data["classified_speeches"])
    td_count = int(data["td_count"])
    period = profile.resolve_monthly_period(data["period"])

    asset = resolve_party_asset(party)
    if asset.party_key != party_key:
        raise RuntimeError(
            f"Registry party_key mismatch for {party!r}: manifest={party_key!r}, registry={asset.party_key!r}"
        )
    logo, asset_lineage = fetch_logo(s3, asset)
    logo = logo.resize((LOGO_SIZE, LOGO_SIZE), Image.Resampling.LANCZOS)

    image = profile._base_slide()
    profile._draw_title(image, [party])
    draw = ImageDraw.Draw(image)
    logo_left = (profile.W - LOGO_SIZE) // 2
    image.paste(logo, (logo_left, LOGO_TOP))

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

    output = OUTPUT_ROOT / f"{party_key}-cover.png"
    lineage_path = OUTPUT_ROOT / f"{party_key}-cover-lineage.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image.save(output)

    lineage = {
        "review_type": "v2_logo_cover_all_party_review",
        "source_metrics_manifest": str(SOURCE_ROOT / party_key / "manifest.json"),
        "source_party_name": source_party,
        "display_party_name": party,
        "party_key": party_key,
        "period": data["period"],
        "classified_speeches": speech_count,
        "td_count": td_count,
        "avg_speeches_per_td": avg,
        "output": str(output),
        "rendered_dimensions": [profile.W, profile.H],
        "logo_geometry": {"size": [LOGO_SIZE, LOGO_SIZE], "top": LOGO_TOP, "centered": True},
        "party_asset_registry": "configs/reference/party_assets_v1.csv",
        "party_asset": asset_lineage,
        "publication_enabled": False,
        "review_state": "review_requested",
    }
    lineage_path.write_text(json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8")
    return output, lineage


def _build_contact_sheet(outputs: list[tuple[Path, dict]]) -> None:
    cols = 3
    rows = 4
    thumb_w = 324
    thumb_h = 405
    gap = 18
    margin = 18
    sheet_w = margin * 2 + cols * thumb_w + (cols - 1) * gap
    sheet_h = margin * 2 + rows * thumb_h + (rows - 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    for index, (path, _) in enumerate(outputs):
        row, col = divmod(index, cols)
        image = Image.open(path).convert("RGB")
        image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + col * (thumb_w + gap)
        y = margin + row * (thumb_h + gap)
        sheet.paste(image, (x, y))

    sheet.save(CONTACT_SHEET)


def main() -> None:
    source_manifests = sorted(SOURCE_ROOT.glob("*/manifest.json"))
    if len(source_manifests) != EXPECTED_PARTY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARTY_COUNT} July party manifests under {SOURCE_ROOT}; found {len(source_manifests)}"
        )

    s3 = boto3.client("s3", region_name="ca-central-1")
    rendered: list[tuple[Path, dict]] = []
    for manifest_path in source_manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("period") != PERIOD:
            raise RuntimeError(f"Unexpected period in {manifest_path}: {data.get('period')!r}")
        rendered.append(_render_cover(data, s3))

    rendered.sort(key=lambda item: item[1]["display_party_name"].casefold())
    _build_contact_sheet(rendered)

    review_manifest = {
        "review_type": "v2_logo_cover_all_party_review",
        "project_id": "party_issue_monthly_profile_v2",
        "period": PERIOD,
        "party_count": len(rendered),
        "party_asset_registry": "configs/reference/party_assets_v1.csv",
        "logo_contract": "s3://eirepolitic-data/processed/reference/party_assets/v1/assets/{party_key}/logo.png",
        "logo_source_dimensions": [1600, 1600],
        "cover_logo_geometry": {"size": [LOGO_SIZE, LOGO_SIZE], "top": LOGO_TOP, "centered": True},
        "covers": [lineage for _, lineage in rendered],
        "contact_sheet": str(CONTACT_SHEET),
        "publication_enabled": False,
        "review_state": "review_requested",
    }
    REVIEW_MANIFEST.write_text(json.dumps(review_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Rendered {len(rendered)} party covers")
    for path, lineage in rendered:
        print(f"{lineage['display_party_name']}: {path}")
    print(CONTACT_SHEET)


if __name__ == "__main__":
    main()
