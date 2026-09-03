from __future__ import annotations

import json
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.oireachtas_production import resolve_production_key
from instagram.factory.party_asset_registry import fetch_logo, resolve_party_asset
from instagram.factory.run_party_monthly_profile import _install_member_name_aliases

PROJECT_ID = "party_issue_monthly_profile_v2"
COVER_TITLE = "Party Speech Breakdown"
LOGO_SIZE = 500
LOGO_TOP = 300
LOGO_BORDER_WIDTH = 6
LOGO_SCALE_OVERRIDES = {
    "fine-gael": 1.10,
    "independent-ireland": 1.10,
    "labour-party": 1.10,
}
SOCIAL_DEMOCRATS_KEY = "social-democrats"
MAX_NEUTRAL_SPREAD = 18
MAX_NEUTRAL_VALUE = 244
_asset_lineage: dict[str, dict] = {}
_cover_lineage: dict[str, dict] = {}


def _display_party_name(party: str) -> str:
    return "Independents" if party == "Independent" else party


def _remove_resampling_neutral_fringe(logo: Image.Image) -> tuple[Image.Image, int]:
    rgb = logo.convert("RGB")
    cleaned: list[tuple[int, int, int]] = []
    changed = 0
    for pixel in rgb.getdata():
        low = min(pixel)
        high = max(pixel)
        if high <= MAX_NEUTRAL_VALUE and (high - low) <= MAX_NEUTRAL_SPREAD:
            cleaned.append((255, 255, 255))
            if pixel != (255, 255, 255):
                changed += 1
        else:
            cleaned.append(pixel)
    output = Image.new("RGB", rgb.size, "white")
    output.putdata(cleaned)
    return output, changed


def _prepare_logo(logo: Image.Image, party_key: str) -> tuple[Image.Image, float, int]:
    scale = LOGO_SCALE_OVERRIDES.get(party_key, 1.0)
    if scale > 1.0:
        crop_size = round(logo.width / scale)
        left = (logo.width - crop_size) // 2
        top = (logo.height - crop_size) // 2
        logo = logo.crop((left, top, left + crop_size, top + crop_size))

    logo = logo.resize((LOGO_SIZE, LOGO_SIZE), Image.Resampling.LANCZOS)
    neutral_cleanup_pixels = 0
    if party_key == SOCIAL_DEMOCRATS_KEY:
        logo, neutral_cleanup_pixels = _remove_resampling_neutral_fringe(logo)
    return logo, scale, neutral_cleanup_pixels


def _install_v2_context_centering() -> None:
    original = profile._render_chart

    def wrapped(path, party, period, title_lines, supporting, rows, value_mode):
        original(path, party, period, title_lines, supporting, rows, value_mode)

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
        draw.text(
            ((profile.W - meta_w) // 2, block_top),
            meta_text,
            font=meta_font,
            fill=profile.ACCENT,
            anchor="lt",
        )
        draw.text(
            ((profile.W - support_w) // 2, block_top + meta_h + line_gap),
            supporting,
            font=support_font,
            fill=profile.TEXT,
            anchor="lt",
        )
        image.save(path)

    profile._render_chart = wrapped


def _install_logo_cover_renderer(s3) -> None:
    def render_cover(path: Path, party: str, speech_count: int, td_count: int, period) -> None:
        image = profile._base_slide()
        profile._draw_title(image, [COVER_TITLE, period.label])
        draw = ImageDraw.Draw(image)

        asset = resolve_party_asset(party)
        logo, lineage = fetch_logo(s3, asset)
        logo, logo_scale, neutral_cleanup_pixels = _prepare_logo(logo, asset.party_key)
        logo_left = (profile.W - LOGO_SIZE) // 2
        logo_right = logo_left + LOGO_SIZE - 1
        logo_bottom = LOGO_TOP + LOGO_SIZE - 1
        image.paste(logo, (logo_left, LOGO_TOP))
        draw.rectangle(
            (logo_left, LOGO_TOP, logo_right, logo_bottom),
            outline=profile.ACCENT,
            width=LOGO_BORDER_WIDTH,
        )

        _asset_lineage[party] = lineage
        _cover_lineage[party] = {
            "display_party_name": _display_party_name(party),
            "cover_title": COVER_TITLE,
            "cover_title_period": period.label,
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
                "enabled": asset.party_key == SOCIAL_DEMOCRATS_KEY,
                "neutral_pixels_replaced": neutral_cleanup_pixels,
                "purpose": "remove_neutral_gray_pixels_introduced_by_lanczos_downscaling",
            },
        }

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

    profile._render_cover = render_cover


def _install_asset_lineage() -> None:
    original_build = profile.build

    def wrapped(period_value: str, output_root: Path):
        period_root = original_build(period_value, output_root)
        manifest_path = period_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["project_id"] = PROJECT_ID
        manifest["party_asset_registry"] = "configs/reference/party_assets_v1.csv"
        manifest["party_assets"] = _asset_lineage
        manifest["cover_title"] = COVER_TITLE
        manifest["cover_logo_geometry"] = {
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
        }
        manifest["party_display_aliases"] = {"Independent": "Independents"}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        for party_manifest in manifest.get("parties", []):
            party = party_manifest.get("party")
            party_manifest_path = period_root / "parties" / party_manifest["party_key"] / "manifest.json"
            if not party_manifest_path.exists():
                continue
            data = json.loads(party_manifest_path.read_text(encoding="utf-8"))
            data["project_id"] = PROJECT_ID
            data["display_party_name"] = _display_party_name(str(party))
            data["party_asset_registry"] = "configs/reference/party_assets_v1.csv"
            data["party_asset"] = _asset_lineage.get(party)
            data.update(_cover_lineage.get(party, {}))
            party_manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return period_root

    profile.build = wrapped


def configure_v2() -> None:
    _asset_lineage.clear()
    _cover_lineage.clear()
    s3 = boto3.client("s3", region_name="ca-central-1")
    resolved_key, pointer = resolve_production_key(
        s3,
        bucket=profile.S3_BUCKET,
        production_key=profile.CLASSIFIED_KEY,
    )
    print(f"Resolved unified production batch: {pointer.get('batch_id') or pointer.get('mode')}")
    print(f"Resolved classified source: {resolved_key}")
    profile.CLASSIFIED_KEY = resolved_key
    profile.PROJECT_ID = PROJECT_ID
    _install_member_name_aliases()
    _install_v2_context_centering()
    _install_logo_cover_renderer(s3)
    _install_asset_lineage()


def main() -> None:
    configure_v2()
    profile.main()


if __name__ == "__main__":
    main()
