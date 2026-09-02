from __future__ import annotations

import json
from pathlib import Path

import boto3
from PIL import ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.oireachtas_production import resolve_production_key
from instagram.factory.party_asset_registry import fetch_logo, resolve_party_asset
from instagram.factory.run_party_monthly_profile import (
    _install_measured_context_centering,
    _install_member_name_aliases,
    _install_party_display_names,
)

PROJECT_ID = "party_issue_monthly_profile_v2"
LOGO_SIZE = 500
LOGO_TOP = 300
_asset_lineage: dict[str, dict] = {}


def _install_logo_cover_renderer(s3) -> None:
    def render_cover(path: Path, party: str, speech_count: int, td_count: int, period) -> None:
        image = profile._base_slide()
        profile._draw_title(image, [party])
        draw = ImageDraw.Draw(image)

        asset = resolve_party_asset(party)
        logo, lineage = fetch_logo(s3, asset)
        logo = logo.resize((LOGO_SIZE, LOGO_SIZE))
        logo_left = (profile.W - LOGO_SIZE) // 2
        image.paste(logo, (logo_left, LOGO_TOP))
        _asset_lineage[party] = lineage

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
        manifest["cover_logo_geometry"] = {"size": [LOGO_SIZE, LOGO_SIZE], "top": LOGO_TOP, "placement": "centered"}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        for party_manifest in manifest.get("parties", []):
            party = party_manifest.get("party")
            party_manifest_path = period_root / "parties" / party_manifest["party_key"] / "manifest.json"
            if party_manifest_path.exists():
                data = json.loads(party_manifest_path.read_text(encoding="utf-8"))
                data["project_id"] = PROJECT_ID
                data["party_asset"] = _asset_lineage.get(party)
                party_manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return period_root

    profile.build = wrapped


def configure_v2() -> None:
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
    _install_party_display_names()
    _install_measured_context_centering()
    _install_logo_cover_renderer(s3)
    _install_asset_lineage()


def main() -> None:
    configure_v2()
    profile.main()


if __name__ == "__main__":
    main()
