from __future__ import annotations

import json
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

from instagram.factory import party_monthly_profile as profile
from instagram.factory.party_asset_registry import fetch_logo, resolve_party_asset

SOURCE_MANIFEST = Path(
    "instagram/commissioning/output/party_issue_monthly_profile_v1/period=2026-07/parties/fianna-fail/manifest.json"
)
OUTPUT = Path("instagram/commissioning/output/party-logo-cover-v2-review/fianna-fail-cover.png")
LINEAGE = Path("instagram/commissioning/output/party-logo-cover-v2-review/fianna-fail-cover-lineage.json")
LOGO_SIZE = 500
LOGO_TOP = 300


def main() -> None:
    data = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    party = data["party"]
    speech_count = int(data["classified_speeches"])
    td_count = int(data["td_count"])
    period = profile.resolve_monthly_period(data["period"])

    s3 = boto3.client("s3", region_name="ca-central-1")
    asset = resolve_party_asset(party)
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    LINEAGE.write_text(
        json.dumps(
            {
                "review_type": "v2_logo_cover_representative",
                "source_metrics_manifest": str(SOURCE_MANIFEST),
                "party": party,
                "period": data["period"],
                "classified_speeches": speech_count,
                "td_count": td_count,
                "avg_speeches_per_td": avg,
                "logo_geometry": {"size": [LOGO_SIZE, LOGO_SIZE], "top": LOGO_TOP, "centered": True},
                "party_asset": asset_lineage,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
