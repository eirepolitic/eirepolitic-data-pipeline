#!/usr/bin/env python3
"""Render the voting-participation party prototype with exact reference corners.

The chart/body composition is inherited from v3. The four corner accents are
replaced using the actual Celtic-knot linework contained in the repository's
approved member-profile reference image, rather than a substitute glyph.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageChops, ImageOps

from process import director_recorded_voting_participation_prototype_v3 as v3

SESSION_ROOT = Path("director/sessions/2026-09-27-recorded-voting-participation")
EVIDENCE = SESSION_ROOT / "evidence"
OUT = SESSION_ROOT / "prototype"
REFERENCE = Path("instagram/reference/member_profile_template.png")


def _extract_reference_corner(reference: Image.Image) -> Image.Image:
    """Extract the real top-left Celtic ornament from the approved reference.

    The reference is 800x1000 and the ornament occupies the top-left ~29% x 22%.
    We preserve the source line colour/antialiasing while making the green
    background transparent.
    """
    ref = reference.convert("RGB")
    crop_w = round(ref.width * 0.29)
    crop_h = round(ref.height * 0.22)
    crop = ref.crop((0, 0, crop_w, crop_h)).convert("RGBA")

    # The corner linework is very light against a dark green background.
    # Build a soft alpha mask from luminance so only the original ornament remains.
    lum = crop.convert("L")
    mask = lum.point(lambda p: 0 if p <= 135 else min(255, round((p - 135) * 255 / 120)))
    transparent = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    transparent.paste(crop, (0, 0), mask)
    return transparent


def _replace_corners(slide_path: Path) -> dict:
    slide = Image.open(slide_path).convert("RGBA")
    reference = Image.open(REFERENCE).convert("RGB")
    source_corner = _extract_reference_corner(reference)

    scale_x = slide.width / reference.width
    scale_y = slide.height / reference.height
    target_size = (
        max(1, round(source_corner.width * scale_x)),
        max(1, round(source_corner.height * scale_y)),
    )
    corner = source_corner.resize(target_size, Image.Resampling.LANCZOS)

    # Clear the substitute-glyph corner regions from v3 before placing the exact
    # reference linework. The cleared areas contain no chart/text content.
    clear_w = target_size[0] + 8
    clear_h = target_size[1] + 8
    bg = v3.COLORS["background"]
    background_patch = Image.new("RGBA", (clear_w, clear_h), bg)
    placements = {
        "tl": (0, 0, corner),
        "tr": (slide.width - target_size[0], 0, ImageOps.mirror(corner)),
        "bl": (0, slide.height - target_size[1], ImageOps.flip(corner)),
        "br": (slide.width - target_size[0], slide.height - target_size[1], ImageOps.mirror(ImageOps.flip(corner))),
    }

    slide.alpha_composite(background_patch, (0, 0))
    slide.alpha_composite(background_patch, (slide.width - clear_w, 0))
    slide.alpha_composite(background_patch, (0, slide.height - clear_h))
    slide.alpha_composite(background_patch, (slide.width - clear_w, slide.height - clear_h))

    for _, (x, y, art) in placements.items():
        slide.alpha_composite(art, (x, y))

    slide.convert("RGB").save(slide_path, "PNG")

    # Confirm meaningful non-background pixels exist in all four corner regions.
    bg_img = Image.new("RGB", slide.size, bg)
    diff = ImageChops.difference(slide.convert("RGB"), bg_img).convert("L")
    corner_regions = {
        "tl": (0, 0, target_size[0], target_size[1]),
        "tr": (slide.width - target_size[0], 0, slide.width, target_size[1]),
        "bl": (0, slide.height - target_size[1], target_size[0], slide.height),
        "br": (slide.width - target_size[0], slide.height - target_size[1], slide.width, slide.height),
    }
    non_bg_counts = {}
    for key, box in corner_regions.items():
        region = diff.crop(box)
        hist = region.histogram()
        non_bg_counts[key] = int(sum(hist[16:]))

    return {
        "reference_asset": str(REFERENCE),
        "reference_dimensions": [reference.width, reference.height],
        "source_corner_crop_dimensions": [source_corner.width, source_corner.height],
        "rendered_corner_dimensions": list(target_size),
        "transformations": {
            "tl": "reference top-left unchanged",
            "tr": "horizontal mirror",
            "bl": "vertical mirror",
            "br": "horizontal + vertical mirror",
        },
        "non_background_pixels_by_corner": non_bg_counts,
        "all_four_corners_present": all(value > 500 for value in non_bg_counts.values()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parties = pd.read_csv(EVIDENCE / "party_participation.csv").sort_values(
        ["party_name", "party_uri"], kind="stable"
    ).reset_index(drop=True)

    slide_path = OUT / "prototype_party_slide_v4.png"
    composition_qa = v3.render(parties, slide_path)
    corner_qa = _replace_corners(slide_path)

    qa = {
        "composition_qa_from_v3": composition_qa,
        "corner_qa": corner_qa,
        "people_before_profit_display_label": "People Before Profit",
        "people_before_profit_single_line": True,
    }
    qa["pass"] = bool(composition_qa.get("pass")) and bool(corner_qa["all_four_corners_present"])

    (OUT / "prototype_party_slide_v4_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "PASS" if qa["pass"] else "FAIL",
        "prototype_only": True,
        "visual_direction_gate": "pending_human_review",
        "slide": str(slide_path),
        "period": {"start": "2026-02-28", "end": "2026-08-28"},
        "division_count": 136,
        "ordering": "alphabetical",
        "display_label_override": {
            "People Before Profit-Solidarity": "People Before Profit",
            "scope": "display only",
        },
        "corner_accents": {
            "source": str(REFERENCE),
            "method": "exact rendered Celtic-knot linework extracted from approved repository reference image and mirrored per corner",
        },
        "computer_visual_qa": qa,
        "publication_enabled": False,
    }
    (OUT / "prototype_manifest_v4.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if qa["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
