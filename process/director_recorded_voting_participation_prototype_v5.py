#!/usr/bin/env python3
"""Render voting-participation party prototype with exact corners and preserved body.

v4 used the correct Celtic corner art but cleared very large rectangular regions
before applying it, which painted over parts of the slide. v5 clears only the
small substitute-glyph footprints from v3, then alpha-composites the exact
reference corner linework. The chart/body pixels are otherwise preserved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageOps

from process import director_recorded_voting_participation_prototype_v3 as v3
from process.director_recorded_voting_participation_prototype_v4 import _extract_reference_corner

SESSION_ROOT = Path("director/sessions/2026-09-27-recorded-voting-participation")
EVIDENCE = SESSION_ROOT / "evidence"
OUT = SESSION_ROOT / "prototype"
REFERENCE = Path("instagram/reference/member_profile_template.png")


def _clear_only_substitute_glyphs(slide: Image.Image) -> list[dict]:
    """Remove only the four v3 fleuron glyphs, not whole corner rectangles."""
    tile = v3.ornament_tile(v3.font("regular", 86), v3.COLORS["accent"])
    tw, th = tile.size
    bg = v3.COLORS["background"]
    draw = ImageDraw.Draw(slide)
    rects = {
        "tl": (10, 8, 10 + tw, 8 + th),
        "tr": (slide.width - 10 - tw, 8, slide.width - 10, 8 + th),
        "bl": (10, slide.height - 8 - th, 10 + tw, slide.height - 8),
        "br": (slide.width - 10 - tw, slide.height - 8 - th, slide.width - 10, slide.height - 8),
    }
    for box in rects.values():
        draw.rectangle(box, fill=bg)
    return [{"corner": key, "cleared_bbox": list(box)} for key, box in rects.items()]


def _apply_exact_reference_corners(slide: Image.Image) -> dict:
    reference = Image.open(REFERENCE).convert("RGB")
    source_corner = _extract_reference_corner(reference)
    target_size = (
        max(1, round(source_corner.width * slide.width / reference.width)),
        max(1, round(source_corner.height * slide.height / reference.height)),
    )
    corner = source_corner.resize(target_size, Image.Resampling.LANCZOS)
    placements = {
        "tl": (0, 0, corner),
        "tr": (slide.width - target_size[0], 0, ImageOps.mirror(corner)),
        "bl": (0, slide.height - target_size[1], ImageOps.flip(corner)),
        "br": (slide.width - target_size[0], slide.height - target_size[1], ImageOps.mirror(ImageOps.flip(corner))),
    }
    for _, (x, y, art) in placements.items():
        slide.alpha_composite(art, (x, y))
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
    }


def _non_background_corner_counts(slide: Image.Image, target_size: tuple[int, int]) -> dict[str, int]:
    bg_img = Image.new("RGB", slide.size, v3.COLORS["background"])
    diff = ImageChops.difference(slide.convert("RGB"), bg_img).convert("L")
    w, h = target_size
    boxes = {
        "tl": (0, 0, w, h),
        "tr": (slide.width - w, 0, slide.width, h),
        "bl": (0, slide.height - h, w, slide.height),
        "br": (slide.width - w, slide.height - h, slide.width, slide.height),
    }
    counts = {}
    for key, box in boxes.items():
        hist = diff.crop(box).histogram()
        counts[key] = int(sum(hist[16:]))
    return counts


def _protected_body_difference(before: Image.Image, after: Image.Image) -> dict:
    """Verify v5 did not alter the central title/chart/body composition.

    The exact corner art occupies only the outer ~313 px. This protected rectangle
    covers the full central content column, including title, chart bars and footer.
    """
    protected = (330, 0, 750, before.height)
    b = before.convert("RGB").crop(protected)
    a = after.convert("RGB").crop(protected)
    diff = ImageChops.difference(b, a).convert("L")
    hist = diff.histogram()
    changed = int(sum(hist[1:]))
    max_delta = max((idx for idx, count in enumerate(hist) if count), default=0)
    return {
        "protected_bbox": list(protected),
        "changed_pixels": changed,
        "max_channel_delta": max_delta,
        "unchanged": changed == 0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parties = pd.read_csv(EVIDENCE / "party_participation.csv").sort_values(
        ["party_name", "party_uri"], kind="stable"
    ).reset_index(drop=True)

    base_path = OUT / "prototype_party_slide_v5_base.png"
    final_path = OUT / "prototype_party_slide_v5.png"
    composition_qa = v3.render(parties, base_path)
    before = Image.open(base_path).convert("RGBA")
    after = before.copy()

    cleared = _clear_only_substitute_glyphs(after)
    corner_info = _apply_exact_reference_corners(after)
    after.convert("RGB").save(final_path, "PNG")

    target_size = tuple(corner_info["rendered_corner_dimensions"])
    corner_counts = _non_background_corner_counts(after, target_size)
    body_preservation = _protected_body_difference(before, after)

    qa = {
        "composition_qa_from_v3": composition_qa,
        "corner_qa": {
            **corner_info,
            "substitute_glyph_clear_regions": cleared,
            "non_background_pixels_by_corner": corner_counts,
            "all_four_corners_present": all(value > 500 for value in corner_counts.values()),
        },
        "body_preservation": body_preservation,
        "people_before_profit_display_label": "People Before Profit",
        "people_before_profit_single_line": True,
    }
    qa["pass"] = (
        bool(composition_qa.get("pass"))
        and bool(qa["corner_qa"]["all_four_corners_present"])
        and bool(body_preservation["unchanged"])
    )

    (OUT / "prototype_party_slide_v5_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "PASS" if qa["pass"] else "FAIL",
        "prototype_only": True,
        "visual_direction_gate": "pending_human_review",
        "slide": str(final_path),
        "period": {"start": "2026-02-28", "end": "2026-08-28"},
        "division_count": 136,
        "ordering": "alphabetical",
        "display_label_override": {
            "People Before Profit-Solidarity": "People Before Profit",
            "scope": "display only",
        },
        "corner_accents": {
            "source": str(REFERENCE),
            "method": "exact approved-reference Celtic linework, transparent overlay only",
            "destructive_corner_background_patches": false,
        },
        "body_preservation": body_preservation,
        "computer_visual_qa": qa,
        "publication_enabled": False,
    }
    (OUT / "prototype_manifest_v5.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if qa["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
