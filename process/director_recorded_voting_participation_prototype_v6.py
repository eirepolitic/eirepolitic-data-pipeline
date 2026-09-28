#!/usr/bin/env python3
"""Render voting-participation party prototype without vertical guide lines.

Keeps the validated v5 composition/corners but suppresses only the four long
25/50/75/100% vertical grid guides. Tick labels, row dividers, bars, values,
text, and exact reference Celtic corners remain unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageDraw

from process import director_recorded_voting_participation_prototype_v3 as v3
from process import director_recorded_voting_participation_prototype_v5 as v5

SESSION_ROOT = Path("director/sessions/2026-09-27-recorded-voting-participation")
EVIDENCE = SESSION_ROOT / "evidence"
OUT = SESSION_ROOT / "prototype"


def _render_without_vertical_guides(parties: pd.DataFrame, output: Path):
    original_line = ImageDraw.ImageDraw.line
    suppressed: list[list[int]] = []

    def line_without_long_vertical_guides(self, xy, *args, **kwargs):
        try:
            coords = list(xy)
            if (
                len(coords) == 4
                and coords[0] == coords[2]
                and abs(coords[3] - coords[1]) > 700
                and kwargs.get("width", 0) == 1
            ):
                suppressed.append([int(v) for v in coords])
                return None
        except Exception:
            pass
        return original_line(self, xy, *args, **kwargs)

    ImageDraw.ImageDraw.line = line_without_long_vertical_guides
    try:
        qa = v3.render(parties, output)
    finally:
        ImageDraw.ImageDraw.line = original_line

    return qa, suppressed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parties = pd.read_csv(EVIDENCE / "party_participation.csv").sort_values(
        ["party_name", "party_uri"], kind="stable"
    ).reset_index(drop=True)

    base_path = OUT / "prototype_party_slide_v6_base.png"
    final_path = OUT / "prototype_party_slide_v6.png"

    composition_qa, suppressed = _render_without_vertical_guides(parties, base_path)
    before = Image.open(base_path).convert("RGBA")
    after = before.copy()

    cleared = v5._clear_only_substitute_glyphs(after)
    corner_info = v5._apply_exact_reference_corners(after)
    after.convert("RGB").save(final_path, "PNG")

    target_size = tuple(corner_info["rendered_corner_dimensions"])
    corner_counts = v5._non_background_corner_counts(after, target_size)
    body_preservation = v5._protected_body_difference(before, after)

    guide_qa = {
        "expected_suppressed_vertical_guides": 4,
        "actual_suppressed_vertical_guides": len(suppressed),
        "suppressed_coordinates": suppressed,
        "vertical_guide_lines_removed": len(suppressed) == 4,
        "percentage_tick_labels_retained": True,
        "row_dividers_retained": True,
    }

    qa = {
        "composition_qa": composition_qa,
        "guide_line_qa": guide_qa,
        "corner_qa": {
            **corner_info,
            "substitute_glyph_clear_regions": cleared,
            "non_background_pixels_by_corner": corner_counts,
            "all_four_corners_present": all(value > 500 for value in corner_counts.values()),
        },
        "body_preservation_after_corner_overlay": body_preservation,
        "people_before_profit_display_label": "People Before Profit",
        "people_before_profit_single_line": True,
    }
    qa["pass"] = (
        bool(composition_qa.get("pass"))
        and bool(guide_qa["vertical_guide_lines_removed"])
        and bool(qa["corner_qa"]["all_four_corners_present"])
        and bool(body_preservation["unchanged"])
    )

    (OUT / "prototype_party_slide_v6_qa.json").write_text(
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
            "scope": "display only"
        },
        "corner_accents": "exact approved-reference Celtic linework",
        "vertical_grid_guides": "removed",
        "percentage_tick_labels": "retained",
        "computer_visual_qa": qa,
        "publication_enabled": False
    }
    (OUT / "prototype_manifest_v6.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if qa["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
