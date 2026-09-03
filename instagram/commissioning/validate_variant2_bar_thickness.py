from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("instagram/commissioning/output/party_issue_monthly_profile_v2/period=2026-07")
MAX_SHORT_BAR_THICKNESS_PX = 169.88
MIN_VISUAL_ROWS = 4


def check(name: str, condition: bool, detail: object = "") -> None:
    if not condition:
        message = f"{name}: {detail}"
        print(f"::error title=Variant 2 bar thickness QA::{message}")
        raise AssertionError(message)
    print(f"PASS {name}")


def main() -> None:
    run_manifest = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    geometry = run_manifest.get("chart_geometry") or {}
    check("min visual rows", geometry.get("min_visual_rows") == MIN_VISUAL_ROWS, geometry)
    check(
        "documented max short-chart thickness",
        abs(float(geometry.get("max_short_chart_bar_thickness_px")) - MAX_SHORT_BAR_THICKNESS_PX) < 0.01,
        geometry,
    )

    manifests = sorted(ROOT.glob("parties/*/metadata/variant-2/*-visual-manifest.json"))
    check("33 analytical visual manifests", len(manifests) == 33, len(manifests))

    short_charts = 0
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        readability = data.get("readability") or {}
        displayed = int(readability.get("displayed_item_count") or 0)
        effective = int(readability.get("effective_visual_row_count") or 0)
        min_rows = int(readability.get("min_visual_rows") or 0)
        thickness = float(readability.get("bar_thickness_px") or 0.0)

        check(f"{path.name} min_visual_rows", min_rows == MIN_VISUAL_ROWS, readability)
        check(f"{path.name} effective rows", effective >= max(displayed, MIN_VISUAL_ROWS), readability)

        if 0 < displayed < MIN_VISUAL_ROWS:
            short_charts += 1
            check(
                f"{path.name} short-chart thickness cap",
                thickness <= MAX_SHORT_BAR_THICKNESS_PX + 0.01,
                {"displayed_item_count": displayed, "bar_thickness_px": thickness},
            )
            check(
                f"{path.name} uses four-row thickness",
                abs(thickness - MAX_SHORT_BAR_THICKNESS_PX) < 0.01,
                {"displayed_item_count": displayed, "bar_thickness_px": thickness},
            )

    check("short charts exercised", short_charts > 0, short_charts)
    print(
        f"PASS: Variant 2 short-chart bar thickness cap — {short_charts} charts with 1-3 rows capped at {MAX_SHORT_BAR_THICKNESS_PX}px"
    )


if __name__ == "__main__":
    main()
