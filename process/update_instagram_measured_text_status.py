from __future__ import annotations

from pathlib import Path

PLAN_PATH = Path("instagram/CONTENT_FACTORY_PLAN.md")

OLD = """- automatic whitespace, media-fill, aspect-ratio, plot-utilization, font-size, bar-thickness, wrapping, and value-label-headroom gates
- shared tall metric covers and shortened titles for party and constituency projects
- generic, party, constituency, and historical-fallback regression coverage

Latest live historical validation loaded eight historical batches, replaced zero current waivers, and retained three waivers. This confirms the fallback process; it does not imply the waived edge cases are impossible in future data.

Remaining validation gap:

- direct rendered-text bounding-box clipping detection and dynamic font sizing remain future renderer work
"""

NEW = """- automatic whitespace, media-fill, aspect-ratio, plot-utilization, font-size, bar-thickness, wrapping, and value-label-headroom gates
- measured title shrink, line-count, truncation, and bounding-box validation
- pixel-measured category-label wrapping and dynamic font sizing
- adaptive chart margins and value-axis headroom
- direct category-label and value-label clipping detection with zero-tolerance thresholds
- shared tall metric covers and shortened titles for party and constituency projects
- generic, party, constituency, historical-fallback, and live S3 measured-text regression coverage

Latest live historical validation loaded eight historical batches, replaced zero current waivers, and retained three waivers. This confirms the fallback process; it does not imply the waived edge cases are impossible in future data.
"""


def main() -> None:
    text = PLAN_PATH.read_text(encoding="utf-8")
    if OLD in text:
        PLAN_PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        return
    if "direct rendered-text bounding-box clipping detection and dynamic font sizing remain future renderer work" in text:
        raise SystemExit("Measured-text status exists in an unexpected format")


if __name__ == "__main__":
    main()
