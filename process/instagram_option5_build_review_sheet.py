from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    manifest_csv = run_root / "metadata" / "generated_manifest.csv"
    review_dir = run_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_csv = review_dir / "review_sheet.csv"

    rows: List[Dict[str, str]] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    **row,
                    "brand_consistency": "",
                    "factual_correctness_visible_text": "",
                    "text_legibility": "",
                    "repeatability_note": "",
                    "better_than_deterministic_template": "",
                    "approved": "",
                    "review_notes": "",
                }
            )

    with review_csv.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Review sheet refreshed: {review_csv.resolve()}")


if __name__ == "__main__":
    main()
