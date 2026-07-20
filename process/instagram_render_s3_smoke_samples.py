from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SMOKE_SAMPLES = [
    {
        "id": "debate_issues_horizontal_bar",
        "sample": "instagram/visuals/samples/horizontal_bar_s3_debate_issues_draft_v1.sample.yml",
        "output_root": "generated_visuals/s3_smoke/debate_issues_horizontal_bar",
    },
    {
        "id": "debate_issues_vertical_bar",
        "sample": "instagram/visuals/samples/vertical_bar_s3_debate_issues_draft_v1.sample.yml",
        "output_root": "generated_visuals/s3_smoke/debate_issues_vertical_bar",
    },
    {
        "id": "member_parties_donut_chart",
        "sample": "instagram/visuals/samples/donut_chart_s3_member_parties_draft_v1.sample.yml",
        "output_root": "generated_visuals/s3_smoke/member_parties_donut_chart",
    },
    {
        "id": "member_parties_horizontal_bar",
        "sample": "instagram/visuals/samples/horizontal_bar_s3_member_parties_draft_v1.sample.yml",
        "output_root": "generated_visuals/s3_smoke/member_parties_horizontal_bar",
    },
]


def run_samples(samples: list[dict[str, str]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for sample in samples:
        command = [
            sys.executable,
            "process/instagram_render_visual.py",
            "--sample",
            sample["sample"],
            "--output-root",
            sample["output_root"],
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        result = {
            "id": sample["id"],
            "sample": sample["sample"],
            "output_root": sample["output_root"],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "success": completed.returncode == 0,
        }
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    return {"success": True, "sample_count": len(results), "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render all mapped S3 smoke visual samples.")
    parser.add_argument("--manifest", default="generated_visuals/s3_smoke/smoke_samples.manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_samples(SMOKE_SAMPLES)
    manifest_path = REPO_ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
