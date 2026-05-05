from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import yaml


GENERATOR_MODULES = {
    "horizontal_bar_chart": "instagram.media_generators.horizontal_bar_chart.generator",
    "ranking_table": "instagram.media_generators.ranking_table.generator",
}


def run_spec(spec_path: str | Path, test_case: str = "default") -> dict:
    spec_path = Path(spec_path)
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if "cases" in doc:
        results = []
        for case in doc["cases"]:
            if test_case != "all" and case.get("id") != test_case:
                continue
            merged = dict(doc.get("base_spec", {}))
            merged["input"] = {"mode": "inline", "rows": case.get("rows", [])}
            merged["params"] = {**merged.get("params", {}), **case.get("params", {})}
            merged.setdefault("output", {})["slug"] = case["id"]
            results.append(run_single(merged))
        return {"success": True, "results": results}
    return run_single(doc)


def run_single(spec: dict) -> dict:
    generator = spec["generator"]
    module_name = GENERATOR_MODULES.get(generator)
    if not module_name:
        raise RuntimeError(f"Unsupported generator: {generator}")
    module = importlib.import_module(module_name)
    output = spec.get("output", {})
    root = Path(output.get("root", "generated_media"))
    slug = output.get("slug", "test")
    output_dir = root / generator / slug
    return module.render(spec, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Instagram media generator specs.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--test-case", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_spec(args.spec, args.test_case), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
