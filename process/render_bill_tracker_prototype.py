#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import yaml

from instagram.renderer.template_renderer import render_template
from instagram.renderer.template_renderer import load_json

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "instagram/campaigns/bill_tracker_v1/prototype_bindings.yml"
OUT = ROOT / "instagram/campaigns/bill_tracker_v1/previews"


def main() -> None:
    doc = yaml.safe_load(BINDINGS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    cover = load_json(ROOT / "instagram/templates/layouts/bill_tracker_cover_v1.json")
    card = load_json(ROOT / "instagram/templates/layouts/bill_tracker_card_v1.json")
    render_template(cover, doc["cover"], OUT / "01_cover.png")
    render_template(card, doc["bill_card"], OUT / "02_bill_card_gas_reserve.png")
    print(OUT / "01_cover.png")
    print(OUT / "02_bill_card_gas_reserve.png")


if __name__ == "__main__":
    main()
