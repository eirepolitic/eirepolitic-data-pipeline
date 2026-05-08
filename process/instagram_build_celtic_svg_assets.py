from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path("instagram/assets/celtic_corners")

SVG_TEMPLATE = """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 300 300\" width=\"300\" height=\"300\">
  <rect width=\"300\" height=\"300\" fill=\"none\"/>
  <g fill=\"none\" stroke=\"#f4ead7\" stroke-width=\"{stroke}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
{paths}
  </g>
</svg>
"""

VARIANTS = {
    "interlace_arch": {
        "stroke": 11,
        "paths": [
            "M28 154 C44 104,88 54,148 32 C199 14,246 26,274 58",
            "M28 154 C76 154,109 121,115 84 C121 46,91 28,58 43 C30 56,20 91,42 119 C67 151,114 154,148 126 C178 101,177 58,144 45",
            "M52 270 C38 238,42 197,74 170 C106 143,151 150,174 181 C195 209,185 249,153 267",
            "M44 214 C78 246,126 239,150 203 C172 171,156 124,116 119",
            "M86 52 C116 92,159 111,206 100 C244 91,269 66,281 39",
            "M140 35 C124 74,133 110,166 130 C200 151,247 139,274 104",
            "M33 34 L83 34 M34 33 L34 83 M266 58 C252 82,232 98,207 104",
            "M37 263 C64 255,83 236,92 212"
        ],
    },
    "rounded_scroll": {
        "stroke": 10,
        "paths": [
            "M24 52 C68 20,131 18,174 54 C206 81,214 127,187 156 C161 184,112 181,94 146 C78 113,111 79,145 94 C173 107,178 145,155 166",
            "M62 28 C50 72,69 110,105 128 C149 150,203 134,236 98 C252 80,264 58,270 36",
            "M30 270 C42 235,69 212,104 205 C143 213,166 252,142 278",
            "M32 212 C62 250,113 257,148 226 C182 196,179 139,137 121",
            "M37 37 C67 60,93 72,122 64 C152 56,171 36,181 20",
            "M20 181 C36 171,52 148,46 120 C39 91,22 75,15 62",
            "M213 42 C237 44,256 53,276 72",
            "M42 213 C44 237,53 256,72 276"
        ],
    },
    "manuscript_panel": {
        "stroke": 8,
        "paths": [
            "M24 24 H172 V70 H70 V172 H24 Z",
            "M56 24 V139 C56 166,78 188,105 188 C132 188,154 166,154 139 C154 112,132 90,105 90 C78 90,56 112,56 139",
            "M24 56 H139 C166 56,188 78,188 105 C188 132,166 154,139 154 C112 154,90 132,90 105 C90 78,112 56,139 56",
            "M24 172 C76 128,128 76,172 24",
            "M172 70 C122 102,102 122,70 172",
            "M194 24 C230 24,260 39,280 66",
            "M24 194 C24 230,39 260,66 280",
            "M203 46 C230 69,249 89,274 101",
            "M46 203 C69 230,89 249,101 274"
        ],
    },
    "floral_long": {
        "stroke": 8,
        "paths": [
            "M26 80 C64 40,122 34,165 64 C197 86,205 124,181 148 C157 172,116 164,105 132 C93 98,133 76,158 100",
            "M92 32 C118 68,154 88,200 82 C239 76,264 50,282 22",
            "M25 275 C46 250,72 239,103 244 C134 250,160 270,172 292",
            "M27 180 C44 156,49 126,39 96 C33 79,24 66,13 56",
            "M180 27 C156 44,126 49,96 39 C79 33,66 24,56 13",
            "M34 216 C72 248,124 247,160 210 C197 172,192 112,148 91",
            "M70 271 C82 247,108 234,135 239",
            "M271 70 C247 82,234 108,239 135",
            "M188 38 C225 36,258 48,286 74",
            "M38 188 C36 225,48 258,74 286"
        ],
    },
    "minimal_triquetra": {
        "stroke": 10,
        "paths": [
            "M28 78 C66 28,127 28,164 78 C128 72,100 87,82 116 C65 143,68 178,90 205 C47 188,20 149,28 78",
            "M164 78 C204 28,265 28,287 78 C248 73,220 91,205 121 C188 153,199 188,230 210 C186 212,143 194,122 158 C103 125,114 95,164 78",
            "M90 205 C119 226,155 228,185 210 C167 258,112 286,62 260 C75 246,84 228,90 205",
            "M82 116 C120 128,156 114,164 78",
            "M122 158 C135 123,170 102,205 121",
            "M90 205 C114 184,123 169,122 158",
            "M32 34 C70 54,95 58,126 48",
            "M34 266 C67 248,91 242,122 248",
            "M266 34 C248 67,242 91,248 122"
        ],
    },
}


def build_all(out_dir: Path = OUT_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, spec in VARIANTS.items():
        path_lines = [f'    <path d="{d}"/>' for d in spec["paths"]]
        svg = SVG_TEMPLATE.format(stroke=spec["stroke"], paths="\n".join(path_lines))
        out = out_dir / f"{name}.svg"
        out.write_text(svg, encoding="utf-8")
        paths.append(str(out))
    manifest = {"success": True, "assets": paths, "notes": "White transparent SVG corner assets for deterministic Instagram template rendering."}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate white transparent Celtic corner SVG assets.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    print(json.dumps({"success": True, "assets": build_all(Path(args.out_dir))}, indent=2))


if __name__ == "__main__":
    main()
