from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_review_page(output_root: str | Path) -> Path:
    root = Path(output_root)
    review_manifest_path = root / "review_manifest.json"
    context_path = root / "metadata" / "post_context.json"
    caption_path = root / "caption.txt"

    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    caption = caption_path.read_text(encoding="utf-8").strip()

    slide_files = [Path(path) for path in review.get("slide_files", [])]
    if len(slide_files) != 3:
        raise RuntimeError(f"Expected exactly 3 carousel slides, found {len(slide_files)}")
    for path in slide_files:
        if not path.exists():
            raise FileNotFoundError(path)

    cards = []
    labels = ["Latest support", "Change since previous model", "90-day trend"]
    for index, (path, label) in enumerate(zip(slide_files, labels), start=1):
        cards.append(
            f'''<article class="card">
              <div class="card-head"><span>Slide {index}</span><strong>{html.escape(label)}</strong></div>
              <img src="{_data_uri(path)}" alt="{html.escape(label)}" loading="eager" />
            </article>'''
        )

    source = (context.get("source_attributions") or [{}])[0]
    source_name = html.escape(str(source.get("display_name") or "Irish Polling Indicator (IPI)"))
    source_url = html.escape(str(source.get("reference_url") or ""), quote=True)
    latest_date = html.escape(str(context.get("latest_model_date") or ""))
    previous_date = html.escape(str(context.get("previous_model_date") or ""))

    document = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>EirePolitic · IPI polling carousel review</title>
<style>
:root {{ color-scheme: dark; --bg:#0b2119; --panel:#12352a; --text:#f4ead7; --muted:#cbbf9f; --line:#315448; --accent:#d8b45f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Arial,Helvetica,sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width:1500px; margin:0 auto; padding:28px; }}
h1 {{ margin:0 0 8px; font-size:clamp(26px,4vw,46px); }}
.meta {{ color:var(--muted); margin-bottom:24px; line-height:1.5; }}
.grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; align-items:start; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; overflow:hidden; box-shadow:0 8px 30px rgba(0,0,0,.22); }}
.card-head {{ padding:12px 14px; display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:13px; }}
.card-head strong {{ color:var(--text); font-weight:600; text-align:right; }}
.card img {{ display:block; width:100%; height:auto; background:#0f2f24; }}
section {{ margin-top:24px; background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; }}
pre {{ white-space:pre-wrap; word-break:break-word; margin:0; font:14px/1.55 Arial,Helvetica,sans-serif; color:var(--text); }}
a {{ color:var(--accent); }}
.badge {{ display:inline-block; padding:4px 8px; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px; margin-right:8px; }}
@media (max-width:1000px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:16px; }} .card {{ max-width:620px; margin:0 auto; width:100%; }} }}
</style>
</head>
<body>
<main>
  <h1>IPI polling carousel review</h1>
  <div class="meta">
    <span class="badge">Human review required</span>
    <span class="badge">Not published to Instagram</span><br><br>
    Latest model date: <strong>{latest_date}</strong> · Previous model date: <strong>{previous_date}</strong><br>
    Source: <a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_name}</a>
  </div>
  <div class="grid">{''.join(cards)}</div>
  <section>
    <h2>Caption</h2>
    <pre>{html.escape(caption)}</pre>
  </section>
  <section>
    <h2>Review checks</h2>
    <ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in review.get('checks', []))}</ul>
  </section>
</main>
</body>
</html>'''

    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    output = review_dir / "review_index.html"
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained browser review page for the IPI polling carousel")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    path = build_review_page(args.output_root)
    print(path)


if __name__ == "__main__":
    main()
