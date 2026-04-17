# Instagram infographic test pipeline

This test pipeline renders Instagram carousel visuals directly from structured data using a deterministic Python stack.

Scope:

- visuals only
- no captions
- no post text
- no image generation model
- no Power BI dependency
- no HTML/CSS renderer for this pipeline

## First test target

The first target remains a constituency carousel because the repo already has the best supporting inputs for it:

- member table
- member photo URL table
- member summary table
- constituency image index
- debate issue classification table

## Rendering stack

- Python orchestrator
- Pillow for slide composition and typography
- matplotlib for chart rendering
- cairosvg for SVG constituency assets when needed
- requests for remote image fetches
- boto3 and pandas for S3 table loading

## Repo structure

```text
instagram/
  README.md
  specs/
    constituency_test_post.yml
  renderer/
    __init__.py
    charts.py
    constants.py
    context.py
    data_loader.py
    fonts.py
    render.py
    slides.py
    util.py
process/
  instagram_render_post.py
.github/workflows/
  instagram_constituency_test.yml
tests/
  test_instagram_renderer.py
  fixtures/instagram/
```

## Existing datasets reused

The renderer loads the first available CSV from these existing pipeline outputs:

- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_summaries.csv`
- `processed/members/member_photos/members_photo_urls.csv`
- `processed/members/members_photo_urls.csv`
- `processed/debates/debate_speeches_classified.csv`
- `processed/constituencies/constituency_images.csv`

## Output convention

```text
generated_posts/<post_slug>/
  png/
    01_overview.png
    02_member_profile.png
    03_top_issues.png
    04_member_top_issues.png
    05_glossary.png
  post_context.json
  render_manifest.json
```

Filenames are deterministic from slide order and slide key.

## Local test run with fixtures

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

python process/instagram_render_post.py \
  --spec instagram/specs/constituency_test_post.yml \
  --data-source local \
  --data-root tests/fixtures/instagram \
  --output-dir generated_posts_local
```

## S3 run

```bash
python process/instagram_render_post.py \
  --spec instagram/specs/constituency_test_post.yml
```

Optional overrides:

```bash
python process/instagram_render_post.py \
  --spec instagram/specs/constituency_test_post.yml \
  --constituency "Wicklow-Wexford" \
  --member-name "Example TD"
```

## Font policy

The renderer uses system fonts already available on GitHub Actions Ubuntu runners:

- DejaVu Sans
- Liberation Sans fallback
- PIL default fallback if neither exists

No font files are stored in the repo.

## Current limits

- vote participation and ranking values remain spec-driven placeholders until a source table is connected
- speaker-to-member joins rely on normalized name matching
- constituency image matching still depends on filename matching in the constituency image index
- remote image URLs must be reachable during render runs
