# Instagram Infographic Test Pipeline

This folder contains a **test pipeline** for generating Instagram carousel slide visuals from existing Eirepolitic datasets.

Scope:
- visuals only
- no captions
- no post text
- no publishing automation
- HTML/CSS rendered to PNG via Playwright

## First test output

The first test output is a **constituency carousel** with 4 slides:
1. constituency overview
2. top issues from debate speeches by TDs in the constituency
3. member profile card
4. methodology / data sources

## Proposed file structure

```text
instagram/
  README.md
  specs/
    constituency_test_post.yml
  templates/
    components.html
    slide_overview.html
    slide_top_issues.html
    slide_member_profile.html
    slide_methodology.html
    styles.css
process/
  instagram_render_post.py
.github/workflows/
  instagram_constituency_test.yml
```

## Data inputs used

Primary inputs are existing S3 outputs already aligned to repo pipelines:

- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_summaries.csv`
- `processed/members/member_photos/members_photo_urls.csv`
- `processed/members/members_photo_urls.csv` fallback
- `processed/debates/debate_speeches_classified.csv`
- `processed/constituencies/constituency_images.csv`

## How the test pipeline works

`process/instagram_render_post.py`:

1. Loads a YAML post spec.
2. Reads existing datasets from S3.
3. Builds a post context for one constituency.
4. Renders HTML slides using Jinja2 templates.
5. Uses Playwright to export each slide to Instagram-sized PNG.
6. Writes outputs into one deterministic folder per post.

## Output convention

Default local output root:

```text
generated_posts/<post_slug>/
  html/
    01_overview.html
    02_top_issues.html
    03_member_profile.html
    04_methodology.html
  png/
    01_overview.png
    02_top_issues.png
    03_member_profile.png
    04_methodology.png
  post_context.json
```

The GitHub Actions workflow uploads the post folder as a workflow artifact.

## Rendering approach

- HTML templates with Jinja2
- shared CSS in `instagram/templates/styles.css`
- small macro/component layer in `instagram/templates/components.html`
- SVG bars generated in Python for deterministic issue charts
- Playwright Chromium screenshots at portrait Instagram size

## Defaults

- slide size: 1080 x 1350
- deterministic output names
- 4-slide test carousel
- first member card defaults to the highest speech-count TD in the chosen constituency

## Local run

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jinja2 playwright
python -m playwright install chromium
```

Set AWS environment variables, then run:

```bash
python process/instagram_render_post.py --spec instagram/specs/constituency_test_post.yml
```

Optional overrides:

```bash
python process/instagram_render_post.py \
  --spec instagram/specs/constituency_test_post.yml \
  --constituency "Dublin Bay South" \
  --output-dir generated_posts
```

## GitHub Actions run

Use the workflow **Generate Instagram Constituency Test Post (Manual)**.

Inputs:
- `constituency`
- `member_name` (optional)
- `spec_path`

The workflow builds the HTML and PNG files, then uploads them as an artifact.

## What is already implemented now

- spec-driven post generation
- S3 dataset loading with fallback paths
- constituency/member/debate issue joins
- HTML/CSS templates
- Playwright screenshot export
- manual GitHub Actions workflow

## Decisions that still genuinely need user input later

Only the minimum design choices remain:
- final palette / branding tweaks
- final preferred constituency default
- whether the member card should default to highest speech count or a fixed named TD
