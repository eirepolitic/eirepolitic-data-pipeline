# Instagram Infographic Test Pipeline

This folder contains a test pipeline for generating Instagram carousel slide visuals from existing Eirepolitic datasets.

Scope:
- visuals only
- no captions
- no post text
- no publishing automation
- HTML/CSS rendered to PNG via Playwright

## First implemented test target

The first implemented target is a constituency-style carousel based on your older post format.

Supported slide types now:
- `constituency_overview`
- `member_profile`
- `top_issues`
- `glossary`

You can control what each post contains from the YAML spec.

## File structure

```text
instagram/
  README.md
  specs/
    constituency_test_post.yml
  templates/
    components.html
    slide_glossary.html
    slide_member_profile.html
    slide_overview.html
    slide_top_issues.html
    styles.css
process/
  instagram_render_post.py
.github/workflows/
  instagram_constituency_test.yml
```

## Existing datasets reused

Primary inputs are existing S3 outputs already aligned to repo pipelines:

- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_summaries.csv`
- `processed/members/member_photos/members_photo_urls.csv`
- `processed/members/members_photo_urls.csv` fallback
- `processed/debates/debate_speeches_classified.csv`
- `processed/constituencies/constituency_images.csv`

## Post spec control

The YAML spec controls:
- constituency
- optional named member override
- slide order
- which slides are enabled
- per-slide content options
- optional manually supplied metrics such as vote participation and rank
- glossary text
- theme settings

### Example controls

```yaml
slides:
  - key: overview
    type: constituency_overview
    enabled: true
    content:
      show_map: true
      show_top_issue: true
      show_vote_participation: true
      show_speech_rank: true

  - key: member_top_issues
    type: top_issues
    enabled: true
    content:
      scope: member
      issue_limit: 8
```

## Current design direction

The default theme now follows the previously created post examples more closely:
- dark green background
- ivory text and bars
- centered headline layout
- framed map / photo area
- classic ornamental corners
- simple horizontal bar charts

## How the renderer works

`process/instagram_render_post.py`:

1. Loads the YAML spec.
2. Reads the first available CSV from each configured S3 dataset path.
3. Builds a constituency context.
4. Selects a TD automatically or from the spec.
5. Computes constituency and member issue counts from classified debate speeches.
6. Renders enabled slides to HTML.
7. Uses Playwright to export PNG files.
8. Writes outputs into a deterministic post folder.

## Output convention

```text
generated_posts/<post_slug>/
  html/
    01_overview.html
    02_member_profile.html
    03_top_issues.html
    04_member_top_issues.html
    05_glossary.html
  png/
    01_overview.png
    02_member_profile.png
    03_top_issues.png
    04_member_top_issues.png
    05_glossary.png
  post_context.json
```

## Metric handling

The current renderer computes issue counts from real datasets.

Metrics such as these are currently spec-driven placeholders unless a dedicated source table is added later:
- vote participation percentage
- speech rank
- constituency speech rank

That keeps the test pipeline honest and avoids inventing data.

## Local run

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
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
  --constituency "Wicklow-Wexford" \
  --member-name "Fionntán Ó Súilleabháin" \
  --output-dir generated_posts
```

## GitHub Actions run

Use the workflow **Generate Instagram Constituency Test Post (Manual)**.

Inputs:
- `constituency`
- `member_name` optional
- `spec_path`

The workflow renders the enabled slides and uploads the generated post folder as an artifact.

## Known limits in the current test build

- debate issue joins depend on normalized speaker-to-member name matching
- vote participation and ranking values are not auto-derived yet unless added later from a real dataset
- ornamental corners are implemented as a simple classic motif, not a custom vector asset from your old manual process
- image matching for constituency maps depends on filename matching in the constituency image index
