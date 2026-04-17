# Instagram Infographic Template Test Pipeline

This folder now supports two render paths for Instagram infographic visuals:

- `bannerbear`: external template API path
- `local_html`: existing repo-local HTML/CSS mock renderer

Scope:
- visuals only
- no caption generation
- no social copy
- no publishing automation
- no LLM-generated visual design

## Chosen test path

The selected implementation path is a **Bannerbear template API test with local HTML fallback**.

Why this was chosen:

- the repo already had a working constituency post context builder and a manual Instagram workflow
- the repo already follows a strong pattern of explicit YAML config plus Python runner plus GitHub Actions
- Bannerbear uses named template layer modifications, which fits this repo's existing spec-driven style well
- the manual work we cannot do yet is limited to creating templates and supplying template IDs / API key
- everything else can be built now: data prep, field mapping, workflow entry point, request payloads, fallback rendering, and docs

## Current repo structure for the template test

```text
instagram/
  README.md
  mappings/
    bannerbear_constituency_basic.yml
  specs/
    bannerbear_constituency_test.yml
    constituency_test_post.yml
  templates/
    components.html
    slide_glossary.html
    slide_member_profile.html
    slide_methodology.html
    slide_overview.html
    slide_top_issues.html
    styles.css
process/
  instagram_render_post.py
  instagram_template_pipeline.py
.github/workflows/
  instagram_constituency_test.yml
  instagram_template_test.yml
```

## What was already in the repo and is now reused

The pipeline reuses the existing S3-backed dataset loading pattern and constituency carousel context builder.

Primary inputs:

- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_summaries.csv`
- `processed/members/member_photos/members_photo_urls.csv`
- `processed/members/members_photo_urls.csv` fallback
- `processed/debates/debate_speeches_classified.csv`
- `processed/constituencies/constituency_images.csv`

## Template test architecture

`process/instagram_template_pipeline.py` does this:

1. Loads the YAML post spec.
2. Builds the same constituency/member context already used by the local renderer.
3. Enriches the context with explicit computed text fields for template binding.
4. Loads the Bannerbear placeholder mapping file.
5. Builds one modification payload per enabled slide.
6. Tries to render via Bannerbear.
7. If Bannerbear credentials or template IDs are missing, falls back to `local_html` unless disabled.
8. Writes render status, request payloads, responses, context JSON, and generated images into a deterministic post folder.

## Output convention

```text
generated_posts/<post_slug>/
  post_context.json
  render_status.json
  png/
    01_overview.png
    02_member_profile.png
    03_top_issues.png
    04_glossary.png
  bannerbear/
    requests/
      01_overview.json
      02_member_profile.json
      03_top_issues.json
      04_glossary.json
    responses/
      ...
  html/
    ... only when fallback local_html is used
```

## Chosen first test asset

The default first test asset is a **constituency-based slide set** using real repo data.

Default spec:

- constituency: `Dublin Bay South`
- automatic member selection unless overridden
- slides:
  - `overview`
  - `member_profile`
  - `top_issues`
  - `glossary`

## Post spec

The new spec is:

- `instagram/specs/bannerbear_constituency_test.yml`

It controls:

- output slug and dimensions
- provider selection
- fallback behavior
- template mapping file
- constituency and optional member override
- enabled slides
- theme metadata and glossary text

## Explicit template mapping

The explicit placeholder mapping file is:

- `instagram/mappings/bannerbear_constituency_basic.yml`

It defines, per slide key:

- which Bannerbear template ID to use
- which placeholder names must exist in that template
- which context field populates each placeholder
- whether the placeholder is text or image
- any transform to apply

### Placeholder names expected by the current mapping

`overview`
- `headline`
- `constituency_name`
- `td_count`
- `party_count`
- `top_issue`
- `vote_participation`
- `speech_rank`
- `constituency_map`
- `footer_note`
- `account_name`

`member_profile`
- `headline`
- `member_name`
- `party_name`
- `constituency_name`
- `member_top_issue`
- `member_vote_participation`
- `member_speech_rank`
- `member_background`
- `member_photo`
- `footer_note`
- `account_name`

`top_issues`
- `headline`
- `constituency_name`
- `issue_summary`
- `issue_note`
- `footer_note`
- `account_name`

`glossary`
- `headline`
- `issues_glossary`
- `vote_participation_glossary`
- `speech_rank_glossary`
- `footer_note`
- `account_name`

## Local run

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

Set AWS environment variables.

### Run the external-template path with fallback enabled

```bash
python process/instagram_template_pipeline.py \
  --spec instagram/specs/bannerbear_constituency_test.yml
```

### Force the local mock renderer

```bash
python process/instagram_template_pipeline.py \
  --spec instagram/specs/bannerbear_constituency_test.yml \
  --provider local_html
```

### Override the constituency or member

```bash
python process/instagram_template_pipeline.py \
  --spec instagram/specs/bannerbear_constituency_test.yml \
  --constituency "Wicklow-Wexford" \
  --member-name "Jennifer Whitmore"
```

## GitHub Actions run

Use the workflow:

- **Generate Instagram Template Test Post (Manual)**

Inputs:

- `constituency`
- `member_name`
- `provider`
- `spec_path`

The workflow will:

1. install dependencies
2. attempt Bannerbear rendering
3. fall back to local HTML if Bannerbear is not fully configured yet
4. upload the generated post folder as an artifact

## Secrets expected for the Bannerbear path

These are wired into the workflow already:

- `BANNERBEAR_API_KEY`
- `BANNERBEAR_TEMPLATE_UID_OVERVIEW`
- `BANNERBEAR_TEMPLATE_UID_MEMBER_PROFILE`
- `BANNERBEAR_TEMPLATE_UID_TOP_ISSUES`
- `BANNERBEAR_TEMPLATE_UID_GLOSSARY`

If these are absent, the pipeline still produces a mock output via `local_html` unless fallback is disabled.

## What is already autonomous vs manual

Already built in-repo:

- data fetching and normalization from existing S3 datasets
- post spec format for the external-template test
- explicit placeholder mapping config
- Bannerbear request payload generation
- Bannerbear API render adapter
- local fallback renderer
- manual GitHub Actions entry point
- deterministic artifact output structure

Still manual later:

- creating the actual Bannerbear templates
- ensuring layer names match the mapping file
- copying the template UIDs
- adding the Bannerbear API key and template IDs as GitHub secrets

## Current design tradeoff for the test

The `top_issues` external-template slide is intentionally mapped as a multiline ranked issue list instead of a dynamic bar chart.

That keeps the first external-template test:

- explicit
- low-friction
- easy to bind
- easy to inspect for text overflow
- easy to recreate visually in Bannerbear

The existing local HTML renderer still keeps the richer bar-chart mock path available for comparison.
