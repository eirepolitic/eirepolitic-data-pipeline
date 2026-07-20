# Instagram Infographic Template Test Pipeline

This folder now supports three render paths for Instagram infographic visuals:

- `bannerbear`: external template API path
- `placid`: external template API path
- `local_html`: existing repo-local HTML/CSS mock renderer

Scope:
- visuals only
- no caption generation
- no social copy
- no publishing automation
- no LLM-generated visual design

## Chosen test paths

The repo now supports **two concrete external-template tests**:

- Bannerbear
- Placid

Why these were chosen:

- the repo already had a working constituency post context builder and a manual Instagram workflow
- the repo already follows a strong pattern of explicit YAML config plus Python runner plus GitHub Actions
- both Bannerbear and Placid are template-driven image APIs built around dynamic fields/layers, which fits this repo's explicit data-binding style
- the manual work we cannot do yet is limited to creating templates and supplying template IDs / API keys
- everything else can be built now: data prep, field mapping, workflow entry point, request payloads, fallback rendering, and docs

## Current repo structure for the template tests

```text
instagram/
  README.md
  mappings/
    bannerbear_constituency_basic.yml
    placid_constituency_basic.yml
  specs/
    bannerbear_constituency_test.yml
    placid_constituency_test.yml
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
4. Loads the provider-specific mapping file.
5. Builds one request payload per enabled slide.
6. Tries to render via Bannerbear or Placid.
7. If provider credentials or template IDs are missing, falls back to `local_html` unless disabled.
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
    responses/
  placid/
    requests/
    responses/
  html/
    ... only when fallback local_html is used
```

## Chosen first test asset

The default first test asset is a **constituency-based slide set** using real repo data.

Default setup:

- constituency: `Dublin Bay South`
- automatic member selection unless overridden
- slides:
  - `overview`
  - `member_profile`
  - `top_issues`
  - `glossary`

## Test specs

Two provider-specific specs are now included:

- `instagram/specs/bannerbear_constituency_test.yml`
- `instagram/specs/placid_constituency_test.yml`

Both control:

- output slug and dimensions
- provider selection
- fallback behavior
- template mapping file
- constituency and optional member override
- enabled slides
- theme metadata and glossary text

## Explicit template mapping

Provider mapping files:

- `instagram/mappings/bannerbear_constituency_basic.yml`
- `instagram/mappings/placid_constituency_basic.yml`

They define, per slide key:

- which external template ID to use
- which placeholder or layer names must exist in that template
- which context field populates each placeholder or layer
- whether it is text or image
- any transform to apply

### Shared layer / placeholder names expected by the current tests

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

### Run the Bannerbear test with fallback enabled

```bash
python process/instagram_template_pipeline.py \
  --spec instagram/specs/bannerbear_constituency_test.yml
```

### Run the Placid test with fallback enabled

```bash
python process/instagram_template_pipeline.py \
  --spec instagram/specs/placid_constituency_test.yml
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
- `provider_suite`

`provider_suite` values:

- `bannerbear`
- `placid`
- `both`
- `local_html`

The workflow will:

1. install dependencies
2. run the requested provider suite
3. fall back to local HTML if the chosen external provider is not fully configured yet
4. upload the generated post folder as an artifact

## Secrets expected for the Bannerbear path

- `BANNERBEAR_API_KEY`
- `BANNERBEAR_TEMPLATE_UID_OVERVIEW`
- `BANNERBEAR_TEMPLATE_UID_MEMBER_PROFILE`
- `BANNERBEAR_TEMPLATE_UID_TOP_ISSUES`
- `BANNERBEAR_TEMPLATE_UID_GLOSSARY`

## Secrets expected for the Placid path

- `PLACID_API_TOKEN`
- `PLACID_TEMPLATE_UUID_OVERVIEW`
- `PLACID_TEMPLATE_UUID_MEMBER_PROFILE`
- `PLACID_TEMPLATE_UUID_TOP_ISSUES`
- `PLACID_TEMPLATE_UUID_GLOSSARY`

If these are absent, the pipeline still produces a mock output via `local_html` unless fallback is disabled.

## What is already autonomous vs manual

Already built in-repo:

- data fetching and normalization from existing S3 datasets
- provider-specific post spec formats for external-template tests
- explicit placeholder and layer mapping configs
- Bannerbear request payload generation
- Placid request payload generation
- Bannerbear API render adapter
- Placid API render adapter
- local fallback renderer
- manual GitHub Actions entry point for one or both providers
- deterministic artifact output structure

Still manual later:

- creating the actual Bannerbear templates
- creating the actual Placid templates
- ensuring layer names match the mapping files
- copying the template IDs
- adding API keys and template IDs as GitHub secrets

## Current design tradeoff for the first external-template tests

The `top_issues` slide is intentionally mapped as a multiline ranked issue list instead of a dynamic bar chart.

That keeps the first external-template tests:

- explicit
- low-friction
- easy to bind
- easy to inspect for text overflow
- easy to recreate visually in Bannerbear or Placid

The existing local HTML renderer still keeps the richer bar-chart mock path available for comparison.
