# Instagram systems

This folder contains several related but distinct Instagram subsystems. Start here to identify the correct path before changing code.

## Start here

- **Complete architecture:** `instagram/ARCHITECTURE.md`
- **Standalone visual system:** `instagram/visuals/SYSTEM.md`
- **Visual quick start:** `instagram/visuals/README.md`

The current repository is review-first. It generates and validates visual/post assets, but it does not contain an approved automated Instagram publishing path.

## System map

| Subsystem | Primary paths | Purpose | Status |
| --- | --- | --- | --- |
| Standalone visual generation | `instagram/visuals/`, `process/instagram_render_visual*.py` | Reusable charts, maps, tables, and image assets | Active and validated |
| Post layout renderer | `instagram/renderer/`, `instagram/templates/`, `process/instagram_render_post.py` | Complete Instagram post and carousel frames | Active |
| Campaign renderer | `instagram/campaigns/`, `process/instagram_render_campaign.py` | Batch/campaign orchestration around post layouts | Active/specialized |
| External template tests | `instagram/mappings/`, `instagram/specs/`, `process/instagram_template_pipeline.py` | Bannerbear, Placid, and local HTML experiments | Experimental/optional |
| Legacy media generators | `instagram/media_generators/`, `process/instagram_generate_media.py` | Earlier chart/table generator tests | Regression only |
| Option 5 AI image tests | `process/instagram_option5_*`, `instagram/OPTION5_*` | AI-assisted image generation/editing experiments | Separate manual-review path |
| Preview publishing | `.github/workflows/instagram_media_test.yml`, `instagram-preview-output` | Publish review artifacts and diagnostics | Active and validated |
| Production publishing | Not implemented as an approved path | Scheduling and posting to Instagram | Out of scope |

## Core architectural rule

The standalone visual layer and the post layout layer are separate.

```text
standalone visual PNG
  -> inserted into post layout
  -> complete Instagram post/carousel
```

Standalone visuals should contain only the visual itself. Titles, subtitles, branding, source/footer text, overlays, and decorative elements belong to the post layout layer.

## Repository map

```text
instagram/
  README.md                 This landing page
  ARCHITECTURE.md           Complete Instagram architecture
  visuals/                  Standalone visual generation system
    README.md               Visual quick start
    SYSTEM.md               Detailed visual implementation reference
    templates/              Visual style, dimensions, and limits
    samples/                Data bindings and provenance
    renderers/              Python visual renderers
    tests/                  Stress cases and fixture data
    data_mappings/          Raw-data normalization for visuals
  renderer/                 Complete post/carousel rendering engine
  templates/                Post layouts, palettes, HTML/CSS assets
  campaigns/                Campaign briefs, specs, fixtures, and logs
  media_generators/         Legacy generator-specific implementations
  mappings/                 External template provider field mappings
  specs/                    Post/provider test specs
```

Supporting scripts live under `process/`, and manual validation workflows live under `.github/workflows/`.

## Where do I make this change?

| I want to... | Correct location |
| --- | --- |
| Add a reusable chart, map, table, or visual asset | `instagram/visuals/` |
| Change a standalone visual's dimensions, palette, or limits | `instagram/visuals/templates/` |
| Add or change deterministic visual stress cases | `instagram/visuals/tests/` |
| Add raw-data normalization for visuals | `instagram/visuals/data_mappings/` and `process/instagram_prepare_visual_data.py` |
| Add a complete post layout | `instagram/templates/layouts/` and the post renderer |
| Change full-post text, branding, or media placement | `instagram/renderer/` or `instagram/templates/` |
| Add campaign/batch orchestration | `instagram/campaigns/` and `process/instagram_render_campaign.py` |
| Test Bannerbear or Placid | `instagram/mappings/`, `instagram/specs/`, and `process/instagram_template_pipeline.py` |
| Maintain an older media generator | `instagram/media_generators/` |
| Add a new reusable visual type | Do **not** use `instagram/media_generators/`; use `instagram/visuals/` |
| Work on AI image experiments | Option 5 files and workflows, with separate review |
| Add publishing or scheduling | Create a separately approved subsystem; do not attach it to review workflows |

## Standalone visual generation

Canonical documentation:

```text
instagram/visuals/SYSTEM.md
```

The visual system currently supports 17 draft renderer families:

- horizontal bar
- vertical bar
- line chart
- area chart
- stacked bar
- ranking table
- donut chart
- scatter plot
- dot plot
- lollipop chart
- slope chart
- table card
- small multiples
- point map
- choropleth map
- tile map
- sourced image asset

It produces:

- PNG assets
- metadata JSON
- render manifests
- variation test packs
- contact sheets
- local and S3-backed review outputs

Current canonical validation workflow:

```text
.github/workflows/instagram_media_test.yml
```

This workflow renders all fixture samples/test packs, performs mapping regression checks, optionally runs live S3 smoke tests, publishes preview outputs, and uploads a complete Actions artifact.

## Post rendering

Primary implementation:

```text
instagram/renderer/
instagram/templates/
process/instagram_render_post.py
```

This layer is responsible for complete post composition, including:

- titles and subtitles
- branding
- account name
- source/footer text
- image placement
- slide layout
- overlays and decorative elements

Current JSON layouts:

```text
instagram/templates/layouts/big_media_title_v1.json
instagram/templates/layouts/profile_card_main_v1.json
instagram/templates/layouts/profile_card_v1.json
instagram/templates/layouts/title_text_media_v1.json
```

## Campaign rendering

Primary paths:

```text
instagram/campaigns/
process/instagram_render_campaign.py
.github/workflows/instagram_campaign_render.yml
```

Campaign folders organize briefs, decisions, data sources, fixtures, render specs, media plans, progress logs, and review notes.

Campaign logic should reuse approved post templates and standalone visual assets instead of duplicating rendering code.

## External template provider tests

The repository includes an older but still useful provider-test pipeline for:

- Bannerbear
- Placid
- local HTML fallback

Primary paths:

```text
instagram/mappings/
instagram/specs/
process/instagram_template_pipeline.py
.github/workflows/instagram_template_test.yml
```

Use this path to evaluate external template services. Do not use it to register new standalone chart renderers.

## Legacy media generators

Primary paths:

```text
instagram/media_generators/
process/instagram_generate_media.py
```

Current legacy generators include:

- horizontal bar chart
- ranking table

They remain in the primary media test workflow for regression coverage. New visual types belong under `instagram/visuals/`.

## Option 5 AI image experiments

Specialized references:

```text
instagram/OPTION5_LLM_IMAGE_TEST.md
instagram/OPTION5_MEMBER_PROFILE_AI_EDIT.md
```

Related scripts/workflows:

```text
process/instagram_option5_generate_images.py
process/instagram_option5_prepare_constituency_cover_test.py
process/instagram_option5_build_review_sheet.py
.github/workflows/instagram_option5_constituency_cover_ai.yml
.github/workflows/instagram_option5_member_profile_ai.yml
```

These experiments have different sourcing, likeness, attribution, and review requirements from deterministic chart rendering.

## Data and S3

The standalone visual loader supports:

- inline rows
- local CSV
- one S3 CSV
- first available S3 CSV from candidate keys

Current validated live mappings:

| Dataset | Source | Validated field |
| --- | --- | --- |
| Debate issue counts | `processed/debates/debate_speeches_classified.csv` | `PoliticalIssues` |
| Member party counts | `raw/members/oireachtas_members_34th_dail.csv` | `party` |

The S3 schema profiler uses byte-range reads for discovery. Public schema diagnostics omit sampled raw values by default and include a guard that refuses unsafe profiles.

## Review workflow

The current visual review lifecycle is:

```text
fixture sample
  -> variation stress pack
  -> individual PNGs
  -> metadata/manifests
  -> contact sheet
  -> preview branch
  -> human review
  -> live S3 smoke test
  -> explicit approval
```

Current preview branch:

```text
instagram-preview-output
```

Preview outputs are review artifacts. They do not represent automatic content approval or publication.

## Approval and publishing boundary

The following are intentionally not automated:

- final visual approval
- removal of `draft` identifiers
- Instagram scheduling
- Instagram posting
- image licensing approval
- real sourced-image acquisition

The expected promotion sequence is:

```text
fixture validation
  -> contact-sheet review
  -> corrections
  -> live-data smoke validation
  -> explicit approval
  -> remove draft identifiers
  -> post/campaign integration
  -> separately approved publishing design
```

## Documentation hierarchy

Read in this order:

1. `instagram/README.md` — navigation and subsystem routing
2. `instagram/ARCHITECTURE.md` — full architecture, boundaries, workflows, and technical debt
3. `instagram/visuals/SYSTEM.md` — detailed standalone visual implementation
4. `instagram/visuals/README.md` — operational quick start
5. Specialized campaign, Option 5, provider-test, and generator documentation as needed

Additional references:

```text
instagram/OPTION2_PYTHON_DETERMINISTIC_RENDERER.md
instagram/OPTION5_LLM_IMAGE_TEST.md
instagram/OPTION5_MEMBER_PROFILE_AI_EDIT.md
instagram/campaigns/*/*.md
instagram/media_generators/*/LIMITS.md
```

## Current recommendation

For new visual development:

1. Work under `instagram/visuals/`.
2. Keep fixture tests deterministic.
3. Run `.github/workflows/instagram_media_test.yml`.
4. Inspect contact sheets and warnings.
5. Add live S3 mappings only after fixture stability.
6. Treat outputs as review-only.
7. Keep `draft` identifiers until explicit approval.
8. Integrate approved visual PNGs into the post/campaign layer rather than duplicating visual logic.
