# Instagram system architecture

This document is the canonical entry point for understanding the Instagram-related systems in this repository.

The repository contains several generations of Instagram tooling. They serve different purposes and should not be treated as one interchangeable pipeline.

## 1. System map

| Subsystem | Primary paths | Purpose | Current status |
| --- | --- | --- | --- |
| Standalone visual generation | `instagram/visuals/`, `process/instagram_render_visual*.py` | Produce reusable chart/map/table/image PNG assets with metadata, manifests, test packs, and contact sheets | Active and validated |
| Post layout renderer | `instagram/renderer/`, `instagram/templates/`, `process/instagram_render_post.py` | Produce complete Instagram posts/carousels with text, images, branding, and layout | Active |
| Campaign renderer | `instagram/campaigns/`, `process/instagram_render_campaign.py` | Batch orchestration around post layouts and campaign specs | Active/specialized |
| External template pipeline | `instagram/mappings/`, `instagram/specs/`, `process/instagram_template_pipeline.py` | Test Bannerbear, Placid, and local HTML template rendering | Experimental/optional |
| Legacy media generators | `instagram/media_generators/`, `process/instagram_generate_media.py` | Earlier generator-specific chart/table tests | Retained for regression only |
| Option 5 AI image experiments | `process/instagram_option5_*`, `instagram/OPTION5_*`, related workflows | AI-assisted image generation/editing experiments | Separate manual-review path |
| Preview publishing | `.github/workflows/instagram_media_test.yml`, `instagram-preview-output` branch | Publish review artifacts and diagnostic links | Active and validated |
| Future publishing | Not yet implemented as an approved production path | Scheduling/approval/posting to Instagram | Explicitly out of scope |

## 2. Architectural boundary

There are two main production-oriented layers:

### 2.1 Standalone visual layer

Creates visual assets only:

- charts
- maps
- ranking tables
- table cards
- sourced-image placeholders

It should not be responsible for:

- post titles
- captions
- source footers
- branding overlays
- decorative corners
- scheduling or publishing

Canonical documentation:

```text
instagram/visuals/SYSTEM.md
```

### 2.2 Post layout layer

Creates complete Instagram post/carousel frames:

- headline placement
- subtitle placement
- media placement
- branding
- overlays
- slide layout
- full post composition

Primary implementation:

```text
instagram/renderer/
instagram/templates/
process/instagram_render_post.py
```

The intended integration is:

```text
standalone visual PNG
  -> inserted into post layout
  -> complete rendered Instagram post
```

## 3. Standalone visual generation system

Primary paths:

```text
instagram/visuals/
process/instagram_render_visual.py
process/instagram_render_visual_test_pack.py
process/instagram_prepare_visual_data.py
process/instagram_profile_s3_visual_data.py
process/instagram_check_s3_mapping_readiness.py
process/instagram_render_s3_smoke_samples.py
process/instagram_build_s3_smoke_contact_sheet.py
.github/workflows/instagram_media_test.yml
```

Responsibilities:

- reusable renderer templates
- sample bindings
- deterministic fixture rendering
- variation stress packs
- contact sheets
- local/S3 CSV loading
- raw-data mappings
- S3 schema profiling
- mapping readiness checks
- privacy-safe diagnostics
- preview publication

Current renderer families:

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

Canonical reference:

```text
instagram/visuals/SYSTEM.md
```

## 4. Post rendering system

Primary paths:

```text
instagram/renderer/
instagram/templates/
process/instagram_render_post.py
process/instagram_render_template.py
```

Key responsibilities:

- build post context
- resolve template layout
- render full slides
- place media and text
- use shared constants and fonts
- create final post/carousel PNGs

Important modules:

| Module | Purpose |
| --- | --- |
| `instagram/renderer/context.py` | Build structured post/member/constituency context |
| `instagram/renderer/data_loader.py` | Load repository/S3 data used by post rendering |
| `instagram/renderer/template_renderer.py` | Render JSON-defined layouts |
| `instagram/renderer/slides.py` | Slide-specific rendering logic |
| `instagram/renderer/charts.py` | Small embedded chart helpers used by the post renderer |
| `instagram/renderer/constants.py` | Shared bucket, region, font, and rendering constants |
| `instagram/renderer/fonts.py` | Font resolution |
| `instagram/renderer/render.py` | General rendering orchestration |
| `instagram/renderer/util.py` | Shared rendering utilities |

Layout definitions:

```text
instagram/templates/layouts/
```

Current JSON layouts include:

- `big_media_title_v1.json`
- `profile_card_main_v1.json`
- `profile_card_v1.json`
- `title_text_media_v1.json`

Palettes:

```text
instagram/templates/palettes/eirepolitic_dark.json
instagram/templates/palettes/eirepolitic_light.json
```

The post renderer is the correct place for titles, source text, account branding, and ornament overlays.

## 5. Campaign system

Primary paths:

```text
instagram/campaigns/
process/instagram_render_campaign.py
.github/workflows/instagram_campaign_render.yml
```

Purpose:

- define campaign briefs
- organize campaign decisions and data sources
- define media plans
- render batches of related posts
- record progress and review notes

Current campaign content includes:

```text
instagram/campaigns/member_profile_batch_v1/
```

Campaign folders may contain:

- `campaign_brief.md`
- `data_sources.md`
- `decisions.md`
- `media_plan.yml`
- `render_spec.yml`
- fixtures
- progress log
- review notes

Campaign orchestration should consume approved templates and visuals rather than duplicating renderer logic.

## 6. External template provider pipeline

Primary paths:

```text
instagram/mappings/
instagram/specs/
process/instagram_template_pipeline.py
.github/workflows/instagram_template_test.yml
.github/workflows/instagram_template_render_test.yml
```

Supported provider modes:

- Bannerbear
- Placid
- local HTML fallback

Purpose:

- test provider-specific dynamic field mappings
- produce request payloads
- exercise external template APIs
- fall back to local HTML when credentials/template IDs are missing

This path is separate from the standalone visual renderer registry.

Use it when evaluating external template services, not when adding a new chart renderer.

## 7. Legacy media generators

Primary paths:

```text
instagram/media_generators/
process/instagram_generate_media.py
```

Current legacy generators:

- horizontal bar chart
- ranking table

They include:

- generator-specific Python
- example specs
- fake-data cases
- limits documentation

Current policy:

- retain for regression coverage
- do not add new visual types here
- add new reusable visuals under `instagram/visuals/`

The primary media test workflow still runs these fake-data cases so existing behavior does not silently break.

## 8. Option 5 AI image experiments

Primary paths:

```text
instagram/OPTION5_LLM_IMAGE_TEST.md
instagram/OPTION5_MEMBER_PROFILE_AI_EDIT.md
process/instagram_option5_generate_images.py
process/instagram_option5_prepare_constituency_cover_test.py
process/instagram_option5_build_review_sheet.py
.github/workflows/instagram_option5_constituency_cover_ai.yml
.github/workflows/instagram_option5_member_profile_ai.yml
```

Purpose:

- AI-assisted image generation/editing tests
- constituency cover experiments
- member-profile image experiments
- review-sheet generation

This path must remain separately reviewed because image sourcing, likeness handling, and attribution/licensing differ from deterministic chart rendering.

## 9. Data and S3 architecture

Default project S3 settings are defined in:

```text
instagram/renderer/constants.py
```

Current visual S3 defaults:

```text
bucket: eirepolitic-data
region: ca-central-1
```

The standalone visual system supports:

- inline rows
- local CSV
- one S3 CSV
- first available S3 CSV from a candidate list

Visual mapping configs live in:

```text
instagram/visuals/data_mappings/
```

Currently validated live mappings:

| Mapping | Source | Field |
| --- | --- | --- |
| Debate issue counts | `processed/debates/debate_speeches_classified.csv` | `PoliticalIssues` |
| Member party counts | `raw/members/oireachtas_members_34th_dail.csv` | `party` |

The S3 schema profiler uses byte-range reads for lightweight discovery. Mapped rendering currently loads the full selected CSV object.

## 10. Review and validation model

The system uses two complementary validation paths.

### 10.1 Deterministic fixture validation

For every standalone visual:

- render one sample
- render multiple stress cases
- create one PNG per case
- create metadata and manifests
- create a labelled contact sheet
- publish outputs for review

Fixture stress tests include:

- item count extremes
- long labels
- wide/tight value ranges
- zero/negative values
- missing/blank values where relevant
- phone-readability cases
- map/geography fixtures
- attribution checks for sourced images

### 10.2 Live S3 smoke validation

The canonical live smoke flow:

1. profile S3 schema
2. create privacy-safe Markdown summary
3. verify mapping readiness
4. prepare chart-ready CSVs
5. render canonical smoke samples
6. create combined contact sheet
7. publish diagnostics and previews

Current canonical live smoke suite:

- debate issues → horizontal bar
- debate issues → vertical bar
- member parties → donut
- member parties → horizontal bar

## 11. Privacy and data handling

Public S3 schema diagnostics intentionally exclude sampled raw values.

The profiler defaults to:

```text
sampled_values_included: false
```

The Markdown summary refuses profiles containing:

- `example_values`
- `top_values`
- sampled-values flags set to true

Important distinction:

- schema diagnostics are privacy-hardened
- renderer metadata can contain actual rendered row values

Renderer metadata must therefore be treated as review data, not automatically public-safe data.

## 12. GitHub Actions map

| Workflow | Purpose |
| --- | --- |
| `instagram_media_test.yml` | Canonical manual standalone-visual regression and S3 smoke workflow |
| `instagram_visual_preview.yml` | Standalone visual preview workflow; requires default-branch availability for dispatch |
| `instagram_visual_s3_smoke.yml` | Standalone S3 smoke workflow; same dispatch limitation |
| `instagram_campaign_render.yml` | Campaign/post batch rendering |
| `instagram_constituency_test.yml` | Constituency post test |
| `instagram_template_test.yml` | External provider template tests |
| `instagram_template_render_test.yml` | Template render test |
| `instagram_s3_preview_test.yml` | Existing S3-backed post preview path |
| `instagram_option5_constituency_cover_ai.yml` | AI constituency cover experiment |
| `instagram_option5_member_profile_ai.yml` | AI member profile experiment |

Canonical current validation entry point:

```text
.github/workflows/instagram_media_test.yml
```

## 13. Preview output model

Dedicated review branch:

```text
instagram-preview-output
```

The primary visual workflow publishes:

- sample PNGs
- sample metadata/manifests
- variation PNGs
- test-pack metadata/manifests
- contact sheets
- S3 status
- schema diagnostics
- mapping readiness diagnostics
- S3 smoke visuals
- generated preview README links

These outputs are for review only.

## 14. Approval and publishing boundary

The current repository does not have an approved automated Instagram publishing path.

Explicitly out of scope:

- automatic approval
- automatic scheduling
- automatic posting
- silent removal of draft status
- unreviewed image acquisition

Promotion sequence should be:

```text
fixture validation
  -> contact-sheet review
  -> corrections
  -> live-data smoke validation
  -> explicit approval
  -> remove draft identifiers
  -> integrate with post/campaign layer
  -> separately design publishing workflow
```

## 15. Current technical debt

1. Renderer registry is duplicated in two scripts.
2. Visual filters support exact equality only.
3. Mapping transforms support only `count_by` and `sum_by`.
4. S3 render loading downloads complete CSV files.
5. Map fixtures are synthetic, not production boundary data.
6. Sourced-image acquisition/licensing is not implemented.
7. Metadata privacy classification is not enforced automatically.
8. Standalone visual workflows cannot be dispatched until present on the default branch.
9. Top-level Instagram documentation historically focused on external template tests and required this architecture index.
10. Several older and newer systems coexist; contributors must use the correct subsystem rather than extending the nearest similar file.

## 16. Where to make changes

| Goal | Correct location |
| --- | --- |
| Add a new reusable chart/map/table visual | `instagram/visuals/` |
| Change a standalone visual's style/limits | `instagram/visuals/templates/` |
| Add fixture stress cases | `instagram/visuals/tests/` |
| Add raw-data normalization for visuals | `instagram/visuals/data_mappings/` + `process/instagram_prepare_visual_data.py` |
| Add a complete post layout | `instagram/templates/layouts/` + post renderer |
| Change full-slide branding/text layout | `instagram/renderer/` or `instagram/templates/` |
| Add a campaign | `instagram/campaigns/` |
| Test Bannerbear/Placid | `instagram/mappings/`, `instagram/specs/`, template pipeline |
| Add AI image experiment | Option 5 files/workflows, with separate review |
| Add publishing/scheduling | New explicitly approved subsystem; do not attach to review workflows |

## 17. Canonical documentation hierarchy

Start here:

```text
instagram/ARCHITECTURE.md
```

Then use:

```text
instagram/visuals/SYSTEM.md
```

for detailed standalone visual implementation.

Existing specialized references remain useful:

```text
instagram/README.md
instagram/OPTION2_PYTHON_DETERMINISTIC_RENDERER.md
instagram/OPTION5_LLM_IMAGE_TEST.md
instagram/OPTION5_MEMBER_PROFILE_AI_EDIT.md
instagram/campaigns/*/*.md
instagram/media_generators/*/LIMITS.md
```

## 18. Operational recommendation

For current standalone visual work:

1. Make changes under `instagram/visuals/`.
2. Use `instagram_media_test.yml` for validation.
3. Inspect fixture and live S3 contact sheets.
4. Treat all outputs as review-only.
5. Keep `draft` identifiers until explicit approval.
6. Integrate approved visual PNGs into the post/campaign layer rather than duplicating visual logic there.
