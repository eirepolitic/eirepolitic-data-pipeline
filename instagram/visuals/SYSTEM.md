# Instagram visual system

This document is the canonical technical reference for the standalone Instagram visual generation system under `instagram/visuals/`.

The system is review-only. It generates visual assets and diagnostics. It does not publish, schedule, approve, or upload Instagram posts.

## 1. Purpose and boundary

The visual system creates reusable PNG assets such as charts, tables, maps, and sourced-image placeholders. These assets are intended to be inserted into a separate post-layout system.

A generated visual should contain the visual itself. Post-level concerns belong elsewhere:

- titles and subtitles
- captions and social copy
- source/footer text
- branding overlays
- corner ornamentation
- post scheduling and publishing

The repo contains other Instagram systems. They are related but distinct:

| System | Primary paths | Purpose | Relationship to this system |
| --- | --- | --- | --- |
| Standalone visual layer | `instagram/visuals/`, `process/instagram_render_visual*.py` | Reusable charts, maps, tables, and visual assets | This document |
| Post/template renderer | `instagram/renderer/`, `instagram/templates/`, `process/instagram_render_post.py` | Complete Instagram post or carousel layouts | Can place standalone visual PNGs into post layouts |
| Campaign renderer | `instagram/campaigns/`, `process/instagram_render_campaign.py` | Batch/campaign orchestration around post templates | Future consumer of approved standalone visuals |
| External template tests | `instagram/mappings/`, `instagram/specs/`, `process/instagram_template_pipeline.py` | Bannerbear/Placid/local HTML provider experiments | Separate render path; not the visual renderer registry |
| Legacy media generators | `instagram/media_generators/`, `process/instagram_generate_media.py` | Earlier generator-specific fake-data tests | Retained for regression; do not expand for new visual types |
| Option 5 AI-image tests | `process/instagram_option5_*`, related workflows/docs | Experimental AI image workflows | Separate, manually reviewed path |

## 2. Design principles

1. **Configuration before code.** Visual style and limits live in YAML templates. Data bindings and provenance live in YAML samples.
2. **Deterministic review outputs.** Fixture test packs generate repeatable PNGs, metadata, manifests, and contact sheets.
3. **Renderer isolation.** Each visual type has one renderer module with a shared function contract.
4. **Data-source independence.** Renderers receive rows and do not need to know whether rows came from inline data, local CSV, or S3.
5. **Review before approval.** All current visual IDs and templates use `draft_v1`; remove `draft` only after visual approval.
6. **Visual-only output.** Titles, source text, and post decoration belong to the post-layout layer.
7. **Mobile-first styling.** Default dimensions, label limits, item limits, warnings, and stress cases target phone readability.
8. **No automated publishing.** Workflows publish only to review artifacts and the `instagram-preview-output` branch.
9. **Privacy-safe public diagnostics.** Public S3 schema summaries omit sampled raw values by default and refuse unsafe profiles.

## 3. Directory structure

```text
instagram/visuals/
  README.md                 Quick-start and operational notes
  SYSTEM.md                 Canonical architecture and component reference
  templates/                Visual style, dimensions, limits, renderer selection
  samples/                  Example bindings and canonical review inputs
  renderers/                Python implementations
  data_mappings/            Raw-data to chart-ready CSV transformations
  fixtures/                 Small deterministic shared datasets
  tests/<visual_id>/        Variation registry, fixture CSVs, optional geography
```

Supporting orchestration lives in `process/` and `.github/workflows/`.

## 4. End-to-end data flow

### 4.1 Single sample render

```text
sample YAML
  -> template YAML
  -> rows_from_sample()
  -> optional equals filters
  -> renderer registry lookup
  -> renderer.render(...)
  -> PNG + metadata JSON + render manifest JSON
```

Command:

```bash
python process/instagram_render_visual.py \
  --sample instagram/visuals/samples/horizontal_bar_draft_v1.sample.yml \
  --output-root generated_visuals/horizontal_bar_draft_v1
```

### 4.2 Variation test pack

```text
cases.yml
  -> one local CSV input per case
  -> renderer for every case
  -> one PNG/metadata/manifest per case
  -> three-column labelled contact sheet
  -> test_pack_manifest.json
```

Command:

```bash
python process/instagram_render_visual_test_pack.py \
  --registry instagram/visuals/tests/horizontal_bar_draft_v1/cases.yml \
  --output-root generated_visual_tests/horizontal_bar_draft_v1
```

### 4.3 S3-backed smoke flow

```text
mapping YAML
  -> ranged schema profile
  -> privacy-safe Markdown summary
  -> mapping-readiness gate
  -> full CSV load for approved mapping
  -> count_by/sum_by transform
  -> chart-ready local CSV
  -> canonical smoke sample renders
  -> combined S3 smoke contact sheet
  -> preview branch publication
```

The schema profiler reads only an S3 byte range. The actual mapped render step currently downloads the complete selected CSV object.

## 5. Core configuration contracts

### 5.1 Visual template YAML

Location:

```text
instagram/visuals/templates/<visual_id>.yml
```

Common fields:

| Field | Purpose |
| --- | --- |
| `template_id` | Stable template identifier |
| `status` | Current lifecycle status, normally `draft` |
| `renderer` | Key in the renderer registry |
| `description` | Intended use |
| `params` | Dimensions, item limits, formats, layout-specific settings |
| `palette` | Visual colors; defaults are supplied by `load_palette()` |
| `constraints` | Design restrictions such as no borders or built-in ornaments |

Common default palette:

| Token | Default |
| --- | --- |
| `background` | `#0f2f24` |
| `panel` | `#173d30` |
| `panel_alt` | `#214a3b` |
| `text` | `#f4ead7` |
| `muted` | `#cbbf9f` |
| `accent` | `#d8b45f` |
| `accent_2` | `#9ec5a2` |
| `grid` | `#cbbf9f` |
| `warning` | `#b55b5b` |

### 5.2 Sample YAML

Location:

```text
instagram/visuals/samples/<sample>.sample.yml
```

Common fields:

| Field | Purpose |
| --- | --- |
| `visual_id` | Output filename/asset identifier |
| `template` | Path to template YAML |
| `input` | Input mode and source configuration |
| `bindings` | Maps renderer roles to dataset column names |
| `filters` | Optional pre-render row filters |
| `grouping` | Batch intent and future grouping metadata |
| `source_note` | Review provenance note |
| `attribution` | Source name, URL, retrieval time, licence, notes |
| `geography` | Optional geography/layout fixture for map renderers |

Some fixture samples still contain `title` and `subtitle` as descriptive metadata. Standalone renderers should not draw these into the PNG.

### 5.3 Input modes

Implemented by `instagram/visuals/renderers/common.py`:

| Mode | Required configuration | Behavior |
| --- | --- | --- |
| `inline` | `rows` | Uses rows embedded in YAML |
| `local_csv` | `path` | Reads a repository/local CSV with UTF-8 BOM support |
| `s3_csv` | `key`, optional `bucket`, `region`, `required` | Downloads one full CSV object |
| `s3_csv_first_available` | `keys`, optional `bucket`, `region`, `required` | Tries keys in order and downloads the first available object |

Bucket resolution order:

1. `input.bucket`
2. `INSTAGRAM_VISUAL_S3_BUCKET`
3. repository default bucket

Region resolution order:

1. `input.region`
2. `AWS_REGION`
3. `AWS_DEFAULT_REGION`
4. repository default region

### 5.4 Filters

`process/instagram_render_visual.py` and the test-pack runner currently support only:

```yaml
filters:
  - field: constituency
    operator: equals
    value: Dublin Bay South
```

Unsupported operators raise an error. Filtering is exact string equality and is performed after data loading.

### 5.5 Renderer function contract

Every renderer module exposes:

```python
render(
    template,
    sample,
    rows,
    output_png,
    metadata_path,
    manifest_path,
    input_metadata,
) -> dict
```

Expected behavior:

1. Resolve bindings and clean rows.
2. Apply renderer-specific sorting, limits, formatting, and validation.
3. Record non-fatal issues in `warnings`.
4. Create parent directories.
5. Write the PNG.
6. Write metadata JSON.
7. Write render-manifest JSON.
8. Return the manifest.

### 5.6 Output contracts

Single visual:

```text
<output-root>/png/<visual_id>.png
<output-root>/metadata/<visual_id>.json
<output-root>/manifests/<visual_id>.render_manifest.json
```

Test pack:

```text
<output-root>/png/<case_id>.png
<output-root>/metadata/<case_id>.json
<output-root>/manifests/<case_id>.render_manifest.json
<output-root>/contact_sheet.png
<output-root>/test_pack_manifest.json
```

Metadata generally contains:

- visual/template/renderer identifiers
- creation time
- input metadata
- bindings, filters, grouping
- attribution and source note
- cleaned/rendered row data
- warnings

Render manifests generally contain:

- success flag
- identifiers
- output paths
- dimensions
- warnings
- creation time

**Data classification:** render metadata can contain actual rendered values. It is a review artifact, not a privacy-sanitized schema diagnostic. Do not publicly publish render metadata for sensitive datasets without an explicit review.

## 6. Renderer registry and component catalog

The registry is duplicated in:

- `process/instagram_render_visual.py`
- `process/instagram_render_visual_test_pack.py`

A new visual type must be added to both registries until the registry is centralized.

| Renderer | Intended use | Typical bindings | Current notes/limitations |
| --- | --- | --- | --- |
| `horizontal_bar` | Ranked categories | `label`, `value`, optional `group` | Sorts, truncates to max items, wraps labels, warns on long/negative/empty data |
| `vertical_bar` | Compact category comparison | `label`, `value` | Better for fewer categories; long labels produce readability warnings |
| `line_chart` | Ordered/time trend | x/time, value, optional series | Fixture validated; live time-series mapping not yet added |
| `area_chart` | Filled trend/volume | x/time, value, optional series | Fixture validated; best for ordered numeric/time axes |
| `stacked_bar` | Composition by category | category, segment, value | Fixture validated; requires long-form grouped rows |
| `ranking_table` | Ordered rows with values/ranks | label/name, value, optional detail | Fixture validated; legacy ranking-table generator also exists separately |
| `donut_chart` | Part-to-whole distribution | label, value | Live S3 member-party smoke validated |
| `scatter_plot` | Relationship between two metrics | x, y, optional label/group | Fixture validated; live two-metric mapping not yet added |
| `dot_plot` | Compact ranked/comparative values | label, value | Fixture validated |
| `lollipop_chart` | Ranked values with emphasis | label, value | Fixture validated |
| `slope_chart` | Change between two endpoints | label, start, end | Fixture validated; requires paired values per entity |
| `table_card` | Key/value or ranked tabular panel | renderer-specific columns | Fixture validated |
| `small_multiples` | Repeated mini-series/panels | panel/group, x, value | Fixture validated; more data-dense and requires phone review |
| `choropleth_map` | Values assigned to polygons | geography key, value | Uses deterministic fixture geometry; not a production electoral boundary map |
| `point_map` | Locations/markers | latitude, longitude, optional label/value | Fixture validated; does not provide basemap acquisition/geocoding |
| `tile_map` | Simplified geographic tile layout | geography key, value | Uses synthetic tile-layout JSON; not a true geographic map |
| `sourced_image_asset` | Review wrapper/placeholder for sourced imagery | image/source metadata fields | Does not download live images; fixture placeholder only; licensing/attribution gate required before real image ingestion |

## 7. Testing and QA

Each visual family normally has:

```text
instagram/visuals/tests/<visual_id>/cases.yml
instagram/visuals/tests/<visual_id>/data/*.csv
instagram/visuals/tests/<visual_id>/geography/*   # where applicable
```

Test registries provide shared bindings and attribution plus a list of cases. Each case identifies a local CSV and may override filters or geography.

Stress dimensions represented across the suite include:

- minimum, normal, maximum, and overflow item counts
- short, medium, long, and very long labels
- small, large, tight-range, and wide-range values
- zero and negative values where relevant
- missing/blank inputs where relevant
- multiple groups or series
- phone-readability worst cases
- map/geography fixture coverage
- attribution completeness for sourced-image assets

The contact sheet is a review surface, not an assertion framework. A successful test-pack command means the renders completed; reviewers must still inspect warnings and visual quality.

## 8. Data preparation and mappings

Runner:

```text
process/instagram_prepare_visual_data.py
```

Config location:

```text
instagram/visuals/data_mappings/*.yml
```

Supported operations:

| Operation | Required fields | Output |
| --- | --- | --- |
| `count_by` | one matched label field | one row per label with a count |
| `sum_by` | matched label and numeric value fields | one row per label with summed value |

Candidate-field lists make mappings tolerant of known schema variants. The first matching candidate is selected.

Current mappings:

| Mapping | Source | Validated field | Output |
| --- | --- | --- | --- |
| `fixture_issue_counts_local` | Horizontal-bar fixture CSV | `issue`, `speech_count` | `generated_visual_data/fixture_issue_counts_local.csv` |
| `debate_issue_counts_s3` | `processed/debates/debate_speeches_classified.csv` | `PoliticalIssues` | `generated_visual_data/debate_issue_counts_s3.csv` |
| `member_party_counts_s3` | `raw/members/oireachtas_members_34th_dail.csv` | `party` | `generated_visual_data/member_party_counts_s3.csv` |

Mappings can cap output rows and combine overflow into an `Other` row. They write a transformation manifest alongside the chart-ready CSV.

## 9. S3 schema profiling, readiness, and privacy

### 9.1 Profiler

```text
process/instagram_profile_s3_visual_data.py
```

Purpose:

- discover columns without downloading the complete dataset
- inspect a limited number of rows from an S3 range read
- report populated/blank counts
- identify likely numeric columns
- show which mapping candidates match available columns

Defaults:

- 262,144-byte range
- 25 sampled rows
- sampled raw values excluded

The JSON contains `sampled_values_included: false` by default. `--include-sampled-values` exists for explicit private debugging and must not be used for public preview outputs.

### 9.2 Markdown summary and privacy guard

```text
process/instagram_summarize_s3_schema_profile.py
```

The public Markdown summary shows:

- S3 object metadata
- columns
- sample coverage counts
- likely numeric columns
- mapping candidate matches

It does not show example or top sampled values.

The summary generator refuses profiles containing:

- top-level `sampled_values_included: true`
- schema-level `sampled_values_included: true`
- `example_values`
- `top_values`

`--allow-sampled-values` is an explicit override intended only for private debugging.

### 9.3 Mapping readiness

```text
process/instagram_check_s3_mapping_readiness.py
```

Rules:

- `count_by` requires a matched label field
- `sum_by` requires matched label and value fields
- every mapping requires detected columns and sampled rows
- unknown operations are treated as needing label and value matches and produce a warning

With `--fail-on-not-ready`, the command exits non-zero when a mapping cannot safely proceed.

## 10. Canonical S3 smoke suite

Driver:

```text
process/instagram_render_s3_smoke_samples.py
```

Contact-sheet builder:

```text
process/instagram_build_s3_smoke_contact_sheet.py
```

Validated canonical samples:

| Dataset | Visual |
| --- | --- |
| Debate issue counts | Horizontal bar |
| Debate issue counts | Vertical bar |
| Member party counts | Donut chart |
| Member party counts | Horizontal bar |

Canonical output roots:

```text
generated_visuals/s3_smoke/debate_issues_horizontal_bar/
generated_visuals/s3_smoke/debate_issues_vertical_bar/
generated_visuals/s3_smoke/member_parties_donut_chart/
generated_visuals/s3_smoke/member_parties_horizontal_bar/
```

The contact-sheet builder renders the canonical suite by default, discovers render manifests, prefers canonical roots over legacy roots, deduplicates by `visual_id`, and writes:

```text
generated_visuals/s3_smoke/contact_sheet.png
generated_visuals/s3_smoke/contact_sheet.manifest.json
generated_visuals/s3_smoke/smoke_samples.manifest.json
```

## 11. GitHub Actions

### 11.1 Primary validated workflow

```text
.github/workflows/instagram_media_test.yml
```

Manual workflow responsibilities:

1. Install Python dependencies.
2. Run legacy media-generator fake-data cases as regression coverage.
3. Run the deterministic local mapping fixture.
4. Render all draft visual samples.
5. Render all visual variation packs/contact sheets.
6. If AWS credentials exist:
   - profile S3 schemas
   - create privacy-safe Markdown summary
   - run mapping-readiness validation
   - prepare mapped CSVs
   - render canonical S3 smoke suite
   - build S3 smoke contact sheet
7. Write `s3_smoke_status.json`.
8. Publish review outputs to `instagram-preview-output`.
9. Upload a complete Actions artifact.

The S3 smoke block is non-blocking for the overall workflow. Possible status values:

- `succeeded`
- `skipped_missing_aws_credentials`
- `failed_non_blocking`

This design keeps fixture regression outputs available when AWS is unavailable or a live schema changes.

### 11.2 Other workflows

| Workflow | Purpose/status |
| --- | --- |
| `instagram_visual_preview.yml` | Standalone visual preview workflow; dispatch requires the workflow to exist on the default branch |
| `instagram_visual_s3_smoke.yml` | Standalone S3 smoke workflow; same default-branch dispatch constraint |
| `instagram_campaign_render.yml` | Campaign/post rendering system, separate from visual test packs |
| `instagram_template_*` | Post/external-template tests, separate from standalone visual registry |
| `instagram_s3_preview_test.yml` | Existing post/preview S3 path, not the canonical visual smoke workflow |
| `instagram_option5_*` | AI-image experiments requiring separate review |

Until the standalone visual workflows are present on the default branch, the dispatchable `instagram_media_test.yml` workflow is the canonical validation entry point.

## 12. Preview branch and artifacts

Preview branch:

```text
instagram-preview-output
```

Fixture outputs:

```text
preview/visuals/samples/<visual_id>/...
preview/visuals/tests/<visual_id>/png/<case_id>.png
preview/visuals/tests/<visual_id>/metadata/<case_id>.json
preview/visuals/tests/<visual_id>/manifests/<case_id>.render_manifest.json
preview/visuals/tests/<visual_id>/contact_sheet.png
preview/visuals/tests/<visual_id>/test_pack_manifest.json
```

S3 outputs:

```text
preview/visuals/smoke/s3/status/s3_smoke_status.json
preview/visuals/smoke/s3/generated_visual_data/
preview/visuals/smoke/s3/contact_sheet/
preview/visuals/smoke/s3/visuals/
```

The workflow regenerates `preview/visuals/README.md` with direct links to contact sheets, samples, S3 diagnostics, and canonical smoke folders.

## 13. Dependencies and runtime assumptions

Important runtime dependencies include:

- Python 3.11 in the primary workflow
- Pillow for contact sheets and several renderers
- Matplotlib for chart rendering
- PyYAML for configuration
- boto3/botocore for S3
- pandas/pyarrow elsewhere in the repository pipeline

Renderers use headless/non-interactive output. Matplotlib renderers set the `Agg` backend.

Fonts are selected from repository/system candidates in `instagram.renderer.constants`. When no candidate exists, Pillow contact sheets fall back to a default font.

## 14. Known limitations and technical debt

1. Renderer registry is duplicated between the single-render and test-pack scripts.
2. Only exact `equals` filtering is supported.
3. Mapping operations are limited to `count_by` and `sum_by`.
4. S3 render loading downloads complete CSV objects; it does not stream or use parquet/query pushdown.
5. Schema range reads can end inside a quoted/multiline CSV record; profiling is lightweight, not a full CSV validation service.
6. Contact-sheet success confirms execution, not visual approval.
7. Current map fixtures are synthetic. Production constituency maps require reviewed geographic source data and joins.
8. The sourced-image renderer is a placeholder/review wrapper and does not perform licensed image acquisition.
9. Visual metadata may contain rendered row values and should not automatically be treated as public-safe.
10. Sample YAMLs contain attribution fields but attribution completeness is not uniformly enforced across every renderer.
11. Titles/subtitles can exist in sample YAML as descriptive fields; renderers should continue to ignore them for standalone PNGs.
12. All visual IDs remain draft and are not approved production assets.

## 15. Adding a new visual type

1. Add `instagram/visuals/templates/<visual_id>.yml`.
2. Add `instagram/visuals/renderers/<renderer>.py` implementing the shared contract.
3. Add the renderer key to both registries.
4. Add `instagram/visuals/samples/<visual_id>.sample.yml`.
5. Add `instagram/visuals/tests/<visual_id>/cases.yml` and deterministic fixture data.
6. Add the visual ID to the sample/test loops in the primary workflow.
7. Run the workflow and inspect the contact sheet and warnings.
8. Correct layout/readability issues without weakening the test cases.
9. Add an S3 mapping/sample only after the fixture renderer is stable.
10. Keep `draft` in the ID until explicit approval.

## 16. Adding a new live S3 mapping

1. Identify the S3 CSV and expected schema.
2. Add a mapping YAML under `data_mappings/` with candidate fields.
3. Add it to the schema-profile command.
4. Review the privacy-safe schema summary.
5. Add it to the readiness command and require readiness.
6. Run `instagram_prepare_visual_data.py` to produce a normalized CSV.
7. Add one or more mapped sample YAMLs using `local_csv` against the normalized output.
8. Add canonical sample definitions to `instagram_render_s3_smoke_samples.py`.
9. Verify contact sheet, manifests, and preview publication.
10. Document whether the mapping is fixture-only, smoke-validated, or production-approved.

## 17. Validation status

Validated through the primary GitHub Actions workflow:

- all 17 draft renderer families execute against deterministic fixtures
- sample PNG generation
- metadata and render manifests
- variation test packs
- contact-sheet generation
- preview-branch publication
- local mapping regression
- live S3 range schema profiling
- privacy-safe JSON/Markdown schema diagnostics
- mapping readiness validation
- debate `PoliticalIssues` mapping
- member `party` mapping
- four-case canonical S3 smoke contact sheet
- complete Actions artifact upload

Validated does not mean visually approved or production-ready. Approval remains a separate human review step.

## 18. Current operational recommendation

Use `instagram_media_test.yml` as the canonical manual regression workflow. Keep fixture tests deterministic. Add live S3 coverage incrementally after each renderer is visually stable. Treat preview outputs as review artifacts, and do not remove `draft` identifiers or connect publishing automation until explicit approval.
