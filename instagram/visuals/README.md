# Instagram visual generation layer

Review-only reusable visual asset generation for the Eirepolitic Instagram content factory.

This layer is separate from `instagram/templates/layouts/`, which defines complete post layouts. Visuals generated here are standalone PNG assets that can be placed inside those post layouts. Visual PNGs contain only the visual itself; titles, subtitles, source text, and post ornamentation belong to the post layout layer.

## Draft visual programme

Build each planned visual as a draft first:

1. Add a visual template/config and deterministic renderer.
2. Add a realistic sample binding.
3. Add a variation test registry covering item counts, label lengths, value ranges, zero/negative values where relevant, and phone-readability stress cases.
4. Render one standalone PNG per test case.
5. Render one labelled contact sheet for the complete test pack.
6. Publish PNGs, metadata, manifests, and contact sheet to `instagram-preview-output`.
7. Continue to the next draft visual without waiting for detailed review.
8. Complete a later bulk review and correction pass across all contact sheets.
9. Remove `draft` from names and internal IDs only after approval.

This contact-sheet process is the standard QA and regression-testing method for every visual type.

## Current draft visuals

- `horizontal_bar_draft_v1`
- `vertical_bar_draft_v1`
- `line_chart_draft_v1`
- `stacked_bar_draft_v1`
- `ranking_table_draft_v1`
- `choropleth_map_draft_v1`
- `point_map_draft_v1`
- `table_card_draft_v1`
- `small_multiples_draft_v1`
- `area_chart_draft_v1`
- `scatter_plot_draft_v1`
- `dot_plot_draft_v1`
- `lollipop_chart_draft_v1`
- `slope_chart_draft_v1`
- `donut_chart_draft_v1`
- `tile_map_draft_v1`
- `sourced_image_asset_draft_v1`

## Visual data mappings

Raw datasets do not need to match renderer bindings directly. Use the data preparation CLI to map raw rows into chart-ready CSVs:

```text
process/instagram_prepare_visual_data.py --config <mapping.yml>
```

Supported mapping operations:

- `count_by`
- `sum_by`

Current mapping configs:

```text
instagram/visuals/data_mappings/fixture_issue_counts_local.yml
instagram/visuals/data_mappings/debate_issue_counts_s3.yml
instagram/visuals/data_mappings/member_party_counts_s3.yml
```

The standard preview workflow runs the local fixture mapping as a regression check and uploads `generated_visual_data/` as an artifact. S3 mappings run inside a non-blocking smoke step in the dispatchable media workflow.

## S3-backed visual smoke tests

The shared visual loader supports these input modes:

- `inline`
- `local_csv`
- `s3_csv`
- `s3_csv_first_available`

S3 modes are for review-only smoke testing and future live-data bindings. They do not publish, schedule, or approve Instagram content.

Mapped S3 samples:

```text
instagram/visuals/samples/horizontal_bar_s3_debate_issues_draft_v1.sample.yml
instagram/visuals/samples/donut_chart_s3_member_parties_draft_v1.sample.yml
```

Run requirements for live S3 rendering:

- repository secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- bucket defaults to `eirepolitic-data`
- region defaults to `ca-central-1`

The dispatchable media workflow always writes this summary artifact:

```text
generated_visual_data/s3_smoke_status.json
```

Expected `status` values:

```text
succeeded
skipped_missing_aws_credentials
failed_non_blocking
```

When S3 smoke succeeds, the preview workflow also publishes the smoke-only outputs separately:

```text
branch: instagram-preview-output
preview/visuals/smoke/s3/status/s3_smoke_status.json
preview/visuals/smoke/s3/contact_sheet/contact_sheet.png
preview/visuals/smoke/s3/contact_sheet/contact_sheet.manifest.json
preview/visuals/smoke/s3/generated_visual_data/s3_schema_profile.json
preview/visuals/smoke/s3/generated_visual_data/
preview/visuals/smoke/s3/visuals/debate_issues/
preview/visuals/smoke/s3/visuals/member_parties/
```

Create the lightweight S3 schema profile without downloading full datasets:

```text
process/instagram_profile_s3_visual_data.py \
  --config instagram/visuals/data_mappings/debate_issue_counts_s3.yml \
  --config instagram/visuals/data_mappings/member_party_counts_s3.yml \
  --output generated_visual_data/s3_schema_profile.json
```

The schema profile uses S3 range reads and records:

```text
column names
sample row coverage
non-empty and blank counts for sampled rows
example values
top sampled values
likely numeric columns
mapping candidate matches
```

Build the combined smoke contact sheet locally after generating S3 smoke renders:

```text
process/instagram_build_s3_smoke_contact_sheet.py \
  --input-root generated_visuals/s3_smoke \
  --output generated_visuals/s3_smoke/contact_sheet.png
```

These smoke previews are not approved fixture contact sheets. They are live-data plumbing previews only.

The S3 smoke step is non-blocking. Fixture previews and contact sheets remain deterministic even if S3 is unavailable.

## Planned visual sequence

- inspect real S3 schema output from `generated_visual_data/s3_schema_profile.json`, `generated_visual_data/s3_smoke_status.json`, and mapping manifests
- add more mapped S3 samples for approved visuals
- real sourced image lookup/download workflow, gated by attribution and license review
- final approval pass to remove `draft` from visual IDs

## Preview outputs

```text
branch: instagram-preview-output
preview/visuals/png/
preview/visuals/metadata/
preview/visuals/manifests/
preview/visuals/tests/<visual_id>/png/<case_id>.png
preview/visuals/tests/<visual_id>/metadata/<case_id>.json
preview/visuals/tests/<visual_id>/manifests/<case_id>.render_manifest.json
preview/visuals/tests/<visual_id>/contact_sheet.png
preview/visuals/tests/<visual_id>/test_pack_manifest.json
preview/visuals/smoke/s3/
```

This system is review-only. It does not publish, schedule, or approve Instagram posts.
