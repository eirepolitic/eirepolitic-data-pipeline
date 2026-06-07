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

## Planned visual sequence

- vertical bar chart
- line chart
- stacked bar chart
- ranking table
- choropleth map
- point map
- table/card visual
- small multiples
- area chart
- scatter plot
- dot plot
- lollipop chart
- slope chart
- donut/pie chart where useful
- constituency/geographic map variants
- sourced internet image assets with attribution metadata

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
```

This system is review-only. It does not publish, schedule, or approve Instagram posts.
