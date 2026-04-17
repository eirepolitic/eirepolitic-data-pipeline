# Option 5: Experimental LLM Image-Generation-Assisted Visual Test

This is the repo-side scaffold for **pipeline option 5**.

Scope:
- visuals only
- no caption generation
- no social copy
- experimental evaluation workflow
- not production-ready

## Why the first test target is a constituency cover

The safest first experiment is a **constituency cover slide** with:
- one large exact constituency name
- no charts
- no dense factual labels
- no dependence on generated small-text rendering

The generated image is used as the **decorative background layer under test**.

Exact visible title text remains deterministic during render review.

That makes failures easier to spot and keeps source truth separate from generated appearance.

## What this scaffold builds

- deterministic prep from existing constituency data
- prompt-spec generation jobs
- repeatable naming conventions
- image generation script
- render-spec generation for deterministic overlay review
- review sheet output
- GitHub Actions workflow for manual test runs

## Existing repo pieces reused

This test reuses the existing Instagram HTML/Playwright renderer instead of replacing it.

That means:
- the generated layer is isolated
- exact text can still be overlaid consistently
- comparison against the deterministic pipeline remains straightforward

## Workflow shape

1. Prepare a run folder for one constituency.
2. Build prompt jobs for two candidate style directions:
   - `map_poster`
   - `textured_editorial`
3. Generate one or more image variants.
4. Create render specs that point the renderer at the generated background file.
5. Render deterministic cover-slide PNGs.
6. Review outputs in the generated review CSV.

## Output layout

```text
generated_visual_tests/option5_constituency_cover/<constituency>__<timestamp>/
  inputs/
    source_snapshot.json
    base_render_spec.yml
  jobs/
    generation_jobs.jsonl
    generation_jobs.pretty.json
  images/
    <record>.png
  metadata/
    <record>.json
    generated_manifest.jsonl
    generated_manifest.csv
  render_specs/
    <record>.yml
  rendered_posts/
    <record>/
      html/
      png/
      post_context.json
  review/
    review_sheet.csv
```

## Manual workflow inputs

Workflow: **Generate Instagram Option 5 Constituency Cover AI Test (Manual)**

Inputs:
- `constituency`
- `variant_count`
- `model`
- `style_mode`
- `spec_path`

## Review criteria

Use the generated review sheet to score each output on:
- brand consistency
- factual correctness of visible text
- text legibility
- repeatability
- usefulness versus deterministic templating

## Important risk rule

Do **not** trust the generated visual because it looks plausible.

The sidecar metadata and deterministic overlay are there so the visual can be judged against source truth, not by appearance alone.
