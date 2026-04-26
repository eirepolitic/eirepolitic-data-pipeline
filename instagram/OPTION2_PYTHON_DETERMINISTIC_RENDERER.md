# Option 2: Python deterministic Instagram infographic renderer

## Purpose of this document

This document records only the pipeline version built in this chat.

It is intended to be a complete working note for later comparison against other pipeline versions, but it does **not** compare them. It only documents this version on its own terms.

This version is the **fully Python-based deterministic renderer** built on branch:

- `feature/instagram-infographic-test-pipeline`

Its goal is to replace a manual Power BI-based post creation process with a more automatable code-driven rendering pipeline.

---

## Summary

This pipeline renders Instagram carousel slides directly from structured data using a deterministic Python rendering stack.

It does **not** generate captions.
It does **not** generate social copy.
It does **not** use an image generation model.
It does **not** rely on Power BI.
It does **not** rely on HTML/CSS or browser rendering in the current version.

The first implemented test target is a **constituency carousel**.

The initial slide set is:
- constituency overview
- TD profile card
- constituency top issues chart
- selected TD top issues chart
- methodology / glossary slide

The current architecture is production-oriented in direction, but not production-ready yet. It successfully runs end to end in GitHub Actions and produces PNG outputs, but there are still known data-mapping and layout issues.

---

## High-level design goals used for this version

This version was designed around the following priorities:

- reliability over cleverness
- deterministic rendering over generative rendering
- reusable slide builders instead of one giant script
- direct rendering from structured source data
- explicit dimensions and explicit output paths
- clear failure when required source data is missing
- local testability without AWS as well as S3-backed real runs

---

## Current repo location

### Main branch used for this work
- `feature/instagram-infographic-test-pipeline`

### Main entry point
- `process/instagram_render_post.py`

### Python renderer package
- `instagram/renderer/__init__.py`
- `instagram/renderer/constants.py`
- `instagram/renderer/data_loader.py`
- `instagram/renderer/context.py`
- `instagram/renderer/fonts.py`
- `instagram/renderer/charts.py`
- `instagram/renderer/slides.py`
- `instagram/renderer/render.py`

### Spec and docs
- `instagram/specs/constituency_test_post.yml`
- `instagram/README.md`

### Workflow
- `.github/workflows/instagram_constituency_test.yml`

### Tests and fixtures
- `tests/test_instagram_renderer.py`
- `tests/fixtures/instagram/members.csv`
- `tests/fixtures/instagram/member_summaries.csv`
- `tests/fixtures/instagram/member_photos.csv`
- `tests/fixtures/instagram/debate_issues.csv`
- `tests/fixtures/instagram/constituency_images.csv`

---

## What changed in this version

This version replaced the earlier HTML/Jinja/Playwright rendering path with a Python-first rendering stack.

### Main architectural changes

1. The Instagram entry point was changed so that:
   - `process/instagram_render_post.py` now calls the Python renderer package
   - rendering is done directly in Python instead of through HTML templates

2. A new renderer package was added under `instagram/renderer/`.

3. The old HTML template files were removed from this branch after confirmation that this branch should be Python-only.

4. The workflow was updated so it no longer installs or depends on Playwright.

5. The requirements file was updated to support:
   - Pillow
   - matplotlib
   - cairosvg

6. A local fixture-based smoke test was added so the renderer can be tested without AWS or live S3 data.

---

## Core methodology

### Step 1: load a post spec

The renderer starts from a YAML post spec.

Current test spec:
- `instagram/specs/constituency_test_post.yml`

The spec defines:
- post slug
- output root
- slide dimensions
- theme metadata
- constituency under test
- optional TD override
- slide order
- enabled slides
- glossary text
- placeholder metrics
- branding palette

### Step 2: load structured datasets

The renderer loads the first available CSV for each required or optional dataset.

The current dataset candidates in code are:
- members table
- member summaries table
- member photo URLs table
- debate issue classifications table
- constituency images table

These reuse the existing repo’s S3 data products rather than introducing new source-of-truth tables.

Relevant reused data products already exist in the repo ecosystem for constituency images, issue classifications, member summaries, and member photo URLs. fileciteturn2file1 fileciteturn2file2 fileciteturn2file3 fileciteturn2file4

### Step 3: build a deterministic post context

The renderer constructs a single context object that includes:
- post metadata
- branding and palette
- constituency information
- selected member information
- issue counts
- glossary content
- datasets used

This includes member selection logic.

Current behavior:
- if a member is supplied in the spec, the renderer tries to use that member
- otherwise, it chooses the highest-speech-count member within the target constituency

### Step 4: render slides with dedicated Python builders

Each slide type is rendered by a dedicated Python function rather than by a shared browser template.

Current slide renderers:
- `constituency_overview`
- `member_profile`
- `top_issues`
- `glossary`

### Step 5: save deterministic outputs

The renderer writes:
- PNG slides
- `post_context.json`
- `render_manifest.json`

Current output convention:

```text
generated_posts/<post_slug>/
  png/
    01_overview.png
    02_member_profile.png
    03_top_issues.png
    04_member_top_issues.png
    05_glossary.png
  post_context.json
  render_manifest.json
```

Filenames are deterministic from slide order and slide key.

---

## Rendering stack

### Main stack
- Python
- Pillow
- matplotlib
- cairosvg
- requests
- boto3
- pandas
- pyyaml

### What each component does

#### Pillow
Used for:
- canvas creation
- text placement
- panel drawing
- image compositing
- final PNG writing

#### matplotlib
Used for:
- deterministic chart image generation for issue-count slides

#### cairosvg
Used for:
- converting SVG image assets to raster images if needed

#### requests
Used for:
- fetching remote image URLs such as member photos or constituency images

#### boto3 and pandas
Used for:
- reading input CSVs from S3
- building context tables

---

## Package breakdown

### `instagram/renderer/constants.py`

Defines:
- default bucket and region
- dataset candidate locations
- local fixture filenames
- font candidate paths
- default palette
- default output size
- slide geometry defaults

This centralizes hard-coded defaults so they are not spread across the codebase.

### `instagram/renderer/fonts.py`

Handles font resolution and fallback.

Current policy:
- try DejaVu Sans
- fall back to Liberation Sans
- fall back to PIL default if system fonts are unavailable

This was chosen to avoid storing font files in the repo while keeping GitHub Actions compatibility.

### `instagram/renderer/util.py`

Contains shared helpers for:
- name normalization
- constituency normalization
- missing-value handling
- wrapped text drawing
- rounded panels
- image loading from URLs or local paths
- image fit/cover helpers
- rank and percent formatting

### `instagram/renderer/data_loader.py`

Implements two loader modes:

#### S3 loader
For real runs against repo datasets in S3.

#### Local loader
For fixture-based local tests.

This is a key design decision because it allows real-data rendering and isolated smoke tests without changing the main rendering code.

### `instagram/renderer/context.py`

Builds the post context from loaded datasets.

Current responsibilities:
- validate members table structure
- merge member summaries and photos onto the member table
- select the target constituency
- pick the target member
- derive constituency issue counts
- derive member issue counts
- prepare glossary and metrics values

### `instagram/renderer/charts.py`

Builds chart images using matplotlib.

Current chart type:
- horizontal bar chart

This is currently used for the issue slides.

### `instagram/renderer/slides.py`

Contains the slide renderer functions and shared layout helpers.

Current responsibilities:
- create new slide canvas
- draw headers and footers
- render overview slide
- render member profile slide
- render top issues slide
- render glossary slide
- save PNGs and manifest file

### `instagram/renderer/render.py`

Main CLI runner for the renderer package.

Responsibilities:
- parse args
- load spec
- choose local or S3 loader
- build context
- write context JSON
- render all enabled slides

---

## Input data used by this version

This renderer reuses existing source tables from the wider repo data lake pattern.

### Required core input
- members table

### Optional but expected inputs
- member summaries
- member photo URLs
- debate issue classifications
- constituency images

### Current expected S3 paths in code
- `raw/members/oireachtas_members_34th_dail.csv`
- `processed/members/members_summaries.csv`
- `processed/members/member_photos/members_photo_urls.csv`
- `processed/members/members_photo_urls.csv`
- `processed/debates/debate_speeches_classified.csv`
- `processed/constituencies/constituency_images.csv`

These align with existing member summary, member image, constituency image, and debate issue pipelines in the repo. fileciteturn2file1 fileciteturn2file2 fileciteturn2file3 fileciteturn2file4

---

## Slide set currently implemented

### 1. Constituency overview

Purpose:
- establish the subject of the carousel
- show image or map
- show key headline metrics

Current content:
- slide title
- account name
- constituency image or fallback text
- constituency name
- TD count
- party count
- issue-labelled speech count
- top issue
- footer

### 2. TD profile card

Purpose:
- identify one representative TD from the constituency
- combine identity, photo, and short background information

Current content:
- slide title
- account name
- member photo or fallback text
- member name
- party
- constituency
- top issue
- issue-labelled speech count
- background summary
- footer

### 3. Constituency top issues chart

Purpose:
- show issue distribution using existing classified speech data

Current content:
- slide title
- account name
- explanatory note
- chart panel
- chart for constituency issue counts
- footer

### 4. Selected TD top issues chart

Purpose:
- show issue distribution for the selected TD only

Current content:
- same general structure as the constituency issue slide
- chart built from member issue counts

### 5. Methodology / glossary slide

Purpose:
- explain issue classification and placeholder metrics

Current content:
- three content boxes
- issues glossary
- vote participation placeholder glossary
- speech rank placeholder glossary
- footer

---

## CLI interface

Current CLI arguments:
- `--spec`
- `--constituency`
- `--member-name`
- `--output-dir`
- `--data-source`
- `--data-root`
- `--s3-bucket`
- `--aws-region`

### Real S3-backed run

```bash
python process/instagram_render_post.py \
  --spec instagram/specs/constituency_test_post.yml
```

### Local fixture run

```bash
python process/instagram_render_post.py \
  --spec instagram/specs/constituency_test_post.yml \
  --data-source local \
  --data-root tests/fixtures/instagram \
  --output-dir generated_posts_local
```

---

## Workflow details

### Workflow file
- `.github/workflows/instagram_constituency_test.yml`

### Workflow name
- `Generate Instagram Constituency Test Post (Manual)`

### Current workflow behavior
- manual dispatch
- runs on `ubuntu-latest`
- Python 3.11
- installs requirements
- runs the Python renderer
- uploads generated output as an artifact

### Workflow inputs
- `constituency`
- `member_name`
- `spec_path`

### Current default constituency in workflow
- `Wicklow-Wexford`

---

## Local test strategy added in this version

A local smoke test was added so the pipeline can be run without AWS access.

### Test file
- `tests/test_instagram_renderer.py`

### Test approach
- run the main renderer entry point as a subprocess
- point it at local fixture CSV files
- assert output PNG files exist
- assert expected fields appear in `post_context.json`

### Current fixtures
- members fixture
- member summaries fixture
- member photos fixture
- debate issues fixture
- constituency images fixture

### What this test gives
- verifies package wiring works
- verifies local-data path works
- verifies deterministic output structure
- reduces the chance of breaking the renderer while editing layout code

### What this test does not give
- it does not validate live S3 data behavior
- it does not guarantee layout quality
- it does not detect all real-data schema mismatches

---

## GitHub Actions test completed in this chat

A real workflow run was triggered and checked on branch:
- `feature/instagram-infographic-test-pipeline`

### Confirmed run
- workflow: `Generate Instagram Constituency Test Post (Manual)`
- run id: `24545728450`
- conclusion: **success**

### Confirmed successful steps
- checkout
- Python setup
- dependency install
- render constituency test post
- upload artifact

### Artifact created
- artifact name: `instagram-constituency-test`
- artifact size: `487,009 bytes`

This confirms the Python deterministic pipeline runs end to end in GitHub Actions using the real workflow path.

---

## Visual output review completed in this chat

The generated PNG slides from this version were visually reviewed in this chat.

### Files reviewed
- `01_overview.png`
- `02_member_profile.png`
- `03_top_issues.png`
- `04_member_top_issues.png`
- `05_glossary.png`

---

## What looked good in the reviewed outputs

### Overall
- the pipeline produced a complete five-slide output set
- slide dimensions looked appropriate for Instagram portrait format
- visual style was consistent across slides
- the dark green / light text palette was coherent
- general card structure was clear

### Overview slide
- title hierarchy was strong
- constituency image did load
- metric-card structure was understandable
- the slide felt recognizably like an infographic slide rather than a debug render

### TD profile slide
- strongest slide in the set
- member photo crop looked good
- layout balance between photo, identity, and summary worked reasonably well
- background panel was useful and readable
- this slide felt closest to a production-ready direction

### Glossary slide
- clean and readable block structure
- clear explanation layout
- easiest slide to understand immediately

---

## What looked bad in the reviewed outputs

### 1. Issue data did not populate correctly

This is the biggest current problem.

Observed symptoms:
- issue-labelled speech count showed `0`
- top issue fallback text appeared instead of real issue values
- both issue chart slides rendered with `No classified issue data available`

This means the chart slides are currently not using real classified issue counts successfully in the live run.

### 2. Footer rendering was broken

Observed symptoms:
- footer source text was too long
- footer text overlapped visually
- bottom of slide looked cluttered and unreadable

### 3. Chart intro line overflowed

Observed symptoms:
- chart intro text extended too far across the slide
- it ran into the right side visually

### 4. Empty-state chart styling looked unfinished

Observed symptoms:
- too much empty space
- fallback message felt like a debug state
- black text in the empty chart state clashed with the design language

### 5. Long fallback phrases wrapped badly

Observed symptoms:
- `No classified issue yet` wrapped awkwardly
- fallback text on overview and profile cards did not feel polished

### 6. Constituency image section on the overview slide was underused

Observed symptoms:
- image/map sat inside a large panel with too much unused space around it
- it worked technically, but not optimally as a design treatment

---

## Most likely current root cause of the issue-data problem

The issue-data failure appears more likely to be a **data mapping bug** than a renderer failure.

Most likely causes:
- the real classified debate table may use a different issue column name than the renderer currently expects
- the speaker-name matching may be too brittle
- issue values may need broader normalization

Current issue-column detection in this version is narrow.

Current likely mismatch risk:
- the live classified debate output may contain a field name not currently included in the renderer detection list

This is consistent with the fact that:
- the renderer itself completed successfully
- chart panels rendered
- but the data payload into those charts was effectively empty

---

## What is honest and complete versus placeholder in this version

### Already real
- constituency name
- selected member identity
- member photo if found
- member background if found
- constituency image if found
- slide dimensions
- output filenames
- slide structure
- glossary text from spec

### Currently placeholder or partially real
- vote participation text is placeholder unless manually supplied in spec
- speech rank text is placeholder unless manually supplied in spec
- constituency speech rank text is placeholder unless manually supplied in spec
- issue charts are intended to be real, but are currently failing in the live run because of the issue-data mapping problem

This distinction matters because the pipeline should not pretend unknown metrics are real.

---

## Why this version is still important despite the bugs

Even with the current data issue bug, this version proved several major things:

1. a fully Python deterministic renderer is viable in the repo
2. the end-to-end GitHub Actions workflow can run successfully
3. structured-data-driven slide composition is working
4. the modular slide-builder approach is workable
5. the remaining main problems are fixable implementation details rather than architectural blockers

In other words, the current status is:
- the architecture works
- the rendering framework works
- the test harness works
- the live issue-data binding needs correction
- some visual polish still needs work

---

## Current status assessment

### Current maturity
This version is best described as:
- **working prototype**
- **not yet production ready**
- **architecturally promising**

### What is already strong
- direction of architecture
- code organization
- repeatability
- workflow automation
- local testability

### What is not yet strong enough
- live issue data binding
- footer design
- empty-state quality
- some long-text handling
- overview image treatment

---

## Recommended next implementation steps for this version

### Highest priority

#### 1. Fix issue-data binding
- inspect actual live columns in `processed/debates/debate_speeches_classified.csv`
- broaden issue-column detection
- strengthen speaker-to-member matching
- rerun live workflow

#### 2. Simplify footer treatment
- remove long file names from visible slide artwork
- replace with concise source or account text

#### 3. Fix chart intro overflow
- shorten copy or wrap it in-theme

### Secondary priority

#### 4. Improve empty-state design
- themed fallback text
- better color handling
- clearer but less debug-like messaging

#### 5. Improve long-value typography
- dynamic font-size reduction or better fallback phrases

#### 6. Improve constituency image treatment
- enlarge image usage or redesign panel composition

---

## Files removed in this version

After the branch was confirmed to be Python-only, the old HTML/CSS Instagram renderer files were removed from this branch.

That means this branch is intentionally focused on the deterministic Python path only.

---

## Why this version matters as a decision candidate

This version is the clearest attempt in the repo to meet the original requirements directly:
- Python-first
- deterministic
- structured-data-driven
- reusable slide builders
- no browser renderer
- no external template vendor
- no image generation model

The fact that it already ran end to end in GitHub Actions makes it a serious candidate for continued development.

---

## Bottom line

This version successfully established a real deterministic Python infographic renderer inside the repo.

It currently does the following successfully:
- builds context from structured data
- renders a five-slide Instagram carousel as PNG files
- runs in GitHub Actions
- supports local fixture testing
- uses reusable slide builders rather than one-off scripts

Its main current problem is not that the renderer architecture failed.

Its main current problem is that the live issue-classification data is not yet being mapped into the chart slides correctly, and several presentation details still need polish.

So the most accurate summary is:

**working deterministic renderer foundation, with live data-mapping fixes and visual refinement still required.**
