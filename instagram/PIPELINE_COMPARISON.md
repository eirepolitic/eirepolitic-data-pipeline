# Instagram infographic pipeline comparison

## Purpose of this document

This file compares the four Instagram visual pipeline versions currently built across the repo so they can be reviewed side by side and a preferred production direction can be chosen.

This comparison is intentionally practical.

It covers:
- what each version is
- where it lives
- methodology
- dependencies
- data reuse
- operational tradeoffs
- test status currently known from this review session
- output quality notes currently known from this review session
- recommended next decision

---

## Scope and decision framing

Original target constraints for the replacement pipeline were:
- visuals only
- no caption generation
- no post text writing
- no image generation model for the preferred production path
- no Power BI dependency
- direct rendering of infographic image slides from structured data
- reliability and repeatability first

Because of that, not all four versions are equally aligned to the intended production direction.

Some versions are best understood as:
- production candidates
- comparison baselines
- external-service experiments
- exploratory visual R&D

---

## The four compared versions

### Version 1: local HTML/CSS renderer

**Branch / reference**
- `main`
- same renderer also appears in `gpt/process-instagram_render_post.py-f79a0488`

**Core files**
- `process/instagram_render_post.py`
- `instagram/templates/*`
- `instagram/specs/constituency_test_post.yml`
- `.github/workflows/instagram_constituency_test.yml`

**Methodology**
- builds post context from S3-backed structured datasets
- renders slides as HTML using Jinja2 templates
- converts HTML slides to PNG using Playwright screenshots

**Rendering stack**
- Python orchestrator
- Jinja2
- HTML/CSS templates
- Playwright / Chromium for PNG export

**Output style**
- deterministic slide structure
- browser-rendered text and layout
- easier to mock and iterate visually than pure Pillow

**Strengths**
- already had a working end-to-end run path
- strong separation between data prep and presentation templates
- relatively quick to tweak layout if comfortable with HTML/CSS
- good as a mock or reference renderer

**Weaknesses**
- not Python-first rendering
- depends on browser automation at render time
- introduces layout/runtime variability through browser rendering
- not the closest match to the target requirement of a fully Python deterministic renderer

**Fit to intended production direction**
- medium
- useful baseline, but not ideal as the final long-term architecture

---

### Version 2: external template API pipeline

**Branch / reference**
- `feature/bannerbear-instagram-template-test`

**Core files**
- `process/instagram_template_pipeline.py`
- `instagram/specs/bannerbear_constituency_test.yml`
- `instagram/specs/placid_constituency_test.yml`
- `instagram/mappings/bannerbear_constituency_basic.yml`
- `instagram/mappings/placid_constituency_basic.yml`
- `.github/workflows/instagram_template_test.yml`

**Methodology**
- reuses the same S3-backed post context builder as Version 1
- maps context fields into provider-specific placeholder or layer definitions
- sends per-slide payloads to Bannerbear or Placid
- can fall back to the local HTML renderer when provider credentials or template IDs are missing

**Rendering stack**
- Python orchestrator
- Bannerbear and/or Placid API templates
- YAML mappings for placeholder binding
- local HTML fallback renderer for mock output

**Output style**
- external template service renders the slides
- slide content remains data-bound, but template composition lives outside the repo

**Strengths**
- explicit field mapping
- easier for a designer to fine-tune visually inside a template platform
- strong audit trail through saved request and response payloads
- good if the team wants to separate data logic from template design

**Weaknesses**
- depends on third-party rendering platforms
- requires manual creation and upkeep of templates
- requires secrets and template IDs
- final determinism depends partly on external provider behavior and template state
- chart-heavy layouts become more awkward unless simplified into text summaries or static layer tricks

**Fit to intended production direction**
- medium to low
- operationally viable, but less aligned with the desire for an internal code-based replacement

---

### Version 3: Option 5 AI visual generation test

**Branch / reference**
- `feature/option5-ai-visual-test`

**Core files**
- `process/instagram_option5_prepare_constituency_cover_test.py`
- `process/instagram_option5_generate_images.py`
- `process/instagram_option5_build_review_sheet.py`
- `instagram/OPTION5_LLM_IMAGE_TEST.md`
- `instagram/specs/constituency_cover_ai_test.yml`
- `.github/workflows/instagram_option5_constituency_cover_ai.yml`

**Methodology**
- prepares a deterministic test package for a constituency cover slide
- generates image prompts for one or more style directions
- uses an image model to generate background artwork variants
- writes sidecar metadata and review sheets
- then reuses the deterministic renderer to overlay exact visible text for review

**Rendering stack**
- Python orchestration
- OpenAI image generation
- metadata manifests and review sheet generation
- deterministic overlay render for the visible title layer

**Output style**
- hybrid
- generated background art plus deterministic text overlay

**Strengths**
- useful for exploratory cover art and style-direction testing
- separates decorative generated layer from exact visible truth text
- includes a review-sheet workflow, which is good process discipline
- good for R&D and creative comparison

**Weaknesses**
- directly conflicts with the preferred production constraint of no image generation model
- not reliable enough for factual infographic production
- repeatability is inherently weaker than deterministic rendering
- best limited to decorative use cases, not trusted fact-heavy slides

**Fit to intended production direction**
- low
- good experiment, poor primary production candidate for this project brief

---

### Version 4: fully Python deterministic renderer

**Branch / reference**
- `feature/instagram-infographic-test-pipeline`

**Core files**
- `process/instagram_render_post.py`
- `instagram/renderer/constants.py`
- `instagram/renderer/data_loader.py`
- `instagram/renderer/context.py`
- `instagram/renderer/fonts.py`
- `instagram/renderer/charts.py`
- `instagram/renderer/slides.py`
- `instagram/renderer/render.py`
- `instagram/specs/constituency_test_post.yml`
- `.github/workflows/instagram_constituency_test.yml`
- `tests/test_instagram_renderer.py`
- `tests/fixtures/instagram/*`

**Methodology**
- builds post context directly from structured data
- loads images from local paths or URLs
- composes slides with Pillow
- renders charts with matplotlib
- writes deterministic PNG outputs directly from Python

**Rendering stack**
- Python orchestrator
- Pillow for composition and typography
- matplotlib for chart slides
- cairosvg for SVG conversion when needed
- no browser renderer

**Output style**
- deterministic image generation
- explicit layout primitives in code
- reusable slide builder modules

**Strengths**
- best match for the intended production brief
- avoids browser rendering and third-party template vendors
- highly testable in pure Python
- deterministic filenames and structure
- good foundation for future reusable slide types

**Weaknesses**
- harder to fine-tune visually than HTML/CSS for some collaborators
- typography/layout utilities must be built and maintained in code
- current real-data issue-mapping bug still needs fixing
- some layout polish remains unfinished

**Fit to intended production direction**
- high
- strongest production candidate from the four versions

---

## Common data reused across the versions

The pipelines were designed to reuse existing repo outputs rather than introduce new primary source tables.

Core reused datasets include:
- members table
- member summaries table
- member photo URL table
- constituency images index
- debate issue classifications

Representative existing tables already in the repo ecosystem:
- constituency images index: `processed/constituencies/constituency_images.csv`
- debate issue classifier output: `processed/debates/debate_speeches_classified.csv`
- member summaries: `processed/members/members_summaries.csv`
- member photo URLs: `processed/members/members_photo_urls.csv`

This is important because the comparison is mainly about rendering methodology, not about creating new source data pipelines.

---

## Methodology comparison table

| Version | Main methodology | Determinism | External dependency level | Design flexibility | Best use |
|---|---|---:|---:|---:|---|
| V1 | HTML/CSS + Playwright | Medium | Medium | High | local mock baseline |
| V2 | Bannerbear / Placid templates | Medium | High | High | external-template workflow |
| V3 | image generation + deterministic overlay | Low | High | Medium to High | exploratory visual R&D |
| V4 | Pillow + matplotlib | High | Low | Medium | preferred production path |

---

## Operational comparison

### Version 1

**Operational requirements**
- Python dependencies
- Playwright / Chromium install
- AWS access to source data

**Failure surface**
- browser install/runtime issues
- HTML/CSS layout regressions
- data-binding bugs

### Version 2

**Operational requirements**
- AWS access
- Bannerbear and/or Placid credentials
- maintained external templates
- layer and placeholder names must stay in sync with mappings
- Playwright still required for local fallback path

**Failure surface**
- missing API keys
- wrong template IDs
- drift between repo mapping and template layers
- provider-side failures or rate limits

### Version 3

**Operational requirements**
- AWS access
- OpenAI API key
- image model usage cost
- deterministic overlay renderer still required for review output

**Failure surface**
- generated outputs may vary unpredictably
- visual plausibility may hide factual weakness
- review overhead is mandatory

### Version 4

**Operational requirements**
- Python dependencies only
- AWS access
- system fonts or sensible fallback fonts

**Failure surface**
- image URL fetches
- SVG conversion
- layout code bugs
- data-binding bugs

---

## Test summary currently known from this review session

This section only records what could be confirmed directly during this review session.

If you ran additional tests outside this chat, those are not automatically visible here and should be added later if you want this document to become the full historical log.

### Version 1 test status currently known

Workflow:
- `Generate Instagram Constituency Test Post (Manual)`

Confirmed runs visible from the repo:
- run `24524677835` on `main` -> **failed** at `Render constituency test post`
- run `24526208994` on `main` -> **success**

Artifact visible for the successful run:
- artifact `instagram-constituency-test`
- size `665,867 bytes`

What is currently known:
- the HTML/Playwright pipeline did achieve a successful end-to-end GitHub Actions run
- the successful artifact contents were not visually reviewed in this session
- the cause of the earlier failed run was not recoverable from the tooling used in this review

### Version 2 test status currently known

Workflow file exists:
- `Generate Instagram Template Test Post (Manual)`

What is currently known:
- the code and workflow scaffolding exist for Bannerbear, Placid, and fallback local HTML paths
- in this review session, no provider-rendered output artifact was visually inspected
- active workflow-run metadata for this branch-specific workflow was not directly retrievable through the currently available repo actions index

Practical interpretation:
- this version is built enough to compare architecturally
- it is not yet sufficiently verified in this review session to score visual quality confidently

### Version 3 test status currently known

Workflow file exists:
- `Generate Instagram Option 5 Constituency Cover AI Test (Manual)`

What is currently known:
- the prepare / generate / review / deterministic-overlay workflow is scaffolded
- in this review session, no output artifact or rendered images from this branch were visually inspected

Practical interpretation:
- methodology is documented and the workflow exists
- visual quality and repeatability are still unverified here

### Version 4 test status currently known

Workflow:
- `Generate Instagram Constituency Test Post (Manual)`

Confirmed local and CI status:
- local fixture smoke test was added and passes as part of the build work
- GitHub Actions run `24545728450` on `feature/instagram-infographic-test-pipeline` -> **success**

Artifact visible for the successful run:
- artifact `instagram-constituency-test`
- size `487,009 bytes`

Visual review completed in this session:
- 5 PNG slides were reviewed directly in chat

---

## Output review notes currently known

### Version 1 output quality currently known

Not enough evidence captured in this session for a full visual-quality judgment.

Known minimum conclusion:
- it can run successfully end to end
- it remains a valid baseline renderer for comparison

### Version 2 output quality currently known

Not enough evidence captured in this session for a full visual-quality judgment.

What can be said from the methodology alone:
- likely strongest when a human-designed external template is already mature
- likely weakest when fast-changing slide requirements must be encoded through many placeholder mappings

### Version 3 output quality currently known

Not enough evidence captured in this session for a full visual-quality judgment.

What can be said from the methodology alone:
- likely useful for cover-art exploration
- poor fit for text-heavy factual slides
- should only be trusted with deterministic overlay and explicit review

### Version 4 output quality currently known

This is the only version whose rendered PNGs were directly reviewed in this session.

#### What looked good
- pipeline worked end to end
- image dimensions looked correct for Instagram portrait format
- visual style was coherent across slides
- the overview slide and TD profile slide were the strongest outputs
- member photo crop looked good
- glossary slide structure was clear and readable

#### What looked bad
- issue data did not populate correctly on the chart slides
- overview slide showed `Issue-labelled speeches: 0`
- top issue fallback text was appearing instead of real issue counts
- both issue chart slides rendered empty-state charts
- chart intro text overflowed the right edge
- footer source text overlapped and was not readable
- empty-state chart message looked like a debug fallback rather than finished design
- long fallback phrases such as `No classified issue yet` wrapped awkwardly
- constituency image treatment worked, but used panel space inefficiently

#### Most likely current root cause of the data problem
- the renderer’s issue-column detection is too narrow for the real classified speeches table
- speaker-name matching may also be too brittle

#### Immediate implication
- the framework is working
- the main blocker is data mapping, not the core rendering architecture

---

## Visual review detail for Version 4

### Slide-by-slide notes

#### Overview slide

**Good**
- title hierarchy was strong
- constituency image loaded
- metric card structure was clear

**Bad**
- issue-labelled speech count was zero
- top issue fallback text was not production-ready
- the map/image panel could use better fill or better composition
- footer was broken

#### TD profile slide

**Good**
- strongest slide overall
- member photo displayed well
- general panel layout worked
- background summary block felt useful and close to production direction

**Bad**
- top issue fallback text wrapped poorly
- issue-labelled speech count was zero
- footer was broken

#### Top issues slide

**Good**
- slide framing and chart area are present
- chart panel size is usable

**Bad**
- no issue data appeared
- intro text overflowed
- empty-state message styling was weak
- footer was broken

#### Selected TD issue profile slide

**Good**
- same structural positives as the constituency issue slide

**Bad**
- same issue-data failure
- same intro text overflow
- same empty-state weakness
- footer was broken

#### Methodology / glossary slide

**Good**
- content boxes were clean and readable
- easiest slide to understand immediately

**Bad**
- footer was broken
- still needs final polish, but less urgent than data-binding issues

---

## Best candidate by use case

### Best production candidate under the original brief
- **Version 4**

Why:
- fully Python-based
- deterministic
- no browser dependency
- no image generation model required
- no external template vendor required
- easiest to keep close to source truth in code

### Best baseline / mock renderer
- **Version 1**

Why:
- fastest to iterate visually if HTML/CSS is acceptable
- useful as a comparison reference even if not chosen for final production

### Best external-service experiment
- **Version 2**

Why:
- strongest if visual design is primarily owned in Bannerbear or Placid
- weaker if long-term control and internal reproducibility are priorities

### Best exploratory visual R&D path
- **Version 3**

Why:
- useful for testing decorative cover art directions
- not suitable as the main factual infographic system

---

## Recommended decision at this stage

### Recommendation

Choose **Version 4** as the primary path to production.

### Reason

It is the best fit to the original goals:
- deterministic
- code-based
- structured-data-first
- reliable reruns
- no Power BI
- no image model dependence
- no template-vendor dependence

### Recommendation on the other three versions

- keep **Version 1** as a reference baseline for quick visual comparison
- keep **Version 2** only if there is serious interest in an external design-tool workflow
- keep **Version 3** as an experimental branch only, not as the main delivery path

---

## Highest-priority fixes if Version 4 is chosen

1. fix debate issue data binding
   - broaden issue-column detection
   - inspect actual speech-classification columns
   - strengthen speaker-to-member matching

2. simplify and shorten footer rendering
   - remove long file names from visible slide artwork
   - use a concise source label instead

3. wrap or shorten chart intro copy

4. improve empty-state design
   - use themed copy and styling
   - avoid black debug-looking fallback text

5. improve long-value typography handling
   - dynamic font-size reduction or better fallback wording

6. refine constituency image treatment on the overview slide

---

## Open questions still worth answering later

- do you want to preserve a browser-rendered baseline for visual parity comparisons?
- is there any real organizational reason to prefer Bannerbear or Placid over an internal renderer?
- should AI-generated cover art remain part of the roadmap as a separate experimental track?
- what exact constituency and TD should become the canonical golden test case for regression testing?

---

## Bottom line

If the goal is to decide on the best long-term approach for an automatable, repeatable, structured-data Instagram infographic pipeline, the current evidence points to:

**Version 4 as the best core architecture.**

It already runs end to end and the remaining problems are mostly fixable implementation details, especially data binding and layout polish.

The other versions are still useful, but mainly as:
- a baseline reference
- an external-template experiment
- an AI-visual experiment

not as the best primary production path.
