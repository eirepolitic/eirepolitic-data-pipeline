# Instagram content creation factory plan

This is the canonical product, architecture, implementation, and handoff plan for the Eirepolitic Instagram content creation factory.

It replaces the original informal plan and should be read together with:

- `instagram/README.md`
- `instagram/ARCHITECTURE.md`
- `instagram/visuals/SYSTEM.md`
- `docs/oireachtas_unified_data_model_plan.md`

The intended user experience is conversational: a GPT with repository, GitHub, AWS, S3, workflow, and rendering access should be able to help define a post concept, test it, generate a full batch, return previews, accept feedback, selectively repair outputs, and preserve an auditable project record.

---

## 1. Desired end state

The completed factory should support this workflow:

1. Upstream scheduled data pipelines refresh CSV/Parquet datasets in S3 automatically.
2. Historical data is preserved and new weekly/monthly/yearly records are appended or replaced according to the table contract.
3. Optional enrichment pipelines, such as LLM speech-topic classification, run independently and publish validated outputs.
4. The user opens a chat with a GPT that has access to the repository, GitHub Actions, and AWS resources.
5. The user describes what a new post series should communicate.
6. The GPT and user agree:
   - intended audience and message
   - granularity (`member`, `constituency`, `party`, `county`, `issue`, `time_period`, etc.)
   - one-off versus recurring generation
   - slide sequence
   - post layouts
   - visuals
   - text and facts
   - data sources and time period
7. Existing approved post/visual types are selected from catalogues, or new draft types are created and added to the catalogues.
8. A machine-readable post-project specification is created and committed.
9. The system validates data availability, field mappings, granularity keys, joins, and expected output counts.
10. The system generates three pre-production test sets for every slide:
    - minimum-case version
    - maximum-case version
    - one real-data example
11. The maximum/minimum versions combine the largest/smallest relevant variable values for the selected granularity, including text length, image dimensions, visual item counts, and numeric values.
12. The real-data example is checked against source data and, when necessary, an external authoritative source.
13. The user reviews the pre-production renders and requests changes through chat.
14. Once approved, the system generates one complete post set for every item in the granularity.
15. Outputs are written to a dedicated S3 project prefix with one subfolder per granularity item and complete manifests.
16. The user manually reviews every generated post.
17. If all outputs are acceptable, the project is marked ready.
18. If all outputs need changes, the project is redesigned and regenerated.
19. If only selected outputs need changes, only those items/slides are regenerated; unaffected outputs remain unchanged.
20. Recurring projects can generate a new version after each period closes and data readiness checks pass.
21. Automatic posting, scheduling, and content suggestions may be added later as separately approved subsystems.

---

## 2. Scope

### Included in the current factory goal

- conversational post design
- post/visual catalogues
- machine-readable project specifications
- data-source and metric definitions
- min/max/real testing
- complete slide rendering
- batch generation by granularity
- S3 output storage
- manifests and provenance
- manual review
- targeted regeneration
- recurring period-based generation
- GitHub Actions review workflows
- direct preview links returned in chat

### Explicitly out of scope for the first complete version

- automatic Instagram publishing
- automatic scheduling to social platforms
- automatic approval
- fully autonomous content suggestions
- unreviewed sourced-image acquisition
- unreviewed factual claims
- removal of draft status without explicit approval

---

## 3. Current implementation status

### 3.1 Unified data platform — built and active

The repository now contains a unified Oireachtas data model with:

- Bronze/Silver/Gold layers
- compatibility outputs
- production batch pointers
- historical batch storage
- downstream contracts
- current and historical tables
- scheduled or workflow-driven refresh pipelines
- support for enriched outputs such as classified speeches

The Instagram visual layer can now resolve logical unified-model keys through the production pointer and follow the active production batch without hard-coded physical batch paths.

Validated unified-model consumers:

- debate issue counts from the debate compatibility dataset using `PoliticalIssues`
- member party counts from the member compatibility dataset using `party`

### 3.2 Standalone visual generation — substantially built

Implemented under `instagram/visuals/`:

- YAML visual templates
- YAML sample bindings
- Python renderer registry
- deterministic fixtures
- stress-test registries
- metadata JSON
- render manifests
- contact sheets
- S3/local/inline input modes
- data preparation mappings
- schema profiling
- mapping-readiness validation
- privacy-safe diagnostics
- preview publication

Current draft renderer families:

1. horizontal bar
2. vertical bar
3. line chart
4. area chart
5. stacked bar
6. ranking table
7. donut chart
8. scatter plot
9. dot plot
10. lollipop chart
11. slope chart
12. table card
13. small multiples
14. point map
15. choropleth map
16. tile map
17. sourced-image asset placeholder

All remain draft until explicit visual approval.

### 3.3 Visual QA — built and validated

For every visual type the system supports:

- minimum/normal/maximum/overflow item counts
- long-label stress cases
- small/large/tight/wide value ranges
- zero/negative/missing cases where relevant
- phone-readability cases
- one PNG per test case
- one contact sheet per visual type
- metadata and manifests
- preview branch publication

This testing is visual-level only. Complete-slide min/max/real testing remains to be built.

### 3.4 Live S3 smoke tests — built and validated

Canonical live smoke suite:

- debate issues → horizontal bar
- debate issues → vertical bar
- member parties → donut
- member parties → horizontal bar

Validated capabilities:

- production pointer resolution
- logical `compat`/`latest` keys
- schema range profiling
- privacy-safe schema summaries
- mapping-readiness checks
- full mapped CSV rendering
- S3 smoke contact sheet
- preview output publication

### 3.5 Data mapping layer — partially built

Current transforms:

- `count_by`
- `sum_by`

Current row filters:

- `equals`
- `latest_value`

The mapping layer can normalize raw/unified datasets into renderer-friendly CSVs and produce transformation manifests.

Still needed:

- `average_by`
- `percentage_by`
- `distinct_count_by`
- `rank_by`
- `top_n`
- `bottom_n`
- `between_dates`
- `latest_complete_period`
- multi-column grouping
- joins
- pivots/unpivots
- percentage-of-total
- period-over-period change
- rolling windows
- per-capita/normalized metrics where appropriate

### 3.6 Post layout renderer — built but not integrated into the new orchestrator

Existing components:

- JSON post layouts under `instagram/templates/layouts/`
- palettes
- deterministic post rendering
- campaign specs
- full-slide text/media placement
- member/constituency context loaders

Existing layouts include:

- `big_media_title_v1`
- `profile_card_main_v1`
- `profile_card_v1`
- `title_text_media_v1`

Important boundary:

- standalone visual layer creates visual-only PNGs
- post layout layer creates complete slides with text, branding, source/footer text, overlays, and media placement

Still needed:

- formal integration between visual assets and post-layout slots
- approved post-type catalogue
- required/optional variable definitions per layout
- complete-slide stress generation
- complete-post project orchestration

### 3.7 Campaign/batch structure — partially built

Existing campaign folder pattern:

```text
instagram/campaigns/<campaign_id>/
  campaign_brief.md
  data_sources.md
  decisions.md
  media_plan.yml
  render_spec.yml
  fixtures/
  progress_log.md
  review_notes.md
```

Existing campaign renderer can batch some post layouts, but it does not yet provide the full conversational factory described here.

### 3.8 Review workflows — built for visuals, partial for complete posts

Existing review capabilities:

- manual GitHub Actions dispatch
- preview output branch
- direct raw PNG links
- uploaded artifacts
- contact sheets
- iterative edit/rerun/review loop
- S3 smoke diagnostics

Missing:

- post-project-level review dashboard/index
- per-item/per-slide review status
- targeted regeneration workflow
- S3 project output structure
- immutable approved-output tracking

### 3.9 Documentation — built

Canonical documentation:

- `instagram/README.md`
- `instagram/ARCHITECTURE.md`
- `instagram/visuals/README.md`
- `instagram/visuals/SYSTEM.md`
- this plan

---

## 4. Core concepts for the completed factory

### 4.1 Post project

A post project defines one repeatable post series.

Examples:

- one five-slide carousel per TD
- one three-slide post per constituency
- one ranking post per party
- one monthly national post
- one annual constituency activity series

Every project must have a stable `project_id` and version.

### 4.2 Granularity

Granularity is the item that receives one complete generated post set.

Supported planned grains:

- `member`
- `constituency`
- `party`
- `county`
- `committee`
- `issue`
- `debate`
- `vote`
- `time_period`
- `national`
- custom composite grain

A project must define:

- grain name
- unique key field(s)
- display-label field
- optional selector/filter
- expected item count
- stable ordering

### 4.3 Slide definition

Each slide definition must specify:

- slide order
- post layout ID
- text variables
- optional visual definition
- image/media definition
- source/footer rules
- accessibility/alt-text inputs
- required/optional fields
- fallback behavior

### 4.4 Visual definition

A visual definition must specify:

- visual template ID
- metric/data mapping ID
- bindings
- filters
- grouping
- sort order
- item limits
- time period
- attribution
- empty-data behavior

### 4.5 Test scenario

Every slide must have:

- `minimum`
- `maximum`
- `real_example`

Optional additional scenarios:

- missing image
- missing optional text
- zero values
- tied rankings
- overflow item count
- unusual characters
- negative values
- incomplete period

### 4.6 Generation run

A generation run is an immutable record containing:

- project/version
- source data batch ID
- production pointer state
- selected period
- generation timestamp
- Git commit SHA
- workflow run ID
- item list
- slide outputs
- hashes
- warnings/errors
- review state

---

## 5. Required new component: post/visual catalogues

The user should be able to select from menus rather than remember file names.

### 5.1 Post type catalogue

Proposed path:

```text
instagram/catalogues/post_types.yml
```

Each entry should define:

- `post_type_id`
- display name
- description
- status (`draft`, `approved`, `deprecated`)
- layout path
- supported aspect ratios
- supported slot types
- required variables
- optional variables
- text-length limits
- visual/media slot dimensions
- fallback rules
- example preview link
- version

### 5.2 Visual type catalogue

Proposed path:

```text
instagram/catalogues/visual_types.yml
```

Each entry should define:

- `visual_type_id`
- display name
- renderer
- template path
- status
- intended use
- required bindings
- optional bindings
- supported data shapes
- recommended min/max item counts
- slot dimensions/aspect ratios
- known limitations
- test contact-sheet link
- version

### 5.3 Metric/data product catalogue

Proposed path:

```text
instagram/catalogues/metrics.yml
```

Each metric should define:

- `metric_id`
- display name
- description
- source logical key/table
- layer (`compat`, `silver`, `gold`)
- grain
- measure field(s)
- dimension fields
- period field(s)
- mapping config
- freshness expectation
- source note
- validation rules
- privacy classification
- external verification requirements

### 5.4 Catalogue validation

A new validator should confirm:

- referenced files exist
- IDs are unique
- required fields are present
- approved entries have previews/tests
- post slots and visual dimensions are compatible
- deprecated IDs are not used by new projects

---

## 6. Required new component: post project specification

Proposed path:

```text
instagram/projects/<project_id>/project.yml
```

Example structure:

```yaml
project_id: constituency_speech_activity_v1
version: 1
status: draft
purpose: Compare recent parliamentary speech activity within each constituency.

granularity:
  grain: constituency
  key_fields: [constituency_code]
  label_field: constituency_name
  source_metric: constituency_activity_yearly
  selector:
    mode: all
  ordering:
    field: constituency_name
    direction: ascending

period:
  mode: latest_complete_year
  field: year

slides:
  - slide_id: cover
    order: 1
    post_type_id: big_media_title_v1
    text:
      title: "{{ constituency_name }}"
      subtitle: "Parliamentary activity in {{ period_label }}"
    media:
      type: sourced_or_generated_image
      binding: constituency_image

  - slide_id: speech_ranking
    order: 2
    post_type_id: title_text_media_v1
    text:
      title: "Who spoke most?"
      body: "Speech contributions by TDs representing {{ constituency_name }}."
    visual:
      visual_type_id: ranking_table_draft_v1
      metric_id: member_speech_count_by_constituency
      filters:
        constituency_code: "{{ constituency_code }}"
        year: "{{ selected_period }}"

validation:
  scenarios: [minimum, maximum, real_example]
  real_example_selector:
    mode: median_complexity

output:
  s3_prefix: processed/instagram_factory/projects/constituency_speech_activity_v1
  preview_branch: instagram-preview-output

review:
  require_all_items_reviewed: true
  allow_targeted_regeneration: true

schedule:
  enabled: false
```

### Required project files

```text
instagram/projects/<project_id>/
  project.yml
  brief.md
  decisions.md
  data_sources.md
  validation_plan.yml
  review_notes.md
  progress_log.md
```

---

## 7. Required new component: project orchestrator

Proposed CLI:

```text
process/instagram_factory.py
```

Planned commands:

```text
instagram_factory.py validate-project --project <path>
instagram_factory.py list-granularity-items --project <path>
instagram_factory.py build-test-scenarios --project <path>
instagram_factory.py render-tests --project <path>
instagram_factory.py generate-batch --project <path>
instagram_factory.py regenerate --project <path> --items ... --slides ...
instagram_factory.py build-review-index --project <path>
instagram_factory.py mark-review --project <path> ...
instagram_factory.py publish-run-to-s3 --project <path>
```

### Orchestrator responsibilities

1. Load and validate the project specification.
2. Resolve catalogue entries.
3. Resolve active unified-data production batch.
4. Check source freshness and completeness.
5. Determine selected period.
6. Enumerate granularity items.
7. Build item contexts.
8. Generate visual assets.
9. Render complete slides.
10. Write manifests.
11. Generate test/contact sheets.
12. Publish previews.
13. Upload project run outputs to S3.
14. Support targeted regeneration.
15. Preserve previous runs.

---

## 8. Min/max/real testing design

This is a core requirement and must be implemented before general batch generation.

### 8.1 Test scenario builder

Proposed component:

```text
process/instagram_build_project_test_scenarios.py
```

It should inspect all candidate granularity items and generate:

#### Minimum scenario

- shortest title
- shortest subtitle/body text
- smallest non-empty numeric values
- fewest visual rows
- smallest image/media case
- fewest optional fields

#### Maximum scenario

- longest title
- longest subtitle/body text
- largest numeric values
- highest visual item count
- longest labels
- largest combined set of layout variables
- largest available image/media case
- all optional fields present

The maximum scenario should optimize for combined layout stress, not merely choose one naturally occurring row if different records contain different extreme values.

#### Real example

A real item selected using a documented rule, for example:

- median complexity
- user-selected ID
- highest-priority constituency/member
- first complete item

The real example must use internally consistent actual data.

### 8.2 Synthetic composite test contexts

Min/max scenarios may combine values from different records to create a worst-case synthetic context. Such contexts must be labelled synthetic and must never be treated as factual posts.

Required manifest fields:

- scenario type
- source fields used
- originating item IDs for each selected extreme
- synthetic flag
- no-publication flag

### 8.3 Complete-slide rendering

Testing must render the final post layout, not just the standalone visual.

For every slide:

```text
minimum/<slide_id>.png
maximum/<slide_id>.png
real_example/<slide_id>.png
```

The system should also build:

- one contact sheet per slide
- one contact sheet per scenario
- one project-wide validation index

### 8.4 Acceptance criteria

A project cannot move to batch generation until:

- all test renders succeed
- no text overflows
- no clipped visual labels
- no missing required data
- all real-example facts are validated
- warnings are reviewed
- user explicitly approves the design

---

## 9. Batch generation design

### 9.1 Item enumeration

The orchestrator should derive the complete item list from the selected grain and source table.

Each item requires:

- stable item key
- display label
- slug
- expected slide count
- source-row references

### 9.2 Render behavior

For each item:

1. Build item context.
2. Resolve slide variables.
3. Generate required visuals.
4. Render final slides.
5. Validate dimensions/files.
6. Write item manifest.
7. Continue independently if another item fails.

### 9.3 Failure behavior

Batch generation must not silently discard partial results.

Run summary states:

- succeeded
- succeeded_with_warnings
- partially_failed
- failed

Each item/slide must record its own state.

### 9.4 Idempotency

Generation should be deterministic for:

- project version
- data batch
- period
- item
- slide

Output filenames and manifest IDs should be stable.

---

## 10. S3 output design

Proposed root:

```text
s3://eirepolitic-data/processed/instagram_factory/projects/<project_id>/
```

Recommended structure:

```text
<project_id>/
  project.yml
  latest.json
  runs/
    <run_id>/
      run_manifest.json
      source_snapshot.json
      test_renders/
        minimum/
          <slide_id>.png
        maximum/
          <slide_id>.png
        real_example/
          <slide_id>.png
        contact_sheets/
      generated/
        <item_slug>/
          slide-01.png
          slide-02.png
          ...
          item_manifest.json
      review/
        review_state.json
        review_index.html
      logs/
      artifacts/
  approved/
    <approved_version_or_run>/
```

### Required manifests

#### Run manifest

- project/version
- run ID
- Git SHA
- workflow run ID
- active data batch ID
- selected period
- item count
- success/failure counts
- timestamps
- output prefix
- source table references

#### Item manifest

- item key/label/slug
- slide list
- input context hash
- visual files
- slide files
- warnings/errors
- review state
- generation timestamp

#### Slide manifest

- layout ID/version
- visual ID/version
- resolved text
- source references
- dimensions
- output hash
- warnings

---

## 11. Review and selective regeneration

### 11.1 Review state

Proposed file:

```text
review/review_state.json
```

Structure:

```json
{
  "project_id": "...",
  "run_id": "...",
  "items": {
    "dublin-bay-south": {
      "status": "approved",
      "slides": {
        "cover": "approved",
        "ranking": "approved"
      }
    },
    "cork-south-central": {
      "status": "changes_requested",
      "slides": {
        "cover": "approved",
        "ranking": "changes_requested"
      }
    }
  }
}
```

### 11.2 Targeted regeneration

The user should be able to request:

```text
Regenerate Cork South-Central, ranking slide only.
```

The orchestrator should:

- render only the specified items/slides
- preserve unaffected files
- update hashes/manifests
- increment regeneration metadata
- produce fresh preview links

### 11.3 Project-wide redesign

If the design changes globally:

- increment project version
- rerun min/max/real validation
- generate a new run
- preserve the prior run unchanged

---

## 12. Recurring projects

### 12.1 Schedule definition

A project may define:

```yaml
schedule:
  enabled: true
  cadence: monthly
  trigger: data_ready
  period_field: month
  completeness_check: latest_complete_period
```

### 12.2 Data readiness

A recurring generation must not run only because a date passed. It must confirm:

- expected upstream workflows succeeded
- required source tables are present
- selected period is complete
- enrichment pipelines finished
- mapping readiness passes

### 12.3 Generated period versions

Each period should create a distinct immutable run.

Examples:

```text
2026-01
2026-02
2026-Q1
2026
```

### 12.4 Notification

Future workflow should notify that a new draft batch is ready for review. Notification method is not yet selected.

---

## 13. Text and fact generation

### Current state

Text variables can be supplied manually through specs/templates. There is no formal factory-level text generation/validation component.

### Required design

Each text block should declare its source:

- static copy
- data interpolation
- deterministic rule
- LLM-generated draft
- externally validated fact

LLM-generated text must record:

- prompt/template ID
- source context
- model/provider
- generation timestamp
- review state

No LLM-generated factual claim should be treated as approved without source validation.

### Proposed component

```text
instagram/text_templates/
process/instagram_generate_text.py
```

It should support:

- deterministic templating first
- optional LLM drafting
- length limits per post layout
- prohibited/required wording rules
- provenance
- human approval

---

## 14. Image/media sourcing

### Current state

- post renderer supports media placement
- sourced-image visual is a placeholder/review wrapper
- Option 5 AI experiments exist separately
- no general licensed-image acquisition pipeline exists

### Required design

For internet-sourced images, record:

- source URL
- retrieval timestamp
- original filename/hash
- licence
- attribution requirements
- crop/transform metadata
- fallback behavior

For AI-generated/edited images, record:

- source image(s)
- prompt
- provider/model
- generation parameters
- review state

No image should be used in a final post without attribution/licence review or explicit approved internal-generation status.

---

## 15. Workflow design

### Existing canonical visual workflow

```text
.github/workflows/instagram_media_test.yml
```

Keep this as the visual regression/smoke workflow.

### Required new workflows

#### Project test workflow

```text
.github/workflows/instagram_factory_project_test.yml
```

Responsibilities:

- validate project
- generate min/max/real tests
- upload artifact
- publish preview branch output
- return links

#### Project batch workflow

```text
.github/workflows/instagram_factory_generate.yml
```

Responsibilities:

- validate approved project version
- enumerate granularity items
- generate complete batch
- upload to S3
- write run manifest
- build review index

#### Targeted regeneration workflow

```text
.github/workflows/instagram_factory_regenerate.yml
```

Inputs:

- project ID/version
- run ID
- item IDs
- slide IDs
- reason

#### Recurring generation workflow

```text
.github/workflows/instagram_factory_recurring.yml
```

Responsibilities:

- scheduled/data-ready trigger
- determine latest complete period
- generate new draft run
- notify reviewer

### Workflow requirements

- manual dispatch available
- exact errors surfaced
- artifacts uploaded even on partial failure
- no publishing permissions
- concurrency controls
- deterministic run IDs
- S3 writes limited to project prefix
- credentials kept in GitHub secrets/OIDC

---

## 16. Security, privacy, and governance

- No AWS credentials in code or project specs.
- Use existing repository secret/OIDC patterns.
- Public preview diagnostics must not expose sampled raw values.
- Render metadata containing real values must be classified before public publication.
- Sensitive/private metrics must not be placed on the public preview branch.
- Every final slide must have source/provenance metadata.
- Draft and approved outputs must be clearly separated.
- Automatic posting remains disabled.
- Production approval must be explicit and auditable.

---

## 17. Planned implementation phases

### Phase 0 — completed foundation

- unified data model access
- production pointer resolution
- compatibility mappings
- visual renderer framework
- fixture stress packs
- contact sheets
- preview workflow
- schema/readiness diagnostics
- initial live-data smoke suite
- documentation

### Phase 1 — catalogues and project schema

**Status: completed (2026-07-19) on `feature/instagram-content-factory-phase1`.**

Implemented:

- post, visual, and metric catalogues
- deterministic catalogue validation and option listing
- first reusable project specification template
- deterministic project validation with catalogue compatibility checks
- focused unit tests and GitHub Actions validation

Build:

- `instagram/catalogues/post_types.yml`
- `instagram/catalogues/visual_types.yml`
- `instagram/catalogues/metrics.yml`
- catalogue validator
- `instagram/projects/_template/`
- project schema validator

Acceptance:

- GPT can list available post/visual/metric options
- new approved types can be added without code changes to menus
- invalid projects fail with actionable errors

### Phase 2 — complete-slide test generator

**Status: constituency pilot implemented and technically validated (2026-07-19).**

Validated pilot:

- project: `instagram/projects/constituency_issue_profile_v1/project.yml`
- grain: constituency
- slides: cover and classified issue profile
- minimum, maximum, and real-example complete-slide renders
- per-scenario and per-slide contact sheets
- local fixture and live S3 execution
- production-pointer resolution and join-coverage manifests
- assistant-visible preview publication
- final live validation run: `29703335986`
- preview root: `instagram-preview-output/preview/factory/projects/constituency_issue_profile_v1/`

This validates the Phase 2 architecture for one pilot. A generic multi-project scenario builder remains future work. Human factual and visual approval is still required before batch generation.

Build:

- min/max/real scenario builder
- complete-slide render orchestration
- project validation contact sheets
- factual real-example validation manifest

Acceptance:

- every slide has minimum, maximum, and real-example renders
- no project can proceed without explicit approval

### Phase 3 — batch generation and S3 project storage

Build:

- granularity enumerator
- item context builder
- complete batch renderer
- run/item/slide manifests
- S3 uploader
- review index

Acceptance:

- one complete post set is generated for every selected item
- output structure matches the project S3 contract
- partial failures are isolated and documented

### Phase 4 — review state and targeted regeneration

Build:

- review-state file/API
- item/slide status changes
- selective rerender command/workflow
- immutable prior-run preservation

Acceptance:

- one bad item/slide can be fixed without regenerating the batch

### Phase 5 — recurring generation

Build:

- project schedule config
- latest-complete-period logic
- upstream readiness checks
- recurring draft generation
- reviewer notification

Acceptance:

- a new period creates one new draft run automatically after data readiness

### Phase 6 — future optional capabilities

Potential additions:

- content suggestion pipeline
- automatic source discovery
- automated external fact checks
- approval UI
- social scheduling
- automatic Instagram posting

These require separate design and approval.

---

## 18. Immediate next development tasks

Phase 1 is complete. The next milestone is Phase 2 complete-slide testing.

Recommended order for the next chat instance:

1. Read this file and the architecture/system docs.
2. Inspect current live repo state; do not assume this plan is perfectly current.
3. Inspect the completed catalogues, validators, and project template; do not duplicate them.
4. Choose one simple pilot project, preferably constituency or member grain with two slides.
5. Implement the min/max/real scenario builder for the pilot.
6. Integrate standalone visual assets into complete post-layout slots.
7. Render complete-slide test sets and contact sheets.
8. Add factual real-example validation manifests.
9. Review and refine before building batch mode.
10. Add S3 project storage only after the pilot test flow is approved.

Do not jump directly to full batch generation before complete-slide validation is working.

---

## 19. Definition of factory v1 complete

Factory v1 is complete when a user can, through conversation with an authorized GPT:

1. Define a post concept.
2. Select or create post/visual types from catalogues.
3. Define granularity, metrics, period, slides, text, and sources.
4. Commit a valid project specification.
5. Generate minimum, maximum, and real-example complete-slide tests.
6. Review and revise those tests through chat.
7. Approve the project design.
8. Generate one complete post set per granularity item.
9. Store outputs and manifests in the project S3 prefix.
10. Review all items.
11. Regenerate selected items/slides only.
12. Mark the run ready for posting.
13. Optionally configure recurring generation for future complete periods.

Automatic Instagram posting is not required for v1.

---

## 20. Handoff instructions for future GPT sessions

A future GPT should:

- treat this as the canonical product plan
- inspect the live repo before making decisions
- use only the repository name in GitHub tool calls
- preserve the separation between visual and post-layout layers
- work one component at a time
- use manual review workflows and direct preview links
- keep draft identifiers until explicit approval
- follow the unified data model and production pointer
- avoid duplicating existing renderer/mapping functionality
- update this plan whenever major architecture or scope changes
- report exact workflow/backend errors
- perform repository edits directly when tools are available

The desired interaction model is conversational, but every project, decision, data source, render, and review state must remain reproducible and auditable in the repository and S3.
