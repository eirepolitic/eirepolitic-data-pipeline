# EirePolitic Director — Agent Reference

*As of 2026-09-27.*

## What this is

This page is the standalone briefing for any LLM agent picking up operation of **EirePolitic's** Irish political-data Instagram content pipeline with no prior context beyond this page plus live tool access.

It summarizes the current operating model, safety rules, Director knowledge tree, content-development workflow, factory workflow, and the now-production Instagram publishing/scheduling system. Read this page first; then read the live repository files it points to before acting.

**Repository:** `Eirepolitic-data-pipeline`, owner `eirepolitic`, default branch `main`.

**Prerequisite tools for an agent to actually operate the pipeline:**

- GitHub access to this repository with the ability to read workflows, runs, branches, files, PRs, Actions variables, and dispatch approved workflows;
- AWS access through the existing GitHub → `HighDirectorAwsAdmin` STS path when infrastructure/runtime maintenance is required.

Agents should not request or expose Meta tokens, AWS secret values, or other stored credentials.

## First reads for every substantive task

1. Start at `director/refs.yml`. It is the keystone for capability state (`merged | pinned | draft | experimental | superseded`). Never assume a capability is on `main` because it sounds finished.
2. Read `director/semantics.md` before generating or describing political content. Its rules are non-negotiable.
3. Read `director/README.md` for the map of the operational knowledge tree.
4. Read `director/references.yml` before making visual/design decisions. The latest render is not automatically the canonical visual target. For a new post, inspect the completed-post archive through the repository's reference workflow before designing from scratch.
5. Read `director/visuals.yml` before modifying layouts or renderers. Prefer existing house-style primitives and reusable knobs over one-off visual systems.
6. For concrete task procedures, read `director/workflows_v1.md`.
7. For publishing capability/state, read `director/publishing.yml` and the canonical runbook `docs/operations/instagram_publishing_standard.md`.
8. For agent-level publishing instructions, read `instagram/PUBLISHING.md`.
9. Before choosing, joining, interpreting, or describing EirePolitic political datasets, read the **Irish Politics Data Model** catalogue: https://eirepolitic.github.io/projects/data/irish-politics-data-model/. It is the human- and agent-readable map of the current usable model, including dataset grain, source, transformations, relationships, current columns, physical location, caveats, and real example rows.
10. For AWS/data questions and operational data-product state, also read `director/data_products.yml` and then verify live production pointers/files rather than trusting stale assumptions. The catalogue explains what data is available; `director/data_products.yml` and live state determine what is operationally current.
11. For substantive content-generation work, use the session mechanism under `director/sessions/<id>/` so later agents can reconstruct intent, references, data sources, decisions, runs, feedback, and approval state.

## Using the data documentation

The live **Irish Politics Data Model** catalogue is the default starting point when an agent needs to answer questions such as “what data do we have?”, “which table contains this?”, “what does one row represent?”, “which identifier should I join on?”, or “what do the actual values look like?”.

**Catalogue:** https://eirepolitic.github.io/projects/data/irish-politics-data-model/

Use it to discover and understand the current production datasets before writing new extraction/query logic. In particular, inspect the documented **grain, source, transformations, relationships, caveats, schema, and real example rows** for every dataset you intend to use. Do not guess table or column names from memory, and do not assume that similarly named identifiers are enforced foreign keys unless the catalogue or implementation establishes the relationship.

The catalogue is documentation, not a substitute for live verification. When freshness, availability, batch identity, or operational state matters, resolve the active production pointer/manifest and inspect the current files/workflows. If the catalogue and live implementation disagree, prefer the current production implementation and update the documentation rather than silently relying on stale catalogue text.

## What an agent can do without asking, and what requires explicit human approval

### Without asking

- Read-only repository/AWS inspection.
- Dispatching existing approved workflows on `main` when the task clearly calls for them.
- Ordinary code, documentation, config, branch, and PR work within the repository's existing architecture.
- Re-running deterministic render/validation/healthcheck workflows.
- Running the read-only Instagram schedule inventory before choosing a new scheduled slot.
- Using the maintenance workflow for already-authorized infrastructure status or deployment operations when the task requires it.

### Requires explicit human approval

- Final approval of political content for publication.
- Publishing immediately when the user has not already explicitly approved that exact reviewed output.
- Scheduling a publication when the user has not explicitly approved that exact reviewed output and requested/approved the schedule.
- Any content-idea, visual-direction, or final-approval gate reserved for the human operator in `director/workflows_v1.md`.
- Any architectural change that weakens the separation between factory rendering/review and production publishing authority.

When a genuine blocker is hit — a missing approval, missing permission, missing evidence, or unrecoverable failure — stop and report it clearly rather than guessing or bypassing the control.

## Critical distinction: the factory still does not publish

The content factory and the production publisher are intentionally separate systems.

The factory continues to enforce:

- `publication_enabled=false` / `publishing_allowed=false`;
- review-only output until every required review item/slide is approved;
- `ready_for_posting` as a handoff state, not a publication action.

**Do not change those factory flags to `true`.** The production publishing system does not require that and should never depend on it.

A successful generic factory render now emits a portable `publication_handoff.json` in its uploaded `generated_render/` artifact. That handoff contains relative slide paths, caption path when present, project/period identity, QA state, and factory safety flags.

The separate production publishing workflow consumes that reviewed artifact and creates the actual publication authority in DynamoDB by recording an approval fingerprint tied to the exact assets, caption, options, and publication version.

## Production Instagram publishing is now enabled

The old v1 rule that "publishing is blocked" is obsolete.

Production publishing was proven end to end on the real Eirepolitic Instagram Professional account on 2026-09-26:

- **Gate 4:** immediate single-image test post successfully published through the production S3 → DynamoDB → Lambda → Meta `/media` → `/media_publish` path; the returned permanent Instagram media ID was persisted and the test post was manually deleted.
- **Gate 5:** one-time EventBridge Scheduler invocation successfully published the scheduled test post at the requested Pacific time; the schedule target/role/DLQ/payload were verified and the post was manually deleted afterward.

The standard system is now the supported mechanism for both immediate and scheduled posts.

## Where scheduled posts are kept and how to inspect the calendar

Future agents must not infer schedule availability from chat history, old workflow runs, or EventBridge alone.

### Canonical schedule calendar: DynamoDB

The authoritative record of future Instagram publications is the DynamoDB table:

`eirepolitic-publications`

A future scheduled publication has a control record with:

- `state = scheduled`;
- `scheduled_local`;
- `timezone`;
- `scheduled_at_utc`;
- publication ID/version;
- project/period identity;
- approval and immutable asset references.

The table's `state-scheduled_at-index` index is the canonical chronological schedule/calendar query.

### Live execution mirror: EventBridge Scheduler

The corresponding one-time execution job lives in EventBridge Scheduler group:

`eirepolitic-instagram`

EventBridge is **not** the durable calendar. One-time schedules use `ActionAfterCompletion=DELETE`, so the live Scheduler job disappears after execution. DynamoDB remains the authoritative historical/control record.

### Mandatory pre-scheduling workflow

Before proposing or choosing a time for a new scheduled post, run:

**Instagram schedule inventory**  
`.github/workflows/instagram_schedule_inventory.yml`

This read-only workflow:

1. queries future `state=scheduled` publication records from DynamoDB in chronological order;
2. shows each post's local time, timezone, project ID, and publication ID;
3. looks up the expected matching EventBridge Scheduler job;
4. reports the live Scheduler state and whether it matches the ledger record.

If the inventory is empty, there are no upcoming scheduled Instagram posts in the canonical ledger.

If an inventory row says the Scheduler job is missing or mismatched, treat that as an operational inconsistency to resolve before adding another schedule.

Use this inventory to decide where a new post fits relative to already-planned content. There is currently **no automatic minimum-spacing rule** enforced by the code; editorial cadence/spacing remains an operator decision informed by the visible inventory and the user's instructions.

## Canonical publishing workflow

For normal publication, use exactly this sequence:

1. Run **Instagram factory render (generic)** (`.github/workflows/instagram_factory_render.yml`).
2. Review the generated preview and obtain explicit approval for the exact output.
3. Record/copy the factory GitHub Actions run ID.
4. **If scheduling, run Instagram schedule inventory and review all upcoming posts before selecting the new time.**
5. Run **Instagram publish (standard)** (`.github/workflows/instagram_publish_standard.yml`).
6. Supply:
   - `factory_run_id`;
   - `approved_by`;
   - `mode` = `scheduled` or `immediate`.
7. For scheduled mode, also supply:
   - `scheduled_local` = `YYYY-MM-DDTHH:MM:SS`;
   - `timezone` = IANA timezone, normally `America/Vancouver` when Pacific time was requested.
8. Leave `options_json={}` unless advanced Instagram fields were explicitly approved.
9. Use `caption_override` only when the reviewed factory artifact has no caption file.

The standard workflow automatically:

- downloads the exact factory artifact by run ID;
- validates `publication_handoff.json` and factory QA/safety state;
- converts slides to deterministic delivery JPEGs;
- uploads immutable content-addressed assets to the private approved-assets S3 bucket;
- persists the `AssetPackage` in DynamoDB;
- creates the exact `PublicationRequest`;
- parses hashtags and caption mentions from the exact caption;
- records the approval fingerprint;
- either invokes the publisher Lambda immediately or creates/verifies a one-time EventBridge Scheduler job.

Agents must **not** recreate this with direct Meta API calls, manual S3 uploads, custom EventBridge schedules, or temporary Lambda publish actions.

## Agent/High Director trigger path

Some GitHub integrations can dispatch workflows but cannot pass `workflow_dispatch` inputs directly. In that case, set the repository Actions variables below and dispatch the **same** `instagram_publish_standard.yml` workflow:

- `INSTAGRAM_FACTORY_RUN_ID`
- `INSTAGRAM_APPROVED_BY`
- `INSTAGRAM_PUBLISH_MODE` = `scheduled` or `immediate`
- `INSTAGRAM_SCHEDULED_LOCAL` for scheduled mode
- `INSTAGRAM_TIMEZONE`
- `INSTAGRAM_CAPTION_OVERRIDE` only when needed
- `INSTAGRAM_OPTIONS_JSON` (normally `{}`)

There is no separate agent-only publishing path.

For scheduled mode, still run `instagram_schedule_inventory.yml` first. Do not select a time solely from repository variables or prior conversation context.

For maintenance, `HIGH_DIRECTOR_AWS_OPERATION` is used by `.github/workflows/deploy_instagram_publisher_lambda.yml`. Leave it at `infrastructure-status` except during an explicit maintenance operation.

## Publication runtime behavior

The production Lambda accepts:

- `{"action":"healthcheck"}` — read-only Meta connectivity check;
- `{"action":"execute_publication","publication_id":"...","expected_version":N}` — immediate execution of an already-approved publication;
- scheduler payload `{"publication_id":"...","expected_version":N}` — scheduled execution.

The Lambda does **not** accept raw caption/image content as publication authority. It reloads the approved `PublicationRequest`, immutable `AssetPackage`, approval fingerprint, credentials, and durable execution state from AWS before contacting Meta.

Durable execution state stores Meta container/media IDs and operation results so retries reuse prior provider objects rather than blindly creating duplicate posts.

Scheduled posts use:

- DynamoDB table `eirepolitic-publications` as the canonical schedule ledger;
- DynamoDB index `state-scheduled_at-index` for the upcoming calendar;
- EventBridge Scheduler group `eirepolitic-instagram` as the live execution mirror;
- dedicated scheduler execution role;
- SQS DLQ;
- bounded retry policy;
- `ActionAfterCompletion=DELETE` for one-time schedules.

## Canonical publishing documentation

Use these files as the source of truth:

- `docs/operations/instagram_publishing_standard.md` — full operating contract;
- `instagram/PUBLISHING.md` — concise agent/operator quick start;
- `director/publishing.yml` — current capability/state summary;
- `.github/workflows/instagram_schedule_inventory.yml` — read-only upcoming schedule/calendar view;
- `.github/workflows/instagram_publish_standard.yml` — normal publication interface;
- `.github/workflows/instagram_factory_render.yml` — source render/review workflow;
- `.github/workflows/deploy_instagram_publisher_lambda.yml` — maintenance only;
- `publishing/schedule_inventory.py` — canonical schedule-ledger/Scheduler cross-check;
- `publishing/standard_pipeline.py` — factory artifact promotion/approval;
- `publishing/lambda_handler.py` — generic immediate/scheduled execution;
- `publishing/aws_runtime.py`, `publishing/dynamodb_runtime.py`, `publishing/scheduler.py` — runtime/idempotency/scheduling internals.

If this briefing and live code/docs disagree, the current `main` implementation plus `docs/operations/instagram_publishing_standard.md` win.

## Tree navigation (`director/README.md`)

`director/` is the Director's platform-neutral operational truth.

1. `refs.yml` — capability state and refs.
2. `semantics.md` — non-negotiable political-content and publishing-boundary rules.
3. `capabilities.yml` — current factory/content capabilities.
4. `projects.yml` — project/schedule mapping.
5. `references.yml` — canonical design references.
6. `data_products.yml` — datasets/tables/pointers.
7. `visuals.yml` — renderer tuning knobs.
8. `workflows.yml` — workflow inventory/currentness.
9. `publishing.yml` — current production publishing state.
10. `sessions/<id>/` — per-conversation decision state.
11. `workflows_v1.md` — step-by-step content workflow procedures.

Generated sections in `workflows.yml` / `projects.yml` are built by `process/build_director_catalogue.py` and drift-checked by CI. If a generated catalogue is stale, trust live GitHub state and regenerate it.

## Task routing

| Request | Route |
|---|---|
| "Generate this month's X" | Existing-series workflow — resolve project, period, data readiness, then dispatch the factory render. |
| "Let's make a post" | New-post collaboration loop — inspect references/archive, establish evidence, prototype one representative slide, confirm direction, then scale to the full render. |
| "Slide 3 is too crowded" | Modify-from-feedback — use `visuals.yml`, make the smallest reusable change, and re-render in the same session. |
| "What data do we have?" / "which table should I use?" | Start with the [Irish Politics Data Model](https://eirepolitic.github.io/projects/data/irish-politics-data-model/), then verify the relevant live production pointer/files before using the data. |
| "Add this dataset" / "add a metric" | Review the existing model in the [Irish Politics Data Model](https://eirepolitic.github.io/projects/data/irish-politics-data-model/) first, then use the data-product workflow — `data_products.yml` / `workflows_v1.md`. |
| "I need a visual like this" | Check `capabilities.yml` and `references.yml` before creating a new subsystem. |
| "Fix this broken post" | Resolve live ref/workflow state first, then diagnose. |
| "Schedule this" | After exact content approval, first run **Instagram schedule inventory**; then use **Instagram publish (standard)** with `mode=scheduled`. |
| "What is already scheduled?" | Run **Instagram schedule inventory** and treat DynamoDB `eirepolitic-publications` as the canonical calendar. |
| "Publish this now" | After exact content approval, use **Instagram publish (standard)** with `mode=immediate`. |
| "Is publishing working?" | Read `director/publishing.yml`; run the maintenance `healthcheck` if live verification is needed. |

## New Instagram post development playbook

This is the preferred operating method for a new political-data post or a substantial new visual format. It is intentionally more detailed than the three gates in `director/workflows_v1.md` because it captures the practical lessons from recent post-development work. It is generic: reuse the method, not the subject matter or a particular Bill Tracker layout.

### 1. Establish live state before designing

Before writing copy or rendering anything:

- read `director/refs.yml` and verify whether the relevant project/capability is `merged`, `pinned`, `draft`, `experimental`, or `superseded`;
- read `director/references.yml` and inspect the closest **approved** completed post via `.github/workflows/completed_post_reference.yml` when one exists;
- read `director/visuals.yml` and the relevant `instagram/projects/<project_id>/` files;
- inspect the actual workflow that will render the project rather than relying on a historical description;
- resolve data through the active production pointer/immutable batch where the project uses the Oireachtas pipeline. Do not mix rows from different production batches in one post.

Do not treat a newer preview as aesthetically canonical merely because it is newer. Do not treat a draft branch or unapproved visual-review render as a completed-post reference.

### 2. Reuse the house style and renderer architecture

Start from the closest approved visual system rather than rebuilding the renderer from scratch.

For the current recurring factory this normally means:

- reuse the EirePolitic dark-green / cream / gold palette and shared render primitives recorded in `director/visuals.yml`;
- use the approved outer-layout/template system where it fits;
- respect pinned pixel-critical assets and the hash checks already enforced by `.github/workflows/instagram_factory_render.yml`;
- extend an existing renderer/project adapter before creating a parallel rendering subsystem;
- keep content/data configuration separate from drawing code where practical so the same visual grammar can be reused with different datasets.

If a new post is not yet compatible with the generic factory, a temporary analysis renderer/workflow is acceptable for prototyping. It must remain review-only, should live on a clearly named analysis branch, and any existing unrelated workflow temporarily repurposed for a preview must be restored immediately after the run. Once the format is approved, migrate the reviewed work into the normal project/factory structure rather than leaving production dependent on a temporary script.

### 3. Prototype one representative slide before scaling

Do not build a full carousel before the visual direction is proven.

Choose the slide that exercises the hardest or most representative layout: long labels, stacked bars, dense annotations, multiple parties, long title, methodology content, etc. Render that one slide, publish it to a hosted preview, and get the human visual-direction decision first.

For a two-slide recurring unit, it is often efficient to approve the most complex visualization first and then its paired explainer. Preserve approved components during later revisions; do not redesign them because an unrelated slide is being changed.

### 4. Audit political evidence at proposition level

A dataset relationship is not enough to justify an editorial claim. For political and legislative content, verify the **meaning** of the underlying event before labelling it.

For a recorded parliamentary vote:

1. identify the exact division, House, date, stage, and proposition;
2. read the official debate/division record when necessary to establish what was actually being decided;
3. state explicitly what a Tá and a Níl meant for that proposition;
4. only describe Tá/Níl as support/opposition to the whole Bill when the proposition itself genuinely supports that interpretation;
5. treat amendment votes, procedural votes, guillotine/combined questions, stage motions, and final-passage questions as different editorial objects;
6. do not infer support for a Bill because a TD spoke in a Bill debate;
7. do not infer that `no recorded vote` means `absent`;
8. do not assume Bills with the same status have equivalent vote data. A Bill can be enacted without a recorded member-by-member division at the stage you want to illustrate.

Where useful, the official Houses of the Oireachtas Bill/debate/division pages, `data.oireachtas.ie`, and `api.oireachtas.ie/v1/votes` can be used to certify the proposition in addition to the repository's processed data.

If no suitable recorded division exists, say so explicitly and change the slide type. Never invent a party split merely to preserve carousel symmetry.

### 5. Reconstruct party breakdowns with temporal joins, not present-day labels

When a vote visualization groups individual members by party, use the party affiliation valid **on the vote date**.

The proven repository pattern is:

- select the exact division from `silver_member_votes`;
- build the eligible member universe from `silver_member_memberships` for the correct House/term/date;
- join `silver_member_parties` using `party_start` / `party_end` against the vote date;
- detect ambiguous overlapping party histories rather than silently selecting one;
- calculate Tá, Níl, abstain/other, and `no recorded vote` as separate values;
- reconcile party totals back to the overall division total and eligible-member count before rendering.

The residual between eligible members and members present in the division data is `no recorded vote`; it is not automatically absence. If affiliation history is ambiguous, stop or label the limitation rather than assigning a convenient party.

### 6. Write for zero assumed procedural knowledge

A factual post can still fail if the reader needs specialist knowledge to understand it. Explain unfamiliar Irish parliamentary terms at the point they matter or in a compact glossary/methodology slide.

For an explainer slide, prefer this order where applicable:

- what the measure/data means in plain English;
- practical effect;
- who introduced or owns the measure where relevant;
- arguments made in favour;
- concerns or arguments raised against;
- what the accompanying visualization or vote was actually measuring/deciding;
- what the result changed.

Keep advocacy and attribution separate: phrases such as “supporters argued…” and “critics raised…” should reflect sourced arguments, not the Director's own judgement.

A glossary subtitle must match its content. If it promises a process (for example, how a Bill moves through Parliament), include a compact process explanation; otherwise rename it to describe the definitions actually shown.

### 7. Treat layout as deterministic geometry, not eyeballing

Instagram review output should be reproducible and should fail loudly on overflow.

Useful practices from recent review work:

- target the project's declared dimensions (normally 1080×1350 for these carousels);
- wrap or shrink long titles to a bounded number of lines before they collide with charts or ornaments;
- center numbers/text from measured bounding boxes or anchors, not hand-tuned visual offsets;
- size repeated text from the most constrained/longest instance, then apply the same size consistently where the design calls for consistency;
- reserve deliberate vertical breathing room between explanatory copy, labels such as `RESULT`, and the result itself;
- use explicit `max_lines`, bounds, and overflow assertions in the renderer;
- validate every generated slide's dimensions before publishing a review branch;
- keep source/methodology text legible rather than treating it as decorative microtext.

Human feedback such as “make the description text bigger”, “center this”, or “title overlaps” should map to a concrete reusable parameter where possible. Record the literal feedback and the technical change in the active Director session.

### 8. Use hosted review branches, individual slides, and contact sheets

The human reviewer should not have to open local files or infer which render is current.

For each meaningful review iteration:

- publish deterministic assets to a `previews/<slug>` branch;
- return a browser-viewable `raw.githack.com/.../index.html` review page;
- provide direct hosted image links for the changed slide(s);
- generate a contact sheet when reviewing a sequence or full post;
- include the exact intended slide order in the contact sheet/review page;
- keep the preview branch slug stable across iterations when the reviewer is following one evolving post, unless parallel alternatives genuinely need separate branches.

The generic factory already stages a review page, contact sheets, `slides.zip`, and a preview branch. Prefer it once the project is integrated. Temporary analysis previews should emulate the same reviewer experience rather than returning `/mnt/data` paths as the final review surface.

### 9. Do not call a “full review” complete while placeholders remain

A full-post contact sheet is a content-completeness gate, not just a visual collage.

Before presenting a post as ready for final review:

- every slide must contain the actual intended data/copy or be explicitly labelled as an unresolved draft;
- every chart must be backed by verified data and semantics;
- every exception (for example, no recorded division) must be handled truthfully in the visual rather than hidden behind a generic placeholder;
- cover/title, explainer, visualization, glossary/methodology, and source treatments should all be present in their intended order;
- unresolved evidence questions must be resolved before the post is described as ready to publish.

If a review package contains placeholders, say **which slides are incomplete and why**. Do not present it as a final/full-post review merely because all image slots exist.

### 10. Iterate narrowly from feedback

When the human reviewer gives feedback:

- change exactly the requested dimension first — copy, font size, spacing, centering, label, etc.;
- do not silently rewrite approved copy or redesign unrelated slides;
- make the smallest reusable code/config change;
- re-render and validate;
- publish the new hosted preview;
- record the feedback loop in the same Director session.

Once a pattern repeats, parameterise it in the project/render configuration instead of accumulating one-off constants. Update `director/visuals.yml` when a new feedback phrase has become a reusable knob.

### 11. Promote only after approval

After the full post is approved:

- migrate any prototype-only code into the normal project/factory structure;
- keep factory publication flags false;
- record the exact approved render/run in the Director session and, where appropriate, `director/references.yml`;
- archive the completed post through the completed-post workflow with an agent summary that truthfully records the data sources, rendering tools, important decisions, QA, limitations, and approvals actually used;
- only then use the separate standard publishing workflow for immediate or scheduled publication.

A successful prototype is not automatically a canonical reference. Canonical status is a deliberate human-reviewed decision recorded in `director/references.yml`.

### Working example without making it a universal template

The 2026 Bill Tracker development session is a useful example of this method because it exercised long-title handling, explainer copy, 100% stacked vote bars, party-level temporal joins, glossary/methodology design, exact-proposition verification, iterative visual feedback, contact-sheet review, and truthful handling of a Bill with no recorded division. However, `director/refs.yml` currently classifies `bill_tracker_series` as **draft**. Use the workflow lessons above; do not treat the Bill Tracker assets themselves as an approved canonical design reference unless `refs.yml` / `references.yml` later say otherwise.

## Non-negotiable political-content semantics

EirePolitic publishes factual, source-grounded political data content. An agent must:

- stay neutral, factual, and source-grounded;
- trace every figure, quote, or claim to a specific dataset, table, batch, or document;
- never invent missing evidence;
- never infer an individual's political preference, voting behaviour, or affiliation beyond what source data records;
- never make voting recommendations, endorse a candidate or party, or rank parties by favourability;
- never create covert persuasion, microtargeting, or content designed to appear organic/grassroots when it is not;
- preserve attribution/sourcing in generated assets, captions, manifests, and methodology slides.

### Publishing semantics

- The **factory** must keep `publication_enabled=false` / `publishing_allowed=false`.
- `ready_for_posting` requires complete approval of the factory review state.
- The **separate standard publisher** may publish or schedule only after explicit human approval of the exact reviewed output.
- Before selecting a scheduled time, the agent must run `instagram_schedule_inventory.yml` and review the canonical upcoming schedule from DynamoDB.
- Publication execution must go through `instagram_publish_standard.yml`; agents must not bypass its immutable-assets + approval-fingerprint boundary.
- Scheduled times must preserve the requested local time/timezone exactly and should be reported back to the user in both local and resolved UTC forms.

## Evidentiary discipline for the agent itself

- Do not describe repository, AWS, workflow, schedule, or data state that has not been verified live in the current session (or by a clearly identified prior verification with date).
- "Probably", "should be", and "I'd expect" are not substitutes for checking.
- Do not infer the current Instagram calendar from chat history. Run the schedule inventory when schedule state matters.
- If the tree marks something `draft`, `experimental`, `superseded`, or `needs_review`, say so rather than presenting it as production truth.

## Background

This page is designed to be enough for a fresh agent to begin operating the system safely with live repository/AWS access.

For historical implementation detail, consult the repository's Director planning/history documents, but do not treat historical plans as current operational truth when they conflict with `main`, `director/publishing.yml`, or `docs/operations/instagram_publishing_standard.md`.
