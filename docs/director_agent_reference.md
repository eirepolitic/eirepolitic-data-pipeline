# EirePolitic Director — Agent Reference

*As of 2026-09-27.*

## What this is

This page is the standalone briefing for any LLM agent picking up operation of **EirePolitic's** Irish political-data Instagram content pipeline with no prior context beyond this page plus live tool access.

It summarizes the current operating model, safety rules, Director knowledge tree, factory workflow, and the now-production Instagram publishing/scheduling system. Read this page first; then read the live repository files it points to before acting.

**Repository:** `Eirepolitic-data-pipeline`, owner `eirepolitic`, default branch `main`.

**Prerequisite tools for an agent to actually operate the pipeline:**

- GitHub access to this repository with the ability to read workflows, runs, branches, files, PRs, Actions variables, and dispatch approved workflows;
- AWS access through the existing GitHub → `HighDirectorAwsAdmin` STS path when infrastructure/runtime maintenance is required.

Agents should not request or expose Meta tokens, AWS secret values, or other stored credentials.

## First reads for every substantive task

1. Start at `director/refs.yml`. It is the keystone for capability state (`merged | pinned | draft | experimental | superseded`). Never assume a capability is on `main` because it sounds finished.
2. Read `director/semantics.md` before generating or describing political content. Its rules are non-negotiable.
3. Read `director/README.md` for the map of the operational knowledge tree.
4. For concrete task procedures, read `director/workflows_v1.md`.
5. For publishing capability/state, read `director/publishing.yml` and the canonical runbook `docs/operations/instagram_publishing_standard.md`.
6. For agent-level publishing instructions, read `instagram/PUBLISHING.md`.
7. For AWS/data questions, read `director/data_products.yml` and then verify live state rather than trusting stale assumptions.
8. For substantive content-generation work, use the session mechanism under `director/sessions/<id>/` so later agents can reconstruct intent, decisions, runs, feedback, and approval state.

## What an agent can do without asking, and what requires explicit human approval

### Without asking

- Read-only repository/AWS inspection.
- Dispatching existing approved workflows on `main` when the task clearly calls for them.
- Ordinary code, documentation, config, branch, and PR work within the repository's existing architecture.
- Re-running deterministic render/validation/healthcheck workflows.
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

## Canonical publishing workflow

For normal publication, use exactly this sequence:

1. Run **Instagram factory render (generic)** (`.github/workflows/instagram_factory_render.yml`).
2. Review the generated preview and obtain explicit approval for the exact output.
3. Record/copy the factory GitHub Actions run ID.
4. Run **Instagram publish (standard)** (`.github/workflows/instagram_publish_standard.yml`).
5. Supply:
   - `factory_run_id`;
   - `approved_by`;
   - `mode` = `scheduled` or `immediate`.
6. For scheduled mode, also supply:
   - `scheduled_local` = `YYYY-MM-DDTHH:MM:SS`;
   - `timezone` = IANA timezone, normally `America/Vancouver` when Pacific time was requested.
7. Leave `options_json={}` unless advanced Instagram fields were explicitly approved.
8. Use `caption_override` only when the reviewed factory artifact has no caption file.

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

For maintenance, `HIGH_DIRECTOR_AWS_OPERATION` is used by `.github/workflows/deploy_instagram_publisher_lambda.yml`. Leave it at `infrastructure-status` except during an explicit maintenance operation.

## Publication runtime behavior

The production Lambda accepts:

- `{"action":"healthcheck"}` — read-only Meta connectivity check;
- `{"action":"execute_publication","publication_id":"...","expected_version":N}` — immediate execution of an already-approved publication;
- scheduler payload `{"publication_id":"...","expected_version":N}` — scheduled execution.

The Lambda does **not** accept raw caption/image content as publication authority. It reloads the approved `PublicationRequest`, immutable `AssetPackage`, approval fingerprint, credentials, and durable execution state from AWS before contacting Meta.

Durable execution state stores Meta container/media IDs and operation results so retries reuse prior provider objects rather than blindly creating duplicate posts.

Scheduled posts use:

- EventBridge Scheduler group `eirepolitic-instagram`;
- dedicated scheduler execution role;
- SQS DLQ;
- bounded retry policy;
- `ActionAfterCompletion=DELETE` for one-time schedules.

## Canonical publishing documentation

Use these files as the source of truth:

- `docs/operations/instagram_publishing_standard.md` — full operating contract;
- `instagram/PUBLISHING.md` — concise agent/operator quick start;
- `director/publishing.yml` — current capability/state summary;
- `.github/workflows/instagram_publish_standard.yml` — normal publication interface;
- `.github/workflows/instagram_factory_render.yml` — source render/review workflow;
- `.github/workflows/deploy_instagram_publisher_lambda.yml` — maintenance only;
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
| "Let's make a post" | New-post collaboration loop — idea, references, prototype, full render, feedback, approval. |
| "Slide 3 is too crowded" | Modify-from-feedback — use `visuals.yml` and re-render. |
| "Add this dataset" / "add a metric" | Data-product workflow — `data_products.yml` / `workflows_v1.md`. |
| "I need a visual like this" | Check `capabilities.yml` before creating a new subsystem. |
| "Fix this broken post" | Resolve live ref/workflow state first, then diagnose. |
| "Schedule this" | After exact content approval, use **Instagram publish (standard)** with `mode=scheduled`. |
| "Publish this now" | After exact content approval, use **Instagram publish (standard)** with `mode=immediate`. |
| "Is publishing working?" | Read `director/publishing.yml`; run the maintenance `healthcheck` if live verification is needed. |

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
- Publication execution must go through `instagram_publish_standard.yml`; agents must not bypass its immutable-assets + approval-fingerprint boundary.
- Scheduled times must preserve the requested local time/timezone exactly and should be reported back to the user in both local and resolved UTC forms.

## Evidentiary discipline for the agent itself

- Do not describe repository, AWS, workflow, or data state that has not been verified live in the current session (or by a clearly identified prior verification with date).
- "Probably", "should be", and "I'd expect" are not substitutes for checking.
- If the tree marks something `draft`, `experimental`, `superseded`, or `needs_review`, say so rather than presenting it as production truth.

## Background

This page is designed to be enough for a fresh agent to begin operating the system safely with live repository/AWS access.

For historical implementation detail, consult the repository's Director planning/history documents, but do not treat historical plans as current operational truth when they conflict with `main`, `director/publishing.yml`, or `docs/operations/instagram_publishing_standard.md`.
