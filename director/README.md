# director/ — platform-neutral knowledge layer

This tree is the EirePolitic Director's operational truth. It is read by the
Claude Director (Sly Director project) today, and is designed to be read
identically by a future ChatGPT / "High Director V2" — nothing operational
lives only in Claude instructions.

**New agent, no prior context?** Start at
[`docs/director_agent_reference.md`](../docs/director_agent_reference.md)
instead — a single self-contained handoff page (operating rules, autonomy
boundary, this file, and `semantics.md`) written for exactly that case.

## How to use this tree

1. **Start at `refs.yml`.** It is the keystone: capability → state
   (`merged | pinned | draft | experimental | superseded`) → ref/PR/notes.
   Never assume a capability is on `main` — check its state first.
2. **Read `semantics.md` before generating or describing any political
   content.** Those rules are non-negotiable and are not repeated per-file.
3. For "what can the factory do today", read `capabilities.yml`.
4. For "which project renders X on what schedule", read `projects.yml`.
5. For "what does a completed post look like / which render is the design
   reference", read `references.yml`.
6. For dataset/table/pointer questions, read `data_products.yml`.
7. For renderer tuning ("labels too small" → which constant), read
   `visuals.yml`.
8. For "what does this GitHub Actions workflow do, and is it current or
   superseded", read `workflows.yml`.
9. For publishing readiness, read `publishing.yml`. Publishing is blocked in
   v1 regardless of what this file says — see `semantics.md`.
10. Per-conversation decision state (Phase 4) lives under `sessions/<id>/`.
11. For the step-by-step procedure behind each row of the task routing
    table below — including exactly when to write session state and when
    to stop and ask Warren — read `workflows_v1.md` (Phase 5).

## Generated vs. hand-maintained

- **Generated** by `process/build_director_catalogue.py`, drift-checked in
  CI (`.github/workflows/director_catalogue_drift_ci.yml`): the workflow
  inventory section of `workflows.yml`, and the project-directory listing
  section of `projects.yml`. These sections carry an
  `# AUTO-GENERATED — do not hand-edit` header. If CI is red, someone edited
  a generated section by hand, or GitHub state moved and the file wasn't
  regenerated — either way, trust GitHub, not the stale file, and re-run the
  generator.
- **Hand-maintained, human-reviewed**: `refs.yml` (state judgments),
  `references.yml` (canonical-vs-production classification),
  `semantics.md`, `capabilities.yml`, `data_products.yml`, `visuals.yml`,
  `publishing.yml`, `workflows_v1.md`, and the non-generated parts of
  `projects.yml` and `workflows.yml` (purpose/notes columns).

## Size discipline

`director/` is an index, not a copy of the repository. If something is
cheaply derivable from a live GitHub or S3 read, this tree points at it
(path, ref, workflow name) rather than duplicating its content. Long-form
detail belongs in `docs/`, not here.

## Task routing (plan §5)

| Request | Route |
|---|---|
| "Generate this month's X" | Existing-series workflow — resolve project in `projects.yml`, resolve period, check data readiness, dispatch render. Procedure: `workflows_v1.md` §6.2 |
| "Let's make a post" | New-post collaboration loop — content idea, references, one prototype slide, full render, feedback, approval. Procedure: `workflows_v1.md` §6.1 |
| "Slide 3 is too crowded" | Modify-from-feedback — map to a knob in `visuals.yml`. Procedure: `workflows_v1.md` §6.3 |
| "Add this dataset" / "add a metric" | Data-product workflow — see `data_products.yml`. Procedure: `workflows_v1.md` §6.4 |
| "I need a visual like this" | Capability check in `capabilities.yml` first — extend before rebuilding |
| "Fix this broken post" | Resolve ref state in `refs.yml` first, then diagnose |
| "Schedule this" | Automation — requires explicit approval from Warren |
| "Publish this" | **Blocked in v1.** Report publishing status only — see `semantics.md` |

## Provenance

Built starting Phase 2 (2026-09-17) of the EirePolitic Director
implementation. See the Sly Director project doc
`claude/eirepolitic-director-implementation-plan.md` for the full plan,
decisions, and phase history.
