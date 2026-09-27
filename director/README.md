# director/ — platform-neutral knowledge layer

This tree is the EirePolitic Director's operational truth. It is read by the
Claude Director today and is designed to be read identically by future agents —
nothing operational should live only in one model's instructions.

**New agent, no prior context?** Start at
[`docs/director_agent_reference.md`](../docs/director_agent_reference.md)
instead — a single self-contained handoff page covering operating rules,
autonomy boundaries, content semantics, and the production Instagram publishing
path.

## How to use this tree

1. **Start at `refs.yml`.** It is the keystone: capability → state
   (`merged | pinned | draft | experimental | superseded`) → ref/PR/notes.
   Never assume a capability is on `main` — check its state first.
2. **Read `semantics.md` before generating or describing political content.**
   Those rules are non-negotiable and are not repeated per-file.
3. For "what can the factory do today", read `capabilities.yml`.
4. For "which project renders X on what schedule", read `projects.yml`.
5. For "what does a completed post look like / which render is the design
   reference", read `references.yml`.
6. For dataset/table/pointer questions, read `data_products.yml`.
7. For renderer tuning ("labels too small" → which constant), read
   `visuals.yml`.
8. For "what does this GitHub Actions workflow do, and is it current or
   superseded", read `workflows.yml`.
9. For publishing state and the supported production mechanism, read
   `publishing.yml`, `../docs/operations/instagram_publishing_standard.md`, and
   `../instagram/PUBLISHING.md`.
10. Per-conversation decision state lives under `sessions/<id>/`.
11. For the step-by-step procedure behind each content-generation route —
    including exactly when to stop and ask the human operator — read
    `workflows_v1.md`.

## Generated vs. hand-maintained

- **Generated** by `process/build_director_catalogue.py`, drift-checked in CI
  (`.github/workflows/director_catalogue_drift_ci.yml`): the workflow inventory
  section of `workflows.yml`, and the project-directory listing section of
  `projects.yml`. If CI is red, trust live GitHub state and regenerate rather
  than hand-editing generated sections.
- **Hand-maintained, human-reviewed**: `refs.yml`, `references.yml`,
  `semantics.md`, `capabilities.yml`, `data_products.yml`, `visuals.yml`,
  `publishing.yml`, `workflows_v1.md`, and the non-generated parts of
  `projects.yml` / `workflows.yml`.

## Size discipline

`director/` is an index, not a copy of the repository. If something is cheaply
derivable from a live GitHub or AWS read, this tree points at it rather than
duplicating it. Long-form operating detail belongs in `docs/`.

## Task routing

| Request | Route |
|---|---|
| "Generate this month's X" | Existing-series workflow — resolve project in `projects.yml`, resolve period, check data readiness, dispatch render. Procedure: `workflows_v1.md` §6.2 |
| "Let's make a post" | New-post collaboration loop — content idea, references, one prototype slide, full render, feedback, approval. Procedure: `workflows_v1.md` §6.1 |
| "Slide 3 is too crowded" | Modify-from-feedback — map to a knob in `visuals.yml`. Procedure: `workflows_v1.md` §6.3 |
| "Add this dataset" / "add a metric" | Data-product workflow — see `data_products.yml`. Procedure: `workflows_v1.md` §6.4 |
| "I need a visual like this" | Capability check in `capabilities.yml` first — extend before rebuilding |
| "Fix this broken post" | Resolve ref state in `refs.yml` first, then diagnose |
| "Schedule this" | After explicit approval of the exact reviewed output, use `.github/workflows/instagram_publish_standard.yml` with `mode=scheduled`; see `publishing.yml` and `../docs/operations/instagram_publishing_standard.md` |
| "Publish this now" | After explicit approval of the exact reviewed output, use `.github/workflows/instagram_publish_standard.yml` with `mode=immediate` |
| "Is publishing working?" | Read `publishing.yml`; use the maintenance workflow healthcheck only when live verification is needed |

## Publishing boundary

The Instagram content factory itself remains non-publishing by design:
`publication_enabled=false` / `publishing_allowed=false` must remain enforced in
factory code. A reviewed run may become `ready_for_posting`, but that does not
publish it.

Production publishing is a separate system. The standard workflow consumes the
reviewed factory artifact, promotes immutable assets to production storage,
records an approval fingerprint for the exact publication request, and then
publishes immediately or creates a one-time EventBridge schedule.

Do not re-create this mechanism with direct Meta calls, manual S3 uploads,
custom EventBridge schedules, or temporary Lambda actions. The canonical
operating contract is `docs/operations/instagram_publishing_standard.md`.

## Provenance

Built starting Phase 2 (2026-09-17) of the EirePolitic Director
implementation and updated after production Instagram immediate and scheduled
publishing were verified on 2026-09-26.
