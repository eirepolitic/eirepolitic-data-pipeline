# The four v1 workflows — concrete procedures (plan §6)

`director/README.md`'s task routing table says *which* of these a request maps to. This file says *how* each one actually runs. All four are read by the Claude Director (Sly Director project) today and are designed to be followed identically by a future ChatGPT/"High Director V2".

Built Phase 5 (2026-09-20). See `claude/eirepolitic-director-implementation-plan.md` §1.15 for the phase's full results, including the session-state wiring bug this phase found and fixed (PR #147).

---

## 6.2 Existing series — fully mechanical, no content gate

"Generate this month's Party Speech Breakdown", "render the latest polling carousel".

1. Resolve `project_id` in `projects.yml` (`directory_listing`). If it's not there, check `not_yet_a_project` — a draft-PR series (e.g. Bill Tracker) is not dispatchable yet; report its `refs.yml` state instead of guessing.
2. Resolve the period: read the project's `cadence` in `projects.yml`. `monthly` → default to `last_completed_month` unless a specific period is named. `latest` → leave the `period` input blank, the adapter resolves it itself (see `ipi_polling_factory_v1`'s `cadence_note`).
3. Check data readiness: for Oireachtas-sourced projects, the batch behind `processed/oireachtas_unified/pointers/production.json` must be `validated` — `data_products.yml`'s `pointer_resolution.enforcement` note explains why this doesn't need a separate manual check (the adapter hard-fails otherwise).
4. (Optional but recommended) Create a session branch first — `sly/session/<id>` — with a minimal `state.json`/`log.md` recording intent and the resolved project/period, so the run below has somewhere to record itself.
5. Dispatch `instagram_factory_render.yml` (`workflow_id: 362791776`) with `project_id`, optionally `period`, and `session_id` set to the branch's id.
6. Poll the run; read declarative QA off `render_summary.json`'s step output rather than parsing logs.
7. If QA fails: diagnose from the failed check's `detail` field before touching anything else — do not silently retry or loosen a check.
8. If QA passes: the review URL is `https://raw.githack.com/eirepolitic/eirepolitic-data-pipeline/previews/<slug>/index.html`. Report it. If a session was used, its run record now lives at `director/sessions/<id>/runs/<run_id>.json` on the session branch — update `state.json`'s `runs[]` and `approval_state` to match.

**No AWS-credentialed workflow push is needed for this.** Dispatching the already-merged workflow is unrestricted (plan §1.11/§1.13) — this entire workflow runs without Warren, end to end, demonstrated live in session `2026-09-20-phase5-existing-series-demo` (run `35544517116`).

## 6.1 New post — collaborative, three real gates

"Let's make a post about X."

1. **Content idea (gate — stop and ask).** What's the post about, and why now? Check `references.yml` and the completed-post archive for the closest approved example first — don't design from nothing when a precedent exists.
2. Check `capabilities.yml` — can an existing renderer/layout do this, or does it need a new one? Extending is strongly preferred over building a parallel system (see `capabilities.yml`'s `needs_review` entries for what happens when this isn't done — `instagram/media_generators/` vs. `instagram/visuals/renderers/` is an open question from exactly this kind of drift).
3. Create the session branch now, before any rendering — `intent`, `references_consulted`, and the content-idea decision go into `state.json`/`log.md` as they're made, not reconstructed afterward.
4. **Visual options (gate — stop and ask).** Don't build straight to a full carousel. Prototype ONE representative slide (the Bill Tracker lesson, plan §6.1) and get a visual-direction decision before scaling up.
5. Render the full set once direction is confirmed — same dispatch mechanics as 6.2, with `session_id` set.
6. **Approval (gate — stop and ask).** Present the review link. `approval_state` moves to `pending_review` → `approved`/`rejected` based on Warren's actual response, never assumed.
7. On approval, the session folder is the input to a completed-post archive entry (plan §3.5) — write the git-tracked summary the way Phase 2's backfill did (`completed_posts/summaries/`), noting §1.9's caveat that the S3-side archive entry is a separate, not-yet-automated step.

Everything between the three gates is mechanical; the gates themselves are not skippable or fake-able. This procedure has not been run end-to-end this phase — it will be, the first time Warren actually asks for a new post, and that run will be its real proof, not a synthetic one.

## 6.3 Modify from feedback

"Slide 3 is too crowded", "labels too small".

1. Parse the feedback into `visuals.yml`'s `feedback_to_knob` table. If the phrase doesn't map to an existing entry, that itself is worth recording — append a new row rather than solving it as a one-off and forgetting the mapping.
2. Decide one-off vs. reusable: is this knob already a `project.yml` field (reusable, safe to just edit) or a module constant (one-off until it's touched twice, per `visuals.yml`'s `feedback_to_knob_note` — then parameterise it once).
3. Make the edit — `project.yml` field or the renderer module — as its own small commit with a clear message tying it back to the feedback.
4. Re-render via the same dispatch mechanics as 6.2, same `session_id` (continuing the session, not starting a new one — this is a loop, not a fresh request).
5. Record the loop in `state.json`'s `feedback[]`: `raw` (Warren's literal words), `interpretation`, `technical_change`, and the resulting `run_id`.
6. Repeat from step 4 until approved.

Like 6.1, this is gated on a real interpretation of Warren's actual words — not run synthetically this phase. `visuals.yml`'s table already exists and is the reusable part; what's new here is the procedure wrapping it.

## 6.4 Data / capability work

"Add this dataset", "add a metric".

No new pipeline architecture (plan §6.4, §9). Route through the existing pattern: source → schema → extraction → transformation → immutable batch → validation → downstream contract → availability, using `extract/` + `political_metrics/` + `configs/*/downstream_contracts.yml` as-is. `data_products.yml`'s `metrics_and_contracts` entry is the pointer; read the contracts directly when a specific metric question comes up rather than duplicating their content here. Nothing to build or demonstrate for Phase 5 — this capability already exists and is unchanged by this phase.

---

## Why 6.1/6.3 aren't "demonstrated end-to-end" the way 6.2 is

Plan §8's Phase 5 acceptance criterion is "each of the four request types completes end-to-end and returns a clickable link." For 6.2 and 6.4 that's a mechanical claim, provable without Warren, and 6.2 was proven live this phase. For 6.1 and 6.3, "completing end-to-end" means a real content idea, a real visual-direction call, or a real feedback interpretation — decisions that belong to Warren, not something to simulate on his behalf. Faking them to tick the acceptance box would produce a plausible-looking but meaningless test. These two are validated instead by (a) the procedure above being concrete and complete, (b) the supporting data (`visuals.yml`'s knob table, `references.yml`'s archive) already existing and correct, and (c) the gates being explicit and impossible to skip. Their real proof is the first time Warren actually uses them.
