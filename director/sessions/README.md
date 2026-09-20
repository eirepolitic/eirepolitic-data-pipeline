# director/sessions/

Per-conversation decision state (plan §3.5, Phase 4 — **built 2026-09-20**).

## What this is

Every substantial Director conversation (a render, a new-post collaboration, a
feedback loop, a data-product change) gets its own session record: enough
structured and plain-English state that the *next* conversation — by Claude
or by a future ChatGPT-based Director — can reconstruct what happened without
re-reading the whole chat transcript. Warren never edits these directly.

## Where it lives

Each session gets a branch named `sly/session/<session-id>` and two files
under `director/sessions/<session-id>/`:

- `state.json` — structured state, schema below
- `log.md` — the same story in plain English, for a human or an agent that
  just wants the narrative

**Branch-naming note:** the original plan sketch (§3.5) described the branch
as `director/session/<session-id>`. In practice, this repository's GitHub
write path (the Sly Director GitHub connector) automatically prefixes every
branch it creates with `sly/`, so the real, working convention is
`sly/session/<session-id>` — adjusted to match what the tooling actually
produces. `director/sessions/<session-id>/` (the *file path*, not the
branch name) is unaffected and still matches the original plan.

Session branches are **not** merged into `main` — they are read directly by
`ref` (the same way preview branches are), so `main` doesn't accumulate one
folder per conversation forever. A session only needs merging if its output
becomes something Warren-facing and permanent (e.g. `references.yml`'s
`canonical_visual_reference`, `completed_posts/summaries/`), and that's a
deliberate, separate edit to the relevant hand-maintained `director/*.yml`
file — not something session state does automatically.

## `state.json` schema

```jsonc
{
  "session_id": "YYYY-MM-DD-short-slug",       // matches the branch suffix
  "branch": "sly/session/<session_id>",
  "created_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC",                 // bump on every write
  "intent": "free text — what was actually asked for, in plain terms",

  "resolved_project": "project_id or null",     // from projects.yml, once known
  "resolved_period": "YYYY-MM or null",

  "data_sources": ["free-text pointers into S3/pointer files actually consulted"],
  "references_consulted": ["director/*.yml paths, archive entries, canonical refs read"],

  "decisions": [
    {
      "question": "what was actually being decided",
      "choice": "what was chosen",
      "rationale": "why",
      "timestamp": "ISO-8601 UTC"
    }
  ],

  "slide_plan": [],       // only populated by the new-post workflow (§6.1) —
                           // one entry per planned slide before the first render
  "visual_decisions": {}, // free-form knob choices keyed by knob name (§3.7),
                           // e.g. {"legend_variant": "underneath"}

  "runs": [
    {
      "run_id": "GitHub Actions run id, as a string",
      "workflow": "instagram_factory_render.yml",
      "project_id": "...",
      "period": "the period INPUT given, or null if left blank",
      "resolved_period": "the period the run actually resolved to",
      "review_url": "https://raw.githack.com/.../previews/<slug>/index.html",
      "commit": "head_sha of the run",
      "qa_all_passed": true,
      "dispatched_by": "warren | claude",   // see the dispatch-vs-push note below
      "timestamp": "ISO-8601 UTC"
      // optional: "purpose" for a run made for a specific reason
      // (regression check, feedback re-render, etc.) rather than a routine one
    }
  ],

  "feedback": [
    {
      "raw": "Warren's literal words",
      "interpretation": "what it was taken to mean",
      "technical_change": "what was actually changed as a result, or 'No plan/code change' if it was a style/process note rather than a build change",
      "run_id": "the run this produced, if any, else null"
    }
  ],

  "approval_state": "in_progress | pending_review | approved | rejected"
}
```

All fields are present in every `state.json`, even when empty (`[]`/`{}`/`null`)
— a consumer should never have to check for a missing key, only an empty one.

## `log.md`

No fixed schema — a short, plain-English narrative of the session written
for a human or another agent to read top-to-bottom, in the tone of a status
update: what was asked, what was found, what was built or run, what it
proved, what's left. It should stand on its own without needing `state.json`
open alongside it, and `state.json` should stand on its own without needing
`log.md` — they're the same information in two shapes for two different
kinds of reader, not one being a subset of the other.

## Reading a past session

A future Director reads the branch directly (`ref: sly/session/<id>`) via
the GitHub connector's `read_file` — no merge, no PR, no special tooling.
`director/refs.yml` and `references.yml` are the durable, hand-reviewed
knowledge base; a session record is context for *how* something in them got
that way, not a second source of truth for what's currently approved.

## A worked example

`sly/session/2026-09-20-phase3-close-out` is the first real session record,
written by this same Phase 4 change: it documents the Phase 3 close-out
(PR #145 merge, the Batch 1 handoff and results, and the July 2026
regression dispatch that closed Phase 3's acceptance criterion). Read it as
a concrete example of a filled-in `state.json`/`log.md` pair, including the
`dispatched_by: "claude"` case — a routine dispatch of an
*already-merged* AWS-credentialed workflow doesn't need Warren (only
*pushing a new one* does — see plan §1.11/§1.13), so most `runs` entries a
future session writes will legitimately have `dispatched_by: "claude"`.

## What's still open (deliberately, not an oversight)

No CI validates `state.json` against this schema yet — there's no
`director_sessions_schema_ci.yml` the way `director_factory_v1_identity_ci.yml`
guards the frozen v1 files or `director_catalogue_drift_ci.yml` guards the
generated inventories. Session branches aren't part of any push/PR to
`main`, so the existing drift-CI pattern doesn't naturally extend to them
without inventing a new trigger shape. This is left for a future session to
add if malformed session state ever actually causes a problem, rather than
built speculatively now.
