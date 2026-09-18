# director/sessions/

Per-conversation decision state (plan §3.5, Phase 4 — not built yet).

Once built, each session gets `director/sessions/<session-id>/state.json`
(intent, resolved project/period, data sources, decisions, slide plan,
runs, feedback, approval state) plus a plain-English `log.md`, committed on
a `director/session/<session-id>` branch. Warren never edits these
directly. On approval, the session folder becomes the input to the
completed-post agent summary.

This directory is a structural placeholder created in Phase 2 so the path
exists and is documented; no session-state code or schema has been built
yet. Do not write ad hoc content here before Phase 4 defines the real
schema — anything added before then risks being incompatible with what
Phase 4 lands.
