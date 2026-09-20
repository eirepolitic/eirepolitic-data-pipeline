# Session log — 2026-09-20-phase5-existing-series-demo

Phase 5 (plan §5/§6.2) self-test: prove the existing-series workflow runs end-to-end through the actual routing table in `director/README.md`, and that a session-tagged dispatch correctly records its run on this session branch (not `main` — see PR #147, which fixed exactly this wiring gap moments before this session branch was created).

Route taken, per `director/README.md`'s task routing table ("Generate this month's X" -> existing-series workflow, plan §6.2): resolve project from `projects.yml` -> resolve period -> dispatch `instagram_factory_render.yml` with `session_id` set -> declarative QA -> review link recorded back here.

`ipi_polling_factory_v1` was chosen over `party_issue_monthly_profile_v2` because it has no monthly content decision attached (cadence: latest, dispatched on-demand) — this keeps the demo honest as a mechanical proof rather than a stand-in for a real editorial choice only Warren can make.

See `state.json` for the structured run record once the dispatch completes.
