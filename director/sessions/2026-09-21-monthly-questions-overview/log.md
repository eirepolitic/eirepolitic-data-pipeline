# Session 2026-09-21 — Monthly Questions Overview

## What was asked
Warren wants a new recurring monthly Instagram post: a high-level overview of Dáil Parliamentary Questions asked and answered in the previous month. It is one of three new monthly post types, alongside a Monthly Speeches Overview and a Monthly Bill Summary.

## Discovery (done)
The pipeline already has a questions metrics layer (political_metrics/calculators/questions.py, a metrics catalogue, and a commissioning script) plus parsed written-answer foundations (answer status, answering minister, grouped answers, and a keyword-based referral flag).

July 2026 was analysed live from production batch written-pq-answers-20260905-1 (validated). August 2026 is not in production yet.

July 2026 findings:
- 8,911 questions (8,752 written, 159 oral); 133 TDs asked; June was 7,568.
- 4,313 (48%) carry the date 28 July, after the Dáil rose. The cause has not been verified.
- Top askers: Ken O'Flynn 418, Pa Daly 322, John Brady 254. The list is volatile month to month.
- 32 of 40 office-holders asked none. Eight asked some (e.g. two Ministers of State, the Leas-Cheann Comhairle); office dates should be checked.
- Excluding office-holders, the median TD asked 40 and the mean was about 66. Only one non-office TD asked zero.
- Health was about 24% of questions; 57% of Health answers were referred for direct reply (mostly to the HSE).
- 103 questions (1.2%) had 'reply not received'.

## Agreed content
Seven slides: Headline; Most questions submitted (top 10); Fewest questions submitted (bottom 10, with gap to the median); Questions per TD by party; Departments asked most; What happened to the answers; Methodology.

The constituency slide was dropped. Oral questions and office-holders asking become notes.

The fewest-questions slide excludes office-holders, party leaders (via a hand-maintained list) and TDs not seated all month. It uses a single month and compares each TD against the median ('typical TD'). The wording is direct but factual. Absences are ignored for now; Warren will add absence data later.

## Build: prototype slide (2026-09-21, this build)
Built `instagram/projects/pq_monthly_overview_v1/` on branch `feature/pq-monthly-overview-v1` (commit `4bf3ce5`), modeled on `party_issue_monthly_profile_v2`:

- **adapter.py** loads `silver_questions` + membership/party/constituency/member tables from the validated production batch (`written-pq-answers-20260905-1`), filters to July 2026, **dedupes duplicate `question_id` rows** (keep-first) before eligibility/ranking rather than hard-failing like `process/political_metrics_question_commission.py` does, computes eligible-TD question counts via the existing `political_metrics.calculators.questions` layer, and ranks the top 10.
- Reuses `instagram/visuals/renderers/horizontal_bar.py` and the `title_text_media_v1.json` outer layout **unmodified** — no new renderer.
- Added a `normalize_result()` shim for the new project_id in `instagram/factory/normalize.py` (not part of the frozen v1 file set, so free to extend) so the project runs through the existing generic `instagram_factory_render.yml` workflow.
- `publication.enabled: false` throughout; QA declares `expected_slide_count: 1`, `require_publication_disabled`, `require_review_state`.

Dispatched `instagram_factory_render.yml` (run `35634477456`, from `sly/feature/pq-monthly-overview-v1`, period `2026-07`, `session_id` set) — **passed on the first attempt**. Review page: https://raw.githack.com/eirepolitic/eirepolitic-data-pipeline/previews/pq-monthly-overview-v1/index.html

Rendered top 10 matches the discovery-phase spot-check (Ken O'Flynn 418, Pa Daly 322, John Brady 254), confirming the dedupe/eligibility logic reproduces the same counts as the earlier manual analysis.

**Visual issue found:** the `Name (Party)` label format produces a 3-line wrap for at least one TD ("Conor D. McGuinness (Sinn Féin)"), and horizontal_bar.py doesn't add extra row spacing for wrapped labels, so it visually overlaps the neighbouring rows. Declarative QA (bbox-vs-figure clipping) doesn't catch row-to-row overlap, so this passed QA but is a real defect — flagged for Warren's visual-direction decision along with the label format itself, since party_issue_monthly_profile_v2's renderer was tuned against short one-line issue labels, not `Name (Party)` combinations.

## Build notes / open issues
- Duplicate `question_id` handling: **resolved** for this build — deduped (keep-first), exact counts recorded in the run's `run_manifest.json` (`data_quality.question_dedupe`), currently only retained in the run's 30-day GitHub Actions artifact, not yet copied into this session folder.
- Department labels need a display-name mapping — not yet needed (no slide built yet uses departments).
- A party-leader list needs creating with its sources — needed for the fewest-askers slide, not yet built.
- The 28 July date concentration needs explaining before any 'busiest day' framing.
- **Open for Warren's visual-direction decision:** the top-askers label overlap above, and whether the `Name (Party)` bar-label format and current palette/layout are right before building the other six slides.

## Status
Content gate passed. Prototype slide built and rendered (run `35634477456`, QA PASS). **approval_state: pending_visual_direction** — waiting on Warren's review of the link above before scaling to the full seven-slide carousel.
