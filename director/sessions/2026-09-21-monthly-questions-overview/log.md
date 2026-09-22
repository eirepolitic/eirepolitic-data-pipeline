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

## Build: prototype slide (2026-09-21, first build)
Built `instagram/projects/pq_monthly_overview_v1/` on branch `feature/pq-monthly-overview-v1` (commit `4bf3ce5`), modeled on `party_issue_monthly_profile_v2`:

- **adapter.py** loads `silver_questions` + membership/party/constituency/member tables from the validated production batch (`written-pq-answers-20260905-1`), filters to July 2026, **dedupes duplicate `question_id` rows** (keep-first) before eligibility/ranking rather than hard-failing like `process/political_metrics_question_commission.py` does, computes eligible-TD question counts via the existing `political_metrics.calculators.questions` layer, and ranks the top 10.
- Reuses `instagram/visuals/renderers/horizontal_bar.py` and the `title_text_media_v1.json` outer layout **unmodified** — no new renderer.
- Added a `normalize_result()` shim for the new project_id in `instagram/factory/normalize.py` (not part of the frozen v1 file set, so free to extend) so the project runs through the existing generic `instagram_factory_render.yml` workflow.
- `publication.enabled: false` throughout; QA declares `expected_slide_count: 1`, `require_publication_disabled`, `require_review_state`.

Dispatched `instagram_factory_render.yml` (run `35634477456`) — **passed on the first attempt**.

Rendered top 10 matches the discovery-phase spot-check. **Visual issue found:** the `Name (Party)` label format produces a 3-line wrap for at least one TD ("Conor D. McGuinness (Sinn Féin)"), overlapping the neighbouring rows — flagged for Warren's visual-direction decision.

## Warren's feedback (round 1)
"Your assessment is correct. The chart looks good. However, the labels are too long and therefore overwrap the rows." Asked for two comparable versions instead of picking a fix blind: (A) a two/one-letter party acronym in the label, e.g. "Conor McGuinness (SF)"; (B) bars colored by party with a legend detailing which party is which color.

## Build: two visual-direction variants (2026-09-21, this build)

**Option A — `top_askers_v1_acronym`**: `horizontal_bar.py` used completely unmodified. Label format changed from `Name (Party)` to `Name (XX)`, a first-cut 2-4 letter acronym per party (`PARTY_ACRONYM` dict in `adapter.py`; not derived from any registry — worth promoting to a proper reference file alongside `configs/reference/party_assets_v1.csv` if this direction is picked). This directly fixes the wrap/overlap since the labels are now short enough to stay on one line for all 10 TDs.

**Option B — `top_askers_v2_colored_legend`**: new `instagram/visuals/renderers/horizontal_bar_grouped.py`. `horizontal_bar.py` is in the CI-frozen v1 file set (byte-identity-checked against `386b933` by `director_factory_v1_identity_ci.yml`) so it can't be edited for a one-off feature — this new sibling module imports its proven label-wrap/clip/truncation-detection helpers unchanged and adds only what it doesn't do: one bar color per party (`ax.barh(..., color=bar_colors)`) plus a `fig.legend(...)` mapping color to party. Colors are the Anthropic dataviz skill's validated default categorical palette (dark-mode steps) — abstract per-party identity slots, not official party brand colors — validated with `validate_palette.js` against this project's actual `#0f2f24` chart background: **ALL CHECKS PASS**, one WARN on the green slot's contrast (2.92:1), mitigated per the skill's own rule by the always-present value labels and legend text (the required secondary encoding).

`adapter.py`'s `generate()` now renders both slides in one call; `project.yml` updated to `expected_count: 2` / `qa.expected_slide_count: 2` with both slide definitions.

### Debugging this build (three failed attempts before it rendered clean)
1. **Run `35642966347` — ImportError.** `instagram_factory_render.yml`'s injection step only copies a fixed list of `instagram/factory/*.py` files plus the requested project directory into the pinned/frozen factory worktree — it had no way to know about the new `horizontal_bar_grouped.py` renderer module, so importing it in the pinned worktree failed. **Fix:** edited the workflow (existing file, not its first appearance, so within normal Director authority — no AWS-secrets-first-appearance or publish gate involved) to also copy any `instagram/visuals/renderers/*.py` file the project ships that isn't already in the pinned worktree, guarded so it never touches the frozen renderers themselves.
2. **Run `35643254682` — failed again, no visibility why.** GitHub's own log-download endpoint isn't reachable from this environment's network (egress proxy blocks `results-receiver.actions.githubusercontent.com`), so the failure was a black box. **Fix:** added a diagnostics step (`if: always()`) that tees the render step's stdout/stderr and pushes it to `director/sessions/<id>/runs/<run_id>-render-output.log` on this session branch, readable via the GitHub connector.
3. **Run `35645238352` — RuntimeError: `legend_clipped_to_figure` for option B.** The diagnostics log pinpointed it immediately: the legend used full party names (some quite long, e.g. "People Before Profit-Solidarity") across up to 4 columns on a narrow 1032px canvas — comfortably wider than the figure on plausible party combinations, even though there was always enough vertical room. **Fix:** capped the on-image legend label length (18 characters + ellipsis; full names still live in the manifest's `group_legend_labels`), dropped to 2 legend columns, trimmed the legend font size and spacing, and gave the legend band a bit more vertical headroom.
4. **Run `35645890623` — success.** Both slides rendered and were visually verified (downloaded both PNGs via the preview branch and inspected them directly): option A shows all 10 acronym labels on one line with no overlap; option B shows 6 distinct, legible party colors with a clean 2-column legend, no clipping. The run's diagnostics-log push worked, but the follow-up push of `run_manifest.json` (added to also capture the exact dedupe count, see below) looked in the wrong directory and found nothing.
5. **Run `35646123979` — re-run purely to capture the manifest** (no code change to the adapter or either renderer, just the workflow's manifest-search path fixed from the pinned worktree to `$GITHUB_WORKSPACE`). Confirms the same output as run `35645890623`. Manifest: `director/sessions/2026-09-21-monthly-questions-overview/runs/35646123979-run-manifest.json`.

### Dedupe count — captured, doesn't match the kickoff note, still unresolved
This run's `data_quality.question_dedupe`: `raw_row_count=8911`, `deduped_row_count=8911`, **`duplicate_question_id_count=0`**, `duplicate_row_count_removed=0`. The kickoff instruction said "July has 11 duplicate question IDs" — this run found none. Not root-caused; flagged to Warren alongside the option A/B review link. Warren has not yet responded to this specific point — still open.

## Warren's feedback (round 2) — visual direction decided
Sent Warren the review link with both options and the dedupe discrepancy. His reply: "I prefer option B. One modification should be that instead of two columns of three rows for the legend, make it three columns, two rows, and fill in the remaining space by making the chart slightly larger."

## Build: option B legend/chart tuning per Warren's pick (2026-09-22)
Edited `horizontal_bar_grouped.py` on `sly/feature/pq-monthly-overview-v1`:
- `LEGEND_MAX_COLUMNS` 2 → 3 (Warren's ask: 3 columns × 2 rows instead of 2 × 3, for the same 6 parties).
- `PLOT_HEIGHT` 0.66 → 0.70 — a 2-row legend needs less vertical room than the earlier 3-row layout, so the freed space (now a 0.16 figure-fraction gap above the plot, vs. 0.185 before) goes into the chart itself, per Warren's "fill in the remaining space by making the chart slightly larger."
- `LEGEND_MAX_LABEL_CHARS` 18 → 13, plus tighter `columnspacing`/`handletextpad` — three columns leaves less horizontal room per column on the fixed 1032px canvas than two did, and the earlier real failure (run `35645238352`) was a *horizontal* legend overflow (at `ncol=4`, uncapped names), not a vertical one — so the horizontal safety margin, not the vertical one, is what needed tightening for the extra column.

One slip along the way, caught and fixed before it could matter: the first commit for this change (`9266e575`) accidentally pushed placeholder text instead of the real file content — caught immediately by reading the file back from the branch before dispatching anything, and corrected in a follow-up commit (`8e304045`) before any render ran against it.

Dispatched `instagram_factory_render.yml` (run `35789255000`) — **passed on the first attempt**. QA PASS (`expected_slide_count: 2`), option B's readability block shows `legend_clipped_to_figure: false`, `legend_group_count: 6`, zero clipped/truncated labels. Downloaded the rendered PNG and inspected it directly: clean 3-column/2-row legend (Fianna Fáil / Fine Gael / Social Democrats on row one, Sinn Féin / Independent Ireland / Aontú on row two), no overlap or clipping, and the chart is visibly larger than the prior 2-col/3-row build.

Review page (still shows both slides — option A is unchanged from run `35646123979`; option B is now the final, approved layout): https://raw.githack.com/eirepolitic/eirepolitic-data-pipeline/previews/pq-monthly-overview-v1-options/index.html

## Build notes / open issues
- **Dedupe count still unresolved**: this run's actual count is 0, contradicting the kickoff note of 11 — still needs Warren's input on which is right, or whether it no longer matters now that the pipeline dedupes defensively either way.
- Department labels need a display-name mapping — not yet needed (no slide built yet uses departments).
- A party-leader list needs creating with its sources — needed for the fewest-askers slide, not yet built.
- The 28 July date concentration needs explaining before any 'busiest day' framing.
- Option B's legend still ellipsizes any party name over 13 characters (tightened from 18 for the 3-column layout) — cosmetically minor but worth a cleaner truncation (e.g. a short-name lookup instead of a character cut) at some point; full names remain correct in the run manifest regardless.
- Option A (`PARTY_ACRONYM` dict in `adapter.py`) is now the non-selected variant — kept rendering alongside option B for reference/comparison, but no further investment planned unless Warren changes his mind.
- `instagram_factory_render.yml` now also injects any project-added `instagram/visuals/renderers/*.py` module and pushes render diagnostics + the run manifest to the session branch on every run — this is a general improvement, not specific to this project, and should keep working for future projects that add their own non-frozen renderer variants.
- **Visual direction is now settled (option B, 3-col/2-row legend, enlarged chart).** Next: scale this same renderer/treatment approach to the remaining six slides in the agreed content spec, then the full-carousel render and final-approval gates (workflows_v1.md §6.2–6.3, both Warren's calls).

## Status
Content gate passed. Visual-direction gate passed — Warren picked option B with the 3-col/2-row legend + enlarged-chart modification, confirmed rendering clean (run `35789255000`, QA PASS, `legend_clipped_to_figure: false`). **approval_state: visual_direction_approved_pending_full_carousel** — next step is building out the remaining six slides using this settled visual treatment, then the full-carousel render and Warren's final approval.
