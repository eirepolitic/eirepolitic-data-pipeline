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

## Build notes / open issues (as of end of previous build)
- **Dedupe count still unresolved**: this run's actual count is 0, contradicting the kickoff note of 11 — still needs Warren's input on which is right, or whether it no longer matters now that the pipeline dedupes defensively either way.
- Department labels need a display-name mapping — not yet needed (no slide built yet uses departments).
- A party-leader list needs creating with its sources — needed for the fewest-askers slide, not yet built.
- The 28 July date concentration needs explaining before any 'busiest day' framing.
- Option B's legend still ellipsizes any party name over 13 characters (tightened from 18 for the 3-column layout) — cosmetically minor but worth a cleaner truncation (e.g. a short-name lookup instead of a character cut) at some point; full names remain correct in the run manifest regardless.
- Option A (`PARTY_ACRONYM` dict in `adapter.py`) is now the non-selected variant — kept rendering alongside option B for reference/comparison, but no further investment planned unless Warren changes his mind.
- `instagram_factory_render.yml` now also injects any project-added `instagram/visuals/renderers/*.py` module and pushes render diagnostics + the run manifest to the session branch on every run — this is a general improvement, not specific to this project, and should keep working for future projects that add their own non-frozen renderer variants.

## Build: full 7-slide carousel (2026-09-22, this build)

Warren: "leave as is for the dedupes. We can investigate that in the future. It's not a big enough proportion to be concerned about. Let's proceed now to the other slides." — authorized building the remaining six slides using the settled top_askers treatment.

Built the remaining slides in `adapter.py`: headline (text), fewest_askers (chart), party_per_td (chart), departments (chart), answers (chart, from written_question_answer_bridge/sections), methodology (text). Added a new non-frozen layout `instagram/templates/layouts/text_block_v1.json` (title + up to 6 independently-bound text lines + footer) for the three text-only slides, since `template_renderer.py`'s `text_lines()` collapses embedded newlines within a single placeholder — six separate placeholders were needed instead of one multi-line string.

Office-holder/party-leader exclusion (from the originally agreed content spec) and the median-comparison framing for `fewest_askers` were **not** implemented — no reference list of current office-holders/party leaders exists anywhere in the pipeline (confirmed by searching the repo), so rather than fabricate one, this was disclosed as a data gap: a caveat on the slide, in the methodology slide, and in the run manifest's `caveats` list. `fewest_askers` as built is a plain ascending ranking of the bottom 10 eligible TDs seated the full period, with no office-holder exclusion and no median comparison — see "Issue found" below for why this matters more than expected.

### Debugging this build (four failed runs before a clean full carousel)
1. **Run `35795646995` — failure.** Dispatched with `factory_ref: "sly/feature/pq-monthly-overview-v1"` by mistake. `factory_ref` pins a *separate*, hash-verified "approved" worktree to a fixed commit (`386b933`) — it is not where the project's own code comes from (that's the dispatch's `ref`, which was already set correctly). Broke the worktree's corner-PNG hash verification step. **Fix:** redispatched without `factory_ref` at all, letting it default to `386b933`.
2. **Run `35795811070` — failure.** `botocore.errorfactory.NoSuchKey` inside `load_csv_tables()`: `written_question_answer_sections`/`written_question_answer_bridge` are not present in the *current* production batch (`written-pq-answers-20260905-1`), even though the schema is contract-approved (`configs/political_metrics/written_question_answers.yml`). **Fix:** restructured `adapter.py` to load these two tables in a separate, `try/except`-wrapped call; on failure, the "answers" slide degrades to an honest text placeholder (no fabricated figures) instead of aborting the whole 7-slide render, with the exact error recorded in `caveats`. Trimmed `project.yml`'s hard `required_tables` back to the 5 core silver_* tables.
3. **Run `35796492112` — failure.** After the fix above, `RuntimeError: No Dáil-eligible questions found for 2026-08`. `period: last_completed_month` resolved to August 2026 given today's date (2026-09-22), but — as already noted in this session's own discovery phase — **August 2026 is not in production data yet**; July is still the last month with real data. Not a code bug. **Fix:** for this verification render, dispatched with an explicit `period: "2026-07"` instead (the same month already analyzed and shown to Warren), rather than letting the default resolve into a known data gap.
4. **Run `35796641892` — failure.** `FileNotFoundError: instagram/templates/layouts/text_block_v1.json`. The workflow's injection step copies new `instagram/visuals/renderers/*.py` modules a project adds, but had no equivalent for a new `instagram/templates/layouts/*.json` file — `text_block_v1.json` only existed on the feature branch, not in the pinned worktree. **Fix:** edited `instagram_factory_render.yml` again (existing file, not first appearance) to add the same "copy only if it doesn't already exist in the pinned worktree" pattern for `instagram/templates/layouts/*.json`.
5. **Run `35796810745` — success.** All 7 slides rendered, declarative QA passed (`expected_slide_count: 7`, publication disabled, review state pending). Numbers match the July discovery-phase spot-check exactly (8,911 questions, Ken O'Flynn 418, etc.). `answers` correctly degraded to the placeholder (confirms the answer tables really are absent from the current batch, not a bug). Manifest: `director/sessions/2026-09-21-monthly-questions-overview/runs/35796810745-run-manifest.json`.

### Visual verification — 6 of 7 slides clean, one real issue found
Downloaded and visually inspected all 7 rendered PNGs from the preview branch (`previews/pq-monthly-overview-v1-full-carousel`):

- **headline** (new text_block_v1.json layout, first live use) — clean, no clipping or wrap collisions, all 5 stat lines legible.
- **top_askers** — unchanged from Warren's approved design (run `35789255000`), renders identically clean.
- **fewest_askers** — **broken: renders as "No data available" instead of a chart.** See "Issue found" below.
- **party_per_td** — clean 11-row chart, no clipping, party names wrap sensibly ("People Before Profit-Solidarity" wraps to 2 lines without overlap).
- **departments** — clean 8-row chart, no issues.
- **answers** — clean placeholder text slide (matches the headline/methodology text-slide style), correctly explains the data gap.
- **methodology** (new text_block_v1.json layout) — clean, all 5 bullet points + dedupe note legible, no clipping.

### Issue found: fewest_askers has no usable content, not just a rendering bug
All 10 of July 2026's lowest-count eligible TDs (seated the full month) asked **exactly zero** questions (Alan Dillon, Charlie McConalogue, Colm Brophy, Dara Calleary, Darragh O'Brien, Emer Higgins, Helen McEntee, Hildegarde Naughton, Jack Chambers, James Browne — confirmed in the run manifest's `fewest_askers` array). 

The frozen `instagram/visuals/renderers/horizontal_bar.py` (byte-identity-checked, cannot be edited) treats this as "no data": `empty_state = not clean_rows or max(values, default=0.0) <= 0` — it can't distinguish "no rows were returned" from "the rows are real but every value is 0", so it shows the "No data available" placeholder instead of the (10-real-record) ranking.

But even setting that renderer quirk aside: a bar chart has nothing to show when every bar is zero-length. All 10 names on this list read like office-holders and senior figures (McEntee, Chambers, O'Brien, Calleary, McConalogue, Naughton...) — exactly the pattern the original discovery phase already flagged ("32 of 40 office-holders asked none") and the originally *agreed* content spec was designed around: exclude office-holders/party leaders, and compare each remaining TD's count to the median ("typical TD") rather than a raw ascending ranking. That design was never built this session — only a caveat ("does not exclude office-holders...") was added in its place, on the reasoning that no office-holder/party-leader reference list exists anywhere in the pipeline to build it from. That reasoning still holds (no such list exists, confirmed by search), but the consequence — a chart with literally nothing to show most months, since office-holders structurally cluster at zero — was not anticipated until this render made it visible.

**This is not something to silently redesign or route around** (e.g. faking a non-zero value to force a bar to draw, which would misrepresent the data). It's the content/visual-direction gate (`workflows_v1.md` §6.1, Warren's call) surfacing a real design question: how should `fewest_askers` work when the bottom of the ranking is dominated by a tie at zero, which — given office-holders structurally ask few or no PQs — looks likely to recur most months, not just this one? Options include (not decided, listed for Warren to choose from or propose his own): (a) build the office-holder/party-leader exclusion using a hand-maintained list Warren supplies, restoring the originally agreed design; (b) render this case as a text slide (names only) rather than a chart when there's a mass zero-tie; (c) drop to a shorter list of only the non-zero low askers, if any; (d) something else entirely.

### Status
Content gate passed. Visual-direction gate passed (top_askers). Full 7-slide carousel now builds and QA-passes, 6 of 7 slides visually clean. **Not ready for the final-approval gate (§6.1 step 6)** — `fewest_askers` needs a design decision from Warren before this carousel can be presented for approval. **approval_state: full_carousel_built_pending_fewest_askers_design_decision.**

## Build: fewest_askers office-holder exclusion + 25%-threshold (2026-09-22/23, this build)

Warren, resuming after the AWS connection was restored: "Hopefully the AWS connection has been reconnected now. Please check in terms of developing further. Let's take it slide by slide, iterate one by one, and then we can determine whether the file is ready." Checked the AWS connector live (`GetCallerIdentity` succeeded) and re-confirmed the production data situation was genuinely unchanged (same pointer, same batch, August still absent) rather than assuming it from the prior build's notes.

Then, on the `fewest_askers` issue flagged at the end of the previous build, Warren replied: "Without the folders should be identifiable from the data. If it's not tracked anywhere in our current data set, then we need to flag that as a missing component that we need to add in the future. And we need to remove it from our current scope. As for those who didn't ask any questions, what proportion of TDs asked no questions? If it's less than 25%, then we should just list essentially all the TDs that did not ask a question." "Without the folders" was read as a transcription artifact for "Office-holders" — the only sensible reading given it directly follows the office-holder-exclusion discussion at the end of the last build. This was not explicitly re-confirmed with Warren before acting on it; flagged here so he can correct it if wrong.

### Checked live: office-holder data (correcting an earlier, incomplete claim)
The previous build's caveat said "no office-holder/party-leader reference list exists anywhere in the pipeline" — based on a repo search, not a live data check. Reading `silver_member_offices` directly from the current production batch (`written-pq-answers-20260905-1`) via the AWS connector found this was wrong for the office-holder half: the table exists, 123 rows, 68 `is_current=True`, and its distinct `office_name` values are all ministerial/parliamentary-office roles (`Minister*`, `Minister of State*`, `Taoiseach`, `Ceann Comhairle`, `Leas-Cheann Comhairle`) — no party-leader roles. Manually computing "office-holder overlapping July 2026" (an `office_start`/`office_end` overlap test) gave **40 office-holders**, which matches the discovery-phase note ("32 of 40 office-holders asked none") exactly — confirming both the table and the count are right. Party-leader status genuinely is not tracked anywhere in the pipeline (re-confirmed) — per Warren's instruction, this stays out of scope as a caveat/future-work item rather than a hand-maintained guess.

Attempted to also compute the exact July zero-question proportion by hand in the AWS sandbox, to sanity-check before touching code, but `silver_questions.csv` (~104MB, 121,355 rows) reliably timed out on every approach tried (full parse, streamlined parse, S3 Select — not permitted on this bucket, and a range-limited fetch) — abandoned and left the actual computation to the render pipeline itself, which has already proven it can process this file quickly.

### Code changes (`sly/feature/pq-monthly-overview-v1`)
- `adapter.py`: added `_office_holder_codes()` (member_codes holding a ministerial-type office overlapping the period, from `silver_member_offices`) and `_chunk_into_lines()` (packs a list of formatted name strings into `text_block_v1.json`'s 6 line slots, merging overflow into the last slot rather than dropping it). Rewrote `fewest_askers`: excludes office-holders from the full-period-seated roster, computes what proportion of the remainder asked exactly zero questions, and branches — under 25%, renders every such TD by name as a text-list slide (reusing `text_block_v1.json`); at or above 25%, falls back to the previous bottom-10-by-count chart (now also office-holder-excluded), flagged in its own caveat as untested since it wasn't exercised. Added a `fewest_askers_stats` block to the run manifest (`office_holders_excluded_count`, `eligible_pool_after_exclusion`, `zero_question_count`, `zero_question_proportion`, `rendered_as`) for transparency.
- `project.yml`: added `silver_member_offices` to `required_tables`; updated the `fewest_askers` slide description and the project's `purpose` text to match.
- Methodology slide's `fewest_askers` line rewritten to describe the office-holder exclusion and the party-leader gap accurately (dropped the now-superseded median-comparison language).

### Verification render
Dispatched `instagram_factory_render.yml` (ref `sly/feature/pq-monthly-overview-v1`, period `2026-07`, same `preview_slug` `pq-monthly-overview-v1-full-carousel`) — **run `35804522996`, success on the first attempt.**

`fewest_askers_stats` from the manifest: `office_holders_excluded_count: 40`, `eligible_pool_after_exclusion: 134`, `zero_question_count: 1`, `zero_question_proportion: 0.0075`, `rendered_as: "text_list_all_zero_askers"`. For July 2026, only **1 of 134** eligible (non-office-holder, seated-full-month) TDs asked zero questions — Richard O'Donoghue (Independent Ireland) — well under the 25% threshold, so the text-list branch fired.

All 7 slides re-downloaded from the preview branch and inspected directly:
- **fewest_askers** — now renders clean and legible: "1 of 134 non-office-holder TDs seated all of July 2026 (1%) asked no parliamentary questions: Richard O'Donoghue (II)." No more "No data available" failure. (Visually sparse — a lot of empty space below the one name — because there's genuinely only one name to show this month; not a defect, just what a 1-name result looks like in a 6-slot layout.)
- **headline, top_askers, party_per_td, departments, answers (placeholder), methodology** — all re-confirmed clean and unaffected by this change; figures unchanged from run `35796810745` (same source batch and period, only `fewest_askers` logic changed).

The bottom-10-by-count chart fallback (for a month where the zero-question proportion is at or above 25%) was not exercised by this run and remains untested against real data — flagged in the run's caveats for a future month to confirm.

### Status
`fewest_askers` design decision resolved and verified working end-to-end for July 2026. All 7 slides now render clean. Session docs (`state.json`, this log) updated to record the live office-holder-data correction, the code change, and the verification run. **Not yet marked approved** — Warren asked to go "slide by slide, iterate one by one" and had only seen slide 1 (headline) presented before the conversation moved to this `fewest_askers` fix; his explicit sign-off on each slide, and the carousel overall, is still outstanding. **approval_state: fewest_askers_rebuilt_and_verified_pending_warren_slide_by_slide_review.**
