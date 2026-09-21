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

## Build notes / open issues
- The existing commissioning script hard-fails on duplicate question_id. July has 11 duplicates, so a decision or fix is needed.
- Department labels need a display-name mapping.
- A party-leader list needs creating with its sources.
- The 28 July date concentration needs explaining before any 'busiest day' framing.
- Existing instagram/visuals/renderers/horizontal_bar.py covers most slides. Per §6.1, one prototype slide comes first, then a visual-direction gate with Warren.

## Status
Content gate passed. Build not started. approval_state: in_progress.
