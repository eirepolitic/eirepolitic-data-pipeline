# First Stage Bill Tracker development session — 2026-10-03

The task is to develop the First Stage variant of the merged `bill_tracker_factory_v1` Instagram series, beginning with a live inventory refresh and a single representative Bill-slide prototype. No publishing or scheduling is authorised.

## Repository review

Read the required Director operating documentation and inspected the merged Bill Tracker project, its current content, adapter, renderer, generic factory workflow, and the two completed Enacted-post summaries. Current repository truth confirms `bill_tracker_series` is merged and the existing factory must be extended rather than replaced.

The canonical Enacted family establishes the 1080×1350 geometry, dark green/cream/gold palette, corner ornaments, bounded title fitting, source/footer treatment, shared-fit comparable typography, and left-to-right legislative-process timeline. The original adapter is Enacted-specific: it accepts `post1`/`post2`, requires exactly three Bills, emits two slides per Bill, and assumes nine slides. The session branch adds a stage-aware adapter path for review-only First Stage development while preserving the existing Enacted behavior.

## Production pointer and live reconciliation

The validated production pointer currently resolves to batch `written-pq-answers-20260905-1`, promoted on 6 September 2026. Its manifest is validated and the relevant Bill, stage, sponsor, debate, speech, division and member-vote tables were inspected. For 2026, the stale snapshot exposes seven Bills whose latest recorded stage row is First Stage.

That production snapshot is not editorially current on 3 October 2026. A live Houses of the Oireachtas legislation refresh and official Bill/debate-document cross-check yields six Bills currently at First Stage across the current session:

1. Protection of Tenants' Deposits Bill 2025 — Bill 4/2025 — Dáil Éireann — Paul Murphy, Richard Boyd Barrett, Ruth Coppinger.
2. Energy Poverty Reduction (Use of Surplus Renewable Energy) Bill 2025 — Bill 6/2025 — Dáil Éireann — Paul McAuliffe.
3. Criminal Justice (Trespass on Land) Bill 2025 — Bill 55/2025 — Dáil Éireann — Carol Nolan.
4. Criminal Law (Adult Safeguarding) Bill 2026 — Bill 44/2026 — Dáil Éireann — David Cullinane, Matt Carthy, Natasha Newsome Drennan.
5. Defence (Amendment) Bill 2026 — Bill 63/2026 — Dáil Éireann — Minister for Defence.
6. Railway Safety Bill 2026 — Bill 82/2026 — Dáil Éireann — Minister for Transport.

The old seven-Bill assumption must therefore not be forced into the new posts. Within the stale 2026 batch, Bills 26/2026, 40/2026 and 65/2026 are already recorded as defeated after later events, and Bill 76/2026 is no longer in the live First Stage set. Bills 44/2026, 63/2026 and 82/2026 remain. The live set also includes three current 2025 First Stage Bills that were outside the handoff's earlier seven-2026 working assumption.

## Proposed grouping

Use a 3/3 split rather than 3/4:

- First Stage · Post 1: Bills 4/2025, 6/2025 and 55/2025.
- First Stage · Post 2: Bills 44/2026, 63/2026 and 82/2026.

Each post would therefore have six slides: cover, three Bill slides, parliamentary-process glossary and First-Stage explainer glossary. The session branch `project.yml` now records period-specific expected counts of 6/6 while keeping publication disabled.

## Representative prototype

Prototype Bill: **Criminal Law (Adult Safeguarding) Bill 2026 (44/2026)**.

Selection is based only on layout/data complexity: it has a long title, three named sponsors, substantive initiated text and explanatory memorandum, and enough distinct legal-effect material to stress-test the one-slide hierarchy without inventing a vote or second Bill slide.

The one-slide information hierarchy is:

1. Bill identity / metadata — formal title, Bill number, House, introduction date and sponsors.
2. WHAT THE BILL PROPOSES — concise neutral legal-policy summary.
3. PRACTICAL EFFECT — what the proposed offences/orders would change if enacted.
4. WHY IT WAS INTRODUCED — attributed to the sponsors' explanatory memorandum where rationale is not a neutral fact.
5. WHERE IT IS NOW / WHAT HAPPENS NEXT — exact First Stage status and plain-English procedural context.
6. Official source footer — Houses of the Oireachtas, initiated Bill and explanatory memorandum.

## Prototype approval and requested revision

The human approved the prototype direction on 3 October 2026 with one visual change: enlarge the body copy inside the `WHAT FIRST STAGE MEANS` panel so that it uses the available box more fully. The Bill-slide renderer now fits that note from a substantially larger starting size while retaining deterministic bounds and overflow assertions. All other approved prototype geometry and information hierarchy were preserved.

## Full First Stage · Post 2 review

The approved component system was scaled to the six-slide Post 2 period:

1. Cover
2. Criminal Law (Adult Safeguarding) Bill 2026
3. Defence (Amendment) Bill 2026
4. Railway Safety Bill 2026
5. Parliamentary-process glossary with FIRST highlighted
6. HOW TO READ FIRST STAGE BILLS explainer

An inherited Enacted-process glossary layout initially failed the shared-fit typography assertion because the First Stage definitions required more vertical room. The failure was isolated with a temporary review-only diagnostic render. No diagnostic output was accepted as final. A First-Stage-specific process-glossary layout was then added within the same Bill Tracker factory, retaining the canonical left-to-right timeline and one shared body font across comparable term boxes. The temporary diagnostic fallback was removed before the clean run.

Clean generic factory workflow run `37154378301` rendered and published the full Post 2 review successfully from commit `90e7457cd6062e8651b5a12ba80613b97bf9835f`. The stable preview branch contains six real 1080×1350 slides, the Post 2 contact sheet, `index.html`, caption and ZIP. The clean preview caption contains no diagnostic/review-only leakage.

Review URL:

https://raw.githack.com/eirepolitic/eirepolitic-data-pipeline/previews/bill-tracker-first-stage/index.html

Direct contact sheet:

https://raw.githack.com/eirepolitic/eirepolitic-data-pipeline/previews/bill-tracker-first-stage/contact-first-stage-post2.jpg

The full Post 2 state remains `pending_human_review`. Publication and scheduling remain disabled.
