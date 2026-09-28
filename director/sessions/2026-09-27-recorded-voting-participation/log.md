# Recorded Dáil voting participation — Director session

The operator asked for a new EirePolitic Instagram carousel explaining recorded Dáil division participation over the most recent complete six-month period supported by production data, ideally 27 March 2026 to 27 September 2026. The requested measure is recorded participation opportunities divided by eligible TD × division opportunities, with event-date party and constituency attribution and explicit numerator/denominator retention.

The Director reference and linked live documents were read before investigation. This is a new-post collaboration, not an existing approved series. The operator's prompt supplies the content-idea gate, but not the visual-direction or final approval gates. Publishing and scheduling remain out of scope.

## Production evidence and resolved period

After the Director reference was updated to point explicitly to the Irish Politics Data Model catalogue, the production data path could be resolved directly. The promoted immutable production batch is `written-pq-answers-20260905-1`.

A session-specific read-only audit was added on the Director branch and executed through the existing batch-status workflow. It resolved the production Dáil division range as 18 December 2024 through 28 August 2026. Because production voting data does not reach 27 September 2026, the post period resolves to **28 February 2026 through 28 August 2026**, inclusive.

The period contains **136 Dáil divisions**. **Zero divisions** were excluded for incomplete member-vote coverage.

The final validated member-level universe contains **23,408 eligible TD × division opportunities**, of which **19,956** have a recorded Tá, Níl or formal abstention. The production data contain **73 formal recorded abstentions (`staon`) across 3 divisions** in the period.

## Validation results

All required denominator checks pass:

- production pointer remained stable during the read and the batch manifest is validated;
- TD × division opportunities are unique;
- member vote rows are unique at member × division grain;
- recorded member-vote rows reconcile to the official division tallies for all 136 divisions;
- all vote codes are recognised (`ta`, `nil`, `staon`);
- membership histories are unambiguous;
- party histories are unambiguous;
- constituency histories are unambiguous;
- every eligible opportunity and every recorded vote receives an event-date party and constituency attribution;
- members entering during the period are handled using their actual membership start dates (including Daniel Ennis and Seán Kyne from 25 May 2026);
- party and constituency aggregate numerators and denominators reconcile exactly to the member-level universe;
- no recorded vote falls outside the final eligible universe.

The identified presiding member was resolved for every division using debate/speech evidence. The audit removes an ordinary presiding-member opportunity unless the presiding member has a recorded vote in that division, preserving a recorded casting-vote case rather than assuming one. This removes **128 ordinary presiding-member opportunities**. Verona Murphy is identified in the office table as the current Ceann Comhairle, while the debate evidence also captures Leas-Cheann Comhairle and acting-chair cases.

No canonical production pairing or statutory-leave field was found in the promoted batch manifest, so the denominator is **not adjusted for pairing or leave**. No such evidence is inferred from external text or absence-reason tooling.

The audit evidence is committed at `director/sessions/2026-09-27-recorded-voting-participation/evidence/`.

## Metric interpretation

Recorded participation means a recorded `ta`, `nil`, or `staon` entry for an eligible member-division opportunity. `No recorded vote` remains exactly that: it is not treated as proof that a TD was physically absent.

Party and constituency percentages use total recorded participation opportunities divided by total eligible member-division opportunities. Individual TD percentages are not averaged to create group figures.

Required editorial caveat remains fixed: **Recorded voting participation does not by itself measure a TD’s attendance, workload, effectiveness, or overall job performance.**

## Visual reference and representative prototype

The canonical analytical reference is the July 2026 `party_issue_monthly_profile_v2` horizontal-bar treatment. The new post preserves alphabetical ordering instead of ranking by value.

A small reusable renderer extension was added on the session branch so horizontal bars can preserve input order (`sort: input`). This supports neutral alphabetical presentation without introducing a parallel renderer.

Per the new-post Director workflow, only **one representative slide** has been rendered before visual approval: the Parties slide. It shows all 11 parties/groups alphabetically, with participation percentages plus the recorded/eligible numerator and denominator beside each label.

The prototype uses:

- title: `Recorded voting participation — parties`;
- body: definition of recorded participation plus explicit statement that this is not an attendance measure;
- alphabetical bars rather than a leaderboard order;
- exact numerator/denominator beside every party/group;
- period footer: 28 Feb–28 Aug 2026, 136 divisions;
- the approved EirePolitic dark analytical palette and existing title/text/media layout.

Prototype QA passes with no chart or layout warnings, no text clipping and no label truncation. The rendered slide and metadata are under `director/sessions/2026-09-27-recorded-voting-participation/prototype/`.

## Current Director gate

The session is now at **pending visual-direction review**. The full carousel has **not** been rendered. Planned structure remains: cover; metric explainer; parties; constituencies; alphabetically paginated TDs; methodology/source slide.

Scaling to those slides requires the operator's visual-direction decision on the representative Parties prototype first. Final post approval remains a separate later gate.

No publishing or scheduling action has been taken.
