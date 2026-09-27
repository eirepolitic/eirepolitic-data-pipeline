# Recorded Dáil voting participation — Director session

The operator asked for a new EirePolitic Instagram carousel explaining recorded Dáil division participation over the most recent complete six-month period supported by production data, ideally 27 March 2026 to 27 September 2026. The requested measure is recorded participation opportunities divided by eligible TD × division opportunities, with event-date party and constituency attribution and explicit numerator/denominator retention.

The Director reference and linked live documents were read before investigation. This is a new-post collaboration, not an existing approved series. The operator's prompt supplies the content-idea gate, but does not supply visual-direction approval or final post approval. Publishing and scheduling remain out of scope.

## Updated data-documentation evidence

The Director reference was reread after the operator updated it to point explicitly to the Irish Politics Data Model documentation. That documentation materially resolves the earlier data-discovery blocker.

The published catalogue identifies the promoted production batch as `written-pq-answers-20260905-1` and documents `silver_divisions` as a 401-row division-event table resolved through that production batch. Its examples include a Dáil division dated 2026-08-28 with snapshot date 2026-09-01, so the originally requested 27 March 2026 to 27 September 2026 window is not fully supported by the documented production voting snapshot.

The same catalogue documents the voting and member-history products needed for this analysis, including division events, member votes, membership histories, party histories and constituency histories. The political-metrics catalogue already defines the requested recorded-voting-participation concepts and denominator semantics, so the analysis should reuse those foundations rather than invent a new denominator.

A read-only `Oireachtas batch control` status workflow was run from `main` as run `36360350524` and completed successfully. It produced a status artifact. The connector does not expose the artifact contents directly, but the published catalogue independently names the promoted batch above.

## Interpretation rules already established

Repository voting semantics confirm that member participation uses eligible member × division opportunities and that a non-recorded vote must not be interpreted as proof of absence. The source vote flattener reads `taVotes`, `nilVotes` and `staonVotes`; the production vote commissioner accepts only `ta`, `nil` and `staon`, so formal abstention (`staon`) is a recorded participation event.

Membership eligibility is start-inclusive/end-exclusive. Historical party and constituency attribution use the same start-inclusive/end-exclusive rule on the division date. Ambiguous temporal history matches raise rather than being silently resolved. Party and constituency participation are calculated from total recorded member-vote opportunities divided by total eligible member-division opportunities, not by averaging TD percentages.

The canonical visual reference for analytical comparison slides is the July 2026 `party_issue_monthly_profile_v2` completed post identified in `director/references.yml`, using the approved horizontal-bar analytical treatment. If this post advances, party, constituency and TD ordering should remain alphabetical rather than ranked.

## Current status

The previous broad blocker is cleared. The task can proceed using the documented production batch and existing political-metrics foundations.

Before any carousel prototype is rendered, the remaining validation work is to calculate the exact latest six-month period ending within production coverage, establish the exact number of divisions in that period, prove TD × division uniqueness, reconcile member-vote records to division totals, test entered/left members and any party changes, test the Ceann Comhairle separately, confirm whether pairing/statutory-leave evidence exists, identify any incomplete divisions that require exclusion, and reconcile party/constituency aggregates back to the member-level universe.

Required editorial caveat remains fixed for the eventual post: **Recorded voting participation does not by itself measure a TD’s attendance, workload, effectiveness, or overall job performance.**

No publishing or scheduling action has been taken.
