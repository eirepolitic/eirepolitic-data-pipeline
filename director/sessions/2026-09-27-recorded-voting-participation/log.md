# Recorded Dáil voting participation — Director session

The operator asked for a new EirePolitic Instagram carousel explaining recorded Dáil division participation over the most recent complete six-month period supported by production data, ideally 27 March 2026 to 27 September 2026. The requested measure is recorded participation opportunities divided by eligible TD × division opportunities, with event-date party and constituency attribution and explicit numerator/denominator retention.

The Director reference and linked live documents were read before investigation. This is a new-post collaboration, not an existing approved series. The operator's prompt supplies the content-idea gate, but does not supply visual-direction approval or final post approval. Publishing and scheduling remain out of scope.

## What is already established

Repository voting semantics confirm that member participation uses eligible member × division opportunities and that a non-recorded vote must not be interpreted as proof of absence. The source vote flattener reads `taVotes`, `nilVotes` and `staonVotes`; the production vote commissioner accepts only `ta`, `nil` and `staon`, so formal abstention (`staon`) is a recorded participation event.

Membership eligibility is start-inclusive/end-exclusive. Historical party and constituency attribution use the same start-inclusive/end-exclusive rule on the division date. Ambiguous temporal history matches raise rather than being silently resolved. Party and constituency participation are calculated from total recorded member-vote opportunities divided by total eligible member-division opportunities, not by averaging TD percentages.

The canonical visual reference for analytical comparison slides is the July 2026 `party_issue_monthly_profile_v2` completed post identified in `director/references.yml`, using the approved horizontal-bar analytical treatment. If this post advances, party, constituency and TD ordering should remain alphabetical rather than ranked.

A read-only production historical audit was dispatched from `main` as GitHub Actions run `36353072089`. It completed successfully. The connector can report the successful run and step results, but exposes the detailed logs only as a signed ZIP URL that is not readable in the current environment.

## Evidence blocker

The post cannot yet advance to a representative-slide prototype because the denominator is not fully defensible from the evidence accessible through the approved Director path in this session.

The current repository identifies the required production sources and formulas, but this session has not been able to inspect the live contents of one current immutable production batch for:

- exact latest `silver_divisions` coverage and therefore the exact six-month period;
- exact division count in scope;
- `silver_member_votes` completeness by division;
- TD × division uniqueness across the chosen period;
- reconciliation of member-vote rows to division totals;
- representative entered/left-member eligibility checks on the live batch;
- event-date party and constituency coverage for the live voting universe;
- Ceann Comhairle/presiding-member treatment in the denominator;
- whether pairing or statutory-leave information exists in the current production pipeline and can be joined safely;
- whether any division requires exclusion for incomplete member-vote coverage;
- whether any live party/constituency history overlap would affect a division-date join.

The successful existing historical audit proves its own speech/history checks, but it does not expose the division/member-vote evidence needed for this voting denominator through the current connector. Creating and pushing a new AWS-credentialed workflow solely to extract those data would cross the Director's separate approval boundary for new credentialed workflow code, so that route was not used.

## Review point

Status is `pending_review` at an evidence blocker, before prototype rendering. No percentage, party comparison, constituency comparison, TD figure, slide image, schedule, or publication action has been produced from an unverified denominator.

Once the live production voting evidence is made accessible through an approved path, the next Director steps are: resolve the exact one-batch period; run all requested uniqueness, reconciliation, eligibility, temporal-attribution, Ceann Comhairle, abstention, pairing/leave and incomplete-division checks; populate the slide plan; then render one representative comparison slide using the canonical analytical reference for human visual-direction review.

Required editorial caveat remains fixed for the eventual post: **Recorded voting participation does not by itself measure a TD’s attendance, workload, effectiveness, or overall job performance.**

No publishing or scheduling action has been taken.
