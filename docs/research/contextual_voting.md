# Contextual voting analysis

## Status

Investigation completed on 2026-09-03.

No production architecture or voting metric changed during this investigation.

The current production batch already supports deterministic parliamentary context for all 401 live Dáil divisions through exact debate-section relationships. The evidence supports a future additive one-row-per-division context foundation, but that is a separate production architecture change and should be planned before implementation.

## Goal

Revisit voting analysis after deployment of:

- certified `bill_debate_sections`; and
- broader deterministic `speech_context`.

The purpose was to determine whether divisions can be described with safer substantive parliamentary context without weakening the existing voting denominators or relying on division subject text alone.

## Evidence

Temporary investigation branch:

- `ops/investigate-contextual-voting-20260903`

Production batch examined:

- `broader-speech-context-20260903-1`

Runs:

- `33833310590` — first contextual-voting profile attempt; failed during read-only diagnostic logic and made no production change.
- `33833380526` — corrected contextual-voting profile; successful.
- `33833423049` — compact digest; successful.

Artifacts:

- `analysis/contextual_voting_profile.json`
- `analysis/contextual_voting_digest.json`

Current production source counts:

- 401 divisions
- 59,325 member-vote rows
- 371 certified Bill sections
- 66,192 speech-context rows

## Existing voting metrics remain methodologically valid

The existing voting calculations already use conservative denominators:

- member voting participation is based on eligible member × division opportunities, not all sittings or all theoretical votes;
- party voting unity/cohesion is based on recorded party votes inside each division;
- low-vote and low-division samples are guarded rather than treated as stable comparison metrics.

This investigation does **not** recommend changing those denominators.

Context should be added as a dimension around existing vote calculations, not used to redefine participation or cohesion.

## Confirmed division contexts

Every one of the 401 current divisions falls into one of four section-level contexts:

- `bill_or_legislation`: 168 divisions
- `motion_proceeding`: 153 divisions
- `procedural_business`: 53 divisions
- `other`: 27 divisions

No current division falls in:

- `oral_question_exchange`
- `leaders_questions`
- `statements`

This result is consistent with the parliamentary purpose of recorded divisions in the current Dáil data.

## 1. Bill-linked divisions are deterministic

168 divisions are linked to certified Bills through exact `debate_section_id` membership in `bill_debate_sections`.

Current Bill-linked vote profile:

- 168 divisions
- 24,427 recorded member votes
- median recorded votes per division: 146
- median absolute Tá/Níl margin: 19 votes

Examples include:

- Second Stage votes;
- Committee Stage amendments;
- Report/Final Stage amendments;
- final passage questions.

Many division subjects are generic, for example:

- `Amendment put:`
- `Question put:`

Therefore the Bill relationship is materially more informative and safer than division-subject text alone.

**Decision:** Bill context for a division must come from the certified Bill-section bridge, never from subject text, shared debate ID or debate date.

## 2. Motion proceedings form a large non-Bill voting context

153 divisions occur in sections certified as `motion_proceeding`.

Current profile:

- 153 divisions
- 22,777 recorded member votes
- median recorded votes per division: 149
- median absolute Tá/Níl margin: 16 votes

These include amendments and final questions on motions.

As established in the speech-context research, `motion_proceeding` describes parliamentary form only. It must not be presented as one substantive political topic or one uniform category of political importance.

**Decision:** motion context is safe as a filter/dimension, but not as a claim that all motion votes are substantively comparable.

## 3. Procedural-business divisions are a distinct voting context

53 divisions occur in certified `procedural_business` sections.

Current profile:

- 53 divisions
- 7,979 recorded member votes
- median recorded votes per division: 150
- median absolute Tá/Níl margin: 17 votes

Typical subjects concern arrangements for the week's business or amendments to the Order of Business.

This category should remain separate from substantive Bill or motion voting because the underlying parliamentary purpose is different.

## 4. Twenty-seven divisions remain `other`

27 divisions are not covered by the certified Bill, motion or procedural-business context rules.

Current profile:

- 27 divisions
- 4,142 recorded member votes
- median recorded votes per division: 156
- median absolute Tá/Níl margin: 23 votes

Examples include financial-resolution proceedings and other source structures not yet covered by the current speech-context categories.

**Decision:** preserve these as `other`. Do not classify them from generic division subjects without a separately certified source relationship.

## Close-vote observations

The current data contains close divisions in all major contexts.

Examples from the diagnostic include:

- a 2-vote margin in an `other` division;
- a 5-vote margin in a certified Bill division;
- 7-vote margins in motion proceedings;
- 8- or 9-vote margins in Bill, motion and procedural-business divisions.

These are descriptive vote outcomes only.

A close margin does **not** by itself imply political importance, effectiveness, rebellion, government instability or policy significance.

If close-vote reporting is exposed publicly, it should state the actual Tá/Níl/Staon totals and parliamentary context rather than attaching evaluative language.

## Bill stage attribution remains unresolved at exact division grain

Bill stage history is source-supported, but the current stage table is a lifecycle history rather than a certified division-to-stage relationship.

The investigation found examples where:

- stage-history dates align with the date of a Bill-linked division;
- multiple stage records can exist for the same Bill and date;
- stages span both Dáil and Seanad;
- source headings sometimes describe combined stages such as `Committee and Remaining Stages` or `Report and Final Stages`;
- `Cream List` remains a source stage value whose public meaning has not been certified.

Therefore a division must **not** inherit a stage merely because one stage-history row has the same Bill and date.

**Decision:** keep Bill stage history separate until an exact stage/proceeding relationship is certified.

## Safe contextual voting measures

The following are supported by current foundations when denominators remain explicit:

- number of divisions by parliamentary context;
- recorded member-vote counts by context;
- member voting participation within a context, using eligible member × division opportunities restricted to that context;
- party voting cohesion within a context, using recorded party votes and the existing minimum-sample rules;
- Tá/Níl/Staon distributions for a specific division;
- division margin, with explicit vote totals;
- Bill-linked vote histories for a certified Bill;
- contextual comparisons such as Bill divisions vs motion divisions, provided they are described as different parliamentary forms rather than measures of quality or effectiveness.

## Unsafe or misleading interpretations

Do not describe:

- high voting participation as political effectiveness;
- high party cohesion as party quality or discipline success;
- low cohesion as rebellion without inspecting the specific votes and party membership context;
- a close division as politically important solely because the margin is small;
- Bill stage from date coincidence;
- Bill context from generic division subject text;
- non-recorded votes as proven absence;
- all motion proceedings as one substantive political topic.

## Production architecture implication

The current data supports an additive one-row-per-division context foundation.

A useful future dataset could be named:

- `division_context`

Recommended grain:

- one row per `division_id`

Likely fields:

- `division_id`
- `division_date`
- `debate_section_id`
- `division_context`
- `evidence_method`
- `linked_entity_type`
- `linked_entity_id`
- optional source section heading
- `source_batch_id`
- `context_version`
- `calculated_at_utc`
- `contract_version`

For Bill divisions:

- `linked_entity_type = bill`
- `linked_entity_id = bill_id`

For non-Bill contexts, linked entity fields should remain blank unless another certified entity relationship exists.

The foundation should derive section context without joining one division to many speech rows. It should use a deterministic section-context projection from the already certified context rules/foundations.

## Why a dedicated division-context layer is preferable

It avoids several downstream hazards:

1. joining divisions directly to speech-level `speech_context` would multiply division rows by the number of speeches in the section;
2. recomputing context independently in dashboards risks drift from production certification rules;
3. generic division subjects are often insufficient to identify the underlying Bill or parliamentary form;
4. one-row-per-division context gives Power BI, Appsmith and APIs a safe lookup dimension without changing vote denominators.

## Living next-step plan

1. Prepare a short implementation plan for an additive `division_context` foundation before changing production architecture.
2. Define exact precedence/derivation rules using existing certified relationships:
   - certified Bill section → `bill_or_legislation` + Bill ID;
   - certified motion section → `motion_proceeding`;
   - certified procedural-business section → `procedural_business`;
   - otherwise `other`.
3. Ensure output grain is exactly one row per source `division_id`.
4. Add audits for:
   - one row per source division;
   - no missing or extra divisions;
   - no duplicate `division_id`;
   - Bill-linked divisions resolving to one certified Bill;
   - allowed context values only;
   - context-count drift by source batch;
   - no division multiplication when joined to member votes.
5. Keep existing `division_party_vote_components`, member participation and party-cohesion calculations unchanged initially.
6. After deploying `division_context`, investigate contextual variants of existing voting metrics rather than creating new evaluative metrics.
7. Before publishing party/member contextual comparisons, validate sample sizes within each context and keep the existing reliability thresholds.
8. Investigate exact Bill stage/proceeding linkage separately; do not infer stage from Bill + date.
9. Investigate the 27 `other` divisions only if a concrete downstream use case requires another deterministic source category.
10. Preserve the rule that voting activity/cohesion/context are descriptive measures, not evidence of political effectiveness, quality or performance.
