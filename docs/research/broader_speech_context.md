# Broader deterministic speech context

## Status

Investigation completed and refreshed on 2026-09-03 after deployment of the certified Bill-section foundation.

Production speech-context architecture has **not** changed during this follow-up.

The evidence now supports a broader deterministic `speech_context` foundation. Six non-`other` contexts can be assigned safely with narrow source-backed rules and no observed overlap in the current production batch:

1. `oral_question_exchange`
2. `bill_or_legislation`
3. `leaders_questions`
4. `statements`
5. `procedural_business`
6. `motion_proceeding`
7. `other`

The current rules classify 53,452 of 66,192 speeches (80.75%). The remaining 12,740 speeches stay `other`.

This is now an architecture recommendation, not a production implementation. A short implementation plan must be approved before building the broader foundation.

## Goal

Determine whether every speech can receive at most one deterministic top-level parliamentary context without AI/classifier calls, with explicit precedence and an `other` fallback.

The governing principles remain:

- use source relationships before heading rules;
- use narrow certified source-heading forms rather than broad semantic matching;
- keep ambiguous or unsupported cases as `other`;
- never infer context from vague interpretation of speech text;
- never multiply speeches because multiple source records point at one parliamentary section.

## Evidence

### Initial investigation

Temporary branch:

- `ops/investigate-broader-speech-context`

Runs:

- `33703889234` — initial profile; successful.
- `33809629511` — failed Bill-link follow-up using a non-existent speech URI field.
- `33811157221` — corrected Bill-link diagnostic through debate records; successful.
- `33811273798` — compact diagnostic digest; successful.

Primary artifacts:

- `analysis/broader_speech_context_profile.json`
- `analysis/broader_speech_context_profile_digest.json`
- `analysis/broader_speech_context_link_diagnostics.json`
- `analysis/broader_speech_context_link_diagnostics_digest.json`

### Post-legislation follow-up

Temporary branch:

- `ops/investigate-speech-context-v2-20260903`

Production batch examined:

- `certified-bill-sections-20260903-1`

Runs:

- `33825553971` — first refreshed diagnostic attempt; failed before output because the temporary profile expected a section date column that is not present in `silver_debate_sections`.
- `33825622455` — diagnostic retry; failed before persisted output because the temporary capture step did not add the traceback file independently when the result file was missing.
- `33825695492` — corrected broad profile using the live Bill-section foundation; successful.
- `33825788414` — final narrow certification/precedence pass; successful.

Artifacts:

- `analysis/broader_speech_context_v2_digest.json`
- `analysis/broader_speech_context_certification_digest.json`

The failed investigation runs were read-only and made no production changes.

## Confirmed context rules

### 1. `oral_question_exchange`

Source:

- existing certified `speech_question_context`

Current footprint:

- 2,127 sections
- 18,485 speeches
- 2025: 10,811 speeches
- 2026: 7,674 speeches

This remains the strongest relationship-backed speech context and should retain first precedence.

### 2. `bill_or_legislation`

Source:

- live certified `bill_debate_sections`

Current footprint:

- 371 sections
- 7,352 speeches
- 2025: 3,605 speeches
- 2026: 3,747 speeches

The production Bill bridge already guarantees section-grain attribution and excludes debate-wide, conflicting and multi-Bill cases.

**Certification decision:** safe for broader speech context through exact `debate_section_id` membership in `bill_debate_sections`.

Do not infer Bill context from shared debate ID, URI, date, heading similarity or speech text.

### 3. `leaders_questions`

Source:

- exact Oireachtas section-heading allowlist

Certified headings:

- `Ceisteanna ó Cheannairí - Leaders' Questions`
- `Ceisteanna ó Cheannairí (Atógáil) - Leaders' Questions (Resumed)`

Current footprint:

- 158 sections
- 10,821 speeches
- 2025: 6,800 speeches
- 2026: 4,021 speeches

The earlier broad substring rule produced a false positive from a standing-order motion referring to Leaders' Questions. The exact allowlist removes that case.

**Certification decision:** safe for public use with exact heading equality only.

### 4. `statements`

Source:

- exact source-heading form, not broad substring search

Certified heading forms are headings ending exactly in:

- `: Statements`
- `: Statements (Resumed)`
- `: Ráitis`
- `: Ráitis (Atógáil)`

Current footprint:

- 123 sections
- 4,906 speeches
- 2025: 3,347 speeches
- 2026: 1,559 speeches

This deliberately excludes headings that merely contain the word “Statement”, including:

- `Budget Statement 2026`
- `Statement of Estimates for the Houses of the Oireachtas Commission: Motion`

Those are not automatically assigned to the statements category.

**Certification decision:** safe as a narrow source-form context. This is a proceeding-type label, not a semantic statement-topic classifier.

### 5. `procedural_business`

Source:

- exact Oireachtas section-heading allowlist

Certified headings:

- `An tOrd Gnó - Order of Business`
- `An tOrd Gnó - Order of Business (Resumed)`
- `An tOrd Gnó (Atógáil) - Order of Business (Resumed)`
- `Ceisteanna ar Reachtaíocht a Gealladh - Questions on Promised Legislation`
- `Gnó na Dála - Business of Dáil`

Current footprint:

- 87 sections
- 5,067 speeches
- 2025: 3,287 speeches
- 2026: 1,780 speeches

The diagnostic heading `Sittings and Business of the Dáil: Motion` is deliberately excluded from this allowlist and remains a motion proceeding.

**Certification decision:** safe for public use as a narrow parliamentary-business context.

### 6. `motion_proceeding`

Source:

- exact source-heading form

Certified rule:

A heading must end in a formal Oireachtas motion form:

- `: Motion`
- `: Motion (Resumed)`
- `: Motion [Private Members]`
- `: Motion (Resumed) [Private Members]`
- `: Motions`
- `: Motions (Resumed)`

Current footprint:

- 296 sections
- 6,821 speeches
- 2025: 3,927 speeches
- 2026: 2,894 speeches

The source family includes many different parliamentary purposes, for example:

- Private Members' motions;
- confidence motions;
- statutory approval motions;
- treaty/EU motions;
- standing-order and committee motions;
- administrative motions.

Therefore the public meaning must remain narrow:

**`motion_proceeding` means the Oireachtas source identifies the section as a formal motion proceeding. It does not imply that all such motions are substantively comparable.**

This resolves the earlier concern about treating every motion as one homogeneous political topic. The category describes parliamentary form, not subject matter or political significance.

**Certification decision:** safe with this narrow interpretation and exact source-form rule.

### 7. `other`

Current footprint:

- 1,848 sections in the section foundation not assigned to one of the certified section contexts
- 12,740 speeches
- 2024: 119 speeches
- 2025: 7,499 speeches
- 2026: 5,122 speeches

`other` remains mandatory. It is not an error state and must not be filled through semantic guessing.

## Final overlap result

The final narrow certification pass found **zero section overlaps** among:

- `oral_question_exchange`
- `bill_or_legislation`
- `leaders_questions`
- `statements`
- `procedural_business`
- `motion_proceeding`

This means the current source rules are mutually exclusive in the examined production batch.

Precedence should still be encoded explicitly so future source changes cannot create unstable assignments.

## Recommended precedence

Recommended production precedence:

1. `oral_question_exchange`
2. `bill_or_legislation`
3. `leaders_questions`
4. `statements`
5. `procedural_business`
6. `motion_proceeding`
7. `other`

Rationale:

- certified relationship-backed contexts come before heading-derived contexts;
- the exact heading families are currently mutually exclusive;
- explicit ordering protects against future overlap without relying on incidental current-data separation.

If any future overlap appears, it should be surfaced by audit rather than silently changing classification behaviour.

## Coverage

Current production speech count:

- 66,192 speeches

Certified non-`other` contexts:

- 53,452 speeches
- 80.75% of speeches

Remaining `other`:

- 12,740 speeches
- 19.25% of speeches

This is sufficient coverage to justify a broader deterministic foundation without AI classification.

## Rejected approaches

### Debate-wide Bill attribution

Rejected. A Bill can be one section among many unrelated proceedings in the same sitting.

Use only `bill_debate_sections`.

### Global `dbsect_*` joins

Rejected. Source-local section EIDs are only meaningful within a debate.

### Broad Leaders' Questions substring matching

Rejected because administrative/motion headings can mention Leaders' Questions without being a Leaders' Questions session.

### Broad “statement” substring matching

Rejected because it captures headings such as `Budget Statement 2026` and motion titles containing “Statement”.

### Broad “business” substring matching

Rejected because topical headings can contain the word “business”. Use the exact procedural allowlist.

### Treating motion proceedings as one substantive topic

Rejected. The certified `motion_proceeding` category records parliamentary form only.

### Speech-text interpretation

Rejected for top-level context assignment. No certified rule depends on semantic interpretation of speech text.

### AI/classifier calls

Not needed. Deterministic source structure now covers more than 80% of speeches with a safe `other` fallback.

## Methodological guardrails

1. Every source speech must appear exactly once in a future broader context foundation.
2. Every speech must have at most one top-level context.
3. `other` remains a valid explicit category.
4. Relationship-backed rules take precedence over source-heading rules.
5. Heading rules must use certified exact forms/allowlists, not broad keyword regexes.
6. Future overlaps must fail or warn in audit; they must not silently depend on DataFrame ordering.
7. `motion_proceeding` describes parliamentary form, not political topic, importance, quality or effectiveness.
8. `bill_or_legislation` must inherit only from the certified Bill-section bridge.
9. Keep recipient, Bill ID, section heading and other linked entities as separate dimensions rather than embedding them into the top-level label.
10. Keep existing `speech_question_context` for compatibility during any broader-context rollout.
11. Do not run Parliamentary Question topic classification merely to fill top-level speech context.
12. Do not describe activity/context measures as political effectiveness or performance.

## Production implications

The follow-up evidence changes the earlier architecture decision.

A broader deterministic `speech_context` foundation is now justified because:

- the Bill relationship is production-certified;
- Leaders' Questions has a stable exact allowlist;
- statements have a narrow bilingual source-form rule;
- procedural/business has a bounded exact allowlist;
- motion proceedings can be safely represented as parliamentary form rather than substantive category;
- all six certified non-`other` contexts are mutually exclusive in the current production data;
- deterministic coverage reaches 80.75% of speeches.

This should be an **additive** production foundation initially. Do not remove or replace `speech_question_context` during the first deployment.

A likely schema remains:

- `speech_id`
- `debate_date`
- `debate_section_id`
- `speech_context`
- `evidence_method`
- optional linked entity identifier where structurally relevant, especially `bill_id`
- source/provenance version
- source batch ID

The exact implementation/schema must be planned before production changes.

## Living next-step plan

1. Prepare a short implementation plan for an additive `speech_context` foundation.
2. The plan should preserve one row per speech and include all 66,192 current speeches, including `other`.
3. Implement explicit precedence in code even though the current certified rules have zero overlap.
4. Add audits for:
   - one row per source speech;
   - no missing source speeches;
   - no duplicate `speech_id`;
   - allowed context values only;
   - no unexpected rule overlap;
   - Bill-linked speeches resolving to one certified Bill at most;
   - context counts/coverage changes by year.
5. Keep `speech_question_context` deployed for compatibility and verify the new `oral_question_exchange` assignments exactly agree with it.
6. Preserve `bill_debate_sections` as the legislation relationship foundation; do not duplicate Bill-certification logic inside speech context.
7. After successful broader-context deployment, revisit voting analysis using certified Bill/motion context.
8. Continue source-structure investigation for the remaining `other` speeches only when a concrete downstream need exists; do not force them into categories.
9. Expand Seanad/committee canonical speech coverage separately before presenting this as whole-Oireachtas context coverage.
10. Keep full Parliamentary Question issue classification deferred unless a specific downstream use case demonstrates that deterministic context, recipient, heading and legislation dimensions remain insufficient.
