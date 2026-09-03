# Broader deterministic speech context

## Status

Investigation completed on 2026-09-03.

Production architecture was not changed.

Current recommendation: do **not** build a broad `speech_context` production foundation yet. Keep the existing certified `speech_question_context` for Oral-question exchanges. A narrowly defined Leaders' Questions rule is ready to certify, but the remaining candidate categories are not yet sufficiently complete and homogeneous to justify a new top-level production schema.

## Goal

Determine whether speeches can be assigned one deterministic top-level parliamentary context without AI/classifier calls, with explicit precedence and an `other` fallback.

Candidate contexts investigated:

1. `oral_question_exchange`
2. Bill / legislation debate
3. Leaders' Questions
4. motions
5. statements
6. parliamentary / procedural business
7. `other`

The intended guardrail remains: each speech may have at most one top-level context, and ambiguous cases remain `other`.

## Evidence

Temporary investigation branch:

- `ops/investigate-broader-speech-context`

Runs:

- `33703889234` — initial speech-context profile; successful.
- `33809629511` — follow-up Bill-link diagnostic; failed before producing output because it assumed `silver_speeches.source_debate_uri`, which does not exist in the live speech table.
- `33811157221` — corrected read-only diagnostic using `silver_debate_records` as the URI/debate bridge; successful.
- `33811273798` — compact digest of the corrected diagnostics; successful.

Primary temporary artifacts:

- `analysis/broader_speech_context_profile.json`
- `analysis/broader_speech_context_profile_digest.json`
- `analysis/broader_speech_context_link_diagnostics.json`
- `analysis/broader_speech_context_link_diagnostics_digest.json`

Production batch examined:

- `structure-oral-exchange-participants-20260902-1`

## Confirmed findings

### 1. Oral-question exchanges remain certified

The existing deterministic Oral-question relationship remains the strongest context rule.

Observed profile in the current production batch:

- 18,485 speeches
- 2,127 debate sections / Oral-question exchanges
- 2025: 10,811 speeches
- 2026: 7,674 speeches

This category should retain highest precedence over broader heading-based categories because it is backed by an already-certified question/exchange relationship rather than heading interpretation.

### 2. Leaders' Questions can be identified safely with an exact source-heading allowlist

The diagnostic substring rule returned 10,822 speeches across 159 sections, but manual overlap review exposed one false positive:

- `Report on Standing Orders and Dáil Reform on Rota for Leaders’ Questions Pursuant to Standing Order 38: Motion`

That row contains the words “Leaders’ Questions” but is a motion, not a Leaders' Questions session.

The source-generated headings themselves are otherwise highly stable. The two genuine observed forms are:

- `Ceisteanna ó Cheannairí - Leaders' Questions`
- `Ceisteanna ó Cheannairí (Atógáil) - Leaders' Questions (Resumed)`

They account for 10,821 speeches in the current data:

- main heading: 10,702 speeches
- resumed heading: 119 speeches

Coverage is present in both 2025 and 2026.

**Certification decision:** safe for public use if implemented as an exact allowlist of certified source headings, not a substring/regex rule.

### 3. Bill `debate_uri` / `debate_id` is debate-wide, not section-level

The Bill table is built from Oireachtas legislation debate records in `extract/oireachtas/table_bill_debates.py`.

The corrected diagnostics established that the Bill `debate_uri` values are whole debate/sitting identifiers. Example matched debates contained tens of unrelated sections and hundreds of speeches, including:

- Leaders' Questions;
- Oral-question sections;
- motions;
- statements;
- several distinct Bills;
- unrelated administrative business.

Current counts:

- 1,222 Bill-debate rows
- 399 distinct Bills
- 652 distinct Bill debate IDs / URIs
- 134 Bill debate IDs exactly overlap speech `debate_id`
- those 134 debates contain 55,722 speeches across 3,821 sections

Therefore, joining speeches to Bills at the debate URI / debate ID grain would massively over-label speeches and is rejected.

### 4. A promising Bill section-level relationship exists, but it is not fully certified yet

Two narrower relationships were found:

1. `silver_bill_debates.debate_section_id` behaves like a local Oireachtas `dbsect_*` identifier and overlaps `silver_debate_sections.section_eid` in 64 cases.
2. Within the same matched debate, `silver_bill_debates.debate_show_as` can often be matched exactly to one section heading.

The diagnostic found:

- 382 unique exact same-debate heading-to-section matches
- 7,413 speeches in those matched sections

Examples include exact section matches for Second Stage, Committee and Remaining Stages, Report and Final Stages, referral stages, and other Bill proceedings.

This is materially better than debate-wide joining, but it is not yet a production rule because:

- only part of the Bill table overlaps the current speech/debate coverage;
- `debate_section_id` needs to be treated explicitly as a source-local section identifier and validated against `section_eid` across chambers/years;
- unmatched Bill rows need explanation rather than fallback inference;
- exact heading matching should be validated as a source relationship, not treated as a semantic text classifier;
- Bill stages and linked entities should be investigated together in the planned legislation work.

**Certification decision:** promising deterministic route; defer production implementation to the legislation investigation.

### 5. Motions are detectable but too heterogeneous for one broad production label yet

The diagnostic heading patterns found:

- 6,821 speeches
- 294 sections
- coverage in 2025 and 2026

The source headings include several materially different forms:

- Private Members' motions;
- confidence motions;
- statutory approval motions;
- treaty / EU motions;
- procedural and standing-order motions;
- administrative motions.

Overlap review also found examples where a naive text rule collides with other contexts, including:

- a Leaders' Questions-related standing-order motion;
- `Statement of Estimates ...: Motion`;
- `Sittings and Business of the Dáil: Motion`.

**Certification decision:** do not publish one homogeneous `motions` category yet. First decide whether the intended top-level context should include all formal motions or whether substantive, statutory and procedural motions should be distinguished.

### 6. Statements are strongly source-signalled but not yet complete enough to certify as a broad rule

The current diagnostic patterns found:

- 5,081 speeches
- 125 sections
- coverage in 2025 and 2026

Most matches are explicit source headings ending in forms such as `: Statements` or `: Statements (Resumed)`.

However, the current pattern also picks up headings such as:

- `Budget Statement 2026`

and the investigation has not yet established a complete bilingual/source-heading allowlist for all statement-like proceedings.

**Certification decision:** likely deterministic, but not yet certified as a complete top-level rule. Prefer an exact source-heading family/allowlist after a completeness review rather than broad substring matching.

### 7. Procedural/business context is real, but the first rule under-covered it

The first candidate profile found:

- 4,711 speeches
- 59 sections

It correctly captured examples such as:

- `An tOrd Gnó - Order of Business`
- resumed Order of Business variants
- `Ceisteanna ar Reachtaíocht a Gealladh - Questions on Promised Legislation`

But diagnostic heading discovery also found a major omitted source heading:

- `Gnó na Dála - Business of Dáil` — 357 speeches

A naive `business` substring rule is also unsafe because unrelated topical headings can contain the word “business”.

**Certification decision:** a procedural/business category is plausible, but it needs an exact source-heading allowlist and a clear scope definition before public use.

## Rejected approaches

### Bill section join by `silver_bill_debates.debate_section_id = silver_speeches.debate_section_id`

Rejected. The identifiers are not in the same identifier space.

Observed overlaps:

- Bill `debate_section_id` -> speech `debate_section_id`: 0
- Bill `debate_section_id` -> silver section `debate_section_id`: 0
- Bill `debate_section_id` -> silver section `section_uri`: 0
- Bill `debate_section_id` -> silver section `section_eid`: 64

### Bill join by `debate_uri` / `debate_id` alone

Rejected. The URI represents the whole debate/sitting and includes many unrelated sections.

### Broad substring/regex context classification

Rejected as a production rule.

The diagnostics intentionally used broad patterns to discover candidate headings. Manual overlap review demonstrated false positives from headings that mention another context as part of a motion or administrative title.

Production rules should use exact certified source-generated headings or stronger source relationships.

### Speech-text interpretation

Not used and not recommended for this foundation.

No context in this investigation is certified from vague semantic interpretation of speech text.

## Methodological guardrails

1. Prefer exact source relationships over heading rules.
2. If headings are used, use certified exact source-heading allowlists, not broad regexes.
3. Keep `other` as an explicit fallback.
4. Do not label all speeches in a debate URI as Bill speeches.
5. Do not treat diagnostic regex coverage as certification.
6. Do not use speech-text similarity to infer formal parliamentary context.
7. Preserve one-row-per-speech coverage if a broader context foundation is eventually built.
8. Apply explicit precedence only between independently certified rules.
9. Keep the existing `speech_question_context` for compatibility unless/until a broader foundation is deployed and downstream contracts are migrated deliberately.
10. Describe these categories as parliamentary context, not as measures of political effectiveness or quality.

## Production implications

No production data or schema changed during this investigation.

The evidence does **not** yet justify replacing `speech_question_context` with a broader `speech_context` foundation.

A future production design remains plausible, but it should wait until legislation and the remaining heading families are certified. If built later, likely fields remain:

- `speech_id`
- `debate_date`
- `debate_section_id`
- `speech_context`
- source/evidence method
- optional linked entity identifier, such as Bill ID
- provenance/version

The exact architecture, schema and precedence should be proposed separately before implementation.

## Provisional precedence after this investigation

Only certified rules should participate in precedence.

Current safe ordering is therefore:

1. `oral_question_exchange`
2. `leaders_questions` using the exact certified heading allowlist
3. `other`

This is **not** a recommendation to deploy a three-value production foundation now; it only records the ordering that would be safe among currently certified rules.

Bills, motions, statements and procedural/business remain outside production precedence until separately certified.

## Living next-step plan

1. **Move into the legislation investigation next.** This is the highest-value unresolved dependency for broader speech context.
   - establish Bill IDs and debate-record relationships;
   - validate `silver_bill_debates.debate_section_id` against `silver_debate_sections.section_eid` across chambers/years;
   - quantify why only part of the Bill-debate universe overlaps the current speech/debate coverage;
   - validate exact same-debate `debate_show_as` -> section-heading matching;
   - capture Bill stages/readings from source-supported fields;
   - link speeches to Bills only at certified section grain;
   - investigate divisions linked to Bills;
   - investigate sponsors only where source-supported.
2. During or immediately after legislation work, finish exact source-heading allowlists for:
   - statements;
   - procedural/business proceedings.
3. Decide the intended public meaning of `motions` before implementation:
   - one formal-motion umbrella; or
   - separate substantive/private-members, statutory, confidence, and procedural motion families.
4. Recompute overlap/precedence using only certified rules.
5. If the resulting categories cover enough speeches to justify a broader foundation, prepare a short architecture/schema implementation plan before changing production.
6. Keep full Parliamentary Question topic classification deferred unless a specific downstream need demonstrates that deterministic recipient, heading, legislation and speech-context fields are insufficient.
7. After legislation context is established, revisit voting analysis so divisions can be described using substantive Bill/motion context rather than raw party-unity measures alone.
