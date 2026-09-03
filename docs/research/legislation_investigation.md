# Legislation relationships investigation

## Status

Investigation completed on 2026-09-03.

No production architecture or data changed.

Current recommendation: the source supports a conservative deterministic Bill-to-debate-section relationship for a substantial subset of current Dáil coverage. This is sufficient to support a future legislation context foundation, but production implementation should be proposed separately before changing schemas or downstream contracts.

## Goal

Investigate deterministic relationships among:

- Bills;
- Bill stages/events;
- sponsors;
- parliamentary debate sections;
- speeches;
- divisions.

The key requirement was to avoid debate-day attribution. A Bill relationship is only useful for public metrics if it reaches the correct debate section.

## Evidence

Temporary investigation branch:

- `ops/investigate-legislation-20260903`

Runs:

- `33819091517` — first legislation relationship profile; successful, but it exposed an investigation bug caused by treating `section_eid` as globally unique.
- `33819178835` — corrected profile using debate-scoped section identifiers; successful.
- `33819238120` — compact digest of the corrected profile; successful.
- `33819289813` — conservative Bill-section certification subset and edge-case review; successful.

Temporary artifacts:

- `analysis/legislation_relationships_profile.json`
- `analysis/legislation_relationships_profile_v2.json`
- `analysis/legislation_relationships_digest.json`
- `analysis/legislation_certification_digest.json`

Production batch examined:

- `structure-oral-exchange-participants-20260902-1`

## Source foundations already present

The repository already contains confirmed source-backed tables for:

- `silver_bills`
- `silver_bill_stages`
- `silver_bill_sponsors`
- `silver_bill_debates`
- `silver_bill_events`
- `silver_debate_sections`
- `silver_speeches`
- `silver_divisions`

Current production snapshot counts used in this investigation:

- 406 Bills
- 1,395 Bill-stage rows
- 1,213 Bill-sponsor rows
- 1,222 Bill-debate rows
- 711 Bill-event rows
- 5,010 debate sections
- 66,192 speeches
- 401 divisions

All 406 Bills currently have stage, sponsor and event records. 399 have at least one Bill-debate row.

## Confirmed findings

### 1. Bill debate URI / debate ID is whole-debate grain

This confirms the earlier broader-speech-context finding.

A Bill debate's `debate_id` / `debate_uri` identifies a sitting/debate that can contain many unrelated sections. It is not a safe Bill-to-speech join by itself.

**Rule:** never label every speech or division sharing a debate ID as Bill-related.

### 2. `dbsect_*` identifiers are debate-local, not globally unique

The first legislation diagnostic incorrectly joined `silver_bill_debates.debate_section_id` to `silver_debate_sections.section_eid` globally. This produced obviously wrong cross-date matches because values such as `dbsect_18` repeat across debates.

The correct candidate relationship is the composite key:

`(bill_debate.debate_id, bill_debate.debate_section_id)`

→

`(debate_section.debate_id, debate_section.section_eid)`

The local section identifier must never be used without the debate identifier.

### 3. The composite Bill-section relationship is strongly supported for current Dáil coverage

The corrected profile found 424 Bill-debate rows where the debate-scoped source section identifier maps to a current debate section.

A second independent check compared the Bill's source `debate_show_as` value with the exact section heading within the same debate.

Results:

- 419 rows: source section ID and unique exact source heading identify the same section;
- 5 rows: conflict or partial ambiguity;
- 798 rows: no matching current section in the present debate-section coverage.

The 419 agreements are strong evidence that the composite source key represents the intended section relationship in normal cases.

### 4. A conservative certified subset should exclude multi-Bill section anomalies

Among the 419 agreement rows, 11 debate sections were associated with more than one Bill ID, affecting 23 Bill-debate rows.

These are concentrated in First Stage records and include examples where distinct Bill IDs share the same source section/title relationship. The investigation did not establish whether these are legislation-source corrections, renumbering/replacement behaviour, or another source-data artefact.

Therefore they should not be assigned to one Bill for public metrics without further source-specific resolution.

After excluding them, the conservative certified subset is:

- 396 Bill-debate rows
- 168 distinct Bills
- 371 distinct debate sections
- 7,352 speeches
- 168 divisions across 94 Bill-linked sections

Coverage by year within this certified subset:

- 2025: 183 Bill-debate rows
- 2026: 213 Bill-debate rows

### 5. The five conflict/ambiguity cases must remain uncertified

Three rows had an apparent source section-ID offset where the section ID points at a different proceeding while the exact Bill heading appears in the adjacent section. Examples included First Stage records where the source section ID pointed to `Questions on Policy or Legislation` or an unrelated motion.

Two additional rows involved a resumed Bill heading repeated in two sections on the same debate date, so exact heading matching was not unique.

These cases demonstrate why neither source section ID nor heading should be used blindly on its own.

**Certification rule:** for the current conservative foundation, require both:

1. debate-scoped source section ID resolves to a section; and
2. the Bill's exact source heading uniquely resolves to the same section.

Then exclude any section linked to more than one Bill.

### 6. Current unmatched Bill-debate rows are primarily a coverage issue, not evidence that the relationship is invalid

798 of the 1,222 Bill-debate rows do not map to the current `silver_debate_sections` coverage.

Observed examples are dominated by:

- Seanad debates;
- select committee debates;
- older historical debate records outside the currently materialized debate/speech range.

The current corrected matches are all Dáil rows because the present debate-section/speech production coverage used here does not contain the corresponding Seanad/committee sections.

Do not treat the 798 unmatched rows as failed Bill relationships. They are currently outside the available section foundation.

### 7. Divisions can be linked to Bills at section grain for the certified subset

`silver_divisions` already contains canonical `debate_section_id` values.

Using the conservative certified Bill-section set identifies:

- 168 divisions
- across 94 Bill-linked sections

Examples include Second Stage votes, amendments, Report/Final Stage votes and Bill-passage questions.

This is substantially safer than trying to infer Bill context from division subject text or debate-date co-occurrence.

**Decision:** a future Bill-division relationship should flow through the certified Bill section, not through text matching.

### 8. Bill stages are already source-supported and useful as a separate progress/history structure

Current stage table:

- 1,395 rows
- 406 Bills

Observed stage names include:

- First Stage
- Second Stage
- Committee Stage
- Report Stage
- Fifth Stage
- Enacted
- `Cream List`

Stage outcome is frequently blank, with explicit values such as `Current`, `Enacted`, `Defeated` and `Lapsed` present on a subset.

The stage table should remain a Bill progress/history source. It should not be assumed that each stage-history row maps one-to-one to a transcript section without an explicit source link.

`Cream List` should be investigated before exposing stage names directly as a public taxonomy because its meaning is not self-explanatory.

### 9. Sponsors are source-supported, but member and office sponsorship must remain distinct

The sponsor table contains:

- 1,213 rows
- all 406 Bills
- 408 rows marked primary

476 sponsor rows, covering 228 Bills, match a known member directly by source URI (`sponsor_uri` -> `member_uri`). This is the preferred member linkage and can map to `member_code` deterministically.

72 rows have no sponsor person name but instead contain an office/ministerial role such as:

- Minister for Defence
- Minister for Transport
- Minister for Justice, Home Affairs and Migration
- Minister for Housing, Local Government and Heritage

These must not be forced onto a named current office-holder without a date-aware office relationship.

**Rule:** preserve source sponsor type/evidence. A source-backed member sponsor can link by URI; an office sponsor should remain an office sponsor unless a date-aware office-to-person attribution is separately certified.

## Rejected approaches

### Global join on `section_eid`

Rejected. `dbsect_*` values repeat across debate records and are only meaningful within a debate.

### Debate ID / URI as a Bill-to-speech join

Rejected. It labels unrelated sections in the same sitting.

### Heading-only Bill classification

Rejected as the primary relationship. Exact heading agreement is valuable validation evidence, but headings can repeat and a few records disagree with the source section field.

### Division subject-text inference

Not needed for Bill attribution where a certified section relationship exists. Subjects such as `Amendment put:` are often too generic by themselves.

### Sponsor name matching

Not recommended as the primary identity link. Use source URI where available.

### Assigning ministerial office sponsors to a person by current office-holder

Rejected. That would create historical-attribution risk unless office tenure is matched to the Bill sponsorship date/source event.

## Methodological guardrails

1. Treat Bill IDs as the canonical legislation entity identifier.
2. Treat `dbsect_*` identifiers as debate-local.
3. Never label an entire debate/sitting as a Bill debate because one Bill appears in it.
4. For the current certified subset, require agreement between the debate-scoped section identifier and unique exact source heading.
5. Exclude multi-Bill section anomalies until their source semantics are understood.
6. Keep unmatched Seanad/committee/historical Bill-debate rows as unresolved coverage, not negative matches.
7. Link divisions to Bills through certified debate sections.
8. Keep Bill stage history conceptually separate from transcript-section context unless a source relationship is explicit.
9. Link member sponsors by source URI, not names.
10. Preserve ministerial/office sponsorship as office-level evidence unless date-aware person attribution is independently certified.
11. Do not use speech text similarity to create Bill relationships.
12. Keep public descriptions neutral and descriptive; activity around legislation is not evidence of political quality or effectiveness.

## Production implications

No production architecture changed during this investigation.

The investigation does support a future deterministic legislation-context implementation, but this would be a production architecture change and should therefore be planned separately before implementation.

A useful foundation could expose a certified Bill-to-section relationship with fields such as:

- `bill_id`
- `bill_debate_id`
- `debate_id`
- `debate_section_id`
- source-local `section_eid`
- `debate_date`
- source debate heading / stage wording
- evidence method
- certification status/version

From that relationship, speeches and divisions can inherit Bill context by exact `debate_section_id` without duplication at debate-day grain.

The production design should decide explicitly whether this is materialized as:

- a dedicated Bill-section bridge used by downstream metrics; or
- part of a broader future `speech_context` / parliamentary-context foundation.

The existing source tables should remain intact either way.

## Living next-step plan

1. Prepare a short implementation plan for a production **certified Bill-section bridge** before changing architecture.
2. In that plan, define the conservative certification rule currently supported by evidence:
   - debate-scoped section-ID match;
   - unique exact heading agreement;
   - exclude multi-Bill sections;
   - retain unmatched/conflicting records with explicit status rather than guessing.
3. Decide whether to materialize only certified links or also retain unresolved Bill-debate rows in the same foundation with status/evidence fields.
4. Add permanent audits for:
   - source-section/heading disagreement;
   - multiple Bills per certified section;
   - duplicate Bill-section links;
   - speech/division joins multiplying rows;
   - coverage by chamber/year.
5. Expand debate-section/speech coverage to Seanad and committee debates before claiming whole-Oireachtas Bill speech coverage.
6. Investigate the small source anomalies:
   - the five section/heading conflict or ambiguity rows;
   - the 11 multi-Bill sections / 23 rows;
   - `Cream List` stage semantics.
7. Build a safe sponsor interpretation layer only if needed:
   - member sponsor via source URI;
   - office/ministerial sponsor kept separate;
   - date-aware office-holder attribution only if a real downstream use case requires it.
8. Once Bill-section context is deployed, revisit broader speech context:
   - `oral_question_exchange`
   - `bill_or_legislation`
   - exact Leaders' Questions allowlist
   - remaining statement/procedural/motion categories after certification.
9. Revisit voting analysis using certified Bill context so division behaviour can be described by legislation/stage rather than only raw party-unity measures.
10. Keep full Parliamentary Question issue classification deferred unless deterministic context foundations prove insufficient for a specific downstream use case.
