# Oral-question exchange participant metrics

Status: **Certified research findings; production design not yet implemented**  
Date: **2 September 2026**  
Parent research record: [Parliamentary questions investigation](parliamentary_questions_investigation.md)

This note records the certification investigation for deterministic participant and word-volume measures inside Oral Parliamentary Question exchanges.

## Scope

The production snapshot examined contains:

- **2,127 oral-question exchanges**;
- **18,485 unique transcript interventions** inside those exchanges;
- **2,763,441 words** across those interventions;
- an exploratory participant aggregation of **6,131 exchange-participant rows** before separating mixed roles.

No classifier or model call is required for these measures.

## Certification result

The proposed exchange-level measures reconciled exactly back to the production speech source.

Confirmed invariants:

- sum of exchange `related_speech_count` = unique oral-exchange `speech_id` count;
- sum of exchange word counts = source `silver_speeches.word_count` total;
- ministerial + chair + ordinary-member + unmatched intervention totals = unique oral-exchange speech total;
- the same role partition reconciles exactly for word counts;
- participant-level intervention totals reconcile exactly to speech rows;
- participant-level word totals reconcile exactly to source word counts;
- one exchange row per `debate_section_id` is unique.

These checks all passed in workflow run **33580902432**.

## Word-count quality

`silver_speeches.word_count` is exceptionally clean for this use case:

- **0 missing** word counts across the 18,485 oral-exchange interventions;
- **0 zero-word** rows;
- source word counts matched a direct whitespace-token recount in **100%** of rows examined;
- total = **2,763,441 words**.

This supports using exchange word volume as a deterministic structural measure.

It remains a transcript-volume measure, not a measure of argument quality, importance or influence.

## Certified participant role buckets

Role attribution uses both:

1. event-date `silver_member_offices`; and
2. explicit transcript speaker labels for chair/procedural roles.

Current deterministic role buckets:

- `ministerial`;
- `chair`;
- `ordinary_member`;
- `collective_or_unidentified` (recommended replacement for the exploratory `unmatched` label).

### Aggregate role totals

Across all oral-question exchanges:

| Role | Interventions | Words |
| --- | ---: | ---: |
| Ordinary member | **9,262** | **1,061,771** |
| Ministerial/respondent | **7,967** | **1,684,996** |
| Chair/procedural | **1,251** | **16,659** |
| Collective/unidentified | **5** | **15** |

This confirms an important structural distinction:

- ordinary members produce more transcript interventions;
- ministerial/respondent speakers produce substantially more words;
- chair interventions are numerous enough to affect intervention counts but contribute very little text volume.

## Proposed exchange-level fields

The following measures are now considered technically certifiable from current sources:

- `question_count`;
- `grouped_exchange`;
- `submitting_member_count`;
- `participating_submitting_member_count`;
- `participating_submitter_share`;
- `ordinary_non_submitter_td_count`;
- `related_speech_count`;
- `related_speech_word_count`;
- `ministerial_intervention_count`;
- `ministerial_word_count`;
- `ministerial_word_share`;
- `chair_intervention_count`;
- `chair_word_count`;
- `ordinary_member_intervention_count`;
- `ordinary_member_word_count`;
- `collective_or_unidentified_intervention_count`;
- `collective_or_unidentified_word_count`.

Most naturally belong directly in the existing `oral_question_sections` foundation because they are one-row-per-exchange attributes.

## Distribution of exchange structure

Across all 2,127 exchanges:

### Questions per exchange

- median: **1**;
- 90th percentile: **2**;
- 95th percentile: **6**.

### Transcript interventions

- median: **6**;
- 90th percentile: **12**;
- 95th percentile: **21**.

### Word volume

- median: **1,173 words**;
- 75th percentile: **1,313 words**;
- 90th percentile: about **1,670 words**;
- 95th percentile: about **2,525 words**.

### Ministerial word share

- median: **62.4%**;
- 25th percentile: **57.7%**;
- 75th percentile: **66.7%**;
- 90th percentile: **71.2%**.

This is descriptive transcript composition only. It should not be interpreted as answer quality, responsiveness, dominance or effectiveness.

### Participating submitter share

- median: **100%**;
- 10th percentile: **0%**;
- 25th percentile: **100%**.

The unusual distribution reflects two structural facts already established:

- most exchanges are single-question exchanges where the submitter participates;
- a minority involve substitution, non-participating submitters, or grouped-question procedures.

## Single versus grouped exchanges

### Single-question exchanges

Count: **1,865**.

Typical structure:

- median **6 interventions**;
- median **1,147 words**;
- median participating-submitter share: **100%**;
- median ministerial word share: **62.6%**;
- about **21.4%** contain at least one ordinary TD participant who was not the recorded submitter.

### Grouped-question exchanges

Count: **262**.

Typical structure:

- median **12 interventions**;
- median **2,275.5 words**;
- median participating-submitter share: **66.7%**;
- median ministerial word share: **59.3%**;
- about **43.9%** contain at least one ordinary TD participant who was not a recorded submitter.

Grouped exchanges are therefore materially different parliamentary events and should remain explicitly identifiable.

## Large-exchange examples

The largest exchange in this snapshot by word volume was the 4 February 2026 section previously identified in the question research:

- **22 submitted oral questions**;
- **21 submitting members**;
- **18 participating submitting members**;
- **37 transcript interventions**;
- **6,687 words**;
- ministerial word share about **48.4%**.

Other very large grouped exchanges contained more than 5,000 words and dozens of interventions.

A 26 November 2025 grouped exchange contained:

- **13 questions**;
- **98 transcript interventions**;
- **4,932 words**;
- **18 chair interventions**;
- **47 ordinary-member interventions**;
- **33 ministerial interventions**.

This reinforces the earlier conclusion that intervention count and word volume are different structural measures.

## Residual role anomalies

A targeted follow-up inspected all residual anomalies from the certification pass.

### Same member, two roles in one exchange

There were exactly **2 exchange-participant combinations** where one member legitimately had two role buckets during the same section.

#### Erin McGreehan — 18 February 2025

Within one exchange, the same member appeared as:

- `Deputy Erin McGreehan` contributing as an ordinary member; and later
- `An Cathaoirleach Gníomhach (Deputy Erin McGreehan)` making chair interventions.

#### Jennifer Whitmore — 10 April 2025

Within one exchange, the same member appeared as:

- `Deputy Jennifer Whitmore` contributing as an ordinary member; and later
- `An Cathaoirleach Gníomhach (Deputy Jennifer Whitmore)` making a chair intervention.

### Schema consequence

A participant-level foundation must **not** use only:

`debate_section_id + member_code`

as its unique key.

The correct grain should preserve the role, for example:

`debate_section_id + participant_key + participant_role`

This prevents ordinary-member contributions and acting-chair interventions by the same person from being merged into one ambiguous participant row.

## Collective and unidentified speakers

Only **5 oral-exchange interventions** lacked an identifiable member and role under the deterministic member-role logic.

They were legitimate transcript entities, not missing rows:

- `Deputies: Hear, hear.` — three instances;
- `Deputies: Shame, shame, shame.` — one instance;
- `A Deputy: Are they working in NATO headquarters?` — one instance.

These should not be attributed to an individual TD.

Recommended role/status:

`collective_or_unidentified`

They should remain in total exchange intervention/word counts but be excluded from member-level TD participant counts.

## Recommended participant-level foundation

A participant foundation would add analytical value beyond exchange aggregates, especially for member profiles and participation networks.

Recommended grain:

> one row per oral-question exchange + participant identity + participant role.

Recommended fields:

- `debate_section_id`;
- `debate_date`;
- `participant_key`;
- `member_code` when identified;
- `speaker_name`;
- `participant_role`;
- `is_recorded_submitter`;
- `intervention_count`;
- `word_count`;
- provenance/version fields.

Recommended role values:

- `ministerial`;
- `chair`;
- `ordinary_member`;
- `collective_or_unidentified`.

### Why a separate participant foundation is useful

The exchange-level fields answer questions such as:

- how large was the exchange?
- how many submitters participated?
- how much of the transcript came from the respondent?
- how many other TDs joined?

The participant-level foundation would answer:

- which specific TDs participated in the exchange?
- how many interventions/words did each contribute?
- which TDs regularly participate in exchanges they did not submit questions for?
- how often does an office-holder participate as respondent?
- how often does a member act as chair and also contribute substantively in the same section?

This should remain distinct from `question_taking_relationships`. Participation is observable; taker attribution is a separate, more constrained concept.

## Additivity and interpretation rules

### Additive across disjoint exchanges

These component counts can be summed across non-overlapping exchange sets:

- question count;
- speech/intervention count;
- word count;
- ministerial intervention/word counts;
- chair intervention/word counts;
- ordinary-member intervention/word counts.

Participant counts require care because the same person may appear in multiple exchanges.

### Non-additive / recalculate

Do not sum or average blindly across exchanges:

- `participating_submitter_share`;
- `ministerial_word_share`;
- unique participant counts;
- unique ordinary non-submitting TD counts.

For combined periods, recompute shares from their numerator/denominator components.

### Public interpretation cautions

Do not call these:

- answer quality;
- effectiveness;
- dominance;
- scrutiny success;
- speaking performance.

Safe language includes:

- transcript word share;
- recorded intervention count;
- exchange participation;
- exchange size;
- grouped-question exchange;
- recorded respondent/ministerial contribution.

## Production recommendation

The evidence now supports a concrete architecture recommendation:

### Extend `oral_question_sections`

Add the certified one-row-per-exchange component fields listed above.

### Add `oral_question_exchange_participants`

Create a separate deterministic participant foundation at:

> exchange + participant + role grain.

This is preferable to trying to encode participant lists into JSON arrays because it is easier to validate, aggregate and consume in Power BI/Appsmith/API models.

The participant foundation should include collective/unidentified transcript speakers without assigning them a TD identity.

### Do not combine with question-taker attribution

The participant foundation should describe **observed participation only**.

The separately researched `taken_by_member_code` concept should remain bounded to explicitly evidenced cases and should not block participant metrics.

## Evidence

- main exchange/participant certification: workflow run **33580902432**;
- residual mixed-role/unmatched-speaker investigation: workflow run **33580988078**.

Both were read-only production-data investigations apart from committing analysis files to temporary branches. No production data or pointer changed.

## Revised next-steps plan

### 1. Design the production schema and tests for exchange metrics

This is the immediate next step if implementation is approved.

Design:

- expanded `oral_question_sections` columns;
- new `oral_question_exchange_participants` dataset;
- exact primary keys;
- role precedence and role naming;
- component/additivity metadata;
- downstream contract paths;
- materialization manifest requirements;
- unit/regression tests for reconciliation.

Required regression cases:

- one speech belongs to one exchange only;
- source speech/word totals reconcile exactly;
- same member can occupy ordinary-member and chair roles in one exchange;
- collective/unidentified speakers are retained but never counted as identified TDs;
- grouped exchanges do not multiply question or speech rows.

### 2. Implement in an isolated candidate, not directly in production

After schema review:

- build on a feature branch;
- seed a disposable candidate from production;
- materialize revised exchange + participant foundations;
- verify exact reconciliation against source speech rows;
- run the full political-metrics/downstream validation stack;
- document findings before promotion.

### 3. Then investigate section-heading normalization

Once exchange participant structure is production-ready, return to the parent plan and investigate whether Oireachtas section headings can provide a stable no-AI topic hierarchy for Oral questions.

### 4. Then compare Oral vs Written scrutiny

With certified exchange participation available, compare:

- Oral vs Written recipient patterns;
- TD use of each channel;
- portfolio specialization;
- grouped oral participation;
- departments receiving high Written PQ volume but relatively little Oral exchange activity, and vice versa.

### 5. Question issue classification remains deferred

The deterministic data continues to yield useful structure. There is still no reason to classify the full ~121k question history until a concrete unmet use case justifies the cost.
