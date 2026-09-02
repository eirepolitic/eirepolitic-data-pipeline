# Oral-question section heading investigation

Status: **Research complete; no new production dataset recommended**  
Date: **2 September 2026**  
Parent research: [Parliamentary questions investigation](parliamentary_questions_investigation.md)

This note records the investigation into whether Oireachtas debate-section headings need normalization and whether they can provide a useful deterministic topical layer for Oral Parliamentary Question exchanges.

## Scope

Production batch examined:

`structure-oral-exchange-participants-20260902-1`

Live `oral_question_sections` contains:

- **2,127 oral-question exchanges**;
- **493 distinct non-blank section headings**;
- **0 blank headings**.

No AI classifier or model call was used.

## Main conclusion

The Oireachtas section headings are already sufficiently clean to use **as source-generated labels without a normalization layer**.

The correct public interpretation is:

> **Oireachtas section heading**

They should not be presented as:

- EirePolitic issue classifications;
- department classifications;
- a fixed policy taxonomy;
- inferred question topics.

The recipient/department must remain a separate dimension because the same heading can legitimately appear under different recipients or move between portfolios over time.

## Heading reuse

Of the **493** distinct headings:

- **327** occur in more than one exchange;
- **166** occur once;
- **136** occur in at least 5 exchanges;
- **51** occur in at least 10 exchanges;
- **234** appear in both 2025 and 2026.

Most importantly, **92.20% of all 2,127 exchanges occur under a heading that is reused at least once**.

This makes the raw heading much more useful as a filter than the raw unique-heading count initially suggests.

## Normalization test

A deliberately conservative normalization was tested using only:

- Unicode normalization;
- case folding;
- whitespace normalization;
- punctuation normalization;
- `&` → `and`;
- hyphen/underscore spacing normalization.

Result:

- raw unique headings: **493**;
- conservative-normalized unique headings: **493**;
- reduction: **0**;
- collision groups: **0**.

Therefore there is currently **no case/punctuation/spacing duplication problem** to solve.

### Fuzzy near-duplicate check

A high-threshold fuzzy comparison was also run for review purposes only.

Only one high-similarity pair survived the blocking/threshold rules:

- `Budget 2026`;
- `Budget 2027`.

These are obviously **different headings and must not be merged**.

This is strong evidence against introducing fuzzy normalization: textual similarity would create false semantic merges before solving any real data-quality problem.

## Frequently reused headings

Examples of highly reused headings include:

| Heading | Exchanges |
| --- | ---: |
| Defence Forces | **57** |
| Cabinet Committees | **46** |
| Childcare Services | **38** |
| Special Educational Needs | **33** |
| Public Transport | **31** |
| Housing Provision | **31** |
| Disability Services | **28** |
| An Garda Síochána | **25** |
| Middle East | **24** |
| Bus Services | **20** |

Many persist across both 2025 and 2026 and across numerous months.

Examples with particularly stable recipient relationships include:

- `Defence Forces` → Defence, 100% of recorded question recipients in the sample;
- `Cabinet Committees` → Taoiseach, 100%;
- `Bus Services` → Transport, 100%;
- `Schools Building Projects` → Education, 100%;
- `Social Welfare Payments` → Social, 100%;
- `Military Neutrality` → Defence, 100%;
- `Business Supports` → Enterprise, 100%;
- `Third Level Education` → Further and Higher Education, 100%.

These show that many headings are stable, human-readable topic labels.

## Relationship to question recipient

The heading is useful, but it is **not equivalent to recipient/department**.

Across all headings:

- about **68.8%** have only one recorded recipient in the current history;
- among repeated headings, about **52.9%** have only one recipient.

For headings appearing at least 5 times:

- **77.2%** have at least 75% of their question records going to one top recipient;
- **50.0%** have at least 90% going to one top recipient.

For headings appearing at least 10 times:

- about **60.8%** have at least 90% going to one top recipient.

This is strong enough to show meaningful relationships, but not strong enough to derive recipient from heading.

### Cross-portfolio/generic headings

Some source headings are deliberately broad and span many recipients.

Examples:

- `Departmental Schemes` — 19 exchanges, 8 recipients, top recipient only about 47.6%;
- `Legislative Measures` — 15 exchanges, 7 recipients, top recipient about 41.2%;
- `Artificial Intelligence` — 16 exchanges, 7 recipients, top recipient about 61.9%;
- `Departmental Funding` — 13 exchanges, 6 recipients, top recipient about 35.7%;
- `Departmental Strategies` — 8 exchanges, 5 recipients.

These remain valid source headings, but they are poor substitutes for a department or issue taxonomy.

## Recipient drift across time

Of the **234 headings used in both 2025 and 2026**, **57 (24.4%)** had a different most-common recipient between the two years.

This does not mean the heading itself is unstable. Often it reflects:

- government portfolio changes;
- questions being put to the Taoiseach as well as a line department;
- genuinely cross-departmental subjects;
- small sample counts in one year.

Examples include:

- `Disability Services`: Children was the most common recipient in 2025, Taoiseach in 2026;
- `Sports Funding`: Tourism in 2025, Culture in 2026;
- `Legislative Measures`: Justice in 2025, Public Expenditure in 2026;
- `Renewable Energy Generation`: Climate in 2025, Taoiseach in 2026;
- `Defective Building Materials`: Public Expenditure in 2025, Housing in 2026;
- `Urban Development`: Tourism in the small 2025 sample, Taoiseach in 2026.

### Important modelling rule

Do not attach a permanent recipient/department to a section heading.

The safe model is:

- `section_heading` = source-generated description of the exchange subject;
- question recipient = event/question-level recipient field;
- historical recipient changes remain visible rather than being overwritten by a heading-to-department lookup.

## Grouped-question behaviour

Headings also describe different parliamentary structures.

Examples:

- `Cabinet Committees` is **100% grouped** in the current history, with a median **14.5 questions per exchange**;
- `Military Neutrality` is grouped in about **53.3%** of exchanges;
- many service-delivery headings such as `Bus Services`, `Social Welfare Payments`, `Third Level Education` and `Childcare Services` are overwhelmingly single-question exchanges.

Therefore heading can be useful alongside `grouped_exchange`, but it should not encode grouping status itself.

## Procedural-like headings

The investigation explicitly searched for generic/procedural labels such as:

- generic `Questions` / `Oral Questions`;
- `Priority Questions`;
- generic `Questions to the Minister`;
- `Order of Business` / `Business of the House`.

None appeared as recurring Oral-question section headings in this live dataset under those patterns.

Two recurring special structures were observed:

- `Cabinet Committees` — **46 exchanges**;
- `Taoiseach's Meetings and Engagements` — **10 exchanges**.

These are somewhat procedural in parliamentary form but still useful, intelligible source labels. They should not be stripped or recoded as generic `procedural` without a separate use case.

## Production recommendation

### Do not create a heading-normalization dataset now

There is no demonstrated data-quality benefit:

- raw headings are non-blank;
- conservative normalization reduces zero values;
- fuzzy merging introduces semantic risk;
- the current `section_heading` field already carries the useful deterministic information.

Creating a separate normalization dimension now would add maintenance and versioning without improving the data.

### Use the raw heading directly

Consumers may safely use `oral_question_sections.section_heading` as a filter or display field labelled:

**Oireachtas section heading**

Recommended uses:

- browsing recurring Oral-question subjects;
- exchange drilldowns;
- post/story selection;
- comparison of exchange size/participation by source heading;
- pairing with recipient as a separate dimension.

### Do not call it an issue taxonomy

The heading is source-generated and narrower/broader in inconsistent ways depending on parliamentary practice.

For example:

- `Departmental Schemes` is broad and cross-portfolio;
- `Defence Forces` is stable and department-specific;
- `Middle East` is a broad geographic/policy subject;
- `Cabinet Committees` describes a parliamentary questioning structure more than one policy issue.

These differences are acceptable for a source heading and problematic for a curated issue taxonomy.

## Evidence

Read-only production investigations:

- heading uniqueness/reuse/normalization profile: **33679467828** — SUCCESS;
- recipient stability and year-drift profile: **33679576540** — SUCCESS;
- compact stability digest: **33679660506** — SUCCESS.

No production data changed.

No classifier was called.

## Living next-steps plan

### 1. Compare Oral and Written parliamentary scrutiny

This is now the immediate research task.

Use the certified structures already available rather than building another heading layer.

Compare:

- overall Oral vs Written volumes;
- recipient/department mix by question type;
- member use of Oral vs Written channels;
- party and constituency channel profiles where denominators are safe;
- recurring Oral section headings alongside Written recipient patterns;
- departments with high Written-question volume but comparatively little Oral exchange activity;
- departments with relatively strong Oral exchange presence compared with Written volume;
- grouped Oral-question behaviour by recipient;
- Oral exchange word/intervention volume by recipient;
- whether certain TDs specialize strongly in one channel.

Keep the interpretation descriptive: this measures use of parliamentary-question channels, not political effectiveness.

### 2. Decide whether new reusable channel-comparison components are needed

First attempt the comparison using existing production foundations:

- `daily_question_dimensions`;
- `oral_question_sections`;
- `oral_question_exchange_participants`;
- source question recipient/type fields.

Only add a new materialized dataset if the analysis reveals a recurring calculation that cannot be produced safely from those components.

### 3. Continue deterministic speech-context research

After Oral-vs-Written analysis, return to speech context categories where the source data can prove the context safely, including possible:

- Leaders' Questions;
- Bills/legislation;
- motions;
- statements;
- parliamentary business/procedure;
- other.

### 4. Return to legislation investigation

The broader project plan still includes a proper legislation analysis after the question/exchange structure has been exhausted.

### 5. Question issue classification remains deferred

Section headings provide a useful no-AI topical layer for Oral exchanges, reducing the need for an Oral-question classifier.

Written questions still lack that exchange-heading structure, but do not classify the full question history until the Oral-vs-Written analysis identifies a concrete unmet use case that justifies the cost.
