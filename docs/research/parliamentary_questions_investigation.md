# Parliamentary questions investigation

Status: **Living research record**  
Last major investigation cycle: **September 2026**  
Production structure introduced by: **PR #46** (`e1e75b94…`)  
Current purpose: document what the Oireachtas parliamentary-question data means, what we have learned from it, what has already been implemented, and what should be investigated next before considering large-scale AI classification.

---

## 1. Why this investigation exists

Parliamentary questions initially appeared to be an unusually large dataset compared with Dáil speech activity. That raised a basic concern: were we accidentally counting question-led debate speeches as questions?

The investigation therefore began by answering four structural questions:

1. Is the question count inflated by duplicate API records?
2. Are written and oral questions fundamentally different records?
3. Do oral questions act as anchors for debate sections containing separate transcript speeches?
4. What is the correct unit of analysis when multiple oral questions are grouped into one parliamentary exchange?

The investigation later expanded into:

- written-question batching and redundancy;
- oral-question grouping;
- question-to-debate-section relationships;
- exchange size and word volume;
- question submitter participation;
- ministers/respondents, chairs and other TD participants;
- proxy/substitute taking of oral questions;
- opportunities for new deterministic metrics;
- whether question issue classification is worth the cost.

The central lesson is that **submitted parliamentary questions, oral-question exchanges, and speech interventions within those exchanges are different entities and must remain separate in the data model.**

---

## 2. Settled terminology

### Parliamentary question record

A distinct Oireachtas `question_id` returned by the Oireachtas `/questions` endpoint.

This is the unit counted by the existing question-count metrics.

A parliamentary question record may be:

- **Written**; or
- **Oral**.

### Written parliamentary question

A submitted question record with Oireachtas `question_type = Written`.

In the data examined, written questions do **not** have associated transcript speech interventions in `silver_speeches`.

Written PQs are therefore best treated as a high-volume formal scrutiny/submission channel rather than a debate event.

### Oral parliamentary question

A submitted question record with Oireachtas `question_type = Oral`.

An oral question carries a `debate_section_id` and belongs to a question-led debate section. Multiple oral question records may share the same section.

### Oral-question section / oral-question exchange

One debate section associated with one or more oral question records.

This is the container representing the parliamentary exchange in the chamber.

It can contain:

- one or many submitted oral questions;
- the questioning TD or TDs;
- a minister/Taoiseach/respondent;
- other TD participants;
- chair/procedural interventions;
- many transcript speech/intervention records.

### Speech intervention

A separate `speech_id` parsed from the debate XML.

A speech intervention inside an oral-question exchange is **not** a parliamentary question record. It remains a normal speech/intervention record whose context is `oral_question_exchange`.

### Submitted by vs taken by

The TD attached to the official question record is the **submitter**.

The TD who actually takes or opens the question in the chamber may be someone else, for example when another TD takes the question on behalf of the submitter.

These roles should not be assumed to be identical.

---

## 3. Source architecture confirmed

The question and speech tables are built from separate source paths:

- `silver_questions` comes from the Oireachtas `/questions` endpoint.
- `silver_speeches` comes from debate XML.
- `silver_debate_sections` provides section-level debate structure.

`silver_questions` is deduplicated by Oireachtas `question_id`.

The question count therefore does **not** arise from treating transcript speech rows as question rows.

---

## 4. Question volume audit

### Available history examined

Question data:

- **121,355** question records;
- **121,355 unique `question_id` values**;
- approximately **117,695 Written**;
- approximately **3,660 Oral**;
- question dates from **22 January 2025 through 28 July 2026** in the production snapshot examined.

Speech data in the same broad production snapshot:

- approximately **66,192 speech/intervention rows**;
- debate dates from **18 December 2024 through 28 August 2026**.

The datasets do not have identical date coverage, so the historical totals should not be interpreted as a direct rate comparison.

### July 2026 diagnostic month

July 2026 contained:

- **8,911 question records**;
- **8,911 unique question IDs**;
- **0 duplicate question IDs**;
- **0 repeated exact question texts**;
- **0 duplicate same-member / same-date / same-text records**;
- **8,752 Written questions**;
- **159 Oral questions**;
- **2,839 recorded Dáil speech interventions**.

The apparent question-volume anomaly is therefore real and is explained overwhelmingly by **Written Parliamentary Questions**, not duplication.

### Key interpretation

Written PQs can be submitted in very large numbers without becoming spoken debate interventions.

Therefore:

> `question_count` and `speech_count` are not comparable measures of the same parliamentary activity.

They measure different parliamentary mechanisms.

---

## 5. Written PQ structure and redundancy investigation

The high number of Written PQs raised a second question: even if the IDs are unique, are the texts mainly templated variations that could share one classifier result?

### Exact and normalized uniqueness

Across the written-question history examined:

- **117,695 written records**;
- all exact question texts were unique;
- after conservative normalization of changing numbers, dates, money values, percentages and URLs, approximately **115,277 distinct templates remained**.

Only about **3.5%** of historical written-question rows belonged to a repeated conservative numeric/date template.

July 2026 similarly contained:

- **8,752 written PQs**;
- about **8,632 distinct conservative normalized templates**.

### Batching is common

A submission batch was defined exploratorily as:

> same TD + same question date + same recipient.

In July 2026:

- about **84.7%** of written PQ rows were in batches of at least 2;
- about **55.6%** were in batches of at least 5;
- about **35.6%** were in batches of at least 10;
- about **21.4%** were in batches of at least 20;
- the largest observed same-TD/date/recipient batch contained **140 questions**.

This shows that **batch submission is common**, but batch membership does not imply duplicate content.

### Similarity clustering sensitivity

Within conservative same-TD/date/recipient batches, near-duplicate wording was tested after conservative number/date normalization.

Estimated July classifier-call savings if one call were reused per wording cluster:

| Similarity rule | Potential calls saved | Reduction |
| --- | ---: | ---: |
| exact normalized template | about 112 | about **1.3%** |
| >=98% wording similarity | about 209 | about **2.4%** |
| >=95% wording similarity | about 400 | about **4.6%** |
| >=92% wording similarity | about 527 | about **6.0%** |

Even the aggressive 92% threshold would save only about 6% of calls.

More importantly, very similar wording can still concern substantively different subjects. Examples included questions using the same template for different services or different environmental effects.

### Decision

**Do not build similarity-based issue-label propagation into production.**

If question classification is eventually approved, each question should remain its own factual entity and be classified independently, with exact source-text hash reuse for unchanged records.

---

## 6. Oral-question structure

The production relationship audit identified:

- **3,660 Oral question records**;
- **2,127 distinct oral-question sections/exchanges**.

Therefore the ratio is about **1.7 submitted Oral questions per exchange** overall.

### Typical exchange shape

Across the 2,127 exchanges examined:

- median oral questions per section: **1**;
- about 90% had **2 questions or fewer**;
- maximum grouped oral questions in one section: **25**;
- median transcript interventions: **6**;
- about 90% had **12 interventions or fewer**;
- maximum transcript interventions in one section: **98**;
- median distinct speakers: **2**;
- maximum distinct speakers observed: **20**;
- median total spoken word volume: approximately **1,173 words**.

This means the typical oral question is a relatively compact exchange, but the distribution has a long tail of very large grouped or highly interactive exchanges.

---

## 7. Single-question versus grouped exchanges

The exchange-level investigation separated:

- **1,865 single-question exchanges**;
- **262 grouped-question exchanges**.

### Grouped exchanges

A grouped exchange had approximately:

- median **3 submitted oral questions**;
- median **4 speakers**;
- median **12 transcript interventions**;
- median **2,276 words**.

Grouped exchanges are therefore structurally different from single-question exchanges.

Only about **24%** of grouped exchanges contained transcript participation from **every recorded submitting TD**.

This is important:

> a TD having a submitted question grouped into an oral-question section does not prove that the TD personally spoke in the resulting exchange.

### Cabinet Committees as a special recurring structure

`Cabinet Committees` was one of the clearest examples of a recurring grouped-question format.

In the snapshot examined:

- approximately **46 exchanges** used that heading;
- grouping occurred essentially all the time;
- median submitted questions per exchange was around **14.5**;
- median interventions around **30.5**;
- median word volume around **3,696 words**.

Many TDs submit similar questions such as when a particular Cabinet committee will next meet, and the Taoiseach responds to them collectively.

This should not be modelled as many independent debates.

---

## 8. Representative structural examples

These examples are preserved because they demonstrate how misleading a flat question count can be.

### Urban Development — 15 July 2026

Observed structure:

- **16 oral question records**;
- one debate section;
- recipient: Taoiseach;
- **24 transcript interventions**;
- **10 speakers**.

The submitted questions included different aspects of urban task-force work. The Taoiseach/Chair treated the questions together and the transcript then contained responses, TD interventions and procedural turns.

Correct interpretation:

> 16 submitted questions -> 1 grouped exchange -> 24 transcript interventions.

### Middle East — 2 July 2026

Observed structure:

- **6 oral question records**;
- one grouped exchange;
- recipient: Foreign Affairs;
- **9 transcript interventions**;
- **4 speakers**.

The grouped questions covered related but not identical matters, including occupied-territories legislation, EU-Israel relations, trade in services, settlement goods and implementation timing.

This demonstrates why individual speeches in a grouped section must **not** automatically be attributed to every question in that section.

The speech belongs safely to the **exchange**. A more specific question-to-speech relationship would require additional evidence.

### Cabinet Committees — 10 February 2026

Observed structure:

- **25 oral question records**;
- questions from approximately **24 TDs**;
- one debate section;
- approximately **81 transcript interventions**;
- approximately **12 speakers**.

Many submitted questions were variants of when a Cabinet committee would next meet.

### Arts Funding — 2 October 2025

Observed structure:

- **1 submitted oral question**;
- question associated with Aengus Ó Snodaigh;
- recipient: Culture;
- approximately **81 transcript interventions**;
- approximately **5 speakers**.

This is an example of a **deep individual exchange**: very few submitted questions but a very large back-and-forth transcript.

### Local Authorities — 15 May 2025

Observed structure:

- **1 submitted oral question**;
- question associated with Eoin Ó Broin;
- recipient: Housing;
- approximately **42 transcript interventions**;
- approximately **5 speakers**.

Again, raw oral-question count alone would miss the scale of the exchange.

### Programme for Government — 4 February 2026

One of the largest exchanges by word volume in the data examined:

- **22 submitted oral questions**;
- approximately **21 submitting TDs**;
- **20 speakers**;
- **37 transcript interventions**;
- about **6,687 words**.

### Cabinet Committees — 26 November 2025

Illustrates why intervention count and word volume should remain separate measures:

- approximately **98 transcript interventions**;
- about **4,932 words**.

A high intervention count may include many short procedural/back-and-forth turns and is not identical to a high volume of substantive text.

---

## 9. Important correction: avoid section-speech multiplication

An early exploratory join attached each oral question record to the speech count of its section and then summed that count across question rows.

That produced inflated totals such as:

- **66,994** historical "related speech rows" when summed across question records;
- **1,727** for July 2026 under the same row-expanded calculation.

Those are **not counts of unique speech interventions**. They multiply the same section speeches when multiple oral questions share that section.

The corrected unit is the unique oral-question section.

Later exchange-level analysis identified approximately **18,485 unique transcript interventions** across the 2,127 oral-question exchanges in the production snapshot examined.

For July 2026 specifically, the speech-side relationship audit found about **900 unique speech interventions** in oral-question-linked sections, or roughly **31.7% of July's 2,839 speech interventions**.

### Permanent rule

> Never multiply a section's speech interventions by the number of oral question records sharing the section.

This rule is now documented in the downstream contract and protected by regression tests.

---

## 10. Speech intervention semantics inside oral exchanges

A `speech_id` is a transcript intervention, not necessarily a formal or substantial speech.

Examples inside oral-question exchanges include:

- a substantive TD contribution;
- a ministerial answer;
- a follow-up question;
- a short reply;
- chair/timekeeping intervention;
- procedural interruption.

Therefore public wording should prefer **speech intervention** or **transcript intervention** where precision matters.

Do not interpret 81 interventions as "81 substantive speeches".

---

## 11. Role composition inside oral-question exchanges

Using dated `silver_member_offices` data plus explicit chair naming in transcript metadata, the investigation separated participant roles conservatively.

Across approximately **18,485 unique oral-exchange transcript interventions**:

- ministers/office-holders produced roughly **43% of interventions** but approximately **61% of all words**;
- ordinary members produced roughly **53% of interventions** and approximately **39% of words**;
- chair/procedural speakers produced roughly **4% of interventions** but only around **0.4% of words**.

This reinforces the fact that transcript intervention count and word volume describe different things.

### Typical word balance

Across all exchanges, the median exchange had approximately:

- **62% of words from the minister/respondent side**;
- **35% of words from submitting TDs**.

These are descriptive transcript shares, not measures of quality, responsiveness or effectiveness.

---

## 12. Participation by TDs who did not submit a grouped question

The first exploratory calculation incorrectly treated all non-submitting member-coded speakers as "other TDs", which included ministers because ministers are also TDs/member-coded speakers.

That definition was corrected.

The corrected definition of an **ordinary non-submitting TD participant** excludes:

- recorded question submitters;
- ministers/ministerial office-holders;
- chair/procedural office-holders.

Under the corrected definition:

- about **30.7% of all oral-question exchanges** contained at least one ordinary TD who had not submitted a question in that section;
- about **27.3% of single-question exchanges** contained such participation;
- about **55.3% of grouped exchanges** contained such participation.

So an oral-question exchange cannot safely be modelled as simply:

> submitter + minister.

Other TDs frequently participate, especially in grouped exchanges.

### Examples

A `Road Projects` exchange on 27 March 2025 had:

- one submitted oral question;
- approximately six additional ordinary TD participants;
- non-submitting TDs contributing roughly **36% of the words**.

A `Water Charges` exchange had:

- one submitted question;
- approximately five additional ordinary TD participants;
- those non-submitting TDs contributing more than half of the exchange's words.

These are useful examples of why exchange participation deserves its own model.

---

## 13. Submitter absent from transcript: substitution/proxy-taking

The investigation found **221 of 2,127 exchanges**, about **10.4%**, where none of the recorded question submitters appeared under their own member ID in the related transcript.

This initially looked like a possible data-linkage problem.

Manual inspection showed that many are legitimate parliamentary substitutions: another TD takes the question on behalf of the submitter.

### Conservative deterministic phrase audit

A deliberately conservative phrase-based scan of the first transcript interventions confirmed explicit substitution/proxy language in at least:

- **77 exchanges**;
- covering approximately **83 submitted question records**.

Examples of detectable language included formulations equivalent to:

- "on behalf of" another deputy;
- "I am covering for Deputy ...";
- "I am asking this question for Deputy ...";
- the chair announcing that another TD is taking the question.

At least one no-submitter exchange was clearly procedural/interrupted rather than a normal completed exchange.

### Remaining unexplained cases

Approximately **143** no-submitter exchanges were not resolved by the conservative English-language rules.

They must **not** be interpreted as errors automatically.

Manual examples indicate that some are also substitutions that were missed because:

- the chair announces the substitution in Irish;
- wording does not contain the simple English trigger phrases;
- the substitute TD begins directly with the substantive question.

### Structural conclusion

The data model needs to distinguish:

- `submitted_by_member_code`;
- `taken_by_member_code`;
- whether the question was taken by a substitute;
- evidence used to establish that relationship.

The submitter and in-chamber question taker are not always the same person.

---

## 14. Section headings as a source-generated topical layer

The Oireachtas debate-section headings provide a useful human/source-generated subject signal for oral exchanges.

Frequently recurring headings in the snapshot examined included approximately:

| Heading | Exchange count |
| --- | ---: |
| Defence Forces | 57 |
| Cabinet Committees | 46 |
| Childcare Services | 38 |
| Special Educational Needs | 33 |
| Public Transport | 31 |
| Housing Provision | 31 |
| Disability Services | 28 |
| An Garda Síochána | 25 |
| Middle East | 24 |
| Bus Services | 20 |

These headings are not a replacement for the EirePolitic issue taxonomy because they are more granular and not necessarily standardized across all contexts.

However, they are valuable metadata and should be exploited before introducing expensive classification of oral-question exchanges.

---

## 15. Recipient patterns already available without classification

The existing question data includes `to_minister_or_department`.

This has already produced useful descriptive findings without AI classification.

For July 2026:

- Health received approximately **2,167 question records**, about **24.3%** of eligible questions;
- Health was the largest recipient across the recent active months examined;
- several TDs showed strong recipient specialization.

Examples from July recipient-coded questions included approximately:

- David Cullinane: **89 of 102** to Health, about **87.3%**;
- Pádraig Rice: **182 of 210** to Health, about **86.7%**;
- Gary Gannon: **25 of 31** to Justice, about **80.6%**;
- Pearse Doherty: **44 of 67** to Finance, about **65.7%**;
- Rory Hearne: **24 of 40** to Housing, about **60%**;
- Eoghan Kenny: **44 of 80** to Education, about **55%**.

This suggests that recipient data may already answer many useful questions about **where formal scrutiny is directed**, even before any issue classifier exists.

---

## 16. Question activity concentration

In July 2026:

- the top 10 question submitters generated approximately **28.6%** of eligible question records.

This is useful descriptively, but raw question-volume rankings should be treated cautiously.

Question volume can reflect:

- bulk Written PQ practice;
- portfolio specialization;
- party research organization;
- constituency casework strategies;
- procedural access to Oral questions;
- other parliamentary working styles.

It does not directly measure effectiveness or quality.

---

## 17. Cross-metric finding: questions are a different activity channel

A July 2026 cross-metric analysis found essentially no relationship between member speech volume and Parliamentary Question volume.

Spearman correlation:

- speeches vs questions: approximately **-0.04**.

Recorded voting participation was also only weakly related to either speaking or question volume.

### Decision

Do **not** create a single parliamentary "activity score" combining speech, question and voting metrics.

These appear to represent different forms of parliamentary activity.

Prefer descriptive profiles or scatterplots showing different activity channels separately.

---

## 18. Production changes already implemented

PR #46, merged to `main` as commit beginning `e1e75b94`, introduced the deterministic structural model.

### New production foundations

#### `oral_question_sections`

One row per debate section anchored by one or more Oral question records.

Key concepts include:

- `debate_section_id`;
- debate date;
- section heading;
- oral question count;
- grouped question IDs;
- asking-member count;
- related unique speech/intervention count;
- related speaker count;
- provenance.

#### `speech_question_context`

One row per speech/intervention.

Current deterministic context values:

- `oral_question_exchange`;
- `other`.

The dataset records the question IDs grouped into the same exchange as the speech.

### Important semantic warning

The grouped question-ID field means:

> these questions belong to the same oral-question exchange as this speech.

It does **not** prove that the individual speech directly responds to every question ID in that section.

A future field rename such as `oral_exchange_question_ids_json` or `section_question_ids_json` would reduce this ambiguity.

### Candidate lifecycle

Political-metric candidate manifests now require **seven** metric/foundation datasets instead of the original five:

1. `daily_activity_components`
2. `daily_issue_activity`
3. `division_party_vote_components`
4. `daily_question_dimensions`
5. `oral_question_sections`
6. `speech_question_context`
7. `monthly_metric_results`

### Production deployment evidence

- PR #46 merged successfully.
- Disposable production-seeded candidate integration: run **33550268537**, passed.
- Final feature-branch unit/history/relationship audit: run **33550915907**, passed.
- Structure-only candidate build/full downstream validation: run **33551096144**, passed.
- Production promotion and seven-dataset verification: run **33551448689**, passed.
- Final production inventory audit: run **33551546645**, passed.
- Final merged-main metrics/history/question-relationship audit: run **33551538202**, passed.

Production batch after the structural promotion:

`structure-question-context-33550915907-1`

No question classifier was run during this work.

---

## 19. Public terminology decisions

Existing metric IDs are preserved for compatibility, but their public definitions were clarified.

`member_question_count`, `party_question_count`, and `constituency_question_count` mean:

> distinct submitted Oireachtas parliamentary-question records (`question_id`).

They do **not** mean:

- question-led debate sections;
- oral-question exchanges;
- transcript interventions within those exchanges.

Recommended public terms:

- Parliamentary Questions submitted;
- Written Parliamentary Questions;
- Oral Parliamentary Questions;
- Oral-question exchanges;
- Transcript interventions during oral-question exchanges.

Avoid the word "questions" alone where the unit could be ambiguous.

---

## 20. Questions and issues: classification remains deferred

A question issue classifier has deliberately **not** been built or run.

Reasons:

1. The dataset contains approximately 121k records, creating a meaningful one-time classification cost.
2. The investigation has already uncovered substantial useful structure from deterministic fields such as recipient, type, section heading, grouping and transcript participation.
3. We should first determine which additional public/research questions genuinely require issue classification.
4. Similarity/template reuse would save too few calls and could introduce incorrect label propagation.

If classification is eventually approved, the recommended safe design remains:

- one factual question record per `question_id`;
- one independent classification result per question text;
- same core issue taxonomy as speech classification where appropriate;
- source-text hash;
- classifier/model version;
- provenance/status;
- changed/new only after initial backfill;
- exact hash reuse only;
- no approximate similarity-based propagation.

Oral and Written questions should remain filterable separately.

---

## 21. Potential future generated dimensions

These are ideas, not approved production classifiers.

### Question issue

Could allow comparison of:

- what parties/TDs talk about in speeches;
- what they formally ask ministers about in Parliamentary Questions.

Recipient and issue are not equivalent. For example, a Health question can concern hospitals, staffing, waiting lists, disability services, expenditure, medicines or other topics.

### Question purpose

Possible descriptive categories:

- statistics/data request;
- funding/expenditure;
- waiting-list/status request;
- staffing;
- implementation timeline;
- policy explanation;
- eligibility;
- individual case/status;
- legislation;
- meeting/correspondence request;
- other.

This should only be built after its analytical value is demonstrated.

### Response form

Potential objective categories for answers:

- direct figure supplied;
- substantive narrative response;
- referred elsewhere;
- information unavailable;
- cannot provide;
- matter for another body;
- promised later response;
- procedural response.

Avoid subjective classifiers such as "good answer" or "effective answer".

---

## 22. Promising deterministic post/research formats

The investigation already supports several content families without question issue classification.

### Written-question activity

- Which departments receive the most Written PQs?
- Which TDs direct the largest share of their questions to one portfolio?
- How does departmental scrutiny pressure change over time?
- Which TDs submit broad government-wide batches versus concentrated portfolio batches?

### Oral-question activity

- Which submitted oral questions generated the longest exchanges?
- Which oral-question headings repeatedly return to the Dáil?
- Which topics attract questions from the largest number of TDs?
- Which exchanges involve the most additional TDs beyond the original submitters?
- Which exchanges have unusually high respondent word share versus TD word share?
- Which exchanges are deep one-question discussions versus broad grouped-question sessions?

### Parliamentary procedure explainers

- One oral question does not equal one debate.
- Many questions can be grouped into one exchange.
- A submitted question may be taken in the chamber by another TD.
- Transcript intervention count is not the same thing as substantive speech count.

These explainers could be valuable public methodology content as well as internal guidance.

---

## 23. Proposed next deterministic data structures

### A. Extend `oral_question_sections`

Recommended fields/components to investigate and potentially materialize:

- `submitting_member_count`;
- `participating_submitting_member_count`;
- `participating_submitter_share`;
- `ordinary_non_submitter_td_count`;
- `related_speech_word_count`;
- `ministerial_intervention_count`;
- `ministerial_word_count`;
- `ministerial_word_share`;
- `chair_intervention_count`;
- `chair_word_count`;
- `grouped_exchange`;
- `questions_per_exchange`.

These are deterministic or derivable from current sources, provided office-role attribution is certified.

### B. Question-taking relationship

Proposed new foundation concept:

- `question_id`;
- `submitted_by_member_code`;
- `taken_by_member_code`;
- `is_substitute`;
- `evidence_method`;
- `confidence/status` limited to deterministic evidence categories;
- source section/speech evidence reference.

Start conservatively with explicit transcript evidence.

Unknown should remain unknown; do not infer a substitute merely because the submitter is absent.

### C. General speech context

Current deterministic context only identifies `oral_question_exchange` vs `other`.

Future investigation should determine whether section metadata can safely distinguish:

- Leaders' Questions;
- Bills/legislation;
- motions;
- statements;
- oral-question exchanges;
- procedural/business;
- other.

Prefer deterministic section/source metadata before AI classification.

---

## 24. Methodological cautions

1. **Question record != debate section != speech intervention.**
2. Multiple Oral question records may share one debate section.
3. Never multiply section-level speech counts by question records.
4. A speech in a grouped exchange should not automatically be attributed to every individual question in that exchange.
5. Submitted-by member is not necessarily the same as the member who takes the question in chamber.
6. A missing submitter in the transcript is not automatically a data error.
7. Ministers are also member-coded speakers; "non-submitting member" does not automatically mean ordinary opposition/backbench TD.
8. Chair interventions are often short procedural turns and can inflate intervention counts.
9. Word volume and intervention count measure different structural properties.
10. Written and Oral PQs should remain separable in analysis.
11. Question count is not a measure of question quality, effectiveness or answer quality.
12. Recipient share is not equivalent to policy-issue share.
13. Section headings are useful but not a standardized issue taxonomy.
14. Do not create a composite parliamentary activity/performance score from questions, speeches and votes.

---

## 25. Evidence and investigation runs

The following workflow runs are important evidence for this research cycle.

### Initial metric opportunity/question analysis

- question opportunity analysis and recipient specialization: runs around **33539925603**, **33540043679**, **33540135196**.

### Question volume audit

- question/speech volume and uniqueness audit: **33547236392**.

Key conclusion: July's 8,911 question records were 8,911 unique IDs; volume was driven by 8,752 Written PQs.

### Written PQ structure

- written-PQ batching/template analysis: **33547869141**.
- similarity clustering sensitivity: **33548050476**.

Key conclusion: approximate template reuse produces only modest savings and is not safe enough to justify automatic label propagation.

### Question-section relationship audit

- question-to-debate-section/speech relationship audit: **33549306719**.

Key conclusion: Written questions had no related speech rows; Oral questions anchor sections containing separate transcript interventions.

### Structural production implementation

- candidate integration: **33550268537**.
- final feature audit: **33550915907**.
- structure-only full downstream validation: **33551096144**.
- production promotion: **33551448689**.
- final production audits: **33551546645**, **33551538202**.

### Oral exchange content/participation investigations

- exchange structure/content profiling: **33576191998**.
- compact digest: **33576247281**.
- event-date role/participation analysis: **33576915361**.
- corrected ordinary-non-submitter analysis: **33577017364**.
- proxy/substitution follow-up: **33577219522** after correcting a temporary analysis-script key mismatch.

Temporary investigation-script failures or diagnostics were not production failures and should not be confused with source/pipeline defects.

---

## 26. Living next-steps plan

The order matters. Investigation should precede production metric expansion, and documentation should be updated before external summary.

### Next 1 — certify question-taking/substitution structure

Goal: determine how reliably `taken_by_member_code` can be derived without AI.

Tasks:

- expand deterministic proxy/substitution phrase coverage, including Irish-language chair formulations;
- inspect unresolved no-submitter cases by heading/recipient/time period;
- distinguish explicit substitute-taking from interrupted/not-reached/procedural cases;
- test whether transcript ordering reliably identifies the substitute TD;
- establish conservative evidence/status values;
- determine whether a production `question_taking_relationships` foundation is justified.

Do not infer a taker where evidence is insufficient.

### Next 2 — certify exchange participant metrics

Goal: make oral-question exchange structure reusable instead of leaving it in exploratory JSON.

Tasks:

- certify event-date minister/chair attribution rules;
- validate word-count completeness;
- define ordinary non-submitting TD precisely;
- validate grouped/single exchange flag;
- define additive/non-additive properties;
- decide which fields belong directly in `oral_question_sections` and which belong in a separate participant foundation;
- add tests preventing double counting.

### Next 3 — investigate section-heading normalization

Goal: determine whether Oireachtas headings can provide a useful no-AI topic hierarchy for Oral questions.

Tasks:

- measure heading uniqueness and reuse;
- identify spelling/format variants;
- test stable normalization rules;
- map headings to recipients and existing speech issue categories descriptively;
- decide whether a deterministic heading taxonomy would be useful for public filters.

Do not force headings into the issue taxonomy if they represent different concepts.

### Next 4 — investigate Written PQ behaviour more deeply

Goal: understand how the high-volume Written channel is used before classifying it.

Tasks:

- member-day and member-recipient batch distributions;
- recurring portfolio campaigns over time;
- breadth of departments questioned per TD/month;
- focused vs broad submission patterns;
- recipient concentration and change over time;
- oral vs written usage profiles by TD/party/constituency;
- inspect whether answer metadata creates useful deterministic response measures.

### Next 5 — compare Oral scrutiny with Written scrutiny

Goal: establish whether Oral and Written questions reveal meaningfully different parliamentary behaviour.

Potential comparisons:

- recipient mix;
- TD specialization;
- party mix;
- constituency mix;
- question frequency;
- exchange participation;
- recurring headings;
- ministers/departments that receive many Written questions but few Oral exchanges, and vice versa.

### Next 6 — integrate with broader parliamentary behaviour research

After the question structure is certified, connect it with:

- speech activity/context;
- voting behaviour;
- legislation;
- existing speech issue data;
- historical office/role context.

Do not create one combined performance score.

### Deferred — question issue classifier

Only reconsider once the previous deterministic investigations establish a concrete need.

Before approval, document:

- which user/public questions cannot be answered without issue classification;
- expected number of useful outputs/posts/dashboards;
- one-time backfill cost;
- ongoing changed/new cost;
- classifier validation sample and accuracy;
- treatment of Oral vs Written questions;
- whether question text alone is sufficient for the intended issue label.

Until then: **do not classify the 121k question history.**

---

## 27. Research workflow for future updates

For this topic, future work should follow this sequence:

1. **Investigate** using production data or an isolated candidate/read-only audit.
2. **Update this document** with confirmed findings, corrections, caveats and evidence run IDs.
3. **Update the living next-steps plan** based on what changed.
4. **Only then return a concise external summary** of the important result and immediate next action.

The detailed research record should remain here so short operational summaries do not become the only institutional memory of the work.
