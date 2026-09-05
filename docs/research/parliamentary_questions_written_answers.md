# Written Parliamentary Question answers investigation

## Status

Read-only investigation completed on 2026-09-05 against production batch:

- `contextual-monthly-voting-20260904-1`

Production was not changed.

Key evidence runs:

- `33996825667` — successful 100-question XML accessibility/parse audit.
- `33996894128` — successful optimized repeat of the 100-question source audit.
- `33996928835` — XML structure inspection on representative Written-answer sections.
- `33997006966` — successful 100-question answer-status/edge-case profile.

Two earlier temporary audit runs (`33996741779`, `33996782973`) failed because the diagnostic script initially used the wrong case/schema assumptions. They did not change production and were not source-data failures.

## Why this investigation was needed

The production `silver_questions` table already contains:

- question ID;
- question date/number;
- Written vs Oral type;
- question text;
- submitting member;
- recipient minister/department;
- `debate_section_id`;
- official XML source URI/URL.

However, a dedicated production audit confirmed:

- 121,355 total question rows;
- 117,695 Written questions;
- 3,660 Oral questions;
- all 121,355 rows have a source XML URI;
- **0 rows currently have populated `answer_text`** in `silver_questions`.

Therefore Written-answer content is not available from the current `/questions` JSON extraction alone. The official section XML must be treated as a separate source layer.

## Source structure confirmed

For Written Questions, `source_xml_url` points to a **section-specific Akoma Ntoso XML document**, for example a URL ending in:

`.../writtens/mul@/dbsect_225.xml`

This is not a whole sitting-day XML file. It is the XML for the particular Written-answer debate section.

In the representative XML structure audit:

- every sampled document contained one `debateSection` matching the local `dbsect_*` suffix from the production `debate_section_id`;
- the section is explicitly marked `name="writtenAnswer"`;
- the submitted question is represented by one or more `<question>` elements;
- the ministerial answer is normally represented by a `<speech>` element;
- the answer text is contained inside the `<speech>` element and its paragraph children;
- respondent people/roles are available through Akoma Ntoso references such as `TLCPerson` and `TLCRole`;
- grouped-answer relationships may be represented by multiple `<question>` elements plus `<summary>` text.

This is a strong deterministic structure. A language model is not required simply to separate question text from answer text.

## 100-question source reliability sample

A deterministic random sample of 100 Written Questions was drawn from the 117,695 production Written records.

Results:

- **100/100 source XML URLs fetched successfully**;
- **100/100 XML files parsed successfully**;
- **100/100 section-local `dbsect_*` IDs matched the production `debate_section_id` suffix**;
- 0 XML fetch failures in the sample.

This is strong evidence that the current XML source references are operationally usable.

It is not yet a whole-history guarantee; a production backfill would still need explicit fetch/error/retry accounting.

## Answer-status profile

Across the 100 sampled Written-answer sections:

- **98** contained a ministerial `<speech>` reply;
- **2** explicitly recorded `Reply not received from Department.`;
- **25** showed grouped/joint-answer structure;
- **22** contained more than one `<question>` element;
- **14** contained referral/direct-reply language, for example the minister asking the HSE or another body to respond directly;
- **5** contained embedded XML tables;
- no sampled section contained more than one answer `<speech>` element.

### Answer length

For the 98 sampled sections with a ministerial reply, extracted `<speech>` text length was approximately:

- median: **1,704 characters**;
- mean: **1,851 characters**;
- 25th percentile: **579 characters**;
- 75th percentile: **2,739 characters**;
- minimum: **96 characters**;
- maximum: **4,658 characters**.

The answers therefore range from very short procedural/referral replies to multi-paragraph substantive policy/data responses.

## Grouped answers are a first-class relationship

Written Questions do not always have a one-question-to-one-answer relationship.

Examples in the source include wording such as:

- a minister proposing to take several question numbers together;
- `Question No. 694 answered with Question No. 693.`;
- several `<question>` elements followed by one ministerial `<speech>` reply.

In the 100-question sample:

- 25 rows showed grouped/joint-answer evidence;
- 22 XML sections contained multiple question elements.

Therefore copying the same answer text independently onto every question row would lose important structure and could create misleading duplicate answer counts.

The safer conceptual relationship is:

`one Written-answer section -> one ministerial reply -> one or more submitted question records`

## Question-ID mapping anomaly

The production `debate_section_id` matched the XML section in **100/100** sampled rows.

The production question ID suffix matched a `<question eId="pq_*">` inside the section in **93/100** rows.

Seven sampled rows did not contain the expected local `pq_*` ID even though:

- the section URL fetched successfully;
- the section `dbsect_*` matched exactly;
- the section contained a valid Written-answer structure;
- a ministerial reply was present.

Some mismatches may be explained by grouped/taken-with-answer relationships, but several appeared as single-question XML sections and should be treated as unresolved source/linkage anomalies until examined individually.

### Guardrail

Do **not** certify a one-question-to-answer relationship solely from the question-ID suffix.

The section-level source relationship is currently more robust than the individual question-ID-in-XML relationship.

## Referral/direct-response cases

Fourteen of the 100 sampled sections contained referral/direct-response language.

Typical form:

- minister gives a short jurisdictional/policy response;
- minister states that the HSE, NTA, Irish Rail, another agency, or another body has been asked to reply directly;
- sometimes the XML summary confirms a referred reply was forwarded later.

This means there are at least two distinct response concepts:

1. **the official ministerial Written-answer text published in the Oireachtas section**;
2. **a potentially later substantive reply from an external/state body**.

A future answer dataset should not treat a referral as equivalent to the later agency reply unless that later document is separately available and linked.

## Missing replies

Two of the 100 sampled sections explicitly stated:

`Reply not received from Department.`

This should be represented as an objective source status, not as an empty-string parsing failure.

Recommended deterministic status concept:

- `ministerial_reply_present`;
- `reply_not_received`;
- `source_fetch_failed`;
- `source_parse_failed`;
- `unresolved_structure`.

Referral/grouping should be separate flags rather than replacing the answer-status field.

## Tables and attachments

Five sampled answer sections contained embedded XML `<table>` elements.

This confirms that some Written answers contain structured tabular data directly inside the source XML.

The audit also observed many generic XML links, but those include normal Akoma Ntoso metadata/reference links and should **not** be treated as attachments automatically.

A separate attachment-specific audit is still needed to distinguish:

- embedded tables;
- downloadable XLS/XLSX/CSV/DOC/DOCX/PDF files;
- ordinary metadata/navigation links.

Do not create an attachment dataset from generic `href` counts alone.

## What is safe to automate now

The following extraction appears deterministic enough to design without AI:

- fetch the official section-specific XML from `source_xml_url`;
- verify the `debateSection` local `dbsect_*` ID against production `debate_section_id`;
- verify `name="writtenAnswer"`;
- enumerate all `<question>` elements and their local `pq_*`, `by`, and `to` attributes;
- extract the ministerial `<speech>` text when present;
- extract respondent role/person references;
- extract heading;
- preserve `<summary>` text;
- count/preserve embedded tables;
- classify `Reply not received from Department.` deterministically;
- flag grouped/joint-answer structure;
- flag referral/direct-response wording conservatively.

## What should not be automated yet

Do not yet:

- infer that every production question ID maps directly to a same-ID `<question>` element;
- duplicate answer text across grouped question rows without preserving section grain;
- treat referrals as complete substantive answers from the referred body;
- treat all XML links as downloadable attachments;
- classify answers as good/bad, evasive/substantive, effective/ineffective, or truthful/untruthful;
- use an LLM merely to split question and answer text when the XML structure already provides that separation.

## Recommended production architecture — not yet approved

A production answer foundation now appears justified, but it should be designed at **answer-section grain**.

### Proposed `written_question_answer_sections`

One row per certified `debate_section_id`.

Candidate fields:

- `debate_section_id`;
- `answer_date`;
- `section_heading`;
- `question_count`;
- `question_ids_json` based on certified/observed relationships;
- `answer_status`;
- `answer_text`;
- `respondent_member_code` where deterministically resolvable;
- `respondent_name`;
- `respondent_role`;
- `grouped_answer`;
- `referred_or_direct_reply`;
- `summary_text_json` or equivalent provenance;
- `embedded_table_count`;
- `source_xml_uri` / `source_xml_url`;
- provenance/version fields.

### Proposed question-to-answer bridge

A separate bridge may be needed at:

`question_id -> debate_section_id`

with evidence/status values so the 7/100 observed question-ID mismatches are not silently certified.

This preserves the distinction between:

- the submitted question entity;
- the shared Written-answer section;
- the published ministerial response.

## Analytical opportunities after extraction

Once the deterministic answer foundation exists, useful research can be done without subjective scoring.

Examples:

- answer length distributions by department/recipient;
- share of answers referred to another body;
- share where no departmental reply was received;
- grouped-answer frequency;
- embedded-table frequency;
- prevalence of explicit figures, dates, named programmes, funding amounts, or future commitments using deterministic/entity extraction methods;
- follow-up chains where a question explicitly references an earlier Parliamentary Question;
- repeated questions over time and whether the official response changes;
- departments/areas where substantive detail is often supplied via external agencies.

Any future semantic categorization should be framed descriptively, not as an answer-quality score.

## Methodological guardrails

1. Written question record != Written-answer section.
2. One answer section may contain multiple question records.
3. One ministerial reply may answer several questions jointly.
4. Preserve the answer at section grain to avoid duplicate counting.
5. A referral is still an official ministerial answer, but may not contain the ultimate substantive agency response.
6. `Reply not received from Department.` is a source status, not a parser failure.
7. The production `debate_section_id` relationship was stronger than direct question-ID-in-XML matching in the sample.
8. Embedded tables should be preserved as structured content where feasible.
9. Generic XML links are not automatically attachments.
10. Do not infer answer quality, effectiveness, evasiveness, or truthfulness from structural features alone.

## Living next-step plan

The source is sufficiently stable to justify a **production design phase**, but not immediate implementation without architecture confirmation.

Next steps:

1. Design the exact `written_question_answer_sections` schema at one-row-per-answer-section grain.
2. Define the question-to-answer bridge evidence rules, including how to handle the 7/100 question-ID mismatches.
3. Run a whole-history lightweight source-reference audit before backfill:
   - unique XML URL count;
   - duplicate/grouped section distribution;
   - missing URLs;
   - HTTP/fetch failure rate on a broader stratified sample;
   - section-ID consistency.
4. Certify respondent extraction from `<speech>` / `TLCPerson` / `TLCRole` references.
5. Define deterministic answer statuses and grouped/referral flags.
6. Design preservation of embedded XML tables before flattening answer text.
7. Run a dedicated attachment/file-link audit; do not infer attachments from generic XML links.
8. Add regression tests around grouped answers, referrals, missing replies, tables, and question-ID mismatches.
9. Only after those rules are approved, implement a candidate-only backfill and validate counts before production promotion.
10. After the deterministic answer corpus exists, separately investigate objective information extraction such as monetary amounts, dates, statistics, organisations, commitments, and references to earlier PQs.

No AI answer classifier is recommended at this stage.
