# Parliamentary question-taking certification

Status: **Current follow-up investigation**  
Date: **2 September 2026**  
Parent research record: [Parliamentary questions investigation](parliamentary_questions_investigation.md)

This note records the follow-up investigation into whether the TD who actually takes an Oral Parliamentary Question in the chamber can be derived deterministically from transcript order, explicit substitution/proxy language, and question-to-speech text matching.

## Confirmed findings

The production snapshot contains:

- **2,127 oral-question exchanges**;
- **3,660 Oral question records**.

### Submitter normally appears in the exchange

In **1,906 exchanges (89.6%)**, at least one recorded submitter appears under their own member code in the related transcript.

At question-record level, **3,410 of 3,660 Oral questions (93.2%)** belong to exchanges where a submitter is present in the transcript.

### Transcript ordering is highly reliable, but not perfect

After excluding ministers and chair/procedural speakers using both dated office history and transcript role labels, the **first non-officeholder member speaker** was one of the recorded question submitters in:

- **1,890 of 1,901 eligible submitter-present exchanges**;
- **99.42%**.

This is strong evidence that transcript order is useful supporting evidence for identifying the person taking a question.

However, the remaining **11 exceptions** show that ordering alone is not safe enough to materialize `taken_by_member_code` universally.

### Explicit substitution/proxy-taking is demonstrable

The deterministic rules now recognize both English and Irish formulations, plus explicit chair/procedural assignment language, including examples equivalent to:

- "I am taking this question on behalf of Deputy ...";
- "I am covering for Deputy ...";
- "I ask this question on behalf of ...";
- "Question No. 4 is being taken by Deputy Neville";
- "Question No. 75 is in the name of Deputy O'Connor and is being taken by Deputy McCormack";
- "Deputy Lawlor on Question No. 103" when the official question record names a different submitter;
- "He is subbing for the Deputy";
- "I am covering the questions for Deputy ...";
- Irish chair language such as a TD `ag glacadh Ceist ... faoi choinne an Teachta ...`;
- Irish wording where a question is `á tógáil ag an Teachta ...`.

Because grouped exchanges can contain several submitted questions, an exchange-level substitute candidate must not automatically be assigned to every question in that section.

### Strictly certified single-question substitute relationships

The current strict rule is:

> single-question exchange + different first ordinary-member speaker + explicit substitution/proxy/procedural assignment evidence tied to that question.

Under the expanded evidence rules, **65 single-question substitute relationships** are now strictly certifiable.

This supersedes the earlier figure of 57, which reflected the narrower initial phrase set.

The latest pass added **7 relationships** that had previously remained unresolved:

- Ruairí Ó Murchú taking Mark Ward's Question No. 7 after the chair explicitly stated that he was substituting for Deputy Ward;
- George Lawlor taking Robert O'Donoghue's Question No. 103 after the chair called "Deputy George Lawlor on Question No. 103";
- George Lawlor taking Eoghan Kenny's Question No. 2 after the Minister asked if he was subbing and the chair confirmed that he was;
- Joe Neville taking Barry Ward's Question No. 4 after the Leas-Cheann Comhairle explicitly said it was being taken by Deputy Neville;
- Tony McCormack taking James O'Connor's Question No. 75 after the chair explicitly stated that the question was in Deputy O'Connor's name and being taken by Deputy McCormack;
- Pa Daly taking Rose Conway-Walsh's Question No. 137 after stating that he was covering the questions for Deputy Conway-Walsh;
- Malcolm Byrne taking Naoise Ó Cearúil's Question No. 7 after the chair explicitly stated that it was being taken by Deputy Malcolm Byrne.

These are examples of a real parliamentary mechanism rather than anomalous data.

## Remaining unresolved structure

The single-question text-matching audit identified **207** cases where the first ordinary-member speaker was not the official submitter.

With the expanded strict evidence rules:

- **65** are now explicitly certified substitute-taken questions;
- **142** remain unresolved and should stay `unknown` unless stronger evidence is found.

The unresolved cases should **not** be treated as erroneous. Many may be genuine substitutions, but the current evidence is insufficient to publish that attribution as fact.

At the broader exchange level, grouped oral-question exchanges remain structurally different and often do not support question-level taker attribution at all.

## Role-detection correction made during this investigation

An earlier pass incorrectly treated some acting chairs as ordinary members because dated office history did not always identify the temporary chair role.

The refined method also checks transcript speaker labels for:

- Ceann Comhairle;
- Leas-Cheann Comhairle;
- Cathaoirleach;
- Cathaoirleach Gníomhach;
- Acting Chairman / Acting Chairperson.

This improved the validation of the transcript-order rule from approximately **98.3% to 99.4%**.

This is another reason source transcript role labels should supplement office history when classifying procedural speakers.

## Investigation of the 11 transcript-order exceptions

All 11 exceptions were inspected individually, including a raw-XML check of the first speech in each affected debate section.

### Raw XML result

The parsed first-speaker attribution matched the Oireachtas XML in all 11 cases examined.

This rules out the suspected general speaker/text alignment bug in `silver_speeches` for these cases.

Two apparently suspicious grouped-answer introductions were correctly attributed in the raw XML to the Taoiseach. The earlier compact exception digest had shown the **first ordinary-member candidate** separately from the **first speech text**, which made those rows look mismatched. The underlying production speech records were correct.

### What the 11 exceptions actually represent

The exceptions break down structurally as:

- **8 grouped-question exchanges** where an ordinary TD who was not one of the recorded submitters spoke before the first recorded submitter;
- **2 single-question exchanges with explicit substitute-taking**, where the substitute spoke first even though the original submitter later appeared in the same exchange;
- **1 single-question exchange with a likely substitute/alternate taker but no currently certified explicit proxy phrase**.

### Consequence for the ordering rule

The 99.42% figure remains useful as a **validation statistic**, but the first ordinary-member speaker cannot be treated as the question taker by default.

There are two reasons:

1. grouped oral-question exchanges can contain other TD participants before any recorded submitter speaks;
2. an explicit substitute can take the question first even when the original submitter later participates.

Therefore:

> `submitter_present` is not equivalent to `submitter_took_question`, and `first ordinary speaker` is not equivalent to `taken_by_member_code`.

Transcript order should remain supporting evidence only unless combined with stronger question-level evidence.

## Question-text to opening-speech matching

A second deterministic signal was tested on **single-question oral exchanges only**.

The audit found **1,856 single-question exchanges** with an identifiable ordinary-member opening speaker:

- **1,649** where the opening ordinary-member speaker was the official submitter;
- **207** where the opening ordinary-member speaker was not the official submitter.

### Main result: text similarity does not distinguish self from substitute

The official question wording is often repeated or closely paraphrased by whoever takes the question in the chamber.

Median normalized question-token coverage in the benchmark was approximately:

- self-first cases: **35.0%**;
- explicit substitutes: **52.6%**;
- unresolved non-submitter-first cases: **35.3%**.

Median token-set similarity was approximately:

- self-first cases: **63.1**;
- explicit substitutes: **74.9**;
- unresolved non-submitter-first cases: **63.2**.

A substitute can therefore match the official wording more closely than the original submitter does in many self-taken cases.

### Decision on text matching

Question-to-opening-speech similarity should be treated as:

- **review prioritization evidence**;
- a potential feature in a deterministic evidence bundle;
- a way to identify likely cases where a non-submitting TD voiced the official question.

It should **not** be treated as:

- proof that the speaker was the official submitter;
- proof that the speaker was an authorized substitute;
- a standalone production `taken_by_member_code` rule.

The proposed `substitute_text_match` status is therefore **not certified**.

## High-match unresolved-case pattern discovery

Text similarity was then used only to prioritize unresolved cases for manual/deterministic pattern discovery.

This produced **53 high-match unresolved single-question cases** under the review threshold used for the exploratory pass.

The review uncovered several explicit procedural patterns that the earlier proxy rules had missed.

### Most useful new source signal: chair assignment by question number

The strongest newly identified pattern is the chair explicitly assigning a numbered question to a TD who is different from the official submitter.

Examples include:

- `Question No. 4 is being taken by Deputy Neville.`
- `Question No. 75 is in the name of Deputy James O'Connor and is being taken by Deputy McCormack.`
- `Question No. 7 in the name of Deputy Ó Cearúil is being taken by Deputy Malcolm Byrne.`
- `Deputy George Lawlor on Question No. 103` where Question No. 103 belongs to Robert O'Donoghue in the official question record.

For a **single-question exchange**, when the chair names a different TD against the exact official question number, this is strong deterministic evidence that the named TD is taking that question.

### Other new explicit formulations

Additional certifiable wording included:

- `Deputy Ó Murchú is substituting for Deputy Ward on Question No. 7.`
- a Minister asking whether a TD is `subbing for` the official submitter, followed by the chair explicitly confirming that they are;
- `I am covering the questions for Deputy Conway-Walsh.`

### Pattern-audit result

The strict follow-up scan found:

- **58** single-question relationships already covered by the expanded existing explicit rules used in that scan;
- **7 additional relationships** caught only by the newly identified chair/subbing/covering rules;
- **65 total strict single-question substitute relationships** after combining the evidence classes.

The 7 newly recovered relationships were supported by:

- 3 `question ... is being taken by ...` chair assignments;
- 2 chair/member-on-question-number assignments;
- 1 explicit `subbing for` confirmation;
- 1 explicit `covering the questions for` statement.

### Important guardrail

The phrase discovery was driven by high-match cases, but **text similarity did not certify any relationship**.

Certification came only from explicit procedural/substitution evidence.

## Current methodological decision

Question-taking attribution is now **deliberately bounded** rather than pursued toward full coverage.

The certified rules are:

- Do not infer `taken_by_member_code` from submitter presence alone.
- Do not infer it from first ordinary-member transcript position alone.
- Do not infer it from question-text similarity alone.
- For a **single-question exchange**, an explicit substitution/proxy statement or explicit chair assignment tied to that numbered question may certify a substitute taker.
- For grouped exchanges, do not assign one exchange-level participant to every grouped question.
- For unresolved cases, retain `unknown`.

The current certified subset is **65 single-question substitute relationships**.

The remaining **142 single-question non-submitter-first cases** should remain unresolved unless new explicit evidence emerges naturally from future work.

### Production-foundation decision

A dedicated `question_taking_relationships` production foundation is **not yet necessary** solely for these 65 records.

The evidence model is now clear enough to implement later if a downstream consumer needs it, but continuing to optimize regex coverage has diminishing analytical value.

The useful structural lesson has been established:

> official submitter, in-chamber question taker, and exchange participant are distinct concepts.

The research effort should now move back to exchange-level participant metrics, where the coverage and analytical value are much larger.

## Evidence

Important runs for this follow-up:

- initial deterministic question-taking audit: **33578667504**;
- compact initial audit digest: **33578749629**;
- refined chair-role + Irish-language substitution audit: **33578804943**;
- refined compact digest: **33578865429**;
- 11-case transcript-order exception extraction: **33579196132**;
- compact exception digest: **33579257409**;
- raw XML speaker-attribution check: **33579429125**;
- single-question text-match audit: **33579960048**;
- explicit-substitute/unresolved text-match benchmark: **33580045161**;
- high-match unresolved substitution-pattern review: **33580298794**;
- compact unresolved-pattern digest: **33580355636**;
- strict chair/procedural assignment audit: **33580414547**.

No production pipeline or production data was changed during these investigations.

## Revised next-steps plan

### 1. Certify reusable oral-question exchange participant metrics

This is now the immediate next task.

The taker-attribution problem is sufficiently bounded for current purposes. The next higher-value work is to convert the already observed exchange structure into reusable deterministic measures.

Investigate and certify:

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

Before materialization:

- validate event-date minister/chair role attribution;
- validate transcript word-count completeness;
- define ordinary non-submitting TD precisely;
- establish additive/non-additive semantics;
- decide which measures belong directly in `oral_question_sections` and which require a participant-level foundation.

### 2. Design exchange-participant structure if needed

If exchange-level aggregates are insufficient for downstream analysis, consider a separate participant foundation keyed by:

- `debate_section_id`;
- `member_code` / participant identity;
- participant role category;
- submitted-question relationship;
- intervention count;
- word count;
- provenance.

This would support richer analysis without forcing question-level taker attribution.

### 3. Preserve question-taker evidence rules for future use

Do not discard the certified evidence model.

If a future consumer needs `taken_by_member_code`, implement only the explicit evidence rules documented here and leave all other cases unknown.

Do not resume broad regex expansion unless there is a concrete downstream requirement.

### 4. Then investigate section-heading normalization

After exchange participant metrics are settled, return to the parent plan and determine whether Oireachtas section headings can provide a useful no-AI topic hierarchy for Oral questions.

### 5. Question issue classification remains deferred

Nothing in this follow-up changes the earlier decision: do not classify the roughly 121k question history until deterministic structure has been exhausted and a concrete use case justifies the cost.
