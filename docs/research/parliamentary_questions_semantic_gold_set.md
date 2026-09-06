# Written Parliamentary Question semantic gold-review set

## Status

A blind human-review set was generated successfully on 2026-09-05/06 for the next phase of Written PQ semantic-classification research.

This is research-only. No semantic labels have been published to production and the production pointer was not changed.

## Purpose

The earlier semantic-classification research established that model-vs-model agreement is not enough to choose between the richer V2 classifier and the cheaper routing-only variant.

The next valid comparison requires labels created independently of the models being evaluated.

The gold-review set therefore contains source question and answer text plus blank human-label fields. It is deliberately blind to model outputs so reviewers are not anchored by either classifier.

## Generation evidence

Workflow run: `34008131593`

Deterministic seed: `20260905`

Generated set:

- answer sections: **100**;
- linked questions: **139**;
- unique recipients: **20**;
- 2025 sections: **62**;
- 2026 sections: **38**;
- grouped-answer sections: **27**;
- referral/direct-reply sections: **27**;
- non-`ministerial_reply_present` sections: **10**;
- sections containing embedded tables: **20**;
- minimum answer length: **0 characters**;
- median answer length: **1,035.5 characters**;
- maximum answer length: **9,794 characters**.

The set was intentionally sampled to include structural and semantic edge cases rather than act as a prevalence sample of the whole corpus.

## Review files

The research branch `research/written-pq-semantic-gold-set-20260905` contains:

- `analysis/written_pq_semantic_gold_set/gold_set_manifest.csv`
- `analysis/written_pq_semantic_gold_set/question_review.csv`
- `analysis/written_pq_semantic_gold_set/answer_review.csv`
- `analysis/written_pq_semantic_gold_set/summary.json`

### Question review fields

Each linked question has source identifiers/text plus blank reviewer fields for:

- `human_topic_tags_json`;
- `human_question_intents_json`;
- `human_missing_topic_notes`;
- `reviewer_notes`.

### Answer review fields

Each answer section has source text/metadata plus blank reviewer fields for:

- `human_topic_tags_json`;
- `human_answer_characteristics_json`;
- `human_missing_topic_notes`;
- `reviewer_notes`.

## Why the gold labels must be human-reviewed

The purpose of the gold set is to evaluate model accuracy independently.

Using the same LLM, or another unreviewed LLM, to fill the gold labels would turn the exercise back into model-vs-model agreement and would not establish which output is correct.

LLMs may later help with annotation tooling or disagreement summaries, but the reference labels used to decide production accuracy thresholds should be human-reviewed.

## Recommended annotation procedure

1. Review question text independently from answer text.
2. Apply only controlled taxonomy labels that are substantively present in that scope.
3. Record all applicable question intents or answer characteristics; do not force a single label.
4. Use `human_missing_topic_notes` when the controlled taxonomy lacks a recurring concept instead of selecting a misleading nearby tag.
5. Do not infer political support/opposition, truthfulness, effectiveness, evasiveness or quality.
6. For referral-only answers, classify only what the answer itself contains rather than inheriting the question topic.
7. Where a label is genuinely ambiguous, record the ambiguity in `reviewer_notes` rather than manufacturing certainty.

## Evaluation to perform after annotation

Compare both semantic variants against the human labels separately for:

- question topic tags;
- question intents;
- answer topic tags;
- answer characteristics;
- missing/new taxonomy concepts.

Do not collapse these into one headline accuracy score. A cheaper model path may be acceptable for one output family and inadequate for another.

Important comparison questions include:

- Does the cheaper router retain acceptable question-intent accuracy?
- Does it retain acceptable answer-characteristic accuracy?
- Are rich V2 topic tags materially more accurate than cheap topic tags, or merely different?
- Which topic labels produce systematic ambiguity?
- Which taxonomy gaps recur across reviewers?
- Are referral/no-answer/grouped-answer cases handled consistently?

## Living next-step plan

1. Complete human annotation of the 100-section / 139-question blind review set.
2. If possible, double-review a smaller subset so reviewer disagreement can be distinguished from model error.
3. Produce per-output-family evaluation for rich V2 and the cheaper router against the human labels.
4. Inspect errors by tag and by structural edge case rather than relying only on average agreement.
5. Refine taxonomy definitions only where human review shows systematic ambiguity or missing concepts.
6. Define explicit production acceptance/quarantine thresholds.
7. Decide which semantic fields belong in the cheapest first pass and which should be deferred to specialised later calls.
8. Only after that decision, design production semantic tables and the initial/incremental model-call workflow.
9. Then begin a separate bounded claim-extraction pilot for commitments, statistics, funding and legislation using the validated router to select source text.

No production semantic-classification dataset is approved yet.
