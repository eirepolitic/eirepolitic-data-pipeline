# Research documentation

This directory is the durable research record for investigations into EirePolitic/Oireachtas data and derived political metrics.

The purpose is to preserve not only final metrics, but also what was investigated, what the source data actually means, methodological caveats, data-quality discoveries, decisions taken, and promising future work.

## Working process

For each research topic:

1. Investigate the source and derived data.
2. Record confirmed findings and corrections in the topic document.
3. Update the living next-steps plan in that document.
4. Only then summarize the result externally.

Research documents should distinguish clearly between:

- confirmed source/data structure;
- derived descriptive findings;
- methodological cautions;
- pipeline changes already implemented;
- open questions;
- proposed enrichments or classifications;
- ideas deliberately deferred.

Do not promote exploratory conclusions into production metrics until their denominator, attribution rules, reliability and public interpretation have been reviewed.

## Topics

- [Parliamentary questions](parliamentary_questions_investigation.md) — main durable record covering submitted question records, written vs oral questions, oral-question sections/exchanges, transcript interventions, grouped questions, participant roles, substitution/proxy-taking, batching, recipient patterns, and future metrics/enrichments.
- [Written Parliamentary Question answers](parliamentary_questions_written_answers.md) — read-only investigation of the official section-level Written-answer XML, including 100/100 fetch/parse success, deterministic question/answer structure, grouped answers, referrals, missing replies, embedded tables, question-ID linkage anomalies, and the proposed answer-section production design.
- [Written PQ semantic classification](parliamentary_questions_semantic_classification.md) — research-only design and benchmarking of LLM-based semantic routing over question text and official answer text, with controlled hierarchical tags, separate question/answer views, deterministic combined topics, evidence validation, repair rules, taxonomy-gap handling and cost/scale analysis.
- [Parliamentary question-taking certification](parliamentary_questions_question_taking.md) — submitted-by vs actually-taken-by attribution, transcript ordering, explicit substitution evidence, bounded unresolved cases, and the conservative evidence model.
- [Oral-question exchange participant metrics](parliamentary_questions_exchange_metrics.md) — certified exchange word/intervention components, submitter participation, minister/chair/ordinary-member roles, mixed-role and anonymous-speaker edge cases, recommended participant foundation, and production implementation plan.
- [Oral-question exchange participant implementation](parliamentary_questions_exchange_metrics_implementation.md) — implemented schema, regression coverage, production deployment, live counts, permanent audits, and deployment evidence.
- [Oral-question section headings](parliamentary_questions_section_headings.md) — heading reuse, stability, normalization tests, recipient drift, public-filter interpretation, and the decision not to add a normalization dataset.
- [Oral versus Written parliamentary questions](parliamentary_questions_oral_vs_written.md) — channel volumes, recipient concentration, Taoiseach/department differences, member/party/constituency channel profiles, safe reusable measures, and the decision that existing foundations are sufficient.
- [Broader deterministic speech context](broader_speech_context.md) — investigation and certification record for deterministic top-level speech contexts, exact source rules, precedence, rejected approaches and coverage evidence.
- [Broader speech context implementation](speech_context_implementation.md) — deployed one-row-per-speech `speech_context` foundation, live production batch, context counts, validation/deployment runs, permanent audits, downstream guardrails and the current living next-step plan into contextual voting analysis.
- [Contextual voting analysis](contextual_voting.md) — read-only investigation linking all current divisions to certified Bill, motion, procedural-business or `other` section context, preserving existing voting denominators and rejecting unsafe stage/text inference.
- [Division context implementation](division_context_implementation.md) — deployed one-row-per-division `division_context` foundation, live production batch, exact context counts, validation/deployment runs, member-vote no-multiplication audits, denominator guardrails and the living next-step plan for contextual versions of existing voting metrics.
- [Context-filtered voting metrics](contextual_voting_metrics.md) — read-only validation of member participation and party recorded-vote agreement inside each division context, including denominator sizes, existing reliability thresholds, Independent-group wording, publishability rules and the recommended context-aware additive foundation design.
- [Context-aware voting foundations implementation](contextual_vote_foundations_implementation.md) — deployed additive daily contextual vote numerators/denominators and context-aware party division-vote components, exact reconciliation to existing unfiltered foundations, live production counts and the handoff to the deployed monthly contextual result layer.
- [Contextual monthly voting results implementation](contextual_monthly_voting_implementation.md) — deployed `division_context` rows for monthly member recorded-vote participation and party recorded-vote agreement, formal member small-sample rules, live reliability/suppression counts, safe deployment-failure fixes and the handoff to consumer-readiness auditing.
- [Contextual voting consumer readiness](contextual_voting_consumer_readiness.md) — read-only audit of live CSV/Parquet parity, metadata completeness, ranking/suppression behavior, Independent wording and human-label readiness, including the finding that naïve top-10 sorting would surface 703 `not_certified` rows unless consumers filter first.
- [Legislation relationships](legislation_investigation.md) — Bill identifiers, stages, sponsors, debate-section relationships, speech and division linkage, conservative certification rules, source anomalies, coverage limits, production implications, and the living implementation plan.
- [Certified Bill-section bridge implementation plan](legislation_bridge_implementation_plan.md) — approved design for the additive production foundation, exact certified grain, duplicate-source handling, audits, downstream compatibility and deployment sequence.
- [Certified Bill-section bridge implementation](legislation_bridge_implementation.md) — deployed production foundation, live batch and counts, validation/deployment run IDs, permanent audits, production semantics, guardrails and the current living next-step plan.
- [Instagram post candidates](instagram_post_candidates.md) — researched and ranked cross-pipeline shortlist of ten defensible Instagram post ideas spanning issue-labelled speeches, speech structure, divisions, voting, legislation, Leaders' Questions and Parliamentary Questions, with exact evidence, denominators, rejected ideas, unsafe-but-promising areas, and the living editorial next-step plan.
- [Irish political data sources](irish_political_data_sources.md) — completed research-only investigation of Irish polling and other political datasets, including access, licensing, pricing, methodology, automation feasibility, rejected sources, rankings, and top future ingestion candidates.
- [Irish political data ingestion feasibility](irish_political_data_ingestion_feasibility.md) — read-only follow-up on the top five candidates, covering concrete access patterns, schema/versioning risks, future ingestion modes and implementation-readiness order.
- [Irish Polling Indicator read-only proof](ipi_readonly_proof.md) — bounded source-level validation of the highest-ranked polling candidate, including current files, schemas, coverage, versioning and future validation rules.
- [Irish Polling Indicator bounded validation](ipi_bounded_validation.md) — exact pinned-source diagnostics for duplicates, fieldwork-date anomalies, modeled interval validity, election-cycle date behavior and future source-quality rules.
- [Irish Polling Indicator ingestion implementation](ipi_ingestion_implementation.md) — production-quality repository integration, S3 contract, validation/publication rules, CI coverage, scheduled workflow and explicit source-rights publication gate.
- [Irish demographic polling dataset validation](irish_demographic_polling_validation.md) — bounded validation of the second-ranked polling source, including file structure, crosstab reliability, exact diagnostics and licensing/republication clarity.
- [Irish general-election count and transfer data validation](election_count_data_validation.md) — bounded validation of official 2016/2020 count-detail files and the 2024 results/transfer workbook, including schema differences, source quality and safe normalization rules.
- [CSO PxStat validation](cso_pxstat_validation.md) — bounded validation of the CSO API using a Census 2022 population table, including JSON-stat structure, geography/version safeguards, source metadata and future ingestion rules.
- [Irish referendum result data validation](referendum_data_validation.md) — bounded cross-year validation of official referendum CSVs, including schema drift, arithmetic checks, constituency-boundary cautions and safe normalization rules.
- [Official recurring political-adjacent data validation](official_recurring_data_validation.md) — read-only validation of eTenders procurement data and Department of Finance monthly Exchequer/tax-receipt data, including access, schema, update cadence, data-quality caveats and recurring-graphics suitability.

Future investigations should get their own topic documents here, for example voting behaviour, speech structure, issue analysis, constituency representation and cross-metric profiles.
