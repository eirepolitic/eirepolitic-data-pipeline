# Step 17 — Monitoring, Auditability and Operator Queries

Status: **complete**

Research date: 2026-09-04

Scope: define how Eirepolitic should monitor scheduler/runtime health, record publication history, surface failures/uncertain outcomes, notify operators, and support High Director queries about scheduled and published posts.

No CloudWatch alarm, SNS topic, dashboard, Lambda, database, scheduler, or live publication was created.

---

## Short conclusion

Use two distinct observability layers:

```text
Publication ledger
  → authoritative business/audit state
  → what was intended, approved, scheduled, attempted and published

CloudWatch / AWS operational telemetry
  → infrastructure health
  → scheduler/Lambda/DLQ/runtime failures and latency
```

High Director should answer normal operator questions from the **publication ledger**, not by interpreting raw logs.

Recommended initial monitoring stack for direct Meta:

```text
EventBridge Scheduler metrics
Lambda metrics + structured logs
SQS DLQ metrics
custom publication-state metrics
CloudWatch alarms
SNS or AWS User Notifications
publication ledger reconciliation
```

At Eirepolitic's low volume, alerts should be **event-oriented**: one failed or uncertain scheduled publication is important enough to notify an operator.

---

# 1. Source-of-truth separation

## Publication ledger

Answers:

```text
What was approved?
What is scheduled?
Did the publication succeed?
What Instagram Media ID/permalink resulted?
Is the outcome uncertain?
Why did the application classify it as failed?
```

## CloudWatch

Answers:

```text
Did EventBridge attempt the Lambda invocation?
Did the Lambda error or time out?
Was the Lambda throttled?
Did the scheduler exhaust retries?
Did something enter the DLQ?
How long did execution take?
```

These are not interchangeable.

Example:

```text
Lambda timeout
```

does **not** prove:

```text
Instagram publication failed
```

because Step 16 established that Meta may have successfully published before the HTTP response/Lambda failed.

The publication ledger must therefore remain in `publishing_unknown` / reconciliation state until provider status is known.

---

# 2. Existing repository monitoring finding

A repository search found no existing CloudWatch-specific publishing/monitoring implementation.

Therefore the monitoring layer described here would be new infrastructure accompanying the future publisher.

This should remain small and directly tied to publication safety rather than introducing a large observability platform.

---

# 3. EventBridge Scheduler metrics

AWS currently exposes Scheduler metrics in the `AWS/Scheduler` namespace including:

```text
InvocationAttemptCount
TargetErrorCount
TargetErrorThrottledCount
InvocationThrottleCount
InvocationDroppedCount
InvocationsSentToDeadLetterCount
InvocationsFailedToBeSentToDeadLetterCount
```

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/monitoring-cloudwatch.html

### Recommended alarms

For Eirepolitic's low volume:

```text
TargetErrorCount >= 1
InvocationDroppedCount >= 1
InvocationsSentToDeadLetterCount >= 1
InvocationsFailedToBeSentToDeadLetterCount >= 1
```

within an appropriate monitoring window should be considered noteworthy.

Do not use percentage-based alerting initially; a single failed publication invocation may be operationally significant.

---

# 4. Lambda metrics

AWS Lambda exposes metrics including:

```text
Invocations
Errors
Duration
Throttles
DeadLetterErrors
AsyncEventsDropped
```

Source:

- https://docs.aws.amazon.com/lambda/latest/dg/monitoring-metrics-types.html

### Recommended initial alarms

```text
Errors >= 1
Throttles >= 1
AsyncEventsDropped >= 1
```

for the publishing Lambda should trigger investigation.

`Duration` should be monitored for trend/timeout risk rather than alarming on every normal slow Meta response.

A future alarm could fire when duration approaches the configured Lambda timeout consistently.

---

# 5. SQS DLQ monitoring

The Step 9 design uses an SQS standard queue as the EventBridge Scheduler DLQ.

Recommended metrics include:

```text
ApproximateNumberOfMessagesVisible
ApproximateAgeOfOldestMessage
```

If the queue has one visible message, operator attention is required.

A growing `ApproximateAgeOfOldestMessage` means a failed trigger has not been resolved.

Source:

- AWS CloudWatch metrics include standard SQS queue metrics: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/appinsights-metrics-datapoint-requirements.html

### Important meaning

A DLQ message means EventBridge could not deliver/invoke its target under the configured retry policy.

It does **not** by itself establish whether Meta published content.

The operator/recovery worker must load the publication ledger before deciding the next action.

---

# 6. Application-level custom metrics

AWS infrastructure metrics alone are insufficient.

The publication service should emit low-cardinality custom metrics such as:

```text
PublicationScheduled
PublicationExecutionStarted
PublicationPublished
PublicationFailed
PublicationNeedsAttention
PublicationOutcomeUncertain
PublicationAuthBlocked
PublicationReconciled
FirstCommentFailed
```

Useful dimensions should remain low-cardinality, for example:

```text
environment
platform
provider
```

Avoid dimensions such as `publication_id` in CloudWatch metrics because every unique publication ID creates a new metric series and is better handled in logs/ledger queries.

---

# 7. Recommended structured logs

Each publisher log event should contain safe structured fields such as:

```json
{
  "event": "meta_publish_attempt",
  "publication_id": "pub_...",
  "publication_version": 3,
  "attempt_id": "attempt_...",
  "account_ref": "eirepolitic",
  "provider": "meta",
  "operation": "media_publish",
  "state": "reconciling_publish",
  "http_status": 500,
  "provider_error_code": null,
  "duration_ms": 1234
}
```

Never log:

- Meta access token;
- Authorization headers;
- app secrets;
- full presigned URLs;
- raw secret strings;
- unnecessary raw HTTP payloads containing sensitive values.

This follows Step 15's secret-redaction requirements.

---

# 8. Correlation identifiers

Every log and execution record should carry:

```text
publication_id
publication_version
attempt_id
```

where relevant.

Also capture provider IDs once known:

```text
container_id
instagram_media_id
provider_post_id
```

This allows an operator to move from:

```text
High Director query
→ ledger record
→ execution attempt
→ CloudWatch logs
```

without searching by caption text or timestamp guesses.

---

# 9. Log messages are not ledger state

Do not infer final publication state by scanning log text such as:

```text
"publish call completed"
```

The deterministic publisher must explicitly persist state transitions in the publication ledger.

Logs are diagnostic evidence.

The ledger is the application state.

---

# 10. Publication audit trail

The ledger should preserve an append-oriented history of important events:

```text
publication draft created
content package selected
publication version created
human approval recorded
schedule created
schedule changed
schedule cancelled
execution attempt started
provider operation started
provider operation result recorded
outcome became uncertain
reconciliation performed
published result recorded
first comment result recorded
operator recovery action recorded
```

Each event should store:

```text
event type
timestamp UTC
publication_id/version
actor type/reference when human/system action
attempt_id where applicable
safe reason/details
```

Do not overwrite history simply because the current aggregate state changed.

---

# 11. Audit actor types

Useful actor types:

```text
human
high_director
scheduler
publisher
reconciler
operator_recovery
external_provider
```

Example:

```yaml
event: publication_approved
actor:
  type: human
  actor_ref: ...
```

versus:

```yaml
event: execution_started
actor:
  type: publisher
```

This prevents High Director-generated intent from being confused with human approval or provider-observed facts.

---

# 12. High Director query: "What's scheduled this week?"

Query the publication store, not EventBridge directly as the primary source.

Conceptual query:

```text
PublicationSchedule
WHERE state = scheduled
AND scheduled_at_utc in requested interval
ORDER BY scheduled_at_utc
```

Join/load current approved publication summaries.

High Director can return:

```text
Tuesday 19:30 Dublin — Party Speech Breakdown — @eirepolitic
Friday 18:00 Dublin — Member Profile — @eirepolitic
```

A background/explicit reconciliation process should detect if a supposedly scheduled publication has no matching scheduler/provider job.

---

# 13. High Director query: "What went out yesterday?"

Query:

```text
PublishedMedia.published_at
```

within yesterday's `Europe/Dublin` local-day boundaries converted to UTC.

Return data such as:

```text
post name/project
actual published time
account
permalink
provider result
```

Use the associated approved `PublicationRequest` for caption/media details.

---

# 14. High Director query: "Why did tonight's post fail?"

Recommended resolution:

```text
Publication
   ↓
latest ExecutionAttempt
   ↓
classified error + operation state
   ↓
provider status/reconciliation
```

High Director should translate technical details into a concise explanation while preserving certainty.

Examples:

```text
The post did not start because the Meta credential is no longer valid.
```

or:

```text
The publish request timed out, but Meta may have accepted it. The system is reconciling the existing container and will not retry blindly.
```

Do not say:

```text
Instagram failed
```

when the actual state is uncertain.

---

# 15. High Director query: "Is anything stuck?"

Query for operational exception states such as:

```text
publishing_unknown
needs_attention
auth_blocked
scheduled but scheduler_missing
publishing with expired execution lease
published_result_pending_reconciliation
```

These are more useful than asking the user to inspect CloudWatch manually.

---

# 16. Reconciliation monitor

The architecture should run a small periodic reconciliation process for active/exception publications.

Examples:

```text
scheduled publication
→ confirm expected external scheduler job exists

publishing_unknown
→ query existing Meta container/provider state

published_result_pending_reconciliation
→ recover/verify final media metadata

auth health
→ verify account/token health periodically
```

This can be a periodic Lambda/EventBridge rule rather than a continuously running service.

The exact cadence is implementation work.

---

# 17. Scheduled-job drift detection

For direct Meta/EventBridge:

```text
ledger says scheduled
but EventBridge schedule missing/disabled/wrong instant
```

should become:

```text
needs_attention
```

For Buffer hybrid:

```text
ledger approved caption/time
but Buffer job differs
```

should likewise become:

```text
provider_drift / needs_attention
```

Do not silently change Eirepolitic's approved record to match external state.

---

# 18. Publication lateness detection

Monitoring should compare:

```text
scheduled_at_utc
actual_started_at_utc
published_at_utc
```

This allows metrics such as:

```text
schedule_to_start_latency_seconds
schedule_to_publish_latency_seconds
```

A publication that remains `scheduled` beyond its expected trigger window should be flagged even if no AWS alarm fired.

This detects logical/configuration drift that infrastructure metrics may miss.

---

# 19. Alert severity

Recommended simple severity model:

## Critical / immediate operator attention

```text
PublicationOutcomeUncertain
PublicationNeedsAttention near/after schedule
PublicationAuthBlocked with scheduled posts due
DLQ message
InvocationDroppedCount > 0
provider reports published but result reconciliation incomplete for prolonged period
```

## Warning

```text
known token expiry approaching
provider rate limiting
repeated transient retries
schedule/provider drift detected well before due time
Lambda duration approaching timeout
```

## Informational

```text
publication scheduled
publication published successfully
schedule moved/cancelled
```

Do not notify humans for every internal container operation.

---

# 20. Notification channel

AWS CloudWatch alarms can notify through SNS, and AWS User Notifications can deliver through channels including email and supported chat/mobile destinations.

Sources:

- CloudWatch alarm notifications: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Notify_Users_Alarm_Changes.html
- CloudWatch alarms: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Alarms.html

### Recommended initial approach

Keep v1 simple:

```text
CloudWatch alarm / publication exception
        ↓
SNS or AWS User Notifications
        ↓
operator email
```

The final notification destination should be chosen during implementation based on what the user actually wants to monitor.

Power Automate could later consume an email/webhook/event if a Microsoft-facing workflow is useful, but it is not required for the publication core.

---

# 21. Avoid alert storms

CloudWatch supports composite alarms and normal threshold evaluation, which can reduce noise.

Source:

- https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Alarms.html

At Eirepolitic scale, start with a small number of precise alarms rather than dozens of low-value thresholds.

For example, one `publication-needs-attention` application alarm/event can be more useful than separate notifications for every retry attempt.

---

# 22. Recommended v1 alarms

Minimal direct-Meta set:

```text
1. SchedulerTargetError
   TargetErrorCount >= 1

2. SchedulerDroppedInvocation
   InvocationDroppedCount >= 1

3. SchedulerDLQMessage
   InvocationsSentToDeadLetterCount >= 1
   OR SQS messages visible >= 1

4. PublisherLambdaError
   Lambda Errors >= 1

5. PublisherLambdaThrottle
   Lambda Throttles >= 1

6. PublicationNeedsAttention
   custom application event/metric >= 1

7. PublicationOutcomeUncertain
   custom application event/metric >= 1

8. InstagramAuthBlocked
   custom application event/metric >= 1
```

This is enough to catch the important failure classes without building a full observability platform.

---

# 23. Dashboard

A CloudWatch dashboard is optional but useful once production exists.

Recommended tiles:

```text
Scheduled publications next 7 days   # ideally sourced from application/ledger view, not pure CloudWatch
Publisher Lambda errors
Publisher Lambda duration
Scheduler target errors
DLQ message count
PublicationPublished count
PublicationFailed count
PublicationOutcomeUncertain count
AuthBlocked count
```

A custom application/operator page may eventually be more useful than a CloudWatch dashboard for editorial status.

High Director itself should remain the primary conversational status interface.

---

# 24. Audit retention

Publication approval and final publication history should be retained long-term enough to answer historical editorial/audit questions.

Examples:

```text
Which exact caption was approved?
Who approved it?
Which asset hashes were published?
When was it scheduled?
What provider ID/permalink resulted?
Were there retries or uncertain outcomes?
```

CloudWatch logs do not need to be retained forever merely because the publication ledger is permanent.

Use a deliberate log-retention policy to control cost and privacy, while keeping durable business records in the ledger.

The exact retention period can be set during implementation.

---

# 25. CloudTrail role

AWS control-plane actions such as secret access/management and infrastructure changes can be audited through AWS-native services such as CloudTrail, as noted in Step 15.

CloudTrail is useful for questions like:

```text
Which AWS principal accessed or changed a production secret/resource?
```

It should not replace the application publication audit trail.

The publication ledger records business actions; CloudTrail records AWS API/control-plane actions.

---

# 26. No secret leakage in monitoring

Monitoring/alert payloads must follow Step 15's rules.

Never include:

```text
Meta token
Buffer API key
Authorization header
full presigned URL
secret values
```

in:

- CloudWatch logs;
- custom metrics;
- alarm descriptions;
- SNS messages;
- High Director summaries.

Provider error messages should be sanitized before notification.

---

# 27. Success verification

A successful Lambda invocation is not sufficient success evidence.

A publication reaches `published` only after provider success is durably recorded/reconciled.

For direct Meta, examples include:

```text
/media_publish returned Instagram Media ID
OR
container reports PUBLISHED and result is reconciled/guarded
```

The ledger should also record the resulting permalink when retrievable.

---

# 28. Failure versus uncertainty

Monitoring must preserve the distinction:

```text
failed
```

means a known terminal/blocked failure.

```text
publishing_unknown
```

means the side effect may have happened and reconciliation is required.

The latter should often be treated as higher urgency because the system must avoid both duplicate publication and missed visibility.

---

# 29. Daily/weekly operational summary

High Director should be able to generate summaries directly from ledger queries such as:

```text
scheduled next 7 days
published last 7 days
failed last 7 days
needs_attention now
authentication status
```

This supports an operator workflow without requiring a dedicated social publishing dashboard at first.

Do not create recurring automated reports unless explicitly requested later.

---

# 30. Buffer hybrid monitoring differences

If Buffer is selected:

```text
Eirepolitic ledger
      ↓
Buffer scheduled post state
      ↓
Instagram
```

Monitoring must include provider reconciliation for Buffer states such as:

```text
scheduled
sending
sent
error
```

Because Step 7 did not identify a reliable public Buffer post-status webhook, initial design should assume polling/reconciliation.

CloudWatch then monitors our Buffer adapter/reconciler, while Buffer's state remains an external data source.

Eirepolitic's ledger remains authoritative for approved intent and normalized status.

---

# 31. Direct Meta monitoring advantage

Direct Meta removes one external scheduler state layer.

The state chain is:

```text
Eirepolitic ledger
→ EventBridge/Lambda
→ Meta
```

instead of:

```text
Eirepolitic ledger
→ Buffer
→ Meta
```

This makes drift/reconciliation slightly simpler and is another modest advantage of direct Meta.

---

# 32. Step 17 verdict

Recommended monitoring architecture:

```text
Publication ledger
        ↓
High Director operator queries

EventBridge Scheduler ─┐
Lambda metrics/logs    ├→ CloudWatch alarms → SNS/User Notifications
SQS DLQ                │
Custom app metrics     ┘

Periodic reconciler
        ↓
provider/scheduler state
        ↓
publication ledger
```

Key rules:

1. The publication ledger is the business source of truth; CloudWatch is operational telemetry.
2. High Director should query structured publication records, not parse raw logs.
3. Correlate everything using `publication_id`, version and `attempt_id`.
4. At low volume, one failed/uncertain publication is significant enough to alert.
5. Monitor Scheduler errors/drops/DLQ, Lambda errors/throttles and custom application states.
6. Treat `publishing_unknown` distinctly from known failure.
7. Reconcile scheduled/provider state periodically to detect drift and stuck jobs.
8. Keep notification channels simple initially—SNS/User Notifications/email is sufficient.
9. Retain permanent publication history separately from shorter-lived technical logs.
10. Never leak credentials or full presigned URLs through monitoring/alerts.

---

## Sources

### AWS EventBridge Scheduler

- CloudWatch monitoring/metrics: https://docs.aws.amazon.com/scheduler/latest/UserGuide/monitoring-cloudwatch.html
- Scheduler troubleshooting: https://docs.aws.amazon.com/scheduler/latest/UserGuide/troubleshooting.html

### AWS Lambda

- Lambda metric definitions: https://docs.aws.amazon.com/lambda/latest/dg/monitoring-metrics-types.html

### CloudWatch / notifications

- CloudWatch alarms: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Alarms.html
- CloudWatch alarm notifications: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Notify_Users_Alarm_Changes.html

### SNS

- SNS monitoring/CloudWatch alarm examples: https://docs.aws.amazon.com/sns/latest/dg/sns-monitoring-using-cloudwatch.html

### Repository

- Targeted repo search found no current CloudWatch publishing-monitoring implementation.

---

## Confidence / unresolved items

**High confidence:**

- EventBridge Scheduler and Lambda expose the required operational metrics;
- the publication ledger must remain separate from raw infrastructure telemetry;
- High Director can answer operator status questions cleanly from the logical records designed in Steps 10–11;
- uncertain publish outcomes require explicit monitoring/reconciliation rather than being flattened into generic failure;
- simple CloudWatch + SNS/User Notifications is sufficient for initial scale.

**Still to determine during implementation:**

- exact CloudWatch alarm periods/threshold windows;
- exact log-retention duration;
- final notification channel/email recipient;
- reconciler cadence;
- physical ledger/query/index design;
- whether a CloudWatch dashboard is worth creating in v1.

**Next research step:**

Step 18 will define the multi-platform extension approach so the publication control layer can later support other networks without making Instagram/Meta concepts the core data model.
