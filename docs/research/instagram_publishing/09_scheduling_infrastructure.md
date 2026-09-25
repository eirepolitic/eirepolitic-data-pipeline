# Step 9 — Scheduling and Runtime Infrastructure

Status: **complete**

Research date: 2026-09-03

Scope: compare appropriate scheduling/runtime options for the direct-Meta architecture, with emphasis on reliability, timing accuracy, retries, idempotency, cost, monitoring, secrets, cancellation/rescheduling and future multi-platform use.

Options reviewed:

- AWS EventBridge Scheduler;
- AWS Lambda;
- AWS Step Functions;
- Amazon SQS;
- GitHub Actions scheduled workflows;
- Power Automate scheduled cloud flows.

No infrastructure was provisioned and no production scheduler was created.

---

## Short conclusion

For direct Meta publishing, the best current fit is:

```text
EventBridge Scheduler
       ↓
small deterministic Lambda publisher
       ↓
Meta Instagram API
```

with:

```text
SQS standard queue
```

used initially as the EventBridge Scheduler dead-letter queue (DLQ), not as the publication clock.

Do **not** use GitHub Actions cron as the authoritative social-publication scheduler. GitHub explicitly warns that scheduled workflows can be delayed during high load and that queued jobs can be dropped if load is high enough.

Do **not** introduce Step Functions in the first static-image/carousel implementation unless the eventual Meta worker proves to need a genuinely multi-stage/long-running orchestration flow. Step Functions remains a good later option for video/Reel processing or complex recovery workflows.

Power Automate is suitable for optional notifications/operator workflows, but is a weaker choice for the authoritative publication clock because it adds another runtime/licensing layer and Microsoft documents trigger delays/plan-dependent polling behaviour.

At Eirepolitic's expected volume, EventBridge Scheduler + Lambda + an SQS DLQ should cost effectively nothing or only pennies under normal AWS free-tier/low-volume usage.

---

# 1. Scheduling requirements

The publication scheduler must support:

- one-time future jobs;
- exact approved date/time;
- explicit timezone handling;
- safe daylight-saving behaviour;
- cancellation;
- rescheduling;
- retry on trigger-delivery failure;
- dead-letter handling;
- deterministic target payload;
- low operational overhead;
- auditability;
- multi-platform reuse later.

It does **not** need millisecond or second-level precision.

For social publishing, a target such as 19:30 means publication should begin within the scheduled minute, not necessarily exactly at `19:30:00.000`.

---

# 2. AWS EventBridge Scheduler

## Fit: **excellent — recommended publication clock**

EventBridge Scheduler is purpose-built for scheduled tasks and supports:

- one-time schedules;
- recurring cron/rate schedules;
- timezone selection;
- daylight-saving-aware scheduling;
- schedule enable/disable;
- update/delete operations;
- configurable retry policy;
- SQS dead-letter queues;
- automatic deletion after schedule completion;
- direct Lambda/Step Functions/SQS/AWS API targets.

Sources:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule.html
- https://aws.amazon.com/eventbridge/scheduler/

---

## One schedule per publication

The simplest model is:

```text
publication_id = pub_123
scheduled_at = approved instant
       ↓
EventBridge one-time schedule
       ↓
Lambda payload:
{"publication_id":"pub_123"}
```

The scheduler should not contain the full approved caption/media configuration.

At execution, Lambda loads the authoritative approved publication version from the publication store.

This keeps schedule management small and prevents stale copies of captions/tags existing in several systems.

---

## Timing precision

AWS currently documents **60-second precision** for EventBridge Scheduler targets.

If a schedule is configured for `19:30`, the target invocation occurs between:

```text
19:30:00
and
19:30:59
```

when no flexible delivery window is configured.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html

### Assessment

This is sufficient for Eirepolitic social publishing.

Do not add more complex infrastructure merely to try to publish at the exact zero-th second of a minute.

---

## Timezones and DST

EventBridge Scheduler supports explicit timezones and daylight-saving-aware schedules.

This maps well to Eirepolitic's `Europe/Dublin` requirement.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html

The final timezone model is covered in Step 14; Step 9 only establishes that EventBridge has the required scheduling capability.

---

## Cancellation

EventBridge Scheduler exposes a direct `DeleteSchedule` API operation.

AWS also documents client tokens for idempotent schedule-management API calls.

Source:

- https://docs.aws.amazon.com/scheduler/latest/APIReference/API_DeleteSchedule.html

Therefore:

```text
"Cancel Friday's post"
       ↓
resolve publication
       ↓
remove/disable EventBridge schedule
       ↓
verify result
       ↓
ledger → cancelled
```

No Meta post needs to exist yet.

---

## Rescheduling

EventBridge Scheduler exposes `UpdateSchedule`, and its current API includes create/get/list/update/delete schedule operations.

Source:

- https://docs.aws.amazon.com/scheduler/latest/APIReference/Welcome.html

Therefore moving a post can update/recreate the one-time schedule after the publication's approval rules are satisfied.

As with cancellation, the Eirepolitic publication ledger remains authoritative; the EventBridge schedule is execution infrastructure.

---

## Retry policy

EventBridge Scheduler supports configurable retries with exponential backoff.

Current limits allow:

- event retention up to 24 hours;
- up to 185 retry attempts.

Sources:

- https://docs.aws.amazon.com/scheduler/latest/APIReference/API_RetryPolicy.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/getting-started.html

### Eirepolitic recommendation

Do **not** configure 185 retries merely because AWS allows it.

The actual retry policy should be deliberately smaller and aligned with publication behaviour.

Also distinguish:

```text
EventBridge failed to invoke Lambda
```

from:

```text
Lambda invoked successfully but Meta publishing failed
```

The latter is application-level retry/reconciliation and belongs in Step 16.

---

## At-least-once delivery

AWS documents EventBridge Scheduler as using **at-least-once delivery** to targets.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html

This means duplicate Lambda invocation must be considered possible.

Therefore:

```text
EventBridge Scheduler
       ↓ maybe duplicate delivery
Lambda
       ↓
publication idempotency lock/state check
```

is mandatory.

EventBridge reliability does not remove the need for application-level idempotency.

---

## Dead-letter queue

EventBridge Scheduler can send failed target invocations to an **Amazon SQS standard queue** after retries are exhausted.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/configuring-schedule-dlq.html

Recommended initial use:

```text
EventBridge Scheduler
       ↓ invocation repeatedly fails
SQS DLQ
       ↓
monitor/alert/operator investigation
```

A FIFO queue cannot be used as the Scheduler DLQ; AWS specifically requires a standard SQS queue for this purpose.

---

## Automatic deletion

AWS allows one-time schedules to delete themselves after successful completion/invocation.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule-delete.html

This is useful operationally because old one-time scheduler resources do not need to remain indefinitely.

The permanent historical record belongs in the Eirepolitic publication ledger, not in EventBridge Scheduler.

---

## Cost

AWS currently includes:

```text
14,000,000 EventBridge Scheduler invocations/month free
```

and then charges approximately $1 per million invocations beyond the free tier.

Source:

- https://aws.amazon.com/eventbridge/pricing/

Eirepolitic will use perhaps tens of invocations/month initially, not millions.

### Conclusion

Scheduler invocation cost is effectively negligible.

---

# 3. AWS Lambda

## Fit: **excellent as the deterministic publisher worker**

Lambda is not the scheduler itself in the recommended design.

Its role is:

```text
EventBridge triggers publication_id
       ↓
Lambda loads and validates publication
       ↓
Lambda executes Meta API workflow
       ↓
Lambda records result
```

Benefits:

- no server to operate;
- native EventBridge integration;
- IAM/secrets integration;
- CloudWatch logs/metrics;
- inexpensive at low volume;
- straightforward Python implementation that fits the repository's existing language.

---

## Runtime suitability

Static image/carousel publication is a bounded API workflow and is well suited to a normal Lambda function.

Typical work:

- database reads/writes;
- S3 URL generation;
- HTTP calls to Meta;
- short processing/status polling;
- result persistence.

If later Reel/video processing requires substantially longer asynchronous waits or complex recovery, Step Functions can be added then.

---

## Idempotency

AWS recommends idempotent serverless/application logic because duplicate events/retries can occur.

The publication Lambda must therefore use the publication ledger/state machine to prevent duplicate publication.

The detailed design is Step 16.

---

## Cost

AWS Lambda's current free tier includes:

- 1 million function requests/month;
- 400,000 GB-seconds/month.

Source:

- https://aws.amazon.com/lambda/pricing/

At Eirepolitic publication volume, Lambda cost should be effectively negligible unless an unexpectedly expensive video workflow is introduced.

---

# 4. AWS Step Functions

## Fit: **good technology, not required for v1**

Step Functions can orchestrate long-running deterministic workflows.

A Standard Workflow:

- can run for up to one year;
- records execution history;
- supports visual debugging;
- supports service integrations;
- has exactly-once workflow execution semantics at the workflow level.

AWS also provides `Wait` states that can wait until an absolute RFC3339 timestamp.

Sources:

- https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html
- https://docs.aws.amazon.com/step-functions/latest/dg/state-wait.html

---

## Could Step Functions itself be the scheduler?

Technically yes.

One possible design would be:

```text
start state machine when publication approved
       ↓
Wait until scheduled timestamp
       ↓
publish
```

But this creates a long-lived workflow execution for every future publication.

EventBridge Scheduler already provides a cleaner purpose-built one-time scheduling resource with update/delete/retry/DLQ semantics.

### Recommendation

Do **not** use Step Functions merely to wait for publication time.

---

## Where Step Functions could become useful

Later, a Reel/video workflow may look like:

```text
create Meta video container
       ↓
wait
       ↓
check status
       ↓
retry wait/check
       ↓
publish
       ↓
first comment
       ↓
reconcile
```

If this becomes complex enough, Step Functions could make the orchestration and recovery more visible than custom Lambda polling logic.

That should be justified by real complexity, not introduced pre-emptively.

---

## Cost

Step Functions Standard includes 4,000 free state transitions/month and then bills per transition.

Source:

- https://aws.amazon.com/step-functions/pricing/

At Eirepolitic scale the cost would still be tiny, but cost is not the reason to avoid it in v1; unnecessary architecture is.

---

# 5. Amazon SQS

## Fit: **useful as DLQ; unnecessary as primary scheduler in v1**

SQS is a durable message queue, not a calendar scheduler.

Standard SQS queues provide at-least-once delivery and messages can occasionally be delivered more than once or out of order.

Source:

- https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html

### Recommended initial role

Use one standard SQS queue as the EventBridge Scheduler DLQ.

This gives a durable place for trigger-delivery failures without adding an unnecessary queue hop to every successful publication.

Potential v1:

```text
EventBridge Scheduler
       ↓
Lambda

failed trigger delivery
       ↓
SQS DLQ
```

not:

```text
EventBridge → SQS → Lambda
```

unless later load/decoupling requirements justify it.

---

## Why not use SQS as the publication clock?

SQS is not designed to hold arbitrary calendar-time jobs over long horizons as the main scheduling abstraction.

EventBridge Scheduler already provides the one-time date/time semantics directly.

SQS can be added later for buffering/concurrency control if Eirepolitic begins sending many platforms/accounts simultaneously.

---

## Cost

SQS currently includes 1 million requests/month in its free tier.

Source:

- https://aws.amazon.com/sqs/pricing/

A low-volume DLQ should incur effectively negligible cost.

---

# 6. GitHub Actions scheduled workflows

## Fit: **not sufficiently reliable as the publication clock**

GitHub Actions cron looks attractive because the repo already uses GitHub Actions.

However, GitHub's own current documentation explicitly warns:

- scheduled workflows may be delayed during periods of high load;
- high load commonly occurs at the start of the hour;
- if load is sufficiently high, queued scheduled jobs may be **dropped**;
- scheduled workflows run from the default branch;
- the shortest schedule interval is five minutes.

Sources:

- https://docs.github.com/en/actions/how-tos/troubleshoot-workflows
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows

GitHub now supports IANA timezone strings for scheduled workflows, which is useful, but it does not solve the documented delivery-reliability issue.

---

## Why this matters

For CI/report generation, a delayed cron run may be acceptable.

For:

```text
"Publish the post at 7:30pm"
```

an authoritative scheduler should not be based on a service whose documentation says the run can be delayed or dropped under load.

### Verdict

Keep GitHub Actions for:

- generation;
- tests;
- CI/CD;
- manual dry runs;
- infrastructure deployment;

but **not** as the production social-publication clock.

---

# 7. Power Automate

## Fit: **possible, but not preferred as authoritative publication scheduler**

Power Automate supports scheduled cloud flows with start times and recurrence rules.

Source:

- https://learn.microsoft.com/en-us/power-automate/run-scheduled-tasks

However, Microsoft also documents trigger-delay behaviour and plan-dependent trigger/polling timing. Depending on licensing/trigger type, flows can wait several minutes or be queued according to plan limits.

Sources:

- https://learn.microsoft.com/en-us/troubleshoot/power-platform/power-automate/flow-run-issues/triggers-troubleshoot
- https://learn.microsoft.com/en-us/power-automate/limits-and-config

### Why not select it

Using Power Automate for the publication clock would introduce:

- another cloud/runtime platform;
- separate connection/secrets management;
- Microsoft licensing/flow limits;
- another audit/logging surface;
- less natural integration with S3/AWS runtime state;
- more difficult infrastructure-as-code/version-control compared with a small AWS scheduler setup.

It does not provide enough benefit to offset those costs when the publication assets already use AWS/S3.

### Useful future role

Power Automate could still be useful for human-facing operational tasks, for example:

```text
publication fails
       ↓
SNS/email/other signal
       ↓
optional Power Automate notification/approval workflow
```

but it should not be the authoritative publication timer.

---

# 8. Comparison table

| Option | One-time scheduling | Timing suitability | Retries/DLQ | Cancel/reschedule | Cost at Eirepolitic scale | Operational complexity | Recommendation |
|---|---|---|---|---|---|---|---|
| **EventBridge Scheduler** | **Native** | **Within scheduled minute** | **Native retries + SQS DLQ** | **Native API** | **Negligible** | Low | **Use as clock** |
| **Lambda** | Not itself | N/A | Invocation/application retries | N/A | **Negligible** | Low | **Use as worker** |
| Step Functions | Can wait to timestamp | Good | Strong workflow retry/state | Execution management possible | Very low | Medium | Later if workflow complexity justifies it |
| SQS | Not a calendar scheduler | N/A | Durable queue/DLQ | Not natural | **Negligible** | Low | **Use as DLQ initially** |
| GitHub Actions cron | Recurring cron | **Documented delays/drops** | Workflow retry must be custom | Awkward per-post workflow/schedule changes | Low | Low initially | **Do not use as publication clock** |
| Power Automate | Scheduled cloud flows | Plan/runtime dependent | Platform mechanisms | Possible but operationally awkward per post | License-dependent | Medium | Optional human workflow only |

---

# 9. Recommended direct-Meta runtime after Step 9

If direct Meta is selected later, the smallest appropriate runtime is:

```text
High Director / publication control layer
       ↓
Publication ledger
       ↓
EventBridge Scheduler
       ↓
Lambda publisher
       ↓
Meta Instagram API

EventBridge trigger failures
       ↓
SQS DLQ
```

Supporting services likely required later:

```text
S3             approved assets
Secrets store  Meta credential
CloudWatch     logs/metrics/alarms
```

The exact ledger/secrets/monitoring decisions are later research steps.

---

# 10. Cancellation/rescheduling behaviour

The architecture should make the Eirepolitic publication record authoritative.

## Schedule

```text
ledger: approved
       ↓
create EventBridge one-time schedule
       ↓
verify schedule
       ↓
ledger: scheduled
```

## Reschedule

```text
new approved scheduled instant
       ↓
update/recreate EventBridge schedule
       ↓
verify schedule
       ↓
ledger records new schedule
```

## Cancel

```text
cancel requested/confirmed
       ↓
delete/disable EventBridge schedule
       ↓
verify deletion/state
       ↓
ledger: cancelled
```

Do not mark the ledger operation complete before the scheduler operation is confirmed.

---

# 11. Scheduler payload design

Recommended payload:

```json
{
  "publication_id": "pub_..."
}
```

Optionally include a non-authoritative expected publication version/fingerprint for additional protection:

```json
{
  "publication_id": "pub_...",
  "expected_version": 3
}
```

The Lambda must still load and validate the authoritative publication record.

Do not place:

- Meta tokens;
- full captions;
- presigned media URLs;
- temporary Meta container IDs;

in the EventBridge schedule payload.

---

# 12. Monitoring requirements

EventBridge Scheduler exposes CloudWatch metrics including invocation attempts, target errors and DLQ delivery counts.

AWS specifically recommends monitoring metrics such as:

- `InvocationAttemptCount`;
- `TargetErrorCount`;
- `InvocationsSentToDeadLetterCount`.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/troubleshooting.html

The eventual monitoring design is Step 17, but EventBridge already exposes the necessary operational signals.

---

# 13. Reliability model

Important distinction:

```text
Reliable scheduler
≠
exactly-once social publication
```

EventBridge Scheduler uses at-least-once delivery.

Lambda/API calls can timeout.

Meta can accept a publication while our response is lost.

Therefore the full system must combine:

```text
reliable schedule trigger
+
idempotent publication worker
+
publication ledger
+
Meta reconciliation
```

No scheduler choice can replace Step 16's idempotency design.

---

# 14. Impact on direct Meta vs Buffer decision

Step 8 asked whether Buffer removes enough scheduling infrastructure to justify the additional vendor dependency.

Step 9 finds that the direct scheduling layer is quite small:

```text
one EventBridge schedule per publication
+
one Lambda publisher
+
one SQS DLQ
```

At Eirepolitic scale, the AWS scheduling/runtime cost is effectively negligible.

Therefore Buffer's advantage is **not primarily the clock**.

Its meaningful simplification is instead:

- managing the Instagram connection/token;
- abstracting the Meta media-container/publish API;
- providing an operator social calendar/UI;
- potentially serving multiple social platforms through one API.

Direct Meta remains somewhat more engineering work, but owning the timer itself is **not a significant complexity or cost problem**.

This finding modestly strengthens the direct-Meta option relative to Buffer.

---

# 15. Step 9 verdict

If direct Meta is ultimately selected:

## Recommend

```text
EventBridge Scheduler → Lambda publisher
```

with:

```text
SQS standard DLQ
```

and normal CloudWatch monitoring.

## Do not use initially

- GitHub Actions cron as publication clock;
- Step Functions solely to wait until a timestamp;
- SQS as a replacement calendar scheduler;
- Power Automate as authoritative publication timer.

## Add later only if justified

- Step Functions for complex video/Reel processing;
- SQS in the normal execution path for higher concurrency/decoupling;
- Power Automate for human notifications/approvals.

This is proportional to the size of the project and leaves a clean path to multi-platform publishing later.

---

## Sources

### AWS EventBridge Scheduler

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule.html
- https://docs.aws.amazon.com/scheduler/latest/APIReference/Welcome.html
- https://docs.aws.amazon.com/scheduler/latest/APIReference/API_DeleteSchedule.html
- https://docs.aws.amazon.com/scheduler/latest/APIReference/API_RetryPolicy.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/configuring-schedule-dlq.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule-delete.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/troubleshooting.html
- https://aws.amazon.com/eventbridge/scheduler/
- https://aws.amazon.com/eventbridge/pricing/

### AWS Lambda

- https://aws.amazon.com/lambda/pricing/
- https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html

### AWS Step Functions

- https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html
- https://docs.aws.amazon.com/step-functions/latest/dg/state-wait.html
- https://aws.amazon.com/step-functions/pricing/

### Amazon SQS

- https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html
- https://aws.amazon.com/sqs/pricing/

### GitHub Actions

- https://docs.github.com/en/actions/how-tos/troubleshoot-workflows
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows

### Power Automate

- https://learn.microsoft.com/en-us/power-automate/run-scheduled-tasks
- https://learn.microsoft.com/en-us/troubleshoot/power-platform/power-automate/flow-run-issues/triggers-troubleshoot
- https://learn.microsoft.com/en-us/power-automate/limits-and-config

---

## Confidence / unresolved items

**High confidence:**

- EventBridge Scheduler is the best-fit direct publication clock among the reviewed options;
- it supports one-time jobs, update/delete, timezone/DST handling, retries, DLQ and 60-second precision;
- EventBridge uses at-least-once delivery, so application idempotency is mandatory;
- GitHub Actions scheduled workflows are explicitly documented as subject to delay/drop under high load;
- Lambda is appropriate for the first static image/carousel publisher;
- Step Functions is not needed merely to wait until publication time;
- SQS is useful as the scheduler DLQ;
- direct AWS scheduling cost is negligible at Eirepolitic volume.

**Still to design later:**

- publication ledger/database;
- approval/state model;
- exact asset storage/delivery;
- timezone canonicalization rules;
- secrets store;
- application retry/idempotency logic;
- monitoring/notifications.

**Next research step:**

Step 10 will design the publication data model and cleanly separate content intent from execution state.
