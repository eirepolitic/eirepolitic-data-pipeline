# Step 14 — Timezones and Daylight-Saving Handling

Status: **complete**

Research date: 2026-09-04

Scope: define how Eirepolitic should represent, confirm and execute publication times for an Ireland-focused audience, including `Europe/Dublin`, UTC conversion, daylight-saving transitions, ambiguous/nonexistent local times and EventBridge Scheduler behaviour.

No scheduler, timezone conversion code, account connection, or live publication was created.

---

## Short conclusion

Use **IANA timezone identifiers**, specifically:

```text
Europe/Dublin
```

for Irish publication intent.

For each one-time publication, store all three of these values:

```yaml
scheduled_local: "2026-09-08T19:30:00"
timezone: Europe/Dublin
scheduled_at_utc: "2026-09-08T18:30:00Z"
```

The human-facing meaning is the local time + IANA timezone.

The deterministic execution identity is the resolved UTC instant.

For one-time approved publications, the recommended scheduler design is to execute the **resolved UTC instant** after validating the local Dublin time. This freezes the exact approved instant and prevents later timezone-database/rule changes from silently changing an already-approved job.

Do not store `GMT`, `IST`, `UTC+1` or a fixed numeric offset as the canonical timezone rule.

If a requested local time falls inside a daylight-saving transition and is ambiguous or nonexistent, High Director/system validation must **ask the human to resolve it** rather than guessing.

---

# 1. Why `Europe/Dublin`

`Europe/Dublin` is an IANA Time Zone Database identifier.

AWS EventBridge Scheduler uses the IANA Time Zone Database for timezone-aware scheduling.

Source:

- AWS EventBridge Scheduler schedule types: https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html

Using an IANA zone means the system applies the historical/current civil-time rules for Ireland rather than manually maintaining fixed GMT/summer offsets.

---

# 2. Do not use fixed offsets as the human timezone

Avoid storing the canonical schedule as:

```text
UTC+0
UTC+1
GMT
IST
```

Reasons:

- a fixed offset does not encode daylight-saving transitions;
- labels such as `IST` are ambiguous internationally;
- manually choosing GMT vs summer time is error-prone;
- future timezone-rule changes belong in the IANA timezone database, not application code.

Use:

```text
Europe/Dublin
```

and let a timezone-aware library resolve the offset for the requested date.

---

# 3. Recommended stored schedule fields

A `PublicationSchedule` should store at least:

```yaml
scheduled_local: "2026-09-08T19:30:00"
timezone: Europe/Dublin
scheduled_at_utc: "2026-09-08T18:30:00Z"
```

Optionally also store audit metadata such as:

```yaml
resolved_utc_offset: "+01:00"
time_resolution:
  status: unambiguous
  resolved_at: "..."
```

The offset is useful audit information but is not the timezone identity.

---

# 4. Why store both local and UTC

## Local + timezone answers

```text
What did the human ask for?
```

Example:

```text
19:30 Europe/Dublin
```

## UTC answers

```text
Exactly what instant should the scheduler execute?
```

Example:

```text
18:30Z
```

Both should remain in the ledger.

This lets High Director say:

```text
Scheduled for 7:30pm Dublin time (18:30 UTC).
```

and gives the execution system one unambiguous instant.

---

# 5. High Director input behaviour

Examples:

```text
"next Tuesday at 7:30pm"
"Friday at 8"
"tomorrow evening at 7"
```

High Director should resolve these using the publication/account's configured default timezone unless the user specifies another timezone.

For Eirepolitic's Irish audience, the default should be:

```text
Europe/Dublin
```

The final confirmation must display the resolved absolute date, local time and timezone.

Example:

```text
Tuesday 8 September 2026, 19:30 Europe/Dublin
```

not merely:

```text
next Tuesday at 7:30
```

This protects against relative-date misunderstandings.

---

# 6. Final confirmation must include UTC resolution

Recommended final schedule confirmation:

```text
Tuesday 8 September 2026
19:30 Europe/Dublin
18:30 UTC
```

The human primarily confirms the Dublin wall-clock time.

The UTC value makes the execution instant explicit and auditable.

Both become part of the publication approval/schedule record.

---

# 7. Daylight-saving transitions

Civil clocks can create two unusual cases.

## Nonexistent local time

When clocks move forward, a range of local wall-clock times does not occur.

If the human requests a time inside that missing range, the system must not silently move it forward/backward.

Required behaviour:

```text
requested local time
      ↓
timezone resolver says nonexistent
      ↓
no schedule created
      ↓
High Director asks for a valid time
```

Example user-facing concept:

```text
That local time does not occur in Europe/Dublin because the clocks change that night. Choose another time.
```

---

## Ambiguous local time

When clocks move backward, some local wall-clock times occur twice with different UTC offsets.

If the human requests one of those repeated times, the system must not guess which occurrence is intended.

Required behaviour:

```text
requested local time
      ↓
timezone resolver finds two UTC candidates
      ↓
no schedule created
      ↓
High Director asks which occurrence is intended
```

The confirmation can show both candidates clearly using local offset/UTC time.

---

# 8. Deterministic timezone resolver

The publication-control service, not the LLM alone, should validate local time.

Python's standard `zoneinfo` module uses IANA timezone data and supports disambiguation of repeated times through the `fold` attribute.

Source:

- Python `zoneinfo`: https://docs.python.org/3/library/zoneinfo.html

A future deterministic resolver should:

1. parse the requested local date/time;
2. load `ZoneInfo("Europe/Dublin")`;
3. determine whether the wall time is valid;
4. determine whether it maps to one or two UTC instants;
5. reject nonexistent times;
6. require an explicit choice for ambiguous times;
7. return one resolved UTC instant;
8. store the local value, timezone and UTC result.

Do not rely on the LLM to calculate daylight-saving offsets mentally.

---

# 9. EventBridge Scheduler timezone behaviour

AWS EventBridge Scheduler supports:

- one-time schedules;
- cron schedules;
- explicit IANA timezone selection;
- daylight-saving-aware evaluation.

AWS documents that:

- during spring-forward, a recurring cron time that does not exist is skipped;
- during fall-back, an ambiguous recurring cron time runs once rather than twice;
- target invocation precision is within the scheduled minute when flexible windows are disabled.

Source:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html

---

# 10. Recommendation for one-time publication jobs

Eirepolitic is scheduling **individual publications**, not an unattended recurring daily cron.

Therefore use this approach:

```text
human selects Dublin local time
       ↓
validate with Europe/Dublin
       ↓
resolve exact UTC instant
       ↓
human confirms
       ↓
store local + timezone + UTC
       ↓
create one-time EventBridge schedule for exact approved instant
```

This is preferable to allowing EventBridge to make an implicit DST choice for an ambiguous/nonexistent local request.

---

# 11. Why execute the approved UTC instant

EventBridge can create a one-time schedule using a local time and a timezone.

However, once a human has approved both:

```text
19:30 Europe/Dublin
```

and the resulting:

```text
18:30Z
```

the system should preserve that exact approved instant.

If timezone rules/database data changed between approval and execution, re-evaluating the local time later could theoretically produce a different UTC result.

Therefore for a one-time publication:

```text
approval freezes scheduled_at_utc
```

and the scheduler should execute that frozen instant.

The original `Europe/Dublin` value remains in the ledger for human interpretation/audit.

---

# 12. EventBridge schedule representation

For the recommended one-time model, a future schedule may conceptually use:

```text
at(<resolved UTC date/time>)
```

with UTC semantics, rather than depending on a later re-resolution of the Dublin wall clock.

Alternatively the implementation can supply the local time + timezone while also verifying through `GetSchedule` that the resulting configuration matches the approved intent.

The important invariant is:

```text
scheduler execution instant == approved scheduled_at_utc
```

The final implementation can choose the cleanest AWS API representation that guarantees this.

---

# 13. Flexible time window

EventBridge Scheduler supports flexible delivery windows.

For social publication schedules, use:

```text
FlexibleTimeWindow = OFF
```

because the user requested a specific publication minute.

Even with the flexible window disabled, AWS documents 60-second target precision, so a 19:30 schedule can invoke during the 19:30 minute rather than necessarily at exactly `19:30:00`.

Sources:

- https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html
- https://docs.aws.amazon.com/scheduler/latest/UserGuide/getting-started.html

This is acceptable for Eirepolitic social publishing and should be reflected in expectations/logging.

---

# 14. Rescheduling

When the user says:

```text
"Move tomorrow's post to 8pm."
```

High Director should:

1. identify the exact publication;
2. resolve `20:00 Europe/Dublin` on the absolute date;
3. calculate the new UTC instant;
4. show both local and UTC values;
5. obtain explicit schedule-change confirmation under Step 11;
6. update/recreate the EventBridge schedule;
7. verify scheduler state;
8. write a new `PublicationSchedule` version/history entry.

Do not mutate the UTC instant without preserving the previous schedule history.

---

# 15. Changing timezone explicitly

If the human says:

```text
"Actually schedule it for 7:30pm London time."
```

that is a schedule change.

The system should store the explicitly requested IANA timezone rather than converting it into the account default.

Example:

```yaml
timezone: Europe/London
```

The final confirmation must show the changed timezone.

Do not silently normalize all schedules back to Dublin if the human explicitly chose another zone.

---

# 16. Account default timezone

Recommended account configuration:

```yaml
account_ref: eirepolitic
publication_defaults:
  timezone: Europe/Dublin
```

This allows High Director to interpret ordinary requests without repeatedly asking:

```text
"Which timezone?"
```

The final confirmation still shows the timezone even when it came from the default.

---

# 17. Publication manifest relationship

The publication request/approval fingerprint should bind to the resolved schedule identity or its associated schedule record according to Step 11's policy.

Recommended schedule representation:

```yaml
delivery:
  mode: scheduled
  scheduled_local: "2026-09-08T19:30:00"
  timezone: Europe/Dublin
  scheduled_at_utc: "2026-09-08T18:30:00Z"
```

A schedule-only change can preserve the same approved content version while creating a new confirmed `PublicationSchedule` record.

---

# 18. Database/storage rules

Store UTC timestamps in an unambiguous ISO 8601/RFC 3339 form, for example:

```text
2026-09-08T18:30:00Z
```

Store IANA timezone separately:

```text
Europe/Dublin
```

Do not store only:

```text
2026-09-08 19:30
```

because it is incomplete without timezone semantics.

Do not store only UTC either, because that loses the human's local-time intent.

---

# 19. Logging

Execution records should log both:

```text
scheduled_at_utc
actual_started_at_utc
```

The system can then calculate scheduling latency.

Human-facing views can render those timestamps back into `Europe/Dublin` or the schedule's original timezone.

This allows High Director to answer:

```text
"It was scheduled for 19:30 Dublin time and the publisher started at 19:30:27."
```

without ambiguous clock arithmetic.

---

# 20. Future recurring schedules

The current publication model is based on explicit one-time approved posts.

If Eirepolitic later introduces recurring automation such as:

```text
publish every Friday at 19:30
```

then EventBridge's timezone-aware cron support becomes useful.

But recurring schedules need an explicit DST policy because AWS documents that:

- nonexistent spring-forward occurrences are skipped;
- repeated fall-back occurrences run only once.

Do not reuse the one-time approval model blindly for recurring unattended publication rules.

Recurring publication should be a separate future architecture/policy decision.

---

# 21. High Director safety rule

High Director may interpret conversational dates/times, but deterministic code must validate the result.

High Director should never tell the scheduler merely:

```text
"next Tuesday evening"
```

The scheduler receives only an explicitly resolved schedule record.

Final execution therefore never depends on natural-language date interpretation.

---

# 22. Step 14 verdict

Recommended timezone architecture:

```text
Human request
   ↓
Europe/Dublin local wall time
   ↓
deterministic IANA timezone validation
   ↓
reject nonexistent / disambiguate repeated time
   ↓
resolve UTC instant
   ↓
human confirms local + timezone + UTC
   ↓
store all three
   ↓
one-time scheduler executes frozen UTC instant
```

Key rules:

1. `Europe/Dublin` is the default publication timezone.
2. Never manually switch between GMT/Irish summer offsets.
3. Store local datetime + IANA timezone + resolved UTC instant.
4. Reject nonexistent DST times instead of silently adjusting them.
5. Require human selection when a local time occurs twice.
6. Use deterministic timezone libraries, not LLM arithmetic.
7. Freeze the approved UTC instant for one-time publications.
8. Keep EventBridge flexible windows off.
9. Schedule-only changes require explicit confirmation and create schedule history.
10. Treat future recurring schedules as a separate policy because DST semantics differ.

---

## Sources

### AWS

- EventBridge Scheduler schedule types / timezones / DST: https://docs.aws.amazon.com/scheduler/latest/UserGuide/schedule-types.html
- EventBridge Scheduler CreateSchedule API: https://docs.aws.amazon.com/scheduler/latest/APIReference/API_CreateSchedule.html
- EventBridge Scheduler getting started / flexible windows: https://docs.aws.amazon.com/scheduler/latest/UserGuide/getting-started.html

### Python

- Python standard-library `zoneinfo`: https://docs.python.org/3/library/zoneinfo.html

---

## Confidence / unresolved items

**High confidence:**

- use of IANA `Europe/Dublin` is preferable to manual GMT/summer offsets;
- EventBridge Scheduler uses IANA timezone data and supports DST-aware one-time/cron schedules;
- EventBridge Scheduler has 60-second target precision;
- local and UTC representations should both be stored;
- ambiguous/nonexistent local times must be resolved before schedule creation;
- one-time publication approval is safest when it freezes one resolved UTC execution instant.

**Still to determine during implementation:**

- exact timezone-validation utility implementation/tests;
- exact AWS one-time schedule representation used by the adapter;
- whether UTC and timezone database version metadata need to be persisted for forensic audit (probably unnecessary at current scale);
- final UX wording for ambiguous DST choices.

**Next research step:**

Step 15 will define secrets and Meta token management, including AWS Secrets Manager, GitHub secret boundaries, rotation/renewal, logging redaction and revoked-token handling.
