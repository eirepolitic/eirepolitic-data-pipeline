# Step 10 — Publication Data Model

Status: **complete**

Research date: 2026-09-03

Scope: design the logical records that separate approved content intent from scheduling and API execution state. This is a research/design artifact only; no schemas, database tables, publishing code, or infrastructure were implemented.

---

## Short conclusion

The publication system should **not** use one large mutable manifest containing both human intent and Meta runtime fields.

Use six logical records:

1. `AssetPackage` — immutable approved publication assets and content-readiness evidence;
2. `PublicationRequest` — exact human-visible intent for one publication version;
3. `PublicationApproval` — who approved exactly which request fingerprint;
4. `PublicationSchedule` — when that approved request is intended to execute;
5. `ExecutionAttempt` — runtime/API work, temporary container IDs, retries and errors;
6. `PublishedMedia` — resulting Instagram/media IDs, permalink and final delivery facts.

The existing repo's `publish_ready` / `review_status` fields remain useful, but they should mean **the generated asset package is eligible to be considered for publication**, not that it is authorized to post to Instagram.

---

# 1. Core separation

The model should preserve this boundary:

```text
CONTENT / HUMAN INTENT

AssetPackage
    ↓
PublicationRequest
    ↓
PublicationApproval
    ↓
PublicationSchedule

---------------- deterministic execution boundary ----------------

ExecutionAttempt
    ↓
PublishedMedia
```

Temporary API values belong below the boundary.

Human-approved values belong above it.

---

# 2. AssetPackage

## Purpose

Represents a versioned set of generated assets that has passed the post-generation review process.

It answers:

```text
Exactly which files are approved inputs for publication?
```

## Example logical shape

```yaml
asset_package_id: asset_01J...
schema_version: 1

project_id: party_speech_breakdown
period: "2026-08"
content_version: 2

source:
  repository: Eirepolitic-data-pipeline
  commit_sha: abc123...
  generation_run_id: ...

media:
  - asset_id: slide_01
    ordinal: 1
    object_uri: s3://.../01.jpg
    mime_type: image/jpeg
    width: 1080
    height: 1350
    size_bytes: 123456
    sha256: ...
    alt_text: "..."

  - asset_id: slide_02
    ordinal: 2
    object_uri: s3://.../02.jpg
    mime_type: image/jpeg
    width: 1080
    height: 1350
    size_bytes: 123456
    sha256: ...
    alt_text: "..."

readiness:
  expected_media_count: 2
  qa_status: passed
  human_visual_review_status: approved
  safety_notes: []
  publication_ready: true

review:
  reviewed_by: human_ref
  reviewed_at: 2026-09-03T18:00:00Z
```

## Rules

- immutable after approval;
- ordered media explicit;
- hashes mandatory;
- no temporary presigned URLs;
- no Meta container IDs;
- no schedule;
- no target social account;
- no publication authorization.

If an image changes, create a new `AssetPackage` version/ID.

---

# 3. Relationship to existing repo review fields

Current repo fields include:

```text
publish_ready
review_status
safety_notes
caption
alt_text
hashtags
```

The current queue correctly blocks assets unless content review passes.

However, future semantics should be clearer:

```text
publish_ready=yes
```

means:

> This generated content package is eligible to enter the publication-control workflow.

It must **not** mean:

> Instagram is authorized to publish this now.

That second authorization belongs to `PublicationApproval`.

This prevents an old generation review flag from accidentally becoming a live-post permission.

---

# 4. PublicationRequest

## Purpose

Represents exactly what the human intends to happen on one target platform/account.

It is the central deterministic statement produced from the High Director conversation.

## Recommended identity

```text
publication_id
publication_version
```

Example:

```text
publication_id: pub_01JABC...
publication_version: 3
```

A content edit creates a new publication version, not an invisible mutation of the approved version.

---

## Proposed platform-neutral structure

```yaml
schema_version: 1

publication_id: pub_01J...
publication_version: 3

content:
  project_id: party_speech_breakdown
  period: "2026-08"
  asset_package_id: asset_01J...

caption:
  text: |
    Exact final caption goes here.
  template_ref:
    template_id: party_speech_breakdown
    template_version: 4

caption_entities:
  mentions:
    - username: example
  hashtags:
    - "#EirePolitic"
    - "#IrishPolitics"

target:
  platform: instagram
  account_ref: eirepolitic

media:
  - asset_id: slide_01
    alt_text: "..."
    tags: []

  - asset_id: slide_02
    alt_text: "..."
    tags:
      - username: example
        position:
          x: 0.50
          y: 0.40

platform_options:
  instagram:
    post_type: carousel
    collaborators: []
    location: null
    first_comment:
      text: null

created_by: high_director
created_at: 2026-09-03T18:30:00Z
```

---

# 5. Why hashtags and mentions should be both explicit and structured

The canonical publication text is:

```yaml
caption.text
```

If that exact text contains:

```text
@example
#EirePolitic
```

then that is what the publisher sends.

Structured fields such as:

```yaml
caption_entities:
  mentions: ...
  hashtags: ...
```

are useful for:

- validation;
- conversational summaries;
- analytics;
- checking that requested mentions actually appear;
- template defaults.

But the publisher must not regenerate or reassemble the caption from those structured pieces at execution time.

---

# 6. Platform-neutral intent vs platform-specific extensions

Keep generally reusable fields at the common level:

- content/project identity;
- caption text;
- target account;
- ordered assets;
- alt/accessibility text;
- requested publication schedule.

Put Instagram-only concepts under an extension:

```yaml
platform_options:
  instagram:
    post_type: carousel
    collaborators: []
    location: null
    first_comment: ...
```

This avoids hard-wiring the entire conversation model to Instagram while avoiding premature over-engineering.

A future LinkedIn target could use:

```yaml
platform_options:
  linkedin:
    ...
```

without redesigning the core records.

---

# 7. PublicationApproval

## Purpose

Proves that a human approved a specific immutable publication request.

Recommended shape:

```yaml
approval_id: appr_01J...
publication_id: pub_01J...
publication_version: 3

request_fingerprint: sha256:...

status: approved
approved_by:
  type: human
  actor_ref: ...

approved_at: 2026-09-03T19:00:00Z

confirmation_snapshot:
  account_ref: eirepolitic
  project_id: party_speech_breakdown
  period: "2026-08"
  media_count: 8
  caption_sha256: ...
  scheduled_local: "2026-09-08T19:30:00"
  timezone: Europe/Dublin
```

The approval should reference a cryptographic fingerprint of the canonical publication request.

---

# 8. Approval fingerprint

Recommended concept:

```text
SHA256(canonical JSON of all material publication fields)
```

Material fields should include at least:

- target platform/account;
- project/period;
- publication version;
- asset package ID;
- ordered asset IDs;
- asset hashes;
- exact caption;
- alt text;
- caption mentions/hashtags;
- media tags;
- collaborators;
- location;
- first comment;
- scheduled local time;
- timezone;
- resolved execution instant.

If any material field changes:

```text
old fingerprint != new fingerprint
```

and the old approval cannot authorize the new request.

---

# 9. PublicationSchedule

## Purpose

Separates scheduling/execution timing from content definition.

Recommended shape:

```yaml
schedule_id: sched_01J...
publication_id: pub_01J...
publication_version: 3
approval_id: appr_01J...

mode: scheduled

scheduled_local: "2026-09-08T19:30:00"
timezone: Europe/Dublin
scheduled_at_utc: "2026-09-08T18:30:00Z"

status: active

scheduler:
  provider: eventbridge
  external_schedule_name: null

created_at: ...
updated_at: ...
```

For immediate publishing:

```yaml
mode: immediate
scheduled_at_utc: <confirmed execution instant>
```

Even `immediate` should still pass through the deterministic publication service and ledger; High Director should not bypass the control layer by directly calling Meta.

---

# 10. Why schedule should be separate

The same approved content may need:

```text
Tuesday 19:30
      ↓ user changes mind
Tuesday 20:00
```

Changing schedule should not require copying temporary API state into the publication request.

Whether a schedule change itself requires fresh approval is a policy question addressed in Step 11.

The model supports either policy cleanly.

---

# 11. ExecutionAttempt

## Purpose

Stores one deterministic attempt to execute an approved/scheduled publication.

This is where temporary Meta/API state belongs.

Example:

```yaml
attempt_id: attempt_01J...
publication_id: pub_01J...
publication_version: 3
schedule_id: sched_01J...

attempt_number: 1
started_at: ...
completed_at: ...

provider:
  name: meta
  api_version: ...

state: publishing

containers:
  - asset_id: slide_01
    meta_container_id: "..."
    status: FINISHED

  - asset_id: slide_02
    meta_container_id: "..."
    status: FINISHED

parent_container_id: "..."

operations:
  - operation: create_child_container
    status: succeeded
    provider_request_id: ...

  - operation: media_publish
    status: unknown
    provider_request_id: ...

error:
  category: null
  provider_code: null
  provider_subcode: null
  sanitized_message: null

retry:
  retryable: false
  retry_after: null
```

---

# 12. ExecutionAttempt rules

Execution records may contain:

- Meta container IDs;
- provider request IDs;
- processing status;
- retry counters;
- sanitized provider errors;
- timestamps;
- API version;
- reconciliation results.

They must not contain:

- Meta access tokens;
- app secrets;
- authorization headers;
- unredacted credential-bearing URLs.

Execution attempts are append-oriented audit records, not mutable content manifests.

---

# 13. PublishedMedia

## Purpose

Records the externally observed result after successful publication.

Example:

```yaml
published_media_id: result_01J...
publication_id: pub_01J...
publication_version: 3
attempt_id: attempt_01J...

platform: instagram
account_ref: eirepolitic

provider:
  instagram_media_id: "..."
  permalink: "https://www.instagram.com/p/.../"
  media_type: CAROUSEL_ALBUM

published_at: 2026-09-08T18:30:42Z

final_observed:
  caption: |
    ...

secondary_actions:
  first_comment:
    status: not_requested
```

This record answers:

```text
What actually went out?
```

rather than:

```text
What did we intend to go out?
```

Both are valuable and should remain distinct.

---

# 14. Provider-neutral external delivery reference

Even if direct Meta is selected initially, avoid making `meta_media_id` the system's primary publication identifier.

Use:

```text
publication_id = Eirepolitic identity
```

and store provider identifiers beneath it.

This allows a future Buffer or LinkedIn adapter without changing the conversational identity of the publication.

---

# 15. Logical uniqueness and versioning

A useful human/logical key is approximately:

```text
platform
+ account_ref
+ project_id
+ period
+ publication_version
```

But use a generated stable `publication_id` as the actual primary identity.

Examples:

### Retry of same publication

```text
publication_id: same
publication_version: same
attempt_id: new
```

### Caption corrected before publication

```text
publication_id: same
publication_version: incremented
approval: new
```

### Intentional second post of the same project/period

Use a **new publication_id** or an explicit repeat/republication relationship.

Do not disguise an intentional repost as a retry.

---

# 16. Proposed lifecycle/state ownership

Do not put every state on every record.

### AssetPackage

Example readiness states:

```text
generated
pending_review
approved
rejected
superseded
```

### PublicationRequest / publication aggregate

Conversation/policy states can later include:

```text
draft
pending_publication_approval
approved
scheduled
publishing
published
failed
cancelled
```

The exact state machine is Step 11.

### ExecutionAttempt

Runtime states can include:

```text
started
creating_containers
processing_media
publishing
reconciling
succeeded
failed
unknown
```

Keeping runtime details in the attempt record avoids exposing Meta-specific mechanics as human workflow states unless needed.

---

# 17. High Director read/write boundary

High Director should be allowed to operate on **control-plane records**:

```text
PublicationRequest
PublicationApproval
PublicationSchedule
```

and read:

```text
AssetPackage
ExecutionAttempt
PublishedMedia
```

High Director should not manually invent or edit:

- Meta container IDs;
- retry counters;
- provider request IDs;
- execution success flags.

Those are written by deterministic execution code.

This protects the publication ledger from conversational hallucination or accidental state manipulation.

---

# 18. Example end-to-end record relationship

```text
AssetPackage
asset_100
party_speech_breakdown / 2026-08
        │
        ▼
PublicationRequest
pub_200 / version 3
caption + ordered media + tags
        │
        ▼
PublicationApproval
appr_300
fingerprint(version 3)
        │
        ▼
PublicationSchedule
sched_400
2026-09-08 19:30 Europe/Dublin
        │
        ▼
ExecutionAttempt
attempt_500
Meta child/parent containers
        │
        ▼
PublishedMedia
result_600
Instagram Media ID + permalink
```

This chain makes a later audit straightforward.

---

# 19. Queries this model supports

## "What is scheduled this week?"

Query active `PublicationSchedule` records joined to their current approved `PublicationRequest` summaries.

## "What went out yesterday?"

Query `PublishedMedia.published_at` and load the associated final request.

## "Why did tonight's post fail?"

Load latest `ExecutionAttempt` and its sanitized error/reconciliation history.

## "Cancel Friday's post."

Resolve `PublicationSchedule`, follow approval/cancellation policy, cancel scheduler, persist state.

## "Move tomorrow's post to 8pm."

Create/update schedule version according to Step 11 approval policy.

## "Use the same caption structure as last month's Party Speech Breakdown."

Load prior `PublicationRequest.caption` and/or its referenced caption template—not the execution record.

---

# 20. Storage recommendation intentionally deferred

This step defines **logical records**, not their final physical database representation.

Possible implementations include:

- DynamoDB single-table or small multi-entity design;
- a relational database;
- another durable publication store.

The final architecture recommendation can choose the simplest appropriate implementation later.

Do not let a database technology dictate the logical model prematurely.

---

# 21. Minimum fields required before scheduling

A publication should not be schedulable unless these can be resolved deterministically:

```text
publication_id/version
asset_package_id
platform
account_ref
ordered media
exact final caption
alt text where applicable
caption mention/hashtag configuration
media tag configuration
collaborator configuration
location configuration
first-comment configuration
scheduled local date/time
IANA timezone
resolved UTC instant
valid human approval fingerprint
```

Optional features may explicitly be `none` / empty.

Absence should not be confused with an unanswered decision.

---

# 22. Null versus missing

This is important for conversational control.

For example:

```yaml
collaborators: null
```

could mean:

> no decision has been made yet.

while:

```yaml
collaborators: []
```

means:

> human confirmed there should be no collaborators.

Similarly:

```yaml
first_comment: null
```

can mean unresolved, while:

```yaml
first_comment:
  text: null
  explicitly_none: true
```

means the human explicitly chose no first comment.

The eventual schema should represent this distinction cleanly so High Director asks only genuinely missing questions.

---

# 23. Immutable snapshots versus mutable drafts

Recommended rule:

### Draft publication request

Mutable while discussing.

### Approved publication version

Immutable.

If changed after approval:

```text
clone/create version N+1
   ↓
edit
   ↓
new approval required
```

Execution always references an immutable approved version.

This makes it impossible for a scheduled job to silently pick up a newer unapproved caption.

---

# 24. Proposed conceptual schema summary

```text
AssetPackage
  └─ immutable reviewed assets

PublicationRequest
  └─ exact publication intent/version

PublicationApproval
  └─ human approval + fingerprint

PublicationSchedule
  └─ intended execution time + scheduler reference

ExecutionAttempt
  └─ provider API mechanics/errors/retries

PublishedMedia
  └─ externally observed successful result
```

This is the recommended conceptual model for the rest of the research.

---

# 25. Step 10 verdict

The original concept of a single publication manifest was directionally correct but should be refined into **separate control-plane and execution-plane records**.

The most important rules are:

1. `publish_ready` means content is eligible, not authorized to post.
2. Final caption/media/tags must be explicit before approval.
3. Approval binds to a fingerprint of one immutable publication version.
4. Schedule is separate from content intent.
5. Meta container IDs and retries belong only to `ExecutionAttempt`.
6. Instagram Media ID/permalink belong to `PublishedMedia`.
7. High Director operates on intent/approval/schedule, not low-level provider runtime state.
8. A changed approved field creates a new publication version and invalidates the old approval.
9. Provider IDs are secondary; Eirepolitic `publication_id` remains authoritative.
10. The physical database choice can be made later without changing this logical model.

---

## Repository references

Existing files reviewed for alignment:

- `process/instagram_build_copy_pack.py`
- `process/instagram_build_publish_queue.py`

The future model should preserve their useful principles:

- deterministic explicit caption/alt-text values;
- explicit review state;
- blocking safety notes;
- no direct publication from the generation step.

---

## Confidence / unresolved items

**High confidence:**

- intent and execution state should be separate;
- approved publication versions should be immutable;
- execution attempts need their own records;
- provider IDs should not be Eirepolitic's primary identity;
- existing generation review status must be separate from publication authorization.

**Still to design:**

- exact conversational state/approval transition rules;
- whether schedule-only changes require full reapproval;
- physical datastore/table/index design;
- canonical JSON/fingerprint rules;
- exact timezone normalization;
- idempotency keys/locking mechanics.

**Next research step:**

Step 11 will define the High Director conversational approval model and state machine, including what changes invalidate approval and how scheduling/cancellation/rescheduling should be confirmed safely.
