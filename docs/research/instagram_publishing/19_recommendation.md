# Step 19 — Final Recommendation, Cost, Risk and Phased Implementation Proposal

Status: **complete**

Research date: 2026-09-04

Scope: synthesize Steps 1–18 into a final architecture recommendation for conversationally controlled Instagram publishing. This is the end of the research phase only.

**No account type was changed, no Meta App was created, no Facebook Page or Instagram account was connected, no credentials were created, no infrastructure was provisioned, no scheduler was enabled, and no post was published.**

---

# Executive recommendation

## Recommend: Direct Meta publishing with an Eirepolitic-owned control plane and AWS execution layer

Recommended architecture:

```text
High Director / Overlord
        ↓
Publication control service
        ↓
DynamoDB publication ledger
        ↓
EventBridge Scheduler
        ↓
Lambda deterministic publisher
        ↓
Meta Instagram API with Facebook Login
        ↓
Instagram
```

Supporting services:

```text
S3              immutable approved assets
Secrets Manager Meta runtime credential
SQS             scheduler DLQ
CloudWatch       metrics/logs/alarms
SNS/User Notifications  operator alerts
```

Recommended Instagram/account route:

```text
existing Eirepolitic Instagram account
        ↓ convert in place, if approved
Professional Business account
        ↓
linked Facebook Page
        ↓
Meta Business App
        ↓
Instagram API with Facebook Login
```

The existing Instagram account should be preserved rather than replaced.

### Why this is the preferred option

Direct Meta gives Eirepolitic:

- the strongest match to the desired High Director conversational model;
- direct control of exact approved caption/media/tags/account/time;
- the richest current Instagram metadata surface among the researched options;
- current direct evidence for user tags, collaborators, location, alt text and comments/first-comment operations;
- private canonical S3 asset storage with publication-time retrieval URLs;
- direct reconciliation of Meta containers and published state;
- lower third-party lock-in;
- a clean future provider-adapter boundary;
- negligible scheduling/runtime cost at current publication volume.

The main disadvantage is that Eirepolitic must own Meta authentication lifecycle, provider API maintenance and failure recovery.

The research shows that those responsibilities are manageable with a small serverless architecture and are outweighed by the control gained.

---

# 1. Final option ranking

| Rank | Option | Verdict |
|---|---|---|
| **1** | **Direct Meta + Eirepolitic control plane + AWS scheduler/runtime** | **Recommended** |
| **2** | **Eirepolitic control plane + Buffer scheduled delivery** | Good fallback / optional future adapter |
| 3 | Metricool API | Technically viable but too expensive/uncertain for current scale |
| 4 | Hootsuite | Not enough verified current self-service publishing API clarity for this use |
| 5 | Later | No suitable public customer publishing API identified |
| 6 | Sprout Social | Current Publishing API create path is draft-only and API tier is very expensive |

---

# 2. Architecture decision matrix

Scores: 1 = weak, 5 = excellent.

| Dimension | Direct Meta | Buffer hybrid |
|---|---:|---:|
| Conversational control fit | **5** | **5** |
| Exact human approval/audit model | **5** | **5** |
| Native Instagram feature access | **5** | 3 |
| Media user tags | **5** | 4 |
| Collaborators | **5** | **2** current API gap |
| Location | **5** | 4 |
| Alt text | **5** | 4 |
| First-comment architecture | **4** | 2 current API bug |
| Private unpublished assets | **5** | **2** |
| Scheduling simplicity | 4 | **5** |
| Meta token simplicity | 3 | **5** |
| Failure transparency/control | **5** | 3 |
| Idempotency/reconciliation control | **5** | 4 |
| Vendor lock-in | **4** | 3 |
| Operator social-calendar UI | 2 | **5** |
| Initial engineering effort | 3 | **4** |
| Runtime cost | **5** | 4 |
| Future multi-platform foundation | 4 | **5** |
| Ability to switch delivery provider later | **5** | 4 if Eirepolitic ledger remains authoritative |

### Overall

Direct Meta wins because the user's priority is **conversationally precise control over Instagram publication**, not merely obtaining a social scheduling UI.

Buffer's primary advantage is implementation convenience, but Step 9 established that the part Buffer replaces most obviously—the future publication clock—is only a small EventBridge + Lambda system.

---

# 3. Why direct Meta is preferred over Buffer

## Direct Meta advantages

### A. Instagram feature control

The current Meta Graph API surface provides stronger evidence for:

- image/media tags;
- collaborators;
- location;
- alt text;
- comments/first-comment operations;
- single image;
- carousel;
- Reel;
- Story subject to account/API restrictions.

Buffer's current Instagram metadata input exposes location and first comment but does not expose a collaborator field.

Its current API roadmap also still reports an Instagram `firstComment` persistence issue.

Sources:

- Meta current Instagram API: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Buffer Instagram metadata input: https://developers.buffer.com/types/InstagramPostMetadataInput.html
- Buffer API roadmap: https://developers.buffer.com/roadmap.html

### B. Private asset model

Direct Meta allows this architecture:

```text
private immutable S3 asset
       ↓
publication-time temporary retrieval URL
       ↓
Meta retrieves asset
```

Buffer currently requires scheduled-media URLs to remain publicly reachable until the future publication time and warns against expiring S3 presigned URLs.

That is a meaningful disadvantage for unpublished political/editorial content.

### C. Fewer stateful external systems

Direct:

```text
Eirepolitic ledger
→ AWS runtime
→ Meta
```

Buffer:

```text
Eirepolitic ledger
→ Buffer scheduled job
→ Meta
```

The latter requires an extra synchronization/reconciliation layer.

### D. Own failure recovery

Meta exposes container state such as `PUBLISHED`, allowing Eirepolitic to prevent duplicate posts after uncertain `/media_publish` responses.

Using Buffer introduces another abstraction around the final platform state.

### E. Low AWS scheduling complexity

Direct scheduling is only:

```text
one EventBridge schedule per publication
→ one Lambda publisher
→ one SQS DLQ
```

It does not require a server, always-running process or expensive workflow engine.

---

# 4. Reasons Buffer remains a valuable fallback

Buffer should not be discarded entirely.

A `BufferInstagramPublisher` adapter remains useful if later:

- Meta authentication becomes disproportionately difficult to maintain;
- Eirepolitic wants many social networks quickly;
- Buffer adds collaborator support and resolves first-comment issues;
- an operator social calendar becomes more valuable than direct control;
- direct Meta policy/API changes materially increase maintenance burden.

The core Eirepolitic publication model is deliberately independent of Meta so this migration remains possible.

---

# 5. Account recommendation

## Preserve the existing Instagram account

Research found the existing personal account can be converted in place to a Professional account rather than creating a replacement account.

The objective is to preserve:

- followers;
- posts/media;
- handle;
- messages/history;
- audience continuity.

### Important conversion consequence

A Professional account must be public.

If the existing account is private, pending follower requests must be reviewed before conversion because the account becomes public and Meta guidance indicates pending requests may be accepted during the transition.

### Recommended professional type: Business, subject to explicit approval

For Eirepolitic, **Business** is the preferred initial professional-account type because:

- Eirepolitic is an organizational/project presence rather than an individual creator identity;
- the Page-linked Facebook Login publishing route is the preferred API path;
- current Facebook Login documentation limits Stories publishing to Business accounts;
- it leaves the broadest current direct-publishing route available.

Trade-off:

- Business accounts can have different licensed-music availability than personal/creator accounts.

If music-heavy Reels become a major requirement, this should be reviewed again before conversion.

No conversion should occur until explicitly approved.

---

# 6. Preferred Meta authentication route

Recommend:

```text
Instagram API with Facebook Login
```

rather than Instagram Login for the first direct implementation.

Reason:

The desired feature set includes tagging/collaboration-related functionality, and Meta explicitly says the Instagram Login setup does not provide tagging access.

Required relationship:

```text
Professional Instagram account
↕
linked Facebook Page
↕
Meta App / Facebook authorization
↕
Page Access Token
```

Initial Eirepolitic-only proof should investigate **Standard Access** first because the application serves an account owned/managed by the app owner.

Do not assume Advanced Access/App Review is required until the exact Developer Dashboard configuration proves it.

---

# 7. Recommended control-plane records

Use the logical model from Step 10:

```text
AssetPackage
PublicationRequest
PublicationApproval
PublicationSchedule
ExecutionAttempt
PublishedMedia
```

### Critical rule

Existing repo fields:

```text
publish_ready
review_status
```

remain **content readiness** gates.

They are not permission to publish.

Actual live publication requires a separate explicit human `PublicationApproval` bound to an immutable publication fingerprint.

---

# 8. Recommended physical publication ledger: DynamoDB on-demand

Step 10 intentionally deferred the physical database choice.

After the full architecture review, **DynamoDB on-demand is the best fit for v1**.

Reasons:

- serverless;
- no database server/cluster;
- very low usage cost;
- atomic conditional writes suitable for idempotency locks;
- simple Lambda integration;
- straightforward query/index patterns for current publishing volumes;
- no need to manage connection pools/schema migrations for a tiny control-plane workload.

The logical records should remain platform-neutral even if physically stored in one DynamoDB table.

### Do not over-engineer the table initially

Required access patterns are modest:

- get publication by ID;
- list scheduled publications by date/account;
- list published media by date/account;
- find current publication by project/period;
- load attempts by publication;
- list `needs_attention`/`auth_blocked` items.

A small single-table design or small number of tables is sufficient.

---

# 9. Recommended scheduler/runtime

Use:

```text
EventBridge Scheduler
→ Lambda publisher
```

with:

```text
SQS standard DLQ
```

### Scheduler payload

Prefer:

```json
{
  "publication_id": "pub_...",
  "expected_version": 3
}
```

Do not store captions, tokens or temporary asset URLs in the schedule.

### Timing

Store:

```text
Europe/Dublin local time
+
IANA timezone
+
resolved UTC instant
```

and execute the frozen approved UTC instant.

EventBridge flexible delivery windows should be off.

---

# 10. Recommended asset architecture

Current repo renderer output:

```text
PNG previews
```

Current Meta Content Publishing image requirement:

```text
JPEG
```

Therefore add a future deterministic asset-finalization stage:

```text
reviewed PNG
      ↓
platform finalizer
      ↓
Instagram-compatible JPEG
      ↓
SHA-256 + dimensions + MIME + ordered package
      ↓
private immutable S3 path
```

Never publish from:

```text
instagram/previews/.../latest/
```

and never use temporary GitHub Actions artifact URLs as canonical production assets.

Recommended S3 path pattern:

```text
instagram/approved/<project>/<period>/<asset_package_id>/media/01.jpg
```

Use immutable keys and optional S3 Versioning as defence in depth.

---

# 11. Publication-time media delivery

For direct Meta, preferred candidate:

```text
private S3 object
→ Lambda creates short-lived GET URL at execution
→ Meta cURLs it immediately
```

The exact S3 presigned-URL pattern must be tested with a Meta canary because Meta specifies public reachability but does not explicitly certify S3 presigned URLs in the reviewed guide.

Never log the full temporary URL.

---

# 12. Caption architecture

Move reusable defaults out of hard-coded Python into future versioned template/configuration data.

Example future structure:

```text
instagram/caption_templates/member_profile.yml
instagram/caption_templates/party_speech_breakdown.yml
```

Templates provide drafting defaults only.

High Director may conversationally edit the copy, but before approval the system stores:

```text
one exact complete final caption
```

The publisher sends that exact string unchanged.

No LLM or template rendering occurs at execution time.

---

# 13. Conversational approval model

High Director may freely edit `draft` publications.

Before scheduling/publishing, show a compact exact confirmation containing:

- account;
- post/project/period;
- ordered media/count;
- exact caption;
- hashtags/mentions;
- media tags;
- collaborators;
- location;
- alt text;
- first comment;
- local date/time/timezone;
- resolved UTC time;
- immediate vs scheduled;
- notification behaviour.

A human must explicitly approve the displayed version.

### Material edit rule

Changing any of these after approval:

```text
account
asset/media
caption
hashtags/mentions
alt text
media tags
collaborators
location
first comment
post type
```

creates a new publication version and invalidates the old approval.

### Schedule-only change

A time/date change can use a narrower explicit confirmation without forcing a complete reapproval of unchanged content.

---

# 14. Idempotency / duplicate-post prevention

This is the most important runtime safety requirement.

Use:

```text
publication_id + publication_version
```

as the root idempotency identity.

Before Meta side effects:

```text
atomic conditional execution claim
```

must ensure only one worker proceeds.

Persist every Meta child/parent container ID immediately.

### Critical `/media_publish` rule

If the publish HTTP response is lost:

```text
DO NOT create another publication
```

Query the existing container.

If Meta reports:

```text
PUBLISHED
```

the system permanently blocks republishing that publication version even if the Instagram Media ID still needs reconciliation.

This protects against visible duplicates under at-least-once AWS delivery and network uncertainty.

---

# 15. Secrets/authentication recommendation

Use:

```text
AWS Secrets Manager
```

for production Meta credentials.

Publisher Lambda reads the exact secret through least-privilege IAM.

High Director and publication records see only:

```text
account_ref
credential_ref
auth status/expiry metadata
```

not token values.

### GitHub AWS access

Current repo workflows use static:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

Future publishing/deployment work should migrate to:

```text
GitHub OIDC → restricted AWS IAM role → short-lived credentials
```

Do not place the Meta publishing token in GitHub Actions.

---

# 16. Monitoring recommendation

Use:

```text
Publication ledger = business truth
CloudWatch = infrastructure telemetry
```

Initial alerts:

- scheduler target error;
- dropped invocation;
- DLQ message;
- Lambda error/throttle;
- `PublicationNeedsAttention`;
- `PublicationOutcomeUncertain`;
- `InstagramAuthBlocked`.

Send operator notifications through a simple SNS/AWS notification/email path initially.

High Director should answer publication-history/status questions by querying the ledger, not by parsing CloudWatch logs.

---

# 17. Multi-platform extension

Keep separate concepts:

```text
platform = instagram
provider = meta_direct
```

A future delivery may instead use:

```text
platform = instagram
provider = buffer
```

without changing the human/publication identity.

For multi-platform distribution, create independent child publications under an optional `distribution_id`.

Each child gets its own:

- caption;
- platform derivative assets;
- approval fingerprint;
- schedule;
- execution attempts;
- provider result.

Build only Instagram initially.

---

# 18. Cost estimate — direct Meta architecture

## Meta

No separate per-post Content Publishing API fee was identified in the current official Meta documentation reviewed during this research.

This should be rechecked if Meta introduces pricing/policy changes before implementation.

## AWS expected low-volume monthly cost

Assumption:

```text
1 Instagram account
10–100 posts/month
small static-image/carousel assets
low operator query volume
```

### EventBridge Scheduler

Current free allowance:

```text
14 million Scheduler invocations/month
```

Eirepolitic usage would be dozens, so expected Scheduler cost:

```text
~$0
```

Source:

- https://aws.amazon.com/eventbridge/pricing/

### Lambda

Current free allowance includes:

```text
1 million requests/month
400,000 GB-seconds/month
```

Expected publication/reconciliation usage is far below this.

Expected:

```text
~$0
```

Source:

- https://aws.amazon.com/lambda/pricing/

### SQS

Current free allowance:

```text
1 million requests/month
```

DLQ traffic should normally be near zero.

Expected:

```text
~$0
```

Source:

- https://aws.amazon.com/sqs/pricing/

### DynamoDB on-demand

Current documented example rates for Standard on-demand are approximately:

```text
$0.625 per million writes
$0.125 per million reads
first 25 GB storage included in AWS Free Tier
```

At publication-ledger volume the request cost should be fractions of a cent to pennies.

Source:

- https://aws.amazon.com/dynamodb/pricing/

### Secrets Manager

Current pricing:

```text
$0.40 per secret/month
$0.05 per 10,000 API calls
```

Expected:

```text
$0.40–$1.20/month
```

depending on whether one runtime token or several separated credential values are retained.

Source:

- https://aws.amazon.com/secrets-manager/pricing/

### CloudWatch

Current free tier includes:

- 5 GB Logs usage;
- 10 standard-resolution alarm metrics;
- several other basic metrics/dashboard allowances.

At this volume Eirepolitic should usually stay inside/near those allowances unless verbose logging is enabled.

Source:

- https://aws.amazon.com/cloudwatch/pricing/

### S3

Eirepolitic already uses S3.

Additional approved JPEG assets will add some storage/GET/PUT cost, but static Instagram image volume should remain pennies/month unless asset volume grows dramatically.

### SNS/email alerts

Current SNS allowances include 1 million requests and 1,000 email deliveries/month free.

Expected operational alert cost:

```text
~$0
```

Source:

- https://aws.amazon.com/sns/faqs/

## Expected direct-Meta total

At initial Eirepolitic scale:

```text
approximately $0.50–$2/month incremental AWS cost
```

A conservative planning ceiling is:

```text
< $5/month
```

excluding:

- existing general S3 costs;
- taxes/VAT;
- unexpected heavy CloudWatch logging;
- future video processing/storage;
- paid AWS support;
- future multi-platform traffic.

The most predictable non-zero line item is Secrets Manager.

---

# 19. Cost estimate — Buffer hybrid

Current Buffer pricing:

```text
Free:
$0
3 channels
10 scheduled posts/channel
API access

Essentials:
$5/month per channel
$60 billed yearly
unlimited scheduled posts
API access
```

Source:

- https://buffer.com/pricing

Eirepolitic would still need:

- publication ledger;
- approval logic;
- reconciliation;
- asset finalization/storage;
- High Director integration;
- secret storage for Buffer API token;
- monitoring.

Therefore expected practical cost for one Instagram channel is approximately:

```text
Buffer Essentials $5/month
+
small AWS control-plane cost
≈ $5.50–$7/month
```

The Free plan might be enough for experimentation but its 10-scheduled-post/channel limit is restrictive for production planning.

### Cost is not the deciding factor

Both direct Meta and Buffer are cheap at Eirepolitic scale.

The decision is primarily about:

```text
control + feature coverage + asset privacy + vendor dependency
```

not monthly spend.

---

# 20. Other vendor cost/fit summary

| Vendor | Current relevant entry cost | API delivery fit | Conclusion |
|---|---:|---|---|
| Buffer | $0 free / $5 per channel Essentials yearly billing | Strong | Keep as fallback |
| Metricool | ~€54/month Advanced | Viable API | Too expensive for current need |
| Hootsuite | ~US$99+/month | Publishing API entitlement insufficiently verified | Do not choose without vendor confirmation |
| Later | $18.75+/month | No suitable public publishing API identified | UI scheduler, not backend architecture |
| Sprout Social | $399/seat/month Advanced | API create currently draft-only | Exclude |

Prices can change and must be rechecked immediately before purchasing a subscription.

---

# 21. Complexity comparison

## Direct Meta

### Initial complexity: medium

Requires:

- account conversion/Page link;
- Meta App/auth proof;
- final asset conversion;
- ledger;
- scheduler;
- Lambda publisher;
- idempotency/reconciliation;
- monitoring.

### Ongoing complexity: medium

Main maintenance:

- Meta token health;
- Graph API version changes;
- platform capability changes;
- occasional provider errors.

## Buffer

### Initial complexity: low-medium

Removes:

- direct Meta publishing/container implementation;
- publication timer;
- much Meta-token lifecycle work.

Still requires:

- ledger;
- approval;
- external-provider idempotency;
- reconciliation;
- media hosting;
- monitoring.

### Ongoing complexity: low-medium

But adds:

- Buffer API/version dependency;
- Buffer/Meta double-layer failures;
- API feature lag/bugs;
- stable public media URL requirement.

---

# 22. Risk comparison

| Risk | Direct Meta | Buffer hybrid |
|---|---|---|
| Accidental unapproved publish | Low with proposed approval gate | Low with same Eirepolitic gate |
| Duplicate post on retry | Medium engineering risk; controllable with container reconciliation | Medium distributed-state risk |
| Token/auth outage | Eirepolitic manages directly | Buffer manages connection, but disconnections still happen |
| Asset leak before publication | **Low** with private S3 + execution URL | **Higher** due stable reachable URLs |
| Provider feature mismatch | Lower; native API | Higher; vendor abstraction may lag |
| Vendor pricing/API change | Meta platform dependency only | Meta + Buffer |
| Operational UI absence | Higher; ledger/High Director instead | Low; Buffer UI available |
| Provider-state drift | Lower | Higher |
| API maintenance burden | Higher | Lower |
| Long-term portability | **Higher** | Good only if Eirepolitic remains source of truth |

---

# 23. Lock-in assessment

## Direct Meta

Lock-in:

```text
Meta platform API
```

which is unavoidable for publishing to Instagram unless another vendor sits in front of Meta.

AWS components are conventional serverless primitives and the logical publication model remains portable.

### Rating: low-medium

## Buffer

Lock-in:

```text
Buffer scheduling schema/API
+
Buffer social-account connection
+
Meta behind Buffer
```

The adapter model limits this, but operational dependency remains.

### Rating: medium

---

# 24. Conversational fit

Direct Meta strongly matches the original desired interaction:

```text
User:
"Schedule the August breakdown for Tuesday at 7:30pm, mention @example in the caption, tag them on slide 3, use the normal hashtags, and add no first comment."

High Director:
resolve approved assets
→ resolve deterministic fields
→ show exact final publication
→ obtain human approval
→ write immutable request/approval/schedule
→ EventBridge executes later
→ Lambda publishes exactly that version
```

High Director is the conversational **control surface**, not the background scheduler.

That separation remains one of the strongest architectural decisions from the research.

---

# 25. Recommended implementation phases

Implementation must not begin until the user explicitly approves the recommendation.

## Phase 0 — Explicit architecture approval

User approves or changes:

- direct Meta vs Buffer;
- existing-account conversion;
- Business vs Creator;
- Page linkage;
- AWS serverless architecture;
- rough cost envelope.

No technical changes before this gate.

---

## Phase 1 — Account/API feasibility proof

Goal: prove the Meta route before building production publishing infrastructure.

Actions after explicit approval:

1. review current Instagram privacy/pending follower requests;
2. convert the existing account in place to Professional Business;
3. create/link the appropriate Facebook Page;
4. create/configure the Meta Business App;
5. obtain only the minimum required permissions/token for the owned account;
6. record the exact token type, scopes and expiry properties;
7. prove account/Page/Instagram identity read access;
8. test platform capability metadata in a controlled way;
9. test Meta retrieval of a temporary S3 canary asset by creating a **non-published** container where possible;
10. allow that container to expire rather than calling `/media_publish` unless a separate test-post approval is given.

Decision gate:

```text
Do not proceed if direct Meta cannot reliably provide the required tagging/collaboration/asset/auth capabilities.
```

If it fails materially, reconsider Buffer.

---

## Phase 2 — Security and asset foundation

Implement without enabling live publication:

- GitHub OIDC → AWS IAM deployment role;
- Secrets Manager secret references;
- immutable approved S3 asset prefix;
- deterministic PNG → JPEG finalizer;
- SHA-256/dimensions/MIME checks;
- versioned caption-template framework;
- account/provider capability configuration.

Default:

```text
publishing_enabled = false
```

---

## Phase 3 — Publication control plane

Implement:

- DynamoDB publication ledger;
- AssetPackage records;
- PublicationRequest/versioning;
- PublicationApproval fingerprints;
- PublicationSchedule records;
- ExecutionAttempt/operation records;
- PublishedMedia records;
- High Director query/actions interface;
- validation/state machine.

Still keep live provider publication disabled.

Test using synthetic/dry-run provider responses.

---

## Phase 4 — Scheduler and deterministic publisher dry run

Implement:

- EventBridge Scheduler adapter;
- Lambda publisher;
- execution claim/lease;
- SQS DLQ;
- Meta client;
- container status reconciliation;
- CloudWatch/SNS alerts;
- no-publication/dry-run mode.

Test:

- duplicate scheduler invocation;
- Lambda crash/resume;
- expired lease;
- missing asset;
- wrong hash;
- auth failure;
- rate limit simulation;
- uncertain publish simulation;
- carousel partial failure;
- first-comment failure path.

---

## Phase 5 — Explicit Meta canary publication

This is a separate decision gate.

Before the first visible test post, show exactly:

- account;
- media;
- caption;
- tags;
- collaborators;
- location;
- alt text;
- first comment;
- immediate/scheduled time.

Require explicit approval.

Use one low-risk controlled post and verify:

- Meta media retrieval;
- carousel/media order;
- tags/collaborators/location;
- caption/alt text;
- returned Media ID/permalink;
- ledger reconciliation;
- monitoring;
- no duplicate under replay tests.

Do not immediately enable unattended production scheduling after one successful API call.

---

## Phase 6 — Limited production enablement

Begin with:

- static image/carousel only;
- one Instagram account;
- human approval required for every publication;
- conservative retry policy;
- operator alert on all uncertain/failure states;
- no automatic destructive delete;
- no Reels/Stories until static publishing is stable.

After several successful posts, review operational evidence before expanding.

---

## Phase 7 — Optional enhancements

Only after stable Instagram v1:

- Reels;
- Stories;
- first-comment automation if proven;
- richer collaborator workflows;
- operator dashboard;
- automatic auth-health checks/renewal;
- additional social platform adapters;
- Buffer fallback adapter if desired.

---

# 26. Explicit go/no-go checkpoints

## Gate A — Account conversion

Before changing the existing Instagram account:

```text
explicit user approval required
```

## Gate B — Meta App/Page/account connection

Before creating/configuring Meta objects or linking live account:

```text
explicit user approval required
```

## Gate C — Credential creation/storage

Before generating/storing live Meta credentials:

```text
explicit user approval required
```

## Gate D — AWS production infrastructure

Before provisioning production DynamoDB/EventBridge/Lambda/Secrets/SQS/alerts:

```text
explicit user approval required
```

## Gate E — First visible test post

Before calling Meta `/media_publish` against the live Instagram account:

```text
explicit user approval of the exact test publication required
```

## Gate F — Production scheduling enablement

After test results:

```text
explicit user approval required
```

---

# 27. What should NOT be built

Avoid:

- an LLM process that sleeps until publication time;
- GitHub Actions cron as the authoritative publication clock;
- a single mutable CSV queue as production state;
- publishing directly from `latest/` S3 previews;
- storing full captions inside EventBridge payloads;
- storing Meta tokens in GitHub;
- regenerating captions at execution time;
- creating Meta containers days before publication;
- blind retries after uncertain `/media_publish`;
- a universal multi-platform schema containing every possible social field;
- a third-party scheduler as the authoritative publication database;
- automatic live publication merely because `publish_ready=yes`.

---

# 28. Expected future repo areas

Research recommendation only; not created yet.

A likely implementation may eventually introduce areas similar to:

```text
instagram/
  caption_templates/
  publication_policies/

process/
  instagram_finalize_asset_package.py

publishing/
  models/
  control/
  publishers/
    instagram_meta.py
  scheduling/
    eventbridge.py
  storage/
    ledger.py
  monitoring/

infra/
  publishing/
```

Exact file layout should be planned once implementation is approved.

Do not create parallel generation logic; consume existing reviewed post-generation outputs.

---

# 29. Final recommendation statement

## Recommended target architecture

```text
Existing Instagram generation/review pipeline
       ↓
Immutable approved JPEG AssetPackage in private S3
       ↓
High Director creates exact PublicationRequest
       ↓
Human approves immutable fingerprint
       ↓
DynamoDB publication ledger
       ↓
EventBridge one-time schedule
       ↓
Lambda deterministic Meta publisher
       ↓
Meta Instagram API with Facebook Login
       ↓
Existing Eirepolitic Instagram account converted in place to Business Professional
       ↓
PublishedMedia result reconciled to ledger
```

With:

```text
Secrets Manager  credentials
SQS              DLQ
CloudWatch/SNS   monitoring
GitHub OIDC      AWS deployment authentication
```

### Primary recommendation

**Build direct Meta first.**

### Secondary/fallback recommendation

Keep **Buffer** behind a future provider adapter, but do not make it the initial source of scheduling/delivery truth.

### Initial product scope

Start with:

```text
one Instagram account
static image + carousel
manual human approval for every publication
scheduled + immediate publishing
caption/hashtags/mentions
media tags
collaborator/location capability after canary verification
alt text
optional first comment after canary verification
cancel/reschedule
status/history queries
```

Defer:

```text
Reels/Stories
automatic recurring publication
multi-platform production
published caption editing
automatic destructive deletion
advanced analytics
```

until the core system is proven.

---

# 30. Research completion

Steps 1–19 are now complete.

Detailed research files:

```text
docs/research/instagram_publishing/
  01_repo_review.md
  02_account_conversion.md
  03_meta_api_capabilities.md
  04_meta_authentication.md
  05_instagram_tagging_metadata.md
  06_direct_meta_option.md
  07_third_party_options.md
  08_hybrid_option.md
  09_scheduling_infrastructure.md
  10_publication_data_model.md
  11_conversational_control.md
  12_asset_readiness.md
  13_caption_templates.md
  14_timezones.md
  15_secrets_tokens.md
  16_idempotency_failures.md
  17_monitoring.md
  18_multi_platform.md
  19_recommendation.md
```

The next action is **not implementation**.

The next action is for the user to review/approve/change the final recommendation and decide whether to begin Phase 0/Phase 1.

---

## Current pricing/reference sources checked on 2026-09-04

### Meta

- https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login

### Buffer

- https://buffer.com/pricing
- https://developers.buffer.com/types/InstagramPostMetadataInput.html
- https://developers.buffer.com/roadmap.html

### AWS

- EventBridge: https://aws.amazon.com/eventbridge/pricing/
- Lambda: https://aws.amazon.com/lambda/pricing/
- SQS: https://aws.amazon.com/sqs/pricing/
- DynamoDB: https://aws.amazon.com/dynamodb/pricing/
- Secrets Manager: https://aws.amazon.com/secrets-manager/pricing/
- CloudWatch: https://aws.amazon.com/cloudwatch/pricing/
- S3: https://aws.amazon.com/s3/pricing/
- SNS: https://aws.amazon.com/sns/faqs/

Prices/free allowances vary by region, account/free-tier eligibility and future vendor changes. Recheck immediately before provisioning or purchasing.
