# Instagram Publishing Research Plan

Status: **Research only — no production publishing, credentials, account connection, or live posting permitted without explicit approval.**

Branch: `ops/investigate-instagram-publishing-20260903`

## Goal

Design a deterministic Instagram publishing system controlled conversationally by High Director / Overlord.

The intended separation is:

High Director / Overlord → publication intent and human approval → structured publication manifest → scheduler/publisher → Instagram API → publication ledger.

The language model must not act as the scheduler or long-running worker.

## Important account constraint

The current Eirepolitic Instagram account is believed to be a **personal account**.

Before recommending any API architecture that requires a professional Instagram account, research must verify from current Meta documentation:

- whether the existing personal account can be converted in-place to Creator or Business;
- whether existing followers are retained;
- whether existing posts/media are retained;
- whether the account can later switch between Creator, Business, and Personal;
- any feature, privacy, music/licensing, analytics, messaging, or account-management changes caused by conversion;
- whether Facebook Page linkage is required for the API path we would need;
- whether conversion creates any practical risk to the existing audience/account history.

Default preference: **preserve the existing account, followers, posts, handle, and history. Do not recommend creating a replacement account unless conversion is impossible or materially unsafe.**

## Working method

Research will be completed one step at a time.

For each step:

1. inspect the repo and/or authoritative documentation;
2. write detailed findings to a dedicated Markdown file in `docs/research/instagram_publishing/`;
3. update this plan's status table;
4. report only a very short summary in chat;
5. do not move into implementation.

Primary external sources should be Meta's current official developer/help documentation. Third-party vendor claims should be checked against their own current documentation/pricing pages.

## Research steps

| Step | Topic | Output file | Status |
|---|---|---|---|
| 1 | Existing repo review: Instagram pipeline, review gates, S3, AWS, secrets, scheduling hooks | `01_repo_review.md` | **complete** |
| 2 | Existing Instagram personal account → Creator/Business conversion and preservation risks | `02_account_conversion.md` | **complete** |
| 3 | Current Meta Instagram publishing API capabilities and restrictions | `03_meta_api_capabilities.md` | pending |
| 4 | Meta authentication, Page linkage, permissions, App Review, verification, token lifecycle | `04_meta_authentication.md` | pending |
| 5 | Tagging and metadata: mentions, media tags, collaborators, location, alt text, first comment, product tags | `05_instagram_tagging_metadata.md` | pending |
| 6 | Direct Meta publishing architecture | `06_direct_meta_option.md` | pending |
| 7 | Third-party scheduler/API options: Buffer, Hootsuite, Metricool, Later, Sprout and suitable alternatives | `07_third_party_options.md` | pending |
| 8 | Hybrid architecture: own intent/ledger + third-party delivery | `08_hybrid_option.md` | pending |
| 9 | Scheduling infrastructure comparison: EventBridge Scheduler, Lambda, Step Functions, SQS, GitHub Actions, Power Automate | `09_scheduling_infrastructure.md` | pending |
| 10 | Publication data model: asset package, request, approval, schedule, execution attempt, published-media record | `10_publication_data_model.md` | pending |
| 11 | Conversational approval/state model and safeguards | `11_conversational_control.md` | pending |
| 12 | Asset readiness, S3/media hosting, hashes, stable URLs and Meta retrieval requirements | `12_asset_readiness.md` | pending |
| 13 | Captions/templates and explicit final-caption storage | `13_caption_templates.md` | pending |
| 14 | Timezones and Europe/Dublin scheduling/DST handling | `14_timezones.md` | pending |
| 15 | Secrets/token management | `15_secrets_tokens.md` | pending |
| 16 | Idempotency, failure recovery and retries | `16_idempotency_failures.md` | pending |
| 17 | Monitoring, auditability and operator queries | `17_monitoring.md` | pending |
| 18 | Multi-platform extension approach | `18_multi_platform.md` | pending |
| 19 | Options comparison, estimated cost, recommendation and phased implementation proposal | `19_recommendation.md` | pending |

## Decision gates

No implementation should begin as part of this research branch.

After Step 19, the recommendation must be presented for explicit approval before any of the following occur:

- changing the Instagram account type;
- creating/configuring a Meta App;
- connecting a Facebook Page or live Instagram account;
- creating Meta credentials/tokens;
- provisioning production publishing infrastructure;
- enabling scheduling;
- publishing a test or production post.
