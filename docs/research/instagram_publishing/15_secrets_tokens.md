# Step 15 — Secrets and Meta Token Management

Status: **complete**

Research date: 2026-09-04

Scope: define where Instagram/Meta credentials should live, which systems may access them, how GitHub/AWS authentication should work, how token health/renewal should be handled, and how logs must avoid leaking credentials.

No secret, Meta token, IAM role, OIDC provider, GitHub secret, account connection, or production publishing credential was created or changed.

---

## Short conclusion

For the direct-Meta architecture:

```text
High Director
    ↓ account_ref only
Publication records
    ↓ credential_ref only
Lambda publisher IAM role
    ↓ secretsmanager:GetSecretValue
AWS Secrets Manager
    ↓ Meta runtime credential
Meta API
```

The **Meta access token must not live in GitHub, publication manifests, DynamoDB/ledger records, environment files, logs, or conversation history**.

Store the production Meta runtime credential in **AWS Secrets Manager** and grant only the publication runtime role permission to read the specific secret.

GitHub Actions should **not** receive the Meta publishing token. GitHub should deploy AWS resources/code using **GitHub OIDC → short-lived AWS credentials**, rather than extending the repository's current long-lived `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` pattern.

Meta token lifetime/renewal should remain an explicit runtime/setup concern rather than relying on a hard-coded assumption such as "permanent token" or "always 60 days." The selected credential's actual expiry/scopes must be recorded when the future Meta authentication proof is performed.

---

# 1. Existing repo credential pattern

The current manual Instagram render workflow:

```text
.github/workflows/instagram_campaign_render.yml
```

currently injects:

```yaml
AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
AWS_REGION: ${{ secrets.AWS_REGION }}
```

This is sufficient for the current preview workflow, but it is a **long-lived static AWS credential pattern**.

### Recommendation

Do not extend this pattern into the publication system.

Future AWS-accessing GitHub workflows should prefer:

```text
GitHub Actions OIDC token
      ↓
AWS IAM role assumption
      ↓
short-lived AWS session credentials
```

GitHub's current official AWS OIDC guidance explicitly says OIDC allows workflows to access AWS without storing long-lived AWS credentials as GitHub secrets.

Source:

- GitHub OIDC with AWS: https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws

---

# 2. GitHub OIDC requirements

A future deployment workflow would conceptually require:

```yaml
permissions:
  id-token: write
  contents: read
```

and an AWS IAM trust policy restricted to the correct repository/branch/environment through the OIDC `sub` claim or equivalent conditions.

GitHub and AWS both recommend restricting the trust relationship so untrusted repositories/workflows cannot assume the role.

### Important separation

The GitHub deployment role should have permissions such as:

```text
update Lambda code
update infrastructure
write approved deployment resources
```

as required by the chosen deployment design.

It should **not automatically receive**:

```text
secretsmanager:GetSecretValue
```

for the production Meta token.

Deployment code generally needs to know the secret's ARN/name, not its value.

---

# 3. Recommended production secret store

Use **AWS Secrets Manager** for the Meta runtime credential.

AWS describes Secrets Manager as suitable for:

- OAuth tokens;
- API keys;
- application credentials;
- lifecycle/rotation management.

AWS encrypts secrets at rest with KMS and transmits retrieved secret material over TLS.

Sources:

- AWS Secrets Manager overview: https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html
- AWS Secrets Manager best practices: https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html

### Why Secrets Manager instead of GitHub Secrets

The publisher runs in AWS, not in GitHub Actions.

Using Secrets Manager means:

```text
GitHub never needs the live social-platform credential
```

and Lambda can retrieve the credential at runtime through its AWS identity.

This creates a much cleaner security boundary.

---

# 4. What is actually secret

## Secret

Treat these as secret values where applicable:

```text
Meta Page access token
Meta User access token, if retained for renewal/bootstrap
Meta app secret, if the selected flow/runtime requires it
Buffer API token, if Buffer is selected instead
any future platform OAuth refresh/access token
presigned S3 URLs while valid
```

## Configuration, not normally secret

These identifiers can normally live in application configuration/publication account records:

```text
Meta App ID
Instagram user/account ID
Facebook Page ID
account_ref = eirepolitic
Graph API version
AWS region
Secrets Manager secret ARN/name
```

Do not hide ordinary configuration in a secret merely because it relates to authentication.

For example, the current workflow's `AWS_REGION` does not need to be a secret; it can be normal configuration.

---

# 5. Recommended secret layout

Do not store one giant JSON object containing every production credential in the project.

Conceptual secret names:

```text
/eirepolitic/prod/instagram/meta-page-access-token
/eirepolitic/prod/instagram/meta-bootstrap-token   # only if truly needed
/eirepolitic/prod/instagram/meta-app-secret        # only if truly needed
```

or an equivalent environment-scoped naming convention.

### Why separation can help

Different roles may need different permissions.

For example:

```text
publisher Lambda
  → read page access token

credential-maintenance workflow
  → may need bootstrap/renewal credential

ordinary deployment role
  → reads neither value
```

The exact number of secrets should remain proportional to the final authentication flow.

Do not create unused secret entries merely for theoretical completeness.

---

# 6. Runtime IAM access

AWS recommends least-privilege access to Secrets Manager.

The publication Lambda should conceptually receive only:

```text
secretsmanager:GetSecretValue
```

for the exact required secret ARN(s).

If `DescribeSecret` is useful for metadata/health checks, grant it deliberately rather than granting broad Secrets Manager administration.

The publisher should not need permissions such as:

```text
CreateSecret
DeleteSecret
PutSecretValue
RotateSecret
List all secrets
```

for ordinary publication execution.

Sources:

- AWS Secrets Manager IAM policies: https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_iam-policies.html
- AWS Secrets Manager best practices: https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html

---

# 7. High Director must never receive the credential

High Director needs only deterministic account/configuration state such as:

```yaml
account_ref: eirepolitic
platform: instagram
auth_status: valid
credential_ref: instagram/meta-page-token
```

It does **not** need:

```text
access_token
app_secret
refresh_token
AWS secret value
```

If High Director is asked:

```text
"What is the Instagram token?"
```

the system should not retrieve/reveal it as part of normal conversational control.

This reduces accidental leakage through conversation transcripts and tools.

---

# 8. Publication records store references, not secret values

A publication/account configuration may contain:

```yaml
provider_auth:
  provider: meta
  credential_ref: instagram/meta-page-token
```

The worker maps that reference to the real Secrets Manager ARN through configuration.

Do not put any token value in:

- `PublicationRequest`;
- `PublicationApproval`;
- `PublicationSchedule`;
- `ExecutionAttempt`;
- `PublishedMedia`;
- scheduler payloads.

---

# 9. Current Meta token relationship

For the preferred Facebook Login/Page-linked route, Meta's current official Instagram documentation describes this relationship:

```text
Facebook User Access Token
        ↓
GET /me/accounts
        ↓
Facebook Page Access Token
        ↓
linked instagram_business_account
```

The Page Access Token acts on behalf of the Facebook Page linked to the Instagram professional account.

Current documented permissions include:

```text
pages_show_list
instagram_basic
instagram_content_publish
pages_read_engagement
instagram_manage_comments
```

Sources:

- Meta current Token folder: https://www.postman.com/meta/instagram/folder/i9oo1e6/token
- Meta current Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login

### Runtime implication

If this route is selected, the publisher should use the exact token type documented/verified for the publishing endpoint rather than storing a generic human token and assuming it is interchangeable.

---

# 10. User token versus production runtime token

The initial Meta setup may require a human-authorized User Access Token to discover/manage the Page relationship and obtain the Page token.

That does not mean the human token should automatically become the permanent publisher credential.

Recommended principle:

```text
bootstrap authorization credential
      ≠
normal publication runtime credential
```

If the User Access Token is not required after provisioning/renewal validation, do not retain it unnecessarily.

If it **is** required for renewal, keep it as a separate higher-sensitivity secret accessible only to the credential-maintenance path.

---

# 11. Token lifetime must not be guessed

Step 4 found that the current Meta publishing documentation clearly identifies token types and relationships but does not provide one universal current lifetime rule safe to hard-code across every route/token configuration.

Therefore do not design around assumptions such as:

```text
"the token never expires"
```

or:

```text
"all tokens expire every 60 days"
```

The future controlled authentication proof must record the actual credential properties returned/observable for the selected account/app.

---

# 12. Store token health metadata outside the secret

The token value belongs in Secrets Manager.

Non-secret operational metadata can live in an account/auth record:

```yaml
credential_ref: instagram/meta-page-token
credential_type: facebook_page_access_token
status: valid
issued_at: null
expires_at: null
last_validated_at: "..."
last_refreshed_at: null
required_permissions:
  - instagram_content_publish
permissions_last_checked_at: "..."
```

If Meta provides data-access expiry or other relevant lifecycle metadata during the future proof, record that too.

High Director can safely query this metadata to answer:

```text
"Is Instagram authentication healthy?"
```

without seeing the credential.

---

# 13. Authentication health check

Do not wait until a scheduled publication fails to discover that authentication is broken.

A future lightweight health check should validate, without publishing content, that the configured credential can still access the expected Page/Instagram identity and required read/publishing prerequisites.

Conceptually verify:

```text
credential is accepted by Meta
expected Facebook Page is returned/reachable
expected Instagram professional account remains linked
essential permissions still exist
account remains Professional
```

Do not create media containers merely as an authentication heartbeat.

---

# 14. Expiry/renewal behaviour

Recommended control behaviour:

```text
known expiry approaching
      ↓
credential status = expiring
      ↓
operator notification
      ↓
controlled refresh/reauthorization process
      ↓
validate replacement credential
      ↓
update Secrets Manager value
      ↓
credential status = valid
```

If Meta offers a safe programmatic renewal method for the exact selected token, it can later be automated.

If human reauthorization is required, the system should surface `needs_attention` before posts are due.

---

# 15. Do not assume Secrets Manager can automatically rotate Meta itself

AWS Secrets Manager supports managed and Lambda-based rotation mechanisms.

However, automatic Secrets Manager rotation only helps if the rotation code can legitimately obtain/update the corresponding credential at the external service.

Source:

- AWS rotation documentation: https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html

### Meta implication

Do **not** turn on a generic timed secret rotation job that merely replaces the stored string.

The external Meta token must first be renewed/reissued using Meta's supported authentication flow, then the validated new token can replace the old secret value.

Whether that renewal is automatic or human-assisted depends on the exact selected Meta credential discovered during the future authentication proof.

---

# 16. System-user token question remains conservative

Step 4 found Meta surfaces supporting system-user tokens in parts of the Instagram/Meta ecosystem, but the current Content Publishing path explicitly documents User/Page token flows.

Therefore do not build production security around an assumed permanent system-user token unless the future controlled proof confirms it works correctly for the exact Content Publishing endpoint and account configuration.

If validated, a system-user model could be attractive for unattended server-to-server operation because it may reduce dependence on a particular human authorization lifecycle.

But it is **not yet an architecture assumption**.

---

# 17. Secrets Manager caching in Lambda

AWS supports retrieving/caching Secrets Manager values from Lambda through:

- the AWS Parameters and Secrets Lambda Extension;
- AWS Powertools parameters utilities;
- standard SDK calls.

AWS specifically recommends the extension/Powertools approaches for caching and reducing repeated API calls.

Source:

- AWS Lambda Secrets Manager integration: https://docs.aws.amazon.com/lambda/latest/dg/with-secrets-manager.html

### Eirepolitic scale

At very low publication volume, performance/cost differences are tiny.

Use the simplest supported runtime integration, but ensure cached values have a bounded lifetime so a rotated/revoked credential is not held indefinitely inside a warm Lambda environment.

---

# 18. Meta credential transmission

Meta's current Postman Instagram collection uses Bearer Token authorization helpers for API calls.

Source:

- Meta Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

### Recommendation

Prefer sending the Meta credential in the HTTPS `Authorization: Bearer ...` header where supported by the documented endpoint/client rather than copying it into application URLs/query strings.

Why:

- URLs are more likely to appear in logs/error traces/proxies;
- headers are easier to redact systematically;
- the credential stays outside scheduler/publication data.

If a specific Meta endpoint requires another documented token transport, follow that endpoint's official contract and redact it accordingly.

---

# 19. Logging policy

Production publisher logs must never include:

```text
Meta token value
Meta app secret
Buffer API key
AWS access keys/session credentials
Authorization header
Secrets Manager SecretString
full presigned S3 URL
request URL containing access_token or other credential query values
```

Safe log fields include:

```text
publication_id
attempt_id
account_ref
provider
API operation name
HTTP status
Meta error code/subcode
sanitized error message
container/media ID
credential_ref
credential status
```

---

# 20. HTTP error redaction

HTTP libraries can expose request metadata inside raised exceptions/debug output.

The Meta client should sanitize errors before storing them in `ExecutionAttempt`.

Do not blindly serialize:

```text
request headers
prepared request objects
full response/request URLs
raw HTTP debug traces
```

into CloudWatch or the publication ledger.

The application should extract only approved diagnostic fields.

---

# 21. GitHub logging and masking

GitHub automatically redacts configured Actions secrets in logs, and provides `::add-mask::VALUE` for additional sensitive values.

However, GitHub explicitly warns that redaction is not a perfect security boundary and transformed/derived values may not always be hidden.

Sources:

- GitHub secrets: https://docs.github.com/en/actions/concepts/security/secrets
- GitHub masking workflow command: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands
- GitHub secure use reference: https://docs.github.com/en/actions/reference/security/secure-use

### Recommendation

The strongest policy is:

```text
production Meta token never enters a GitHub Actions runner
```

rather than relying on log masking after injection.

---

# 22. Existing preview URL behaviour

The current review workflow writes presigned preview URLs into:

- `GITHUB_STEP_SUMMARY`;
- `workflow_debug/preview_links.json`;
- workflow artifacts.

That is a **review-only UX pattern** and should remain clearly separated from production publishing credentials/assets.

A valid presigned URL is temporary access to an object and should be treated as sensitive operational data.

For production publication assets, Step 12 recommends generating retrieval URLs inside the publisher at execution time and not writing the full URL to logs/ledger/conversation.

Do not copy the preview-link logging behaviour into the production publisher.

---

# 23. Environment separation

If Eirepolitic later introduces test/staging/production publishing environments, use distinct secrets and account mappings.

Conceptually:

```text
/eirepolitic/dev/instagram/...
/eirepolitic/prod/instagram/...
```

Never let a test publication configuration silently reference the production social credential.

The final canary strategy may use the real account carefully because Meta testing constraints can make a completely separate Instagram sandbox impractical; regardless, credential references/environment state must be explicit.

---

# 24. Manual secret administration

Human operators who need to create/replace a secret should use a controlled AWS administration path.

Do not paste tokens into:

- GitHub issues;
- repo files;
- Markdown research docs;
- chat messages;
- workflow inputs;
- command history unnecessarily.

AWS specifically warns that shell/CLI history and logging can expose secret material when secret values are passed directly through commands.

Source:

- AWS Secrets Manager best practices: https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html

Where practical, use secure console/input mechanisms that avoid storing plaintext in shell history.

---

# 25. Secret compromise response

If a Meta/Buffer/AWS credential may have leaked:

```text
1. stop/disable publication execution if necessary
2. revoke/replace credential at the originating provider
3. update Secrets Manager with validated replacement
4. mark authentication state invalid/rotating until verification succeeds
5. review CloudTrail/CloudWatch/GitHub logs as applicable
6. identify affected scheduled jobs
7. resume only after provider identity/permissions are verified
```

Do not merely delete the leaked value from a log and continue using it.

GitHub likewise recommends deleting/rotating exposed secrets rather than relying only on log redaction.

---

# 26. CloudTrail / secret access auditing

AWS integrates Secrets Manager with CloudTrail/CloudWatch auditing and recommends monitoring secret activity.

Source:

- https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html

This allows future operational investigation such as:

```text
Which AWS principal accessed the production Instagram credential?
```

Do not log the secret value itself; rely on AWS control-plane auditing for access events.

---

# 27. Recommended role boundaries

Conceptual least-privilege roles:

```text
GitHubDeploymentRole
  - deploy/update approved AWS resources
  - no Meta secret value read

InstagramPublisherRole
  - read publication records
  - read approved S3 assets
  - read exact Meta publishing secret
  - write execution/result records
  - publish outbound to Meta

CredentialMaintenanceRole
  - only if needed later
  - update/rotate specific Meta credential
  - validate credential lifecycle

HighDirectorControlRole/API
  - create/edit publication intent/schedule/approval records
  - no secret value read
```

Exact IAM policies are implementation work, not research work.

---

# 28. Direct Meta versus Buffer secret model

## Direct Meta

Eirepolitic Secrets Manager contains the Meta runtime credential.

Advantages:

- direct control;
- one fewer vendor credential/dependency.

Disadvantages:

- Meta token lifecycle is our responsibility.

## Buffer hybrid

Eirepolitic Secrets Manager contains the Buffer API credential.

Buffer stores/manages its connection to Instagram/Meta.

Advantages:

- less Meta-specific authentication handling in Eirepolitic.

Disadvantages:

- trust shifts to Buffer;
- Buffer API token itself remains a production secret;
- account disconnection/expiry still must be monitored through Buffer.

Either architecture still requires proper secret storage and health monitoring.

---

# 29. Recommended migration from current AWS static keys

Do not change the existing workflow during this research branch.

If implementation is later approved, a sensible security-hardening task is:

```text
current GitHub AWS access keys
        ↓
create restricted GitHub OIDC IAM role
        ↓
update workflow to assume role
        ↓
verify S3 preview behaviour
        ↓
remove obsolete long-lived GitHub AWS key secrets
```

This is adjacent to publishing but beneficial even independently because it reduces long-lived cloud credentials in GitHub.

It should be implemented/testing separately from the live Instagram credential setup so failures are easier to isolate.

---

# 30. Step 15 verdict

Recommended secrets architecture:

```text
GitHub Actions
  → AWS via OIDC / short-lived role sessions
  → never receives production Meta token

High Director
  → account_ref / credential health only
  → never receives production Meta token

Lambda publisher
  → IAM execution role
  → Secrets Manager GetSecretValue on exact secret
  → uses credential in-memory for Meta HTTPS call

Publication ledger
  → credential_ref + health metadata only
```

Key rules:

1. Store production Meta/Buffer tokens in AWS Secrets Manager.
2. Keep secret values out of GitHub and publication records.
3. Use least-privilege runtime IAM access to specific secret ARNs.
4. Prefer GitHub OIDC over long-lived AWS access-key secrets for future workflows.
5. Keep bootstrap/renewal credentials separate from normal runtime credentials if both are genuinely needed.
6. Do not assume a universal Meta token lifetime.
7. Record expiry/validation metadata separately and alert before known expiry.
8. Automate renewal only after the exact Meta-supported renewal flow is proven.
9. Never log Authorization headers, access tokens or full presigned URLs.
10. Treat credential compromise as requiring provider-side revocation/replacement, not just log cleanup.

---

## Sources

### Meta

- Current Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Current Token documentation: https://www.postman.com/meta/instagram/folder/i9oo1e6/token
- Current Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login

### AWS

- Secrets Manager overview: https://docs.aws.amazon.com/secretsmanager/latest/userguide/intro.html
- Secrets Manager best practices: https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html
- Secrets Manager IAM policies: https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_iam-policies.html
- Secrets Manager rotation: https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html
- Lambda + Secrets Manager: https://docs.aws.amazon.com/lambda/latest/dg/with-secrets-manager.html

### GitHub

- OIDC with AWS: https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws
- GitHub Actions secrets: https://docs.github.com/en/actions/concepts/security/secrets
- Workflow masking: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands
- Secure use reference: https://docs.github.com/en/actions/reference/security/secure-use

### Repository

- `.github/workflows/instagram_campaign_render.yml`

---

## Confidence / unresolved items

**High confidence:**

- Secrets Manager is appropriate for the production external API credential;
- the publisher runtime, not High Director/GitHub, should retrieve the token;
- GitHub OIDC avoids storing long-lived AWS access keys and is preferable for future AWS workflows;
- current repo uses static AWS access-key secrets;
- Meta's Page-linked route uses a Page Access Token derived/discovered through the documented Page/User authorization flow;
- token values and presigned URLs must be excluded from logs/ledger;
- Meta token lifetime/renewal should not be hard-coded without the actual authentication proof.

**Must be verified during future setup:**

- exact selected Meta runtime token type;
- actual expiry/data-access-expiry properties;
- supported renewal/refresh procedure;
- whether a system-user credential is valid/preferable for the exact publishing path;
- whether Meta app secret/app-secret proof is required or beneficial in the selected runtime configuration;
- exact IAM role/secret ARN structure.

**Next research step:**

Step 16 will design idempotency, failure recovery and retry behaviour, especially how to prevent duplicate Instagram posts when Meta or network responses are uncertain.
