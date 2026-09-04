# Step 4 — Meta Authentication, Page Linkage, Permissions and Access

Status: **complete**

Research date: 2026-09-03

Scope: document the current Meta authentication models for Instagram publishing, including Facebook Page linkage, permissions, Standard/Advanced Access, App Review implications, verification, token relationships and token-lifecycle considerations.

No Meta app, Page connection, credential, token, account conversion or live publication was created.

---

## Short conclusion

Meta currently offers **two authentication models** for professional Instagram accounts:

1. **Instagram API with Instagram Login**
   - does **not** require a linked Facebook Page;
   - uses an Instagram User access token;
   - publishing permission: `instagram_business_content_publish`;
   - Meta explicitly says this setup **cannot access tagging**.

2. **Instagram API with Facebook Login**
   - **does require** a Facebook Page linked to the Instagram professional account;
   - uses a Facebook Page access token for Instagram API calls;
   - publishing permission: `instagram_content_publish` plus the supporting Page/Instagram permissions;
   - supports the broader Page-linked Instagram feature model.

Because Eirepolitic explicitly wants conversational control over account/media tagging, the Page-linked Facebook Login route currently looks more likely to satisfy the final requirements. **That is not yet a final architecture decision**; Step 5 will verify the detailed tagging capability before we select a route.

For a first-party Eirepolitic app managing only an Instagram account that Eirepolitic owns/manages, Meta's current documentation says **Standard Access** is appropriate. Advanced Access is for apps serving professional Instagram accounts the app does not own/manage.

Do not assume that full external-user App Review or Business Verification is required for the initial own-account proof. Confirm the exact Developer Dashboard requirements when the Meta setup phase is explicitly approved, because Meta's verification/review requirements can depend on the app configuration and permissions requested.

---

## 1. A Meta app is required

Meta's current official Instagram API documentation says an app must first have a **Facebook/Meta App** and recommends a **Business App** for the guide's use case.

The app is the security/permission boundary through which the Instagram professional account authorizes Eirepolitic's publishing integration.

Source:

- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

### Architecture implication

A future production publisher will need a dedicated Meta app configuration, but this should be created only after architecture approval.

The app ID/configuration is part of application configuration. Access tokens/app secrets are secrets and must never be stored in publication manifests or conversational history.

---

## 2. Authentication route A — Instagram Login

Meta's current `Instagram API with Instagram Login` supports Instagram professional accounts directly through **Business Login for Instagram**.

Key properties:

| Item | Current Meta requirement |
|---|---|
| Instagram account | Professional — Business or Creator |
| Facebook Page linkage | **Not required** |
| API host | `graph.instagram.com` |
| Main token | Instagram User access token |
| Publishing permission | `instagram_business_content_publish` |
| Basic permission | `instagram_business_basic` |
| Comments permission | `instagram_business_manage_comments` |
| Messaging permission | `instagram_business_manage_messages` |
| Tagging | **Explicitly unavailable in this setup** |

Current scope names use the `instagram_business_*` prefix. Meta notes that older scope names were deprecated in January 2025.

Sources:

- Meta official Instagram Login folder: https://www.postman.com/meta/instagram/folder/1z5vxzu/instagram-api-with-instagram-login
- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

### Relevance to Eirepolitic

This route is operationally attractive because it avoids a Facebook Page dependency.

However, Meta explicitly states:

> This API setup cannot access ads or tagging.

Since account/media tagging is an explicit High Director requirement, this limitation is potentially disqualifying.

Do not reject this route finally until Step 5 establishes exactly which desired tagging features require the Page-linked route.

---

## 3. Authentication route B — Facebook Login

Meta's `Instagram API with Facebook Login` requires the Instagram Professional account to be linked with a Facebook Page.

Key properties:

| Item | Current Meta requirement |
|---|---|
| Instagram account | Professional — Business or Creator |
| Facebook Page linkage | **Required** |
| API host | `graph.facebook.com` |
| Main runtime token | Facebook Page access token |
| Publishing permission | `instagram_content_publish` |
| Other documented permissions | `pages_show_list`, `instagram_basic`, `pages_read_engagement`, `instagram_manage_comments` |
| Stories | Meta currently says Business-only on this route |

Sources:

- Meta official Facebook Login folder: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login
- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

### Important distinction

A Facebook Page is **not required just to convert** the existing personal Instagram account to professional.

The Page is required if we select this specific Page-linked API authentication route.

---

## 4. Page/account/token relationship on the Facebook Login route

Meta's current token documentation describes this flow:

```text
Authorized Facebook user
        ↓
Facebook User Access Token
        ↓
GET /me/accounts
        ↓
managed Facebook Page
        ↓
Facebook Page Access Token
        +
linked instagram_business_account ID
        ↓
Instagram publishing calls
```

The current official example retrieves:

```text
/me/accounts?fields=name,access_token,tasks,instagram_business_account
```

The result includes:

- Page ID;
- Page access token;
- Page tasks;
- linked Instagram professional account ID.

Meta describes the resulting token as a **Page Access Token**, acting on behalf of a Page linked to the Instagram profile.

Source:

- Meta official Token documentation: https://www.postman.com/meta/instagram/folder/i9oo1e6/token
- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

### Architecture implication

Do not store the human/Facebook identity as the publication identity.

Future configuration should instead use stable internal references such as:

```text
account_ref: eirepolitic
instagram_user_id: <Meta ID>
facebook_page_id: <Meta Page ID, if applicable>
```

The actual access token remains in the secret store.

---

## 5. Permissions for direct publishing

### Facebook Login route

Meta's current collection lists these permissions for the broader route:

```text
pages_show_list
instagram_basic
instagram_content_publish
pages_read_engagement
instagram_manage_comments
```

The essential publication permission is:

```text
instagram_content_publish
```

`instagram_manage_comments` is relevant later if Eirepolitic wants to create/manage comments after publication.

Meta also notes that additional ad permissions can be required in a specific Business Manager role scenario. Eirepolitic should not request advertising permissions unless a real requirement appears.

### Instagram Login route

Current permissions include:

```text
instagram_business_basic
instagram_business_content_publish
instagram_business_manage_comments
instagram_business_manage_messages
```

The essential publication permission is:

```text
instagram_business_content_publish
```

### Least-privilege principle

The final app should request only permissions used by approved features.

For example, do not request messaging permissions merely because Meta exposes them if Eirepolitic is only implementing publishing.

---

## 6. Standard Access versus Advanced Access

Meta's current Instagram documentation explicitly distinguishes the two access levels.

### Standard Access

Use Standard Access when the app serves Instagram professional accounts that:

- the app owner owns/manages; and
- have been added to the app in the App Dashboard.

### Advanced Access

Use Advanced Access when the app serves professional Instagram accounts that the app owner **does not own/manage**.

Source:

- Meta official Instagram User Profile/API requirements: https://www.postman.com/meta/instagram/folder/23987686-22b3a5b0-4a51-449a-9299-e3667d69b182
- Same access-level language appears across current Meta Instagram endpoint requirements.

### Eirepolitic implication

For an initial application that publishes only to the Eirepolitic Instagram account and where that account is genuinely owned/managed by Eirepolitic:

**start with the Standard Access model.**

Do not architect Phase 1 as a SaaS platform onboarding arbitrary third-party Instagram accounts.

If future requirements add other independent organisations' Instagram accounts, Advanced Access becomes relevant and the Meta review burden changes.

---

## 7. App Review

### What can be concluded now

Meta's current Instagram documentation clearly says that own/managed accounts added in the App Dashboard can operate under Standard Access, while third-party accounts require Advanced Access.

Therefore an Eirepolitic-only proof should be designed around **Standard Access first**, rather than assuming we immediately need a public multi-customer Advanced Access review.

### What should not be claimed yet

Do not claim that Eirepolitic will never need App Review.

Meta can require review/approval for specific permissions/features, and the final App Dashboard configuration is authoritative for what must be submitted.

A future move to Advanced Access should be expected to involve Meta's App Review process for the requested permissions/features.

### Practical approach

At approved Meta setup time:

1. create/configure the app;
2. add the owned Eirepolitic professional account as required;
3. inspect `App Review → Permissions and Features`;
4. record which requested permissions have Standard Access;
5. do not request Advanced Access unless the actual use case requires it;
6. document any review requirement before submitting anything to Meta.

This prevents us from designing around outdated assumptions about App Review.

---

## 8. Business Verification / organization verification

### Current finding

The current Instagram API documentation reviewed for this step does **not** state that Business Verification is automatically required merely to use Standard Access with an Instagram professional account that the app owner owns/manages.

Meta does maintain developer/business/organization verification processes, and current Meta developer policy increasingly requires verified developer identities/organizations in portions of the developer ecosystem.

However, the exact verification gate for the future Eirepolitic Instagram app depends on what the Meta Developer Dashboard requires for that app, access level and permissions at setup time.

### Architecture conclusion

Do not make `Business Verification complete` a hard prerequisite of the architecture today.

Instead, at Meta setup time explicitly inspect and record:

```text
Developer account verification status
Organization verification status
Business verification status, if requested
Permission access level
App Review requirements
Data Use Checkup/recertification requirements, if applicable
```

If Meta requires verification for the chosen configuration, complete it before production use.

This is safer than either assuming verification is unnecessary or assuming a full business-verification process is definitely mandatory.

---

## 9. User access tokens versus Page access tokens

The Facebook Login route starts with a Facebook **User Access Token** representing the person who authorized access.

Meta then uses that authorization to retrieve a **Page Access Token** for a managed Page linked to the Instagram professional account.

Runtime Instagram calls on the Page-linked route should use the Page token model documented for the endpoint rather than putting a human's raw User Access Token into publication records.

### Eirepolitic implication

The publication worker should retrieve the active runtime token from a secure secret store.

High Director should know only:

```text
account_ref: eirepolitic
auth_status: valid | expiring | invalid
```

not the token value.

---

## 10. System-user tokens

Meta's current Instagram API documentation contains Instagram surfaces that state the API supports both **user and system-user access tokens**.

However, the current **Content Publishing requirements are more specific**:

- Instagram Login publishing documents an **Instagram User access token**;
- Facebook Login uses a **Facebook Page access token**.

### Conclusion for now

System-user tokens exist in the Meta platform/Instagram ecosystem, but **do not design the first publisher around an assumed permanent system-user token until a controlled authentication proof confirms that the exact chosen Content Publishing route supports it as intended**.

For Phase 1 architecture, follow the token type explicitly documented for the chosen publishing endpoint.

If a system-user token can later be used legitimately for unattended server-to-server operation, it may reduce dependence on a particular human authorization lifecycle and should be evaluated during the Meta authentication proof phase.

Source:

- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

---

## 11. Token lifetime and renewal

### Important research result

The current Meta Instagram publishing collection clearly documents the token *types and relationships*, but the pages reviewed in this step do not provide a single reliable current lifetime/renewal rule that should be hard-coded into the Eirepolitic architecture for every authentication route.

Meta token behaviour varies by:

- token type;
- login method;
- app configuration;
- user/Page/business authorization state;
- revocation events.

Older assumptions such as "this token is permanent" or "all Meta tokens last exactly N days" should therefore not be embedded into the design without testing the selected route.

### Required design

Store token metadata separately from the secret value:

```text
auth_provider
credential_type
issued_at
expires_at            # when supplied/known
last_validated_at
last_refreshed_at
status
```

The system should actively validate authentication health and alert before known expiry where possible.

### Token invalidation can happen before nominal expiry

Even a token that has not reached its recorded expiry can become unusable due to authorization/account changes.

The publisher must therefore treat actual Meta authentication failures as authoritative.

Examples of operational causes include:

- user revokes app access;
- Page permissions/roles change;
- Instagram account is disconnected from the Page;
- Instagram account changes back to Personal;
- app/permission access changes;
- credential is invalidated by Meta/security events.

### Setup-time proof required

When Meta integration is later explicitly approved, the authentication proof should record from the actual token/debugging responses:

- token type;
- exact expiry timestamp;
- scopes/permissions granted;
- data-access expiry if present;
- Page/account relationship;
- refresh/renewal method supported for that token;
- behaviour after renewal.

Only then should an automated refresh/reauthorization policy be implemented.

---

## 12. Token health behaviour for the eventual publisher

Before attempting a scheduled publication, the publisher should perform a lightweight deterministic account/authentication validation where appropriate.

On a clear authentication failure:

```text
scheduled
   ↓
auth validation failure
   ↓
auth_blocked / needs_attention
   ↓
operator notification
```

Do **not** repeatedly retry an obviously revoked/invalid credential.

Once authorization is restored, the same publication request can be resumed under its existing idempotency controls.

This prevents token problems from producing duplicate posts or uncontrolled retries.

---

## 13. Login-route comparison at the end of Step 4

| Requirement | Instagram Login | Facebook Login |
|---|---|---|
| Professional account required | Yes | Yes |
| Existing personal account can be converted in place | Yes, from Step 2 | Yes, from Step 2 |
| Facebook Page required | **No** | **Yes** |
| Host | `graph.instagram.com` | `graph.facebook.com` |
| Runtime token model | Instagram User access token | Facebook Page access token |
| Publish permission | `instagram_business_content_publish` | `instagram_content_publish` |
| Standard Access for owned account | Yes | Yes |
| Advanced Access for third-party accounts | Yes | Yes |
| Tagging | **Meta explicitly says unavailable** | Broader route; exact features Step 5 |
| Stories | Supported by current platform docs subject to route/account rules | Business-only per current Facebook Login docs |
| Likely Eirepolitic fit | Simpler auth but tagging limitation | **More likely**, pending Step 5 |

---

## 14. Recommended authentication direction — provisional only

Current evidence favours investigating this as the likely direct-Meta path:

```text
Existing Eirepolitic personal account
      ↓ explicit future approval
Switch same account to appropriate Professional type
      ↓
Link to Eirepolitic Facebook Page
      ↓
Meta Business App
      ↓
Facebook Login for Business
      ↓
Page Access Token
      ↓
Instagram Content Publishing API
```

Why provisional:

Eirepolitic's desired functionality includes media/account tagging, and Meta explicitly excludes tagging from the Instagram Login route.

However, Step 5 must verify exactly what Meta currently supports for:

- caption mentions;
- image/media tags;
- carousel tags;
- collaborators;
- locations;
- alt text;
- first comments;
- product tagging.

Only after that should the authentication route be treated as selected.

---

## 15. What should happen during a future Meta setup phase

Only after explicit architecture approval:

1. confirm current Eirepolitic Instagram account ownership/status;
2. choose Creator or Business based on the completed feature research;
3. switch the same existing account in place if required;
4. if Facebook Login is selected, confirm/create the correct Facebook Page linkage;
5. create/configure the Meta Business App;
6. inspect the current App Dashboard access levels and verification requirements;
7. request only the minimum permissions actually needed;
8. generate a non-production/test authorization where possible;
9. record token type, scopes and expiry metadata;
10. test read-only identity/account discovery first;
11. test `/content_publishing_limit` or other non-publishing endpoint;
12. only later, under another explicit approval, perform a real publication canary.

---

## Sources

Primary authoritative sources used for this step:

- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Meta official Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login
- Meta official Instagram API with Instagram Login: https://www.postman.com/meta/instagram/folder/1z5vxzu/instagram-api-with-instagram-login
- Meta official Token folder: https://www.postman.com/meta/instagram/folder/i9oo1e6/token
- Meta official Instagram User Profile/API requirements: https://www.postman.com/meta/instagram/folder/23987686-22b3a5b0-4a51-449a-9299-e3667d69b182

Additional current Meta developer verification context:

- Meta Developer Verification Policy: https://developers.meta.com/horizon/policy/developer-verification/
- Meta business verification documentation: https://developers.meta.com/horizon/resources/publish-organization-verification-business/

The verification pages above are Meta-wide/current developer-policy context rather than Instagram Content Publishing-specific proof of a mandatory Eirepolitic Business Verification requirement. The future App Dashboard remains authoritative for that requirement.

---

## Confidence / unresolved items

**High confidence:**

- two current login routes exist;
- Instagram Login does not require a Facebook Page;
- Instagram Login cannot access tagging;
- Facebook Login requires a Page linked to the professional Instagram account;
- Facebook Login runtime calls use a Page access token;
- current publishing permissions for both routes;
- Standard Access is for owned/managed accounts, Advanced Access for accounts not owned/managed.

**Conservative / must be verified during setup:**

- exact App Review gates shown by the future App Dashboard;
- whether Business/organization verification is required for Eirepolitic's exact Standard Access configuration;
- exact token lifetime/refresh behaviour for the final selected credential;
- whether a system-user token is appropriate for the exact Content Publishing path.

**Next research step:**

Step 5 will determine whether the Page-linked route is genuinely required by Eirepolitic's tagging/metadata requirements.
