# Enterprise SSO (OpenID Connect)

NightmareNet hosted platform supports enterprise single sign-on via OpenID Connect for Azure AD (Entra ID), Okta, and Auth0. This meets SOC 2 CC6.1 / CC6.2 expectations for logical access and user provisioning.

## What you get

| Capability | Detail |
|---|---|
| OIDC discovery | Provider metadata from `/.well-known/openid-configuration` |
| PKCE (S256) | SPA-safe auth code flow; client secret optional |
| JIT provisioning | Users created on first successful SSO login |
| Role mapping | IdP groups → `admin` / `member` |
| Session lifetime | Default **8 hours** (`NIGHTMARENET_SSO_SESSION_SECONDS`) |

Social login (GitHub/Google) remains under `/auth/*`. Enterprise SSO is under `/api/v1/auth/sso/*`.

---

## Environment variables

```bash
NIGHTMARENET_JWT_SECRET=change-me
NIGHTMARENET_SESSION_SECRET=change-me-too

# Optional single-tenant / local default IdP
NIGHTMARENET_OIDC_DEFAULT_METADATA_URL=https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration
NIGHTMARENET_OIDC_CLIENT_ID=your-app-client-id
NIGHTMARENET_OIDC_CLIENT_SECRET=   # omit for public/PKCE clients
NIGHTMARENET_OIDC_ROLE_CLAIM=groups
NIGHTMARENET_SSO_SESSION_SECONDS=28800
```

Redirect URI to register with the IdP:

```text
https://<your-host>/api/v1/auth/sso/callback
```

---

## Configure a provider per organization (admin API)

```bash
curl -X POST "http://localhost:8000/api/v1/orgs/<org_id>/sso-providers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "azure",
    "issuer": "https://login.microsoftonline.com/<tenant>/v2.0",
    "client_id": "<app-id>",
    "role_claim": "groups",
    "role_mapping": {
      "NightmareNet-Admins": "admin",
      "NightmareNet-Members": "member"
    }
  }'
```

List providers:

```bash
curl "http://localhost:8000/api/v1/orgs/<org_id>/sso-providers"
```

---

## Login flow

1. Browser → `GET /api/v1/auth/sso/login?org_id=<org_id>`
2. NightmareNet discovers the IdP, stores PKCE verifier + state in the session cookie, redirects to the IdP.
3. IdP → `GET /api/v1/auth/sso/callback?code=...&state=...`
4. NightmareNet exchanges the code (PKCE), validates the ID token (signature, `iss`, `aud`, `exp`), JIT-provisions the user, returns:

```json
{
  "user": { "id": "...", "email": "...", "role": "member", "provider": "azure" },
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "bearer",
  "expires_in": 28800
}
```

Frontend helper: `startSsoLogin(orgId)` in `frontend/src/lib/api.ts` redirects to the login endpoint. The callback page at `/auth/sso/callback` stores the access token when the API returns JSON to a browser-handled redirect flow (or when you proxy tokens client-side).

---

## Provider notes

### Azure AD / Entra ID

- Metadata: `https://login.microsoftonline.com/{tenant-id}/v2.0/.well-known/openid-configuration`
- Expose group claims in the token configuration blade if you rely on role mapping.

### Okta

- Metadata: `https://{domain}.okta.com/oauth2/default/.well-known/openid-configuration`
- Map Okta groups via the `groups` claim (or set `role_claim`).

### Auth0

- Metadata: `https://{tenant}.auth0.com/.well-known/openid-configuration`
- Add an Action to include roles/groups in the ID token if needed.

---

## Security notes

- Prefer **PKCE public clients** for SPAs (`client_secret` left empty).
- Application sessions are NightmareNet JWTs (HS256) with lifetime tied to `NIGHTMARENET_SSO_SESSION_SECONDS`. Re-auth when expired for privileged operations.
- ID tokens from the IdP are validated independently (issuer, audience, expiry, signature via JWKS).
