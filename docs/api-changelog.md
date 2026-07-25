# API Changelog

All notable changes to the NightmareNet API will be documented in this file. This project follows [Semantic Versioning 2.0.0](https://semver.org/) for its API endpoints, request schemas, and responses.

---

## [0.2.0] - 2026-07-01

### Change Type
- Added

### Endpoint(s) Affected
- `GET /api/v1/health`
- `POST /api/v1/generate/dream`
- `POST /api/v1/generate/nightmare`
- `POST /api/v1/evaluate/robustness`
- `POST /api/v1/train/config`
- `POST /api/v1/compare`
- `POST /api/v1/demo`
- `POST /api/v1/upload/text`
- `POST /api/v1/pipeline/train`
- `POST /api/v1/pipeline/evaluate`
- `POST /api/v1/pipeline/cancel`
- `POST /api/v1/pipeline/create`
- `GET /api/v1/pipeline/{run_id}/status`
- `POST /api/v1/pipeline/{run_id}/cancel`
- `GET /api/v1/pipeline/{run_id}/report`
- `GET /api/v1/pipeline/runs`
- `POST /api/v1/notifications/test-webhook`
- `POST /settings/webhooks`
- `GET /api/v1/compliance/report/{run_id}`
- `GET /api/v1/compliance/reports`
- `WS /ws/runs/{run_id}`

### Description
Initial release of the NightmareNet FastAPI service. Provides a complete set of REST API endpoints for orchestrating the Wake-Dream-Nightmare-Compress training pipeline, generating distortions, evaluating robustness, and managing compliance reporting.

### Migration Guide
None. (First release)

### Notes
All endpoints require a valid API key via the `X-API-Key` header if the `NIGHTMARENET_API_KEY` environment variable is set.

---

## Contributor Template (Future Releases)

Contributors: when making changes to the API, copy this template to the top of the file under the main title and fill in the details.

```markdown
## [<API Version>] - <Release Date (YYYY-MM-DD)>

### Change Type
- <Breaking / Non-breaking / Deprecated / Added / Fixed / Removed>

### Endpoint(s) Affected
- `<HTTP Method> <Endpoint Path>` (e.g., `POST /api/v1/generate/dream`)

### Description
<Detailed description of the changes, the reasoning, and expected behavior.>

### Migration Guide
<Instructions on how users should adapt their API integration. State "None" if non-breaking.>

### Notes
<Extra context, performance impacts, or configuration requirements.>
```
