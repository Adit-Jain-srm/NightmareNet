# API Versioning Policy

NightmareNet uses header-based API versioning for the public HTTP surface.

## Version header

All responses from the API must include the following header:

- `API-Version: v1`

This header is present on every request, including successful responses, validation errors, authorization failures, and 404 not found responses.

## Implementation

- The active API version is `v1`.
- All responses should also continue to include `X-API-Version` for the service release version.
- A shared HTTP middleware layer is responsible for injecting `API-Version` and any deprecation headers.

## Deprecation strategy

When an endpoint is deprecated, it should expose the following headers:

- `Deprecation: true`
- `Sunset: <RFC 7231 HTTP-date>`
- `Link: </path/to/alternative>; rel="alternate"` (optional)

The `Sunset` header must use a UTC HTTP-date string, not a plain date.

### Example

A deprecated endpoint should be decorated with a helper such as:

```python
from nightmarenet.api.versioning import deprecated

@app.post("/api/v1/compare", deprecated=True)
@deprecated(sunset="2026-12-01", alternative="/api/v1/evaluate/robustness")
async def compare_distortions(...):
    ...
```

And the middleware should expose headers like:

```http
API-Version: v1
X-API-Version: 0.3.0
Deprecation: true
Sunset: Tue, 01 Dec 2026 00:00:00 GMT
Link: </api/v1/evaluate/robustness>; rel="alternate"
```

## OpenAPI documentation

- Deprecated endpoints should be marked as `deprecated: true` in the generated OpenAPI spec.
- The committed spec lives at `docs/api/openapi.json` and should be regenerated when endpoint metadata changes.

## Migration guidance

- Clients must check `API-Version` to confirm they are talking to `v1`.
- Deprecated endpoints should be migrated to their recommended replacement before the listed sunset date.

## Notes

This versioning strategy is intended to support a stable `v1` surface while allowing safe deprecation notices without requiring path-based version negotiation.
