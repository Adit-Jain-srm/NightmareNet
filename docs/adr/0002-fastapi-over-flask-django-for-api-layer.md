# 2. FastAPI over Flask/Django for API Layer

Date: 2026-03-20

## Status 

Accepted

## Context

Need a Python web framework that supports async operations, auto-generates OpenAPI docs, and integrates with Pydantic for type-safe validation.

## Decision 

Use FastAPI as the API framework.

## Consequences

- (+) Native async support for concurrent request handling
- (+) Automatic OpenAPI/Swagger documentation
- (+) Pydantic v2 integration for request/response validation
- (+) High performance (Starlette + uvicorn)
- (-) Smaller community than Flask/Django (mitigated by rapid growth)
- (-) Pydantic v2 incompatibility with `from __future__ import annotations`
