import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match, Mount

logger = logging.getLogger(__name__)

# Tiers configuration
TIERS = {
    "UNLIMITED": {"limit": 999999, "window": 60},
    "STRICT": {"limit": 5, "window": 60},
    "MODERATE": {"limit": 30, "window": 60},
    "GENEROUS": {"limit": 120, "window": 60},
}

# Default mappings based on request path and method
TIER_MAPPINGS = [
    # UNLIMITED
    (r"^/api/v1/(server/)?health$", "UNLIMITED"),
    (r"^/docs", "UNLIMITED"),
    (r"^/redoc", "UNLIMITED"),
    (r"^/openapi.json", "UNLIMITED"),
    # STRICT
    (r"^/api/v1/auth/login", "STRICT"),
    (r"^/api/v1/auth/register", "STRICT"),
    (r"^/auth/.*", "STRICT"),
    # MODERATE
    (r"^/api/v1/pipeline/create", "MODERATE"),
    (r"^/api/v1/pipeline/[^/]+/cancel", "MODERATE"),
    (r"^/api/v1/settings/webhooks", "MODERATE"),
    (r"^/api/v1/notifications/test-webhook", "MODERATE"),
    (r"^/api/v1/experiments/[^/]+$", "MODERATE"),
    # GENEROUS
    (r"^/api/v1/pipeline/runs", "GENEROUS"),
    (r"^/api/v1/pipeline/[^/]+/status", "GENEROUS"),
    (r"^/api/v1/pipeline/[^/]+/report", "GENEROUS"),
    (r"^/api/v1/search", "GENEROUS"),
    (r"^/api/v1/keys", "GENEROUS"),
]

# Redis Client connection state
_redis_client: Any = None
_redis_initialized = False


def get_redis_client() -> Optional[Any]:
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client

    redis_url = os.environ.get("RATE_LIMIT_REDIS_URL")
    if not redis_url:
        _redis_initialized = True
        logger.info("RATE_LIMIT_REDIS_URL not set - using in-memory rate limiting counter.")
        return None

    try:
        import redis

        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        logger.info("Successfully connected to Redis for rate limiting.")
    except Exception as exc:
        logger.warning(
            "Failed to connect to Redis at %s (%s) - "
            "falling back to in-memory rate limiting counter.",
            redis_url,
            exc,
        )
        _redis_client = None

    _redis_initialized = True
    return _redis_client


# In-memory storage structures
_in_memory_db: Dict[str, int] = {}
_in_memory_expiry: Dict[str, float] = {}
_in_memory_lock = threading.Lock()


def clear_rate_limits() -> None:
    """Clear all in-memory rate limiting counters (mainly for tests)."""
    with _in_memory_lock:
        _in_memory_db.clear()
        _in_memory_expiry.clear()


def _check_in_memory(key: str, limit: int, window: int) -> Tuple[int, int]:
    """Check rate limit in memory. Returns (current_count, reset_seconds)."""
    now = time.time()
    current_time_int = int(now)
    window_start = current_time_int - (current_time_int % window)

    with _in_memory_lock:
        # Clean up expired keys
        expired_keys = [k for k, exp in _in_memory_expiry.items() if exp < now]
        for k in expired_keys:
            _in_memory_db.pop(k, None)
            _in_memory_expiry.pop(k, None)

        # Key specific to the window
        window_key = f"{key}:{window_start}"
        current_count = _in_memory_db.get(window_key, 0) + 1
        _in_memory_db[window_key] = current_count

        if window_key not in _in_memory_expiry:
            _in_memory_expiry[window_key] = float(window_start + window)

        reset_seconds = int(window_start + window - current_time_int)
        return current_count, max(0, reset_seconds)


def _check_redis(client: Any, key: str, limit: int, window: int) -> Tuple[int, int]:
    """Check rate limit in Redis. Returns (current_count, reset_seconds)."""
    now = time.time()
    current_time_int = int(now)
    window_start = current_time_int - (current_time_int % window)
    window_key = f"{key}:{window_start}"

    pipe = client.pipeline()
    pipe.incr(window_key)
    pipe.expire(window_key, window)
    results = pipe.execute()

    current_count = int(results[0])
    reset_seconds = int(window_start + window - current_time_int)
    return current_count, max(0, reset_seconds)


async def enforce_rate_limit(request: Request, tier_name: str) -> Tuple[bool, Dict[str, str]]:
    """Enforces rate limit for a given tier.

    Returns (exceeded, headers).
    """
    if tier_name == "UNLIMITED":
        headers = {
            "X-RateLimit-Limit": "999999",
            "X-RateLimit-Remaining": "999999",
            "X-RateLimit-Reset": "0",
        }
        return False, headers

    tier = TIERS.get(tier_name, TIERS["GENEROUS"])
    limit = tier["limit"]
    window = tier["window"]

    # Identify client
    client_id = None
    # 1. API Key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        client_id = f"apikey:{api_key}"
    else:
        # 2. Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            client_id = f"token:{auth_header[7:]}"

    if not client_id:
        # 3. Client IP
        client_id = f"ip:{request.client.host if request.client else 'unknown'}"

    key = f"rate_limit:{client_id}:{tier_name}"

    # Try Redis
    redis_client = get_redis_client()
    current_count = 0
    reset_seconds = 0
    used_redis = False

    if redis_client:
        try:
            current_count, reset_seconds = _check_redis(redis_client, key, limit, window)
            used_redis = True
        except Exception as exc:
            logger.warning("Redis rate limit check failed: %s. Falling back to in-memory.", exc)

    if not used_redis:
        current_count, reset_seconds = _check_in_memory(key, limit, window)

    remaining = max(0, limit - current_count)

    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_seconds),
    }

    exceeded = current_count > limit
    if exceeded:
        headers["Retry-After"] = str(reset_seconds)

    return exceeded, headers


class RateLimitException(HTTPException):
    def __init__(self, detail: str = "Too many requests. Please try again later."):
        super().__init__(status_code=429, detail=detail)


class RateLimiter:
    def __init__(self, tier: str):
        self.tier = tier

    async def __call__(self, request: Request) -> None:
        exceeded, headers = await enforce_rate_limit(request, self.tier)
        request.state.rate_limit_headers = headers
        request.state.rate_limit_enforced = True

        if exceeded:
            raise RateLimitException(detail="Too many requests. Please try again later.")


def find_route_dependencies(app_obj: Any, scope: Any) -> list:
    """Recursively search for route dependencies in app and sub-mounted apps."""
    if not hasattr(app_obj, "routes"):
        return []
    for route in app_obj.routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            if isinstance(route, Mount):
                # Recurse into mounted application with adjusted scope
                return find_route_dependencies(route.app, child_scope)
            if hasattr(route, "dependencies"):
                return route.dependencies
            break
    return []


class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Initialize state properties
        request.state.rate_limit_enforced = False
        request.state.rate_limit_headers = {}

        # 1. Determine if path matches a default tier
        path = request.url.path
        method = request.method

        # Match against our tier mappings
        matched_tier = None
        for pattern, tier in TIER_MAPPINGS:
            if re.match(pattern, path):
                matched_tier = tier
                break

        if not matched_tier:
            # Method-based fallback
            if method == "GET":
                matched_tier = "GENEROUS"
            else:
                matched_tier = "MODERATE"

        # 2. Check if the matched route contains an explicit RateLimiter dependency.
        # If it does, we let the route dependency handle the enforcement.
        has_limiter_dep = False
        dependencies = find_route_dependencies(request.app, request.scope)
        for dep in dependencies:
            if isinstance(dep.dependency, RateLimiter):
                has_limiter_dep = True
                break

        # If route has no explicit RateLimiter dependency, enforce default matched tier now
        if not has_limiter_dep:
            exceeded, headers = await enforce_rate_limit(request, matched_tier)
            request.state.rate_limit_headers = headers
            request.state.rate_limit_enforced = True

            if exceeded:
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": "Too many requests. Please try again later.",
                    },
                    headers=headers,
                )
                return response

        # Execute the next handler in the stack
        response = await call_next(request)

        # Attach the calculated rate limit headers to the response
        rate_limit_headers = getattr(request.state, "rate_limit_headers", {})
        for k, v in rate_limit_headers.items():
            response.headers[k] = v

        return response
