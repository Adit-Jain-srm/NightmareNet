import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Request

from nightmarenet.api.constants import WEBHOOKS_FILE_PATH, limiter
from nightmarenet.api.schemas import (
    ErrorResponse,
    TestWebhookRequest,
    WebhookSettingsRequest,
    WebhookSettingsResponse,
    WebhookTestResponse,
)

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get(
    "/api/v1/settings/webhooks",
    response_model=WebhookSettingsResponse,
    summary="Get webhook settings",
    tags=["settings"],
)
async def get_webhook_settings():
    """Retrieve the current webhook settings with graceful JSON corruption handling."""
    webhook_path = Path(WEBHOOKS_FILE_PATH)
    if not webhook_path.exists():
        return WebhookSettingsResponse(webhooks=[])
    
    try:
        with open(webhook_path, encoding="utf-8") as f:
            # Try to acquire a shared advisory lock for reading if supported (Unix)
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            except (ImportError, OSError):
                pass

            data = json.load(f)
            
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass

            webhooks = data.get("webhooks", []) if isinstance(data, dict) else []
            return WebhookSettingsResponse(webhooks=webhooks)
    except json.JSONDecodeError as jde:
        logger.warning("Webhook settings file is corrupted (%s). Returning empty configuration.", jde)
        return WebhookSettingsResponse(webhooks=[])
    except Exception as e:
        logger.error("Failed to read webhooks: %s", e)
        return WebhookSettingsResponse(webhooks=[])


@router.post(
    "/api/v1/settings/webhooks",
    response_model=WebhookSettingsResponse,
    summary="Save webhook settings",
    tags=["settings"],
)
async def save_webhook_settings(
    request: Request,
    body: WebhookSettingsRequest,
):
    """Save webhook settings using atomic writes and advisory file locking."""
    webhook_path = Path(WEBHOOKS_FILE_PATH)
    webhook_dir = webhook_path.parent
    webhook_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Atomic write: write to temp file in the same directory, then replace
        fd, temp_path = tempfile.mkstemp(dir=str(webhook_dir), prefix="webhooks_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                # Acquire exclusive advisory lock (Unix)
                try:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass

                json.dump({"webhooks": [w.model_dump() for w in body.webhooks]}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

                try:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass

            os.replace(temp_path, webhook_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        return WebhookSettingsResponse(webhooks=body.webhooks)
    except Exception as e:
        logger.error("Failed to save webhooks: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save webhook settings.") from None


_TEST_WEBHOOK_BODY = Body(...)


@router.post(
    "/api/v1/notifications/test-webhook",
    response_model=WebhookTestResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Send a test notification to a webhook URL",
    tags=["notifications"],
)
@limiter.limit("5/minute")
async def test_webhook_endpoint(
    request: Request,
    body: TestWebhookRequest = _TEST_WEBHOOK_BODY,
):
    """Send a test notification payload to verify webhook integration."""
    from nightmarenet.utils.webhooks import trigger_webhook, validate_webhook_url

    if not validate_webhook_url(body.url):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid webhook URL. Must be an allowed HTTPS domain"
                " and not resolve to an internal IP."
            ),
        )

    temp_config = {
        "notifications": {
            "webhooks": [
                {
                    "url": body.url,
                    "events": [body.event_type],
                }
            ]
        }
    }

    try:
        details = {
            "test": "true",
            "message": f"This is a test notification for {body.event_type}.",
        }
        if body.event_type == "run_complete":
            details.update({"run_id": "test-run-12345", "status": "complete", "model": "gpt2"})
        elif body.event_type == "regression_detected":
            details.update(
                {
                    "robustness_delta": "-0.0543",
                    "baseline_auc": "0.8520",
                    "trained_auc": "0.7977",
                }
            )
        elif body.event_type == "alert":
            details.update(
                {
                    "gpu": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
                    "usage_percent": "91.2%",
                }
            )
        elif body.event_type == "deploy":
            details.update({"mode": "full", "output_path": "results/benchmark-v1.json"})

        trigger_webhook(
            temp_config,
            body.event_type,
            f"Test notification: {body.event_type} integration test.",
            details,
        )
        return WebhookTestResponse(status="ok")
    except Exception as e:
        logger.exception("Test webhook failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to dispatch test webhook: {e}",
        ) from None
    