from slowapi import Limiter
from slowapi.util import get_remote_address
from nightmarenet.api.schemas import (
    WebhookSettingsResponse,
    WebhookSettingsRequest,
    WebhookTestResponse,
    ErrorResponse,
    TestWebhookRequest,
)
from fastapi import HTTPException, Request, APIRouter, Body
import os
import json
import logging

router = APIRouter()

WEBHOOKS_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "webhooks.json"
)

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@router.get(
    "/api/v1/settings/webhooks",
    response_model=WebhookSettingsResponse,
    summary="Get webhook settings",
    tags=["settings"],
)
async def get_webhook_settings():
    """Retrieve the current webhook settings."""
    if not os.path.exists(WEBHOOKS_FILE_PATH):
        return WebhookSettingsResponse(webhooks=[])
    try:
        with open(WEBHOOKS_FILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return WebhookSettingsResponse(webhooks=data.get("webhooks", []))
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
    """Save webhook settings."""
    try:
        os.makedirs(os.path.dirname(WEBHOOKS_FILE_PATH), exist_ok=True)
        with open(WEBHOOKS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"webhooks": [w.model_dump() for w in body.webhooks]}, f, indent=2
            )
        return WebhookSettingsResponse(webhooks=body.webhooks)
    except Exception as e:
        logger.error("Failed to save webhooks: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to save webhook settings."
        ) from None


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

    # Temporary configuration dict containing the target webhook
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
        # Build some mock details depending on the event type
        details = {
            "test": "true",
            "message": f"This is a test notification for {body.event_type}.",
        }
        if body.event_type == "run_complete":
            details.update(
                {"run_id": "test-run-12345", "status": "complete", "model": "gpt2"}
            )
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
