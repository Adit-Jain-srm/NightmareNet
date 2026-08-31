# Webhook Notifications Guide

NightmareNet supports real-time notification alerts sent directly to your team's communication platforms (Slack, Discord, Microsoft Teams) or custom endpoints when important events occur.

## Event Types

We support four real-time notification event types:

| Event Type | Trigger | Description |
|---|---|---|
| `run_complete` | Pipeline Completion/Failure | Sent when an end-to-end training pipeline succeeds or fails. Includes run ID, status, and robustness delta. |
| `regression_detected` | Robustness Drop | Sent when the training loop results in a model that is less robust than the baseline (negative robustness delta). |
| `alert` | GPU VRAM Pressure | Sent when training VRAM usage exceeds 85%, indicating potential Out-Of-Memory (OOM) risks. |
| `deploy` | Scheduled Benchmark Completion | Sent when a scheduled SST-2 or GPU benchmark completes and saves results. |

---

## Configuration via YAML

You can configure webhooks by adding a `notifications` section in your training YAML configuration files:

```yaml
notifications:
  webhooks:
    - url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
      events:
        - "run_complete"
        - "regression_detected"
        - "alert"
    - url: "https://discord.com/api/webhooks/1234567890/abc_xyz"
      events:
        - "alert"
    - url: "https://outlook.office.com/webhook/...@.../IncomingWebhook/..."
      # If events list is omitted, it defaults to subscribing to all 4 event types
```

---

## Payload Format Compatibility

NightmareNet automatically detects your communication platform from the webhook URL and structures a beautiful, native payload format:

### 1. Slack (Blocks Format)
URLs matching `slack.com` receive a rich block format containing:
- Section header with the event type.
- Bold summary message.
- Bulleted details (e.g. Run ID, Model, Delta, Status).

### 2. Discord (Embeds Format)
URLs matching `discord.com` or `discordapp.com` receive an embedded card format containing:
- Colored left border (Amber/Red for Alerts/Regressions, Blue/Green for successful completes).
- Structured inline fields for key metadata.

### 3. Microsoft Teams (MessageCard format)
URLs matching `office.com`, `microsoft.com`, or `webhook.office.com` receive a MessageCard containing:
- Theme coloring mapped to severity.
- Activity title and structured facts list.

### 4. Generic Webhook (JSON format)
For all other URLs, a compatible JSON payload is sent containing keys:
- `event`: Name of the event type.
- `message`: Human-readable summary.
- `details`: Dict of structured key-value pairs.
- `text`/`content`: Markdown formatted message blocks.

---

## Testing Webhooks

### 1. Settings UI
Navigate to the **Settings** panel and open the **Notifications** tab. You can add a webhook, select subscribed events, and click **Test Connection** to trigger a test payload and verify setup.

### 2. API Endpoint
Send a POST request to `/api/v1/notifications/test-webhook`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/notifications/test-webhook" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://hooks.slack.com/services/...", "event_type": "run_complete"}'
```

---

## Rate Limiting

To prevent abuse, webhook-related endpoints are subject to rate limiting:
- Webhook settings updates (`POST /api/v1/settings/webhooks`) and test requests (`POST /api/v1/notifications/test-webhook`) are governed by the **MODERATE** rate limit tier (30 requests/minute).

