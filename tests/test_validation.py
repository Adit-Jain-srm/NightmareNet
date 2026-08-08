from fastapi.testclient import TestClient

from nightmarenet.api.app import app

client = TestClient(app)


def test_webhook_settings_validation_fails_on_bad_data():
    # Passing a string instead of a list of webhook configurations
    bad_payload = {"webhooks": "this_is_not_a_valid_list"}
    response = client.post("/api/v1/settings/webhooks", json=bad_payload)

    # We expect a 422 validation error because it's the wrong data type
    assert response.status_code == 422
    assert "detail" in response.json()
