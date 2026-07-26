from fastapi.testclient import TestClient

from nightmarenet.api.app import app

client = TestClient(app)


def test_pipeline_train_validation_fails_on_bad_data():
    # Sending 'cycles' as 500 when the max allowed is 100
    bad_payload = {"config": {"epochs": 5}, "model_name": "gpt2", "cycles": 500}
    response = client.post("/api/v1/pipeline/train", json=bad_payload)

    # We expect a 422 validation error because cycles > 100
    assert response.status_code == 422
    assert "detail" in response.json()


def test_webhook_settings_validation_fails_on_bad_data():
    # Passing a string instead of a list of webhook configurations
    bad_payload = {"webhooks": "this_is_not_a_valid_list"}
    response = client.post("/settings/webhooks", json=bad_payload)

    # We expect a 422 validation error because it's the wrong data type
    assert response.status_code == 422
    assert "detail" in response.json()
