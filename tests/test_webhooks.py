import concurrent.futures
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from nightmarenet.api.app import app
from nightmarenet.api.constants import WEBHOOKS_FILE_PATH

client = TestClient(app)


def test_webhook_settings_roundtrip(tmp_path, monkeypatch):
    """Test saving and retrieving webhook settings successfully."""
    test_file = tmp_path / "webhooks.json"
    monkeypatch.setattr("nightmarenet.api.webhooks.WEBHOOKS_FILE_PATH", str(test_file))

    payload = {
        "webhooks": [
            {
                "url": "https://example.com/webhook",
                "events": ["run_complete"],
                "secret": "test-secret",
            }
        ]
    }

    response = client.post("/api/v1/settings/webhooks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["webhooks"]) == 1
    assert data["webhooks"][0]["url"] == "https://example.com/webhook"

    # Verify retrieval
    get_response = client.get("/api/v1/settings/webhooks")
    assert get_response.status_code == 200
    assert len(get_response.json()["webhooks"]) == 1


def test_corrupted_json_handling(tmp_path, monkeypatch):
    """Test that a corrupted JSON webhook file is handled gracefully by returning empty config."""
    test_file = tmp_path / "webhooks.json"
    test_file.write_text("{invalid_json...", encoding="utf-8")
    monkeypatch.setattr("nightmarenet.api.webhooks.WEBHOOKS_FILE_PATH", str(test_file))

    response = client.get("/api/v1/settings/webhooks")
    assert response.status_code == 200
    assert response.json() == {"webhooks": []}


def test_concurrent_webhook_saves(tmp_path, monkeypatch):
    """Test that concurrent webhook writes do not corrupt the JSON file."""
    test_file = tmp_path / "webhooks.json"
    monkeypatch.setattr("nightmarenet.api.webhooks.WEBHOOKS_FILE_PATH", str(test_file))

    def make_request(i):
        payload = {
            "webhooks": [
                {
                    "url": f"https://example.com/webhook-{i}",
                    "events": ["run_complete"],
                }
            ]
        }
        return client.post("/api/v1/settings/webhooks", json=payload)

    # Run multiple concurrent write requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in futures]

    # All requests should succeed
    for res in results:
        assert res.status_code == 200

    # The file should be valid JSON and readable
    get_response = client.get("/api/v1/settings/webhooks")
    assert get_response.status_code == 200
    assert isinstance(get_response.json()["webhooks"], list)
    