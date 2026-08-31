#!/usr/bin/env python3
"""Post or update a PR comment with the robustness-check markdown report."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

MARKER = "<!-- nightmarenet-robustness-check -->"


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("INPUT_GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    comment_path = os.environ.get("INPUT_COMMENT_PATH") or os.environ.get("COMMENT_PATH")

    if not token or not repo or not event_path:
        print("Missing GitHub context; skipping PR comment.")
        return 0

    if not comment_path or not Path(comment_path).is_file():
        print(f"Comment body not found at {comment_path!r}; skipping.")
        return 0

    try:
        with open(event_path, encoding="utf-8") as fh:
            event_data = json.load(fh)
    except OSError as exc:
        print(f"Could not read event data: {exc}")
        return 0

    pr = event_data.get("pull_request") or event_data.get("issue")
    if not pr or "number" not in pr:
        print("Not a pull_request/issue event; skipping comment.")
        return 0

    pr_number = pr["number"]
    body = Path(comment_path).read_text(encoding="utf-8")
    if MARKER not in body:
        body = f"{MARKER}\n{body}"

    api_base = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "nightmarenet-robustness-check",
    }

    existing_id = None
    req = urllib.request.Request(api_base, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            comments = json.loads(response.read().decode("utf-8"))
        for comment in comments:
            if MARKER in comment.get("body", ""):
                existing_id = comment["id"]
                break
    except urllib.error.URLError as exc:
        print(f"Failed to list comments: {exc}")

    if existing_id:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_id}"
        method = "PATCH"
    else:
        url = api_base
        method = "POST"

    payload = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            print(f"PR comment {method} ok (status {response.status})")
    except urllib.error.URLError as exc:
        print(f"Failed to post comment: {exc}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
