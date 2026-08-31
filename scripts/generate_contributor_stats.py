#!/usr/bin/env python3
"""Generate contributors JSON for frontend/public/contributors.json

Usage: set GITHUB_TOKEN env var (optional but recommended)
       python scripts/generate_contributor_stats.py \
           --owner OWNER --repo REPO \
           --out frontend/public/contributors.json
"""

import argparse
import json
import os
import sys
import time
from typing import List

try:
    import requests
except Exception:
    print("Please install requests: pip install requests")
    raise

API_BASE = "https://api.github.com"


def get_auth_headers(token: str | None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_contributors(owner: str, repo: str, token: str | None) -> List[dict]:
    url = f"{API_BASE}/repos/{owner}/{repo}/contributors"
    params = {"per_page": 100, "anon": "true"}
    headers = get_auth_headers(token)
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


def count_search(query: str, token: str | None) -> int:
    url = f"{API_BASE}/search/issues"
    headers = get_auth_headers(token)
    params = {"q": query}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 422:
        return 0
    resp.raise_for_status()
    data = resp.json()
    return data.get("total_count", 0)


def build_stats(owner: str, repo: str, token: str | None) -> List[dict]:
    contributors = fetch_contributors(owner, repo, token)
    output = []
    for c in contributors:
        login = c.get("login")
        avatar = c.get("avatar_url")
        html = c.get("html_url")
        contributions = c.get("contributions", 0)
        prs = None
        issues = None
        if login and token:
            # Use search API to count PRs and issues authored by this user in the repo
            try:
                prs = count_search(f"repo:{owner}/{repo}+type:pr+author:{login}", token)
                issues = count_search(f"repo:{owner}/{repo}+type:issue+author:{login}", token)
                # slight delay to avoid hitting search rate limits
                time.sleep(0.2)
            except requests.HTTPError:
                prs = None
                issues = None

        output.append(
            {
                "login": login or "",
                "avatar_url": avatar,
                "html_url": html,
                "contributions": contributions,
                "prs": prs,
                "issues": issues,
            }
        )
    # sort by contributions desc
    output.sort(key=lambda x: x.get("contributions", 0), reverse=True)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", default="frontend/public/contributors.json")
    args = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    try:
        stats = build_stats(args.owner, args.repo, token)
    except Exception as e:
        print("Error fetching contributors:", e)
        sys.exit(2)
    dirname = os.path.dirname(args.out)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Wrote {len(stats)} contributors to {args.out}")


if __name__ == "__main__":
    main()
