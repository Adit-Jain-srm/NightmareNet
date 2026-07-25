import json
import os
import urllib.error
import urllib.request


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not token or not repo or not event_path:
        print("Missing required environment variables for posting comment.")
        return

    try:
        with open(event_path) as f:
            event_data = json.load(f)
    except Exception:
        print(f"Could not read event data from {event_path}")
        return

    if "pull_request" not in event_data:
        print("Not a pull request event. Skipping comment.")
        return

    pr_number = event_data["pull_request"]["number"]

    try:
        with open("comment_body.md", encoding="utf-8") as f:
            body = f.read()
    except Exception:
        print("Could not read comment_body.md")
        return

    api_base = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Fetch existing comments
    req = urllib.request.Request(api_base, headers=headers)
    existing_comment_id = None
    try:
        with urllib.request.urlopen(req) as response:
            comments = json.loads(response.read().decode())
            for comment in comments:
                if "<!-- robustness-delta-comment -->" in comment.get("body", ""):
                    existing_comment_id = comment["id"]
                    break
    except urllib.error.URLError as e:
        print(f"Failed to fetch comments: {e}")

    # Post or update comment
    if existing_comment_id:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_comment_id}"
        method = "PATCH"
    else:
        url = api_base
        method = "POST"

    data = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            print(f"Successfully posted comment. Status: {response.status}")
    except urllib.error.URLError as e:
        print(f"Failed to post comment: {e}")


if __name__ == "__main__":
    main()
