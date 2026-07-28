---
title: Backend is self-hosted with no external deployment
slug: backend-is-self-hosted-with-no-external-deployment
tags: 
scope: project
updated_at: 2026-07-28T09:04:34.263Z
source: live
hook: Backend is self-hosted with no external deployment
---

• Backend is self-hosted with no external deployment
• Local development: `uvicorn nightmarenet.api.app:app --host 0.0.0.0 --port 8000 --reload`
• Docker deployment: `docker compose up` or `docker compose --profile hosted up` for full stack
• Production: Users build and deploy Docker images on their own infrastructure
• No hosted SaaS endpoint or staging URL available for external checking
• CI builds Docker images but does not deploy them externally
• No deployed backend to verify existence of
