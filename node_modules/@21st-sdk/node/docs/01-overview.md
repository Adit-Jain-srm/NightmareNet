# AN Platform Overview

AN is a platform for building, deploying, and embedding AI agents. You define an agent in TypeScript, deploy it with one command, and embed a chat UI in any React app.

## Architecture

```
Your Code (agent.ts)
    |
    v
@21st-sdk/cli deploy (CLI)
    |
    v
AN Platform
    |
    +---> E2B Sandbox (isolated Node.js environment)
    |         |
    |         +---> Claude Agent SDK / Codex (executes your agent)
    |         |
    |         +---> Your custom tools run here
    |
    +---> AN Relay (relay.an.dev)
              |
              +---> SSE streaming to clients
              |
              +---> Token exchange (API key -> short-lived JWT)
```

## Packages

| Package | Purpose |
|---------|---------|
| `@21st-sdk/agent` | Define agents and tools with type inference |
| `@21st-sdk/cli` | Deploy agents from your terminal |
| `@21st-sdk/react` | Chat UI components (theming, tool renderers) |
| `@21st-sdk/nextjs` | Next.js server-side token handler |
| `@21st-sdk/node` | Server-side SDK (sandboxes, threads, tokens) |

## Key Concepts

- **Agent**: A TypeScript config defining model, system prompt, tools, and hooks. Runs in a cloud sandbox.
- **Tool**: A function your agent can call, with a Zod schema for input validation and full type inference.
- **Sandbox**: An isolated E2B cloud environment where your agent executes. Has Node.js, git, and system tools.
- **Thread**: A conversation within a sandbox. One sandbox can have multiple threads.
- **Relay**: The streaming gateway at `relay.an.dev`. Handles auth, routing, and SSE streaming.

## Runtimes

AN supports two runtimes:

- **`claude-code`** (default) — Uses the Claude Agent SDK. Full tool use, file editing, bash execution.
- **`codex`** — Uses OpenAI Codex via ACP provider.

## Auth Model

Two layers of authentication:

1. **Client -> Relay**: API key (`an_sk_...`) or short-lived JWT (via token exchange)
2. **Sandbox -> AI Provider**: Handled internally by the platform (Claude Proxy)

For web apps, use `@21st-sdk/nextjs` to exchange your API key for a short-lived JWT on the server, so the key never reaches the browser.
