# @21st-sdk/node

Server-side Node.js SDK for [21st Agents](https://21st.dev/agents). Manage sandboxes, threads, and tokens programmatically.

## Install

```bash
npm install @21st-sdk/node
```

## Quick Start

```ts
import { AgentClient } from "@21st-sdk/node"

const client = new AgentClient({
  apiKey: process.env.API_KEY_21ST!, // an_sk_...
})

// Create a sandbox for your agent
const sandbox = await client.sandboxes.create({ agent: "my-agent" })

// Create a thread
const thread = await client.threads.create({
  sandboxId: sandbox.sandboxId,
  name: "Review PR #42",
})

// Generate a short-lived token for browser clients
const { token, expiresAt } = await client.tokens.create({
  agent: "my-agent",
  expiresIn: "1h",
})
```

## API

### `new AgentClient(config)`

```ts
new AgentClient({
  apiKey: string     // Your an_sk_ API key
  baseUrl?: string   // Default: "https://relay.an.dev"
})
```

### `client.sandboxes`

| Method | Description |
|--------|-------------|
| `create({ agent })` | Create a new sandbox for an agent |
| `get(sandboxId)` | Get sandbox details (status, threads, agent info) |
| `delete(sandboxId)` | Delete a sandbox |

### `client.threads`

| Method | Description |
|--------|-------------|
| `list({ sandboxId })` | List all threads in a sandbox |
| `create({ sandboxId, name? })` | Create a new thread |
| `get({ sandboxId, threadId })` | Get thread with messages |
| `delete({ sandboxId, threadId })` | Delete a thread |

### `client.tokens`

| Method | Description |
|--------|-------------|
| `create({ agent?, userId?, expiresIn? })` | Create a short-lived JWT (default: 1h) |

## License

MIT
