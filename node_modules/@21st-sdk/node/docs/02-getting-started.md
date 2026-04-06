# Getting Started

## 1. Get an API Key

Sign up at [21st.dev/agents](https://21st.dev/agents) and get your API key from [the dashboard](https://21st.dev/agents/dashboard/api).

## 2. Install

```bash
npm install @21st-sdk/agent zod
```

## 3. Define Your Agent

Create `src/agent.ts`:

```ts
import { agent, tool } from "@21st-sdk/agent"
import { z } from "zod"

export default agent({
  model: "claude-sonnet-4-6",
  systemPrompt: "You are a helpful coding assistant.",
  tools: {
    greet: tool({
      description: "Greet a user by name",
      inputSchema: z.object({ name: z.string() }),
      execute: async ({ name }) => ({
        content: [{ type: "text", text: `Hello, ${name}!` }],
      }),
    }),
  },
})
```

## 4. Login & Deploy

```bash
npx @21st-sdk/cli login
# Enter your API key: an_sk_...

npx @21st-sdk/cli deploy
# Bundling src/agent.ts...
# Deploying my-agent...
# https://api.an.dev/v1/chat/my-agent
```

Your agent is live.

## 5. Embed in a React App

```bash
npm install @21st-sdk/nextjs @21st-sdk/react ai @ai-sdk/react
```

Create a token route (keeps your API key on the server):

```ts
// app/api/agent/token/route.ts
import { createTokenHandler } from "@21st-sdk/nextjs/server"

export const POST = createTokenHandler({
  apiKey: process.env.API_KEY_21ST!,
})
```

Add the chat UI:

```tsx
// app/page.tsx
"use client"

import { useChat } from "@ai-sdk/react"
import { AgentChat, createAgentChat } from "@21st-sdk/nextjs"
import "@21st-sdk/react/styles.css"
import { useMemo } from "react"

export default function Chat() {
  const chat = useMemo(
    () => createAgentChat({
      agent: "your-agent-slug",
      tokenUrl: "/api/agent/token",
    }),
    [],
  )

  const { messages, sendMessage, status, stop, error } = useChat({ chat })

  return (
    <AgentChat
      messages={messages}
      onSend={(msg) =>
        sendMessage({ parts: [{ type: "text", text: msg.content }] })
      }
      status={status}
      onStop={stop}
      error={error}
    />
  )
}
```

## Next Steps

- [Defining Agents](./03-defining-agents.md) — Models, tools, hooks, permissions
- [React UI](./04-react-ui.md) — Theming, slots, tool renderers
- [Next.js Integration](./05-nextjs.md) — Server-side token handler
- [Node SDK](./06-node-sdk.md) — Sandboxes, threads, tokens from server code
- [CLI Reference](./07-cli.md) — All CLI commands
- [Custom Tools](./08-custom-tools.md) — Build custom tool renderers
