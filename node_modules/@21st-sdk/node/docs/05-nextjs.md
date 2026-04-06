# Next.js Integration

`@21st-sdk/nextjs` provides a server-side token handler so your `an_sk_` API key never reaches the browser. It also re-exports everything from `@21st-sdk/react` for convenience.

## Install

```bash
npm install @21st-sdk/nextjs @21st-sdk/react ai @ai-sdk/react
```

## Setup

### 1. Set your API key

```env
# .env.local
API_KEY_21ST=an_sk_your_key_here
```

Get your API key from [the dashboard](https://21st.dev/agents/dashboard/api).

### 2. Create the token route

```ts
// app/api/agent/token/route.ts
import { createTokenHandler } from "@21st-sdk/nextjs/server"

export const POST = createTokenHandler({
  apiKey: process.env.API_KEY_21ST!,
})
```

### 3. Use in your page

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

## How It Works

```
Browser                     Your Next.js Server              AN Relay
  |                                |                            |
  |-- POST /api/agent/token ------>|                            |
  |                                |-- POST /v1/tokens -------->|
  |                                |   (with an_sk_ key)        |
  |                                |<-- { token, expiresAt } ---|
  |<-- { token, expiresAt } ------|                            |
  |                                                             |
  |-- POST /v1/chat/:agent ------(with short-lived JWT)------->|
  |<-- streaming response -----(SSE)---------------------------|
```

The client only receives short-lived JWTs. Your API key stays on the server.

## API

### `createTokenHandler(options)`

Returns a Next.js `POST` route handler.

```ts
createTokenHandler({
  apiKey: string       // Your an_sk_ API key
  relayUrl?: string    // Default: "https://relay.an.dev"
  expiresIn?: string   // Default: "1h"
})
```

### `exchangeToken(options)`

Lower-level function for custom token exchange logic.

```ts
import { exchangeToken } from "@21st-sdk/nextjs/server"

const { token, expiresAt } = await exchangeToken({
  apiKey: process.env.API_KEY_21ST!,
  relayUrl: "https://relay.an.dev",
  expiresIn: "1h",
})
```

## Entry Points

- `@21st-sdk/nextjs` — Re-exports everything from `@21st-sdk/react` (components, types, `createAgentChat`)
- `@21st-sdk/nextjs/server` — Server-only: `createTokenHandler`, `exchangeToken`
