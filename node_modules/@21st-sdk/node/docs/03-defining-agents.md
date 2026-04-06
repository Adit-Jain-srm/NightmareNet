# Defining Agents

Agents are defined with the `agent()` and `tool()` functions from `@21st-sdk/agent`. These are config-only — they return exactly what you pass in, with type inference added. The actual execution happens in the AN runtime (E2B sandbox).

## `agent(config)`

```ts
import { agent } from "@21st-sdk/agent"

export default agent({
  // Model (default: "claude-sonnet-4-6")
  model: "claude-sonnet-4-6",

  // For "claude-code" runtime, string prompts are appended
  // to Claude Code's default preset prompt by default.
  systemPrompt: "You are a PR reviewer...",

  // Custom tools (see below)
  tools: { /* ... */ },

  // Runtime: "claude-code" (default) or "codex"
  runtime: "claude-code",

  // Permission mode: "default", "acceptEdits", or "bypassPermissions"
  permissionMode: "default",

  // Max conversation turns (default: 50)
  maxTurns: 50,

  // Max budget in USD
  maxBudgetUsd: 5,

  // Lifecycle hooks (see below)
  onStart: async () => {},
  onToolCall: async ({ toolName, input }) => {},
  onToolResult: async ({ toolName, input, output }) => {},
  onStepFinish: async ({ step }) => {},
  onFinish: async ({ result }) => {},
  onError: async ({ error }) => {},
})
```

### Defaults

| Field | Default |
|-------|---------|
| `model` | `"claude-sonnet-4-6"` |
| `runtime` | `"claude-code"` |
| `systemPrompt` | Claude Code preset prompt when omitted, or preset + appended string when a string is provided and `runtime === "claude-code"` |
| `permissionMode` | `"default"` |
| `maxTurns` | `50` |
| `tools` | `{}` |

## `tool(definition)`

```ts
import { tool } from "@21st-sdk/agent"
import { z } from "zod"

const myTool = tool({
  description: "What this tool does",
  inputSchema: z.object({
    path: z.string(),
    verbose: z.boolean().optional(),
  }),
  execute: async (args) => {
    // args is fully typed: { path: string; verbose?: boolean }
    return {
      content: [{ type: "text", text: "result" }],
    }
  },
})
```

The `execute` function receives args typed from the Zod schema. Return value must have `content` array with `{ type: "text", text: string }` items.

## Hooks

### `onToolCall`

Runs before a tool executes. Return `false` to block it.

```ts
onToolCall: async ({ toolName, input }) => {
  if (toolName === "Bash" && input.command?.includes("rm -rf")) {
    return false // blocked
  }
}
```

### `onToolResult`

Runs after a tool executes with the result.

```ts
onToolResult: async ({ toolName, input, output }) => {
  console.log(`Tool ${toolName} returned:`, output)
}
```

### `onFinish`

Runs when the agent completes successfully.

```ts
onFinish: async ({ result }) => {
  await fetch("https://my-api.com/webhook", {
    method: "POST",
    body: JSON.stringify({ done: true }),
  })
}
```

### `onError`

Runs when the agent encounters an error.

```ts
onError: async ({ error }) => {
  console.error("Agent failed:", error)
}
```

## Permission Modes

| Mode | Behavior |
|------|----------|
| `"default"` | Agent asks for confirmation on risky operations |
| `"acceptEdits"` | Auto-approve file edits, still confirm other actions |
| `"bypassPermissions"` | Auto-approve everything (use with caution) |

## Type Reference

```ts
// Agent config (all fields required — defaults filled by agent())
interface AgentConfig {
  model: string
  systemPrompt?:
    | string
    | { type: "preset"; preset: "claude_code"; append?: string }
  tools: ToolSet
  runtime: "claude-code" | "codex"
  permissionMode: "default" | "acceptEdits" | "bypassPermissions"
  maxTurns: number
  maxBudgetUsd?: number
  onStart?: () => Promise<void>
  onToolCall?: (payload: { toolName: string; input: any }) => Promise<boolean | void>
  onToolResult?: (payload: { toolName: string; input: any; output: any }) => Promise<void>
  onStepFinish?: (payload: { step: any }) => Promise<void>
  onFinish?: (payload: { result: any }) => Promise<void>
  onError?: (payload: { error: Error }) => Promise<void>
}

// Tool definition — generic over Zod schema
interface ToolDefinition<TInput> {
  description: string
  inputSchema: ZodType<TInput>
  execute: (args: TInput) => Promise<{ content: { type: string; text: string }[] }>
}

// Tool set
type ToolSet = Record<string, ToolDefinition<any>>
```
