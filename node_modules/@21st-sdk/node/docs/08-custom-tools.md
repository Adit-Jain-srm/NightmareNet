# Custom Tool Renderers

The `@21st-sdk/react` chat UI automatically renders tool calls from your agent. It includes built-in renderers for all standard Claude tools and supports custom renderers for your own tools.

## Built-in Tool Renderers

These are rendered automatically when your agent uses standard tools:

| Tool | Renderer | Display |
|------|----------|---------|
| `Bash` | `BashTool` | Terminal card with command and output |
| `Edit` | `EditTool` | Diff card with file path and changes |
| `Write` | `WriteTool` | File creation card |
| `WebSearch` | `SearchTool` | Search results list |
| `TodoWrite` | `TodoTool` | Task checklist with progress |
| `EnterPlanMode` | `PlanTool` | Step list with progress bar |
| `Task` | `TaskTool` | Sub-agent task with nested tools |
| `mcp__*` | `McpTool` | MCP tool call with params and output |
| `thinking` | `ThinkingTool` | Collapsible reasoning block |
| Other | `GenericTool` | Fallback JSON display |

## Custom Tool Renderers via Slots

To render your custom tools differently, use the `slots.ToolRenderer` prop:

```tsx
import { AgentChat, ToolRenderer } from "@21st-sdk/react"
import type { ToolPart } from "@21st-sdk/react"

function MyToolRenderer(props: { part: ToolPart; status: string }) {
  const { part, status } = props

  // Handle your custom tool
  if (part.toolInvocation.toolName === "weather") {
    const args = part.toolInvocation.args
    const result = part.toolInvocation.state === "result"
      ? part.toolInvocation.result
      : null

    return (
      <div className="weather-card">
        <h3>Weather for {args.city}</h3>
        {result ? <p>{result.content[0].text}</p> : <p>Loading...</p>}
      </div>
    )
  }

  // Fall back to default renderer for standard tools
  return <ToolRenderer part={part} status={status} />
}

<AgentChat
  slots={{ ToolRenderer: MyToolRenderer }}
  // ... other props
/>
```

## MCP Tool Naming

MCP (Model Context Protocol) tools follow the naming pattern `mcp__<server>__<tool>`. The built-in `McpTool` renderer parses this automatically and shows the server name and tool name.

## Tool States

Each tool invocation has a state:

| State | Meaning |
|-------|---------|
| `"call"` | Tool has been called, waiting for execution |
| `"result"` | Tool has returned a result |

During streaming, tools may be in `"call"` state before transitioning to `"result"`.

## CSS Classes

All tool renderers have stable CSS class names for custom styling:

```css
.an-tool-bash { }
.an-tool-edit { }
.an-tool-write { }
.an-tool-search { }
.an-tool-todo { }
.an-tool-plan { }
.an-tool-task { }
.an-tool-mcp { }
.an-tool-thinking { }
.an-tool-generic { }
```
