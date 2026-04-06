# CLI Reference

`@21st-sdk/cli` provides the `an` command for deploying agents.

## Install

```bash
npm install -g @21st-sdk/cli
# or use npx
npx @21st-sdk/cli <command>
```

## Commands

### `@21st-sdk/cli login`

Authenticate with the AN platform.

```bash
npx @21st-sdk/cli login
# Enter your API key: an_sk_...
# Authenticated as John (team: my-team)
```

Your key is saved to `~/.an/credentials`.

### `@21st-sdk/cli deploy`

Bundle and deploy your agent.

```bash
npx @21st-sdk/cli deploy
# Bundling src/agent.ts...
# Bundled (12.3kb)
# Deploying my-agent...
# https://api.an.dev/v1/chat/my-agent
```

The CLI:
1. Finds your entry point (see detection order below)
2. Bundles your code + dependencies with esbuild
3. Deploys to a secure cloud sandbox
4. Returns your agent's URL

## Entry Point Detection

The CLI looks for your agent file in this order:

1. `src/agent.ts`
2. `src/index.ts`
3. `agent.ts`
4. `index.ts`

Your entry file must `export default agent(...)`.

## Project Linking

After first deploy, the CLI saves `.an/project.json` in your project directory:

```json
{
  "agentId": "abc123",
  "slug": "my-agent"
}
```

Subsequent deploys update the existing agent.

## Bundling

The CLI uses esbuild to bundle your agent code:

- Target: Node 22, ESM
- `@21st-sdk/agent` is externalized (provided by the sandbox runtime)
- All other dependencies are bundled into a single file

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `~/.an/credentials` | Global | API key (`{ "apiKey": "an_sk_..." }`) |
| `.an/project.json` | Per-project | Agent ID and slug for redeployment |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL_21ST` | `https://an.dev/api/v1` | Override API endpoint |
