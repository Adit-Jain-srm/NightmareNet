import type { SandboxConfig, SandboxOptions } from "./types"

export function Sandbox(config: SandboxOptions = {}): SandboxConfig {
  return { _type: "sandbox" as const, ...config }
}
