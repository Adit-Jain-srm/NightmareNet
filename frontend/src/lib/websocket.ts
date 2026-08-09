import { getApiBase } from "@/lib/api";

export type WsConnectionStatus = "connected" | "reconnecting" | "disconnected";

export const MAX_RECONNECT_ATTEMPTS = 10;

/** Exponential backoff with ±20% jitter, capped at 30s. */
export function nextBackoffMs(attempt: number, random: () => number = Math.random): number {
  const base = Math.min(1000 * Math.pow(2, attempt), 30_000);
  const jitter = base * (random() * 0.4 - 0.2);
  return Math.max(0, Math.round(base + jitter));
}

export function buildRunWsUrl(runId: string): string {
  const encoded = encodeURIComponent(runId);
  const apiBase = getApiBase();
  if (apiBase) {
    try {
      const u = new URL(apiBase);
      const wsProtocol = u.protocol === "https:" ? "wss:" : "ws:";
      const basePath = u.pathname.replace(/\/+$/, "");
      return `${wsProtocol}//${u.host}${basePath}/ws/runs/${encoded}`;
    } catch {
      // fall through to window location
    }
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/runs/${encoded}`;
}
