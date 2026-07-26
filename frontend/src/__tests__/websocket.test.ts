import { describe, it, expect, vi, afterEach } from "vitest";
import { buildRunWsUrl, nextBackoffMs } from "@/lib/websocket";

describe("nextBackoffMs", () => {
  it("follows exponential steps before the cap", () => {
    expect(nextBackoffMs(0, () => 0.5)).toBe(1000);
    expect(nextBackoffMs(1, () => 0.5)).toBe(2000);
    expect(nextBackoffMs(2, () => 0.5)).toBe(4000);
    expect(nextBackoffMs(3, () => 0.5)).toBe(8000);
    expect(nextBackoffMs(4, () => 0.5)).toBe(16000);
  });

  it("caps at 30s", () => {
    expect(nextBackoffMs(5, () => 0.5)).toBe(30000);
    expect(nextBackoffMs(10, () => 0.5)).toBe(30000);
  });

  it("applies ±20% jitter", () => {
    expect(nextBackoffMs(0, () => 0)).toBe(800);
    expect(nextBackoffMs(0, () => 1)).toBe(1200);
  });
});

describe("buildRunWsUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("uses NEXT_PUBLIC_API_URL host when set", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.com");
    expect(buildRunWsUrl("abc")).toBe("wss://api.example.com/ws/runs/abc");
  });

  it("falls back to window.location", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "");
    vi.stubGlobal("window", {
      location: { protocol: "http:", host: "localhost:3000" },
    });
    expect(buildRunWsUrl("run-1")).toBe("ws://localhost:3000/ws/runs/run-1");
  });
});
