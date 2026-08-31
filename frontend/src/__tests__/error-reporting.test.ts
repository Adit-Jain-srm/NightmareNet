import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const DSN = "https://public@o123.ingest.sentry.io/456";

describe("error-reporting", () => {
  let reportError: typeof import("@/lib/error-reporting").reportError;
  let initErrorReporting: typeof import("@/lib/error-reporting").initErrorReporting;

  beforeEach(async () => {
    vi.resetModules();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    vi.stubGlobal("sessionStorage", {
      getItem: vi.fn(() => null),
      setItem: vi.fn(),
    });
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { pathname: "/dashboard", search: "?tab=runs" },
    });
    const mod = await import("@/lib/error-reporting");
    reportError = mod.reportError;
    initErrorReporting = mod.initErrorReporting;
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("logs to console in development without calling fetch", () => {
    vi.stubEnv("NODE_ENV", "development");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", DSN);
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    reportError(new Error("panel blew up"), { source: "ErrorBoundary" });

    expect(consoleError).toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("no-ops in production when DSN is missing", () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "");
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);

    reportError(new Error("silent fail"));

    expect(fetch).not.toHaveBeenCalled();
    expect(consoleError).not.toHaveBeenCalled();
  });

  it("posts to the Sentry store endpoint in production", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", DSN);

    reportError(new Error("prod crash"), {
      source: "ErrorBoundary",
      componentStack: "at Panel",
    });

    await vi.waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });

    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("https://o123.ingest.sentry.io/api/456/store/");
    expect((init as RequestInit).headers).toMatchObject({
      "X-Sentry-Auth": expect.stringContaining("sentry_key=public"),
    });
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body.message.formatted).toBe("prod crash");
    expect(body.extra.route).toBe("/dashboard?tab=runs");
    expect(body.extra.component_stack).toBe("at Panel");
  });

  it("rate limits reports to ten per minute", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", DSN);

    for (let i = 0; i < 12; i++) {
      reportError(new Error(`err-${i}`));
    }

    await vi.waitFor(() => {
      expect((fetch as ReturnType<typeof vi.fn>).mock.calls.length).toBe(10);
    });
  });

  it("registers an unhandledrejection handler once", () => {
    const addSpy = vi.spyOn(window, "addEventListener");

    initErrorReporting();
    initErrorReporting();

    expect(addSpy).toHaveBeenCalledWith("unhandledrejection", expect.any(Function));
    expect(addSpy.mock.calls.filter(([evt]) => evt === "unhandledrejection")).toHaveLength(1);
  });

  it("does not throw when the network request fails", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", DSN);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    expect(() => reportError(new Error("still safe"))).not.toThrow();
    await vi.waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });
  });
});
