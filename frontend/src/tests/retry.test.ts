import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { withRetry, ApiError } from "../lib/retry";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockResponse(status: number, body: unknown = {}, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("withRetry", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns response immediately on success", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(200, { ok: true }));
    const res = await withRetry(fn);
    expect(res.status).toBe(200);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries on network error (TypeError) with exponential backoff", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(mockResponse(200));

    const onRetry = vi.fn();
    const promise = withRetry(fn, { baseDelayMs: 1000, onRetry });

    // Advance timers for each retry delay: 1s, 2s

    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(2000);

    const res = await promise;
    expect(res.status).toBe(200);
    expect(fn).toHaveBeenCalledTimes(3);
    expect(onRetry).toHaveBeenCalledTimes(2);
    expect(onRetry).toHaveBeenNthCalledWith(1, 1, 1000);
    expect(onRetry).toHaveBeenNthCalledWith(2, 2, 2000);
  });

  it("retries on 502, 503, 504", async () => {
    for (const status of [502, 503, 504]) {
      const fn = vi
        .fn()
        .mockResolvedValueOnce(mockResponse(status))
        .mockResolvedValue(mockResponse(200));

      const promise = withRetry(fn, { baseDelayMs: 100 });
      await vi.advanceTimersByTimeAsync(100);
      const res = await promise;

      expect(res.status).toBe(200);
      expect(fn).toHaveBeenCalledTimes(2);
      fn.mockReset();
    }
  });

  it("retries on 429 and respects Retry-After header", async () => {
    const fn = vi
      .fn()
      .mockResolvedValueOnce(
        mockResponse(429, { detail: "rate limited" }, { "Retry-After": "3" }),
      )
      .mockResolvedValue(mockResponse(200));

    const onRetry = vi.fn();
    const promise = withRetry(fn, { baseDelayMs: 1000, onRetry });

    // Retry-After says 3 seconds
    await vi.advanceTimersByTimeAsync(3000);
    const res = await promise;

    expect(res.status).toBe(200);
    expect(fn).toHaveBeenCalledTimes(2);
    expect(onRetry).toHaveBeenCalledWith(1, 3000);
  });

  it("does NOT retry on 400 client error", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(400, { detail: "bad request" }));
    await expect(withRetry(fn)).rejects.toThrow(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry on 401 unauthorized", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(401, { detail: "unauthorized" }));
    await expect(withRetry(fn)).rejects.toThrow(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry on 403 forbidden", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(403, { detail: "forbidden" }));
    await expect(withRetry(fn)).rejects.toThrow(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry on 404 not found", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(404, { detail: "not found" }));
    await expect(withRetry(fn)).rejects.toThrow(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry on 422 unprocessable entity", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(422, { detail: "validation error" }));
    await expect(withRetry(fn)).rejects.toThrow(ApiError);
    expect(fn).toHaveBeenCalledTimes(1);
  });

it("throws after max retries are exhausted", async () => {
    const fn = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    const promise = withRetry(fn, { maxRetries: 3, baseDelayMs: 100 });

    // Advance through all retry delays: 100ms, 200ms, 400ms
    await Promise.all([
      vi.advanceTimersByTimeAsync(700),
      expect(promise).rejects.toThrow("Failed to fetch"),
    ]);

    expect(fn).toHaveBeenCalledTimes(4);
  });

  it("ApiError carries status and response", async () => {
    const fn = vi.fn().mockResolvedValue(mockResponse(404, { detail: "not found" }));
    try {
      await withRetry(fn);
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(404);
      expect((e as ApiError).message).toBe("not found");
    }
  });
});