import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

describe("API module", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
    // Clear env var so tests use same-origin
    vi.stubEnv("NEXT_PUBLIC_API_URL", "");
  });

  describe("getApiBase", () => {
    it("returns empty string in browser context (same-origin)", async () => {
      // window is defined in jsdom
      const { getApiBase } = await import("@/lib/api");
      const result = getApiBase();
      expect(result).toBe("");
    });

    it("strips trailing slash from env var", async () => {
      vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:8000/");
      const { getApiBase } = await import("@/lib/api");
      const result = getApiBase();
      expect(result).not.toMatch(/\/$/);
    });
  });

  describe("getHealth", () => {
    it("calls /api/v1/health", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "healthy", version: "0.2.0" }),
      });

      const { getHealth } = await import("@/lib/api");
      const result = await getHealth();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/v1/health",
        expect.objectContaining({ headers: expect.any(Object) })
      );
      expect(result.status).toBe("healthy");
      expect(result.version).toBe("0.2.0");
    });

    it("throws on non-ok response", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: "Server error" }),
      });

      const { getHealth } = await import("@/lib/api");
      await expect(getHealth()).rejects.toThrow("Server error");
    });
  });

  describe("generateDream", () => {
    it("sends POST with correct body", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          original_text: "test",
          distorted_text: "tset",
          distortion_type: "dream",
          strength: 0.5,
          seed: null,
        }),
      });

      const { generateDream } = await import("@/lib/api");
      const result = await generateDream({ text: "test", strength: 0.5 });

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/v1/generate/dream",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ text: "test", strength: 0.5 }),
        })
      );
      expect(result.distortion_type).toBe("dream");
    });
  });

  describe("generateNightmare", () => {
    it("sends POST with correct body", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          original_text: "hello",
          distorted_text: "h3llo",
          distortion_type: "nightmare",
          strength: 0.8,
          seed: 42,
        }),
      });

      const { generateNightmare } = await import("@/lib/api");
      const result = await generateNightmare({ text: "hello", strength: 0.8, seed: 42 });

      expect(result.distortion_type).toBe("nightmare");
      expect(result.seed).toBe(42);
    });
  });

  describe("optimizeData", () => {
    it("posts to /api/v1/data/optimize", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "completed",
          run_id: "abc-123",
          optimized_count: 10,
        }),
      });

      const { optimizeData } = await import("@/lib/api");
      const result = await optimizeData({
        texts: ["hello", "world"],
        column_mapping: { text: "text" },
      });

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/v1/data/optimize",
        expect.objectContaining({ method: "POST" })
      );
      expect(result.status).toBe("completed");
    });
  });

  describe("suggestConfig", () => {
    it("posts to /api/v1/suggest/config", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          suggestions: [{ param: "lr", current: 0.01, suggested: 0.001, reason: "too high" }],
          model: "heuristic",
        }),
      });

      const { suggestConfig } = await import("@/lib/api");
      const result = await suggestConfig({
        current_config: { learning_rate: 0.01 },
      });

      expect(result.suggestions).toHaveLength(1);
      expect(result.model).toBe("heuristic");
    });
  });

  describe("searchExperiments", () => {
    it("posts natural-language queries to /api/v1/search", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          results: [{ run_id: "exp-47", relevance_score: 0.87, summary: "match", metadata: {} }],
          filters: { status: "completed" },
          backend: "faiss",
        }),
      });

      const { searchExperiments } = await import("@/lib/api");
      const result = await searchExperiments("robustness improved", 5, {
        status: "completed",
      });

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/v1/search",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            query: "robustness improved",
            top_k: 5,
            filters: { status: "completed" },
          }),
        })
      );
      expect(result.results[0].run_id).toBe("exp-47");
    });
  });

  describe("error handling", () => {
    it("extracts detail from error response", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({ detail: "Validation failed" }),
      });

      const { getHealth } = await import("@/lib/api");
      await expect(getHealth()).rejects.toThrow("Validation failed");
    });

    it("falls back to status code on non-JSON error", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => { throw new Error("not json"); },
      });

      const { getHealth } = await import("@/lib/api");
      await expect(getHealth()).rejects.toThrow("API error 503");
    });
  });
});
