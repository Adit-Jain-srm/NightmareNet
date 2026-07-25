import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useVisitTracker } from "./useVisitTracker";
import { getSidebarPrefs, clearSidebarPrefs } from "../lib/storage";

describe("useVisitTracker", () => {
  const VALID_KEYS = ["home", "settings", "profile"];

  beforeEach(() => {
    localStorage.clear();
    clearSidebarPrefs();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should initialize with default state", () => {
    const { result } = renderHook(() => useVisitTracker(VALID_KEYS));

    expect(result.current.isLoaded).toBe(true);
    expect(result.current.totalVisits).toBe(0);
    expect(result.current.visitCounts).toEqual({});
    expect(result.current.customOrder).toEqual([]);
  });

  it("should track valid visits", () => {
    const { result } = renderHook(() => useVisitTracker(VALID_KEYS));

    act(() => {
      result.current.registerVisit("home");
    });

    expect(result.current.visitCounts["home"]).toBe(1);
    expect(result.current.totalVisits).toBe(1);

    act(() => {
      result.current.registerVisit("home");
      result.current.registerVisit("settings");
    });

    expect(result.current.visitCounts["home"]).toBe(2);
    expect(result.current.visitCounts["settings"]).toBe(1);
    expect(result.current.totalVisits).toBe(3);
  });

  it("should ignore invalid visits", () => {
    const { result } = renderHook(() => useVisitTracker(VALID_KEYS));

    act(() => {
      result.current.registerVisit("invalid-route");
    });

    expect(result.current.visitCounts["invalid-route"]).toBeUndefined();
    expect(result.current.totalVisits).toBe(0);
  });

  it("should support setting custom order", () => {
    const { result } = renderHook(() => useVisitTracker(VALID_KEYS));

    act(() => {
      result.current.setCustomOrder(["settings", "home", "profile"]);
    });

    expect(result.current.customOrder).toEqual(["settings", "home", "profile"]);
  });

  it("should reset state to defaults", () => {
    const { result } = renderHook(() => useVisitTracker(VALID_KEYS));

    act(() => {
      result.current.registerVisit("home");
      result.current.setCustomOrder(["profile", "home", "settings"]);
    });

    expect(result.current.totalVisits).toBe(1);
    expect(result.current.customOrder.length).toBe(3);

    act(() => {
      result.current.reset();
    });

    expect(result.current.totalVisits).toBe(0);
    expect(result.current.customOrder).toEqual([]);
    expect(result.current.visitCounts).toEqual({});
  });

  it("should handle custom reset event", () => {
    const { result } = renderHook(() => useVisitTracker(VALID_KEYS));

    act(() => {
      result.current.registerVisit("home");
    });

    expect(result.current.totalVisits).toBe(1);

    act(() => {
      window.dispatchEvent(new CustomEvent("reset-sidebar-prefs"));
    });

    expect(result.current.totalVisits).toBe(0);
  });

  it("storage migration should handle empty or invalid data gracefully", () => {
    localStorage.setItem("nightmare_sidebar_prefs", JSON.stringify({ invalid: "data" }));
    const prefs = getSidebarPrefs();
    expect(prefs.version).toBe(1);
    expect(prefs.visitCounts).toEqual({});
    expect(prefs.customOrder).toEqual([]);
  });
});
