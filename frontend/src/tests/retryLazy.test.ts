import React, { Suspense } from "react";
import { render, screen, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { retryLazy } from "../lib/retryLazy";

class TestErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ||
        React.createElement("div", null, `Error: ${this.state.error?.message}`)
      );
    }
    return this.props.children;
  }
}

describe("retryLazy", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders component immediately when import succeeds on first attempt", async () => {
    const DummyComponent = () => React.createElement("div", null, "Dashboard Panel Content");
    const factory = vi.fn().mockResolvedValue({ default: DummyComponent });

    const LazyPanel = retryLazy(factory);

    render(
      React.createElement(
        Suspense,
        { fallback: React.createElement("div", null, "Loading...") },
        React.createElement(LazyPanel),
      ),
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByText("Dashboard Panel Content")).toBeInTheDocument();
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it("retries loading dynamic import when it fails initially and succeeds on retry", async () => {
    const DummyComponent = () => React.createElement("div", null, "Recovered Panel");
    const factory = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network flake 1"))
      .mockRejectedValueOnce(new Error("Network flake 2"))
      .mockResolvedValue({ default: DummyComponent });

    const LazyPanel = retryLazy(factory, 3, 1000, 2);

    render(
      React.createElement(
        Suspense,
        { fallback: React.createElement("div", null, "Loading...") },
        React.createElement(LazyPanel),
      ),
    );

    // Initial attempt
    await act(async () => {
      await Promise.resolve();
    });
    expect(factory).toHaveBeenCalledTimes(1);

    // 1st retry (1000ms)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(factory).toHaveBeenCalledTimes(2);

    // 2nd retry (2000ms)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(factory).toHaveBeenCalledTimes(3);

    expect(screen.getByText("Recovered Panel")).toBeInTheDocument();
  });

  it("surfaces error after all retries are exhausted", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const factory = vi.fn().mockRejectedValue(new Error("Permanent CDN Failure"));

    const LazyPanel = retryLazy(factory, 3, 1000, 2);

    render(
      React.createElement(
        TestErrorBoundary,
        { fallback: React.createElement("div", null, "Boundary caught: Permanent CDN Failure") },
        React.createElement(
          Suspense,
          { fallback: React.createElement("div", null, "Loading...") },
          React.createElement(LazyPanel),
        ),
      ),
    );

    // Initial attempt
    await act(async () => {
      await Promise.resolve();
    });

    // 1st retry (1000ms)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });

    // 2nd retry (2000ms)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    // 3rd retry (4000ms)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });

    expect(factory).toHaveBeenCalledTimes(4);
    expect(screen.getByText("Boundary caught: Permanent CDN Failure")).toBeInTheDocument();

    consoleError.mockRestore();
  });
});
