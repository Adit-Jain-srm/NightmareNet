import { describe, it, expect, vi, beforeAll, afterEach } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";
import React from "react";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("@/lib/sounds", () => ({
  useSounds: () => ({
    playClick: vi.fn(),
    playSuccess: vi.fn(),
    playError: vi.fn(),
    playTransition: vi.fn(),
    playNotification: vi.fn(),
    enabled: true,
    toggle: vi.fn(),
  }),
}));

vi.mock("framer-motion", () => {
  const createMock = (tag: string) =>
    React.forwardRef(function MockMotion(
      { children, ...props }: Record<string, unknown>,
      ref: React.Ref<unknown>,
    ) {
      // Strip motion-specific props to avoid DOM warnings
      const {
        initial, animate, exit, transition, layout,
        whileHover, whileTap,
        ...domProps
      } = props as Record<string, unknown>;
      void initial; void animate; void exit; void transition;
      void layout; void whileHover; void whileTap;
      return React.createElement(tag, { ...domProps, ref }, children as React.ReactNode);
    });

  return {
    motion: { div: createMock("div") },
    AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

import { ToastProvider, useToast } from "@/components/ui/Toast";

/** Renders a ToastProvider with a trigger button that pushes a toast. */
function TestHarness({
  title,
  description,
  variant,
}: {
  title: string;
  description?: string;
  variant: "info" | "success" | "warning" | "error";
}) {
  const toast = useToast();
  return (
    <button
      onClick={() => toast.push({ title, description, variant })}
      data-testid="push-toast"
    >
      Push Toast
    </button>
  );
}

function renderWithProvider(
  title: string,
  variant: "info" | "success" | "warning" | "error",
  description?: string,
) {
  return render(
    <ToastProvider>
      <TestHarness title={title} description={description} variant={variant} />
    </ToastProvider>,
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LiveRegion + ToastProvider aria-live announcements", () => {
  beforeAll(() => {
    vi.useFakeTimers();
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((q: string) => ({
        matches: false,
        media: q,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.clearAllMocks();
  });

  it("renders a hidden aria-live region in the DOM", () => {
    renderWithProvider("Hello", "info");
    const region = document.getElementById("toast-live-region");
    expect(region).not.toBeNull();
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveAttribute("aria-atomic", "true");
    expect(region).toHaveClass("sr-only");
  });

  it("announces toast title after debounce when toast is pushed", async () => {
    renderWithProvider("Export complete", "success");

    fireEvent.click(screen.getByTestId("push-toast"));
    // Advance past the 150 ms debounce
    act(() => { vi.advanceTimersByTime(200); });

    const region = document.getElementById("toast-live-region");
    expect(region?.textContent).toBe("Export complete");
  });

  it("includes description in announcement when provided", async () => {
    renderWithProvider("Pipeline started", "info", "Run will take ~10 minutes");

    fireEvent.click(screen.getByTestId("push-toast"));
    act(() => { vi.advanceTimersByTime(200); });

    const region = document.getElementById("toast-live-region");
    expect(region?.textContent).toBe("Pipeline started: Run will take ~10 minutes");
  });

  it("uses role=alert and aria-live=assertive for error toasts", async () => {
    renderWithProvider("API Error", "error", "Connection refused");

    fireEvent.click(screen.getByTestId("push-toast"));
    act(() => { vi.advanceTimersByTime(200); });

    const region = document.getElementById("toast-live-region");
    expect(region).toHaveAttribute("role", "alert");
    expect(region).toHaveAttribute("aria-live", "assertive");
    expect(region?.textContent).toBe("API Error: Connection refused");
  });

  it("uses role=status and aria-live=polite for non-error toasts", async () => {
    renderWithProvider("Re-run queued", "success");

    fireEvent.click(screen.getByTestId("push-toast"));
    act(() => { vi.advanceTimersByTime(200); });

    const region = document.getElementById("toast-live-region");
    expect(region).toHaveAttribute("role", "status");
    expect(region).toHaveAttribute("aria-live", "polite");
  });

  it("debounces rapid pushes — only the latest message is announced", async () => {
    function MultiPush() {
      const toast = useToast();
      return (
        <button
          onClick={() => {
            toast.push({ title: "First", variant: "info" });
            toast.push({ title: "Second", variant: "info" });
            toast.push({ title: "Third", variant: "success" });
          }}
          data-testid="multi-push"
        >
          Push Many
        </button>
      );
    }

    render(
      <ToastProvider>
        <MultiPush />
      </ToastProvider>,
    );

    fireEvent.click(screen.getByTestId("multi-push"));
    // Before debounce — region should still be empty (debounce pending)
    const regionBefore = document.getElementById("toast-live-region");
    expect(regionBefore?.textContent).toBe("");

    // After debounce — only the last message
    act(() => { vi.advanceTimersByTime(200); });
    const regionAfter = document.getElementById("toast-live-region");
    expect(regionAfter?.textContent).toBe("Third");
  });
});
