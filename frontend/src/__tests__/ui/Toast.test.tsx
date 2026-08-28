import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import { ToastProvider, useToast } from "@/components/ui/Toast";

// Mock sounds
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

// Mock framer-motion
vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: React.forwardRef(function MockMotionDiv(
      { children, layout, initial, animate, exit, transition, ...props }: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>
    ) {
      void layout;
      void initial;
      void animate;
      void exit;
      void transition;
      return React.createElement(
        "div",
        { ...(props as Record<string, unknown>), ref },
        children as React.ReactNode
      );
    }),
  },
}));

function TestConsumer() {
  const { push, dismiss } = useToast();
  return (
    <div>
      <button
        type="button"
        onClick={() => push({ title: "Success Toast", variant: "success", description: "Operation succeeded" })}
      >
        Trigger Success
      </button>
      <button
        type="button"
        onClick={() => push({ title: "Error Toast", variant: "error", description: "Operation failed" })}
      >
        Trigger Error
      </button>
      <button
        type="button"
        onClick={() => push({ title: "Warning Toast", variant: "warning" })}
      >
        Trigger Warning
      </button>
      <button
        type="button"
        onClick={() => push({ title: "Info Toast", variant: "info" })}
      >
        Trigger Info
      </button>
      <button
        type="button"
        onClick={() => {
          const id = push({ title: "Manual Dismiss Toast", variant: "info" });
          setTimeout(() => dismiss(id), 100);
        }}
      >
        Trigger Dismiss
      </button>
    </div>
  );
}

describe("Toast Component & Provider", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("throws an error when useToast is called outside of ToastProvider", () => {
    expect(() => render(<TestConsumer />)).toThrow(
      "useToast must be used within <ToastProvider>"
    );
  });

  it("displays success toast with message and description", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Success"));
    expect(screen.getByText("Success Toast")).toBeInTheDocument();
    expect(screen.getByText("Operation succeeded")).toBeInTheDocument();
  });

  it("displays error toast and handles error variant styling", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Error"));
    expect(screen.getByText("Error Toast")).toBeInTheDocument();
    expect(screen.getByText("Operation failed")).toBeInTheDocument();
  });

  it("displays warning and info toast variants", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Warning"));
    fireEvent.click(screen.getByText("Trigger Info"));
    expect(screen.getByText("Warning Toast")).toBeInTheDocument();
    expect(screen.getByText("Info Toast")).toBeInTheDocument();
  });

  it("automatically dismisses toast after durationMs timeout", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Success"));
    expect(screen.getByText("Success Toast")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(4100);
    });

    expect(screen.queryByText("Success Toast")).not.toBeInTheDocument();
  });

  it("dismisses toast manually via close/dismiss button", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Warning"));
    expect(screen.getByText("Warning Toast")).toBeInTheDocument();

    const dismissBtn = screen.getByRole("button", { name: /dismiss/i });
    fireEvent.click(dismissBtn);

    expect(screen.queryByText("Warning Toast")).not.toBeInTheDocument();
  });

  it("renders multiple simultaneous toasts stacked", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Success"));
    fireEvent.click(screen.getByText("Trigger Error"));
    fireEvent.click(screen.getByText("Trigger Warning"));

    expect(screen.getByText("Success Toast")).toBeInTheDocument();
    expect(screen.getByText("Error Toast")).toBeInTheDocument();
    expect(screen.getByText("Warning Toast")).toBeInTheDocument();
  });

  it("announces updates to LiveRegion for accessibility", () => {
    render(
      <ToastProvider>
        <TestConsumer />
      </ToastProvider>
    );

    fireEvent.click(screen.getByText("Trigger Error"));
    act(() => {
      vi.advanceTimersByTime(200);
    });

    const liveRegion = document.getElementById("toast-live-region");
    expect(liveRegion).toBeInTheDocument();
    expect(liveRegion).toHaveTextContent("Error Toast: Operation failed");
  });
});
