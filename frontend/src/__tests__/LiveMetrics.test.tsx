// frontend/src/__tests__/LiveMetrics.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";

vi.mock("framer-motion", () => {
  const mock = (tag: string) =>
    function Motion(props: Record<string, unknown>) {
      const { children, initial, animate, transition, ...rest } = props;
      void initial;
      void animate;
      void transition;
      return React.createElement(tag, rest, children as React.ReactNode);
    };
  return { motion: new Proxy({}, { get: (_t, p: string) => mock(p) }) };
});

vi.mock("@/lib/hooks", () => ({
  useDemoMode: () => ({ isLive: false, isLoading: false }),
}));

import { LiveMetrics } from "@/components/dashboard/LiveMetrics";

describe("LiveMetrics", () => {
  it("renders with demo metric data and the loss chart", () => {
    render(<LiveMetrics />);
    expect(screen.getByText("Live Metrics")).toBeInTheDocument();
    expect(screen.getByLabelText("Loss curve")).toBeInTheDocument();
    expect(screen.getByText("Throughput")).toBeInTheDocument();
    expect(screen.getByText("1.2 k/s")).toBeInTheDocument();
  });

  it("shows chart phase labels and y-axis ticks", () => {
    render(<LiveMetrics />);
    for (const phase of ["wake", "dream", "nightmare", "compress"]) {
      expect(screen.getByText(phase)).toBeInTheDocument();
    }
    // LOSS_SERIES spans 0.82–2.41; ticks use toFixed(2)
    expect(screen.getByText("0.82")).toBeInTheDocument();
    expect(screen.getByText("2.41")).toBeInTheDocument();
  });

  it("handles empty data via the loading skeleton", () => {
    render(<LiveMetrics loading />);
    expect(screen.queryByLabelText("Loss curve")).not.toBeInTheDocument();
    expect(screen.queryByText("Throughput")).not.toBeInTheDocument();
  });

  it("updates when data arrives and when the robustness tab is selected", () => {
    const { rerender } = render(<LiveMetrics loading />);
    rerender(<LiveMetrics />);
    expect(screen.getByLabelText("Loss curve")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Robustness" }));
    expect(screen.getByLabelText("Robustness chart")).toBeInTheDocument();
    expect(screen.getByText("Hardened")).toBeInTheDocument();
    expect(screen.getByText("0.1")).toBeInTheDocument();
  });
});
