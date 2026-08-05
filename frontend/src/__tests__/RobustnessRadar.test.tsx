import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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

import { RobustnessRadar } from "@/components/dashboard/RobustnessRadar";

const mockSeries = [
  { label: "Hardened", color: "#22c55e", values: [86, 81, 84, 79, 72] },
  { label: "Baseline", color: "#ef4444", values: [62, 58, 71, 65, 41] },
];

describe("RobustnessRadar", () => {
  it("renders with mock robustness data across multiple axes", () => {
    render(<RobustnessRadar series={mockSeries} />);
    expect(screen.getByText("Robustness Radar")).toBeInTheDocument();
    expect(screen.getByLabelText("Robustness radar")).toBeInTheDocument();
    expect(screen.getByText("Hardened")).toBeInTheDocument();
    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("86")).toBeInTheDocument();
    expect(screen.getByText("62")).toBeInTheDocument();
  });

  it("shows empty state when there is no evaluation data", () => {
    render(<RobustnessRadar series={[]} />);
    expect(screen.getByText("No evaluation data")).toBeInTheDocument();
    expect(screen.getByText(/no robustness metrics to display/i)).toBeInTheDocument();
    expect(screen.queryByLabelText("Robustness radar")).not.toBeInTheDocument();
  });

  it("labels axes with the distortion type names", () => {
    render(<RobustnessRadar series={mockSeries} />);
    for (const label of ["Character", "Word", "Semantic", "Syntactic", "Attacks"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });
});
