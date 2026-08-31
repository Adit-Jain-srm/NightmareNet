import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
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
  return {
    motion: new Proxy(
      {},
      {
        get: (_target, property: string | symbol) =>
          typeof property === "string" ? mock(property) : undefined,
      },
    ),
  };
});

vi.mock("@/lib/hooks", () => ({
  useDemoMode: () => ({ isLive: false, isLoading: false }),
}));

import { RobustnessRadar } from "@/components/dashboard/RobustnessRadar";

const AXIS_COUNT = 5;
const SIZE = 280;
const CX = SIZE / 2;
const CY = SIZE / 2;
const RADIUS = SIZE / 2 - 32;

/** Mirror RobustnessRadar polygon geometry so tests lock the SVG `points` contract. */
function expectedPolygonPoints(values: number[]): string {
  return values
    .map((v, i) => {
      const angle = (i / AXIS_COUNT) * Math.PI * 2 - Math.PI / 2;
      const x = CX + Math.cos(angle) * RADIUS * (v / 100);
      const y = CY + Math.sin(angle) * RADIUS * (v / 100);
      return `${x},${y}`;
    })
    .join(" ");
}

function vertexCount(points: string | null): number {
  if (!points?.trim()) return 0;
  return points.trim().split(/\s+/).length;
}

const mockSeries = [
  { label: "Hardened", color: "#22c55e", values: [86, 81, 84, 79, 72] },
  { label: "Baseline", color: "#ef4444", values: [62, 58, 71, 65, 41] },
];

describe("RobustnessRadar", () => {
  it("renders with mock robustness data across multiple axes", () => {
    render(<RobustnessRadar series={mockSeries} />);
    const chart = screen.getByRole("group", { name: "Robustness radar" });

    expect(screen.getByText("Robustness Radar")).toBeInTheDocument();
    expect(chart).toBeInTheDocument();
    expect(chart.querySelector('polygon[fill="#22c55e"]')).not.toBeNull();
    expect(chart.querySelector('polygon[fill="#ef4444"]')).not.toBeNull();
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
    const chart = screen.getByRole("group", { name: "Robustness radar" });
    for (const label of ["Character", "Word", "Semantic", "Syntactic", "Attacks"]) {
      expect(within(chart).getByText(label)).toBeInTheDocument();
    }
  });

  it("renders series polygons with geometry derived from the series prop", () => {
    const { rerender } = render(<RobustnessRadar series={mockSeries} />);
    const chart = screen.getByRole("group", { name: "Robustness radar" });

    expect(chart.getAttribute("viewBox")).toBe(`0 0 ${SIZE} ${SIZE}`);
    expect(chart.getAttribute("width")).toBe(String(SIZE));
    expect(chart.getAttribute("height")).toBe(String(SIZE));

    const hardened = chart.querySelector('polygon[fill="#22c55e"]');
    const baseline = chart.querySelector('polygon[fill="#ef4444"]');
    expect(hardened).not.toBeNull();
    expect(baseline).not.toBeNull();

    const hardenedPoints = hardened!.getAttribute("points");
    const baselinePoints = baseline!.getAttribute("points");
    expect(hardenedPoints).toBe(expectedPolygonPoints(mockSeries[0].values));
    expect(baselinePoints).toBe(expectedPolygonPoints(mockSeries[1].values));
    expect(vertexCount(hardenedPoints)).toBe(AXIS_COUNT);
    expect(vertexCount(baselinePoints)).toBe(AXIS_COUNT);

    // Regression guard: ignoring `series` (hardcoded SERIES) would keep these points.
    const altSeries = [
      { label: "Hardened", color: "#22c55e", values: [10, 20, 30, 40, 50] },
      { label: "Baseline", color: "#ef4444", values: [90, 80, 70, 60, 55] },
    ];
    rerender(<RobustnessRadar series={altSeries} />);
    const chartAlt = screen.getByRole("group", { name: "Robustness radar" });
    const hardenedAlt = chartAlt.querySelector('polygon[fill="#22c55e"]')!.getAttribute("points");
    const baselineAlt = chartAlt.querySelector('polygon[fill="#ef4444"]')!.getAttribute("points");

    expect(hardenedAlt).toBe(expectedPolygonPoints(altSeries[0].values));
    expect(baselineAlt).toBe(expectedPolygonPoints(altSeries[1].values));
    expect(hardenedAlt).not.toBe(hardenedPoints);
    expect(baselineAlt).not.toBe(baselinePoints);
    expect(vertexCount(hardenedAlt)).toBe(AXIS_COUNT);
  });
});
