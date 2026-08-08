import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

vi.mock("framer-motion", () => {
  const mock = (tag: string) =>
    function M(props: Record<string, unknown>) {
      const { children, animate, initial, transition, whileHover, whileTap, exit, variants, layout, ref, ...rest } =
        props;
      void initial; void transition; void whileHover; void whileTap; void exit; void variants; void layout;
      return React.createElement(
        tag,
        { ...rest, ref, "data-animate": animate != null ? JSON.stringify(animate) : undefined },
        children as React.ReactNode,
      );
    };
  return {
    motion: new Proxy({}, { get: (_t, p: string) => mock(p) }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
  };
});

vi.mock("@/components/dashboard/PipelineGraph", () => ({
  PipelineGraph: () => <div data-testid="pipeline-graph" />,
}));

import { PhaseVisualizer } from "@/components/dashboard/PhaseVisualizer";

describe("PhaseVisualizer", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("renders all four phases", () => {
    render(<PhaseVisualizer activePhase={0} />);
    expect(screen.getByText("Phase Visualizer")).toBeInTheDocument();
    for (const label of ["Wake", "Dream", "Nightmare", "Compress"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("highlights the active phase with a thicker arc", () => {
    const { container } = render(<PhaseVisualizer activePhase={2} />);
    const arcs = container.querySelectorAll("path[data-animate]");
    expect(arcs).toHaveLength(4);
    expect(arcs[2].getAttribute("stroke-width")).toBe("14");
    expect(arcs[0].getAttribute("stroke-width")).toBe("8");
  });

  it("animates arcs with full opacity on the active phase", () => {
    const { container } = render(<PhaseVisualizer activePhase={1} />);
    const opacities = [...container.querySelectorAll("path[data-animate]")].map(
      (el) => JSON.parse(el.getAttribute("data-animate")!).opacity,
    );
    expect(opacities[1]).toBe(1);
    expect(opacities.filter((_, i) => i !== 1).every((o) => o === 0.5)).toBe(true);
  });

  it("accepts activePhase and updates the center label", () => {
    const { rerender } = render(<PhaseVisualizer activePhase={1} />);
    expect(screen.getAllByText("Dream").length).toBeGreaterThan(0);
    rerender(<PhaseVisualizer activePhase={3} />);
    expect(screen.getAllByText("Compress").length).toBeGreaterThan(0);
    rerender(<PhaseVisualizer activePhase={0} />);
    expect(screen.getAllByText("Wake").length).toBeGreaterThan(0);
  });
});
