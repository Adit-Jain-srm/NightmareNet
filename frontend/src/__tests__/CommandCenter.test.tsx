import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { CommandCenter } from "@/components/dashboard/CommandCenter";
import * as hooks from "@/lib/hooks";

vi.mock("@/lib/hooks", () => ({
  useDemoMode: vi.fn(),
}));

vi.mock("framer-motion", () => {
  const createMotionMock = (tag: string) =>
    function MockMotionComponent(props: Record<string, unknown>) {
      const {
        children,
        whileHover,
        whileTap,
        initial,
        animate,
        exit,
        transition,
        variants,
        layout,
        ref,
        ...domProps
      } = props;
      void whileHover; void whileTap; void initial; void animate;
      void exit; void transition; void variants; void layout;
      return React.createElement(tag, { ...domProps, ref }, children as React.ReactNode);
    };

  return {
    motion: new Proxy({}, { get: (_t, prop: string) => createMotionMock(prop) }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
    HTMLMotionProps: {},
  };
});

describe("CommandCenter dashboard component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(hooks.useDemoMode).mockReturnValue({ isLive: true, isLoading: false });
  });

  it("renders component layout correctly with default props", () => {
    render(<CommandCenter />);

    expect(screen.getByText("Command Center")).toBeInTheDocument();
    expect(screen.getByText("System Pulse")).toBeInTheDocument();
    expect(screen.getByText(/Operational overview/i)).toBeInTheDocument();
    expect(screen.getByText(/Live runtime telemetry/i)).toBeInTheDocument();
  });

  it("displays the expected dashboard statistics and metric cards", () => {
    render(<CommandCenter />);

    expect(screen.getByText("Active Runs")).toBeInTheDocument();
    expect(screen.getByText("Total Experiments")).toBeInTheDocument();
    expect(screen.getAllByText("Robustness").length).toBeGreaterThan(0);
    expect(screen.getByText("Queue")).toBeInTheDocument();

    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("148")).toBeInTheDocument();
    expect(screen.getAllByText("82.4").length).toBeGreaterThan(0);
    expect(screen.getByText("5")).toBeInTheDocument();

    expect(screen.getByText("Cluster Utilization")).toBeInTheDocument();
    expect(screen.getByText("72%")).toBeInTheDocument();
    expect(screen.getByText("3.1GB / 4.0GB")).toBeInTheDocument();
    expect(screen.getByText("34%")).toBeInTheDocument();
  });

  it("renders the loading skeleton state when the loading prop is true", () => {
    render(<CommandCenter loading={true} />);

    expect(screen.getByText("Command Center")).toBeInTheDocument();
    expect(screen.getByText("System Pulse")).toBeInTheDocument();

    expect(screen.queryByText("Active Runs")).not.toBeInTheDocument();
    expect(screen.queryByText("Total Experiments")).not.toBeInTheDocument();
    expect(screen.queryByText("Cluster Utilization")).not.toBeInTheDocument();
  });

  it("renders the demo data badge when the API is not live", () => {
    vi.mocked(hooks.useDemoMode).mockReturnValue({ isLive: false, isLoading: false });
    render(<CommandCenter />);

    expect(screen.getByText("demo data · API offline")).toBeInTheDocument();
  });
});
