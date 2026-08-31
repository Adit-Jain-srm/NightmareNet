import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { ExperimentListSkeleton } from "@/components/dashboard/skeletons/ExperimentListSkeleton";
import { RunDetailSkeleton } from "@/components/dashboard/skeletons/RunDetailSkeleton";
import { PipelineGraphSkeleton } from "@/components/dashboard/skeletons/PipelineGraphSkeleton";
import { ModelComparisonSkeleton } from "@/components/dashboard/skeletons/ModelComparisonSkeleton";

describe("ExperimentListSkeleton", () => {
  it("renders a busy status container with a sr-only loading label", () => {
    render(<ExperimentListSkeleton />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Loading experiments")).toBeInTheDocument();
  });

  it("renders one placeholder row per requested row count", () => {
    const { container } = render(<ExperimentListSkeleton rows={3} />);
    expect(container.querySelectorAll("li")).toHaveLength(3);
  });
});

describe("RunDetailSkeleton", () => {
  it("renders a busy status container with a sr-only loading label", () => {
    render(<RunDetailSkeleton />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Loading run detail")).toBeInTheDocument();
  });

  it("renders placeholders for the four phase timeline nodes and tabs", () => {
    const { container } = render(<RunDetailSkeleton />);
    const timeline = container.querySelector(".relative.flex.items-center.justify-between");
    expect(
      timeline?.querySelectorAll(":scope > .relative.flex.flex-col.items-center"),
    ).toHaveLength(4);
  });
});

describe("PipelineGraphSkeleton", () => {
  it("renders a busy status container with a sr-only loading label", () => {
    render(<PipelineGraphSkeleton />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Loading phase visualizer")).toBeInTheDocument();
  });

  it("renders placeholders for the four orbiting phase nodes and the legend", () => {
    const { container } = render(<PipelineGraphSkeleton />);
    expect(container.querySelectorAll(".rounded-full")).not.toHaveLength(0);
  });
});

describe("ModelComparisonSkeleton", () => {
  it("renders a busy status container with a sr-only loading label", () => {
    render(<ModelComparisonSkeleton />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Loading model comparison")).toBeInTheDocument();
  });

  it("renders the A/B metric comparison rows", () => {
    const { container } = render(<ModelComparisonSkeleton />);
    expect(
      container.querySelectorAll(".grid-cols-\\[110px_1fr_60px_1fr_60px\\]"),
    ).toHaveLength(4);
  });
});
