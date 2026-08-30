import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import ResilienceLab from "@/components/ResilienceLab";
import * as api from "@/lib/api";

// Mock Framer Motion
vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: React.forwardRef(function MockMotionDiv(
      { children, initial, animate, exit, transition, whileInView, viewport, layoutId, ...props }: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>
    ) {
      void initial;
      void animate;
      void exit;
      void transition;
      void whileInView;
      void viewport;
      void layoutId;
      return React.createElement(
        "div",
        { ...(props as Record<string, unknown>), ref },
        children as React.ReactNode
      );
    }),
    circle: React.forwardRef(function MockMotionCircle(
      { initial, animate, transition, ...props }: Record<string, unknown>,
      ref: React.Ref<SVGCircleElement>
    ) {
      void initial;
      void animate;
      void transition;
      return React.createElement("circle", { ...(props as Record<string, unknown>), ref });
    }),
  },
}));

// Mock Lucide icons
vi.mock("lucide-react", () => ({
  GitCompareArrows: (props: Record<string, unknown>) => <span data-testid="icon-git-compare" {...props} />,
  Shield: (props: Record<string, unknown>) => <span data-testid="icon-shield" {...props} />,
  Loader2: (props: Record<string, unknown>) => <span data-testid="icon-loader" {...props} />,
  BarChart3: (props: Record<string, unknown>) => <span data-testid="icon-barchart" {...props} />,
  TrendingDown: (props: Record<string, unknown>) => <span data-testid="icon-trending-down" {...props} />,
  ArrowLeftRight: (props: Record<string, unknown>) => <span data-testid="icon-arrow-left-right" {...props} />,
}));

// Mock HTMLCanvasElement getContext
HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  clearRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fillText: vi.fn(),
  closePath: vi.fn(),
  createLinearGradient: vi.fn().mockReturnValue({
    addColorStop: vi.fn(),
  }),
  fill: vi.fn(),
  arc: vi.fn(),
  scale: vi.fn(),
}) as unknown as typeof HTMLCanvasElement.prototype.getContext;

HTMLCanvasElement.prototype.getBoundingClientRect = vi.fn().mockReturnValue({
  width: 600,
  height: 240,
  top: 0,
  left: 0,
  right: 600,
  bottom: 240,
});

describe("ResilienceLab Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders initial state with heading, description, textarea, and tabs", () => {
    render(<ResilienceLab />);

    expect(screen.getByText("Resilience")).toBeInTheDocument();
    expect(screen.getByText("Lab")).toBeInTheDocument();
    expect(screen.getByText(/Measure how text degrades under distortion/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /quick compare/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /full spectrum/i })).toBeInTheDocument();

    const textarea = screen.getByRole("textbox", { name: /test text for resilience comparison/i });
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveValue(
      "Attention mechanisms enable transformers to capture long-range dependencies."
    );

    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("Challenge")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /compare/i })).toBeInTheDocument();
  });

  it("allows editing the input test text", () => {
    render(<ResilienceLab />);

    const textarea = screen.getByRole("textbox", { name: /test text for resilience comparison/i });
    fireEvent.change(textarea, { target: { value: "A new prompt to test resilience." } });

    expect(textarea).toHaveValue("A new prompt to test resilience.");
  });

  it("switches to Full Spectrum tab and back to Quick Compare", () => {
    render(<ResilienceLab />);

    const spectrumTab = screen.getByRole("button", { name: /full spectrum/i });
    fireEvent.click(spectrumTab);

    expect(screen.getByText(/Testing at 9 strength levels/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /evaluate/i })).toBeInTheDocument();

    const compareTab = screen.getByRole("button", { name: /quick compare/i });
    fireEvent.click(compareTab);

    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("Challenge")).toBeInTheDocument();
  });

  it("updates baseline and challenge strength sliders", () => {
    render(<ResilienceLab />);

    const baselineSlider = screen.getByLabelText("Baseline distortion strength");
    fireEvent.change(baselineSlider, { target: { value: "0.35" } });
    expect(screen.getByText("0.35")).toBeInTheDocument();

    const challengeSlider = screen.getByLabelText("Challenge distortion strength");
    fireEvent.change(challengeSlider, { target: { value: "0.85" } });
    expect(screen.getByText("0.85")).toBeInTheDocument();
  });

  it("executes compare API call and displays resilience score and metrics", async () => {
    const mockCompareResponse: api.CompareResponse = {
      resilience_score: 0.88,
      dream: {
        baseline: { text: "clean dream", similarity: 0.95 },
        challenge: { text: "distorted dream", similarity: 0.82 },
      },
      nightmare: {
        baseline: { text: "clean nightmare", similarity: 0.92 },
        challenge: { text: "adversarial nightmare", similarity: 0.74 },
      },
      analysis: "High resilience across syntax distortions.",
    };

    vi.spyOn(api, "compareDistortions").mockResolvedValueOnce(mockCompareResponse);

    render(<ResilienceLab />);

    const compareBtn = screen.getByRole("button", { name: /compare/i });
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(api.compareDistortions).toHaveBeenCalledWith({
        text: "Attention mechanisms enable transformers to capture long-range dependencies.",
        baseline_strength: 0.2,
        challenge_strength: 0.7,
      });
    });

    expect(await screen.findByText("88%")).toBeInTheDocument();
    expect(screen.getByText("Resilience Score")).toBeInTheDocument();
    expect(screen.getByText("High resilience across syntax distortions.")).toBeInTheDocument();
    expect(screen.getByText("95.0%")).toBeInTheDocument();
    expect(screen.getByText("74.0%")).toBeInTheDocument();
  });

  it("executes spectrum evaluate API call and renders curve summary", async () => {
    const mockRobustnessResponse: api.RobustnessResponse = {
      scores: {
        dream: {
          "0.1": { similarity: 0.98 },
          "0.2": { similarity: 0.94 },
          "0.3": { similarity: 0.90 },
          "0.4": { similarity: 0.86 },
          "0.5": { similarity: 0.82 },
          "0.6": { similarity: 0.78 },
          "0.7": { similarity: 0.74 },
          "0.8": { similarity: 0.70 },
          "0.9": { similarity: 0.65 },
        },
        nightmare: {
          "0.1": { similarity: 0.95 },
          "0.2": { similarity: 0.90 },
          "0.3": { similarity: 0.84 },
          "0.4": { similarity: 0.78 },
          "0.5": { similarity: 0.72 },
          "0.6": { similarity: 0.66 },
          "0.7": { similarity: 0.60 },
          "0.8": { similarity: 0.54 },
          "0.9": { similarity: 0.48 },
        },
      },
      summary: "Evaluated across 9 strengths with smooth degradation curve.",
    };

    vi.spyOn(api, "evaluateRobustness").mockResolvedValueOnce(mockRobustnessResponse);

    render(<ResilienceLab />);

    const spectrumTab = screen.getByRole("button", { name: /full spectrum/i });
    fireEvent.click(spectrumTab);

    const evaluateBtn = screen.getByRole("button", { name: /evaluate/i });
    fireEvent.click(evaluateBtn);

    await waitFor(() => {
      expect(api.evaluateRobustness).toHaveBeenCalledWith({
        text: "Attention mechanisms enable transformers to capture long-range dependencies.",
        strengths: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      });
    });

    expect(await screen.findByText("Resilience Curve")).toBeInTheDocument();
    expect(screen.getByText("Evaluated across 9 strengths with smooth degradation curve.")).toBeInTheDocument();
    expect(screen.getByText("Levels Tested")).toBeInTheDocument();
  });

  it("handles compare API error gracefully", async () => {
    vi.spyOn(api, "compareDistortions").mockRejectedValueOnce(new Error("Network connection error"));

    render(<ResilienceLab />);

    const compareBtn = screen.getByRole("button", { name: /compare/i });
    fireEvent.click(compareBtn);

    expect(await screen.findByText("Network connection error")).toBeInTheDocument();
  });

  it("disables button when textarea is empty or whitespace", () => {
    render(<ResilienceLab />);

    const textarea = screen.getByRole("textbox", { name: /test text for resilience comparison/i });
    fireEvent.change(textarea, { target: { value: "   " } });

    const compareBtn = screen.getByRole("button", { name: /compare/i });
    expect(compareBtn).toBeDisabled();
  });
});
