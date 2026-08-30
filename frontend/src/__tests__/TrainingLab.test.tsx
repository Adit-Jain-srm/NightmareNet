import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import TrainingLab from "@/components/TrainingLab";
import * as api from "@/lib/api";

// Mock Framer Motion
vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: React.forwardRef(function MockMotionDiv(
      { children, initial, animate, exit, transition, whileInView, viewport, ...props }: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>
    ) {
      void initial;
      void animate;
      void exit;
      void transition;
      void whileInView;
      void viewport;
      return React.createElement(
        "div",
        { ...(props as Record<string, unknown>), ref },
        children as React.ReactNode
      );
    }),
  },
}));

// Mock LiveRegion
vi.mock("@/components/a11y/LiveRegion", () => ({
  LiveRegion: ({ message, assertive }: { message: string; assertive?: boolean }) => (
    <div data-testid="live-region" role="status" aria-live={assertive ? "assertive" : "polite"}>
      {message}
    </div>
  ),
}));

describe("TrainingLab Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders initial state with heading, description, presets, sliders, and empty preview placeholder", () => {
    render(<TrainingLab />);

    expect(screen.getByRole("heading", { name: /training lab/i })).toBeInTheDocument();
    expect(
      screen.getByText(/Configure training hyperparameters and preview the full phase schedule/i)
    ).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /apply quick start preset/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /apply deep training preset/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /apply production preset/i })).toBeInTheDocument();

    expect(screen.getByLabelText("Training Cycles")).toBeInTheDocument();
    expect(screen.getByLabelText("Wake Epochs")).toBeInTheDocument();
    expect(screen.getByLabelText("Dream Epochs")).toBeInTheDocument();
    expect(screen.getByLabelText("Nightmare Epochs")).toBeInTheDocument();
    expect(screen.getByLabelText("Dream Strength")).toBeInTheDocument();
    expect(screen.getByLabelText("Nightmare Strength")).toBeInTheDocument();
    expect(screen.getByLabelText("Learning Rate")).toBeInTheDocument();
    expect(screen.getByLabelText("Pruning Ratio")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /preview configuration/i })).toBeInTheDocument();
    expect(screen.getByText("Configure & Preview")).toBeInTheDocument();
  });

  it("applies preset configuration values when preset button is clicked", () => {
    render(<TrainingLab />);

    const deepPresetBtn = screen.getByRole("button", { name: /apply deep training preset/i });
    fireEvent.click(deepPresetBtn);

    const cyclesSlider = screen.getByLabelText("Training Cycles");
    expect(cyclesSlider).toHaveValue("5");

    const dreamEpochsSlider = screen.getByLabelText("Dream Epochs");
    expect(dreamEpochsSlider).toHaveValue("3");
  });

  it("updates slider values when input range changes", () => {
    render(<TrainingLab />);

    const cyclesSlider = screen.getByLabelText("Training Cycles");
    fireEvent.change(cyclesSlider, { target: { value: "7" } });
    expect(cyclesSlider).toHaveValue("7");

    const lrSlider = screen.getByLabelText("Learning Rate");
    fireEvent.change(lrSlider, { target: { value: "0.0001" } });
    expect(lrSlider).toHaveValue("0.0001");
  });

  it("toggles switch buttons (early stopping, learned adversarial)", () => {
    render(<TrainingLab />);

    const earlyStoppingSwitch = screen.getByRole("switch", { name: /early stopping/i });
    expect(earlyStoppingSwitch).toHaveAttribute("aria-checked", "false");

    fireEvent.click(earlyStoppingSwitch);
    expect(earlyStoppingSwitch).toHaveAttribute("aria-checked", "true");

    const learnedAdversarialSwitch = screen.getByRole("switch", { name: /learned adversarial/i });
    expect(learnedAdversarialSwitch).toHaveAttribute("aria-checked", "false");

    fireEvent.click(learnedAdversarialSwitch);
    expect(learnedAdversarialSwitch).toHaveAttribute("aria-checked", "true");
  });

  it("calls previewTrainingConfig on preview button click and renders phase schedule and recommendations", async () => {
    const mockPreviewResponse: api.TrainingConfigResponse = {
      valid: true,
      total_phases: 9,
      total_epochs: 18,
      estimated_phases: [
        { cycle: 1, phase: "wake", epochs: 3 },
        { cycle: 1, phase: "dream", epochs: 2 },
        { cycle: 1, phase: "nightmare", epochs: 1 },
      ],
      recommendations: ["Increase dream epochs for deeper linguistic robustness."],
    };

    vi.spyOn(api, "previewTrainingConfig").mockResolvedValueOnce(mockPreviewResponse);

    render(<TrainingLab />);

    const previewBtn = screen.getByRole("button", { name: /preview configuration/i });
    fireEvent.click(previewBtn);

    await waitFor(() => {
      expect(api.previewTrainingConfig).toHaveBeenCalled();
    });

    expect(await screen.findByText("Config Summary")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
    expect(screen.getByText("18")).toBeInTheDocument();
    expect(screen.getByText("Phase Schedule")).toBeInTheDocument();
    expect(screen.getByText("C1 wake")).toBeInTheDocument();
    expect(screen.getByText("3ep")).toBeInTheDocument();
    expect(
      screen.getByText("Increase dream epochs for deeper linguistic robustness.")
    ).toBeInTheDocument();
  });

  it("announces live status to LiveRegion during loading and upon completion", async () => {
    let resolvePromise: (value: api.TrainingConfigResponse) => void;
    const promise = new Promise<api.TrainingConfigResponse>((resolve) => {
      resolvePromise = resolve;
    });

    vi.spyOn(api, "previewTrainingConfig").mockReturnValueOnce(promise);

    render(<TrainingLab />);

    const previewBtn = screen.getByRole("button", { name: /preview configuration/i });
    fireEvent.click(previewBtn);

    expect(screen.getByTestId("live-region")).toHaveTextContent(
      "Training configuration preview started"
    );

    resolvePromise!({
      valid: true,
      total_phases: 4,
      total_epochs: 8,
      estimated_phases: [],
      recommendations: [],
    });

    await waitFor(() => {
      expect(screen.getByTestId("live-region")).toHaveTextContent(
        "Training configuration preview ready"
      );
    });
  });

  it("handles preview API errors and displays error message with assertive LiveRegion", async () => {
    vi.spyOn(api, "previewTrainingConfig").mockRejectedValueOnce(
      new Error("Invalid hyperparameter bounds")
    );

    render(<TrainingLab />);

    const previewBtn = screen.getByRole("button", { name: /preview configuration/i });
    fireEvent.click(previewBtn);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Invalid hyperparameter bounds"
    );
    expect(screen.getByTestId("live-region")).toHaveTextContent(
      "Preview failed: Invalid hyperparameter bounds"
    );
    expect(screen.getByTestId("live-region")).toHaveAttribute("aria-live", "assertive");
  });
});
