import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import { RunDetail } from "@/components/dashboard/RunDetail";
import * as api from "@/lib/api";

const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

const mockToastPush = vi.fn();
vi.mock("@/components/ui/Toast", () => ({
  useToast: () => ({
    push: mockToastPush,
  }),
}));

vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: React.forwardRef(function MockMotionDiv(
      { children, initial, animate, exit, transition, ...props }: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>
    ) {
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

describe("RunDetail Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders with run title, subtitle, status badge, and phase tabs", () => {
    render(<RunDetail />);

    expect(screen.getByText("Run · wikitext-resilient-v3")).toBeInTheDocument();
    expect(screen.getByText("exp_4f0a · DistilBERT · cycle 4 of 5")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Wake" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Dream" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Nightmare" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Compress" })).toBeInTheDocument();

    expect(screen.getByText(/Mild distortion cycle exposing the model/i)).toBeInTheDocument();
    expect(screen.getByText("Avg Loss")).toBeInTheDocument();
    expect(screen.getByText("0.97")).toBeInTheDocument();
  });

  it("switches tabs and displays corresponding phase metrics and descriptions", () => {
    render(<RunDetail />);

    const nightmareTab = screen.getByRole("button", { name: "Nightmare" });
    fireEvent.click(nightmareTab);

    expect(
      screen.getByText(/Adversarial stress: typos, swaps, deletions/i)
    ).toBeInTheDocument();
    expect(screen.getByText("PGD")).toBeInTheDocument();
    expect(screen.getByText("+4.1")).toBeInTheDocument();

    const compressTab = screen.getByRole("button", { name: "Compress" });
    fireEvent.click(compressTab);

    expect(
      screen.getByText(/Knowledge distillation \+ pruning into a leaner/i)
    ).toBeInTheDocument();
    expect(screen.getByText("−42%")).toBeInTheDocument();
  });

  it("calls cancelPipeline API and pushes a toast when Cancel button is clicked", async () => {
    vi.spyOn(api, "cancelPipeline").mockResolvedValueOnce({ status: "cancelled", run_id: "exp_4f0a" });

    render(<RunDetail />);

    const cancelBtn = screen.getByRole("button", { name: "Cancel" });
    fireEvent.click(cancelBtn);

    await waitFor(() => {
      expect(api.cancelPipeline).toHaveBeenCalledWith("exp_4f0a");
      expect(mockToastPush).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Run cancelled",
          variant: "warning",
        })
      );
    });
  });

  it("handles cancelPipeline API failure with error toast", async () => {
    vi.spyOn(api, "cancelPipeline").mockRejectedValueOnce(new Error("Unable to cancel active run"));

    render(<RunDetail />);

    const cancelBtn = screen.getByRole("button", { name: "Cancel" });
    fireEvent.click(cancelBtn);

    await waitFor(() => {
      expect(mockToastPush).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Cancel failed",
          description: "Unable to cancel active run",
          variant: "error",
        })
      );
    });
  });

  it("downloads report file on Download Report button click", async () => {
    vi.spyOn(api, "getPipelineReport").mockResolvedValueOnce({
      report_md: "# Pipeline Run Report for exp_4f0a\n\nAll metrics passed.",
    });

    const mockCreateObjectURL = vi.fn().mockReturnValue("blob:http://localhost/mock-blob-id");
    const mockRevokeObjectURL = vi.fn();
    window.URL.createObjectURL = mockCreateObjectURL;
    window.URL.revokeObjectURL = mockRevokeObjectURL;

    render(<RunDetail />);

    const downloadBtn = screen.getByRole("button", { name: "Download report" });
    fireEvent.click(downloadBtn);

    await waitFor(() => {
      expect(api.getPipelineReport).toHaveBeenCalledWith("exp_4f0a");
      expect(mockToastPush).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Report downloaded",
          variant: "success",
        })
      );
    });
  });

  it("opens Re-run menu, selects preset variation, and creates new pipeline run", async () => {
    vi.spyOn(api, "createPipeline").mockResolvedValueOnce({
      run_id: "exp_new_8899",
      status: "queued",
    });

    render(<RunDetail />);

    const rerunTrigger = screen.getByRole("button", { name: /re-run/i });
    fireEvent.click(rerunTrigger);

    expect(screen.getByRole("menu", { name: /re-run with mutated config/i })).toBeInTheDocument();
    expect(screen.getByText("Same config")).toBeInTheDocument();
    expect(screen.getByText("Strength × 1.2")).toBeInTheDocument();
    expect(screen.getByText("Switch to GPT-2")).toBeInTheDocument();

    const switchGpt2Item = screen.getByRole("menuitem", { name: /switch to gpt-2/i });
    fireEvent.click(switchGpt2Item);

    await waitFor(() => {
      expect(api.createPipeline).toHaveBeenCalledWith(
        expect.objectContaining({
          model_name: "GPT-2",
        })
      );
      expect(mockPush).toHaveBeenCalledWith("/run/exp_new_8899");
      expect(mockToastPush).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Re-run queued · Switch to GPT-2",
          variant: "info",
        })
      );
    });
  });
});
