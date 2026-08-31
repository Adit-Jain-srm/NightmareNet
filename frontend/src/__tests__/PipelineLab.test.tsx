import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { act } from "react";
import React from "react";
import PipelineLab from "@/components/PipelineLab";

/* ── Hoisted mocks (must be declared before vi.mock factories run) ── */

const { mockCreatePipeline, mockGetPipelineStatus, mockCancelPipeline, mockGetPipelineReport } =
  vi.hoisted(() => ({
    mockCreatePipeline: vi.fn(),
    mockGetPipelineStatus: vi.fn(),
    mockCancelPipeline: vi.fn(),
    mockGetPipelineReport: vi.fn(),
  }));

const { mockDisconnect, mockReconnect, mockUseWebSocket } = vi.hoisted(() => ({
  mockDisconnect: vi.fn(),
  mockReconnect: vi.fn(),
  mockUseWebSocket: vi.fn(() => ({
    status: "disconnected" as const,
    attempt: 0,
    reconnect: vi.fn(),
    disconnect: vi.fn(),
  })),
}));

/* ── Module mocks ── */

vi.mock("framer-motion", () => {
  const createMotionMock = (tag: string) =>
    function MockMotion(props: Record<string, unknown>) {
      const { children, initial, animate, exit, transition, whileInView, viewport, ...rest } =
        props;
      void initial;
      void animate;
      void exit;
      void transition;
      void whileInView;
      void viewport;
      return React.createElement(tag, rest, children as React.ReactNode);
    };
  return {
    motion: new Proxy({}, { get: (_t, prop: string) => createMotionMock(prop) }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
  };
});

vi.mock("@/lib/api", () => ({
  createPipeline: (...args: unknown[]) => mockCreatePipeline(...args),
  getPipelineStatus: (...args: unknown[]) => mockGetPipelineStatus(...args),
  cancelPipeline: (...args: unknown[]) => mockCancelPipeline(...args),
  getPipelineReport: (...args: unknown[]) => mockGetPipelineReport(...args),
}));

vi.mock("@/lib/websocket", () => ({
  buildRunWsUrl: (id: string) => `ws://localhost/ws/runs/${id}`,
  MAX_RECONNECT_ATTEMPTS: 10,
}));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: mockUseWebSocket,
}));

vi.mock("@/components/ConnectionStatus", () => ({
  ConnectionStatus: ({ status }: { status: string }) =>
    React.createElement("div", { "data-testid": "connection-status" }, status),
}));

vi.mock("@/components/ui/Button", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    loading,
    variant,
  }: {
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
    loading?: boolean;
    variant?: string;
    size?: string;
  }) =>
    React.createElement(
      "button",
      { onClick, disabled, "aria-busy": loading ? "true" : undefined, "data-variant": variant },
      children
    ),
}));

vi.mock("@/components/ui/Card", () => ({
  Card: ({
    children,
    title,
  }: {
    children: React.ReactNode;
    title?: string;
    subtitle?: string;
    glow?: string;
    className?: string;
  }) =>
    React.createElement(
      "div",
      { "data-testid": "card" },
      title && React.createElement("span", null, title),
      children
    ),
}));

/* ── Shared status fixtures ── */

const runningStatus = (overrides = {}) => ({
  run_id: "run-abc",
  status: "running",
  is_running: true,
  current_phase: "wake",
  progress_pct: 10,
  phase_loss: 0.5,
  current_cycle: 0,
  total_cycles: 1,
  has_report: false,
  error: null,
  history: [],
  ...overrides,
});

/* ── Helpers ── */

function advanceToConfig() {
  render(<PipelineLab />);
  fireEvent.click(screen.getByText(/Next: Choose Model/i));
  fireEvent.click(screen.getByText(/Next: Configure/i));
}

async function launchPipeline(statusOverrides = {}) {
  mockCreatePipeline.mockResolvedValueOnce(runningStatus(statusOverrides));
  advanceToConfig();
  fireEvent.click(screen.getByText(/Launch Pipeline/i));
  await waitFor(() => screen.getByText("Pipeline Running"));
}

/* ── Tests ── */

describe("PipelineLab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockUseWebSocket.mockReturnValue({
      status: "disconnected" as const,
      attempt: 0,
      reconnect: mockReconnect,
      disconnect: mockDisconnect,
    });
  });

  /* 1 — Initial render */
  it("renders initial layout with step 1 active and data source options", () => {
    render(<PipelineLab />);

    expect(screen.getByText("Train a Hardened Model")).toBeInTheDocument();
    expect(screen.getByText("Web Scrape")).toBeInTheDocument();
    expect(screen.getByText("HuggingFace")).toBeInTheDocument();
    expect(screen.getByText("Paste Text")).toBeInTheDocument();
    expect(screen.getByText(/Next: Choose Model/i)).toBeInTheDocument();
    // Step indicator shows steps 1-3
    expect(screen.getByText("Data")).toBeInTheDocument();
  });

  /* 2 — Pipeline step addition (wizard navigation) */
  it("navigates through wizard steps: source → model → config", () => {
    render(<PipelineLab />);

    fireEvent.click(screen.getByText(/Next: Choose Model/i));
    expect(screen.getByText("Select Model Architecture")).toBeInTheDocument();
    expect(screen.getByText("DistilBERT")).toBeInTheDocument();

    fireEvent.click(screen.getByText(/Next: Configure/i));
    expect(screen.getByText("Training Configuration")).toBeInTheDocument();
    expect(screen.getByText("Sleep Cycles")).toBeInTheDocument();
  });

  /* 3 — Pipeline step removal / reorder (back navigation) */
  it("navigates back through wizard steps and reorders correctly", () => {
    render(<PipelineLab />);

    fireEvent.click(screen.getByText(/Next: Choose Model/i));
    expect(screen.getByText("Select Model Architecture")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Back"));
    expect(screen.getByText("Web Scrape")).toBeInTheDocument();
    expect(screen.queryByText("Select Model Architecture")).not.toBeInTheDocument();

    // Navigate forward again to config, then back to model
    fireEvent.click(screen.getByText(/Next: Choose Model/i));
    fireEvent.click(screen.getByText(/Next: Configure/i));
    fireEvent.click(screen.getByText("Back"));
    expect(screen.getByText("Select Model Architecture")).toBeInTheDocument();
  });

  /* 4 — Configuration form validation */
  it("switches source type tabs and shows correct input fields", () => {
    render(<PipelineLab />);

    // Default is URLs
    expect(screen.getByPlaceholderText(/https:\/\/en.wikipedia/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText("HuggingFace"));
    expect(screen.getByPlaceholderText("wikitext")).toBeInTheDocument();
    expect(screen.getByText("Dataset Name")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Paste Text"));
    expect(screen.getByPlaceholderText(/Paste articles/i)).toBeInTheDocument();
    expect(screen.getByText("Paste your training text")).toBeInTheDocument();
  });

  /* 5 — Pipeline execution trigger */
  it("calls createPipeline with correct payload and transitions to running step", async () => {
    mockCreatePipeline.mockResolvedValueOnce(runningStatus());

    advanceToConfig();
    fireEvent.click(screen.getByText(/Launch Pipeline/i));

    await waitFor(() => {
      expect(mockCreatePipeline).toHaveBeenCalledOnce();
      expect(mockCreatePipeline).toHaveBeenCalledWith(
        expect.objectContaining({
          model_name: "distilbert-base-uncased",
          source_type: "urls",
        })
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Pipeline Running")).toBeInTheDocument();
    });
  });

  /* 6 — Progress/status display during execution */
  it("displays progress metrics and WebSocket connection status during running step", async () => {
    await launchPipeline({
      run_id: "run-xyz",
      current_phase: "dream",
      progress_pct: 45.5,
      phase_loss: 0.312,
      current_cycle: 0,
      total_cycles: 2,
    });

    expect(screen.getByText("45.5%")).toBeInTheDocument();
    expect(screen.getByText("0.3120")).toBeInTheDocument();
    expect(screen.getByText("1 / 2")).toBeInTheDocument();
    expect(screen.getByTestId("connection-status")).toBeInTheDocument();
    expect(screen.getByTestId("connection-status")).toHaveTextContent("disconnected");
  });

  /* 7 — Error state rendering */
  it("shows user-friendly error message when createPipeline rejects", async () => {
    mockCreatePipeline.mockRejectedValueOnce(new Error("Server unavailable"));

    advanceToConfig();
    fireEvent.click(screen.getByText(/Launch Pipeline/i));

    await waitFor(() => {
      expect(screen.getByText("Server unavailable")).toBeInTheDocument();
    });
    // Still on config step, not running
    expect(screen.queryByText("Pipeline Running")).not.toBeInTheDocument();
  });

  /* 8 — Reset/clear functionality */
  it("resets to empty state when 'Run Another Pipeline' is clicked after completion", async () => {
    // Start in running state, then simulate completion via WebSocket message
    mockGetPipelineReport.mockResolvedValueOnce({ report_md: "## Report", run_id: "run-done", comparison: null });

    let capturedOnMessage: ((data: unknown) => void) | undefined;
    mockUseWebSocket.mockImplementation(({ onMessage }: { onMessage?: (data: unknown) => void }) => {
      capturedOnMessage = onMessage;
      return { status: "connected" as const, attempt: 0, reconnect: mockReconnect, disconnect: mockDisconnect };
    });

    await launchPipeline({ run_id: "run-done" });

    // Simulate a WebSocket completion event wrapped in act to avoid state-update warnings
    await act(async () => {
      capturedOnMessage?.({
        run_id: "run-done",
        status: "complete",
        is_running: false,
        current_phase: "compress",
        progress_pct: 100,
        phase_loss: 0.1,
        current_cycle: 0,
        total_cycles: 1,
        has_report: true,
        error: null,
        history: [],
        event: "complete",
      });
    });

    await waitFor(() => screen.getByText("Pipeline Complete!"));

    fireEvent.click(screen.getByText("Run Another Pipeline"));

    await waitFor(() => {
      expect(screen.getByText("Train a Hardened Model")).toBeInTheDocument();
      expect(screen.getByText("Web Scrape")).toBeInTheDocument();
      expect(screen.queryByText("Pipeline Complete!")).not.toBeInTheDocument();
    });
  });

  /* 9 — WebSocket connection status indicator */
  it("shows WebSocket status indicator with correct status during running step", async () => {
    mockUseWebSocket.mockReturnValue({
      status: "connected" as const,
      attempt: 0,
      reconnect: mockReconnect,
      disconnect: mockDisconnect,
    });

    await launchPipeline();

    const indicator = screen.getByTestId("connection-status");
    expect(indicator).toBeInTheDocument();
    expect(indicator).toHaveTextContent("connected");
  });

  /* 10 — Cancel pipeline */
  it("calls cancelPipeline API when cancel button is clicked", async () => {
    mockCancelPipeline.mockResolvedValueOnce({});
    await launchPipeline({ run_id: "run-cancel" });

    fireEvent.click(screen.getByText("Cancel Pipeline"));

    await waitFor(() => {
      expect(mockCancelPipeline).toHaveBeenCalledWith("run-cancel");
    });
  });

  /* 11 — Accessibility: form labels */
  it("has accessible labels for all config inputs", () => {
    advanceToConfig();

    expect(screen.getByText("Sleep Cycles")).toBeInTheDocument();
    expect(screen.getByText("Wake Epochs")).toBeInTheDocument();
    expect(screen.getByText("Dream Epochs")).toBeInTheDocument();
    expect(screen.getByText("Nightmare Epochs")).toBeInTheDocument();
    expect(screen.getByText("Batch Size")).toBeInTheDocument();
    expect(screen.getByText("Max Samples")).toBeInTheDocument();
  });

  /* 12 — Accessibility: loading button aria-busy state */
  it("sets aria-busy on launch button while submitting", async () => {
    // Never resolves so we can inspect the loading state
    mockCreatePipeline.mockReturnValueOnce(new Promise(() => {}));

    advanceToConfig();
    fireEvent.click(screen.getByText(/Launch Pipeline/i).closest("button")!);

    // After click the button re-renders with loading=true; query it fresh by aria-busy
    await waitFor(() => {
      const btn = screen.getByRole("button", { name: /Launch Pipeline/i });
      expect(btn).toHaveAttribute("aria-busy", "true");
    });
  });
});
