import { describe, it, expect, vi, beforeEach, beforeAll, afterAll } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import { ExperimentList } from "@/components/dashboard/ExperimentList";
import * as api from "@/lib/api";

// Mock next/navigation
const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

// Mock sounds
vi.mock("@/lib/sounds", () => ({
  useSounds: () => ({
    playClick: vi.fn(),
    playSuccess: vi.fn(),
    playError: vi.fn(),
    playTransition: vi.fn(),
    playNotification: vi.fn(),
    enabled: true,
    toggle: vi.fn(),
  }),
}));

// Mock Toast
const mockPushToast = vi.fn();
vi.mock("@/components/ui/Toast", () => ({
  useToast: () => ({ push: mockPushToast }),
}));

// Mock API
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    deleteExperiment: vi.fn(),
    exportExperiment: vi.fn(),
    createPipeline: vi.fn(),
    searchExperiments: vi.fn().mockResolvedValue({ results: [] }),
  };
});

// Mock Framer Motion
vi.mock("framer-motion", () => {
  const createMock = (tag: string) => React.forwardRef(function MockMotionComponent(
    { children, initial, animate, exit, transition, whileHover, whileTap, ...props }: Record<string, unknown>,
    ref: React.Ref<any>
  ) {
    void initial; void animate; void exit; void transition; void whileHover; void whileTap;
    return React.createElement(tag, { ...(props as Record<string, unknown>), ref }, children as React.ReactNode);
  });

  return {
    motion: {
      div: createMock("div"),
      section: createMock("section"),
      button: createMock("button"),
    },
    AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

const mockExperiments = [
  {
    id: "exp_123",
    name: "test-experiment",
    model: "GPT-2",
    status: "complete" as const,
    cycles: 2,
    robustness: 80,
    duration: "10m",
    createdAt: "1m ago",
    config: {
      dream_strength: 0.5,
      nightmare_strength: 0.8,
      source_type: "text" as const,
    },
  },
  {
    id: "exp_456",
    name: "missing-config-exp",
    model: "BERT",
    status: "failed" as const,
    cycles: 1,
    robustness: 0,
    duration: "1m",
    createdAt: "2m ago",
    // No config
  },
];

describe("ExperimentList Row Actions", () => {
  let mockClick: any;

  beforeAll(() => {
    global.URL.createObjectURL = vi.fn(() => "blob:mock-url");
    global.URL.revokeObjectURL = vi.fn();
    mockClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  afterAll(() => {
    mockClick.mockRestore();
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const openMenu = (experimentName: string) => {
    const toggleButton = screen.getByRole("button", { name: `Row actions for ${experimentName}` });
    fireEvent.click(toggleButton);
  };

  it("Compare navigates to /compare?ids=<id>", () => {
    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const compareBtn = screen.getByRole("menuitem", { name: /Compare test-experiment to baseline/i });
    fireEvent.click(compareBtn);

    expect(mockPush).toHaveBeenCalledWith("/compare?ids=exp_123");
  });

  it("Delete opens confirmation dialog, Cancel delete does not call API", async () => {
    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const deleteBtn = screen.getByRole("menuitem", { name: /Delete test-experiment/i });
    fireEvent.click(deleteBtn);

    expect(await screen.findByText("Delete Experiment")).toBeInTheDocument();

    const cancelBtn = screen.getByRole("button", { name: /Cancel/i });
    fireEvent.click(cancelBtn);

    expect(api.deleteExperiment).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.queryByText("Delete Experiment")).not.toBeInTheDocument();
    });
  });

  it("Confirm delete calls deleteExperiment(id) and shows success toast", async () => {
    vi.mocked(api.deleteExperiment).mockResolvedValueOnce({ success: true } as any);
    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const deleteBtn = screen.getByRole("menuitem", { name: /Delete test-experiment/i });
    fireEvent.click(deleteBtn);

    const confirmBtn = await screen.findByRole("button", { name: /Confirm Delete/i });
    fireEvent.click(confirmBtn);

    expect(api.deleteExperiment).toHaveBeenCalledWith("exp_123");

    await waitFor(() => {
      expect(mockPushToast).toHaveBeenCalledWith(expect.objectContaining({
        title: "Experiment deleted",
        variant: "success",
      }));
    });

    expect(screen.queryByText("Delete Experiment")).not.toBeInTheDocument();
  });

  it("Re-run multiplies strengths by 1.2 before calling createPipeline", async () => {
    vi.mocked(api.createPipeline).mockResolvedValueOnce({} as any);
    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const rerunBtn = screen.getByRole("menuitem", { name: /Re-run test-experiment with strength/i });
    fireEvent.click(rerunBtn);

    expect(api.createPipeline).toHaveBeenCalledWith({
      source_type: "text",
      dream_strength: 0.6, // 0.5 * 1.2
      nightmare_strength: 0.96, // 0.8 * 1.2
      model_name: "GPT-2",
    });

    await waitFor(() => {
      expect(mockPushToast).toHaveBeenCalledWith(expect.objectContaining({
        title: "Re-run queued",
        variant: "success",
      }));
    });
  });

  it("Missing config shows error toast for Re-run", async () => {
    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("missing-config-exp");

    const rerunBtn = screen.getByRole("menuitem", { name: /Re-run missing-config-exp with strength/i });
    fireEvent.click(rerunBtn);

    expect(api.createPipeline).not.toHaveBeenCalled();
    expect(mockPushToast).toHaveBeenCalledWith(expect.objectContaining({
      title: "Missing configuration",
      variant: "error",
    }));
  });

  it("Export calls exportExperiment(id, 'csv') and triggers browser download", async () => {
    const mockBlob = new Blob(["test"], { type: "text/csv" });
    vi.mocked(api.exportExperiment).mockResolvedValueOnce(mockBlob);

    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const exportBtn = screen.getByRole("menuitem", { name: /Export test-experiment run report as CSV/i });
    fireEvent.click(exportBtn);

    expect(api.exportExperiment).toHaveBeenCalledWith("exp_123", "csv");

    await waitFor(() => {
      expect(global.URL.createObjectURL).toHaveBeenCalledWith(mockBlob);
      expect(mockClick).toHaveBeenCalled();
      expect(global.URL.revokeObjectURL).toHaveBeenCalled();
    });
  });

  it("Loading state disables actions while async request is pending", async () => {
    let resolveExport: (value: Blob) => void;
    vi.mocked(api.exportExperiment).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveExport = resolve;
      })
    );

    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const exportBtn = screen.getByRole("menuitem", { name: /Export test-experiment run report as CSV/i });
    fireEvent.click(exportBtn);

    // While pending, the menu item should be disabled
    expect(exportBtn).toBeDisabled();

    // Resolve it
    resolveExport!(new Blob(["test"], { type: "text/csv" }));

    await waitFor(() => {
      expect(screen.queryByRole("menuitem", { name: /Export test-experiment run report as CSV/i })).not.toBeInTheDocument();
    });
  });

  it("API failures display error toast", async () => {
    vi.mocked(api.deleteExperiment).mockRejectedValueOnce(new Error("Network Error"));

    render(<ExperimentList experiments={mockExperiments} />);
    openMenu("test-experiment");

    const deleteBtn = screen.getByRole("menuitem", { name: /Delete test-experiment/i });
    fireEvent.click(deleteBtn);

    const confirmBtn = await screen.findByRole("button", { name: /Confirm Delete/i });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      expect(mockPushToast).toHaveBeenCalledWith(expect.objectContaining({
        title: "API Error",
        description: "Network Error",
        variant: "error",
      }));
    });
  });
});
