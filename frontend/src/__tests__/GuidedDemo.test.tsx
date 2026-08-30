import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import GuidedDemo from "@/components/GuidedDemo";
import { runDemo } from "@/lib/api";
import { HANDOFF_DEMO_TEXT_KEY } from "@/lib/handoff";

// ── Mock Framer Motion ──
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
        viewport,
        whileInView,
        layoutId,
        ...domProps
      } = props;
      void whileHover; void whileTap; void initial; void animate;
      void exit; void transition; void variants; void layout;
      void viewport; void whileInView; void layoutId;
      return React.createElement(tag, { ...domProps, ref }, children as React.ReactNode);
    };

  return {
    motion: new Proxy({}, { get: (_t, prop: string) => createMotionMock(prop) }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
  };
});

// ── Mock next/navigation ──
const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// ── Mock api module ──
vi.mock("@/lib/api", () => ({
  runDemo: vi.fn(),
}));

const mockDemoResponse = {
  original_text: "The transformer model processes sequential input through self-attention layers.",
  dream: {
    distorted_text: "The transformer model transforms sequential inputs using self-attention layers.",
    similarity: 0.92,
    length_ratio: 1.0,
  },
  nightmare: {
    distorted_text: "The model sequences inputs through attention layers.",
    similarity: 0.65,
    length_ratio: 0.8,
  },
  resilience_delta: 0.27,
  insight: "The model is robust to slight syntactic shifts, but susceptible to structured sequence omission.",
};

describe("GuidedDemo Component", () => {
  const originalSessionStorage = global.sessionStorage;

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock sessionStorage
    const storageStore: Record<string, string> = {};
    const mockSessionStorage = {
      getItem: vi.fn((key: string) => storageStore[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        storageStore[key] = value;
      }),
      removeItem: vi.fn((key: string) => {
        delete storageStore[key];
      }),
      clear: vi.fn(() => {
        for (const k in storageStore) delete storageStore[k];
      }),
    };
    Object.defineProperty(global, "sessionStorage", {
      value: mockSessionStorage,
      writable: true,
    });
  });

  afterEach(() => {
    global.sessionStorage = originalSessionStorage;
  });

  it("renders with the initial start input and default samples", () => {
    render(<GuidedDemo />);

    // Verify title and description
    expect(screen.getByText("See It In")).toBeInTheDocument();
    expect(screen.getByText("Action")).toBeInTheDocument();

    // Verify input textarea contains the first default sample
    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeInTheDocument();
    expect(textarea.innerHTML).toContain("The transformer model processes sequential input");

    // Verify CTAs
    expect(screen.getByRole("button", { name: /Try another example/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Open in dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Dream It/i })).toBeInTheDocument();
  });

  it("cycles sample texts when clicking 'Try another example'", () => {
    render(<GuidedDemo />);

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    const initialText = textarea.value;

    const cycleBtn = screen.getByRole("button", { name: /Try another example/i });
    fireEvent.click(cycleBtn);

    // Textarea value should have updated
    expect(textarea.value).not.toBe(initialText);
  });

  it("transitions to loading state and then to Step 1 upon successful execution", async () => {
    vi.mocked(runDemo).mockResolvedValueOnce(mockDemoResponse);

    render(<GuidedDemo />);

    const dreamItBtn = screen.getByRole("button", { name: /Dream It/i });
    fireEvent.click(dreamItBtn);

    // Verify API is called
    expect(runDemo).toHaveBeenCalledWith({
      text: expect.stringContaining("The transformer model processes sequential input"),
    });

    // Check loading indicator during transit
    expect(dreamItBtn).toHaveAttribute("disabled");

    // Verify Step 2 results (Step 1 index) are rendered
    expect(await screen.findByText("Step 2 — Dream Distortion")).toBeInTheDocument();
    expect(screen.getByText("92% preserved")).toBeInTheDocument();
    expect(screen.getByText("Now Nightmare It")).toBeInTheDocument();
  });

  it("transitions to Step 2 results when clicking 'Now Nightmare It'", async () => {
    vi.mocked(runDemo).mockResolvedValueOnce(mockDemoResponse);

    render(<GuidedDemo />);

    // Step 0 -> Step 1
    fireEvent.click(screen.getByRole("button", { name: /Dream It/i }));

    // Step 1 -> Step 2
    const nightmareBtn = await screen.findByRole("button", { name: /Now Nightmare It/i });
    fireEvent.click(nightmareBtn);

    // Verify Step 3 results are rendered
    expect(screen.getByText("Step 3 — Nightmare Distortion")).toBeInTheDocument();
    expect(screen.getByText("65% preserved")).toBeInTheDocument();

    // Verify Insight card renders
    expect(screen.getByText("What this means")).toBeInTheDocument();
    expect(screen.getByText(mockDemoResponse.insight)).toBeInTheDocument();
  });

  it("resets to initial step when clicking 'Try Different Text' from the final screen", async () => {
    vi.mocked(runDemo).mockResolvedValueOnce(mockDemoResponse);

    render(<GuidedDemo />);

    // Run complete progression
    fireEvent.click(screen.getByRole("button", { name: /Dream It/i }));
    const nightmareBtn = await screen.findByRole("button", { name: /Now Nightmare It/i });
    fireEvent.click(nightmareBtn);

    // Click Reset
    const resetBtn = screen.getByRole("button", { name: /Try Different Text/i });
    fireEvent.click(resetBtn);

    // Should return back to Step 1 (Index 0) textarea input
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });

  it("handles dashboard handoff by writing to sessionStorage and redirecting", () => {
    render(<GuidedDemo />);

    const dashboardBtn = screen.getByRole("button", { name: /Open in dashboard/i });
    fireEvent.click(dashboardBtn);

    // Should write input text to sessionStorage
    expect(sessionStorage.setItem).toHaveBeenCalledWith(
      HANDOFF_DEMO_TEXT_KEY,
      expect.stringContaining("The transformer model processes sequential input")
    );

    // Should redirect to dashboard path with demo param
    expect(mockPush).toHaveBeenCalledWith("/dashboard?from=demo");
  });

  it("displays error message if API fails during execution", async () => {
    vi.mocked(runDemo).mockRejectedValueOnce(new Error("API internal failure"));

    render(<GuidedDemo />);

    fireEvent.click(screen.getByRole("button", { name: /Dream It/i }));

    // Verify error text is rendered
    expect(
      await screen.findByText(/API internal failure — make sure the API is running/i)
    ).toBeInTheDocument();
  });
});
