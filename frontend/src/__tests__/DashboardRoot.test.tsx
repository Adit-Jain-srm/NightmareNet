import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import { DashboardRoot } from "@/components/dashboard/DashboardRoot";

// Mock retryLazy to immediately return sync mock components
vi.mock("@/lib/retryLazy", () => ({
  retryLazy: (importFn: () => Promise<{ default: React.ComponentType<any> }>) => {
    return function MockLazySection(props: Record<string, unknown>) {
      return (
        <div data-testid="lazy-section" {...props}>
          MockSectionContent
        </div>
      );
    };
  },
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

// Mock framer-motion
vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    div: React.forwardRef(function MockMotionDiv(
      { children, initial, animate, exit, transition, variants, ...props }: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>
    ) {
      void initial;
      void animate;
      void exit;
      void transition;
      void variants;
      return React.createElement(
        "div",
        { ...(props as Record<string, unknown>), ref },
        children as React.ReactNode
      );
    }),
  },
}));

// Mock sub-overlays
vi.mock("@/components/dashboard/OnboardingOverlay", () => ({
  OnboardingOverlay: ({ onNavigate }: { onNavigate?: (sec: string) => void }) => (
    <div data-testid="onboarding-overlay">
      <button type="button" onClick={() => onNavigate?.("experiments")}>
        Tour Start
      </button>
    </div>
  ),
}));

vi.mock("@/components/dashboard/WhatsNew", () => ({
  WhatsNew: () => <div data-testid="whats-new">Whats New Modal</div>,
}));

vi.mock("@/components/dashboard/KeyboardHelp", () => ({
  KeyboardHelp: ({ open, onClose }: { open: boolean; onClose: () => void }) =>
    open ? (
      <div data-testid="keyboard-help">
        Keyboard Shortcuts Help
        <button type="button" onClick={onClose}>
          Close Help
        </button>
      </div>
    ) : null,
}));

vi.mock("@/components/dashboard/AskNightmareDock", () => ({
  AskNightmareDock: ({ section, onNavigate }: { section: string; onNavigate?: (s: any) => void }) => (
    <div data-testid="ask-nightmare-dock">
      <span>Ask Nightmare Dock ({section})</span>
      <button type="button" onClick={() => onNavigate?.("robustness")}>
        Jump To Robustness
      </button>
    </div>
  ),
}));

describe("DashboardRoot Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders initial dashboard layout with AppShell, header, and active section", () => {
    render(<DashboardRoot />);

    expect(screen.getByText("Command Center")).toBeInTheDocument();
    expect(screen.getByTestId("onboarding-overlay")).toBeInTheDocument();
    expect(screen.getByTestId("whats-new")).toBeInTheDocument();
    expect(screen.getByTestId("ask-nightmare-dock")).toHaveTextContent("command-center");
  });

  it("navigates to different section via Sidebar navigation and updates breadcrumbs", () => {
    render(<DashboardRoot />);

    const experimentsNav = screen.getByRole("button", { name: /experiments/i });
    fireEvent.click(experimentsNav);

    expect(screen.getByText("Experiments")).toBeInTheDocument();
    expect(screen.getByTestId("ask-nightmare-dock")).toHaveTextContent("experiments");
  });

  it("opens keyboard shortcut help dialog when shortcut triggered", () => {
    render(<DashboardRoot />);

    expect(screen.queryByTestId("keyboard-help")).not.toBeInTheDocument();

    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "?", bubbles: true }));
    });

    expect(screen.getByTestId("keyboard-help")).toBeInTheDocument();
    expect(screen.getByText("Keyboard Shortcuts Help")).toBeInTheDocument();

    const closeBtn = screen.getByRole("button", { name: "Close Help" });
    fireEvent.click(closeBtn);
    expect(screen.queryByTestId("keyboard-help")).not.toBeInTheDocument();
  });

  it("handles navigation initiated from OnboardingOverlay tour", () => {
    render(<DashboardRoot />);

    const tourBtn = screen.getByRole("button", { name: "Tour Start" });
    fireEvent.click(tourBtn);

    expect(screen.getByText("Experiments")).toBeInTheDocument();
  });

  it("handles keyboard shortcut navigation (1-9) to switch dashboard sections", () => {
    render(<DashboardRoot />);

    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "2", bubbles: true }));
    });

    expect(screen.getByText("Experiments")).toBeInTheDocument();
  });

  it("navigates to various sections (run-detail, robustness, benchmarks, settings)", () => {
    render(<DashboardRoot />);

    const runDetailNav = screen.getByRole("button", { name: /run detail/i });
    fireEvent.click(runDetailNav);
    expect(screen.getByText("Run Detail")).toBeInTheDocument();

    const settingsNav = screen.getByRole("button", { name: /settings/i });
    fireEvent.click(settingsNav);
    expect(screen.getByText("Settings")).toBeInTheDocument();

    const benchmarksNav = screen.getByRole("button", { name: /benchmark suite/i });
    fireEvent.click(benchmarksNav);
    expect(screen.getByText("Benchmark Suite")).toBeInTheDocument();
  });

  it("supports navigation via AskNightmareDock callback", () => {
    render(<DashboardRoot />);

    const jumpBtn = screen.getByRole("button", { name: "Jump To Robustness" });
    fireEvent.click(jumpBtn);

    expect(screen.getByText("Robustness Radar")).toBeInTheDocument();
  });
});
