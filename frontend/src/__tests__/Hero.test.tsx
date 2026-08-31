import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import Hero from "@/components/Hero";
import QuickStart from "@/components/QuickStart";

// Mock GSAP and @gsap/react to prevent layout/animation errors in JSDOM
vi.mock("gsap", () => ({
  default: {
    registerPlugin: vi.fn(),
    to: vi.fn(),
  },
}));

vi.mock("@gsap/react", () => ({
  useGSAP: vi.fn(() => {
    // We intentionally don't execute the callback to avoid GSAP manipulating missing JSDOM nodes
  }),
}));

// Mock framer-motion to avoid SVG animation/IntersectionObserver issues in tests
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
    HTMLMotionProps: {},
  };
});

describe("Landing Page Components", () => {
  const originalClipboard = navigator.clipboard;

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock navigator.clipboard
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  afterEach(() => {
    Object.assign(navigator, {
      clipboard: originalClipboard,
    });
  });

  describe("Hero Component", () => {
    it("renders correctly and displays headline and description", () => {
      render(<Hero />);

      // Verify headline
      expect(screen.getByRole("heading", { name: /NightmareNet/i })).toBeInTheDocument();

      // Verify description texts
      expect(screen.getByText(/Autonomous AI Self-Improvement/i)).toBeInTheDocument();
      expect(
        screen.getByText(/Force neural networks to learn invariant structures/i)
      ).toBeInTheDocument();
    });

    it("renders primary CTAs and ensures they are accessible links", () => {
      render(<Hero />);

      const dashboardLink = screen.getByRole("link", { name: /Launch Dashboard/i });
      const playgroundLink = screen.getByRole("link", { name: /Try Playground/i });

      expect(dashboardLink).toBeInTheDocument();
      expect(dashboardLink).toHaveAttribute("href", "/dashboard");

      expect(playgroundLink).toBeInTheDocument();
      expect(playgroundLink).toHaveAttribute("href", "#playground");
    });
  });

  describe("QuickStart Component", () => {
    it("renders code snippets and switches correctly between tabs", async () => {
      render(<QuickStart />);

      // Default tab should be "Install" and contain clone instructions
      expect(screen.getByText(/git clone/i)).toBeInTheDocument();

      // Switch to "Python" tab
      const pythonTab = screen.getByRole("button", { name: /Python/i });
      fireEvent.click(pythonTab);

      // Verify Python code is rendered
      expect(await screen.findByText(/trainer = Trainer\(config\)/i)).toBeInTheDocument();
    });

    it("renders the copy button, copies code to clipboard, and updates state", async () => {
      render(<QuickStart />);

      const copyButton = screen.getByRole("button", { name: /Copy/i });
      expect(copyButton).toBeInTheDocument();

      fireEvent.click(copyButton);

      // Verify clipboard API was called with the "Install" snippet text
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        expect.stringContaining("git clone")
      );

      // Verify the button text changed to indicate success
      expect(await screen.findByRole("button", { name: /Copied!/i })).toBeInTheDocument();
    });
  });
});
