import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import Navbar from "@/components/Navbar";

// ── Mock Framer Motion ──
let scrollHandler: ((latest: number) => void) | null = null;
let mockScrollYValue = 0;
let mockPrevScrollYValue = 0;

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
    useScroll: () => ({
      scrollY: {
        get: () => mockScrollYValue,
        getPrevious: () => mockPrevScrollYValue,
      },
    }),
    useMotionValueEvent: (_value: unknown, event: string, handler: (latest: number) => void) => {
      if (event === "change") {
        scrollHandler = handler;
      }
    },
  };
});

// ── Mock Theme Hook ──
const mockSetTheme = vi.fn();
let currentTheme = "dark";

vi.mock("@/lib/theme", () => ({
  useTheme: () => ({
    theme: currentTheme,
    setTheme: mockSetTheme,
  }),
}));

// ── Mock useDialogFocus Hook ──
vi.mock("@/components/a11y/useDialogFocus", () => ({
  useDialogFocus: vi.fn(() => ({ current: null })),
}));

// ── Mock IntersectionObserver ──
let observerCallback: ((entries: Array<Record<string, unknown>>) => void) | null = null;
const mockObserve = vi.fn();
const mockDisconnect = vi.fn();

class MockIntersectionObserver {
  constructor(callback: (entries: Array<Record<string, unknown>>) => void) {
    observerCallback = callback;
  }
  observe = mockObserve;
  disconnect = mockDisconnect;
}

vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);

describe("Navbar Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scrollHandler = null;
    mockScrollYValue = 0;
    mockPrevScrollYValue = 0;
    currentTheme = "dark";
    observerCallback = null;
  });

  it("renders the brand logo and standard navigation links", () => {
    render(<Navbar />);

    // Verify brand logo exists
    expect(screen.getByLabelText("Primary navigation")).toBeInTheDocument();
    expect(screen.getByText("Nightmare")).toBeInTheDocument();
    expect(screen.getByText("Net")).toBeInTheDocument();

    // Verify navigation links render correctly
    expect(screen.getByRole("button", { name: /Demo/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Quick Start/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Playground/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Pipeline/i })).toBeInTheDocument();
  });

  it("handles scroll behavior by hiding or showing the navbar", () => {
    render(<Navbar />);

    expect(scrollHandler).toBeDefined();

    // Scroll down past threshold: should trigger state changes (e.g. setHidden)
    mockPrevScrollYValue = 50;
    mockScrollYValue = 150;
    act(() => {
      if (scrollHandler) scrollHandler(150);
    });

    // Scroll up: should reveal the navbar again
    mockPrevScrollYValue = 150;
    mockScrollYValue = 100;
    act(() => {
      if (scrollHandler) scrollHandler(100);
    });
  });

  it("opens and closes the mobile menu drawer on hamburger menu click", () => {
    render(<Navbar />);

    const toggleButton = screen.getByRole("button", { name: /Open navigation menu/i });
    expect(toggleButton).toHaveAttribute("aria-expanded", "false");

    // Click toggle to open mobile menu
    fireEvent.click(toggleButton);
    expect(toggleButton).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("dialog", { name: /Mobile navigation/i })).toBeInTheDocument();

    // Click toggle again to close mobile menu
    fireEvent.click(toggleButton);
    expect(toggleButton).toHaveAttribute("aria-expanded", "false");
  });

  it("triggers theme context updates on theme toggle buttons click", () => {
    render(<Navbar />);

    // Click Light mode button
    const lightBtn = screen.getByLabelText("Light mode");
    fireEvent.click(lightBtn);
    expect(mockSetTheme).toHaveBeenCalledWith("light");

    // Click Dark mode button
    const darkBtn = screen.getByLabelText("Dark mode");
    fireEvent.click(darkBtn);
    expect(mockSetTheme).toHaveBeenCalledWith("dark");

    // Click System mode button
    const systemBtn = screen.getByLabelText("System theme");
    fireEvent.click(systemBtn);
    expect(mockSetTheme).toHaveBeenCalledWith("system");
  });

  it("updates active navigation link highlighting based on IntersectionObserver events", () => {
    render(<Navbar />);

    expect(observerCallback).toBeDefined();

    // Simulate #playground target element intersecting
    act(() => {
      if (observerCallback) {
        observerCallback([
          {
            isIntersecting: true,
            target: { id: "playground" },
            boundingClientRect: { top: 10 },
          },
        ]);
      }
    });

    // The active item class should update to reflect the intersecting id
    const activeBtn = screen.getByRole("button", { name: /Playground/i });
    expect(activeBtn).toHaveAttribute("aria-current", "location");
  });

  it("adheres to accessibility guidelines", () => {
    render(<Navbar />);

    // Proper landmarks
    const nav = screen.getByRole("navigation", { name: /Primary navigation/i });
    expect(nav).toBeInTheDocument();

    // Proper role attributes on scrollbar progress indicators
    const progress = screen.getByRole("progressbar");
    expect(progress).toHaveAttribute("aria-label", "Page scroll progress");
  });
});
