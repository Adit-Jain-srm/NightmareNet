import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { Modal } from "@/components/ui/Modal";

// Mock framer-motion
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

describe("Modal Component", () => {
  const defaultProps = {
    open: true,
    onClose: vi.fn(),
    title: "Test Modal Title",
    subtitle: "Test modal subtitle description",
    children: <div>Modal Inner Content</div>,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    document.body.style.overflow = "";
  });

  it("opens when open=true and renders title, subtitle, and children", () => {
    render(<Modal {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Test Modal Title")).toBeInTheDocument();
    expect(screen.getByText("Test modal subtitle description")).toBeInTheDocument();
    expect(screen.getByText("Modal Inner Content")).toBeInTheDocument();
    expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
  });

  it("is hidden when open=false", () => {
    render(<Modal {...defaultProps} open={false} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("Test Modal Title")).not.toBeInTheDocument();
  });

  it("calls onClose when close button in header is clicked", () => {
    const onClose = vi.fn();
    render(<Modal {...defaultProps} onClose={onClose} />);
    const closeBtn = screen.getByRole("button", { name: /close/i });
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("dismisses on backdrop click when closeOnBackdrop=true", () => {
    const onClose = vi.fn();
    const { container } = render(<Modal {...defaultProps} onClose={onClose} closeOnBackdrop={true} />);
    const backdrop = container.querySelector(".bg-void\\/80");
    expect(backdrop).toBeInTheDocument();
    if (backdrop) {
      fireEvent.click(backdrop);
    }
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not dismiss on backdrop click when closeOnBackdrop=false", () => {
    const onClose = vi.fn();
    const { container } = render(<Modal {...defaultProps} onClose={onClose} closeOnBackdrop={false} />);
    const backdrop = container.querySelector(".bg-void\\/80");
    if (backdrop) {
      fireEvent.click(backdrop);
    }
    expect(onClose).not.toHaveBeenCalled();
  });

  it("dismisses when Escape key is pressed", () => {
    const onClose = vi.fn();
    render(<Modal {...defaultProps} onClose={onClose} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not trigger Escape handler when modal is closed", () => {
    const onClose = vi.fn();
    render(<Modal {...defaultProps} open={false} onClose={onClose} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
  });

  it("locks body overflow when open and restores on unmount/close", () => {
    const { rerender, unmount } = render(<Modal {...defaultProps} open={true} />);
    expect(document.body.style.overflow).toBe("hidden");

    rerender(<Modal {...defaultProps} open={false} />);
    expect(document.body.style.overflow).toBe("");

    rerender(<Modal {...defaultProps} open={true} />);
    expect(document.body.style.overflow).toBe("hidden");

    unmount();
    expect(document.body.style.overflow).toBe("");
  });

  it("renders custom footer content", () => {
    render(
      <Modal
        {...defaultProps}
        footer={<button type="button">Custom Action</button>}
      />
    );
    expect(screen.getByRole("button", { name: "Custom Action" })).toBeInTheDocument();
  });

  it("applies size classes appropriately", () => {
    const { container, rerender } = render(<Modal {...defaultProps} size="sm" />);
    expect(container.querySelector(".max-w-sm")).toBeInTheDocument();

    rerender(<Modal {...defaultProps} size="lg" />);
    expect(container.querySelector(".max-w-2xl")).toBeInTheDocument();

    rerender(<Modal {...defaultProps} size="xl" />);
    expect(container.querySelector(".max-w-4xl")).toBeInTheDocument();
  });
});
