import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

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

describe("ConfirmDialog Component", () => {
  const defaultProps = {
    open: true,
    onClose: vi.fn(),
    onConfirm: vi.fn(),
    title: "Delete Model Checkpoint",
    subtitle: "This action cannot be undone.",
    children: <p>Are you sure you want to permanently delete this checkpoint?</p>,
  };

  it("renders custom title, subtitle, and message children", () => {
    render(<ConfirmDialog {...defaultProps} />);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Delete Model Checkpoint")).toBeInTheDocument();
    expect(screen.getByText("This action cannot be undone.")).toBeInTheDocument();
    expect(
      screen.getByText("Are you sure you want to permanently delete this checkpoint?")
    ).toBeInTheDocument();
  });

  it("calls onConfirm when confirm button is clicked", () => {
    const onConfirm = vi.fn();
    render(<ConfirmDialog {...defaultProps} onConfirm={onConfirm} confirmLabel="Delete" />);

    const confirmBtn = screen.getByRole("button", { name: "Delete" });
    fireEvent.click(confirmBtn);
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when cancel button is clicked", () => {
    const onClose = vi.fn();
    render(<ConfirmDialog {...defaultProps} onClose={onClose} cancelLabel="Dismiss" />);

    const cancelBtn = screen.getByRole("button", { name: "Dismiss" });
    fireEvent.click(cancelBtn);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("disables buttons and shows loading text when isLoading=true", () => {
    const onConfirm = vi.fn();
    const onClose = vi.fn();
    render(
      <ConfirmDialog
        {...defaultProps}
        onClose={onClose}
        onConfirm={onConfirm}
        isLoading={true}
        confirmLabel="Execute"
        cancelLabel="Abort"
      />
    );

    const loadingConfirmBtn = screen.getByRole("button", { name: "Processing..." });
    const abortBtn = screen.getByRole("button", { name: "Abort" });

    expect(loadingConfirmBtn).toBeDisabled();
    expect(abortBtn).toBeDisabled();

    fireEvent.click(loadingConfirmBtn);
    fireEvent.click(abortBtn);

    expect(onConfirm).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("renders with different button variant styles", () => {
    const { rerender } = render(
      <ConfirmDialog {...defaultProps} variant="danger" confirmLabel="Danger Action" />
    );
    expect(screen.getByRole("button", { name: "Danger Action" })).toBeInTheDocument();

    rerender(
      <ConfirmDialog {...defaultProps} variant="primary" confirmLabel="Primary Action" />
    );
    expect(screen.getByRole("button", { name: "Primary Action" })).toBeInTheDocument();

    rerender(
      <ConfirmDialog {...defaultProps} variant="secondary" confirmLabel="Secondary Action" />
    );
    expect(screen.getByRole("button", { name: "Secondary Action" })).toBeInTheDocument();
  });

  it("is hidden when open=false", () => {
    render(<ConfirmDialog {...defaultProps} open={false} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("Delete Model Checkpoint")).not.toBeInTheDocument();
  });
});
