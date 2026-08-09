// frontend/src/__tests__/CommandPalette.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React, { useMemo, useState } from "react";

vi.mock("framer-motion", () => {
  const mock = (tag: string) =>
    React.forwardRef(function Motion(props: Record<string, unknown>, ref: React.Ref<HTMLElement>) {
      const { children, initial, animate, exit, transition, ...rest } = props;
      void initial; void animate; void exit; void transition;
      return React.createElement(tag, { ...rest, ref }, children as React.ReactNode);
    });
  return {
    motion: new Proxy({}, { get: (_t, p: string) => mock(p) }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

vi.mock("@/components/dashboard/useGlobalShortcuts", async (importOriginal) => {
  const mod = await importOriginal<typeof import("@/components/dashboard/useGlobalShortcuts")>();
  return { ...mod, loadRecentPaletteIds: () => [], pushRecentPaletteId: vi.fn() };
});

import { CommandPalette } from "@/components/dashboard/CommandPalette";
import { useGlobalShortcuts } from "@/components/dashboard/useGlobalShortcuts";

function Harness({ onNavigate = vi.fn(), onAction = vi.fn() }) {
  const [open, setOpen] = useState(false);
  const handlers = useMemo(
    () => ({
      onPaletteToggle: () => setOpen((v) => !v),
      onHelpToggle: () => undefined,
      onNavigate,
    }),
    [onNavigate]
  );
  useGlobalShortcuts(handlers);
  return (
    <CommandPalette open={open} onClose={() => setOpen(false)} onNavigate={onNavigate} onAction={onAction} />
  );
}

describe("CommandPalette", () => {
  beforeEach(() => vi.clearAllMocks());

  it("opens on Cmd+K and closes on Escape", () => {
    render(<Harness />);
    fireEvent.keyDown(document, { key: "k", metaKey: true });
    expect(screen.getByRole("dialog", { name: "Command palette" })).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Command palette" })).not.toBeInTheDocument();
  });

  it("filters results as the user types", () => {
    render(<CommandPalette open onClose={vi.fn()} onNavigate={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Command palette search"), { target: { value: "Settings" } });
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.queryByText("Live Metrics")).not.toBeInTheDocument();
  });

  it("highlights the next item on ArrowDown", () => {
    render(<CommandPalette open onClose={vi.fn()} onNavigate={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Go to Command Center/i }).className).toMatch(/bg-neural/);
    fireEvent.keyDown(document, { key: "ArrowDown" });
    expect(screen.getByRole("button", { name: /Go to Experiments/i }).className).toMatch(/bg-neural/);
  });

  it("runs the selected action on Enter", () => {
    const onNavigate = vi.fn();
    const onClose = vi.fn();
    render(<CommandPalette open onClose={onClose} onNavigate={onNavigate} />);
    fireEvent.keyDown(document, { key: "Enter" });
    expect(onNavigate).toHaveBeenCalledWith("command-center");
    expect(onClose).toHaveBeenCalled();
  });

  it("shows an empty state when nothing matches", () => {
    render(<CommandPalette open onClose={vi.fn()} onNavigate={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Command palette search"), { target: { value: "zzz-no-match" } });
    expect(screen.getByText(/No commands match/i)).toBeInTheDocument();
  });
});
