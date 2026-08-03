// frontend/src/__tests__/AuditTrail.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";

vi.mock("framer-motion", () => {
    const createMotionMock = (tag: string) =>
        function MockMotionComponent(props: Record<string, unknown>) {
            const {
                children, whileHover, whileTap, initial, animate, exit,
                transition, variants, layout, ref, ...domProps
            } = props;
            void whileHover; void whileTap; void initial; void animate;
            void exit; void transition; void variants; void layout;
            return React.createElement(tag, { ...domProps, ref }, children as React.ReactNode);
        };

    return {
        motion: new Proxy({}, { get: (_t, prop: string) => createMotionMock(prop) }),
        AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
        HTMLMotionProps: {},
    };
});

const pushMock = vi.fn();
vi.mock("@/components/ui/Toast", () => ({
    useToast: () => ({ push: pushMock }),
}));

import { AuditTrail } from "@/components/dashboard/AuditTrail";

describe("AuditTrail component", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("renders without crashing, showing the full event count", () => {
        render(<AuditTrail />);
        expect(screen.getByText("8 events")).toBeInTheDocument();
    });

    it("renders sample events by message text", () => {
        render(<AuditTrail />);
        expect(
            screen.getByText(/Started training run wikitext-resilient-v3/)
        ).toBeInTheDocument();
    });

    it("filters events by search query", () => {
        render(<AuditTrail />);
        const search = screen.getByPlaceholderText("Search events…");
        fireEvent.change(search, { target: { value: "API key rotated" } });

        expect(screen.getByText(/API key rotated/)).toBeInTheDocument();
        expect(
            screen.queryByText(/Started training run wikitext-resilient-v3/)
        ).not.toBeInTheDocument();
    });

    it("shows the no-match empty state when search matches nothing", () => {
        render(<AuditTrail />);
        const search = screen.getByPlaceholderText("Search events…");
        fireEvent.change(search, { target: { value: "zzz-no-such-event" } });

        expect(screen.getByText("No events recorded")).toBeInTheDocument();
        expect(screen.getByText(/Nothing matches the current filter/)).toBeInTheDocument();
        expect(screen.getByText("Clear filters")).toBeInTheDocument();
    });

    it("clears the search and restores the full list", () => {
        render(<AuditTrail />);
        const search = screen.getByPlaceholderText("Search events…");
        fireEvent.change(search, { target: { value: "zzz-no-such-event" } });
        fireEvent.click(screen.getByText("Clear filters"));

        expect(screen.getByText("8 events")).toBeInTheDocument();
        expect(
            screen.getByText(/Started training run wikitext-resilient-v3/)
        ).toBeInTheDocument();
    });

    it("pushes an info toast when 'Open settings' is clicked from the empty state", () => {
        render(<AuditTrail />);
        const search = screen.getByPlaceholderText("Search events…");
        fireEvent.change(search, { target: { value: "zzz-no-such-event" } });
        fireEvent.click(screen.getByText("Open settings"));

        expect(pushMock).toHaveBeenCalledWith(
            expect.objectContaining({ title: "Audit settings", variant: "info" })
        );
    });
});