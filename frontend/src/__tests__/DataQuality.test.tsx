// frontend/src/__tests__/DataQuality.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";

vi.mock("framer-motion", () => {
    const createMotionMock = (tag: string) =>
        React.forwardRef(function MockMotionComponent(
            props: Record<string, unknown>,
            ref: React.Ref<HTMLElement>
        ) {
            const {
                children, whileHover, whileTap, initial, animate, exit,
                transition, variants, layout, ...domProps
            } = props;
            void whileHover; void whileTap; void initial; void animate;
            void exit; void transition; void variants; void layout;
            return React.createElement(tag, { ...domProps, ref }, children as React.ReactNode);
        });

    return {
        motion: new Proxy({}, { get: (_t, prop: string) => createMotionMock(prop) }),
        AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
        HTMLMotionProps: {},
    };
});

const optimizeDataMock = vi.fn();
const optimizeDataStreamMock = vi.fn();
const suggestConfigMock = vi.fn();

vi.mock("@/lib/api", () => ({
    optimizeData: (...args: unknown[]) => optimizeDataMock(...args),
    optimizeDataStream: (...args: unknown[]) => optimizeDataStreamMock(...args),
    suggestConfig: (...args: unknown[]) => suggestConfigMock(...args),
}));

import { DataQuality } from "@/components/dashboard/DataQuality";

describe("DataQuality component", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("renders without crashing, showing the sample dataset overview", () => {
        render(<DataQuality />);
        expect(screen.getByText("Current Dataset")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Upload dataset file" })).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Run data optimization" })).toBeInTheDocument();
    });

    it("shows an error state with a working retry count when optimization fails", async () => {
        optimizeDataMock.mockRejectedValueOnce(new Error("Network timeout"));

        render(<DataQuality />);
        fireEvent.click(screen.getByRole("button", { name: "Run data optimization" }));

        expect(await screen.findByText("Network timeout")).toBeInTheDocument();
        const retryBtn = screen.getByRole("button", { name: "Retry optimization" });
        expect(retryBtn).toHaveTextContent("Retry (1)");
    });

    it("completes a run and shows before/after stats, then loads config suggestions", async () => {
        optimizeDataMock.mockResolvedValueOnce({ estimate: { credits: 5, estimated_minutes: 2 } });
        optimizeDataStreamMock.mockImplementation(async function* () {
            yield { event: "progress", progress_pct: 50, message: "Halfway there" };
            yield {
                event: "complete",
                run_id: "run_1",
                result: { optimized_count: 8, quality: {} },
                elapsed_seconds: 12,
                before_stats: null,
                after_stats: { count: 8, avg_length: 55, total_chars: 440, avg_words: 9 },
            };
        });

        render(<DataQuality />);
        fireEvent.click(screen.getByRole("button", { name: "Run data optimization" }));

        expect(await screen.findByText("Completed")).toBeInTheDocument();
        expect(screen.getByText("Before / After")).toBeInTheDocument();
        expect(screen.getByText("Optimized")).toBeInTheDocument();

        suggestConfigMock.mockResolvedValueOnce({
            model: "gpt-4",
            suggestions: [{ param: "batch_size", current: 8, suggested: 16, reason: "faster convergence" }],
        });
        fireEvent.click(screen.getByRole("button", { name: "Get config suggestions" }));

        expect(await screen.findByText(/Suggestions \(gpt-4\)/)).toBeInTheDocument();
        expect(screen.getByText("batch_size")).toBeInTheDocument();
    });
});

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