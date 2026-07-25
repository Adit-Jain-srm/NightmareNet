import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ErrorBoundary } from "@/components/ui/ErrorBoundary";

function HealthyChild() {
  return <p>Dashboard content</p>;
}

function ThrowingChild(): never {
  throw new Error("Panel failed");
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  it("renders its children when no error is thrown", () => {
    render(
      <ErrorBoundary>
        <HealthyChild />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Dashboard content")).toBeInTheDocument();
  });

  it("catches child errors and displays recovery controls", () => {
    render(
      <ErrorBoundary fallbackTitle="Experiment panel unavailable">
        <ThrowingChild />
      </ErrorBoundary>,
    );

    expect(
      screen.getByRole("heading", {
        name: "Experiment panel unavailable",
      }),
    ).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Report this issue" }),
    ).toHaveAttribute(
      "href",
      "https://github.com/Adit-Jain-srm/NightmareNet/issues/new/choose",
    );
  });

  it("invokes the reset callback when Retry is selected", () => {
    const onReset = vi.fn();

    render(
      <ErrorBoundary onReset={onReset}>
        <ThrowingChild />
      </ErrorBoundary>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(onReset).toHaveBeenCalledOnce();
  });

  it("reports the captured error through onError", () => {
    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError}>
        <ThrowingChild />
      </ErrorBoundary>,
    );

    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0]).toEqual(
      expect.objectContaining({ message: "Panel failed" }),
    );
  });
});
