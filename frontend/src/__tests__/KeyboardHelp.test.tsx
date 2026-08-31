import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { KeyboardHelp } from "@/components/dashboard/KeyboardHelp";

vi.mock("@/components/dashboard/../a11y/useDialogFocus", () => ({
  useDialogFocus: vi.fn(),
}));

describe("KeyboardHelp", () => {
  it("renders the keyboard shortcuts modal when open", () => {
    render(<KeyboardHelp open={true} onClose={vi.fn()} />);

    expect(
      screen.getByRole("dialog", { name: /keyboard shortcuts/i }),
    ).toBeInTheDocument();

    expect(screen.getByText("Global")).toBeInTheDocument();
    expect(screen.getByText("Navigation (press g, then …)")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", () => {
    const onClose = vi.fn();

    render(<KeyboardHelp open={true} onClose={onClose} />);

    fireEvent.click(
      screen.getAllByRole("button", {
        name: /close keyboard shortcuts/i,
      })[0],
    );

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not render the modal when closed", () => {
    render(<KeyboardHelp open={false} onClose={vi.fn()} />);

    expect(
      screen.queryByRole("dialog", { name: /keyboard shortcuts/i }),
    ).not.toBeInTheDocument();
  });
});
