import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React, { createRef } from "react";
import { Input } from "@/components/ui/Input";

describe("Input Component", () => {
  it("renders with placeholder and controlled value", () => {
    render(
      <Input
        placeholder="Enter model name..."
        value="Nightmare-GPT2"
        onChange={() => {}}
      />
    );

    const input = screen.getByPlaceholderText("Enter model name...");
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue("Nightmare-GPT2");
  });

  it("fires onChange callback when input text changes", () => {
    const onChange = vi.fn();
    render(<Input placeholder="Type here" onChange={onChange} />);

    const input = screen.getByPlaceholderText("Type here");
    fireEvent.change(input, { target: { value: "New Value" } });
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("renders label and associates with input via htmlFor", () => {
    render(<Input label="Batch Size" defaultValue={32} />);

    const label = screen.getByText("Batch Size");
    expect(label).toBeInTheDocument();
    const input = screen.getByLabelText("Batch Size");
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("id", "input-batch-size");
  });

  it("renders custom id when provided", () => {
    render(<Input label="Epochs" id="custom-epochs-id" />);
    const input = screen.getByLabelText("Epochs");
    expect(input).toHaveAttribute("id", "custom-epochs-id");
  });

  it("renders validation error message and applies error styles", () => {
    render(
      <Input
        label="Learning Rate"
        error="Learning rate must be positive"
      />
    );

    const errorMsg = screen.getByText("Learning rate must be positive");
    expect(errorMsg).toBeInTheDocument();
    expect(errorMsg.className).toContain("text-nightmare-soft");

    const input = screen.getByLabelText("Learning Rate");
    expect(input.className).toContain("border-nightmare/60");
  });

  it("renders hint text when no error is present", () => {
    render(
      <Input
        label="Seed"
        hint="Optional integer seed for reproducibility"
      />
    );

    expect(screen.getByText("Optional integer seed for reproducibility")).toBeInTheDocument();
  });

  it("handles disabled state correctly", () => {
    const onChange = vi.fn();
    render(<Input placeholder="Disabled input" disabled onChange={onChange} />);

    const input = screen.getByPlaceholderText("Disabled input");
    expect(input).toBeDisabled();
    expect(input.className).toContain("disabled:cursor-not-allowed");
  });

  it("renders leftIcon and rightSlot elements", () => {
    render(
      <Input
        placeholder="Search..."
        leftIcon={<span data-testid="search-icon">🔍</span>}
        rightSlot={<span data-testid="clear-slot">✕</span>}
      />
    );

    expect(screen.getByTestId("search-icon")).toBeInTheDocument();
    expect(screen.getByTestId("clear-slot")).toBeInTheDocument();
  });

  it("forwards ref to HTMLInputElement", () => {
    const ref = createRef<HTMLInputElement>();
    render(<Input ref={ref} placeholder="Ref test" />);
    expect(ref.current).toBeInstanceOf(HTMLInputElement);
  });
});
