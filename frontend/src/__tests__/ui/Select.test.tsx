import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { Select, type SelectOption } from "@/components/ui/Select";

// Mock framer-motion
vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  motion: {
    ul: React.forwardRef(function MockMotionUl(
      { children, initial, animate, exit, transition, ...props }: Record<string, unknown>,
      ref: React.Ref<HTMLUListElement>
    ) {
      void initial;
      void animate;
      void exit;
      void transition;
      return React.createElement(
        "ul",
        { ...(props as Record<string, unknown>), ref },
        children as React.ReactNode
      );
    }),
  },
}));

const sampleOptions: SelectOption[] = [
  { value: "gpt2", label: "GPT-2 Base", hint: "124M params" },
  { value: "gpt2-medium", label: "GPT-2 Medium", hint: "355M params" },
  { value: "gpt2-large", label: "GPT-2 Large", hint: "774M params" },
];

describe("Select Component", () => {
  it("renders label, placeholder, and initial selected value", () => {
    render(
      <Select
        label="Target Architecture"
        value="gpt2"
        options={sampleOptions}
        onChange={() => {}}
      />
    );

    expect(screen.getByText("Target Architecture")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /gpt-2 base/i })).toBeInTheDocument();
  });

  it("renders placeholder when value does not match options", () => {
    render(
      <Select
        value=""
        options={sampleOptions}
        placeholder="Choose a model..."
        onChange={() => {}}
      />
    );

    expect(screen.getByText("Choose a model...")).toBeInTheDocument();
  });

  it("opens options listbox dropdown when button is clicked", () => {
    render(
      <Select
        value="gpt2"
        options={sampleOptions}
        onChange={() => {}}
      />
    );

    const trigger = screen.getByRole("button", { name: /gpt-2 base/i });
    expect(trigger).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("listbox")).toBeInTheDocument();

    const options = screen.getAllByRole("option");
    expect(options.length).toBe(3);
    expect(options[0]).toHaveAttribute("aria-selected", "true");
    expect(options[1]).toHaveAttribute("aria-selected", "false");
  });

  it("calls onChange and closes dropdown when an option is selected", () => {
    const onChange = vi.fn();
    render(
      <Select
        value="gpt2"
        options={sampleOptions}
        onChange={onChange}
      />
    );

    const trigger = screen.getByRole("button", { name: /gpt-2 base/i });
    fireEvent.click(trigger);

    const mediumOption = screen.getByRole("button", { name: /gpt-2 medium/i });
    fireEvent.click(mediumOption);

    expect(onChange).toHaveBeenCalledWith("gpt2-medium");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("closes dropdown when Escape key is pressed", () => {
    render(
      <Select
        value="gpt2"
        options={sampleOptions}
        onChange={() => {}}
      />
    );

    const trigger = screen.getByRole("button", { name: /gpt-2 base/i });
    fireEvent.click(trigger);
    expect(screen.getByRole("listbox")).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("closes dropdown when clicking outside", () => {
    render(
      <div>
        <div data-testid="outside-element">Outside</div>
        <Select
          value="gpt2"
          options={sampleOptions}
          onChange={() => {}}
        />
      </div>
    );

    const trigger = screen.getByRole("button", { name: /gpt-2 base/i });
    fireEvent.click(trigger);
    expect(screen.getByRole("listbox")).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByTestId("outside-element"));
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("is disabled when disabled prop is set", () => {
    const onChange = vi.fn();
    render(
      <Select
        value="gpt2"
        options={sampleOptions}
        disabled={true}
        onChange={onChange}
      />
    );

    const trigger = screen.getByRole("button", { name: /gpt-2 base/i });
    expect(trigger).toBeDisabled();
    fireEvent.click(trigger);
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("renders hints for options inside the dropdown", () => {
    render(
      <Select
        value="gpt2"
        options={sampleOptions}
        onChange={() => {}}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /gpt-2 base/i }));
    expect(screen.getByText("124M params")).toBeInTheDocument();
    expect(screen.getByText("355M params")).toBeInTheDocument();
    expect(screen.getByText("774M params")).toBeInTheDocument();
  });
});
