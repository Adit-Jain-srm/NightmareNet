import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";

interface TestRow {
  id: string;
  name: string;
  score: number;
  status: string;
}

const sampleColumns: DataTableColumn<TestRow>[] = [
  {
    key: "id",
    header: "ID",
    accessor: (r) => r.id,
    sortable: true,
  },
  {
    key: "name",
    header: "Name",
    accessor: (r) => r.name,
    sortable: true,
  },
  {
    key: "score",
    header: "Score",
    accessor: (r) => r.score,
    sortable: true,
    align: "right",
    cell: (r) => <span data-testid={`score-${r.id}`}>{r.score}%</span>,
  },
  {
    key: "status",
    header: "Status",
    accessor: (r) => r.status,
    sortable: false,
  },
];

const sampleRows: TestRow[] = [
  { id: "1", name: "Model Beta", score: 85, status: "Active" },
  { id: "2", name: "Model Alpha", score: 92, status: "Pending" },
  { id: "3", name: "Model Gamma", score: 78, status: "Archived" },
];

describe("DataTable Component", () => {
  it("renders table headers and correct number of data rows", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
      />
    );

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("ID")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Score")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();

    expect(screen.getByText("Model Beta")).toBeInTheDocument();
    expect(screen.getByText("Model Alpha")).toBeInTheDocument();
    expect(screen.getByText("Model Gamma")).toBeInTheDocument();
  });

  it("renders custom cell content via column.cell renderer", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
      />
    );

    expect(screen.getByTestId("score-1")).toHaveTextContent("85%");
    expect(screen.getByTestId("score-2")).toHaveTextContent("92%");
    expect(screen.getByTestId("score-3")).toHaveTextContent("78%");
  });

  it("displays empty state message when rows array is empty", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={[]}
        rowKey={(r) => r.id}
        empty="No models found"
      />
    );

    expect(screen.getByText("No models found")).toBeInTheDocument();
  });

  it("displays default 'No data' text when empty prop is not provided", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={[]}
        rowKey={(r) => r.id}
      />
    );

    expect(screen.getByText("No data")).toBeInTheDocument();
  });

  it("shows skeleton loading rows when isLoading=true", () => {
    const { container } = render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
        isLoading={true}
        loadingRows={4}
      />
    );

    const shimmers = container.querySelectorAll(".animate-shimmer");
    expect(shimmers.length).toBe(4 * sampleColumns.length);
    expect(screen.queryByText("Model Beta")).not.toBeInTheDocument();
  });

  it("sorts by column ascending, descending, then resets on consecutive clicks", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
      />
    );

    const nameHeader = screen.getByText("Name").closest("th");
    expect(nameHeader).toBeInTheDocument();
    expect(nameHeader).toHaveAttribute("aria-sort", "none");

    // First click: Sort Ascending
    fireEvent.click(nameHeader!);
    expect(nameHeader).toHaveAttribute("aria-sort", "ascending");
    let tableCells = screen.getAllByRole("cell");
    let firstRowName = tableCells.find((c) => c.textContent?.includes("Model Alpha"));
    expect(firstRowName).toBeInTheDocument();

    // Second click: Sort Descending
    fireEvent.click(nameHeader!);
    expect(nameHeader).toHaveAttribute("aria-sort", "descending");

    // Third click: Reset sorting
    fireEvent.click(nameHeader!);
    expect(nameHeader).toHaveAttribute("aria-sort", "none");
  });

  it("does not sort non-sortable columns", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
      />
    );

    const statusHeader = screen.getByText("Status").closest("th");
    expect(statusHeader).toBeInTheDocument();
    fireEvent.click(statusHeader!);
    expect(statusHeader).toHaveAttribute("aria-sort", "none");
  });

  it("triggers onRowClick when a row is clicked", () => {
    const onRowClick = vi.fn();
    render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
        onRowClick={onRowClick}
      />
    );

    const targetRow = screen.getByText("Model Beta").closest("tr");
    expect(targetRow).toBeInTheDocument();
    fireEvent.click(targetRow!);
    expect(onRowClick).toHaveBeenCalledWith(sampleRows[0]);
  });

  it("respects initialSort prop", () => {
    render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
        initialSort={{ key: "score", direction: "desc" }}
      />
    );

    const scoreHeader = screen.getByText("Score").closest("th");
    expect(scoreHeader).toHaveAttribute("aria-sort", "descending");
  });

  it("supports compact density styling", () => {
    const { container } = render(
      <DataTable
        columns={sampleColumns}
        rows={sampleRows}
        rowKey={(r) => r.id}
        density="compact"
      />
    );

    expect(container.querySelector(".py-1\\.5")).toBeInTheDocument();
  });
});
