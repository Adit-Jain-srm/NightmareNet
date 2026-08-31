import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import ContributorsPage from "./page";
import { vi } from "vitest";

describe("Contributors page", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders contributors after fetching", async () => {
    const fake = [
      {
        login: "alice",
        avatar_url: "https://example.com/a.png",
        html_url: "https://github.com/alice",
        contributions: 42,
        prs: 5,
        issues: 3,
      },
    ];

    vi.stubGlobal("fetch", vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(fake) } as any)
    ));

    render(<ContributorsPage />);

    // wait for the contributor login to appear
    await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());
    expect(screen.getByText("42 contributions")).toBeInTheDocument();
  });

  it("shows error when fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({ ok: false, status: 500 } as any)));

    render(<ContributorsPage />);

    await waitFor(() => expect(screen.getByText(/Failed to load contributors/i)).toBeInTheDocument());
  });
});
