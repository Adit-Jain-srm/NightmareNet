import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const routes = [
  { name: "landing page", path: "/" },
  { name: "dashboard", path: "/dashboard" },
];

for (const route of routes) {
  test(`${route.name} has no automatically detectable WCAG A/AA violations`, async ({
    page,
  }) => {
    await page.goto(route.path, {
  waitUntil: "domcontentloaded",
});

await page.locator("main").waitFor({
  state: "visible",
});
await page.waitForTimeout(500);

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();

    if (results.violations.length > 0) {
  console.error(
    JSON.stringify(
      results.violations.map((violation) => ({
        id: violation.id,
        impact: violation.impact,
        help: violation.help,
        nodes: violation.nodes.map((node) => ({
          target: node.target,
          html: node.html,
          failureSummary: node.failureSummary,
        })),
      })),
      null,
      2,
    ),
  );
}

expect(results.violations).toEqual([]);
  });
}
