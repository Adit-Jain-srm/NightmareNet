# NightmareNet: research synthesis (2026-04-22)

## Parallel deep research (CLI)

`parallel-cli` was not available in this environment (`command not found`). To run the full flow from the plan, install with `/parallel-setup` (or your org’s install path), then:

```bash
parallel-cli research run "<question>" --processor pro-fast --no-wait --json
parallel-cli research poll "<RUN_ID>" -o "nightmarenet-research-landscape" --timeout 540
```

## Manually synthesized takeaways (until CLI runs)

**Product / positioning**

- “Sleep” or multi-phase (wake → dream → stress → compress) training sits alongside continual learning, curriculum learning, and adversarial training; the differentiator is **narrative + explicit phase control** and **in-product distortion engines** (dream vs nightmare) rather than a single loss.
- **API-first demos** (distortion, robustness, E2E pipeline) match how buyers evaluate MLOps-style tools: fast time-to-first-result beats roadmap slides.

**Engineering (FastAPI + long runs)**

- In-process **threaded** runners with a **bounded registry** avoid unbounded memory from abandoned runs; **cancellation** remains cooperative (check between stages).
- **Health checks** in production should not shell out to `pytest`; optional dev-only test counts behind an env flag is standard.

**Implications for this repo**

- Keep **CORS** and **Next rewrites** aligned so the UI can use same-origin `/api/...` in production.
- Treat **App Insights** / telemetry as a separate **azure-prepare** pass when you pick Azure hosting.
- For GTM list enrichment, **parallel-data-enrichment** is optional and unrelated to the training core.
