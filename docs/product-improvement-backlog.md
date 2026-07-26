# Product Improvement Backlog

Tracked through the `consumer-product-improvement` skill's *Analyze → Critique → Reimagine → Stress Test → Improve → Refine* loop. Append entries; never delete (mark `[shipped]` or `[deferred]` instead).

## Shipped this iteration (2026-05-26)

- **[shipped]** First-run `OnboardingOverlay` — 4-step tour with localStorage-gated dismissal and "Try it now" CTA jumping to Distortion Preview.
- **[shipped]** Global keyboard shortcut vocabulary — `Cmd/Ctrl+K`, `?`, `Esc`, plus `g`-prefix navigation across all 12 sections. Suppressed in text inputs.
- **[shipped]** `KeyboardHelp` overlay — grouped shortcut catalog, ? to toggle.
- **[shipped]** Fuzzy palette ranking — prefix > substring > subsequence scoring, recency bonus, dedicated "Recent" group when query is empty.
- **[shipped]** `AskNightmareDock` — floating-button context-aware copilot with per-section hints and 1-click next-step suggestions. Heuristic v1; LLM streaming endpoint is the next swap-in.
- **[shipped]** `ToastProvider` wired globally — every palette/g-shortcut/action fires structured toast feedback (variant + duration).

## Shipped this iteration (2026-05-27)

- **[shipped]** `EmptyState` primitive — intelligent CTAs, constellation illustration, used in `ExperimentList` (0-runs) and `AuditTrail` (0-events).
- **[shipped]** `SkeletonStatTile` / `SkeletonChart` / `SkeletonRows` — shape-matched loaders; `CommandCenter`, `LiveMetrics`, `ExperimentList` opt-in via `loading` prop.
- **[shipped]** `RowActionsMenu` on `ExperimentList` rows — Compare / Re-run / Export / Open / Delete with toast feedback. `IconKebab` added to the icon set.
- **[shipped]** Marketing → `/dashboard` text handoff — `frontend/src/lib/handoff.ts` persists demo input to `sessionStorage`; `DistortionPreview` prefills on mount and fires an info toast.
- **[shipped]** `/api/v1/copilot/ask` SSE endpoint — heuristic by default, auto-upgrades to OpenAI / Anthropic / Azure OpenAI when keys are set. Unified `{answer, suggestions, model}` shape; `register_copilot_routes(app, limiter)` to share slowapi limiter.
- **[shipped]** `askCopilot()` typed async generator in API client + `AskNightmareDock` wired to it — streaming with thinking cursor, "powered by" footer, abort-on-close, heuristic fallback on network failure.
- **[shipped]** `/api/v1/badge/{score}.svg` + `.json` — shields.io-style robustness badge with 5-band color scale, `Cache-Control: max-age=300`. Embed section in `CIIntegration` panel with copy-to-clipboard HTML and Markdown snippets.
- **[shipped]** `RunDetail` quick-re-run menu — Same / Strength × 1.2 / Strength × 0.8 / Switch to GPT-2 with inline config-diff preview.
- **[shipped]** `WhatsNew` card — top-right of dashboard, build-SHA-gated, suppressed until onboarding is dismissed.
- **[shipped]** OpenAPI documentation for copilot + badge endpoints.
- **[shipped]** **Fixed tautological worker HEALTHCHECK** — real `healthcheck_worker.py` with TCP broker probe + Celery worker ping. 14 regression tests lock the behavior. (See `tasks/lessons.md` 2026-05-27.)

## Shipped this iteration (2026-06-01)

- **[shipped]** Theme toggle (dark/light/system) — `ThemeProvider` context with `localStorage` persistence, CSS variable swapping, light mode glassmorphism overrides, 3-state toggle in Navbar (desktop + mobile).
- **[shipped]** Custom NightmareNet SVG logo — orbital rings, neural core, 4 phase-colored dots, `useId()` for unique gradient IDs.
- **[shipped]** Dashboard navigation from landing page — Hero CTA, Navbar button, Footer link, Playground/Demo contextual links.
- **[shipped]** GSAP animations — floating gradient orbs in Hero using `useGSAP` with proper scope and cleanup.
- **[shipped]** WebSocket live progress — `/ws/runs/{run_id}` endpoint in OSS app, PipelineLab auto-upgrades from polling to WS.
- **[shipped]** 18 code review fixes — critical runtime crash, stale metrics, dead code, license mismatch, deprecated APIs.
- **[shipped]** Repo cleanup — removed 8,400+ lines of junk (duplicate skill dirs, compiled JS in source, dead components, stale docs).
- **[shipped]** README rewrite — real benchmark results, 660+ test badge, frontend section, removed stale PyPI badge.

## Top of backlog (next iteration)

| Rank | Opportunity | Persona | Effort | Notes |
|-----:|-------------|---------|:------:|-------|
| 1 | Per-user palette command history sync | Power user | S | Currently localStorage; sync to server store when logged in. |
| 2 | Wire `RowActionsMenu` actions to real endpoints | Power user | M | "Compare" → `/api/v1/experiments/compare`; "Re-run" → `/api/v1/pipeline/create` with mutated config; "Export" → JSON download. |
| 3 | Wire `RunDetail` re-run-with-mutation to `/api/v1/pipeline/create` | Power user | S | Currently a toast + TODO comment. |
| 4 | Real "What's new" changelog feed | Daily user | S | Currently three hardcoded bullets — drive from CHANGELOG.md. |
| 5 | Live robustness-score endpoint for badge | Growth | M | Wire `/api/v1/badge/latest.svg` that pulls from the most recent run. |
| 6 | "View changelog" deep link from `WhatsNew` | Daily user | S | Route to GitHub releases or in-app changelog modal. |
| 7 | Inline-edit experiment names | Power user | S | Click-to-edit row title in `ExperimentList` with optimistic update. |
| 8 | Empty states across remaining panels (`RobustnessRadar`, `ModelComparison`, `BenchmarkSuite`) | First-time user | S | EmptyState primitive is built; just wire it. |
| 9 | Mobile-responsive sidebar collapse | Mobile user | M | Below `md`, sidebar collapses to bottom navbar or drawer. |
| 10 | Voice mode for copilot dock | Wow / accessibility | M | Web Speech API push-to-talk → POST to `/api/v1/copilot/ask`. |
| 11 | Personalized dashboard ordering | Daily user | M | Track section-visit counts in localStorage; reorder sidebar by frequency. |

## Deeper opportunities (research / experiment)

- **Memory system across sessions** — the dashboard "remembers" which panels each user views most, and reorders the sidebar to surface them first.
- **Semantic search across runs + audit log** — "show me runs where dream@0.7 dropped below 0.6" without writing a query.
- **Slack/Discord integration** — push run-complete + regression-detected events.
- **Compare to industry baseline** — anonymized aggregate from the hosted platform: "your robustness is in the 73rd percentile for DistilBERT/SST-2."
- **Cycle scheduling** — natural-language schedule input ("run a benchmark every Friday at 6pm") routed through Celery beat.
- **Mobile companion view** — read-only run-status view optimized for phones; push notifications.
- **Robustness diff badges in PR comments** — the existing composite Action could write a PR comment with a sparkline diff vs main.
- **Replay mode** — scrub through a run's timeline as if it were a video, with phase boundaries and loss-curve overlay.
- **Distortion DSL** — let power users compose custom distortion chains in YAML, with live preview.

## Review cadence

This file is re-scored after every shipped iteration. Top of backlog moves into "Shipped this iteration" only with: tests passing, lint clean, frontend build green, manual UX walkthrough recorded.
