"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { askCopilot, type CopilotSuggestion } from "../../lib/api";
import { Button } from "../ui/Button";
import { IconCommand, IconRunning, IconWand } from "./icons";
import type { DashboardSectionKey } from "./Sidebar";

interface DockSuggestion {
  id: string;
  label: string;
  detail: string;
  action: () => void;
  /** Raw server-side action key (section navigation target). */
  actionKey: string;
}

interface ContextualAnswer {
  hint: string;
  next: DockSuggestion[];
}

const SECTION_KEYS: ReadonlySet<DashboardSectionKey> = new Set([
  "command-center",
  "experiments",
  "run-detail",
  "phases",
  "metrics",
  "robustness",
  "compare",
  "distortions",
  "audit",
  "benchmarks",
  "ci",
  "settings",
]);

function isSectionKey(value: string): value is DashboardSectionKey {
  return SECTION_KEYS.has(value as DashboardSectionKey);
}

function suggestionIcon(actionKey: string) {
  return actionKey.includes("bench") ||
    actionKey.includes("metrics") ||
    actionKey.includes("run") ? (
    <IconRunning size={13} />
  ) : (
    <IconWand size={13} />
  );
}

/**
 * Context-aware copilot dock.
 *
 * v2: streams answers from `/api/v1/copilot/ask` (SSE). Falls back to the
 * deterministic heuristic when the network fails or the API returns a
 * non-200 — the dock never appears broken even offline.
 *
 * The server is the source of truth for hints + suggestions; the heuristic
 * here is identical to the server-side heuristic but kept on the client as
 * an offline-safety net only.
 */
function buildHeuristicAnswer(
  section: DashboardSectionKey,
  navigate: (s: DashboardSectionKey) => void,
): ContextualAnswer {
  const make = (
    id: string,
    label: string,
    actionKey: DashboardSectionKey,
    detail: string,
  ): DockSuggestion => ({
    id,
    label,
    detail,
    actionKey,
    action: () => navigate(actionKey),
  });
  switch (section) {
    case "command-center":
      return {
        hint: "Welcome back. Your last cycle improved robustness by +13.6% — keep going with the next benchmark or stress-test an unseen attack.",
        next: [
          make(
            "run-bench",
            "Run standard benchmark",
            "benchmarks",
            "DistilBERT · SST-2 · 4-phase cycle",
          ),
          make(
            "stress",
            "Stress test current model",
            "distortions",
            "Sweep dream + nightmare 0.1-0.9",
          ),
        ],
      };
    case "experiments":
      return {
        hint: "Compare your two most recent runs side-by-side to see which configuration is converging fastest.",
        next: [
          make(
            "compare",
            "Open Model Comparison",
            "compare",
            "A/B overlay of latest two runs",
          ),
        ],
      };
    case "run-detail":
      return {
        hint: "This run is in the Nightmare phase. Open the radar to see which attack family it's least robust against — that's where the next cycle should focus.",
        next: [
          make(
            "radar",
            "Inspect robustness radar",
            "robustness",
            "5-axis weakness map",
          ),
        ],
      };
    case "distortions":
      return {
        hint: "Try the same input across strengths 0.1, 0.5, and 0.9 to see how nightmare distortion escalates — and where your model's decision boundary breaks.",
        next: [
          make(
            "metrics",
            "Watch live metrics",
            "metrics",
            "Loss + robustness curves",
          ),
        ],
      };
    case "robustness":
      return {
        hint: "Your weakest axis is semantic distortion at high strength. Schedule a Nightmare-heavy cycle to harden it.",
        next: [
          make(
            "phases",
            "Open Phase Visualizer",
            "phases",
            "Tune nightmare strength schedule",
          ),
        ],
      };
    case "ci":
      return {
        hint: "The robustness-check Action is wired. Set your threshold to your model's current avg distorted accuracy minus 0.02 to catch regressions without false alarms.",
        next: [
          make(
            "settings",
            "Open Settings",
            "settings",
            "Manage API keys + thresholds",
          ),
        ],
      };
    case "audit":
      return {
        hint: "Filter by error events to triage failures faster — most regressions cluster in the first two cycles after a config change.",
        next: [],
      };
    default:
      return {
        hint: "Tip: press Cmd+K to jump anywhere, or ? to see every shortcut.",
        next: [],
      };
  }
}

interface AskNightmareDockProps {
  section: DashboardSectionKey;
  onNavigate: (s: DashboardSectionKey) => void;
}

interface AskState {
  status: "idle" | "streaming" | "done" | "error";
  text: string;
  suggestions: DockSuggestion[] | null;
  model: string | null;
  error: string | null;
}

export function AskNightmareDock({ section, onNavigate }: AskNightmareDockProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [ask, setAsk] = useState<AskState>({
    status: "idle",
    text: "",
    suggestions: null,
    model: null,
    error: null,
  });
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const heuristic = useMemo(
    () => buildHeuristicAnswer(section, onNavigate),
    [section, onNavigate],
  );

  const navigateFromAction = useCallback(
    (actionKey: string) => {
      if (isSectionKey(actionKey)) onNavigate(actionKey);
    },
    [onNavigate],
  );

  const toDockSuggestions = useCallback(
    (raw: CopilotSuggestion[]): DockSuggestion[] =>
      raw.map((s, idx) => ({
        id: `${s.action}-${idx}`,
        label: s.label,
        detail: s.detail,
        actionKey: s.action,
        action: () => navigateFromAction(s.action),
      })),
    [navigateFromAction],
  );

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => inputRef.current?.focus(), 50);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // Reset streamed state when section changes so context is always fresh.
  useEffect(() => {
    setAsk({
      status: "idle",
      text: "",
      suggestions: null,
      model: null,
      error: null,
    });
  }, [section]);

  const handleAsk = useCallback(async () => {
    const q = query.trim();
    if (!q) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setAsk({
      status: "streaming",
      text: "",
      suggestions: null,
      model: null,
      error: null,
    });

    try {
      let acc = "";
      for await (const evt of askCopilot(q, section, undefined, controller.signal)) {
        if ("token" in evt) {
          acc += evt.token;
          setAsk((prev) =>
            prev.status === "streaming" ? { ...prev, text: acc } : prev,
          );
        } else {
          setAsk({
            status: "done",
            text: acc || "",
            suggestions: toDockSuggestions(evt.suggestions),
            model: evt.model,
            error: null,
          });
          break;
        }
      }
      // Keep focus on input so the next question feels instant.
      inputRef.current?.focus();
    } catch (err) {
      if (controller.signal.aborted) return;
      const msg = err instanceof Error ? err.message : "copilot unavailable";
      setAsk({
        status: "error",
        text: heuristic.hint,
        suggestions: heuristic.next,
        model: "heuristic (offline)",
        error: msg,
      });
      inputRef.current?.focus();
    }
  }, [heuristic, query, section, toDockSuggestions]);

  const isStreaming = ask.status === "streaming";
  const showAnswer =
    ask.status === "streaming" ||
    ask.status === "done" ||
    ask.status === "error";

  const displayedHint = showAnswer ? ask.text : heuristic.hint;
  const displayedSuggestions: DockSuggestion[] =
    ask.suggestions !== null ? ask.suggestions : heuristic.next;
  const modelLabel = ask.model ?? null;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label="Ask NightmareNet copilot"
        aria-expanded={open}
        className="fixed bottom-5 right-5 z-[45] flex h-12 w-12 cursor-pointer items-center justify-center rounded-full border border-white/10 bg-gradient-to-br from-neural/30 to-dream/20 text-neural shadow-[0_8px_28px_rgba(6,182,212,0.32)] transition-transform hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/60"
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
          <path
            d="M3 14V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H6.5L3 14Z"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinejoin="round"
          />
          <circle cx="6.8" cy="7.5" r="0.8" fill="currentColor" />
          <circle cx="9" cy="7.5" r="0.8" fill="currentColor" />
          <circle cx="11.2" cy="7.5" r="0.8" fill="currentColor" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, x: 24, y: 8 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            exit={{ opacity: 0, x: 24, y: 8 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="fixed bottom-20 right-5 z-[46] w-full max-w-sm overflow-hidden rounded-2xl border border-white/[0.08] bg-abyss/95 shadow-[0_24px_60px_rgba(0,0,0,0.6)] backdrop-blur-xl"
            role="region"
            aria-label="NightmareNet copilot"
          >
            <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-md bg-neural/[0.15] text-neural">
                  <IconCommand size={12} />
                </span>
                <span className="text-sm font-semibold text-slate-100">Ask NightmareNet</span>
              </div>
              <button
                type="button"
                onClick={() => {
                  abortRef.current?.abort();
                  setOpen(false);
                }}
                aria-label="Close copilot"
                className="cursor-pointer rounded-md px-1.5 py-0.5 text-[11px] text-slate-400 hover:bg-white/5 hover:text-slate-300"
              >
                Esc
              </button>
            </div>

            <div className="space-y-3 px-4 py-4">
              <div
                className="rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-2.5 text-[13px] leading-relaxed text-slate-300"
                aria-live="polite"
                aria-busy={isStreaming}
              >
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-neural">
                  Context · {section}
                  {ask.status === "streaming" && (
                    <span className="ml-2 inline-flex items-center gap-1 normal-case tracking-normal text-slate-400">
                      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-neural" />
                      thinking…
                    </span>
                  )}
                </p>
                <span className="whitespace-pre-wrap">
                  {displayedHint}
                  {isStreaming && (
                    <span
                      aria-hidden="true"
                      className="ml-0.5 inline-block h-3 w-[2px] translate-y-[2px] animate-pulse bg-neural"
                    />
                  )}
                </span>
              </div>

              {displayedSuggestions.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-300">
                    Suggested next steps
                  </p>
                  {displayedSuggestions.map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => {
                        s.action();
                        setOpen(false);
                      }}
                      className="group flex w-full cursor-pointer items-center gap-3 rounded-md border border-white/[0.05] bg-white/[0.02] px-3 py-2 text-left transition-colors hover:border-neural/30 hover:bg-neural/[0.05]"
                    >
                      <span className="flex h-7 w-7 items-center justify-center rounded-md bg-white/[0.04] text-slate-400 group-hover:bg-neural/[0.12] group-hover:text-neural">
                        {suggestionIcon(s.actionKey)}
                      </span>
                      <span className="flex-1">
                        <span className="block text-[13px] text-slate-200">{s.label}</span>
                        <span className="block text-[11px] text-slate-400">{s.detail}</span>
                      </span>
                    </button>
                  ))}
                </div>
              )}

              <div>
                <p className="pb-1.5 text-[10px] font-semibold uppercase tracking-widest text-slate-300">
                  Ask anything
                </p>
                <div className="flex items-center gap-2 rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-2">
                  <input
                    ref={inputRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !isStreaming) {
                        e.preventDefault();
                        void handleAsk();
                      }
                    }}
                    placeholder="e.g. compare last two runs"
                    aria-label="Ask the NightmareNet copilot"
                    disabled={isStreaming}
                    className="flex-1 bg-transparent text-[13px] text-slate-200 placeholder:text-slate-300 focus:outline-none disabled:opacity-60"
                  />
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => void handleAsk()}
                    disabled={isStreaming || query.trim().length === 0}
                  >
                    {isStreaming ? "…" : "Ask"}
                  </Button>
                </div>
                <p className="flex items-center justify-between pt-1.5 text-[10px] text-slate-300">
                  <span>Streaming SSE · context-aware</span>
                  {modelLabel && (
                    <span className="font-mono text-slate-400">
                      powered by {modelLabel}
                    </span>
                  )}
                </p>
                {ask.status === "error" && (
                  <p
                    className="pt-1 text-[10px] text-amber-400/80"
                    role="status"
                  >
                    Live copilot unavailable — showing offline suggestions.
                  </p>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
