"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  IconCommand,
  IconChevronRight,
  IconHome,
  IconBeaker,
  IconActivity,
  IconRadar,
  IconLayers,
  IconHistory,
  IconBenchmark,
  IconGit,
  IconSettings,
  IconRunning,
  IconWand,
  IconTrend,
} from "./icons";
import type { DashboardSectionKey } from "./Sidebar";
import { useDialogFocus } from "../a11y/useDialogFocus";
import {
  loadRecentPaletteIds,
  pushRecentPaletteId,
  rankPaletteItems,
} from "./useGlobalShortcuts";

interface PaletteItem {
  id: string;
  label: string;
  hint?: string;
  group: "Navigate" | "Actions" | "Tools";
  icon: React.ReactNode;
  onRun: () => void;
}

export interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onNavigate: (key: DashboardSectionKey) => void;
  onAction?: (action: string) => void;
}

export function CommandPalette({
  open,
  onClose,
  onNavigate,
  onAction,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const getInitialFocus = useCallback(() => inputRef.current, []);
  const dialogRef = useDialogFocus(open, onClose, getInitialFocus);

  const items: PaletteItem[] = useMemo(
    () => [
      { id: "nav-cc", label: "Go to Command Center", hint: "Overview", group: "Navigate", icon: <IconHome size={14} />, onRun: () => onNavigate("command-center") },
      { id: "nav-exp", label: "Go to Experiments", hint: "Run history", group: "Navigate", icon: <IconBeaker size={14} />, onRun: () => onNavigate("experiments") },
      { id: "nav-run", label: "Open Run Detail", hint: "Phase tabs", group: "Navigate", icon: <IconRunning size={14} />, onRun: () => onNavigate("run-detail") },
      { id: "nav-phases", label: "Phase Visualizer", hint: "4-phase ring", group: "Navigate", icon: <IconLayers size={14} />, onRun: () => onNavigate("phases") },
      { id: "nav-metrics", label: "Live Metrics", hint: "Loss & robustness", group: "Navigate", icon: <IconActivity size={14} />, onRun: () => onNavigate("metrics") },
      { id: "nav-radar", label: "Robustness Radar", hint: "5-axis", group: "Navigate", icon: <IconRadar size={14} />, onRun: () => onNavigate("robustness") },
      { id: "nav-cmp", label: "Model Comparison", hint: "A/B overlay", group: "Navigate", icon: <IconTrend size={14} />, onRun: () => onNavigate("compare") },
      { id: "nav-dis", label: "Distortion Preview", hint: "Dream vs Nightmare", group: "Navigate", icon: <IconWand size={14} />, onRun: () => onNavigate("distortions") },
      { id: "nav-audit", label: "Audit Trail", hint: "Events", group: "Navigate", icon: <IconHistory size={14} />, onRun: () => onNavigate("audit") },
      { id: "nav-bench", label: "Benchmark Suite", hint: "Run benchmarks", group: "Navigate", icon: <IconBenchmark size={14} />, onRun: () => onNavigate("benchmarks") },
      { id: "nav-ci", label: "CI Integration", hint: "GitHub Action", group: "Navigate", icon: <IconGit size={14} />, onRun: () => onNavigate("ci") },
      { id: "nav-set", label: "Settings", hint: "API keys & config", group: "Navigate", icon: <IconSettings size={14} />, onRun: () => onNavigate("settings") },
      { id: "act-new-run", label: "Start new training run", hint: "Pipeline wizard", group: "Actions", icon: <IconRunning size={14} />, onRun: () => onAction?.("new-run") },
      { id: "act-cancel", label: "Cancel active run", hint: "Stops latest", group: "Actions", icon: <IconHistory size={14} />, onRun: () => onAction?.("cancel-run") },
      { id: "act-export", label: "Export report (JSON)", hint: "Latest run", group: "Actions", icon: <IconCommand size={14} />, onRun: () => onAction?.("export") },
      { id: "tool-distort", label: "Quick distort sample", hint: "Open distortions", group: "Tools", icon: <IconWand size={14} />, onRun: () => onNavigate("distortions") },
      { id: "tool-bench-mr", label: "Run MR benchmark suite", hint: "Standard pack", group: "Tools", icon: <IconBenchmark size={14} />, onRun: () => onAction?.("bench-mr") },
    ],
    [onNavigate, onAction]
  );

  const [recentIds, setRecentIds] = useState<string[]>([]);
  useEffect(() => {
    if (open) setRecentIds(loadRecentPaletteIds());
  }, [open]);

  const filtered = useMemo(
    () => rankPaletteItems(items, query, recentIds),
    [items, query, recentIds]
  );

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIdx(0);
      const t = setTimeout(() => inputRef.current?.focus(), 50);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIdx((i) => Math.min(i + 1, filtered.length - 1));
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIdx((i) => Math.max(i - 1, 0));
      }
      if (e.key === "Enter") {
        e.preventDefault();
        const item = filtered[activeIdx];
        if (item) runItem(item);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, filtered, activeIdx, onClose]);

  useEffect(() => {
    setActiveIdx(0);
  }, [query]);

  const grouped = useMemo(() => {
    const map = new Map<string, PaletteItem[]>();
    // When no query, surface a synthetic "Recent" group at the top.
    if (!query.trim() && recentIds.length > 0) {
      const idToItem = new Map(items.map((it) => [it.id, it] as const));
      const recents = recentIds
        .map((id) => idToItem.get(id))
        .filter((x): x is PaletteItem => Boolean(x));
      if (recents.length > 0) map.set("Recent", recents);
    }
    for (const it of filtered) {
      const arr = map.get(it.group) ?? [];
      arr.push(it);
      map.set(it.group, arr);
    }
    return Array.from(map.entries());
  }, [filtered, items, recentIds, query]);

  const runItem = (it: PaletteItem) => {
    pushRecentPaletteId(it.id);
    it.onRun();
    onClose();
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          className="fixed inset-0 z-[55] flex items-start justify-center px-4 pt-24"
        >
          <button
            type="button"
            className="absolute inset-0 cursor-default bg-void/80 backdrop-blur-sm"
            onClick={onClose}
            aria-label="Close command palette"
          />
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            tabIndex={-1}
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.16, ease: "easeOut" }}
            className="relative w-full max-w-xl overflow-hidden rounded-2xl border border-white/[0.08] bg-abyss/95 shadow-[0_24px_60px_rgba(0,0,0,0.6)] backdrop-blur-xl"
          >
            <div className="flex items-center gap-2 border-b border-white/[0.06] px-3 py-2.5">
              <IconCommand size={14} />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Type a command or search…"
                className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-300 focus:outline-none"
                aria-label="Command palette search"
              />
              <kbd className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
                ESC
              </kbd>
            </div>
            <div className="max-h-[420px] overflow-y-auto py-2">
              {filtered.length === 0 ? (
                <p className="px-4 py-8 text-center text-xs text-slate-400">
                  No commands match <span className="font-mono text-slate-300">&quot;{query}&quot;</span>
                </p>
              ) : (
                grouped.map(([group, list]) => (
                  <div key={group} className="px-2">
                    <p className="px-2 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-widest text-slate-300">
                      {group}
                    </p>
                    {list.map((it) => {
                      const idx = filtered.findIndex((f) => f.id === it.id);
                      const active = idx === activeIdx;
                      return (
                        <button
                          key={`${group}-${it.id}`}
                          type="button"
                          onMouseEnter={() => setActiveIdx(idx)}
                          onClick={() => runItem(it)}
                          className={[
                            "group flex w-full items-center gap-3 rounded-md px-2 py-2 text-left cursor-pointer transition-colors",
                            active
                              ? "bg-neural/[0.10] text-slate-100"
                              : "text-slate-300 hover:bg-white/[0.04]",
                          ].join(" ")}
                        >
                          <span
                            className={[
                              "flex h-7 w-7 items-center justify-center rounded-md",
                              active ? "bg-neural/[0.15] text-neural" : "bg-white/[0.04] text-slate-400",
                            ].join(" ")}
                          >
                            {it.icon}
                          </span>
                          <span className="flex-1 truncate text-[13px]">{it.label}</span>
                          {it.hint && (
                            <span className="text-[10px] uppercase tracking-wider text-slate-400">
                              {it.hint}
                            </span>
                          )}
                          <IconChevronRight size={12} />
                        </button>
                      );
                    })}
                  </div>
                ))
              )}
            </div>
            <div className="flex items-center justify-between border-t border-white/[0.06] px-3 py-2 text-[10px] text-slate-400">
              <span className="flex items-center gap-3">
                <kbd className="rounded bg-white/[0.06] px-1 py-0.5 font-mono">↑↓</kbd> navigate
                <kbd className="rounded bg-white/[0.06] px-1 py-0.5 font-mono">↵</kbd> run
              </span>
              <span>{filtered.length} commands</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
