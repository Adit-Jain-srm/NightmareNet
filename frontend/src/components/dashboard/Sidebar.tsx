"use client";

import { useEffect, useMemo } from "react";
import { motion, Reorder } from "framer-motion";
import { useVisitTracker } from "../../hooks/useVisitTracker";
import {
  IconActivity,
  IconBeaker,
  IconBenchmark,
  IconDatabase,
  IconGit,
  IconHistory,
  IconHome,
  IconLayers,
  IconRadar,
  IconRunning,
  IconSettings,
  IconShield,
  IconSparkle,
  IconTrend,
  IconWand,
  IconX,
} from "./icons";

export type DashboardSectionKey =
  | "command-center"
  | "experiments"
  | "run-detail"
  | "phases"
  | "metrics"
  | "robustness"
  | "compare"
  | "distortions"
  | "data-quality"
  | "audit"
  | "benchmarks"
  | "ci"
  | "settings";

interface NavGroup {
  label: string;
  items: { key: DashboardSectionKey; label: string; icon: React.ReactNode; badge?: string }[];
}

const NAV: NavGroup[] = [
  {
    label: "Overview",
    items: [
      { key: "command-center", label: "Command Center", icon: <IconHome size={15} /> },
      { key: "experiments", label: "Experiments", icon: <IconBeaker size={15} />, badge: "12" },
      { key: "run-detail", label: "Run Detail", icon: <IconRunning size={15} /> },
    ],
  },
  {
    label: "Analytics",
    items: [
      { key: "phases", label: "Phase Visualizer", icon: <IconLayers size={15} /> },
      { key: "metrics", label: "Live Metrics", icon: <IconActivity size={15} /> },
      { key: "robustness", label: "Robustness Radar", icon: <IconRadar size={15} /> },
      { key: "compare", label: "Model Compare", icon: <IconTrend size={15} /> },
      { key: "distortions", label: "Distortions", icon: <IconWand size={15} /> },
      { key: "data-quality", label: "Data Quality", icon: <IconDatabase size={15} /> },
    ],
  },
  {
    label: "Operations",
    items: [
      { key: "audit", label: "Audit Trail", icon: <IconHistory size={15} /> },
      { key: "benchmarks", label: "Benchmarks", icon: <IconBenchmark size={15} /> },
      { key: "ci", label: "CI Integration", icon: <IconGit size={15} /> },
      { key: "settings", label: "Settings", icon: <IconSettings size={15} /> },
    ],
  },
];

const ALL_ITEMS = NAV.flatMap((g) => g.items);
const VALID_KEYS = ALL_ITEMS.map((i) => i.key);

export interface SidebarProps {
  activeSection: DashboardSectionKey;
  onSectionChange: (key: DashboardSectionKey) => void;
  collapsed?: boolean;
  mobileMenuOpen?: boolean;
  onMobileMenuClose?: () => void;
}

export function Sidebar({
  activeSection,
  onSectionChange,
  collapsed = false,
  mobileMenuOpen = false,
  onMobileMenuClose,
}: SidebarProps) {
  const { isLoaded, totalVisits, visitCounts, customOrder, registerVisit, setCustomOrder } =
    useVisitTracker(VALID_KEYS);

  useEffect(() => {
    if (isLoaded) {
      registerVisit(activeSection);
    }
  }, [activeSection, isLoaded, registerVisit]);

  const displayItems = useMemo(() => {
    if (customOrder.length > 0) {
      const itemMap = new Map(ALL_ITEMS.map((item) => [item.key, item]));
      const result = [];
      for (const key of customOrder) {
        if (itemMap.has(key)) {
          result.push(itemMap.get(key)!);
          itemMap.delete(key);
        }
      }
      for (const item of ALL_ITEMS) {
        if (itemMap.has(item.key)) {
          result.push(item);
        }
      }
      return result;
    }

    if (totalVisits >= 10) {
      return [...ALL_ITEMS].sort((a, b) => {
        const countA = visitCounts[a.key] || 0;
        const countB = visitCounts[b.key] || 0;
        if (countB !== countA) return countB - countA;
        return ALL_ITEMS.indexOf(a) - ALL_ITEMS.indexOf(b);
      });
    }

    return null;
  }, [customOrder, totalVisits, visitCounts]);

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm transition-opacity md:hidden"
          onClick={onMobileMenuClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={[
          "fixed inset-y-0 left-0 z-40 flex h-full flex-col border-r border-white/[0.05] bg-void/95 backdrop-blur-xl transition-all duration-300 ease-in-out md:sticky md:top-0 md:h-screen md:shrink-0 md:bg-void/80",
          mobileMenuOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full md:translate-x-0 md:shadow-none",
          collapsed ? "md:w-[68px]" : "md:w-[232px]",
          "w-[260px]",
        ].join(" ")}
        aria-label="Sidebar navigation"
      >
        <div className="flex h-14 items-center justify-between gap-2 border-b border-white/[0.05] px-4">
          <div className="flex items-center gap-2">
        <motion.span
          initial={{ scale: 0.85, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="relative flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-dream to-neural shadow-[0_0_16px_rgba(34,211,238,0.4)]"
          aria-hidden="true"
        >
          <IconShield size={14} />
        </motion.span>
          {!collapsed && (
            <div className="min-w-0">
              <p className="text-sm font-semibold tracking-tight text-slate-100">NightmareNet</p>
              <p className="text-[10px] uppercase tracking-widest text-slate-400">Sprint · 03</p>
            </div>
          )}
          </div>
          <button
            type="button"
            onClick={onMobileMenuClose}
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md text-slate-400 hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50 md:hidden"
            aria-label="Close sidebar"
          >
            <IconX size={20} />
          </button>
        </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3">
        {displayItems ? (
          <div className="mt-2">
            {!collapsed && (
              <div className="mb-2 px-2 flex items-center justify-between">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-300">
                  Your Top Views
                </p>
              </div>
            )}
            <Reorder.Group
              axis="y"
              values={displayItems}
              onReorder={(newItems) => setCustomOrder(newItems.map((i) => i.key))}
              className="space-y-0.5"
            >
              {displayItems.map((item) => {
                const active = activeSection === item.key;
                return (
                  <Reorder.Item
                    key={item.key}
                    value={item.key}
                    className="relative"
                  >
                    <button
                      type="button"
                      onClick={() => onSectionChange(item.key)}
                      className={[
                        "group relative flex w-full items-center gap-2.5 rounded-md px-2 min-h-[44px] md:min-h-0 md:py-1.5 text-left text-[13px] cursor-pointer",
                        "transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50",
                        active
                          ? "bg-neural/[0.08] text-neural"
                          : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200",
                      ].join(" ")}
                      aria-current={active ? "page" : undefined}
                      title={collapsed ? item.label : undefined}
                    >
                      {active && (
                        <motion.span
                          layoutId="sidebar-active"
                          className="absolute left-0 top-1.5 h-5 w-0.5 rounded-r bg-neural shadow-[0_0_8px_var(--color-neural)]"
                        />
                      )}
                      <span className="flex h-5 w-5 items-center justify-center">{item.icon}</span>
                      {!collapsed && (
                        <>
                          <span className="flex-1 truncate">{item.label}</span>
                          {item.badge && (
                            <span className="rounded-full bg-white/[0.06] px-1.5 py-0.5 text-[10px] font-mono text-slate-400">
                              {item.badge}
                            </span>
                          )}
                        </>
                      )}
                    </button>
                  </Reorder.Item>
                );
              })}
            </Reorder.Group>
          </div>
        ) : (
          NAV.map((group, gi) => (
            <div key={group.label} className={gi > 0 ? "mt-4" : ""}>
              {!collapsed && (
                <p className="mb-1 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-300">
                  {group.label}
                </p>
              )}
              <ul className="space-y-0.5">
                {group.items.map((item) => {
                  const active = activeSection === item.key;
                  return (
                    <li key={item.key}>
                      <button
                        type="button"
                        onClick={() => onSectionChange(item.key)}
                        className={[
                          "group relative flex w-full items-center gap-2.5 rounded-md px-2 min-h-[44px] md:min-h-0 md:py-1.5 text-left text-[13px] cursor-pointer",
                          "transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50",
                          active
                            ? "bg-neural/[0.08] text-neural"
                            : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200",
                        ].join(" ")}
                        aria-current={active ? "page" : undefined}
                        title={collapsed ? item.label : undefined}
                      >
                        {active && (
                          <motion.span
                            layoutId="sidebar-active"
                            className="absolute left-0 top-1.5 h-5 w-0.5 rounded-r bg-neural shadow-[0_0_8px_var(--color-neural)]"
                          />
                        )}
                        <span className="flex h-5 w-5 items-center justify-center">{item.icon}</span>
                        {!collapsed && (
                          <>
                            <span className="flex-1 truncate">{item.label}</span>
                            {item.badge && (
                              <span className="rounded-full bg-white/[0.06] px-1.5 py-0.5 text-[10px] font-mono text-slate-400">
                                {item.badge}
                              </span>
                            )}
                          </>
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))
        )}
      </nav>

      <div className="border-t border-white/[0.05] px-2 py-3">
        {!collapsed ? (
          <div className="rounded-lg border border-dream/20 bg-dream/[0.04] p-3">
            <div className="mb-1.5 flex items-center gap-1.5">
              <IconSparkle size={12} />
              <span className="text-[10px] font-semibold uppercase tracking-widest text-dream-soft">
                Robustness
              </span>
            </div>
            <p className="font-mono text-lg text-slate-100">82.4</p>
            <p className="text-[10px] text-slate-400">+4.1 vs last cycle</p>
          </div>
        ) : (
          <div className="flex h-9 items-center justify-center rounded-md bg-dream/[0.06] text-dream-soft">
            <IconSparkle size={14} />
          </div>
        )}
      </div>
      </aside>
    </>
  );
}
