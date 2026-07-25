"use client";

import { Badge } from "@/components/ui/Badge";
import { IconBell, IconCommand, IconCpu, IconSearch, IconMenu } from "./icons";

export interface TopbarProps {
  title: string;
  breadcrumb?: { label: string; onClick?: () => void }[];
  onOpenCommandPalette: () => void;
  apiStatus?: "online" | "degraded" | "offline";
  onOpenMobileMenu?: () => void;
  mobileMenuOpen?: boolean;
}

const statusMap = {
  online: { dot: "bg-emerald-400", label: "API · online", tone: "success" as const },
  degraded: { dot: "bg-amber-400", label: "API · degraded", tone: "warning" as const },
  offline: { dot: "bg-nightmare", label: "API · offline", tone: "nightmare" as const },
};

export function Topbar({
  title,
  breadcrumb = [],
  onOpenCommandPalette,
  apiStatus = "online",
  onOpenMobileMenu,
  mobileMenuOpen = false,
}: TopbarProps) {
  const status = statusMap[apiStatus];
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-4 border-b border-white/[0.05] bg-void/80 px-4 backdrop-blur-xl sm:px-6">
      <div className="flex min-w-0 items-center gap-2">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          className="mr-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md text-slate-400 hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50 md:hidden"
          aria-label="Open sidebar"
          aria-expanded={mobileMenuOpen}
        >
          <IconMenu size={20} />
        </button>
        <h1 className="truncate text-sm font-semibold text-slate-100">{title}</h1>
        {breadcrumb.length > 0 && (
          <nav className="hidden items-center gap-1 text-xs text-slate-400 md:flex" aria-label="Breadcrumb">
            <span>·</span>
            {breadcrumb.map((b, i) => (
              <span key={i} className="flex items-center gap-1">
                {b.onClick ? (
                  <button
                    type="button"
                    onClick={b.onClick}
                    className="hover:text-slate-300 cursor-pointer"
                  >
                    {b.label}
                  </button>
                ) : (
                  <span>{b.label}</span>
                )}
                {i < breadcrumb.length - 1 && <span className="text-slate-700">/</span>}
              </span>
            ))}
          </nav>
        )}
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={onOpenCommandPalette}
          className="group flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-xs text-slate-400 transition-colors hover:border-white/[0.12] hover:text-slate-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50"
          aria-label="Open command palette"
        >
          <IconSearch size={13} />
          <span className="hidden sm:inline">Search or jump…</span>
          <kbd className="hidden items-center gap-0.5 rounded bg-white/[0.06] px-1 py-0.5 font-mono text-[10px] text-slate-400 sm:inline-flex">
            <IconCommand size={10} />K
          </kbd>
        </button>

        <Badge variant={status.tone} size="sm" dot>
          {status.label}
        </Badge>

        <button
          type="button"
          className="hidden h-8 w-8 items-center justify-center rounded-md border border-white/[0.06] text-slate-400 hover:border-white/[0.12] hover:text-slate-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50 sm:inline-flex"
          aria-label="GPU status"
          title="GPU"
        >
          <IconCpu size={14} />
        </button>

        <button
          type="button"
          className="relative inline-flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.06] text-slate-400 hover:border-white/[0.12] hover:text-slate-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50"
          aria-label="Notifications"
        >
          <IconBell size={14} />
          <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-nightmare shadow-[0_0_6px_var(--color-nightmare)]" />
        </button>

        <div className="flex h-8 w-8 items-center justify-center rounded-full border border-white/[0.08] bg-gradient-to-br from-dream/30 to-neural/30 font-mono text-[11px] text-slate-100">
          AJ
        </div>
      </div>
    </header>
  );
}
