"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Sidebar, type DashboardSectionKey } from "./Sidebar";
import { Topbar } from "./Topbar";
import { CommandPalette } from "./CommandPalette";
import { NeuralCanvas } from "./NeuralCanvas";

export interface AppShellProps {
  activeSection: DashboardSectionKey;
  onSectionChange: (key: DashboardSectionKey) => void;
  title: string;
  breadcrumb?: { label: string; onClick?: () => void }[];
  apiStatus?: "online" | "degraded" | "offline";
  onPaletteAction?: (action: string) => void;
  /**
   * Increment to ask the shell to toggle the palette from outside.
   *
   * This decouples the global keyboard hook (which lives in DashboardRoot) from
   * AppShell's internal palette state — the parent owns the vocabulary, we own
   * the rendering.
   */
  externalPaletteOpenPulse?: number;
  children: ReactNode;
}

export function AppShell({
  activeSection,
  onSectionChange,
  title,
  breadcrumb,
  apiStatus = "online",
  onPaletteAction,
  externalPaletteOpenPulse,
  children,
}: AppShellProps) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && mobileMenuOpen) {
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mobileMenuOpen]);

  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [mobileMenuOpen]);

  useEffect(() => {
    if (externalPaletteOpenPulse === undefined) return;
    if (externalPaletteOpenPulse === 0) return; // ignore initial mount
    setPaletteOpen((s) => !s);
  }, [externalPaletteOpenPulse]);

  return (
    <div className="relative min-h-screen bg-void text-slate-100">
      <NeuralCanvas />
      <div className="relative z-10 flex">
        <Sidebar
          activeSection={activeSection}
          onSectionChange={(key) => {
            onSectionChange(key);
            setMobileMenuOpen(false);
          }}
          mobileMenuOpen={mobileMenuOpen}
          onMobileMenuClose={() => setMobileMenuOpen(false)}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <Topbar
            title={title}
            breadcrumb={breadcrumb}
            onOpenCommandPalette={() => setPaletteOpen(true)}
            apiStatus={apiStatus}
            onOpenMobileMenu={() => setMobileMenuOpen(true)}
            mobileMenuOpen={mobileMenuOpen}
          />
          <main id="main-content" tabIndex={-1} className="flex-1 overflow-x-hidden px-4 py-5 outline-none sm:px-6 sm:py-6">
            {children}
          </main>
        </div>
      </div>
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onNavigate={(key) => {
          onSectionChange(key);
          setPaletteOpen(false);
        }}
        onAction={onPaletteAction}
      />
    </div>
  );
}
