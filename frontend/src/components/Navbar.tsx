"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from "framer-motion";
import {
  Sparkles,
  Zap,
  Settings2,
  Shield,
  Layers,
  Activity,
  Menu,
  X,
  Code2,
  Workflow,
  LayoutDashboard,
  Sun,
  Moon,
  Monitor,
} from "lucide-react";
import Logo from "./Logo";
import { useTheme } from "@/lib/theme";
import { useDialogFocus } from "./a11y/useDialogFocus";

const navItems = [
  { label: "Demo", href: "#demo", icon: Sparkles },
  { label: "How It Works", href: "#architecture", icon: Layers },
  { label: "Quick Start", href: "#quickstart", icon: Code2 },
  { label: "Playground", href: "#playground", icon: Zap },
  { label: "Resilience", href: "#resilience", icon: Shield },
  { label: "Training", href: "#training", icon: Settings2 },
  { label: "Pipeline", href: "#pipeline", icon: Workflow },
  { label: "Status", href: "#status", icon: Activity },
];

export default function Navbar() {
  const [active, setActive] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const { scrollY } = useScroll();
  const { theme, setTheme } = useTheme();
  const mobileCloseRef = useRef<HTMLButtonElement>(null);
  const closeMobile = useCallback(() => setMobileOpen(false), []);
  const getMobileInitialFocus = useCallback(() => mobileCloseRef.current, []);
  const mobileDialogRef = useDialogFocus(mobileOpen, closeMobile, getMobileInitialFocus);

  /* ── Hide on scroll down, show on scroll up ── */
  useMotionValueEvent(scrollY, "change", (latest) => {
    const prev = scrollY.getPrevious() ?? 0;
    if (latest > 100 && latest > prev) {
      setHidden(true);
    } else {
      setHidden(false);
    }
    // Scroll progress
    const docH = document.documentElement.scrollHeight - window.innerHeight;
    setScrollProgress(docH > 0 ? Math.min(latest / docH, 1) : 0);
  });

  /* ── Scroll spy ── */
  useEffect(() => {
    const ids = navItems.map((n) => n.href.replace("#", ""));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: 0 },
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  /* ── Lock body ── */
  useEffect(() => {
    document.body.style.overflow = mobileOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [mobileOpen]);

  const handleNav = useCallback((href: string) => {
    setMobileOpen(false);
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  }, []);

  return (
    <>
      <motion.nav
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: hidden ? -80 : 0, opacity: hidden ? 0 : 1 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="fixed top-4 left-6 right-6 z-50 glass-navbar rounded-2xl"
        aria-label="Primary navigation"
      >
        <div className="max-w-7xl mx-auto px-5 h-14 flex items-center justify-between">
          {/* Logo */}
          <a
            href="#hero"
            aria-label="NightmareNet home"
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="flex items-center gap-2.5 group cursor-pointer"
          >
            <Logo size="sm" showText={true} animated={true} />
          </a>

          {/* Desktop nav */}
          <div className="hidden lg:flex items-center gap-0.5 ml-8">
            {navItems.map((item) => {
              const isActive = active === item.href.replace("#", "");
              return (
                <button
                  type="button"
                  key={item.label}
                  onClick={() => handleNav(item.href)}
                  aria-current={isActive ? "location" : undefined}
                  className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 cursor-pointer ${
                    isActive
                      ? "text-text"
                      : "text-text-dim hover:text-text hover:bg-white/[0.03]"
                  }`}
                >
                  <item.icon aria-hidden="true" className="w-3.5 h-3.5" />
                  {item.label}
                  {isActive && (
                    <motion.div
                      layoutId="nav-active"
                      className="absolute inset-0 rounded-lg bg-white/[0.06] border border-neural/15 -z-10"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-3">
            {/* Dashboard Link */}
            <a
              href="/dashboard"
              className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-neural bg-neural/10 border border-neural/20 hover:bg-neural/15 hover:border-neural/30 transition-all cursor-pointer"
            >
              <LayoutDashboard aria-hidden="true" className="w-3.5 h-3.5" />
              Dashboard
            </a>

            {/* Theme Toggle */}
            <div className="hidden sm:flex items-center gap-1 p-1 rounded-lg bg-white/[0.03] border border-white/[0.04]">
              <button
                type="button"
                onClick={() => setTheme("light")}
                aria-pressed={theme === "light"}
                className={`p-1.5 rounded-md transition-all cursor-pointer ${
                  theme === "light"
                    ? "bg-warning/20 text-warning"
                    : "text-slate-400 hover:text-text-dim"
                }`}
                aria-label="Light mode"
              >
                <Sun aria-hidden="true" className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setTheme("dark")}
                aria-pressed={theme === "dark"}
                className={`p-1.5 rounded-md transition-all cursor-pointer ${
                  theme === "dark"
                    ? "bg-dream/20 text-dream"
                    : "text-slate-400 hover:text-text-dim"
                }`}
                aria-label="Dark mode"
              >
                <Moon aria-hidden="true" className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setTheme("system")}
                aria-pressed={theme === "system"}
                className={`p-1.5 rounded-md transition-all cursor-pointer ${
                  theme === "system"
                    ? "bg-neural/20 text-neural"
                    : "text-slate-400 hover:text-text-dim"
                }`}
                aria-label="System theme"
              >
                <Monitor aria-hidden="true" className="w-3.5 h-3.5" />
              </button>
            </div>

            <a
              href="https://github.com/Adit-Jain-srm/NightmareNet"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1.5 text-xs font-mono text-text-dim hover:text-neural transition-colors cursor-pointer"
            >
              <svg aria-hidden="true" className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              GitHub
            </a>
            <span className="hidden sm:block text-[10px] font-mono text-slate-400/70 px-2 py-1 rounded-md bg-white/[0.03] border border-white/[0.04]">
              v0.2.0
            </span>

            {/* Mobile hamburger */}
            <button
              type="button"
              onClick={() => setMobileOpen(!mobileOpen)}
              aria-expanded={mobileOpen}
              aria-controls="mobile-navigation"
              className="lg:hidden p-2 rounded-lg hover:bg-white/[0.05] transition-colors cursor-pointer"
              aria-label={mobileOpen ? "Close navigation menu" : "Open navigation menu"}
            >
              {mobileOpen ? (
                <X aria-hidden="true" className="w-5 h-5 text-text" />
              ) : (
                <Menu aria-hidden="true" className="w-5 h-5 text-text-dim" />
              )}
            </button>
          </div>
        </div>

        {/* Scroll progress bar */}
        <div
          className="absolute bottom-0 left-4 right-4 h-[1px] bg-white/[0.03] rounded-full overflow-hidden"
          role="progressbar"
          aria-label="Page scroll progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(scrollProgress * 100)}
        >
          <motion.div
            className="h-full bg-gradient-to-r from-neural via-dream to-nightmare rounded-full"
            style={{ width: `${scrollProgress * 100}%` }}
            transition={{ duration: 0.05 }}
          />
        </div>
      </motion.nav>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.button
              type="button"
              tabIndex={-1}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={closeMobile}
              aria-label="Close navigation menu"
              className="fixed inset-0 z-40 cursor-default bg-void/70 backdrop-blur-sm lg:hidden"
            />
            <motion.div
              ref={mobileDialogRef}
              id="mobile-navigation"
              role="dialog"
              aria-modal="true"
              aria-label="Mobile navigation"
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="fixed top-0 right-0 bottom-0 z-50 w-72 glass p-6 lg:hidden"
            >
              <div className="flex items-center justify-between mb-8">
                <Logo size="sm" showText={true} animated={false} />
                <button
                  ref={mobileCloseRef}
                  type="button"
                  onClick={closeMobile}
                  aria-label="Close navigation menu"
                  className="p-1.5 rounded-lg hover:bg-white/[0.05] cursor-pointer"
                >
                  <X aria-hidden="true" className="w-5 h-5 text-text-dim" />
                </button>
              </div>

              {/* Dashboard Link - Mobile */}
              <a
                href="/dashboard"
                className="flex items-center gap-3 px-4 py-3 mb-4 rounded-xl text-sm font-medium text-neural bg-neural/10 border border-neural/20 hover:bg-neural/15 transition-all cursor-pointer"
              >
                <LayoutDashboard aria-hidden="true" className="w-4 h-4" />
                Open Dashboard
              </a>

              <div className="space-y-1">
                {navItems.map((item, i) => {
                  const isActive = active === item.href.replace("#", "");
                  return (
                    <motion.button
                      type="button"
                      key={item.label}
                      onClick={() => handleNav(item.href)}
                      aria-current={isActive ? "location" : undefined}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all cursor-pointer text-left ${
                        isActive
                          ? "text-text bg-white/[0.06] border border-neural/15"
                          : "text-text-dim hover:text-text hover:bg-white/[0.03] border border-transparent"
                      }`}
                    >
                      <item.icon aria-hidden="true" className="w-4 h-4" />
                      {item.label}
                    </motion.button>
                  );
                })}
              </div>

              {/* Theme Toggle - Mobile */}
              <div className="mt-6 pt-4 border-t border-white/[0.05]">
                <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">Theme</p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setTheme("light")}
                    aria-pressed={theme === "light"}
                    className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                      theme === "light"
                        ? "bg-warning/20 text-warning border border-warning/30"
                        : "text-slate-400 hover:text-text-dim border border-white/[0.06]"
                    }`}
                  >
                    <Sun aria-hidden="true" className="w-3.5 h-3.5" />
                    Light
                  </button>
                  <button
                    type="button"
                    onClick={() => setTheme("dark")}
                    aria-pressed={theme === "dark"}
                    className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                      theme === "dark"
                        ? "bg-dream/20 text-dream border border-dream/30"
                        : "text-slate-400 hover:text-text-dim border border-white/[0.06]"
                    }`}
                  >
                    <Moon aria-hidden="true" className="w-3.5 h-3.5" />
                    Dark
                  </button>
                  <button
                    type="button"
                    onClick={() => setTheme("system")}
                    aria-pressed={theme === "system"}
                    className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                      theme === "system"
                        ? "bg-neural/20 text-neural border border-neural/30"
                        : "text-slate-400 hover:text-text-dim border border-white/[0.06]"
                    }`}
                  >
                    <Monitor aria-hidden="true" className="w-3.5 h-3.5" />
                    Auto
                  </button>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-white/[0.05]">
                <a
                  href="https://github.com/Adit-Jain-srm/NightmareNet"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs font-mono text-text-dim hover:text-neural transition-colors"
                >
                  GitHub →
                </a>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
