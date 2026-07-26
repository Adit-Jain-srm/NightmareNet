"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "../ui/Button";
import { IconCommand, IconLayers, IconRadar, IconWand } from "./icons";
import type { DashboardSectionKey } from "./Sidebar";
import { useDialogFocus } from "../a11y/useDialogFocus";

const STORAGE_KEY = "nightmarenet.onboarding.dismissed.v1";

interface Step {
  icon: React.ReactNode;
  title: string;
  body: string;
  highlight: string;
}

const STEPS: Step[] = [
  {
    icon: <IconLayers size={18} />,
    title: "Four phases, one cycle",
    body:
      "Wake → Dream → Nightmare → Compress. Every cycle hardens your model against adversarial inputs while preserving clean accuracy.",
    highlight: "Phase Visualizer",
  },
  {
    icon: <IconRadar size={18} />,
    title: "Robustness, measured continuously",
    body:
      "Live metrics, multi-strength evals, and a 5-axis radar give you a daily robustness score, not just a one-shot check.",
    highlight: "Live Metrics + Radar",
  },
  {
    icon: <IconWand size={18} />,
    title: "Distortion playground",
    body:
      "Try Dream vs Nightmare side-by-side on any text — instantly see how your model would react under adversarial pressure.",
    highlight: "Distortion Preview",
  },
  {
    icon: <IconCommand size={18} />,
    title: "Press Cmd+K for anything",
    body:
      "Navigate, run benchmarks, distort text, jump to settings — all from the keyboard. Press ? anywhere to see every shortcut.",
    highlight: "Command palette",
  },
];

interface OnboardingOverlayProps {
  onNavigate: (key: DashboardSectionKey) => void;
}

export function OnboardingOverlay({ onNavigate }: OnboardingOverlayProps) {
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.localStorage.getItem(STORAGE_KEY) !== "true") {
      const t = setTimeout(() => setOpen(true), 350);
      return () => clearTimeout(t);
    }
  }, []);

  const dismiss = useCallback(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, "true");
    } catch {
      /* sandboxed storage; ignore */
    }
    setOpen(false);
  }, []);

  const getInitialFocus = useCallback(() => closeButtonRef.current, []);
  const dialogRef = useDialogFocus(open, dismiss, getInitialFocus);

  const tryItNow = () => {
    onNavigate("distortions");
    dismiss();
  };

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="fixed inset-0 z-[60] flex items-center justify-center px-4"
        >
          <button
            type="button"
            className="absolute inset-0 cursor-default bg-void/85 backdrop-blur-md"
            onClick={dismiss}
            aria-label="Dismiss onboarding tour"
          />
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="onboarding-title"
            tabIndex={-1}
            initial={{ opacity: 0, y: 14, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 14, scale: 0.97 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-white/[0.08] bg-abyss/95 shadow-[0_24px_80px_rgba(0,0,0,0.7)]"
          >
            <div className="border-b border-white/[0.06] px-6 py-5">
              <div className="flex items-center justify-between">
                <span className="rounded-md bg-neural/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-neural">
                  Welcome · {step + 1} / {STEPS.length}
                </span>
                <button
                  ref={closeButtonRef}
                  type="button"
                  onClick={dismiss}
                  className="cursor-pointer rounded-md px-2 py-1 text-[11px] text-slate-400 hover:bg-white/5 hover:text-slate-300"
                >
                  Skip tour
                </button>
              </div>
            </div>
            <AnimatePresence mode="wait">
              <motion.div
                key={step}
                initial={{ opacity: 0, x: 14 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -14 }}
                transition={{ duration: 0.18 }}
                className="px-6 py-6"
              >
                <div className="flex items-start gap-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-neural/[0.12] text-neural">
                    {current.icon}
                  </div>
                  <div className="space-y-1.5">
                    <h2 id="onboarding-title" className="text-lg font-semibold text-slate-100">
                      {current.title}
                    </h2>
                    <p className="text-sm leading-relaxed text-slate-400">{current.body}</p>
                    <p className="pt-1.5 text-[10px] uppercase tracking-widest text-slate-300">
                      Find it in: <span className="text-slate-400">{current.highlight}</span>
                    </p>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
            <div className="flex items-center justify-between border-t border-white/[0.06] px-6 py-4">
              <div className="flex gap-1.5" aria-hidden="true">
                {STEPS.map((_, i) => (
                  <span
                    key={i}
                    className={[
                      "h-1.5 w-6 rounded-full transition-colors",
                      i === step ? "bg-neural" : "bg-white/10",
                    ].join(" ")}
                  />
                ))}
              </div>
              <div className="flex items-center gap-2">
                {step > 0 && (
                  <Button variant="ghost" size="sm" onClick={() => setStep((s) => s - 1)}>
                    Back
                  </Button>
                )}
                {isLast ? (
                  <Button variant="primary" size="sm" onClick={tryItNow}>
                    Try it now
                  </Button>
                ) : (
                  <Button variant="primary" size="sm" onClick={() => setStep((s) => s + 1)}>
                    Next
                  </Button>
                )}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
