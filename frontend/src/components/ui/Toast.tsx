"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useSounds } from "@/lib/sounds";
import { LiveRegion } from "@/components/a11y/LiveRegion";

export type ToastVariant = "info" | "success" | "warning" | "error";

interface ToastItem {
  id: string;
  title: string;
  description?: string;
  variant: ToastVariant;
  durationMs: number;
}

interface ToastContextValue {
  push: (toast: Omit<ToastItem, "id" | "durationMs"> & { durationMs?: number }) => string;
  dismiss: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const variantClasses: Record<ToastVariant, string> = {
  info: "border-neural/30 bg-neural/[0.08] text-neural-soft",
  success: "border-emerald-500/30 bg-emerald-500/[0.08] text-emerald-300",
  warning: "border-amber-500/30 bg-amber-500/[0.08] text-amber-300",
  error: "border-nightmare/30 bg-nightmare/[0.08] text-nightmare-soft",
};

const variantIcons: Record<ToastVariant, ReactNode> = {
  info: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7 4.5V7.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="7" cy="9.6" r="0.7" fill="currentColor" />
    </svg>
  ),
  success: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.4" />
      <path d="M4.5 7L6.3 8.8L9.5 5.6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  warning: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M7 1.5L13 12.5H1L7 1.5Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M7 6V8.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="7" cy="10.4" r="0.7" fill="currentColor" />
    </svg>
  ),
  error: (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.4" />
      <path d="M5 5L9 9M9 5L5 9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  ),
};

/**
 * Debounce delay (ms) for the aria-live region.
 * Prevents duplicate or overlapping announcements when multiple toasts fire
 * in rapid succession (e.g. a loading toast immediately followed by a
 * completion toast).
 */
const ANNOUNCE_DEBOUNCE_MS = 150;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const { playSuccess, playError, playNotification } = useSounds();

  // --- Screen reader announcement state ---
  const [liveMessage, setLiveMessage] = useState("");
  const [liveAssertive, setLiveAssertive] = useState(false);
  const announceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Schedule an aria-live announcement, debounced so rapid consecutive pushes
   * produce a single announcement with the latest message.
   */
  const announce = useCallback((message: string, assertive: boolean) => {
    if (announceTimer.current !== null) {
      clearTimeout(announceTimer.current);
    }
    announceTimer.current = setTimeout(() => {
      setLiveMessage(message);
      setLiveAssertive(assertive);
      announceTimer.current = null;
    }, ANNOUNCE_DEBOUNCE_MS);
  }, []);

  const dismiss = useCallback((id: string) => {
    const t = timers.current.get(id);
    if (t) clearTimeout(t);
    timers.current.delete(id);
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback<ToastContextValue["push"]>(
    ({ title, description, variant, durationMs }) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const item: ToastItem = {
        id,
        title,
        description,
        variant,
        durationMs: durationMs ?? 4000,
      };
      setToasts((prev) => [...prev, item]);
      const handle = setTimeout(() => dismiss(id), item.durationMs);
      timers.current.set(id, handle);

      if (variant === "success") playSuccess();
      else if (variant === "error") playError();
      else playNotification();

      // Announce to screen readers — errors are assertive (interrupt),
      // all other variants are polite (wait for idle).
      const announcementText = description
        ? `${title}: ${description}`
        : title;
      announce(announcementText, variant === "error");

      return id;
    },
    [dismiss, playSuccess, playError, playNotification, announce],
  );

  useEffect(() => {
    const ref = timers.current;
    return () => {
      ref.forEach((t) => clearTimeout(t));
      ref.clear();
      // Clean up any pending announce debounce on unmount.
      if (announceTimer.current !== null) {
        clearTimeout(announceTimer.current);
      }
    };
  }, []);

  return (
    <ToastContext.Provider value={{ push, dismiss }}>
      {children}
      {/*
       * Visually-hidden live region — separate from the visual toast stack so
       * screen reader output is not affected by animation or visual styling.
       * Error toasts use assertive mode to interrupt immediately; all others
       * use polite mode to queue behind the current speech.
       */}
      <LiveRegion
        id="toast-live-region"
        message={liveMessage}
        assertive={liveAssertive}
      />
      <div
        aria-hidden="true"
        className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-full max-w-sm flex-col gap-2"
      >
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, x: 24, scale: 0.97 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 24, scale: 0.97 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
              className={[
                "pointer-events-auto flex items-start gap-3 rounded-xl border p-3 backdrop-blur-md",
                variantClasses[t.variant],
              ].join(" ")}
            >
              <span className="mt-0.5">{variantIcons[t.variant]}</span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-slate-100">{t.title}</p>
                {t.description && (
                  <p className="mt-0.5 text-xs text-slate-400">{t.description}</p>
                )}
              </div>
              <button
                type="button"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss"
                className="-m-1 rounded-md p-1 text-slate-500 hover:bg-white/[0.06] hover:text-slate-200 cursor-pointer"
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                  <path d="M3 3L9 9M9 3L3 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within <ToastProvider>");
  return ctx;
}
