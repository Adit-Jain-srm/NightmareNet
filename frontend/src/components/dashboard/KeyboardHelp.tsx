"use client";

import { AnimatePresence, motion } from "framer-motion";
import { SHORTCUT_GROUPS } from "./useGlobalShortcuts";
import { useDialogFocus } from "../a11y/useDialogFocus";

interface KeyboardHelpProps {
  open: boolean;
  onClose: () => void;
}

export function KeyboardHelp({ open, onClose }: KeyboardHelpProps) {
  const dialogRef = useDialogFocus(open, onClose);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.14 }}
          className="fixed inset-0 z-[58] flex items-center justify-center px-4"
        >
          <button
            type="button"
            className="absolute inset-0 cursor-default bg-void/80 backdrop-blur-sm"
            onClick={onClose}
            aria-label="Close keyboard shortcuts"
          />
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="kbd-help-title"
            tabIndex={-1}
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.16, ease: "easeOut" }}
            className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-white/[0.08] bg-abyss/95 shadow-[0_24px_60px_rgba(0,0,0,0.6)]"
          >
            <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-3">
              <h2 id="kbd-help-title" className="text-sm font-semibold text-slate-100">
                Keyboard Shortcuts
              </h2>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close keyboard shortcuts"
                className="cursor-pointer rounded-md px-2 py-1 text-[11px] text-slate-400 hover:bg-white/5 hover:text-slate-300"
              >
                Esc
              </button>
            </div>
            <div className="max-h-[60vh] space-y-5 overflow-y-auto px-5 py-4">
              {SHORTCUT_GROUPS.map((group) => (
                <div key={group.label}>
                  <p className="pb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-300">
                    {group.label}
                  </p>
                  <ul className="space-y-1.5">
                    {group.items.map((item) => (
                      <li
                        key={item.label}
                        className="flex items-center justify-between text-[13px] text-slate-300"
                      >
                        <span>{item.label}</span>
                        <span className="flex gap-1 font-mono text-[11px]">
                          {item.keys.map((k, i) => (
                            <kbd
                              key={i}
                              className="rounded bg-white/[0.07] px-1.5 py-0.5 text-slate-300"
                            >
                              {k}
                            </kbd>
                          ))}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <div className="border-t border-white/[0.06] px-5 py-3 text-[11px] text-slate-400">
              Tip: Shortcuts are disabled while typing into text fields, so they never fight your editing.
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
