"use client";

export default function SkipLink() {
  return (
    <a
      href="#main-content"
      className="fixed left-4 top-4 z-[10000] -translate-y-24 rounded-lg bg-neural px-4 py-3 text-sm font-bold text-void shadow-xl transition-transform focus:translate-y-0 focus:outline-none focus-visible:ring-4 focus-visible:ring-white/80"
    >
      Skip to main content
    </a>
  );
}