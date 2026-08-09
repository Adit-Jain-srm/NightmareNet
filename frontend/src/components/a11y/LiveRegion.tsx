"use client";

type LiveRegionProps = {
  /** The text to announce to screen readers. Update this value to trigger an announcement. */
  message: string;
  /** When true, uses aria-live="assertive" (interrupts the user). Use for errors only. */
  assertive?: boolean;
  /** When true (default), the entire region is announced as a single unit on change. */
  atomic?: boolean;
  /** Optional stable id for the region element. */
  id?: string;
};

/**
 * LiveRegion — visually hidden ARIA live region for screen reader announcements.
 *
 * Render once near the root (e.g. inside ToastProvider) and update `message`
 * to announce dynamic state changes (toast pushes, async completion, errors)
 * to assistive technology without visual duplication.
 *
 * Polite mode (default): announcement queued until the user is idle.
 * Assertive mode: announcement interrupts immediately — use only for errors.
 */
export function LiveRegion({ message, assertive = false, atomic = true, id }: LiveRegionProps) {
  return (
    <div
      id={id}
      className="sr-only"
      role={assertive ? "alert" : "status"}
      aria-live={assertive ? "assertive" : "polite"}
      aria-atomic={atomic}
    >
      {message}
    </div>
  );
}
