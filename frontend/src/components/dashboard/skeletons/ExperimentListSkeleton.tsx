"use client";

import { Skeleton } from "@/components/ui/Skeleton";

/**
 * Suspense fallback for `ExperimentList`. Mirrors its `Panel` shell
 * (icon + title/subtitle + toolbar) and its compact `DataTable`
 * (Experiment · Model · Status · Cycles · Robustness · Duration ·
 * Created · Actions) so the layout doesn't jump once data arrives.
 */
export function ExperimentListSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <section
      role="status"
      aria-busy="true"
      className="relative flex h-full flex-col overflow-hidden rounded-xl border border-white/[0.06] bg-[rgba(2,6,23,0.72)] backdrop-blur-[12px]"
    >
      <span className="sr-only">Loading experiments</span>
      <header className="flex items-start justify-between gap-3 border-b border-white/[0.04] px-4 py-3">
        <div className="flex min-w-0 items-start gap-2.5" aria-hidden="true">
          <Skeleton width={28} height={28} rounded="md" />
          <div className="min-w-0 space-y-1.5">
            <Skeleton height={13} width={90} />
            <Skeleton height={10} width={110} />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5" aria-hidden="true">
          <Skeleton height={26} width={140} rounded="md" />
          <Skeleton height={26} width={90} rounded="md" />
          <Skeleton height={26} width={70} rounded="md" />
          <Skeleton height={26} width={70} rounded="md" />
          <Skeleton height={26} width={26} rounded="md" />
          <Skeleton height={26} width={26} rounded="md" />
          <Skeleton height={26} width={72} rounded="md" />
        </div>
      </header>

      <div className="flex-1 overflow-hidden" aria-hidden="true">
        <div className="hidden items-center gap-3 border-b border-white/[0.04] px-3 py-2 sm:flex">
          {["Experiment", "Model", "Status", "Cycles", "Robustness", "Duration", "Created", ""].map(
            (label, i) => (
              <Skeleton
                key={i}
                height={8}
                width={i === 0 ? "18%" : i === 7 ? 16 : "9%"}
                className={i === 0 ? "flex-1" : i === 7 ? "ml-auto" : ""}
              />
            ),
          )}
        </div>
        <ul className="divide-y divide-white/[0.04]">
          {Array.from({ length: rows }).map((_, i) => (
            <li key={i} className="flex items-center gap-3 px-3 py-2.5">
              <div className="min-w-0 flex-1 space-y-1.5">
                <Skeleton height={10} width={`${55 + ((i * 7) % 25)}%`} />
                <Skeleton height={8} width={64} />
              </div>
              <Skeleton height={10} width={56} className="hidden sm:block" />
              <Skeleton height={14} width={54} rounded="full" />
              <Skeleton height={10} width={24} className="hidden sm:block" />
              <Skeleton height={10} width={40} className="hidden sm:block" />
              <Skeleton height={10} width={56} className="hidden sm:block" />
              <Skeleton height={10} width={48} className="hidden sm:block" />
              <Skeleton height={14} width={14} rounded="sm" />
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}