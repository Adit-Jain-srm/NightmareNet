"use client";

import { Skeleton } from "@/components/ui/Skeleton";

const METRIC_LABEL_WIDTHS = [70, 60, 55, 85];

/**
 * Suspense fallback for `ModelComparison`. Mirrors the `Panel` header,
 * the Model A / Model B selects, the two model summary cards, the
 * A/B metric comparison rows, and the composite score bars.
 */
export function ModelComparisonSkeleton() {
  return (
    <section
      role="status"
      aria-busy="true"
      className="relative flex h-full flex-col overflow-hidden rounded-xl border border-white/[0.06] bg-[rgba(2,6,23,0.72)] backdrop-blur-[12px]"
    >
      <span className="sr-only">Loading model comparison</span>
      <header className="flex items-start justify-between gap-3 border-b border-white/[0.04] px-4 py-3">
        <div className="flex min-w-0 items-start gap-2.5" aria-hidden="true">
          <Skeleton width={28} height={28} rounded="md" />
          <div className="min-w-0 space-y-1.5">
            <Skeleton height={13} width={130} />
            <Skeleton height={10} width={90} />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5" aria-hidden="true">
          <Skeleton height={20} width={70} rounded="full" />
        </div>
      </header>

      <div className="flex-1 px-4 py-3.5" aria-hidden="true">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Skeleton height={9} width={48} />
            <Skeleton height={30} width="100%" rounded="md" />
          </div>
          <div className="space-y-1.5">
            <Skeleton height={9} width={48} />
            <Skeleton height={30} width="100%" rounded="md" />
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2.5 space-y-1.5">
            <Skeleton height={8} width={54} />
            <Skeleton height={12} width="80%" />
            <Skeleton height={8} width="60%" />
          </div>
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2.5 space-y-1.5">
            <Skeleton height={8} width={54} />
            <Skeleton height={12} width="80%" />
            <Skeleton height={8} width="60%" />
          </div>
        </div>

        <div className="mt-4 space-y-2 border-t border-white/[0.04] pt-2">
          {METRIC_LABEL_WIDTHS.map((w, i) => (
            <div
              key={i}
              className="grid grid-cols-[110px_1fr_60px_1fr_60px] items-center gap-2 py-1.5"
            >
              <Skeleton height={8} width={w} />
              <Skeleton height={6} width="100%" rounded="full" />
              <Skeleton height={8} width={36} className="ml-auto" />
              <Skeleton height={6} width="100%" rounded="full" />
              <Skeleton height={8} width={36} className="ml-auto" />
            </div>
          ))}
        </div>

        <div className="mt-3">
          <Skeleton height={8} width={110} className="mb-1" />
          <div className="grid grid-cols-2 gap-2">
            <Skeleton height={14} width="100%" rounded="full" />
            <Skeleton height={14} width="100%" rounded="full" />
          </div>
        </div>
      </div>
    </section>
  );
}