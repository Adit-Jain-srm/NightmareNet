"use client";

import { Skeleton } from "@/components/ui/Skeleton";

/**
 * Suspense fallback for `RunDetail`. Mirrors the `Panel` header, the
 * four-node phase timeline, the phase tab strip, the 2/4-col metric
 * tile grid, and the loss/epochs/progress card row.
 */
export function RunDetailSkeleton() {
  return (
    <section
      role="status"
      aria-busy="true"
      className="relative flex h-full flex-col overflow-hidden rounded-xl border border-white/[0.06] bg-[rgba(2,6,23,0.72)] backdrop-blur-[12px]"
    >
      <span className="sr-only">Loading run detail</span>
      <header className="flex items-start justify-between gap-3 border-b border-white/[0.04] px-4 py-3">
        <div className="flex min-w-0 items-start gap-2.5" aria-hidden="true">
          <Skeleton width={28} height={28} rounded="md" />
          <div className="min-w-0 space-y-1.5">
            <Skeleton height={13} width={160} />
            <Skeleton height={10} width={140} />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5" aria-hidden="true">
          <Skeleton height={20} width={56} rounded="full" />
          <Skeleton height={26} width={26} rounded="md" />
          <Skeleton height={26} width={26} rounded="md" />
          <Skeleton height={26} width={64} rounded="md" />
        </div>
      </header>

      <div className="flex-1 px-4 py-3.5" aria-hidden="true">
        <div className="relative flex items-center justify-between">
          <div className="absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-white/[0.06]" />
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="relative flex flex-col items-center gap-1">
              <Skeleton width={32} height={32} rounded="full" />
              <Skeleton height={8} width={36} />
            </div>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap gap-1.5 border-b border-white/[0.06] pb-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={22} width={72} rounded="md" />
          ))}
        </div>

        <div className="mt-4 space-y-4">
          <Skeleton height={10} width="80%" />

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="rounded-md border border-white/[0.06] bg-white/[0.02] p-2.5"
              >
                <Skeleton height={8} width="60%" className="mb-1.5" />
                <Skeleton height={14} width="40%" />
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3"
              >
                <Skeleton height={8} width="55%" className="mb-1.5" />
                <Skeleton height={16} width="70%" className="mb-1" />
                <Skeleton height={8} width="35%" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
