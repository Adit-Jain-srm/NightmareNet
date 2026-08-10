"use client";

import { Skeleton } from "@/components/ui/Skeleton";

// Elliptical node layout approximating PipelineGraph's 4-phase orbit
// (wake / dream / nightmare / compress) around a central "MODEL" hub.
const NODE_POSITIONS = [
  { left: "50%", top: "6%" },
  { left: "92%", top: "50%" },
  { left: "50%", top: "94%" },
  { left: "8%", top: "50%" },
];

/**
 * Suspense fallback for `PhaseVisualizer` / `PipelineGraph`. Mirrors the
 * `Panel` header, the orbiting node canvas (4 phase nodes + center hub),
 * the circular phase ring, and the phase legend list.
 */
export function PipelineGraphSkeleton() {
  return (
    <section
      role="status"
      aria-busy="true"
      className="relative flex h-full flex-col overflow-hidden rounded-xl border border-white/[0.06] bg-[rgba(2,6,23,0.72)] backdrop-blur-[12px]"
    >
      <span className="sr-only">Loading phase visualizer</span>
      <header className="flex items-start justify-between gap-3 border-b border-white/[0.04] px-4 py-3">
        <div className="flex min-w-0 items-start gap-2.5" aria-hidden="true">
          <Skeleton width={28} height={28} rounded="md" />
          <div className="min-w-0 space-y-1.5">
            <Skeleton height={13} width={120} />
            <Skeleton height={10} width={130} />
          </div>
        </div>
        <div className="shrink-0" aria-hidden="true">
          <Skeleton height={20} width={64} rounded="full" />
        </div>
      </header>

      <div className="flex-1 px-4 py-3.5" aria-hidden="true">
        <div className="relative mb-4 w-full" style={{ minHeight: 220 }}>
          <Skeleton
            width="70%"
            height={140}
            rounded="full"
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-40"
          />
          {NODE_POSITIONS.map((pos, i) => (
            <div
              key={i}
              className="absolute -translate-x-1/2 -translate-y-1/2"
              style={{ left: pos.left, top: pos.top }}
            >
              <Skeleton width={36} height={36} rounded="full" />
            </div>
          ))}
          <Skeleton
            width={56}
            height={56}
            rounded="full"
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
          />
        </div>

        <div className="flex flex-col items-center gap-4 py-2 lg:flex-row lg:items-center lg:gap-6">
          <Skeleton width={260} height={260} rounded="full" className="shrink-0" />
          <div className="w-full flex-1 space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-lg border border-white/[0.05] bg-white/[0.01] p-2.5"
              >
                <Skeleton width={8} height={8} rounded="full" className="mt-1" />
                <div className="min-w-0 flex-1 space-y-1">
                  <Skeleton height={9} width={70} />
                  <Skeleton height={8} width="85%" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}