"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Panel } from "./Panel";
import { Badge } from "@/components/ui/Badge";
import { SkeletonChart } from "@/components/ui/Skeleton";
import { useDemoMode } from "@/lib/hooks";
import { IconActivity, IconTrend } from "./icons";

const LOSS_SERIES = [
  { phase: "wake", color: "var(--color-neural)", values: [2.41, 2.18, 1.93, 1.78, 1.62, 1.45, 1.32, 1.24, 1.21, 1.18] },
  { phase: "dream", color: "var(--color-dream)", values: [1.18, 1.10, 1.04, 0.98, 0.94, 0.91, 0.88, 0.85, 0.83, 0.82] },
  { phase: "nightmare", color: "var(--color-nightmare)", values: [0.82, 0.95, 1.08, 1.18, 1.25, 1.30, 1.32, 1.33, 1.34, 1.34] },
  { phase: "compress", color: "var(--color-warning)", values: [1.34, 1.31, 1.28, 1.26, 1.24, 1.23, 1.22, 1.22, 1.21, 1.21] },
];

const ROBUSTNESS = [
  { strength: 0.1, baseline: 89, hardened: 92 },
  { strength: 0.2, baseline: 81, hardened: 89 },
  { strength: 0.3, baseline: 72, hardened: 84 },
  { strength: 0.4, baseline: 64, hardened: 81 },
  { strength: 0.5, baseline: 55, hardened: 76 },
  { strength: 0.6, baseline: 47, hardened: 71 },
  { strength: 0.7, baseline: 39, hardened: 66 },
  { strength: 0.8, baseline: 31, hardened: 60 },
];

interface LineChartProps {
  series: { phase: string; color: string; values: number[] }[];
  width?: number;
  height?: number;
}

function LossChart({ series, width = 600, height = 220 }: LineChartProps) {
  const all = series.flatMap((s) => s.values);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const totalPoints = series.reduce((acc, s) => acc + s.values.length, 0);
  const pad = { l: 32, r: 12, t: 12, b: 22 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const x = (i: number) => pad.l + (i / Math.max(1, totalPoints - 1)) * innerW;
  const y = (v: number) => pad.t + (1 - (v - min) / Math.max(0.0001, max - min)) * innerH;

  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }).map((_, i) => {
    const v = min + (i / ticks) * (max - min);
    return { v, y: y(v) };
  });

  // Pre-compute each series' starting index without mutating during map; the
  // react-hooks/immutability rule treats post-render reassignments as a smell.
  const startIdxs = series.reduce<number[]>((acc, s, i) => {
    const prev = i === 0 ? 0 : acc[i - 1] + series[i - 1].values.length;
    acc.push(prev);
    return acc;
  }, []);
  const paths = series.map((s, i) => {
    const startIdx = startIdxs[i];
    const points = s.values.map((v, j) => [x(startIdx + j), y(v)] as const);
    const d = points.map(([px, py], j) => `${j === 0 ? "M" : "L"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
    return { ...s, d, startIdx };
  });

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="group" aria-label="Loss curve">
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={width - pad.r} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.04)" />
          <text x={pad.l - 6} y={t.y} textAnchor="end" dominantBaseline="middle" fontSize="9" fill="rgba(148,163,184,0.6)" className="font-mono">
            {t.v.toFixed(2)}
          </text>
        </g>
      ))}
      {paths.map((p) => (
        <motion.path
          key={p.phase}
          d={p.d}
          stroke={p.color}
          strokeWidth="1.6"
          fill="none"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.4, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 4px ${p.color})` }}
        />
      ))}
      {paths.map((p) =>
        p.values.map((v, i) => (
          <circle key={`${p.phase}-${i}`} cx={x(p.startIdx + i)} cy={y(v)} r="1.5" fill={p.color} />
        ))
      )}
    </svg>
  );
}

function RobustnessChart() {
  const width = 600;
  const height = 220;
  const pad = { l: 32, r: 12, t: 12, b: 26 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const x = (i: number) => pad.l + (i / (ROBUSTNESS.length - 1)) * innerW;
  const y = (v: number) => pad.t + (1 - v / 100) * innerH;

  const baselinePts = ROBUSTNESS.map((r, i) => [x(i), y(r.baseline)] as const);
  const hardenedPts = ROBUSTNESS.map((r, i) => [x(i), y(r.hardened)] as const);

  const toPath = (pts: readonly (readonly [number, number])[]) =>
    pts.map(([px, py], i) => `${i === 0 ? "M" : "L"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
  const toArea = (pts: readonly (readonly [number, number])[]) => {
    const top = toPath(pts);
    const last = pts[pts.length - 1][0];
    const first = pts[0][0];
    return `${top} L${last} ${height - pad.b} L${first} ${height - pad.b} Z`;
  };

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="group" aria-label="Robustness chart">
      <defs>
        <linearGradient id="rob-hardened" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--color-success)" stopOpacity="0.35" />
          <stop offset="100%" stopColor="var(--color-success)" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="rob-baseline" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--color-nightmare)" stopOpacity="0.25" />
          <stop offset="100%" stopColor="var(--color-nightmare)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 25, 50, 75, 100].map((tick) => (
        <g key={tick}>
          <line x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} stroke="rgba(255,255,255,0.04)" />
          <text x={pad.l - 6} y={y(tick)} textAnchor="end" dominantBaseline="middle" fontSize="9" fill="rgba(148,163,184,0.6)" className="font-mono">
            {tick}
          </text>
        </g>
      ))}
      <path d={toArea(baselinePts)} fill="url(#rob-baseline)" />
      <path d={toArea(hardenedPts)} fill="url(#rob-hardened)" />
      <motion.path d={toPath(baselinePts)} stroke="var(--color-nightmare)" strokeWidth="1.5" fill="none" strokeDasharray="3 3" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1 }} />
      <motion.path d={toPath(hardenedPts)} stroke="var(--color-success)" strokeWidth="1.8" fill="none" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, delay: 0.2 }} />
      {ROBUSTNESS.map((r, i) => (
        <text key={i} x={x(i)} y={height - 8} textAnchor="middle" fontSize="9" fill="rgba(148,163,184,0.55)" className="font-mono">
          {r.strength.toFixed(1)}
        </text>
      ))}
    </svg>
  );
}

export interface LiveMetricsProps {
  loading?: boolean;
}

export function LiveMetrics({ loading = false }: LiveMetricsProps = {}) {
  const [tab, setTab] = useState<"loss" | "robustness">("loss");
  const { isLive } = useDemoMode();
  return (
    <Panel
      title="Live Metrics"
      subtitle={tab === "loss" ? "Loss curves across phases" : "Robustness vs distortion strength"}
      icon={<IconActivity size={14} />}
      glow="neural"
      toolbar={
        <div className="flex items-center gap-2">
          {!isLive && <Badge variant="warning" size="xs">demo data</Badge>}
          <div className="flex items-center gap-1 rounded-md border border-white/[0.06] p-0.5">
            {([["loss", "Loss"], ["robustness", "Robustness"]] as const).map(([k, l]) => (
              <button
                key={k}
                type="button"
                onClick={() => setTab(k)}
                disabled={loading}
                className={[
                  "rounded px-2 py-1 text-[11px] cursor-pointer transition-colors",
                  tab === k ? "bg-white/[0.06] text-slate-100" : "text-slate-400 hover:text-slate-300",
                  loading ? "pointer-events-none opacity-50" : "",
                ].join(" ")}
              >
                {l}
              </button>
            ))}
          </div>
        </div>
      }
    >
      {loading ? (
        <SkeletonChart height={220} />
      ) : tab === "loss" ? (
        <>
          <LossChart series={LOSS_SERIES} />
          <div className="mt-3 flex flex-wrap items-center justify-center gap-3 text-[11px]">
            {LOSS_SERIES.map((s) => (
              <span key={s.phase} className="inline-flex items-center gap-1.5 capitalize text-slate-400">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
                {s.phase}
              </span>
            ))}
          </div>
        </>
      ) : (
        <>
          <RobustnessChart />
          <div className="mt-3 flex items-center justify-center gap-4 text-[11px]">
            <span className="inline-flex items-center gap-1.5 text-slate-400">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_6px_var(--color-success)]" />
              Hardened
            </span>
            <span className="inline-flex items-center gap-1.5 text-slate-400">
              <span className="inline-block h-2 w-2 rounded-full bg-nightmare shadow-[0_0_6px_var(--color-nightmare)]" />
              Baseline
            </span>
            <span className="ml-auto inline-flex items-center gap-1 text-emerald-300">
              <IconTrend size={11} /> +29.3 net robustness
            </span>
          </div>
        </>
      )}
      {!loading && (
        <>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              { label: "Steps", value: "1,024" },
              { label: "Tokens", value: "204k" },
              { label: "Throughput", value: "1.2 k/s" },
              { label: "Wall time", value: "12m 04s" },
            ].map((m) => (
              <div key={m.label} className="rounded-md border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                <p className="text-[10px] uppercase tracking-widest text-slate-400">{m.label}</p>
                <p className="mt-0.5 font-mono text-xs text-slate-100">{m.value}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}
