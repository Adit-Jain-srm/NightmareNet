"use client";

import { motion } from "framer-motion";
import { Panel } from "./Panel";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { useDemoMode } from "@/lib/hooks";
import { IconRadar } from "./icons";

const AXES = [
  { key: "character", label: "Character" },
  { key: "word", label: "Word" },
  { key: "semantic", label: "Semantic" },
  { key: "syntactic", label: "Syntactic" },
  { key: "attacks", label: "Attacks" },
] as const;

interface RadarSeries {
  label: string;
  color: string;
  values: number[];
}

const SERIES: RadarSeries[] = [
  { label: "Hardened", color: "var(--color-success)", values: [86, 81, 84, 79, 72] },
  { label: "Baseline", color: "var(--color-nightmare)", values: [62, 58, 71, 65, 41] },
];

export function RobustnessRadar() {
  const { isLive } = useDemoMode();
  const hasEvaluationData = SERIES.length >= 2;
  const size = 280;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 32;
  const n = AXES.length;

  const point = (i: number, frac: number) => {
    const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
    return [cx + Math.cos(angle) * radius * frac, cy + Math.sin(angle) * radius * frac] as const;
  };

  const polygonFor = (vals: number[]) =>
    vals.map((v, i) => point(i, v / 100).join(",")).join(" ");

  return (
    <Panel
      title="Robustness Radar"
      subtitle="5-axis · hardened vs baseline"
      icon={<IconRadar size={14} />}
      glow="dream"
      toolbar={
        hasEvaluationData ? (
          <div className="flex items-center gap-2">
            {!isLive && <Badge variant="warning" size="xs">demo data</Badge>}
            <Badge variant="success" size="xs">avg +18.4</Badge>
          </div>
        ) : undefined
      }
    >
      {!hasEvaluationData ? (
        <EmptyState
          icon={<IconRadar size={18} />}
          title="No evaluation data"
          description="There are currently no robustness metrics to display. Run an evaluation to generate radar data."
          primary={{ label: "Run Evaluation", onClick: () => {} }}
        />
      ) : (
        <>
          <div className="flex flex-col items-center gap-4 lg:flex-row lg:items-start lg:justify-between">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="group" aria-label="Robustness radar">
              <defs>
                <radialGradient id="rad-grad" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="var(--color-neural)" stopOpacity="0.04" />
                  <stop offset="100%" stopColor="var(--color-neural)" stopOpacity="0" />
                </radialGradient>
              </defs>
              <circle cx={cx} cy={cy} r={radius} fill="url(#rad-grad)" />
              {[0.25, 0.5, 0.75, 1].map((f) => (
                <polygon
                  key={f}
                  points={AXES.map((_, i) => point(i, f).join(",")).join(" ")}
                  fill="none"
                  stroke="rgba(255,255,255,0.04)"
                />
              ))}
              {AXES.map((_, i) => {
                const [px, py] = point(i, 1);
                return <line key={i} x1={cx} y1={cy} x2={px} y2={py} stroke="rgba(255,255,255,0.04)" />;
              })}

              {SERIES.map((s, idx) => (
                <g key={s.label}>
                  <motion.polygon
                    points={polygonFor(s.values)}
                    fill={s.color}
                    fillOpacity={idx === 0 ? 0.15 : 0.08}
                    stroke={s.color}
                    strokeWidth={idx === 0 ? 1.6 : 1.2}
                    strokeDasharray={idx === 0 ? undefined : "3 3"}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.7, delay: idx * 0.15 }}
                    style={{ transformOrigin: `${cx}px ${cy}px`, filter: idx === 0 ? `drop-shadow(0 0 6px ${s.color})` : undefined }}
                  />
                  {s.values.map((v, i) => {
                    const [px, py] = point(i, v / 100);
                    return <circle key={i} cx={px} cy={py} r={idx === 0 ? 2.5 : 1.5} fill={s.color} />;
                  })}
                </g>
              ))}

              {AXES.map((a, i) => {
                const [px, py] = point(i, 1.16);
                return (
                  <text
                    key={a.key}
                    x={px}
                    y={py}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="10"
                    fill="rgba(148,163,184,0.85)"
                    className="font-mono uppercase tracking-wider"
                  >
                    {a.label}
                  </text>
                );
              })}
            </svg>

            <div className="w-full flex-1 space-y-2 lg:max-w-[180px]">
              {AXES.map((a, i) => {
                const h = SERIES[0].values[i];
                const b = SERIES[1].values[i];
                const delta = h - b;
                return (
                  <div key={a.key} className="rounded-md border border-white/[0.05] bg-white/[0.02] p-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] uppercase tracking-widest text-slate-400">{a.label}</span>
                      <span className="font-mono text-xs text-emerald-300">+{delta}</span>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-[11px]">
                      <span className="font-mono text-slate-100">{h}</span>
                      <span className="text-slate-300">vs</span>
                      <span className="font-mono text-slate-400">{b}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-3 flex items-center justify-center gap-4 text-[11px]">
            {SERIES.map((s) => (
              <span key={s.label} className="inline-flex items-center gap-1.5 text-slate-400">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
                {s.label}
              </span>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}
