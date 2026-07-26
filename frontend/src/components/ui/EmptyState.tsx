"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

export interface EmptyStateAction {
  label: string;
  onClick: () => void;
}

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description: string;
  primary?: EmptyStateAction;
  secondary?: EmptyStateAction;
  illustration?: ReactNode;
  className?: string;
}

function DefaultIllustration() {
  return (
    <svg
      viewBox="0 0 280 160"
      width="280"
      height="160"
      role="group"
      aria-hidden="true"
      className="select-none"
    >
      <defs>
        <radialGradient id="es-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(34,211,238,0.18)" />
          <stop offset="60%" stopColor="rgba(129,140,248,0.05)" />
          <stop offset="100%" stopColor="rgba(0,0,0,0)" />
        </radialGradient>
        <linearGradient id="es-line" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="rgba(129,140,248,0.0)" />
          <stop offset="50%" stopColor="rgba(34,211,238,0.45)" />
          <stop offset="100%" stopColor="rgba(129,140,248,0.0)" />
        </linearGradient>
      </defs>

      <rect width="280" height="160" fill="transparent" />
      <ellipse cx="140" cy="80" rx="120" ry="60" fill="url(#es-glow)" />

      {/* Constellation lines */}
      <path
        d="M 50 110 L 95 60 L 150 90 L 210 50 L 240 100"
        stroke="url(#es-line)"
        strokeWidth="1"
        fill="none"
        strokeLinecap="round"
      />
      <path
        d="M 95 60 L 150 30"
        stroke="rgba(129,140,248,0.25)"
        strokeWidth="0.75"
        fill="none"
        strokeLinecap="round"
      />
      <path
        d="M 150 90 L 110 130"
        stroke="rgba(129,140,248,0.18)"
        strokeWidth="0.75"
        fill="none"
        strokeLinecap="round"
      />

      {/* Stars — primary nodes */}
      {[
        { cx: 50, cy: 110, r: 1.6, c: "rgba(129,140,248,0.9)" },
        { cx: 95, cy: 60, r: 2.2, c: "rgba(34,211,238,0.95)" },
        { cx: 150, cy: 90, r: 1.8, c: "rgba(129,140,248,0.85)" },
        { cx: 210, cy: 50, r: 2.4, c: "rgba(34,211,238,1)" },
        { cx: 240, cy: 100, r: 1.4, c: "rgba(129,140,248,0.8)" },
        { cx: 150, cy: 30, r: 1.2, c: "rgba(248,113,113,0.55)" },
        { cx: 110, cy: 130, r: 1.2, c: "rgba(129,140,248,0.6)" },
      ].map((s, i) => (
        <g key={i}>
          <circle cx={s.cx} cy={s.cy} r={s.r * 3} fill={s.c} opacity="0.12" />
          <circle cx={s.cx} cy={s.cy} r={s.r} fill={s.c} />
        </g>
      ))}

      {/* Faint scattered starfield */}
      {[
        [30, 40], [70, 130], [130, 60], [170, 130], [200, 90],
        [230, 35], [255, 65], [40, 70], [185, 75], [80, 35],
        [225, 130], [125, 115], [60, 95], [195, 115], [165, 50],
      ].map(([cx, cy], i) => (
        <circle
          key={`s-${i}`}
          cx={cx}
          cy={cy}
          r={0.7}
          fill="rgba(148,163,184,0.35)"
        />
      ))}
    </svg>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  primary,
  secondary,
  illustration,
  className = "",
}: EmptyStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={[
        "mx-auto flex w-full max-w-md flex-col items-center justify-center text-center",
        "rounded-xl border border-white/[0.06] bg-void/40 backdrop-blur-sm",
        "px-6 py-12 sm:px-8 sm:py-12 md:px-10 md:py-16",
        className,
      ].join(" ")}
    >
      <div className="relative mb-5 flex items-center justify-center">
        {illustration ?? <DefaultIllustration />}
        {icon && (
          <span
            className="absolute flex h-11 w-11 items-center justify-center rounded-xl border border-white/[0.06] bg-abyss/80 text-neural shadow-[0_0_24px_rgba(34,211,238,0.18)] backdrop-blur"
            aria-hidden="true"
          >
            {icon}
          </span>
        )}
      </div>

      <h3 className="text-sm font-semibold tracking-tight text-slate-100">
        {title}
      </h3>
      <p className="mt-1.5 max-w-[28rem] text-[12.5px] leading-relaxed text-slate-400">
        {description}
      </p>

      {(primary || secondary) && (
        <div className="mt-5 flex flex-col items-center justify-center gap-2 sm:flex-row">
          {primary && (
            <button
              type="button"
              onClick={primary.onClick}
              className={[
                "inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg",
                "border border-neural/40 bg-neural text-void",
                "px-4 py-2 text-xs font-semibold",
                "shadow-[0_0_20px_rgba(6,182,212,0.22)]",
                "transition-colors hover:bg-neural/90",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/60",
              ].join(" ")}
            >
              {primary.label}
            </button>
          )}
          {secondary && (
            <button
              type="button"
              onClick={secondary.onClick}
              className={[
                "inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg",
                "border border-white/10 bg-white/[0.03] text-slate-200",
                "px-4 py-2 text-xs font-medium",
                "transition-colors hover:bg-white/[0.06]",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/40",
              ].join(" ")}
            >
              {secondary.label}
            </button>
          )}
        </div>
      )}
    </motion.div>
  );
}
