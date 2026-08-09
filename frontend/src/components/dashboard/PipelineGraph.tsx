"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

interface PhaseNode {
  key: string;
  label: string;
  color: string;
  description: string;
}

const PHASES: PhaseNode[] = [
  { key: "wake", label: "Wake", color: "#22d3ee", description: "Anchor on clean data" },
  { key: "dream", label: "Dream", color: "#818cf8", description: "Learn invariance via mild distortions" },
  { key: "nightmare", label: "Nightmare", color: "#ef4444", description: "Adversarial stress testing" },
  { key: "compress", label: "Compress", color: "#f59e0b", description: "Distill into leaner student" },
];

const ROTATION_SPEED = 0.3;
const ELLIPSE_RX = 130;
const ELLIPSE_RY = 70;
const CENTER_X = 200;
const CENTER_Y = 140;

export function PipelineGraph() {
  const [angle, setAngle] = useState(0);
  const [paused, setPaused] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const rafRef = useRef<number>(0);
  const reducedMotionRef = useRef(false);
  const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = mq.matches;
    const handler = (e: MediaQueryListEvent) => {
      reducedMotionRef.current = e.matches;
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  useEffect(() => {
    if (paused || reducedMotionRef.current) return;

    const animate = (time: number) => {
      if (lastTimeRef.current === 0) lastTimeRef.current = time;
      const delta = time - lastTimeRef.current;
      lastTimeRef.current = time;
      setAngle((prev) => (prev + ROTATION_SPEED * (delta / 16.67)) % 360);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(rafRef.current);
      lastTimeRef.current = 0;
    };
  }, [paused]);

  useEffect(() => {
    const handleVisibility = () => {
      if (document.hidden) {
        cancelAnimationFrame(rafRef.current);
        lastTimeRef.current = 0;
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, []);

  const getNodePosition = useCallback(
    (index: number) => {
      const baseAngle = (index * 90 + angle) * (Math.PI / 180);
      return {
        x: CENTER_X + ELLIPSE_RX * Math.cos(baseAngle),
        y: CENTER_Y + ELLIPSE_RY * Math.sin(baseAngle),
      };
    },
    [angle]
  );

  const handleNodeClick = (key: string) => {
    if (expanded === key) {
      setPaused(false);
      setExpanded(null);
    } else {
      setPaused(true);
      setExpanded(key);
    }
  };

  const dashOffset = (-angle * 2) % 1000;

  return (
    <div className="relative w-full" style={{ minHeight: 280 }}>
      <svg
        viewBox="0 0 400 280"
        className="h-full w-full"
        role="group"
        aria-label="Interactive pipeline phase graph"
      >
        <defs>
          <filter id="pg-glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {PHASES.map((p) => (
            <radialGradient key={`grad-${p.key}`} id={`pg-rg-${p.key}`}>
              <stop offset="0%" stopColor={p.color} stopOpacity="0.3" />
              <stop offset="100%" stopColor={p.color} stopOpacity="0" />
            </radialGradient>
          ))}
        </defs>

        <ellipse
          cx={CENTER_X}
          cy={CENTER_Y}
          rx={ELLIPSE_RX}
          ry={ELLIPSE_RY}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="1"
          strokeDasharray="6 4"
          strokeDashoffset={dashOffset}
        />

        {PHASES.map((phase, i) => {
          const pos = getNodePosition(i);
          const nextPos = getNodePosition((i + 1) % PHASES.length);
          return (
            <line
              key={`flow-${phase.key}`}
              x1={pos.x}
              y1={pos.y}
              x2={nextPos.x}
              y2={nextPos.y}
              stroke={phase.color}
              strokeOpacity="0.15"
              strokeWidth="1"
              strokeDasharray="4 3"
              strokeDashoffset={dashOffset * 0.5}
            />
          );
        })}

        <circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={28}
          fill="rgba(2,6,23,0.9)"
          stroke="rgba(34,211,238,0.4)"
          strokeWidth="1.5"
        />
        <circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={20}
          fill="none"
          stroke="rgba(34,211,238,0.12)"
          strokeWidth="0.5"
        />
        <text
          x={CENTER_X}
          y={CENTER_Y - 4}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#22d3ee"
          fontSize="9"
          fontWeight="700"
          fontFamily="var(--font-mono)"
        >
          MODEL
        </text>
        <text
          x={CENTER_X}
          y={CENTER_Y + 8}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(148,163,184,0.7)"
          fontSize="7"
          fontFamily="var(--font-mono)"
        >
          DistilBERT
        </text>

        {PHASES.map((phase, i) => {
          const pos = getNodePosition(i);
          const isExpanded = expanded === phase.key;
          const nodeRadius = isExpanded ? 24 : 18;

          return (
            <g
              key={phase.key}
              onClick={() => handleNodeClick(phase.key)}
              onMouseEnter={(e) => {
                const circle = e.currentTarget.querySelector("circle");
                if (circle) circle.setAttribute("r", String(nodeRadius + 3));
              }}
              onMouseLeave={(e) => {
                const circle = e.currentTarget.querySelector("circle");
                if (circle) circle.setAttribute("r", String(nodeRadius));
              }}
              className="cursor-pointer"
              role="button"
              aria-label={`${phase.label}: ${phase.description}`}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") handleNodeClick(phase.key);
              }}
            >
              <circle cx={pos.x} cy={pos.y} r={32} fill={`url(#pg-rg-${phase.key})`} />
              <circle
                cx={pos.x}
                cy={pos.y}
                r={nodeRadius}
                fill="rgba(2,6,23,0.85)"
                stroke={phase.color}
                strokeWidth={isExpanded ? "2" : "1.5"}
                filter={isExpanded ? "url(#pg-glow)" : undefined}
              />
              <text
                x={pos.x}
                y={pos.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={phase.color}
                fontSize={isExpanded ? "8" : "7.5"}
                fontWeight="600"
                fontFamily="var(--font-mono)"
              >
                {phase.label.toUpperCase()}
              </text>
            </g>
          );
        })}
      </svg>

      {expanded && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-lg border border-white/[0.08] bg-[rgba(2,6,23,0.9)] px-4 py-2.5 backdrop-blur-sm"
        >
          {(() => {
            const phase = PHASES.find((p) => p.key === expanded);
            if (!phase) return null;
            return (
              <div className="text-center">
                <p className="text-[11px] font-semibold" style={{ color: phase.color }}>
                  {phase.label}
                </p>
                <p className="mt-0.5 text-[10px] text-slate-400">{phase.description}</p>
              </div>
            );
          })()}
        </motion.div>
      )}
    </div>
  );
}
