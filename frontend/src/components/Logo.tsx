"use client";

import { useId } from "react";
import { motion } from "framer-motion";

interface LogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
  animated?: boolean;
  className?: string;
}

const sizes = {
  sm: { icon: 20, text: "text-xs" },
  md: { icon: 28, text: "text-sm" },
  lg: { icon: 40, text: "text-lg" },
  xl: { icon: 56, text: "text-2xl" },
};

export default function Logo({
  size = "md",
  showText = true,
  animated = true,
  className = "",
}: LogoProps) {
  const uid = useId();
  const { icon, text } = sizes[size];

  const gradNeural = `${uid}-neural`;
  const gradDream = `${uid}-dream`;
  const gradNightmare = `${uid}-nightmare`;
  const filtGlow = `${uid}-glow`;

  const IconSvg = (
    <svg
      width={icon}
      height={icon}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="shrink-0"
    >
      <defs>
        <linearGradient id={gradNeural} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="50%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#f87171" />
        </linearGradient>
        <linearGradient id={gradDream} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#6366f1" />
        </linearGradient>
        <linearGradient id={gradNightmare} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f87171" />
          <stop offset="100%" stopColor="#dc2626" />
        </linearGradient>
        <filter id={filtGlow} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <circle
        cx="32" cy="32" r="28"
        stroke={`url(#${gradNeural})`}
        strokeWidth="2" fill="none" opacity="0.3"
      />

      <ellipse
        cx="32" cy="32" rx="20" ry="12"
        stroke={`url(#${gradDream})`}
        strokeWidth="1.5" fill="none" opacity="0.5"
        transform="rotate(-30 32 32)"
      />
      <ellipse
        cx="32" cy="32" rx="20" ry="12"
        stroke={`url(#${gradNightmare})`}
        strokeWidth="1.5" fill="none" opacity="0.5"
        transform="rotate(30 32 32)"
      />

      <g filter={`url(#${filtGlow})`}>
        <path
          d="M32 16c-8.837 0-16 7.163-16 16s7.163 16 16 16 16-7.163 16-16-7.163-16-16-16z"
          fill="none" stroke={`url(#${gradNeural})`} strokeWidth="2"
        />
        <path
          d="M24 28c2-4 6-6 8-6s6 2 8 6"
          stroke="#22d3ee" strokeWidth="1.5" fill="none" strokeLinecap="round"
        />
        <path
          d="M24 36c2 4 6 6 8 6s6-2 8-6"
          stroke="#818cf8" strokeWidth="1.5" fill="none" strokeLinecap="round"
        />
        <path
          d="M28 24v16"
          stroke="#f87171" strokeWidth="1.5" strokeLinecap="round" opacity="0.7"
        />
        <path
          d="M36 24v16"
          stroke="#f87171" strokeWidth="1.5" strokeLinecap="round" opacity="0.7"
        />
        <circle cx="32" cy="32" r="4" fill={`url(#${gradNeural})`} />
      </g>

      <circle cx="32" cy="8" r="3" fill="#34d399" opacity="0.9" />
      <circle cx="56" cy="32" r="3" fill="#818cf8" opacity="0.9" />
      <circle cx="32" cy="56" r="3" fill="#f87171" opacity="0.9" />
      <circle cx="8" cy="32" r="3" fill="#22d3ee" opacity="0.9" />
    </svg>
  );

  const content = (
    <div className={`flex items-center gap-2.5 ${className}`}>
      {animated ? (
        <motion.div
          whileHover={{ scale: 1.05, rotate: 5 }}
          transition={{ type: "spring", stiffness: 400, damping: 20 }}
        >
          {IconSvg}
        </motion.div>
      ) : (
        IconSvg
      )}
      {showText && (
        <span className={`font-mono font-bold ${text} tracking-tight`}>
          <span className="text-gradient-neural">Nightmare</span>
          <span className="text-text-dim logo-net">Net</span>
        </span>
      )}
    </div>
  );

  return content;
}
