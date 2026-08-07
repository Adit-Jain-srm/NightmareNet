"use client";

import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

const base = ({
  size = 16,
  strokeWidth = 1.5,
  ...rest
}: IconProps) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  ...rest,
});

export const IconHome = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 11l9-8 9 8" />
    <path d="M5 9.5V20a1 1 0 001 1h4v-6h4v6h4a1 1 0 001-1V9.5" />
  </svg>
);

export const IconBeaker = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M9 3h6" />
    <path d="M10 3v5L4.5 18a2 2 0 001.7 3h11.6a2 2 0 001.7-3L14 8V3" />
    <path d="M7 14h10" />
  </svg>
);

export const IconActivity = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 12h4l3-9 4 18 3-9h4" />
  </svg>
);

export const IconRadar = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="5" />
    <circle cx="12" cy="12" r="1.5" fill="currentColor" />
    <path d="M12 12L19 8" />
  </svg>
);

export const IconLayers = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 3l9 5-9 5-9-5 9-5z" />
    <path d="M3 13l9 5 9-5" />
    <path d="M3 18l9 5 9-5" />
  </svg>
);

export const IconSparkle = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" />
    <path d="M19 17l.6 1.8L21 19.5l-1.4.7L19 22l-.6-1.8L17 19.5l1.4-.7L19 17z" />
  </svg>
);

export const IconHistory = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 12a9 9 0 109-9 9 9 0 00-7.5 4" />
    <path d="M3 4v4h4" />
    <path d="M12 7v5l3 2" />
  </svg>
);

export const IconBenchmark = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M4 20h16" />
    <rect x="6" y="10" width="3" height="10" />
    <rect x="11" y="6" width="3" height="14" />
    <rect x="16" y="13" width="3" height="7" />
  </svg>
);

export const IconGit = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="6" cy="6" r="2.5" />
    <circle cx="6" cy="18" r="2.5" />
    <circle cx="18" cy="12" r="2.5" />
    <path d="M6 8.5v7" />
    <path d="M6 12h6a3 3 0 003-3v-.5" />
  </svg>
);

export const IconSettings = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 00.4 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.8-.4 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.8.4l-.1.1A2 2 0 113.4 17l.1-.1a1.7 1.7 0 00.4-1.8 1.7 1.7 0 00-1.5-1H2a2 2 0 110-4h.1A1.7 1.7 0 003.6 9a1.7 1.7 0 00-.4-1.8l-.1-.1A2 2 0 117 4.3l.1.1a1.7 1.7 0 001.8.4H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.8-.4l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.4 1.8V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z" />
  </svg>
);

export const IconSearch = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3" />
  </svg>
);

export const IconBell = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M6 8a6 6 0 0112 0c0 7 3 7 3 9H3c0-2 3-2 3-9z" />
    <path d="M9.5 21a2.5 2.5 0 005 0" />
  </svg>
);

export const IconCpu = (p: IconProps) => (
  <svg {...base(p)}>
    <rect x="5" y="5" width="14" height="14" rx="2" />
    <rect x="9" y="9" width="6" height="6" />
    <path d="M9 1.5V5M15 1.5V5M9 19v3.5M15 19v3.5M1.5 9H5M1.5 15H5M19 9h3.5M19 15h3.5" />
  </svg>
);

export const IconRunning = (p: IconProps) => (
  <svg {...base(p)}>
    <polygon points="6 4 20 12 6 20 6 4" />
  </svg>
);

export const IconCheck = (p: IconProps) => (
  <svg {...base(p)}>
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

export const IconX = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M18 6L6 18M6 6l12 12" />
  </svg>
);

export const IconChevronRight = (p: IconProps) => (
  <svg {...base(p)}>
    <polyline points="9 18 15 12 9 6" />
  </svg>
);

export const IconPlus = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const IconFilter = (p: IconProps) => (
  <svg {...base(p)}>
    <polygon points="3 4 21 4 14 13 14 20 10 20 10 13 3 4" />
  </svg>
);

export const IconDownload = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 3v12" />
    <polyline points="7 10 12 15 17 10" />
    <path d="M5 21h14" />
  </svg>
);

export const IconClock = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="9" />
    <polyline points="12 7 12 12 15 14" />
  </svg>
);

export const IconCommand = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M9 6V4.5A2.5 2.5 0 116.5 7H9zM15 6v1.5a2.5 2.5 0 102.5-2.5H15zM9 18v1.5A2.5 2.5 0 116.5 17H9zM15 18v-1.5a2.5 2.5 0 102.5 2.5H15zM9 7v10h6V7H9z" />
  </svg>
);

export const IconShield = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 2l8 4v6c0 5-3.5 9-8 10-4.5-1-8-5-8-10V6l8-4z" />
  </svg>
);

export const IconKey = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="8" cy="14" r="4" />
    <path d="M11 11l9-9" />
    <path d="M16 6l3 3" />
    <path d="M19 3l2 2" />
  </svg>
);

export const IconGpu = (p: IconProps) => (
  <svg {...base(p)}>
    <rect x="2" y="7" width="20" height="11" rx="2" />
    <circle cx="8" cy="12.5" r="2.2" />
    <circle cx="16" cy="12.5" r="2.2" />
    <path d="M2 18v2M22 18v2" />
  </svg>
);

export const IconTrend = (p: IconProps) => (
  <svg {...base(p)}>
    <polyline points="3 17 9 11 13 15 21 7" />
    <polyline points="14 7 21 7 21 14" />
  </svg>
);

export const IconQueue = (p: IconProps) => (
  <svg {...base(p)}>
    <line x1="8" y1="6" x2="21" y2="6" />
    <line x1="8" y1="12" x2="21" y2="12" />
    <line x1="8" y1="18" x2="21" y2="18" />
    <circle cx="3.5" cy="6" r="1.5" />
    <circle cx="3.5" cy="12" r="1.5" />
    <circle cx="3.5" cy="18" r="1.5" />
  </svg>
);

export const IconWand = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M15 4V2" />
    <path d="M15 16v-2" />
    <path d="M8 9h2" />
    <path d="M20 9h2" />
    <path d="M17.8 11.8L19 13" />
    <path d="M15 9l-9 9-3 1 1-3 9-9" />
    <path d="M17.8 6.2L19 5" />
  </svg>
);

export const IconDatabase = (p: IconProps) => (
  <svg {...base(p)}>
    <ellipse cx="12" cy="5" rx="8" ry="3" />
    <path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
    <path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" />
  </svg>
);

export const IconKebab = (p: IconProps) => (
  <svg {...base({ ...p, size: p.size ?? 14 })}>
    <circle cx="12" cy="5.5" r="1.2" fill="currentColor" stroke="none" />
    <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
    <circle cx="12" cy="18.5" r="1.2" fill="currentColor" stroke="none" />
  </svg>
);

export const IconMenu = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M4 6h16M4 12h16M4 18h16" />
  </svg>
);

export const IconSpinner = (p: IconProps) => (
  <svg {...base({ ...p, strokeWidth: 2 })} className="animate-spin">
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
  </svg>
);
