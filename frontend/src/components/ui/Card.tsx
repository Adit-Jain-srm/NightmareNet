"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface CardProps {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  glow?: "dream" | "nightmare" | "neural" | "none";
}

const glowMap = {
  dream: "shadow-glow-dream border-dream/20",
  nightmare: "shadow-glow-nightmare border-nightmare/20",
  neural: "shadow-glow-neural border-neural/20",
  none: "border-white/10",
};

export function Card({
  title,
  subtitle,
  children,
  className = "",
  glow = "none",
}: CardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={[
        "rounded-xl border bg-void/80 backdrop-blur-md p-6",
        glowMap[glow],
        className,
      ].join(" ")}
    >
      {(title || subtitle) && (
        <header className="mb-4">
          {title ? (
            <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
          ) : null}
          {subtitle ? (
            <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
          ) : null}
        </header>
      )}
      {children}
    </motion.div>
  );
}
