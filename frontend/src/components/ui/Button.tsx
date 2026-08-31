"use client";

import { motion, type HTMLMotionProps } from "framer-motion";
import { forwardRef, type ReactNode } from "react";
import { useSounds } from "@/lib/sounds";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends Omit<HTMLMotionProps<"button">, "ref" | "children"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  children?: ReactNode;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-neural text-void hover:bg-neural/90 border border-neural/40 shadow-glow-button",
  secondary:
    "bg-white/5 text-slate-200 border border-white/10 hover:bg-white/10",
  ghost: "bg-transparent text-slate-300 hover:bg-white/5 border border-transparent",
  danger:
    "bg-nightmare/20 text-red-300 border border-red-500/30 hover:bg-nightmare/30",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2 text-sm",
  lg: "px-6 py-3 text-base",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = "primary",
      size = "md",
      loading = false,
      className = "",
      children,
      disabled,
      onMouseDown,
      ...props
    },
    ref
  ) => {
    const { playClick } = useSounds();

    const handleMouseDown = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (e.detail !== 0) playClick();
      onMouseDown?.(e);
    };

    return (
      <motion.button
        ref={ref}
        whileHover={{ scale: disabled || loading ? 1 : 1.02 }}
        whileTap={{ scale: disabled || loading ? 1 : 0.98 }}
        className={[
          "inline-flex items-center justify-center gap-2 rounded-lg font-medium",
          "transition-colors focus-visible:outline-none focus-visible:ring-2",
          "focus-visible:ring-neural/50 disabled:opacity-50 disabled:pointer-events-none",
          variantClasses[variant],
          sizeClasses[size],
          className,
        ].join(" ")}
        disabled={disabled || loading}
        onMouseDown={handleMouseDown}
        {...props}
      >
        {loading ? (
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
        ) : null}
        {children}
      </motion.button>
    );
  }
);

Button.displayName = "Button";
