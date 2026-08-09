"use client";

import { forwardRef, type InputHTMLAttributes, type ReactNode } from "react";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
  leftIcon?: ReactNode;
  rightSlot?: ReactNode;
  containerClassName?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      hint,
      error,
      leftIcon,
      rightSlot,
      containerClassName = "",
      className = "",
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || (label ? `input-${label.replace(/\s+/g, "-").toLowerCase()}` : undefined);
    return (
      <div className={["w-full", containerClassName].join(" ")}>
        {label && (
          <label
            htmlFor={inputId}
            className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400"
          >
            {label}
          </label>
        )}
        <div className="relative">
          {leftIcon && (
            <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">
              {leftIcon}
            </span>
          )}
          <input
            ref={ref}
            id={inputId}
            className={[
              "w-full rounded-lg border bg-black/30 text-sm text-slate-100 placeholder:text-slate-300",
              "transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-neural/50",
              leftIcon ? "pl-10" : "pl-3",
              rightSlot ? "pr-10" : "pr-3",
              "py-2.5",
              error
                ? "border-nightmare/60 focus:border-nightmare"
                : "border-white/[0.08] focus:border-neural/50",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              className,
            ].join(" ")}
            {...props}
          />
          {rightSlot && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500">
              {rightSlot}
            </span>
          )}
        </div>
        {(hint || error) && (
          <p
            className={[
              "mt-1.5 text-[11px] leading-tight",
              error ? "text-nightmare-soft" : "text-slate-500",
            ].join(" ")}
          >
            {error || hint}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";
