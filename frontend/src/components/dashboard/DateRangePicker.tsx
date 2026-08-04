"use client";

import { useCallback, useId } from "react";

export type DatePreset = "all" | "7d" | "30d" | "custom";

export interface DateRange {
  preset: DatePreset;
  startDate?: string; // ISO date string "YYYY-MM-DD"
  endDate?: string;   // ISO date string "YYYY-MM-DD"
}

interface DateRangePickerProps {
  value: DateRange;
  onChange: (range: DateRange) => void;
  className?: string;
}

const PRESETS: { value: DatePreset; label: string }[] = [
  { value: "all", label: "All time" },
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
  { value: "custom", label: "Custom range" },
];

/**
 * DateRangePicker — toolbar filter for ExperimentList.
 *
 * Renders a preset <select> for common ranges and, when "custom" is chosen,
 * reveals accessible start/end <input type="date"> fields. All state is
 * lifted to the parent; this component is fully controlled.
 *
 * Keyboard: Tab navigates all inputs. Escape propagated naturally by the
 * browser. ARIA labels on every interactive element satisfy a11y requirements.
 */
export function DateRangePicker({ value, onChange, className = "" }: DateRangePickerProps) {
  const baseId = useId();
  const startId = `${baseId}-start`;
  const endId = `${baseId}-end`;

  const handlePresetChange = useCallback(
    (preset: DatePreset) => {
      if (preset !== "custom") {
        onChange({ preset });
      } else {
        onChange({ preset, startDate: value.startDate ?? "", endDate: value.endDate ?? "" });
      }
    },
    [onChange, value.startDate, value.endDate],
  );

  const handleStartChange = useCallback(
    (startDate: string) => onChange({ ...value, startDate }),
    [onChange, value],
  );

  const handleEndChange = useCallback(
    (endDate: string) => onChange({ ...value, endDate }),
    [onChange, value],
  );

  return (
    <div className={["flex items-center gap-2", className].join(" ")} role="group" aria-label="Date range filter">
      <select
        value={value.preset}
        onChange={(e) => handlePresetChange(e.target.value as DatePreset)}
        aria-label="Select date preset"
        className={[
          "h-7 cursor-pointer rounded-md border border-white/[0.08] bg-white/[0.04]",
          "px-2 py-0 text-xs text-slate-300 outline-none",
          "focus-visible:ring-2 focus-visible:ring-neural/50",
          "hover:border-white/20 hover:bg-white/[0.07] transition-colors",
        ].join(" ")}
      >
        {PRESETS.map((p) => (
          <option key={p.value} value={p.value} className="bg-abyss text-slate-200 dark:bg-abyss dark:text-slate-200">
            {p.label}
          </option>
        ))}
      </select>

      {value.preset === "custom" && (
        <div className="flex items-center gap-1.5">
          <label htmlFor={startId} className="sr-only">
            Start date
          </label>
          <input
            id={startId}
            type="date"
            value={value.startDate ?? ""}
            onChange={(e) => handleStartChange(e.target.value)}
            aria-label="Start date"
            className={[
              "h-7 cursor-pointer rounded-md border border-white/[0.08] bg-white/[0.04]",
              "px-2 py-0 text-xs text-slate-300 outline-none",
              "focus-visible:ring-2 focus-visible:ring-neural/50",
              "hover:border-white/20 transition-colors",
            ].join(" ")}
          />
          <span className="text-xs text-slate-500" aria-hidden="true">→</span>
          <label htmlFor={endId} className="sr-only">
            End date
          </label>
          <input
            id={endId}
            type="date"
            value={value.endDate ?? ""}
            min={value.startDate}
            onChange={(e) => handleEndChange(e.target.value)}
            aria-label="End date"
            className={[
              "h-7 cursor-pointer rounded-md border border-white/[0.08] bg-white/[0.04]",
              "px-2 py-0 text-xs text-slate-300 outline-none",
              "focus-visible:ring-2 focus-visible:ring-neural/50",
              "hover:border-white/20 transition-colors",
            ].join(" ")}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Normalises the `createdAt` display strings used in the SAMPLE dataset
 * (e.g. "2m ago", "1d ago", "6h ago") into a comparable absolute timestamp.
 * When a real ISO date string is supplied it is parsed directly.
 *
 * Returns `null` for unrecognised formats so the caller can decide to include
 * the row conservatively.
 */
export function resolveCreatedAt(createdAt: string): Date | null {
  const iso = Date.parse(createdAt);
  if (!isNaN(iso)) return new Date(iso);

  const now = Date.now();
  const relMatch = createdAt.match(/^(\d+)\s*(s|m|h|d|w)(?:\s+ago)?$/i);
  if (!relMatch) return null;

  const amount = parseInt(relMatch[1], 10);
  const unit = relMatch[2].toLowerCase();
  const msMap: Record<string, number> = {
    s: 1_000,
    m: 60_000,
    h: 3_600_000,
    d: 86_400_000,
    w: 604_800_000,
  };
  return new Date(now - amount * (msMap[unit] ?? 0));
}

/**
 * Returns true when the given experiment timestamp falls within the selected
 * DateRange.
 */
export function isInDateRange(createdAt: string, range: DateRange): boolean {
  if (range.preset === "all") return true;

  const ts = resolveCreatedAt(createdAt);
  if (ts === null) return true; // unknown — include conservatively

  if (range.preset === "7d") {
    return ts.getTime() >= Date.now() - 7 * 86_400_000;
  }
  if (range.preset === "30d") {
    return ts.getTime() >= Date.now() - 30 * 86_400_000;
  }

  // custom
  const start = range.startDate ? new Date(range.startDate).getTime() : -Infinity;
  const end = range.endDate ? new Date(range.endDate + "T23:59:59").getTime() : Infinity;
  return ts.getTime() >= start && ts.getTime() <= end;
}
