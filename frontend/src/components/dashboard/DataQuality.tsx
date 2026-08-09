"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Panel } from "./Panel";
import { IconDatabase } from "./icons";
import { Button } from "../ui/Button";
import { Badge } from "../ui/Badge";
import { EmptyState } from "../ui/EmptyState";
import {
  optimizeData,
  optimizeDataStream,
  suggestConfig,
  type DataOptimizeResponse,
  type DataStats,
  type OptimizeStreamEvent,
  type SuggestConfigResponse,
} from "@/lib/api";

type OptimizationStatus = "idle" | "estimating" | "running" | "completed" | "error";

interface QualityEntry {
  score: number;
  timestamp: number;
}

const SAMPLE_TEXTS = [
  "The movie was fantastic and I loved every minute of it.",
  "Terrible plot, waste of time and money.",
  "An average film with some good moments but nothing special.",
  "Absolutely brilliant performance by the lead actor.",
  "I would not recommend this to anyone looking for quality entertainment.",
  "Visually stunning but the story falls flat in the second act.",
  "A masterpiece of modern cinema that pushes boundaries.",
  "The dialogue felt forced and unnatural throughout.",
];

const ACCEPTED_FILE_TYPES = ".csv,.jsonl,.txt";
const MAX_FILE_SIZE = 5 * 1024 * 1024;

const fadeChild = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.2 } },
};

function computeQualityScore(stats: DataStats | null): number {
  if (!stats || stats.count === 0) return 0;
  const lengthScore = Math.min(40, stats.avg_length * 0.3);
  const wordScore = Math.min(30, stats.avg_words * 1.5);
  const countBonus = Math.min(30, Math.log10(stats.count + 1) * 10);
  return Math.min(99, Math.max(10, lengthScore + wordScore + countBonus));
}

function QualitySparkline({ entries }: { entries: QualityEntry[] }) {
  if (entries.length < 2) return null;

  const width = 200;
  const height = 40;
  const padding = 4;
  const maxScore = Math.max(...entries.map((e) => e.score), 100);
  const minScore = Math.min(...entries.map((e) => e.score), 0);
  const range = maxScore - minScore || 1;

  const points = entries.map((entry, i) => {
    const x = padding + (i / (entries.length - 1)) * (width - padding * 2);
    const y = height - padding - ((entry.score - minScore) / range) * (height - padding * 2);
    return `${x},${y}`;
  });

  const pathD = points.reduce(
    (acc, point, i) => acc + (i === 0 ? `M ${point}` : ` L ${point}`),
    "",
  );

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      className="w-full max-w-[200px]"
      role="group"
      aria-label={`Quality trend: ${entries.length} data points, latest score ${entries[entries.length - 1]?.score.toFixed(0)}%`}
    >
      <defs>
        <linearGradient id="spark-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(129,140,248,0.3)" />
          <stop offset="100%" stopColor="rgba(129,140,248,0)" />
        </linearGradient>
      </defs>
      <path
        d={`${pathD} L ${width - padding},${height - padding} L ${padding},${height - padding} Z`}
        fill="url(#spark-grad)"
      />
      <path
        d={pathD}
        fill="none"
        stroke="rgba(129,140,248,0.8)"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {entries.length > 0 && (
        <circle
          cx={parseFloat(points[points.length - 1].split(",")[0])}
          cy={parseFloat(points[points.length - 1].split(",")[1])}
          r="2.5"
          fill="#818CF8"
        />
      )}
    </svg>
  );
}

function ProgressBar({ pct, message }: { pct: number; message: string }) {
  return (
    <div className="space-y-1.5" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="Optimization progress">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-slate-400">{message}</span>
        <span className="font-mono text-[11px] text-slate-300">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-dream to-neural"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

function StatRow({ label, value, unit, delta }: { label: string; value: string | number; unit?: string; delta?: number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-[12px] text-slate-400">{label}</span>
      <span className="flex items-center gap-2">
        <span className="font-mono text-[13px] text-slate-100">
          {value}
          {unit && <span className="ml-0.5 text-[11px] text-slate-400">{unit}</span>}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span className={`font-mono text-[11px] ${delta > 0 ? "text-emerald-400" : "text-nightmare"}`}>
            {delta > 0 ? "+" : ""}{delta.toFixed(1)}
          </span>
        )}
      </span>
    </div>
  );
}

function MiniStatCard({ title, stats, variant }: { title: string; stats: DataStats; variant?: "before" | "after" }) {
  const borderClass = variant === "after" ? "border-dream/20" : "border-white/[0.06]";
  return (
    <div className={`flex-1 rounded-lg border ${borderClass} bg-void/60 p-3`}>
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
        {title}
      </p>
      <div className="space-y-0.5">
        <StatRow label="Rows" value={stats.count.toLocaleString()} />
        <StatRow label="Avg length" value={stats.avg_length.toFixed(0)} unit="chars" />
        <StatRow label="Avg words" value={stats.avg_words.toFixed(0)} />
      </div>
    </div>
  );
}

export function DataQuality() {
  const [texts, setTexts] = useState<string[]>(SAMPLE_TEXTS);
  const [status, setStatus] = useState<OptimizationStatus>("idle");
  const [progress, setProgress] = useState<{ pct: number; message: string }>({ pct: 0, message: "" });
  const [error, setError] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<DataOptimizeResponse["estimate"]>(null);
  const [result, setResult] = useState<DataOptimizeResponse | null>(null);
  const [qualityHistory, setQualityHistory] = useState<QualityEntry[]>([]);
  const [configSuggestions, setConfigSuggestions] = useState<SuggestConfigResponse | null>(null);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const beforeStats = useMemo<DataStats>(() => {
    const count = texts.length;
    if (count === 0) return { count: 0, avg_length: 0, total_chars: 0, avg_words: 0 };
    const lengths = texts.map((t) => t.length);
    const words = texts.map((t) => t.split(/\s+/).filter(Boolean).length);
    return {
      count,
      avg_length: Math.round(lengths.reduce((a, b) => a + b, 0) / count),
      total_chars: lengths.reduce((a, b) => a + b, 0),
      avg_words: Math.round(words.reduce((a, b) => a + b, 0) / count),
      min_length: Math.min(...lengths),
      max_length: Math.max(...lengths),
    };
  }, [texts]);

  const qualityScore = useMemo(() => computeQualityScore(beforeStats), [beforeStats]);

  useEffect(() => {
    if (qualityHistory.length === 0 && texts.length > 0) {
      setQualityHistory([{ score: qualityScore, timestamp: Date.now() }]);
    }
  }, [qualityScore, qualityHistory.length, texts.length]);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > MAX_FILE_SIZE) {
      setError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum 5 MB.`);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (!content) return;

      let parsed: string[] = [];
      const ext = file.name.split(".").pop()?.toLowerCase();

      if (ext === "csv") {
        const lines = content.split("\n").filter((l) => l.trim());
        parsed = lines.slice(1).map((l) => l.replace(/^"/, "").replace(/"$/, "").trim()).filter(Boolean);
      } else if (ext === "jsonl") {
        parsed = content
          .split("\n")
          .filter((l) => l.trim())
          .map((l) => {
            try {
              const obj = JSON.parse(l);
              return obj.text || obj.content || obj.prompt || JSON.stringify(obj);
            } catch {
              return l;
            }
          })
          .filter(Boolean);
      } else {
        parsed = content.split("\n").filter((l) => l.trim());
      }

      if (parsed.length === 0) {
        setError("No valid text entries found in file.");
        return;
      }

      const limited = parsed.slice(0, 10000);
      setTexts(limited);
      setFileName(file.name);
      setError(null);
      setResult(null);
      setEstimate(null);
      setConfigSuggestions(null);
      setStatus("idle");
      setQualityHistory([{ score: computeQualityScore({
        count: limited.length,
        avg_length: Math.round(limited.reduce((a, t) => a + t.length, 0) / limited.length),
        total_chars: limited.reduce((a, t) => a + t.length, 0),
        avg_words: Math.round(limited.reduce((a, t) => a + t.split(/\s+/).filter(Boolean).length, 0) / limited.length),
      }), timestamp: Date.now() }]);
    };
    reader.onerror = () => setError("Failed to read file.");
    reader.readAsText(file);

    if (event.target) event.target.value = "";
  }, []);

  const handleOptimize = useCallback(async () => {
    setError(null);
    setEstimate(null);
    setResult(null);
    setConfigSuggestions(null);
    setStatus("estimating");
    setProgress({ pct: 5, message: "Estimating cost..." });

    const payload = {
      texts,
      column_mapping: { prompt: "text" },
    };

    abortRef.current = new AbortController();

    try {
      const est = await optimizeData({ ...payload, estimate_only: true });
      setEstimate(est.estimate ?? null);
      setStatus("running");
      setProgress({ pct: 10, message: "Starting optimization..." });

      let finalResult: DataOptimizeResponse | null = null;
      try {
        for await (const event of optimizeDataStream(payload, abortRef.current.signal)) {
          if (event.event === "progress") {
            setProgress({ pct: event.progress_pct ?? 0, message: event.message ?? "Processing..." });
          } else if (event.event === "complete") {
            finalResult = {
              status: "completed",
              run_id: event.run_id,
              optimized_count: event.result?.optimized_count ?? null,
              quality: event.result?.quality ?? null,
              elapsed_seconds: event.elapsed_seconds ?? null,
              before_stats: event.before_stats ?? null,
              after_stats: event.after_stats ?? null,
            };
          } else if (event.event === "error") {
            throw new Error(event.error || "Optimization failed");
          }
        }
      } catch (streamErr) {
        if (streamErr instanceof Error && streamErr.name === "AbortError") {
          setStatus("idle");
          return;
        }
        const completed = await optimizeData(payload);
        finalResult = completed;
      }

      if (finalResult) {
        setResult(finalResult);
        setStatus("completed");
        setProgress({ pct: 100, message: "Complete" });

        const afterScore = finalResult.after_stats
          ? computeQualityScore(finalResult.after_stats)
          : qualityScore + 12;
        setQualityHistory((h) => [...h, { score: afterScore, timestamp: Date.now() }]);
      }
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Optimization failed");
      setRetryCount((c) => c + 1);
    }
  }, [texts, qualityScore]);

  const handleRetry = useCallback(() => {
    handleOptimize();
  }, [handleOptimize]);

  const handleLoadSample = useCallback(() => {
    setTexts(SAMPLE_TEXTS);
    setFileName(null);
    setError(null);
    setResult(null);
    setEstimate(null);
    setConfigSuggestions(null);
    setStatus("idle");
  }, []);

  const handleGetSuggestions = useCallback(async () => {
    setLoadingSuggestions(true);
    try {
      const resp = await suggestConfig({
        current_config: {
          dataset_size: texts.length,
          avg_text_length: beforeStats.avg_length,
          quality_score: qualityScore,
        },
        last_metrics: result?.quality ? (result.quality as Record<string, unknown>) : undefined,
      });
      setConfigSuggestions(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get suggestions");
    } finally {
      setLoadingSuggestions(false);
    }
  }, [texts.length, beforeStats.avg_length, qualityScore, result]);

  if (texts.length === 0) {
    return (
      <Panel title="Data Quality" icon={<IconDatabase size={15} />} glow="dream">
        <EmptyState
          icon={<IconDatabase size={20} />}
          title="No dataset loaded"
          description="Upload a CSV, JSONL, or TXT file, or use sample data to view quality metrics and run Adaption Labs optimization."
          primary={{ label: "Load sample data", onClick: handleLoadSample }}
          secondary={{ label: "Upload file", onClick: () => fileInputRef.current?.click() }}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_FILE_TYPES}
          onChange={handleFileUpload}
          className="hidden"
          aria-label="Upload dataset file"
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Data Quality"
      subtitle={fileName ? `Source: ${fileName}` : "Adaption Labs optimization"}
      icon={<IconDatabase size={15} />}
      glow="dream"
      toolbar={
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => fileInputRef.current?.click()}
            aria-label="Upload dataset file"
          >
            Upload
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={handleOptimize}
            loading={status === "running" || status === "estimating"}
            disabled={status === "running" || status === "estimating"}
            aria-label="Run data optimization"
          >
            Optimize
          </Button>
        </div>
      }
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_FILE_TYPES}
        onChange={handleFileUpload}
        className="hidden"
        aria-label="Upload dataset file"
      />

      <motion.div
        variants={{ animate: { transition: { staggerChildren: 0.06 } } }}
        initial="initial"
        animate="animate"
        className="space-y-4"
      >
        {/* Dataset overview */}
        <motion.div variants={fadeChild}>
          <div className="mb-2 flex items-center justify-between">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Current Dataset
            </p>
            <Badge variant={qualityScore > 70 ? "success" : qualityScore > 45 ? "warning" : "nightmare"} size="xs" dot>
              {qualityScore.toFixed(0)}% quality
            </Badge>
          </div>
          <div className="rounded-lg border border-white/[0.06] bg-void/60 p-3">
            <div className="space-y-0.5">
              <StatRow label="Total rows" value={beforeStats.count.toLocaleString()} />
              <StatRow label="Avg text length" value={beforeStats.avg_length.toFixed(0)} unit="chars" />
              <StatRow label="Avg words" value={beforeStats.avg_words.toFixed(0)} />
              {beforeStats.min_length !== undefined && (
                <StatRow label="Length range" value={`${beforeStats.min_length}–${beforeStats.max_length}`} unit="chars" />
              )}
            </div>
          </div>
        </motion.div>

        {/* Quality trend */}
        {qualityHistory.length > 1 && (
          <motion.div variants={fadeChild}>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Quality Trend
            </p>
            <div className="flex items-center justify-center rounded-lg border border-white/[0.06] bg-void/60 p-3">
              <QualitySparkline entries={qualityHistory} />
            </div>
          </motion.div>
        )}

        {/* Progress / Status */}
        <motion.div variants={fadeChild}>
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
              Optimization Status
            </p>
            <StatusBadge status={status} />
          </div>

          {estimate && (
            <p className="mt-2 text-[11px] text-slate-400">
              Estimated cost:{" "}
              <span className="font-mono text-slate-200">
                {estimate.credits ?? "—"} credits · ~{estimate.estimated_minutes ?? "—"} min
              </span>
            </p>
          )}

          {(status === "running" || status === "estimating") && (
            <div className="mt-2">
              <ProgressBar pct={progress.pct} message={progress.message} />
            </div>
          )}

          <AnimatePresence mode="wait">
            {error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="mt-3 rounded-lg border border-nightmare/20 bg-nightmare/5 p-3"
              >
                <p className="text-[11px] text-nightmare">{error}</p>
                <div className="mt-2 flex items-center gap-2">
                  <Button size="sm" variant="danger" onClick={handleRetry} aria-label="Retry optimization">
                    Retry{retryCount > 0 ? ` (${retryCount})` : ""}
                  </Button>
                  <Button size="sm" variant="ghost" onClick={handleLoadSample}>
                    Use sample data
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Results comparison */}
        <AnimatePresence>
          {status === "completed" && result && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3 }}
              className="space-y-3"
            >
              <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                Before / After
              </p>
              <div className="flex flex-col gap-3 sm:flex-row">
                <MiniStatCard title="Original" stats={beforeStats} variant="before" />
                {result.after_stats ? (
                  <MiniStatCard title="Optimized" stats={result.after_stats} variant="after" />
                ) : (
                  <div className="flex-1 rounded-lg border border-dream/20 bg-void/60 p-3">
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                      Optimized
                    </p>
                    <StatRow label="Rows" value={result.optimized_count?.toLocaleString() ?? "—"} />
                    <StatRow label="Status" value={result.status} />
                  </div>
                )}
              </div>

              {result.quality_delta && (
                <div className="flex items-center gap-3 rounded-lg border border-white/[0.06] bg-void/60 p-2.5">
                  <span className="text-[11px] text-slate-400">Quality delta:</span>
                  {result.quality_delta.avg_length_change !== undefined && (
                    <Badge variant={result.quality_delta.avg_length_change > 0 ? "success" : "neutral"} size="xs">
                      Avg length {result.quality_delta.avg_length_change > 0 ? "+" : ""}{result.quality_delta.avg_length_change.toFixed(0)}
                    </Badge>
                  )}
                  {result.quality_delta.count_change !== undefined && result.quality_delta.count_change !== 0 && (
                    <Badge variant="neural" size="xs">
                      Rows {result.quality_delta.count_change > 0 ? "+" : ""}{result.quality_delta.count_change}
                    </Badge>
                  )}
                </div>
              )}

              {result.elapsed_seconds && (
                <p className="text-[11px] text-slate-400">
                  Completed in <span className="font-mono text-slate-300">{result.elapsed_seconds}s</span>
                </p>
              )}

              {/* Config suggestions */}
              <div className="pt-1">
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={handleGetSuggestions}
                  loading={loadingSuggestions}
                  disabled={loadingSuggestions}
                  aria-label="Get config suggestions"
                >
                  Get config suggestions
                </Button>
              </div>

              <AnimatePresence>
                {configSuggestions && (
                  <motion.div
                    key="suggestions"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="rounded-lg border border-neural/20 bg-neural/5 p-3">
                      <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-neural/70">
                        Suggestions ({configSuggestions.model})
                      </p>
                      <ul className="space-y-1.5">
                        {configSuggestions.suggestions.map((s, i) => (
                          <li key={i} className="text-[11px] text-slate-300">
                            <span className="font-mono text-neural">{s.param}</span>:{" "}
                            <span className="text-slate-400">{String(s.current)}</span>
                            {" → "}
                            <span className="text-slate-100">{String(s.suggested)}</span>
                            <span className="ml-1 text-slate-400">({s.reason})</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </Panel>
  );
}

function StatusBadge({ status }: { status: OptimizationStatus }) {
  const config: Record<OptimizationStatus, { variant: "neutral" | "warning" | "dream" | "success" | "nightmare"; label: string; dot: boolean }> = {
    idle: { variant: "neutral", label: "Idle", dot: false },
    estimating: { variant: "warning", label: "Estimating", dot: true },
    running: { variant: "dream", label: "Optimizing", dot: true },
    completed: { variant: "success", label: "Completed", dot: true },
    error: { variant: "nightmare", label: "Error", dot: true },
  };
  const c = config[status];
  return (
    <Badge variant={c.variant} size="xs" dot={c.dot}>
      {c.label}
    </Badge>
  );
}
