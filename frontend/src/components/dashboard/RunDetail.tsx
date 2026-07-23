"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Panel } from "./Panel";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Progress } from "@/components/ui/Progress";
import { useToast } from "@/components/ui/Toast";
import { cancelPipeline, getPipelineReport, createPipeline } from "@/lib/api";
import {
  IconActivity,
  IconClock,
  IconDownload,
  IconRunning,
  IconSparkle,
} from "./icons";

type PhaseTab = "wake" | "dream" | "nightmare" | "compress";

interface RunConfig {
  sourceType: "urls" | "huggingface" | "text";
  modelName: string;
  modelType: string;
  nightmareStrength: number;
  dreamStrength: number;
  numCycles: number;
  textContent: string;
}

interface MutationPreset {
  id: string;
  label: string;
  detail: string;
  mutate: (cfg: RunConfig) => RunConfig;
  // Inline diff description rendered next to each preset.
  diff: (cfg: RunConfig) => string;
}

const BASE_CONFIG: RunConfig = {
  sourceType: "text",
  modelName: "DistilBERT",
  modelType: "causal_lm",
  nightmareStrength: 0.5,
  dreamStrength: 0.25,
  numCycles: 5,
  textContent: "",
};

const PRESETS: MutationPreset[] = [
  {
    id: "same",
    label: "Same config",
    detail: "Re-runs the exact configuration",
    mutate: (cfg) => ({ ...cfg }),
    diff: () => "no changes",
  },
  {
    id: "up",
    label: "Strength × 1.2",
    detail: "Bumps nightmare strength by 20%",
    mutate: (cfg) => ({
      ...cfg,
      nightmareStrength: Math.min(1, Number((cfg.nightmareStrength * 1.2).toFixed(2))),
    }),
    diff: (cfg) =>
      `nightmare_strength: ${cfg.nightmareStrength.toFixed(2)} → ${Math.min(
        1,
        Number((cfg.nightmareStrength * 1.2).toFixed(2))
      ).toFixed(2)}`,
  },
  {
    id: "down",
    label: "Strength × 0.8",
    detail: "Drops nightmare strength by 20%",
    mutate: (cfg) => ({
      ...cfg,
      nightmareStrength: Number((cfg.nightmareStrength * 0.8).toFixed(2)),
    }),
    diff: (cfg) =>
      `nightmare_strength: ${cfg.nightmareStrength.toFixed(2)} → ${(
        cfg.nightmareStrength * 0.8
      ).toFixed(2)}`,
  },
  {
    id: "gpt2",
    label: "Switch to GPT-2",
    detail: "Same training schedule, different backbone",
    mutate: (cfg) => ({ ...cfg, modelName: "GPT-2" }),
    diff: (cfg) => `model_name: ${cfg.modelName} → GPT-2`,
  },
];

const TABS: { key: PhaseTab; label: string; tone: "neural" | "dream" | "nightmare" | "warning" }[] = [
  { key: "wake", label: "Wake", tone: "neural" },
  { key: "dream", label: "Dream", tone: "dream" },
  { key: "nightmare", label: "Nightmare", tone: "nightmare" },
  { key: "compress", label: "Compress", tone: "warning" },
];

const PHASE_DATA: Record<
  PhaseTab,
  { lossStart: number; lossEnd: number; epochs: number; samples: number; description: string; metrics: { label: string; value: string }[] }
> = {
  wake: {
    lossStart: 2.41,
    lossEnd: 1.18,
    epochs: 1,
    samples: 200,
    description: "Standard supervised learning over the clean corpus to anchor representations.",
    metrics: [
      { label: "Avg Loss", value: "1.42" },
      { label: "Steps", value: "256" },
      { label: "Tokens/s", value: "1.2k" },
      { label: "LR", value: "5e-5" },
    ],
  },
  dream: {
    lossStart: 1.18,
    lossEnd: 0.82,
    epochs: 1,
    samples: 200,
    description: "Mild distortion cycle exposing the model to plausible variations and paraphrases.",
    metrics: [
      { label: "Strength", value: "0.25" },
      { label: "Avg Loss", value: "0.97" },
      { label: "Sim", value: "0.86" },
      { label: "Length Δ", value: "+3%" },
    ],
  },
  nightmare: {
    lossStart: 0.82,
    lossEnd: 1.34,
    epochs: 1,
    samples: 200,
    description: "Adversarial stress: typos, swaps, deletions and learned attacks at high strength.",
    metrics: [
      { label: "Strength", value: "0.80" },
      { label: "Attack", value: "PGD" },
      { label: "Avg Loss", value: "1.12" },
      { label: "Δ Robust", value: "+4.1" },
    ],
  },
  compress: {
    lossStart: 1.34,
    lossEnd: 1.21,
    epochs: 1,
    samples: 200,
    description: "Knowledge distillation + pruning into a leaner, deployment-ready student model.",
    metrics: [
      { label: "Pruning", value: "0.40" },
      { label: "KL Wt", value: "0.5" },
      { label: "Size", value: "−42%" },
      { label: "Δ Quality", value: "−0.8" },
    ],
  },
};

function ReRunMenu({ config }: { config: RunConfig }) {
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const toast = useToast();
  const router = useRouter();
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  // Close on Escape and on outside click.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setOpen(false);
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setFocusIdx((i) => (i + 1) % PRESETS.length);
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setFocusIdx((i) => (i - 1 + PRESETS.length) % PRESETS.length);
      }
    };
    const onClick = (e: MouseEvent) => {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
    };
  }, [open]);

  // Move focus when the active index changes (keyboard nav).
  useEffect(() => {
    if (!open) return;
    buttonRefs.current[focusIdx]?.focus();
  }, [focusIdx, open]);

  const fire = async (preset: MutationPreset) => {
    if (loading) return;
    setLoading(true);
    const next = preset.mutate(config);
    try {
      const res = await createPipeline({
        source_type: next.sourceType,
        model_name: next.modelName,
        model_type: next.modelType,
        num_cycles: next.numCycles,
        nightmare_strength: next.nightmareStrength,
        dream_strength: next.dreamStrength,
        ...(next.textContent ? { text_content: next.textContent } : {}),
      });

      toast.push({
        title: `Re-run queued · ${preset.label}`,
        description:
          preset.id === "same"
            ? "Reusing the original configuration."
            : preset.diff(config),
        variant: "info",
        durationMs: 3200,
      });
      setOpen(false);
      router.push(`/run/${res.run_id}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "An unexpected error occurred";
      toast.push({
        title: "Re-run failed",
        description: message,
        variant: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div ref={containerRef} className="relative">
      <Button
        variant="secondary"
        size="sm"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        loading={loading}
        disabled={loading}
      >
        <IconRunning size={12} /> {loading ? "Re-running..." : "Re-run"}
      </Button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.97 }}
            transition={{ duration: 0.16, ease: "easeOut" }}
            role="menu"
            aria-label="Re-run with mutated config"
            className="absolute right-0 top-full z-30 mt-2 w-80 overflow-hidden rounded-xl border border-white/[0.08] bg-abyss/95 shadow-[0_20px_60px_rgba(0,0,0,0.55)] backdrop-blur-xl"
          >
            <div className="border-b border-white/[0.06] px-3 py-2">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                Re-run with mutated config
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Same dataset · same seed · pick a variation below.
              </p>
            </div>
            <ul className="divide-y divide-white/[0.04] py-1">
              {PRESETS.map((preset, i) => (
                <li key={preset.id}>
                  <button
                    ref={(el) => {
                      buttonRefs.current[i] = el;
                    }}
                    type="button"
                    role="menuitem"
                    onClick={() => fire(preset)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        fire(preset);
                      }
                    }}
                    onFocus={() => setFocusIdx(i)}
                    className={[
                      "group flex w-full cursor-pointer flex-col gap-1 px-3 py-2 text-left transition-colors",
                      "hover:bg-white/[0.04] focus-visible:bg-white/[0.04] focus-visible:outline-none",
                    ].join(" ")}
                  >
                    <span className="flex items-center justify-between">
                      <span className="text-[12.5px] font-medium text-slate-200">
                        {preset.label}
                      </span>
                      <span className="rounded bg-white/[0.05] px-1.5 py-0.5 font-mono text-[10px] text-slate-400 group-hover:bg-neural/[0.12] group-hover:text-neural">
                        ↵
                      </span>
                    </span>
                    <span className="text-[11px] text-slate-400">{preset.detail}</span>
                    <span className="font-mono text-[10.5px] text-slate-400">
                      {preset.diff(config)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="border-t border-white/[0.06] px-3 py-2 text-[10px] text-slate-400">
              <span className="font-mono">esc</span> to close ·{" "}
              <span className="font-mono">↑↓</span> to navigate ·{" "}
              <span className="font-mono">↵</span> to launch
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PhaseTimeline({ active }: { active: PhaseTab }) {
  return (
    <div className="relative flex items-center justify-between">
      <div className="absolute left-0 right-0 top-1/2 h-px -translate-y-1/2 bg-white/[0.06]" />
      {TABS.map((tab) => {
        const isActive = tab.key === active;
        return (
          <div key={tab.key} className="relative flex flex-col items-center gap-1">
            <motion.div
              animate={{ scale: isActive ? 1.05 : 1 }}
              className={[
                "flex h-8 w-8 items-center justify-center rounded-full border-2 bg-abyss",
                isActive
                  ? `border-${tab.tone} text-${tab.tone} shadow-[0_0_14px_currentColor]`
                  : "border-white/10 text-slate-400",
              ].join(" ")}
            >
              <span className="font-mono text-[11px]">{tab.label[0]}</span>
            </motion.div>
            <span className={isActive ? `text-[11px] text-${tab.tone}` : "text-[11px] text-slate-400"}>
              {tab.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function RunDetail() {
  const [active, setActive] = useState<PhaseTab>("dream");
  const [cancelling, setCancelling] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const toast = useToast();
  const tab = TABS.find((t) => t.key === active)!;
  const data = PHASE_DATA[active];

  const runId = "exp_4f0a";

  const handleCancel = useCallback(async () => {
    setCancelling(true);
    try {
      await cancelPipeline(runId);
      toast.push({ title: "Run cancelled", description: `${runId} has been stopped.`, variant: "warning" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Cancel failed";
      toast.push({ title: "Cancel failed", description: msg, variant: "error" });
    } finally {
      setCancelling(false);
    }
  }, [toast]);

  const handleDownloadReport = useCallback(async () => {
    setDownloading(true);
    try {
      const report = await getPipelineReport(runId);
      const blob = new Blob([report.report_md], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${runId}-report.md`;
      a.click();
      URL.revokeObjectURL(url);
      toast.push({ title: "Report downloaded", variant: "success" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Download failed";
      toast.push({ title: "Download failed", description: msg, variant: "error" });
    } finally {
      setDownloading(false);
    }
  }, [toast]);

  return (
    <Panel
      title="Run · wikitext-resilient-v3"
      subtitle="exp_4f0a · DistilBERT · cycle 4 of 5"
      icon={<IconRunning size={14} />}
      glow="neural"
      toolbar={
        <>
          <Badge variant="neural" size="xs" dot>running</Badge>
          <ReRunMenu config={BASE_CONFIG} />
          <Button variant="ghost" size="sm" aria-label="Download report" onClick={handleDownloadReport} loading={downloading} disabled={downloading}>
            <IconDownload size={12} />
          </Button>
          <Button variant="danger" size="sm" onClick={handleCancel} loading={cancelling} disabled={cancelling}>Cancel</Button>
        </>
      }
    >
      <PhaseTimeline active={active} />

      <div className="mt-4 flex flex-wrap gap-1.5 border-b border-white/[0.06] pb-2">
        {TABS.map((t) => {
          const a = t.key === active;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => setActive(t.key)}
              className={[
                "rounded-md px-2.5 py-1 text-[11px] font-medium tracking-wide cursor-pointer transition-colors",
                a
                  ? `bg-${t.tone}/10 text-${t.tone}`
                  : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200",
              ].join(" ")}
              aria-pressed={a}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.18 }}
          className="mt-4 space-y-4"
        >
          <p className="text-xs leading-relaxed text-slate-400">{data.description}</p>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {data.metrics.map((m) => (
              <div key={m.label} className="rounded-md border border-white/[0.06] bg-white/[0.02] p-2.5">
                <p className="text-[10px] uppercase tracking-widest text-slate-400">{m.label}</p>
                <p className="mt-0.5 font-mono text-sm text-slate-100">{m.value}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <IconActivity size={11} />
                <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Loss
                </span>
              </div>
              <p className="font-mono text-base text-slate-100">
                {data.lossStart.toFixed(2)} → {data.lossEnd.toFixed(2)}
              </p>
              <p className="text-[10px] text-slate-400">
                Δ {(data.lossEnd - data.lossStart).toFixed(2)}
              </p>
            </div>
            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <IconClock size={11} />
                <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Epochs · Samples
                </span>
              </div>
              <p className="font-mono text-base text-slate-100">
                {data.epochs} · {data.samples}
              </p>
              <p className="text-[10px] text-slate-400">batch_size = 8</p>
            </div>
            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <div className="mb-1.5 flex items-center gap-1.5">
                <IconSparkle size={11} />
                <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  Phase progress
                </span>
              </div>
              <Progress value={68} tone={tab.tone} size="sm" showValue />
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </Panel>
  );
}
