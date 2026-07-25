"use client";

import { useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings2, Loader2, CheckCircle2, AlertTriangle, Brain, Moon, Skull, Minimize2, Zap } from "lucide-react";
import { previewTrainingConfig, type TrainingConfigResponse } from "@/lib/api";
import { LiveRegion } from "@/components/a11y/LiveRegion";

const DEFAULT_CONFIG = {
  model_name: "gpt2",
  model_type: "causal_lm",
  num_cycles: 3,
  wake_epochs: 3,
  dream_epochs: 2,
  nightmare_epochs: 1,
  learning_rate: 5e-5,
  nightmare_lr_multiplier: 2.0,
  batch_size: 8,
  dream_strength: 0.25,
  nightmare_strength: 0.8,
  pruning_ratio: 0.2,
  kl_weight: 0.1,
  early_stopping: false,
  use_learned_adversarial: false,
};

type ConfigKey = keyof typeof DEFAULT_CONFIG;

const PRESETS: { label: string; desc: string; color: string; values: typeof DEFAULT_CONFIG }[] = [
  {
    label: "Quick Start", desc: "3 cycles, balanced settings", color: "success",
    values: { ...DEFAULT_CONFIG },
  },
  {
    label: "Deep Training", desc: "5 cycles, stronger distortions", color: "dream",
    values: {
      ...DEFAULT_CONFIG, num_cycles: 5, dream_epochs: 3, nightmare_epochs: 2,
      dream_strength: 0.3, nightmare_strength: 0.85, pruning_ratio: 0.25,
      early_stopping: true, use_learned_adversarial: true,
    },
  },
  {
    label: "Production", desc: "10 cycles, maximum robustness", color: "nightmare",
    values: {
      ...DEFAULT_CONFIG, num_cycles: 10, wake_epochs: 5, dream_epochs: 4, nightmare_epochs: 3,
      dream_strength: 0.3, nightmare_strength: 0.9, pruning_ratio: 0.3,
      kl_weight: 0.15, early_stopping: true, use_learned_adversarial: true,
      nightmare_lr_multiplier: 3.0,
    },
  },
];

interface SliderField { key: ConfigKey; label: string; min: number; max: number; step: number; format?: (v: number) => string; accent: string; }

const SLIDERS: SliderField[] = [
  { key: "num_cycles", label: "Training Cycles", min: 1, max: 10, step: 1, accent: "neural" },
  { key: "wake_epochs", label: "Wake Epochs", min: 0, max: 10, step: 1, accent: "success" },
  { key: "dream_epochs", label: "Dream Epochs", min: 0, max: 10, step: 1, accent: "dream" },
  { key: "nightmare_epochs", label: "Nightmare Epochs", min: 0, max: 10, step: 1, accent: "nightmare" },
  { key: "dream_strength", label: "Dream Strength", min: 0, max: 1, step: 0.05, format: (v) => v.toFixed(2), accent: "dream" },
  { key: "nightmare_strength", label: "Nightmare Strength", min: 0, max: 1, step: 0.05, format: (v) => v.toFixed(2), accent: "nightmare" },
  { key: "learning_rate", label: "Learning Rate", min: 1e-6, max: 1e-3, step: 1e-6, format: (v) => v.toExponential(1), accent: "neural" },
  { key: "pruning_ratio", label: "Pruning Ratio", min: 0, max: 0.8, step: 0.05, format: (v) => v.toFixed(2), accent: "warning" },
];

const PHASE_ICONS: Record<string, React.ComponentType<{className?: string}>> = { wake: Brain, dream: Moon, nightmare: Skull, compress: Minimize2 };
const PHASE_COLORS: Record<string, string> = { wake: "bg-success", dream: "bg-dream", nightmare: "bg-nightmare", compress: "bg-warning" };

export default function TrainingLab() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [result, setResult] = useState<TrainingConfigResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = useCallback((key: ConfigKey, value: number | boolean) => {
    setConfig((p) => ({ ...p, [key]: value }));
    setResult(null);
  }, []);

  const preview = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await previewTrainingConfig(config);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
    } finally {
      setLoading(false);
    }
  }, [config]);

  const timeline = useMemo(() => {
    if (!result) return null;
    return result.estimated_phases;
  }, [result]);

  return (
    <section id="training" className="relative py-28 px-6" aria-labelledby="training-heading">
      <LiveRegion message={loading ? "Training configuration preview started" : error ? `Preview failed: ${error}` : result ? "Training configuration preview ready" : ""} assertive={Boolean(error)} />
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-[10px] font-mono text-warning uppercase tracking-[0.2em] mb-3 block">
            Configuration
          </span>
          <h2 id="training-heading" className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Training <span className="text-gradient-neural">Lab</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Configure training hyperparameters and preview the full phase schedule before launch.
          </p>
        </motion.div>

        {/* Presets */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="flex flex-wrap gap-3 mb-8 justify-center"
        >
          {PRESETS.map((p) => (
            <button
              type="button"
              key={p.label}
              onClick={() => { setConfig(p.values); setResult(null); }}
              aria-label={`Apply ${p.label} preset: ${p.desc}`}
              className={`glass-card px-4 py-2.5 flex flex-col items-start cursor-pointer hover:border-${p.color}/20 transition-colors`}
            >
              <span className={`text-xs font-semibold text-${p.color}`}>{p.label}</span>
              <span className="text-[10px] text-slate-400">{p.desc}</span>
            </button>
          ))}
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-6">
          {/* Sliders (3 cols) */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="lg:col-span-3 space-y-4"
          >
            {SLIDERS.map((s) => (
              <div key={s.key} className="glass-card p-4">
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor={`training-${s.key}`} className="text-xs font-mono text-slate-400 uppercase tracking-wider">{s.label}</label>
                  <span className={`text-sm font-mono font-bold text-${s.accent}`}>
                    {s.format ? s.format(config[s.key] as number) : String(config[s.key])}
                  </span>
                </div>
                <input
                  id={`training-${s.key}`}
                  aria-label={s.label}
                  type="range"
                  aria-valuetext={s.format ? s.format(config[s.key] as number) : String(config[s.key])}
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={config[s.key] as number}
                  onChange={(e) => update(s.key, parseFloat(e.target.value))}
                  className={`slider-${s.accent}`}
                />
              </div>
            ))}

            {/* Toggle switches */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { key: "early_stopping" as ConfigKey, label: "Early Stopping" },
                { key: "use_learned_adversarial" as ConfigKey, label: "Learned Adversarial" },
              ].map((t) => (
                <button
                  type="button"
                  key={t.key}
                  onClick={() => update(t.key, !config[t.key])}
                  role="switch"
                  aria-checked={Boolean(config[t.key])}
                  className={`glass-card p-4 flex items-center justify-between cursor-pointer ${
                    config[t.key] ? "!border-neural/20" : ""
                  }`}
                >
                  <span className="text-xs font-mono text-slate-400">{t.label}</span>
                  <span aria-hidden="true" className={`w-8 h-4 rounded-full transition-colors ${config[t.key] ? "bg-neural" : "bg-surface"}`}>
                    <motion.div
                      animate={{ x: config[t.key] ? 16 : 0 }}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                      className="w-4 h-4 rounded-full bg-white shadow-sm"
                    />
                  </span>
                </button>
              ))}
            </div>

            {/* Preview button */}
            <button
              type="button"
              onClick={preview}
              aria-busy={loading}
              disabled={loading}
              className="w-full btn-primary justify-center mt-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Loader2 aria-hidden="true" className="w-4 h-4 animate-spin relative z-10" />
              ) : (
                <>
                  <Settings2 aria-hidden="true" className="w-4 h-4 relative z-10" />
                  <span className="relative z-10">Preview Configuration</span>
                </>
              )}
            </button>
          </motion.div>

          {/* Results (2 cols) */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="lg:col-span-2"
          >
            <AnimatePresence mode="wait">
              {error && (
                <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert" className="glass-card p-4 !border-nightmare/20 mb-4">
                  <p className="text-xs text-nightmare font-medium">{error}</p>
                </motion.div>
              )}

              {result && (
                <motion.div key="result" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                  {/* Summary */}
                  <div className="glass-card p-4 mb-4">
                    <div className="flex items-center gap-2 mb-3">
                      {result.valid ? (
                        <CheckCircle2 aria-hidden="true" className="w-4 h-4 text-success" />
                      ) : (
                        <AlertTriangle aria-hidden="true" className="w-4 h-4 text-warning" />
                      )}
                      <span className="text-xs font-mono text-slate-400">Config Summary</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-center p-2 rounded-lg bg-void/40">
                        <p className="text-lg font-bold font-mono text-neural">{result.total_phases}</p>
                        <p className="text-[9px] text-slate-400 uppercase">Phases</p>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-void/40">
                        <p className="text-lg font-bold font-mono text-dream">{result.total_epochs}</p>
                        <p className="text-[9px] text-slate-400 uppercase">Epochs</p>
                      </div>
                    </div>
                  </div>

                  {/* Timeline */}
                  {timeline && timeline.length > 0 && (
                    <div className="glass-card p-4 mb-4">
                      <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">Phase Schedule</p>
                      <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                        {timeline.map((p, i) => {
                          const Icon = PHASE_ICONS[p.phase] || Zap;
                          const color = PHASE_COLORS[p.phase] || "bg-neural";
                          return (
                            <motion.div
                              key={i}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.03 }}
                              className="flex items-center gap-2 py-1.5 px-2 rounded-lg hover:bg-white/[0.02] transition-colors"
                            >
                              <div className={`w-1.5 h-1.5 rounded-full ${color}`} />
                              <Icon aria-hidden="true" className="w-3 h-3 text-slate-400" />
                              <span className="text-[10px] font-mono text-text-dim flex-1">
                                C{p.cycle} {p.phase}
                              </span>
                              <span className="text-[9px] font-mono text-slate-400">
                                {p.epochs}ep
                              </span>
                            </motion.div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {result.recommendations.length > 0 && (
                    <div className="glass-card p-4 !border-warning/15">
                      <p className="text-xs font-mono text-warning uppercase tracking-wider mb-2">Recommendations</p>
                      <ul className="space-y-1.5">
                        {result.recommendations.map((r, i) => (
                          <li key={i} className="text-[11px] text-slate-400 leading-relaxed flex gap-2">
                            <AlertTriangle aria-hidden="true" className="w-3 h-3 text-warning shrink-0 mt-0.5" />
                            {r}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </motion.div>
              )}

              {!result && !error && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-8 text-center">
                  <Settings2 aria-hidden="true" className="w-8 h-8 text-slate-400 mx-auto mb-3" />
                  <p className="text-sm text-text-dim mb-1">Configure & Preview</p>
                  <p className="text-xs text-slate-400">Adjust sliders and click preview to see the training schedule.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
