"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Settings2, Brain, Moon, Skull, Minimize2, Shield,
  GitCompareArrows, RefreshCw, ArrowRight, Sparkles,
  TrendingUp, Target, Layers,
} from "lucide-react";

interface PipelineStep {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  accent: string;
  accentBg: string;
  description: string;
  detail: string;
  metrics: string[];
}

const STEPS: PipelineStep[] = [
  { id: "configure", label: "Configure", icon: Settings2, accent: "text-neural", accentBg: "bg-neural", description: "Define training parameters & schedule", detail: "Set model type, cycle count, phase epochs, distortion strengths, learning rates, and pruning ratios. The system validates your configuration and generates an optimized phase schedule.", metrics: ["Phase count", "Epoch budget", "Pruning ratio"] },
  { id: "wake", label: "Wake", icon: Brain, accent: "text-success", accentBg: "bg-success", description: "Standard supervised training on clean data", detail: "Train on clean, unaugmented data to establish baseline performance. Anchors weights before distortion phases introduce controlled chaos.", metrics: ["Base accuracy", "Loss convergence", "Gradient norms"] },
  { id: "dream", label: "Dream", icon: Moon, accent: "text-dream", accentBg: "bg-dream", description: "Creative exploration through gentle distortion", detail: "Gently distorted text — shuffled tokens, inserted noise, paraphrased segments. Forces flexible, generalizable representations.", metrics: ["Dream similarity", "Representation drift", "Feature diversity"] },
  { id: "nightmare", label: "Nightmare", icon: Skull, accent: "text-nightmare", accentBg: "bg-nightmare", description: "Adversarial stress-testing pushes limits", detail: "Aggressive distortions attack — extreme noise, semantic scrambling, learned adversarial perturbations. Builds true robustness.", metrics: ["Adversarial accuracy", "Worst-case loss", "Robustness score"] },
  { id: "compress", label: "Compress", icon: Minimize2, accent: "text-warning", accentBg: "bg-warning", description: "Knowledge distillation & pruning", detail: "Compressed via knowledge distillation and structured pruning. Smaller student network preserves adversarial resilience.", metrics: ["Compression ratio", "KL divergence", "Retained accuracy"] },
  { id: "evaluate", label: "Evaluate", icon: Shield, accent: "text-neural", accentBg: "bg-neural", description: "Measure resilience across distortion spectrum", detail: "Tested across 11 distortion strength levels for both dream and nightmare modes. Produces a full resilience profile.", metrics: ["Resilience curve", "Break point", "Mean similarity"] },
  { id: "compare", label: "Compare", icon: GitCompareArrows, accent: "text-dream", accentBg: "bg-dream", description: "Side-by-side improvement measurement", detail: "Current cycle's resilience compared against previous baseline. Delta reveals improvement per distortion type.", metrics: ["Δ resilience", "Dream improvement", "Nightmare improvement"] },
  { id: "iterate", label: "Iterate", icon: RefreshCw, accent: "text-success", accentBg: "bg-success", description: "Auto-repeat with refined parameters", detail: "If resilience target not met, new cycle starts with auto-adjusted parameters. Fully autonomous improvement.", metrics: ["Cycles completed", "Convergence rate", "Final score"] },
];

const CASES = [
  { icon: Target, title: "Adversarial Defense", description: "Deploy models that withstand prompt injection, input manipulation, and data poisoning." },
  { icon: Layers, title: "Model Compression", description: "Ship 40-60% smaller models without accuracy loss — reduce inference cost at scale." },
  { icon: TrendingUp, title: "Continuous Improvement", description: "Autonomous training cycles that improve resilience overnight, zero human intervention." },
  { icon: Sparkles, title: "Domain Hardening", description: "Stress-test models on medical, legal, and financial text where accuracy is critical." },
];

export default function AutoPipeline() {
  const [selected, setSelected] = useState("configure");
  const step = STEPS.find((s) => s.id === selected)!;
  const stepIdx = STEPS.findIndex((s) => s.id === selected);

  return (
    <section id="pipeline" className="relative py-28 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-[10px] font-mono text-neural uppercase tracking-[0.2em] mb-3 block">
            End-to-End Pipeline
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Autonomous{" "}
            <span className="text-gradient-neural">Improvement</span>
          </h2>
          <p className="text-text-dim max-w-2xl mx-auto text-sm leading-relaxed">
            NightmareNet runs a complete self-improvement loop. Each cycle forces the model
            through escalating stress, compresses what it learned, measures improvement,
            and repeats until a resilience target is reached.
          </p>
        </motion.div>

        {/* Pipeline steps */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-6"
        >
          {/* Step bar */}
          <div className="flex flex-wrap items-center justify-center gap-1">
            {STEPS.map((s, i) => (
              <div key={s.id} className="flex items-center">
                <button
                  onClick={() => setSelected(s.id)}
                  className={`relative flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 cursor-pointer ${
                    selected === s.id
                      ? `${s.accent} glass-card !border-white/10`
                      : "text-muted hover:text-text-dim hover:bg-white/[0.02]"
                  }`}
                >
                  <s.icon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{s.label}</span>
                </button>
                {i < STEPS.length - 1 && (
                  <div className={`w-6 h-[1px] mx-0.5 hidden md:block transition-colors duration-300 ${
                    i < stepIdx ? "bg-neural/30" : "bg-white/[0.05]"
                  }`} />
                )}
              </div>
            ))}
          </div>

          {/* Progress bar */}
          <div className="mt-3 flex rounded-full overflow-hidden h-1 bg-white/[0.03] max-w-3xl mx-auto">
            {STEPS.map((s, i) => (
              <button
                key={s.id}
                onClick={() => setSelected(s.id)}
                className={`flex-1 transition-all duration-400 cursor-pointer ${
                  selected === s.id ? `${s.accentBg} opacity-80` : i <= stepIdx ? `${s.accentBg} opacity-25` : "bg-white/[0.04]"
                }`}
                aria-label={s.label}
              />
            ))}
          </div>
        </motion.div>

        {/* Detail panel */}
        <AnimatePresence mode="wait">
          <motion.div
            key={step.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="glass-card p-6 md:p-8 mb-14"
          >
            <div className="flex items-start gap-4">
              <div className={`shrink-0 w-12 h-12 rounded-xl bg-void/80 border border-white/[0.06] flex items-center justify-center ${step.accent}`}>
                <step.icon className="w-6 h-6" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className={`text-lg font-bold ${step.accent} mb-1`}>{step.label}</h3>
                <p className="text-sm text-text-dim mb-1">{step.description}</p>
                <p className="text-xs text-muted leading-relaxed">{step.detail}</p>
                <div className="flex flex-wrap gap-2 mt-3">
                  {step.metrics.map((m) => (
                    <span key={m} className="text-[10px] font-mono text-muted bg-void/60 border border-white/[0.04] rounded-full px-2.5 py-0.5">
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            </div>
            {step.id === "iterate" && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="mt-4 flex items-center gap-2 text-xs text-success/80">
                <RefreshCw className="w-3 h-3 animate-[spin_3s_linear_infinite]" />
                <span>Loops back to <strong>Configure</strong> with auto-adjusted parameters → fully autonomous</span>
              </motion.div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Business cases */}
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.1 }}>
          <h3 className="text-lg font-bold text-center mb-8 text-text-dim">Why This Matters</h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {CASES.map((c, i) => (
              <motion.div
                key={c.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="glass-card p-5 cursor-default"
              >
                <c.icon className="w-5 h-5 text-neural mb-3" />
                <h4 className="text-sm font-semibold text-text mb-1">{c.title}</h4>
                <p className="text-xs text-muted leading-relaxed">{c.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.2 }} className="mt-12 text-center">
          <a href="#training" className="btn-primary">
            <span className="relative z-10">Configure Training</span>
            <ArrowRight className="w-4 h-4 relative z-10" />
          </a>
        </motion.div>
      </div>
    </section>
  );
}
