"use client";

import { motion } from "framer-motion";
import { Sun, Moon, Skull, Minimize2, ArrowRight, RefreshCw } from "lucide-react";

const phases = [
  { key: "wake", icon: Sun, label: "Wake", color: "success", description: "Train on clean data to establish baseline competence and anchor model weights." },
  { key: "dream", icon: Moon, label: "Dream", color: "dream", description: "Soft perturbations stretch learned representations while preserving core structure." },
  { key: "nightmare", icon: Skull, label: "Nightmare", color: "nightmare", description: "Aggressive adversarial distortions test the model's invariant knowledge." },
  { key: "compress", icon: Minimize2, label: "Compress", color: "neural", description: "Distill robust representations into a tighter model via knowledge distillation." },
];

const details = [
  { label: "Dream Distortion", value: "Synonym swaps, paraphrase, soft noise", accent: "dream" },
  { label: "Nightmare Distortion", value: "Adversarial attacks, char-swaps, deletions", accent: "nightmare" },
  { label: "Compression", value: "Knowledge distillation, magnitude pruning", accent: "neural" },
];

export default function Architecture() {
  return (
    <section id="architecture" className="relative py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-[10px] font-mono text-neural uppercase tracking-[0.2em] mb-3 block">
            How It Works
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            <span className="text-gradient-neural">4-Phase</span> Training Pipeline
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Each cycle builds robust invariant knowledge that survives even the harshest perturbations.
          </p>
        </motion.div>

        {/* Pipeline cards */}
        <div className="grid md:grid-cols-4 gap-4 relative">
          {/* Desktop connection lines */}
          <div className="hidden md:flex absolute inset-0 items-center pointer-events-none z-0">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="absolute top-1/2 -translate-y-1/2"
                style={{ left: `${(i + 1) * 25}%`, transform: "translate(-50%, -50%)" }}
              >
                <div className="w-8 flex items-center justify-center">
                  <div className="w-full h-[1px] bg-gradient-to-r from-white/[0.03] via-neural/20 to-white/[0.03]" />
                  <ArrowRight className="w-3 h-3 text-neural/30 absolute" />
                </div>
              </div>
            ))}
          </div>

          {phases.map((phase, i) => {
            const Icon = phase.icon;
            return (
              <motion.div
                key={phase.key}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.5, ease: "easeOut" }}
                className={`relative z-10 glass-card p-6 flex flex-col items-center text-center group`}
              >
                <span className="absolute top-3 right-3 text-[10px] font-mono text-muted/40">
                  {String(i + 1).padStart(2, "0")}
                </span>

                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-4 bg-${phase.color}/5 border border-${phase.color}/10 group-hover:bg-${phase.color}/10 transition-colors`}>
                  <Icon className={`w-7 h-7 text-${phase.color}`} strokeWidth={1.5} />
                </div>

                <h3 className={`text-lg font-bold text-${phase.color} mb-2`}>
                  {phase.label}
                </h3>
                <p className="text-xs text-text-dim leading-relaxed">
                  {phase.description}
                </p>
              </motion.div>
            );
          })}
        </div>

        {/* Cycle indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6 }}
          className="mt-8 flex items-center justify-center gap-3 text-xs text-muted"
        >
          <RefreshCw className="w-3.5 h-3.5 text-success animate-[spin_6s_linear_infinite]" />
          <span className="font-mono">
            Repeat each epoch → invariant structure, not surface patterns
          </span>
        </motion.div>

        {/* Technical details */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="mt-14 grid sm:grid-cols-3 gap-4"
        >
          {details.map((item) => (
            <div key={item.label} className={`glass-card p-4`}>
              <p className={`text-[10px] font-mono text-${item.accent} uppercase tracking-wider mb-1`}>{item.label}</p>
              <p className="text-sm text-text-dim">{item.value}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
