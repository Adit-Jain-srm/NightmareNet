"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Terminal, Copy, Check, ArrowRight, Package, Code2, Zap } from "lucide-react";

const TABS = [
  {
    id: "install",
    label: "Install",
    icon: Package,
    code: `git clone https://github.com/Adit-Jain-srm/NightmareNet.git
cd NightmareNet
pip install -e ".[api]"`,
    lang: "bash",
  },
  {
    id: "python",
    label: "Python",
    icon: Code2,
    code: `from nightmarenet.training import Trainer
from nightmarenet.utils.config import load_config

# Load default config (or pass your YAML)
config = load_config("configs/default.yaml")
config["model"]["name"] = "gpt2"  # your model
config["training"]["num_cycles"] = 3

# Train — wake, dream, nightmare, compress 🔁
trainer = Trainer(config)
trainer.train()`,
    lang: "python",
  },
  {
    id: "cli",
    label: "CLI",
    icon: Terminal,
    code: `# One command to start training
nightmarenet train \\
  --model gpt2 \\
  --dataset wikitext \\
  --cycles 3 \\
  --dream-strength 0.25 \\
  --nightmare-strength 0.8

# Evaluate robustness after training
nightmarenet evaluate \\
  --model ./output/best_model \\
  --tasks sst2 mrpc`,
    lang: "bash",
  },
];

const PHASES = [
  { phase: "Wake", time: "~15 min", desc: "Train on clean data", color: "success" },
  { phase: "Dream", time: "~10 min", desc: "Gentle distortions", color: "dream" },
  { phase: "Nightmare", time: "~8 min", desc: "Adversarial stress test", color: "nightmare" },
  { phase: "Compress", time: "~5 min", desc: "Prune & distill", color: "neural" },
];

export default function QuickStart() {
  const [activeTab, setActiveTab] = useState("install");
  const [copied, setCopied] = useState(false);

  const tab = TABS.find((t) => t.id === activeTab)!;

  const handleCopy = () => {
    navigator.clipboard.writeText(tab.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="quickstart" className="relative py-24 px-6" suppressHydrationWarning>
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-[10px] font-mono text-success uppercase tracking-[0.2em] mb-3 block">
            Get Started
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            3 Lines to <span className="text-gradient-neural">Robustness</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Add NightmareNet to your training pipeline in minutes, not days.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card overflow-hidden"
        >
          {/* Tab header */}
          <div className="flex items-center border-b border-white/[0.04] px-4">
            {TABS.map((t) => {
              const Icon = t.icon;
              const isActive = t.id === activeTab;
              return (
                <button
                  key={t.id}
                  onClick={() => { setActiveTab(t.id); setCopied(false); }}
                  className={`relative flex items-center gap-2 px-4 py-3 text-xs font-mono transition-colors cursor-pointer ${
                    isActive ? "text-neural" : "text-slate-400 hover:text-text-dim"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {t.label}
                  {isActive && (
                    <motion.div
                      layoutId="quickstart-tab"
                      className="absolute bottom-0 left-0 right-0 h-[2px] bg-neural"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                </button>
              );
            })}
            <button
              onClick={handleCopy}
              className="ml-auto flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono text-slate-400 hover:text-neural transition-colors cursor-pointer"
            >
              {copied ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>

          {/* Code block */}
          <div className="p-5 bg-void/40">
            <pre className="text-sm font-mono text-text-dim leading-relaxed overflow-x-auto">
              <code>{tab.code}</code>
            </pre>
          </div>
        </motion.div>

        {/* Phase timeline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="mt-8"
        >
          <div className="flex items-center gap-2 mb-4 justify-center">
            <Zap className="w-3.5 h-3.5 text-neural" />
            <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
              What happens each cycle (~38 min on A100)
            </span>
          </div>

          <div className="grid grid-cols-4 gap-3">
            {PHASES.map((p, i) => (
              <motion.div
                key={p.phase}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 + i * 0.08 }}
                className="glass-card p-4 text-center relative group"
              >
                {i < 3 && (
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 z-10 hidden md:block">
                    <ArrowRight className="w-3 h-3 text-slate-400/30" />
                  </div>
                )}
                <p className={`text-xs font-semibold text-${p.color} mb-1`}>{p.phase}</p>
                <p className="text-[10px] text-text-dim mb-1">{p.desc}</p>
                <p className={`text-[10px] font-mono text-${p.color}/60`}>{p.time}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
