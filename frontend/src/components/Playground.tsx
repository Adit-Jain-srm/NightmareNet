"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Moon, Skull, Loader2, Sparkles, RotateCcw, ArrowRight } from "lucide-react";
import { generateDream, generateNightmare, type DistortionResponse } from "@/lib/api";

type Mode = "dream" | "nightmare";

const SAMPLES = [
  "The model achieved 97.3% accuracy on the benchmark dataset.",
  "Neural activations propagate through deep residual connections.",
  "Gradient descent converges toward a local minimum in the loss landscape.",
  "Attention mechanisms enable transformers to capture long-range dependencies.",
  "The training loss plateaued after 50 epochs of fine-tuning on the domain-specific corpus.",
];

export default function Playground() {
  const [mode, setMode] = useState<Mode>("dream");
  const [text, setText] = useState(SAMPLES[0]);
  const [strength, setStrength] = useState(0.5);
  const [result, setResult] = useState<DistortionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const distort = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fn = mode === "dream" ? generateDream : generateNightmare;
      const res = await fn({ text, strength });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API");
    } finally {
      setLoading(false);
    }
  }, [mode, text, strength]);

  const fillRandom = () => {
    setText(SAMPLES[Math.floor(Math.random() * SAMPLES.length)]);
    setResult(null);
  };

  const isDream = mode === "dream";
  const accent = isDream ? "dream" : "nightmare";

  return (
    <section id="playground" className="relative py-28 px-6">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <span className={`text-[10px] font-mono text-${accent} uppercase tracking-[0.2em] mb-3 block`}>
            Interactive
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Distortion{" "}
            <span className={`text-gradient-${accent}`}>Playground</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Feed text through the distortion pipeline and watch it transform in real-time.
          </p>
        </motion.div>

        {/* Mode toggle */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="flex justify-center mb-8"
        >
          <div className="inline-flex rounded-xl glass p-1">
            {([
              { m: "dream" as Mode, icon: Moon, label: "Dream" },
              { m: "nightmare" as Mode, icon: Skull, label: "Nightmare" },
            ]).map(({ m, icon: Icon, label }) => (
              <button
                key={m}
                onClick={() => { setMode(m); setResult(null); }}
                className={`relative flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium transition-colors duration-200 cursor-pointer ${
                  mode === m ? "text-white" : "text-muted hover:text-text-dim"
                }`}
              >
                {mode === m && (
                  <motion.div
                    layoutId="playground-mode"
                    className={`absolute inset-0 rounded-lg ${
                      m === "dream"
                        ? "bg-dream/15 border border-dream/20"
                        : "bg-nightmare/15 border border-nightmare/20"
                    }`}
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className="relative z-10 w-4 h-4" />
                <span className="relative z-10">{label}</span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className={`glass-card p-6 ${isDream ? "!border-dream/10" : "!border-nightmare/10"}`}
        >
          {/* Text input */}
          <div className="mb-5">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-mono text-muted uppercase tracking-wider">Input Text</label>
              <button onClick={fillRandom} className="flex items-center gap-1 text-xs text-muted hover:text-neural transition-colors cursor-pointer">
                <RotateCcw className="w-3 h-3" /> Random
              </button>
            </div>
            <textarea
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null); }}
              rows={3}
              className="w-full bg-void/60 border border-white/[0.06] rounded-xl px-4 py-3 text-sm font-mono text-text placeholder:text-muted/40 focus:outline-none focus:border-neural/30 focus:ring-1 focus:ring-neural/15 resize-none transition-colors"
              placeholder="Enter text to distort..."
            />
          </div>

          {/* Strength slider */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-mono text-muted uppercase tracking-wider">Strength</label>
              <span className={`text-sm font-mono font-bold text-${accent}`}>{strength.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={strength}
              onChange={(e) => setStrength(parseFloat(e.target.value))}
              className={`slider-${accent}`}
            />
            <div className="flex justify-between text-[10px] text-muted/50 mt-1 font-mono">
              <span>Gentle</span>
              <span>Aggressive</span>
            </div>
            <p className="text-[10px] text-muted mt-2 leading-relaxed">
              {strength <= 0.3
                ? "💤 Typos, synonym swaps — gentle generalization pressure"
                : strength <= 0.6
                  ? "🌀 Token drops, paraphrasing — moderate restructuring"
                  : "⚡ Adversarial attacks, semantic scrambling — maximum stress"}
            </p>
          </div>

          {/* Generate */}
          <button
            onClick={distort}
            disabled={loading || !text.trim()}
            className={`w-full flex items-center justify-center gap-2 py-3.5 rounded-xl font-semibold text-sm transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer ${
              isDream
                ? "bg-gradient-to-r from-dream to-dream-glow text-white hover:shadow-[0_0_30px_rgba(129,140,248,0.25)]"
                : "bg-gradient-to-r from-nightmare to-nightmare-glow text-white hover:shadow-[0_0_30px_rgba(248,113,113,0.25)]"
            }`}
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <Zap className="w-4 h-4" />
                {isDream ? "Generate Dream" : "Generate Nightmare"}
              </>
            )}
          </button>
        </motion.div>

        {/* Result */}
        <AnimatePresence mode="wait">
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mt-6 glass-card p-4 !border-nightmare/20"
            >
              <p className="text-nightmare font-medium text-sm mb-1">Connection Error</p>
              <p className="text-text-dim text-xs font-mono">{error}</p>
              <p className="text-muted text-xs mt-2">
                Make sure the API is running at <code className="text-neural">localhost:8000</code>
              </p>
            </motion.div>
          )}

          {result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4 }}
              className={`mt-6 glass-card p-6 ${isDream ? "!border-dream/15" : "!border-nightmare/15"}`}
            >
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className={`w-4 h-4 text-${accent}`} />
                <span className="text-xs font-mono text-muted uppercase tracking-wider">Distorted Output</span>
                <span className={`ml-auto text-xs font-mono text-${accent}`}>
                  {result.distortion_type} @ {result.strength.toFixed(2)}
                </span>
              </div>

              {/* Side by side */}
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <span className="text-[10px] text-muted/70 font-mono block mb-1.5">Original</span>
                  <div className="terminal">
                    <div className="p-3">
                      <p className="text-xs text-text-dim leading-relaxed">{result.original_text}</p>
                    </div>
                  </div>
                </div>
                <div>
                  <span className={`text-[10px] text-${accent}/70 font-mono block mb-1.5`}>
                    {isDream ? "Dream State" : "Nightmare State"}
                  </span>
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.6 }}
                    className={`terminal !border-${accent}/15`}
                  >
                    <div className="p-3">
                      <p className={`text-xs leading-relaxed text-${accent}`}>{result.distorted_text}</p>
                    </div>
                  </motion.div>
                </div>
              </div>

              {/* Use in comparison CTA */}
              <div className="mt-4 flex justify-end">
                <a href="#resilience" className="flex items-center gap-1.5 text-xs text-muted hover:text-neural transition-colors cursor-pointer">
                  Use in Resilience Lab <ArrowRight className="w-3 h-3" />
                </a>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
