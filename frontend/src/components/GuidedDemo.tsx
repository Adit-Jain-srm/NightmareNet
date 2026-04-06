"use client";

import { useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Skull, ArrowRight, Loader2, RotateCcw, Lightbulb } from "lucide-react";
import { runDemo, type DemoResponse } from "@/lib/api";

const SAMPLES = [
  "The transformer model processes sequential input through self-attention layers, allowing it to capture long-range dependencies without recurrence.",
  "Fine-tuning a pre-trained language model on domain-specific data significantly improves downstream task accuracy while reducing the need for large labeled datasets.",
  "Gradient descent iteratively adjusts model weights to minimize the loss function, converging toward optimal parameters for the given training distribution.",
];

type Step = 0 | 1 | 2;

/** Character-level diff renderer */
function CharDiff({
  original,
  distorted,
  color,
}: {
  original: string;
  distorted: string;
  color: "dream" | "nightmare";
}) {
  const chars = useMemo(() => {
    const result: { char: string; status: "same" | "changed" | "added" }[] = [];
    const maxLen = Math.max(original.length, distorted.length);
    for (let i = 0; i < maxLen; i++) {
      if (i >= distorted.length) break;
      if (i >= original.length) {
        result.push({ char: distorted[i], status: "added" });
      } else if (distorted[i] !== original[i]) {
        result.push({ char: distorted[i], status: "changed" });
      } else {
        result.push({ char: distorted[i], status: "same" });
      }
    }
    return result;
  }, [original, distorted]);

  const bg =
    color === "dream"
      ? { changed: "bg-dream/20", added: "bg-dream/30" }
      : { changed: "bg-nightmare/20", added: "bg-nightmare/30" };

  return (
    <span className="font-mono text-sm leading-relaxed whitespace-pre-wrap">
      {chars.map((c, i) => (
        <span
          key={i}
          className={
            c.status === "same"
              ? "text-text-dim"
              : c.status === "changed"
                ? `${bg.changed} text-${color} rounded-sm px-[1px]`
                : `${bg.added} text-${color}-soft rounded-sm px-[1px]`
          }
        >
          {c.char}
        </span>
      ))}
    </span>
  );
}

export default function GuidedDemo() {
  const [step, setStep] = useState<Step>(0);
  const [text, setText] = useState(SAMPLES[0]);
  const [result, setResult] = useState<DemoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showNightmare, setShowNightmare] = useState(false);

  const handleDream = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runDemo({ text });
      setResult(res);
      setStep(1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach API");
    } finally {
      setLoading(false);
    }
  }, [text]);

  const handleNightmare = useCallback(() => {
    setShowNightmare(true);
    setStep(2);
  }, []);

  const reset = useCallback(() => {
    setStep(0);
    setResult(null);
    setShowNightmare(false);
    setError(null);
    setText(SAMPLES[Math.floor(Math.random() * SAMPLES.length)]);
  }, []);

  const steps = [
    { label: "Your Text", active: step >= 0 },
    { label: "Dream", active: step >= 1 },
    { label: "Nightmare", active: step >= 2 },
  ];

  return (
    <section id="demo" className="relative py-24 px-6">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-10"
        >
          <span className="text-[10px] font-mono text-neural uppercase tracking-[0.2em] mb-3 block">
            Try It Now
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            See It In <span className="text-gradient-neural">Action</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Watch your text transform through dream and nightmare distortions in 3 steps.
          </p>
        </motion.div>

        {/* Progress dots */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {steps.map((s, i) => (
            <div key={s.label} className="flex items-center gap-2">
              <div className="flex items-center gap-1.5">
                <div
                  className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                    s.active
                      ? i === 0
                        ? "bg-neural shadow-[0_0_8px] shadow-neural/40"
                        : i === 1
                          ? "bg-dream shadow-[0_0_8px] shadow-dream/40"
                          : "bg-nightmare shadow-[0_0_8px] shadow-nightmare/40"
                      : "bg-surface"
                  }`}
                />
                <span
                  className={`text-[10px] font-mono transition-colors ${
                    s.active ? "text-text" : "text-muted/40"
                  }`}
                >
                  {s.label}
                </span>
              </div>
              {i < 2 && <div className="w-8 h-[1px] bg-surface" />}
            </div>
          ))}
        </div>

        {/* Step content */}
        <AnimatePresence mode="wait">
          {/* Step 0: Input */}
          {step === 0 && (
            <motion.div
              key="step0"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="glass-card p-6"
            >
              <label className="text-xs font-mono text-muted uppercase tracking-wider block mb-2">
                Step 1 — Paste or edit text to distort
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="w-full bg-void/60 border border-white/[0.06] rounded-xl px-4 py-3 text-sm font-mono text-text placeholder:text-muted/40 focus:outline-none focus:border-neural/30 focus:ring-1 focus:ring-neural/15 resize-none transition-colors"
                placeholder="Enter text here..."
              />
              <div className="flex items-center justify-between mt-4">
                <button
                  onClick={() => setText(SAMPLES[Math.floor(Math.random() * SAMPLES.length)])}
                  className="flex items-center gap-1.5 text-xs text-muted hover:text-neural transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3 h-3" /> Try another example
                </button>
                <button
                  onClick={handleDream}
                  disabled={loading || !text.trim()}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-dream to-dream-glow text-white font-semibold text-sm hover:shadow-[0_0_25px_rgba(129,140,248,0.3)] disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer"
                >
                  {loading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Dream It
                      <ArrowRight className="w-3.5 h-3.5" />
                    </>
                  )}
                </button>
              </div>
              {error && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-3 text-xs text-nightmare font-mono"
                >
                  {error} — make sure the API is running at localhost:8000
                </motion.p>
              )}
            </motion.div>
          )}

          {/* Step 1: Dream Result */}
          {step === 1 && result && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="space-y-4"
            >
              <div className="glass-card p-6 !border-dream/15">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="w-4 h-4 text-dream" />
                  <span className="text-sm font-semibold text-dream">
                    Step 2 — Dream Distortion
                  </span>
                  <span className="ml-auto text-xs font-mono text-dream/70">
                    {(result.dream.similarity * 100).toFixed(0)}% preserved
                  </span>
                </div>

                <div className="bg-void/40 rounded-xl p-4 border border-dream/[0.06]">
                  <CharDiff
                    original={result.original_text}
                    distorted={result.dream.distorted_text}
                    color="dream"
                  />
                </div>

                <p className="text-xs text-text-dim mt-3 leading-relaxed">
                  <span className="text-dream font-medium">Gentle:</span> Synonym swaps, typos, and paraphrasing.
                  Your model learns to generalize beyond surface patterns.
                </p>
              </div>

              <div className="flex justify-end">
                <button
                  onClick={handleNightmare}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-nightmare to-nightmare-glow text-white font-semibold text-sm hover:shadow-[0_0_25px_rgba(248,113,113,0.3)] transition-all cursor-pointer"
                >
                  <Skull className="w-4 h-4" />
                  Now Nightmare It
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </motion.div>
          )}

          {/* Step 2: Nightmare Result + Insight */}
          {step === 2 && result && showNightmare && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="space-y-4"
            >
              {/* Dream (compressed) */}
              <div className="glass-card p-4 !border-dream/10 opacity-70">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="w-3.5 h-3.5 text-dream" />
                  <span className="text-xs font-semibold text-dream">Dream</span>
                  <span className="ml-auto text-[10px] font-mono text-dream/60">
                    {(result.dream.similarity * 100).toFixed(0)}% preserved
                  </span>
                </div>
                <p className="text-xs font-mono text-text-dim line-clamp-2">
                  {result.dream.distorted_text}
                </p>
              </div>

              {/* Nightmare (expanded) */}
              <motion.div
                initial={{ scale: 0.98, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="glass-card p-6 !border-nightmare/20"
              >
                <div className="flex items-center gap-2 mb-4">
                  <Skull className="w-4 h-4 text-nightmare" />
                  <span className="text-sm font-semibold text-nightmare">
                    Step 3 — Nightmare Distortion
                  </span>
                  <span className="ml-auto text-xs font-mono text-nightmare/70">
                    {(result.nightmare.similarity * 100).toFixed(0)}% preserved
                  </span>
                </div>

                <div className="bg-void/40 rounded-xl p-4 border border-nightmare/[0.06]">
                  <CharDiff
                    original={result.original_text}
                    distorted={result.nightmare.distorted_text}
                    color="nightmare"
                  />
                </div>

                <p className="text-xs text-text-dim mt-3 leading-relaxed">
                  <span className="text-nightmare font-medium">Aggressive:</span> Adversarial attacks,
                  contradictions, semantic scrambling. If your model can survive this, it&apos;s robust.
                </p>
              </motion.div>

              {/* Insight card */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.4 }}
                className="glass-card p-5 !border-neural/15"
              >
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-neural/5 border border-neural/10 flex items-center justify-center shrink-0 mt-0.5">
                    <Lightbulb className="w-5 h-5 text-neural" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-neural mb-1">What this means</p>
                    <p className="text-xs text-text-dim leading-relaxed">{result.insight}</p>
                  </div>
                </div>

                <div className="mt-4 flex flex-col sm:flex-row gap-3">
                  <a
                    href="#quickstart"
                    className="btn-primary justify-center flex-1"
                  >
                    <span className="relative z-10 flex items-center gap-2">
                      <ArrowRight className="w-4 h-4" />
                      Add to Your Training
                    </span>
                  </a>
                  <button
                    onClick={reset}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-white/[0.06] text-muted text-sm hover:text-neural hover:border-neural/20 transition-colors cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Try Different Text
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
