"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GitCompareArrows, Loader2, ArrowLeftRight, BarChart3, TrendingDown } from "lucide-react";
import { compareDistortions, type CompareResponse } from "@/lib/api";

export default function ComparisonLab() {
  const [text, setText] = useState("Attention mechanisms enable transformers to capture long-range dependencies.");
  const [baselineStr, setBaselineStr] = useState(0.2);
  const [challengeStr, setChallengeStr] = useState(0.7);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compare = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await compareDistortions({ text, baseline_strength: baselineStr, challenge_strength: challengeStr });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
    } finally {
      setLoading(false);
    }
  }, [text, baselineStr, challengeStr]);

  return (
    <section id="compare" className="relative py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <span className="text-[10px] font-mono text-dream uppercase tracking-[0.2em] mb-3 block">Analysis</span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Comparison <span className="text-gradient-dream">Lab</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Compare distortion effects at two strength levels to measure resilience.
          </p>
        </motion.div>

        {/* Input */}
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <div className="glass-card p-6 mb-6">
            <label className="text-xs font-mono text-muted uppercase tracking-wider block mb-2">Test Text</label>
            <textarea
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null); }}
              rows={2}
              className="w-full bg-void/60 border border-white/[0.06] rounded-xl px-4 py-3 text-sm font-mono text-text placeholder:text-muted/40 focus:outline-none focus:border-neural/30 resize-none transition-colors"
              placeholder="Enter text to compare..."
            />

            <div className="grid sm:grid-cols-2 gap-4 mt-4">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-mono text-success">Baseline Strength</label>
                  <span className="text-xs font-mono font-bold text-success">{baselineStr.toFixed(2)}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={baselineStr} onChange={(e) => { setBaselineStr(parseFloat(e.target.value)); setResult(null); }} className="slider-neural" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-mono text-nightmare">Challenge Strength</label>
                  <span className="text-xs font-mono font-bold text-nightmare">{challengeStr.toFixed(2)}</span>
                </div>
                <input type="range" min="0" max="1" step="0.05" value={challengeStr} onChange={(e) => { setChallengeStr(parseFloat(e.target.value)); setResult(null); }} className="slider-nightmare" />
              </div>
            </div>

            <button onClick={compare} disabled={loading || !text.trim()} className="w-full btn-primary justify-center mt-4 disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? <Loader2 className="w-4 h-4 animate-spin relative z-10" /> : <><GitCompareArrows className="w-4 h-4 relative z-10" /><span className="relative z-10">Compare</span></>}
            </button>
          </div>
        </motion.div>

        <AnimatePresence mode="wait">
          {error && (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="glass-card p-4 !border-nightmare/20">
              <p className="text-xs text-nightmare">{error}</p>
            </motion.div>
          )}

          {result && (
            <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              {/* Resilience score */}
              <div className="glass-card p-6 mb-4 text-center">
                <p className="text-xs font-mono text-muted uppercase tracking-wider mb-2">Resilience Score</p>
                <div className="relative inline-flex items-center justify-center">
                  <svg width="120" height="120" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="6" />
                    <motion.circle
                      cx="60" cy="60" r="52" fill="none"
                      stroke="url(#resilience-grad)"
                      strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray={2 * Math.PI * 52}
                      initial={{ strokeDashoffset: 2 * Math.PI * 52 }}
                      animate={{ strokeDashoffset: 2 * Math.PI * 52 * (1 - result.resilience_score) }}
                      transition={{ duration: 1.2, ease: "easeOut" }}
                      transform="rotate(-90 60 60)"
                    />
                    <defs>
                      <linearGradient id="resilience-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#22d3ee" />
                        <stop offset="100%" stopColor="#818cf8" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <span className="absolute text-2xl font-black font-mono text-gradient-neural">
                    {(result.resilience_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Dream vs Nightmare comparison */}
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                {(["dream", "nightmare"] as const).map((type) => {
                  const data = result[type];
                  const isDream = type === "dream";
                  const color = isDream ? "dream" : "nightmare";
                  return (
                    <div key={type} className={`glass-card p-5 !border-${color}/10`}>
                      <div className="flex items-center gap-2 mb-4">
                        <ArrowLeftRight className={`w-4 h-4 text-${color}`} />
                        <span className={`text-sm font-semibold text-${color} capitalize`}>{type}</span>
                      </div>

                      {/* Similarity bars */}
                      {(["baseline", "challenge"] as const).map((level) => {
                        const d = data[level];
                        const isBaseline = level === "baseline";
                        return (
                          <div key={level} className="mb-3">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-[10px] font-mono text-muted capitalize">{level}</span>
                              <span className={`text-[10px] font-mono font-bold text-${color}`}>
                                {(d.similarity * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="h-1.5 rounded-full bg-void/60 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${d.similarity * 100}%` }}
                                transition={{ duration: 0.8, delay: isBaseline ? 0 : 0.2 }}
                                className={`h-full rounded-full bg-${color} ${!isBaseline ? "opacity-60" : ""}`}
                              />
                            </div>
                          </div>
                        );
                      })}

                      {/* Delta */}
                      <div className="flex items-center gap-1.5 mt-2 text-[10px] font-mono text-muted">
                        <TrendingDown className="w-3 h-3" />
                        <span>Δ = {((data.baseline.similarity - data.challenge.similarity) * 100).toFixed(1)}% drop</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Analysis */}
              <div className="glass-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-neural" />
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">Analysis</span>
                </div>
                <p className="text-xs text-text-dim leading-relaxed">{result.analysis}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
