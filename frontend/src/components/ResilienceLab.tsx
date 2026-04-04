"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  GitCompareArrows, Shield, Loader2, BarChart3,
  TrendingDown, ArrowLeftRight,
} from "lucide-react";
import {
  compareDistortions, evaluateRobustness,
  type CompareResponse, type RobustnessResponse,
} from "@/lib/api";

type Tab = "compare" | "spectrum";

/* ── Resilience Chart (Canvas) ── */
function ResilienceChart({ data }: { data: RobustnessResponse }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    const pad = { top: 20, right: 20, bottom: 30, left: 40 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + cw, y);
      ctx.stroke();
    }

    // Y labels
    ctx.fillStyle = "rgba(148,163,184,0.5)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i;
      ctx.fillText(`${100 - i * 25}%`, pad.left - 6, y + 3);
    }

    // X labels
    ctx.textAlign = "center";
    const strengths = Object.keys(data.scores.dream).map(Number).sort((a, b) => a - b);
    strengths.forEach((s, i) => {
      const x = pad.left + (cw / (strengths.length - 1)) * i;
      ctx.fillText(s.toFixed(1), x, h - 8);
    });

    const drawCurve = (scores: Record<string, { similarity: number }>, color: string, glow: string) => {
      const pts = strengths.map((s, i) => ({
        x: pad.left + (cw / (strengths.length - 1)) * i,
        y: pad.top + ch * (1 - (scores[String(s)]?.similarity ?? 0)),
      }));

      // Area
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pad.top + ch);
      pts.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.lineTo(pts[pts.length - 1].x, pad.top + ch);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
      grad.addColorStop(0, glow);
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.fill();

      // Line
      ctx.beginPath();
      pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Dots
      pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });
    };

    drawCurve(data.scores.dream, "#818cf8", "rgba(129,140,248,0.08)");
    drawCurve(data.scores.nightmare, "#f87171", "rgba(248,113,113,0.08)");
  }, [data]);

  return <canvas ref={canvasRef} className="w-full" style={{ height: 240 }} />;
}

/* ── Main Component ── */
export default function ResilienceLab() {
  const [tab, setTab] = useState<Tab>("compare");
  const [text, setText] = useState(
    "Attention mechanisms enable transformers to capture long-range dependencies."
  );

  // Compare state
  const [baselineStr, setBaselineStr] = useState(0.2);
  const [challengeStr, setChallengeStr] = useState(0.7);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // Spectrum state
  const [spectrumResult, setSpectrumResult] = useState<RobustnessResponse | null>(null);
  const [spectrumLoading, setSpectrumLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);

  const DEFAULT_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

  const handleCompare = useCallback(async () => {
    setCompareLoading(true);
    setError(null);
    try {
      const res = await compareDistortions({
        text, baseline_strength: baselineStr, challenge_strength: challengeStr,
      });
      setCompareResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
    } finally {
      setCompareLoading(false);
    }
  }, [text, baselineStr, challengeStr]);

  const handleSpectrum = useCallback(async () => {
    setSpectrumLoading(true);
    setError(null);
    try {
      const res = await evaluateRobustness({ text, strengths: DEFAULT_STRENGTHS });
      setSpectrumResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
    } finally {
      setSpectrumLoading(false);
    }
  }, [text, DEFAULT_STRENGTHS]);

  const loading = compareLoading || spectrumLoading;

  const avgDream = spectrumResult
    ? Object.values(spectrumResult.scores.dream).reduce((s, v) => s + v.similarity, 0) /
      Object.values(spectrumResult.scores.dream).length
    : 0;
  const avgNightmare = spectrumResult
    ? Object.values(spectrumResult.scores.nightmare).reduce((s, v) => s + v.similarity, 0) /
      Object.values(spectrumResult.scores.nightmare).length
    : 0;

  return (
    <section id="resilience" className="relative py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-10"
        >
          <span className="text-[10px] font-mono text-dream uppercase tracking-[0.2em] mb-3 block">
            Analysis
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Resilience <span className="text-gradient-dream">Lab</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Measure how text degrades under distortion. A resilience score of 85% means your text retains 85% of its structure under adversarial pressure.
          </p>
        </motion.div>

        {/* Tab toggle */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-xl glass p-1">
            {([
              { t: "compare" as Tab, icon: GitCompareArrows, label: "Quick Compare" },
              { t: "spectrum" as Tab, icon: Shield, label: "Full Spectrum" },
            ]).map(({ t, icon: Icon, label }) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`relative flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                  tab === t ? "text-white" : "text-muted hover:text-text-dim"
                }`}
              >
                {tab === t && (
                  <motion.div
                    layoutId="resilience-tab"
                    className="absolute inset-0 rounded-lg bg-dream/15 border border-dream/20"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className="relative z-10 w-3.5 h-3.5" />
                <span className="relative z-10">{label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Shared input */}
        <div className="glass-card p-5 mb-6">
          <label className="text-xs font-mono text-muted uppercase tracking-wider block mb-2">
            Test Text
          </label>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setCompareResult(null);
              setSpectrumResult(null);
            }}
            rows={2}
            className="w-full bg-void/60 border border-white/[0.06] rounded-xl px-4 py-3 text-sm font-mono text-text placeholder:text-muted/40 focus:outline-none focus:border-neural/30 resize-none transition-colors"
          />

          <AnimatePresence mode="wait">
            {tab === "compare" ? (
              <motion.div
                key="compare-controls"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid sm:grid-cols-2 gap-4 mt-4"
              >
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs font-mono text-success">Baseline</label>
                    <span className="text-xs font-mono font-bold text-success">{baselineStr.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.05" value={baselineStr} onChange={(e) => { setBaselineStr(parseFloat(e.target.value)); setCompareResult(null); }} className="slider-neural" />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs font-mono text-nightmare">Challenge</label>
                    <span className="text-xs font-mono font-bold text-nightmare">{challengeStr.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.05" value={challengeStr} onChange={(e) => { setChallengeStr(parseFloat(e.target.value)); setCompareResult(null); }} className="slider-nightmare" />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="spectrum-controls"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mt-3 flex items-center gap-2 text-[10px] text-muted font-mono"
              >
                <Shield className="w-3 h-3" />
                Testing at {DEFAULT_STRENGTHS.length} strength levels: {DEFAULT_STRENGTHS.join(", ")}
              </motion.div>
            )}
          </AnimatePresence>

          <button
            onClick={tab === "compare" ? handleCompare : handleSpectrum}
            disabled={loading || !text.trim()}
            className="w-full btn-primary justify-center mt-4 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin relative z-10" />
            ) : (
              <span className="relative z-10 flex items-center gap-2">
                {tab === "compare" ? <GitCompareArrows className="w-4 h-4" /> : <Shield className="w-4 h-4" />}
                {tab === "compare" ? "Compare" : "Evaluate"}
              </span>
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-4 mb-4 !border-nightmare/20">
            <p className="text-xs text-nightmare font-mono">{error}</p>
          </motion.div>
        )}

        {/* Compare Results */}
        <AnimatePresence mode="wait">
          {tab === "compare" && compareResult && (
            <motion.div
              key="compare-result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="glass-card p-6 mb-4 text-center">
                <p className="text-xs font-mono text-muted uppercase tracking-wider mb-2">Resilience Score</p>
                <div className="relative inline-flex items-center justify-center">
                  <svg width="120" height="120" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="6" />
                    <motion.circle
                      cx="60" cy="60" r="52" fill="none" stroke="url(#res-grad)" strokeWidth="6" strokeLinecap="round"
                      strokeDasharray={2 * Math.PI * 52}
                      initial={{ strokeDashoffset: 2 * Math.PI * 52 }}
                      animate={{ strokeDashoffset: 2 * Math.PI * 52 * (1 - compareResult.resilience_score) }}
                      transition={{ duration: 1.2, ease: "easeOut" }}
                      transform="rotate(-90 60 60)"
                    />
                    <defs>
                      <linearGradient id="res-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#22d3ee" /><stop offset="100%" stopColor="#818cf8" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <span className="absolute text-2xl font-black font-mono text-gradient-neural">
                    {(compareResult.resilience_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mb-4">
                {(["dream", "nightmare"] as const).map((type) => {
                  const d = compareResult[type];
                  const color = type === "dream" ? "dream" : "nightmare";
                  return (
                    <div key={type} className={`glass-card p-5 !border-${color}/10`}>
                      <div className="flex items-center gap-2 mb-4">
                        <ArrowLeftRight className={`w-4 h-4 text-${color}`} />
                        <span className={`text-sm font-semibold text-${color} capitalize`}>{type}</span>
                      </div>
                      {(["baseline", "challenge"] as const).map((level) => {
                        const item = d[level];
                        return (
                          <div key={level} className="mb-3">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-[10px] font-mono text-muted capitalize">{level}</span>
                              <span className={`text-[10px] font-mono font-bold text-${color}`}>{(item.similarity * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 rounded-full bg-void/60 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${item.similarity * 100}%` }}
                                transition={{ duration: 0.8 }}
                                className={`h-full rounded-full bg-${color}`}
                              />
                            </div>
                          </div>
                        );
                      })}
                      <div className="flex items-center gap-1.5 mt-2 text-[10px] font-mono text-muted">
                        <TrendingDown className="w-3 h-3" />
                        Δ = {((d.baseline.similarity - d.challenge.similarity) * 100).toFixed(1)}% drop
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="glass-card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-neural" />
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">Analysis</span>
                </div>
                <p className="text-xs text-text-dim leading-relaxed">{compareResult.analysis}</p>
              </div>
            </motion.div>
          )}

          {/* Spectrum Results */}
          {tab === "spectrum" && spectrumResult && (
            <motion.div
              key="spectrum-result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="grid grid-cols-3 gap-4 mb-6">
                {[
                  { label: "Dream Avg", value: `${(avgDream * 100).toFixed(1)}%`, color: "dream" },
                  { label: "Nightmare Avg", value: `${(avgNightmare * 100).toFixed(1)}%`, color: "nightmare" },
                  { label: "Levels Tested", value: String(DEFAULT_STRENGTHS.length), color: "neural" },
                ].map((m) => (
                  <div key={m.label} className="glass-card p-4 text-center">
                    <p className="text-xs font-mono text-muted mb-1">{m.label}</p>
                    <p className={`text-xl font-black font-mono text-${m.color}`}>{m.value}</p>
                  </div>
                ))}
              </div>

              <div className="glass-card p-5">
                <div className="flex items-center gap-2 mb-4">
                  <BarChart3 className="w-4 h-4 text-neural" />
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">Resilience Curve</span>
                  <div className="ml-auto flex items-center gap-4">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-dream" />
                      <span className="text-[10px] font-mono text-muted">Dream</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-nightmare" />
                      <span className="text-[10px] font-mono text-muted">Nightmare</span>
                    </div>
                  </div>
                </div>
                <ResilienceChart data={spectrumResult} />
              </div>

              <div className="glass-card p-4 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingDown className="w-4 h-4 text-muted" />
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">Summary</span>
                </div>
                <p className="text-xs text-text-dim leading-relaxed">{spectrumResult.summary}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
