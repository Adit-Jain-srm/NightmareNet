"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Loader2, BarChart3, TrendingDown } from "lucide-react";
import { evaluateRobustness, type RobustnessResponse } from "@/lib/api";

const DEFAULT_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

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

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + cw, y);
      ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = "rgba(148,163,184,0.5)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i;
      ctx.fillText(`${(100 - i * 25)}%`, pad.left - 6, y + 3);
    }

    // X-axis labels
    ctx.textAlign = "center";
    const strengths = Object.keys(data.scores.dream).map(Number).sort((a, b) => a - b);
    strengths.forEach((s, i) => {
      const x = pad.left + (cw / (strengths.length - 1)) * i;
      ctx.fillText(s.toFixed(1), x, h - 8);
    });

    // Draw curves
    const drawCurve = (scores: Record<string, { similarity: number }>, color: string, glowColor: string) => {
      const points = strengths.map((s, i) => ({
        x: pad.left + (cw / (strengths.length - 1)) * i,
        y: pad.top + ch * (1 - (scores[String(s)]?.similarity ?? 0)),
      }));

      // Area fill
      ctx.beginPath();
      ctx.moveTo(points[0].x, pad.top + ch);
      points.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.lineTo(points[points.length - 1].x, pad.top + ch);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
      grad.addColorStop(0, glowColor);
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.fill();

      // Line
      ctx.beginPath();
      points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Dots
      points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = glowColor;
        ctx.fill();
      });
    };

    drawCurve(data.scores.dream, "#818cf8", "rgba(129,140,248,0.08)");
    drawCurve(data.scores.nightmare, "#f87171", "rgba(248,113,113,0.08)");
  }, [data]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ height: 260 }}
    />
  );
}

export default function RobustnessLab() {
  const [text, setText] = useState("The transformer model processes sequential input through self-attention layers.");
  const [result, setResult] = useState<RobustnessResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const evaluate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await evaluateRobustness({ text, strengths: DEFAULT_STRENGTHS });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
    } finally {
      setLoading(false);
    }
  }, [text]);

  const avgDream = result
    ? Object.values(result.scores.dream).reduce((s, v) => s + v.similarity, 0) / Object.values(result.scores.dream).length
    : 0;
  const avgNightmare = result
    ? Object.values(result.scores.nightmare).reduce((s, v) => s + v.similarity, 0) / Object.values(result.scores.nightmare).length
    : 0;

  return (
    <section id="robustness" className="relative py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <span className="text-[10px] font-mono text-neural uppercase tracking-[0.2em] mb-3 block">Evaluation</span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Robustness <span className="text-gradient-neural">Analysis</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Evaluate how text degrades across the full distortion spectrum.
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
            />
            <div className="mt-3 flex items-center gap-2 text-[10px] text-muted font-mono">
              <Shield className="w-3 h-3" />
              Testing at {DEFAULT_STRENGTHS.length} strength levels: {DEFAULT_STRENGTHS.join(", ")}
            </div>
            <button onClick={evaluate} disabled={loading || !text.trim()} className="w-full btn-primary justify-center mt-4 disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? <Loader2 className="w-4 h-4 animate-spin relative z-10" /> : <><Shield className="w-4 h-4 relative z-10" /><span className="relative z-10">Evaluate Robustness</span></>}
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
              {/* Metric cards */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="glass-card p-4 text-center">
                  <p className="text-xs font-mono text-muted mb-1">Dream Avg</p>
                  <p className="text-xl font-black font-mono text-dream">{(avgDream * 100).toFixed(1)}%</p>
                </div>
                <div className="glass-card p-4 text-center">
                  <p className="text-xs font-mono text-muted mb-1">Nightmare Avg</p>
                  <p className="text-xl font-black font-mono text-nightmare">{(avgNightmare * 100).toFixed(1)}%</p>
                </div>
                <div className="glass-card p-4 text-center">
                  <p className="text-xs font-mono text-muted mb-1">Levels Tested</p>
                  <p className="text-xl font-black font-mono text-neural">{DEFAULT_STRENGTHS.length}</p>
                </div>
              </div>

              {/* Chart */}
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
                <ResilienceChart data={result} />
              </div>

              {/* Summary */}
              <div className="glass-card p-4 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingDown className="w-4 h-4 text-muted" />
                  <span className="text-xs font-mono text-muted uppercase tracking-wider">Summary</span>
                </div>
                <p className="text-xs text-text-dim leading-relaxed">{result.summary}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
