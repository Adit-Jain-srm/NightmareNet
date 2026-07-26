"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Panel } from "./Panel";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import {
  generateDream,
  generateNightmare,
  type DistortionResponse,
} from "@/lib/api";
import { HANDOFF_DEMO_TEXT_KEY } from "@/lib/handoff";
import { IconRunning, IconWand } from "./icons";

const PLACEHOLDER =
  "Neural networks learn by adjusting weights through repeated exposure to training data, gradually improving their predictions.";

interface DistortionState {
  loading: boolean;
  result: DistortionResponse | null;
  error: string | null;
}

const empty = (): DistortionState => ({ loading: false, result: null, error: null });

function DiffText({ original, distorted }: { original: string; distorted: string }) {
  if (!original || !distorted) return <span className="text-slate-400">—</span>;
  const oWords = original.split(/(\s+)/);
  const dWords = distorted.split(/(\s+)/);
  const same = (i: number) => oWords[i] && oWords[i] === dWords[i];
  return (
    <p className="text-[13px] leading-relaxed">
      {dWords.map((w, i) => (
        <span
          key={i}
          className={
            same(i)
              ? "text-slate-300"
              : "rounded bg-amber-500/[0.12] px-0.5 text-amber-200"
          }
        >
          {w}
        </span>
      ))}
    </p>
  );
}

export function DistortionPreview() {
  const [text, setText] = useState(PLACEHOLDER);
  const [strength, setStrength] = useState(0.5);
  const [dream, setDream] = useState<DistortionState>(empty());
  const [nightmare, setNightmare] = useState<DistortionState>(empty());
  const toast = useToast();

  // Pick up text handed off from the marketing demo (GuidedDemo / Playground).
  // Runs once on mount in the browser; the sessionStorage key is cleared so
  // subsequent visits get the default placeholder.
  useEffect(() => {
    if (typeof window === "undefined") return;
    let handoff: string | null = null;
    try {
      handoff = window.sessionStorage.getItem(HANDOFF_DEMO_TEXT_KEY);
    } catch {
      // Privacy / SSR edge cases — silently fall back to placeholder.
      return;
    }
    if (handoff && handoff.trim()) {
      setText(handoff);
      toast.push({
        title: "Continuing from marketing demo",
        description: "We carried your text into the live distortion preview.",
        variant: "info",
      });
    }
    try {
      window.sessionStorage.removeItem(HANDOFF_DEMO_TEXT_KEY);
    } catch {
      // Ignore — best-effort cleanup.
    }
    // We intentionally omit `toast` from deps; pushing a toast is a one-shot
    // side-effect that should fire exactly once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRun = async () => {
    setDream({ loading: true, result: null, error: null });
    setNightmare({ loading: true, result: null, error: null });
    try {
      const [d, n] = await Promise.all([
        generateDream({ text, strength: Math.min(strength, 0.5), seed: 42 }),
        generateNightmare({ text, strength, seed: 42 }),
      ]);
      setDream({ loading: false, result: d, error: null });
      setNightmare({ loading: false, result: n, error: null });
      toast.push({ title: "Distortions generated", variant: "success" });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to call distortion API";
      setDream({ loading: false, result: null, error: msg });
      setNightmare({ loading: false, result: null, error: msg });
      toast.push({ title: "API error", description: msg, variant: "error" });
    }
  };

  return (
    <Panel
      title="Distortion Preview"
      subtitle="Dream vs Nightmare · live API"
      icon={<IconWand size={14} />}
      glow="dream"
      toolbar={
        <>
          <Badge variant="outline" size="xs">/api/v1/generate/*</Badge>
          <Button variant="primary" size="sm" onClick={handleRun} disabled={dream.loading || nightmare.loading} loading={dream.loading || nightmare.loading}>
            <IconRunning size={11} /> Distort
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <div>
          <div className="mb-1.5 flex items-center justify-between text-[10px] uppercase tracking-widest text-slate-400">
            <span>Source text</span>
            <span className="font-mono normal-case text-slate-400">{text.length} chars</span>
          </div>
          <textarea
            id="distortion-source-text"
            aria-label="Source text to distort"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            className="w-full resize-none rounded-lg border border-white/[0.08] bg-black/30 px-3 py-2 text-[13px] text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-neural/50 focus:border-neural/50"
          />
        </div>

        <div className="flex items-center gap-3">
          <label htmlFor="distortion-strength" className="text-[10px] uppercase tracking-widest text-slate-400">Strength</label>
          <input
            id="distortion-strength"
            aria-label="Distortion strength"
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={strength}
            onChange={(e) => setStrength(parseFloat(e.target.value))}
            className="flex-1 accent-dream"
          />
          <span className="font-mono text-xs text-slate-300">{strength.toFixed(2)}</span>
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-lg border border-dream/20 bg-dream/[0.04] p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-widest text-dream-soft">Dream · soft</span>
              {dream.result && <Badge variant="dream" size="xs">seed {dream.result.seed ?? 42}</Badge>}
            </div>
            {dream.loading ? (
              <div className="space-y-2"><Skeleton height={10} /><Skeleton height={10} /><Skeleton height={10} width="60%" /></div>
            ) : dream.error ? (
              <p className="text-[11px] text-nightmare-soft">{dream.error}</p>
            ) : dream.result ? (
              <DiffText original={text} distorted={dream.result.distorted_text} />
            ) : (
              <p className="text-[12px] italic text-slate-400">Click Distort to call /api/v1/generate/dream</p>
            )}
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-lg border border-nightmare/25 bg-nightmare/[0.05] p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-widest text-nightmare-soft">Nightmare · adversarial</span>
              {nightmare.result && <Badge variant="nightmare" size="xs">seed {nightmare.result.seed ?? 42}</Badge>}
            </div>
            {nightmare.loading ? (
              <div className="space-y-2"><Skeleton height={10} /><Skeleton height={10} /><Skeleton height={10} width="80%" /></div>
            ) : nightmare.error ? (
              <p className="text-[11px] text-nightmare-soft">{nightmare.error}</p>
            ) : nightmare.result ? (
              <DiffText original={text} distorted={nightmare.result.distorted_text} />
            ) : (
              <p className="text-[12px] italic text-slate-400">Click Distort to call /api/v1/generate/nightmare</p>
            )}
          </motion.div>
        </div>
      </div>
    </Panel>
  );
}
