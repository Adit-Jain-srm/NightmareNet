"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Panel } from "./Panel";
import { Badge } from "@/components/ui/Badge";
import { Select } from "@/components/ui/Select";
import { Progress } from "@/components/ui/Progress";
import { EmptyState } from "@/components/ui/EmptyState";
import { useDemoMode } from "@/lib/hooks";
import { IconTrend } from "./icons";

interface ModelStat {
  id: string;
  name: string;
  size: string;
  robustness: number;
  accuracy: number;
  latency: number;
  cost: number;
  flops: string;
  trainedOn: string;
}

const MODELS: Record<string, ModelStat> = {
  base: {
    id: "base",
    name: "DistilBERT · baseline",
    size: "67M",
    robustness: 62,
    accuracy: 88.4,
    latency: 18,
    cost: 1.0,
    flops: "11.2G",
    trainedOn: "wikitext-2-raw",
  },
  hardened: {
    id: "hardened",
    name: "DistilBERT · hardened-v3",
    size: "67M",
    robustness: 86,
    accuracy: 87.2,
    latency: 19,
    cost: 1.4,
    flops: "11.2G",
    trainedOn: "wikitext-2-raw + nightmare-v3",
  },
  compressed: {
    id: "compressed",
    name: "Compressed · 2× student",
    size: "34M",
    robustness: 81,
    accuracy: 85.0,
    latency: 9,
    cost: 0.6,
    flops: "5.6G",
    trainedOn: "distilled from hardened-v3",
  },
};

interface MetricRowProps {
  label: string;
  a: number;
  b: number;
  unit?: string;
  higherIsBetter?: boolean;
  format?: (n: number) => string;
}

function MetricRow({ label, a, b, unit = "", higherIsBetter = true, format }: MetricRowProps) {
  const max = Math.max(a, b, 1);
  const fmt = format ?? ((n: number) => `${n}${unit}`);
  const winsA = higherIsBetter ? a >= b : a <= b;
  return (
    <div className="grid grid-cols-[110px_1fr_60px_1fr_60px] items-center gap-2 py-1.5 text-[11px]">
      <span className="text-slate-400">{label}</span>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/[0.04]">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${(a / max) * 100}%` }}
          transition={{ duration: 0.6 }}
          className={["absolute right-0 top-0 h-full rounded-full", winsA ? "bg-emerald-400" : "bg-slate-500"].join(" ")}
          style={{ transformOrigin: "right" }}
        />
      </div>
      <span className={["text-right font-mono", winsA ? "text-emerald-300" : "text-slate-300"].join(" ")}>
        {fmt(a)}
      </span>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/[0.04]">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${(b / max) * 100}%` }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className={["absolute left-0 top-0 h-full rounded-full", winsA ? "bg-slate-500" : "bg-emerald-400"].join(" ")}
        />
      </div>
      <span className={["text-right font-mono", winsA ? "text-slate-300" : "text-emerald-300"].join(" ")}>
        {fmt(b)}
      </span>
    </div>
  );
}

export function ModelComparison() {
  const [a, setA] = useState("base");
  const [b, setB] = useState("hardened");
  const { isLive } = useDemoMode();
  const hasEnoughModels = Object.keys(MODELS).length >= 2;
  const ma = MODELS[a];
  const mb = MODELS[b];

  return (
    <Panel
      title="Model Comparison"
      subtitle="A/B side-by-side"
      icon={<IconTrend size={14} />}
      glow="neural"
      toolbar={
        hasEnoughModels ? (
          <div className="flex items-center gap-2">
            {!isLive && <Badge variant="warning" size="xs">demo data</Badge>}
            <Badge variant="neural" size="xs">5 metrics</Badge>
          </div>
        ) : undefined
      }
    >
      {!hasEnoughModels ? (
        <EmptyState
          icon={<IconTrend size={18} />}
          title="Insufficient models"
          description="Model comparison requires at least two models. Add another model to compare performance."
          primary={{ label: "Add Model", onClick: () => {} }}
        />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Select
              size="sm"
              label="Model A"
              value={a}
              onChange={setA}
              options={Object.values(MODELS).map((m) => ({ value: m.id, label: m.name }))}
            />
            <Select
              size="sm"
              label="Model B"
              value={b}
              onChange={setB}
              options={Object.values(MODELS).map((m) => ({ value: m.id, label: m.name }))}
            />
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-emerald-500/15 bg-emerald-500/[0.04] p-2.5">
              <p className="text-[10px] uppercase tracking-widest text-emerald-300">Model A</p>
              <p className="truncate text-sm text-slate-100">{ma.name}</p>
              <p className="text-[10px] text-slate-400">{ma.size} · {ma.flops} · trained on {ma.trainedOn}</p>
            </div>
            <div className="rounded-lg border border-slate-500/15 bg-white/[0.02] p-2.5">
              <p className="text-[10px] uppercase tracking-widest text-slate-400">Model B</p>
              <p className="truncate text-sm text-slate-100">{mb.name}</p>
              <p className="text-[10px] text-slate-400">{mb.size} · {mb.flops} · trained on {mb.trainedOn}</p>
            </div>
          </div>

          <div className="mt-4 space-y-1 border-t border-white/[0.04] pt-2">
            <MetricRow label="Robustness" a={ma.robustness} b={mb.robustness} format={(n) => n.toFixed(0)} />
            <MetricRow label="Accuracy" a={ma.accuracy} b={mb.accuracy} format={(n) => `${n.toFixed(1)}%`} />
            <MetricRow label="Latency" a={ma.latency} b={mb.latency} higherIsBetter={false} format={(n) => `${n}ms`} />
            <MetricRow label="Compute Cost" a={ma.cost} b={mb.cost} higherIsBetter={false} format={(n) => `${n.toFixed(2)}×`} />
          </div>

          <div className="mt-3">
            <p className="mb-1 text-[10px] uppercase tracking-widest text-slate-400">Composite score</p>
            <div className="grid grid-cols-2 gap-2">
              <Progress value={(ma.robustness + ma.accuracy) / 2} tone="success" size="sm" showValue label={ma.name.split(" · ")[1] ?? "A"} />
              <Progress value={(mb.robustness + mb.accuracy) / 2} tone="neural" size="sm" showValue label={mb.name.split(" · ")[1] ?? "B"} />
            </div>
          </div>
        </>
      )}
    </Panel>
  );
}
