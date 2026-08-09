"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Panel } from "./Panel";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { useToast } from "@/components/ui/Toast";
import { EmptyState } from "@/components/ui/EmptyState";
import { IconBenchmark, IconCheck, IconRunning } from "./icons";

interface Benchmark {
  id: string;
  name: string;
  description: string;
  duration: string;
  lastScore: number | null;
  trend: number;
  category: "char" | "word" | "semantic" | "attack" | "compression";
  default?: boolean;
}

const BENCHMARKS: Benchmark[] = [
  { id: "char-pgd", name: "Character PGD", description: "Iterative character-level perturbation under projected-gradient-descent", duration: "~2m", lastScore: 78.4, trend: 4.1, category: "char", default: true },
  { id: "word-substitute", name: "Word substitution", description: "Synonym/antonym swap pipeline targeting 5-15% of tokens", duration: "~3m", lastScore: 81.2, trend: 2.6, category: "word" },
  { id: "semantic-paraphrase", name: "Semantic paraphrase", description: "BackTrans + T5 paraphrase pairs from 4 languages", duration: "~6m", lastScore: 83.9, trend: 1.4, category: "semantic" },
  { id: "syntactic-reorder", name: "Syntactic reorder", description: "Tree-rotation reordering w/ POS-preservation", duration: "~4m", lastScore: 76.8, trend: -0.5, category: "word" },
  { id: "textfooler", name: "TextFooler", description: "Greedy WIR + saliency-ranked synonym attack (Jin et al.)", duration: "~5m", lastScore: 71.5, trend: 3.2, category: "attack" },
  { id: "bertattack", name: "BERT-Attack", description: "Masked-LM-driven token replacement under similarity guard", duration: "~5m", lastScore: 68.1, trend: 5.7, category: "attack" },
  { id: "compress-quality", name: "Compression quality", description: "PPL-on-WikiText + downstream task delta vs teacher", duration: "~8m", lastScore: 90.3, trend: 0.4, category: "compression" },
];

const CATEGORY_LABEL: Record<Benchmark["category"], string> = {
  char: "Character",
  word: "Word",
  semantic: "Semantic",
  attack: "Attack",
  compression: "Compression",
};

export function BenchmarkSuite() {
  const [running, setRunning] = useState<Record<string, number>>({});
  const toast = useToast();
  const hasBenchmarks = BENCHMARKS.length > 0;

  const start = (b: Benchmark) => {
    if (running[b.id] !== undefined) return;
    toast.push({ title: `Started ${b.name}`, description: b.description, variant: "info" });
    setRunning((r) => ({ ...r, [b.id]: 0 }));
    const total = 1500 + Math.random() * 1200;
    const stepMs = 60;
    const inc = 100 / (total / stepMs);
    const handle = setInterval(() => {
      setRunning((r) => {
        const cur = (r[b.id] ?? 0) + inc;
        if (cur >= 100) {
          clearInterval(handle);
          const next = { ...r };
          delete next[b.id];
          setTimeout(() => {
            toast.push({ title: `${b.name} finished`, description: "Open the run in Live Metrics", variant: "success" });
          }, 0);
          return next;
        }
        return { ...r, [b.id]: cur };
      });
    }, stepMs);
  };

  const startAll = () => {
    BENCHMARKS.forEach((b, i) => {
      setTimeout(() => start(b), i * 250);
    });
  };

  return (
    <Panel
      title="Benchmark Suite"
      subtitle={`${BENCHMARKS.length} standard benchmarks`}
      icon={<IconBenchmark size={14} />}
      glow="neural"
      toolbar={
        hasBenchmarks ? (
          <Button variant="primary" size="sm" onClick={startAll}>
            <IconRunning size={11} /> Run all
          </Button>
        ) : undefined
      }
    >
      {!hasBenchmarks ? (
        <EmptyState
          icon={<IconBenchmark size={18} />}
          title="No benchmarks available"
          description="Your benchmark suite is empty. Create or configure benchmarks to begin testing."
          primary={{ label: "Configure Benchmarks", onClick: () => {} }}
        />
      ) : (
        <ul className="space-y-2">
          {BENCHMARKS.map((b, idx) => {
            const pct = running[b.id];
            const isRunning = pct !== undefined;
            return (
              <motion.li
                key={b.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: idx * 0.04 }}
                className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-3"
              >
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-white/[0.04] text-slate-300">
                    <IconBenchmark size={13} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-semibold text-slate-100">{b.name}</p>
                      <Badge variant="outline" size="xs">{CATEGORY_LABEL[b.category]}</Badge>
                      {b.default && <Badge variant="neural" size="xs">default</Badge>}
                    </div>
                    <p className="mt-0.5 text-[11px] text-slate-400">{b.description}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
                      <span>~ {b.duration}</span>
                      {b.lastScore !== null && (
                        <span>
                          last:{" "}
                          <span className="font-mono text-slate-300">{b.lastScore.toFixed(1)}</span>
                        </span>
                      )}
                      <span className={b.trend >= 0 ? "text-emerald-300" : "text-nightmare-soft"}>
                        {b.trend >= 0 ? "+" : ""}
                        {b.trend.toFixed(1)} pts
                      </span>
                    </div>
                    {isRunning && (
                      <div className="mt-2">
                        <Progress value={pct} tone="neural" size="xs" indeterminate={pct < 5} />
                      </div>
                    )}
                  </div>
                  <Button
                    variant={isRunning ? "ghost" : "secondary"}
                    size="sm"
                    onClick={() => start(b)}
                    disabled={isRunning}
                    loading={isRunning}
                  >
                    {isRunning ? "Running…" : (
                      <>
                        <IconCheck size={11} /> Run now
                      </>
                    )}
                  </Button>
                </div>
              </motion.li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
