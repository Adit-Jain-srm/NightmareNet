"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Globe,
  Database,
  FileText,
  Brain,
  Rocket,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronRight,
  Workflow,
  BarChart3,
  Download,
  AlertTriangle,
  Sparkles,
  Moon,
  Skull,
  Minimize2,
} from "lucide-react";
import {
  createPipeline,
  getPipelineStatus,
  cancelPipeline,
  getPipelineReport,
  type PipelineCreateRequest,
  type PipelineStatusResponse,
  type PipelineReportResponse,
} from "@/lib/api";
import { buildRunWsUrl, MAX_RECONNECT_ATTEMPTS } from "@/lib/websocket";
import { useWebSocket } from "@/hooks/useWebSocket";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

/* ── Phase definitions ── */
const PHASES = [
  { key: "wake", label: "Wake", icon: Sparkles, color: "#06b6d4" },
  { key: "dream", label: "Dream", icon: Moon, color: "#a855f7" },
  { key: "nightmare", label: "Nightmare", icon: Skull, color: "#ef4444" },
  { key: "compress", label: "Compress", icon: Minimize2, color: "#f59e0b" },
] as const;

/* ── Source tabs ── */
const SOURCE_TABS = [
  { key: "urls" as const, label: "Web Scrape", icon: Globe },
  { key: "huggingface" as const, label: "HuggingFace", icon: Database },
  { key: "text" as const, label: "Paste Text", icon: FileText },
];

/* ── Model presets ── */
const MODEL_PRESETS = [
  { name: "distilbert-base-uncased", type: "masked_lm", label: "DistilBERT", desc: "Fast embeddings" },
  { name: "bert-base-uncased", type: "masked_lm", label: "BERT", desc: "Fill-mask / embeddings" },
  { name: "roberta-base", type: "masked_lm", label: "RoBERTa", desc: "Robust embeddings" },
  { name: "gpt2", type: "causal_lm", label: "GPT-2", desc: "Text generation" },
  { name: "distilgpt2", type: "causal_lm", label: "DistilGPT-2", desc: "Fast generation" },
];

type WizardStep = "source" | "model" | "config" | "running" | "complete";

function resolvePhaseIndex(status: PipelineStatusResponse | null): number {
  if (!status?.current_phase) return -1;
  const phase = status.current_phase.toLowerCase();
  const direct = PHASES.findIndex((p) => p.key === phase);
  if (direct >= 0) return direct;
  const stageMap: Record<string, number> = {
    ingest: 0,
    prepare: 0,
    evaluate: 3,
    complete: 3,
  };
  return stageMap[phase] ?? -1;
}

export default function PipelineLab() {
  /* ── Wizard state ── */
  const [step, setStep] = useState<WizardStep>("source");
  const [sourceType, setSourceType] = useState<"urls" | "huggingface" | "text">("urls");

  /* ── Source inputs ── */
  const [urls, setUrls] = useState("https://en.wikipedia.org/wiki/Machine_learning\nhttps://en.wikipedia.org/wiki/Neural_network_(machine_learning)");
  const [hfDataset, setHfDataset] = useState("wikitext");
  const [hfSubset, setHfSubset] = useState("wikitext-2-raw-v1");
  const [textContent, setTextContent] = useState("");

  /* ── Model selection ── */
  const [selectedModel, setSelectedModel] = useState(0);

  /* ── Config ── */
  const [numCycles, setNumCycles] = useState(1);
  const [wakeEpochs, setWakeEpochs] = useState(1);
  const [dreamEpochs, setDreamEpochs] = useState(1);
  const [nightmareEpochs, setNightmareEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(5e-5);
  const [maxSamples, setMaxSamples] = useState(200);
  const [dreamStrength, setDreamStrength] = useState(0.25);
  const [nightmareStrength, setNightmareStrength] = useState(0.8);

  /* ── Pipeline state ── */
  const [runId, setRunId] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusResponse | null>(null);
  const [report, setReport] = useState<PipelineReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const applyStatusUpdate = useCallback((id: string, data: PipelineStatusResponse & { event?: string }) => {
    setPipelineStatus(data);
    if (data.event === "complete" || !data.is_running) {
      stopPolling();
      if (data.has_report || data.status === "complete") {
        getPipelineReport(id).then(setReport).catch(() => {});
        setStep("complete");
      } else if (data.status === "failed") {
        setError(data.error || "Pipeline failed.");
      }
    }
  }, [stopPolling]);

  const startPollingFallback = useCallback((id: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const s = await getPipelineStatus(id);
        applyStatusUpdate(id, s);
      } catch {
        /* network hiccup — keep polling */
      }
    }, 3000);
  }, [applyStatusUpdate, stopPolling]);

  const wsEnabled = step === "running" && !!runId;
  const wsUrl = wsEnabled && runId ? buildRunWsUrl(runId) : null;

  const handleWsMessage = useCallback((raw: unknown) => {
    if (!runId || !raw || typeof raw !== "object") return;
    const data = raw as PipelineStatusResponse & { event?: string };
    if (data.error && data.is_running) {
      setError(data.error);
      return;
    }
    applyStatusUpdate(runId, data);
  }, [runId, applyStatusUpdate]);

  const refreshAfterReconnect = useCallback(async () => {
    if (!runId) return;
    stopPolling();
    try {
      const s = await getPipelineStatus(runId);
      applyStatusUpdate(runId, s);
    } catch {
      /* will catch up on next WS message */
    }
  }, [runId, applyStatusUpdate, stopPolling]);

  const {
    status: wsStatus,
    attempt: wsAttempt,
    reconnect: reconnectWs,
    disconnect: disconnectWs,
  } = useWebSocket({
    url: wsUrl,
    enabled: wsEnabled,
    onMessage: handleWsMessage,
    onReconnect: refreshAfterReconnect,
  });

  /* Poll only after WS retries are exhausted — not during the initial connect. */
  useEffect(() => {
    if (wsEnabled && runId && wsStatus === "disconnected" && wsAttempt >= MAX_RECONNECT_ATTEMPTS) {
      startPollingFallback(runId);
    }
    if (wsStatus === "connected") {
      stopPolling();
    }
  }, [wsEnabled, runId, wsStatus, wsAttempt, startPollingFallback, stopPolling]);

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  /* ── Launch pipeline ── */
  const handleLaunch = async () => {
    setIsLaunching(true);
    setError(null);
    const model = MODEL_PRESETS[selectedModel];
    const req: PipelineCreateRequest = {
      source_type: sourceType,
      model_name: model.name,
      model_type: model.type,
      num_cycles: numCycles,
      wake_epochs: wakeEpochs,
      dream_epochs: dreamEpochs,
      nightmare_epochs: nightmareEpochs,
      learning_rate: learningRate,
      batch_size: batchSize,
      max_samples: maxSamples,
      dream_strength: dreamStrength,
      nightmare_strength: nightmareStrength,
    };

    // Retrieve webhooks from localStorage if configured
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("nightmarenet-webhooks");
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) {
            req.webhooks = parsed
              .map((wh) => ({
                url: wh.url || "",
                events: wh.events || [],
              }))
              .filter((wh) => wh.url);
          }
        } catch {
          // ignore parsing error
        }
      }
    }

    if (sourceType === "urls") {
      req.urls = urls.split("\n").map(u => u.trim()).filter(Boolean);
    } else if (sourceType === "huggingface") {
      req.hf_dataset = hfDataset;
      req.hf_subset = hfSubset || undefined;
    } else {
      req.text_content = textContent;
    }

    try {
      const status = await createPipeline(req);
      setRunId(status.run_id);
      setPipelineStatus(status);
      setStep("running");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to launch pipeline");
    } finally {
      setIsLaunching(false);
    }
  };

  /* ── Cancel ── */
  const handleCancel = async () => {
    if (!runId) return;
    try {
      await cancelPipeline(runId);
      stopPolling();
      disconnectWs();
    } catch { /* best effort */ }
  };

  const phaseIdx = resolvePhaseIndex(pipelineStatus);

  return (
    <section id="pipeline" className="relative py-20 sm:py-28 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-purple-500/30 bg-purple-500/10 text-purple-300 text-sm font-medium mb-4">
            <Workflow className="w-4 h-4" />
            End-to-End Pipeline
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3">
            Train a Hardened Model
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Bring your own data → select a model → configure the sleep cycle → get a production-ready model with before/after robustness metrics.
          </p>
        </motion.div>

        {/* Wizard */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card rounded-2xl p-6 sm:p-8 border border-white/[0.06]"
        >
          {/* Step indicator */}
          {step !== "running" && step !== "complete" && (
            <div className="flex items-center justify-center gap-2 mb-8">
              {(["source", "model", "config"] as const).map((s, i) => (
                <div key={s} className="flex items-center gap-2">
                  <button
                    onClick={() => setStep(s)}
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                      step === s
                        ? "bg-purple-500 text-white shadow-lg shadow-purple-500/30"
                        : "bg-white/5 text-gray-500"
                    }`}
                  >
                    {i + 1}
                  </button>
                  <span className={`text-sm hidden sm:inline ${step === s ? "text-white" : "text-gray-500"}`}>
                    {s === "source" ? "Data" : s === "model" ? "Model" : "Config"}
                  </span>
                  {i < 2 && <ChevronRight className="w-4 h-4 text-gray-600" />}
                </div>
              ))}
            </div>
          )}

          <AnimatePresence mode="wait">
            {/* ── Step 1: Data Source ── */}
            {step === "source" && (
              <motion.div key="source" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                <div className="flex gap-2 mb-6">
                  {SOURCE_TABS.map(tab => (
                    <button
                      key={tab.key}
                      onClick={() => setSourceType(tab.key)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer ${
                        sourceType === tab.key
                          ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                          : "bg-white/5 text-gray-400 border border-white/[0.06] hover:border-white/10"
                      }`}
                    >
                      <tab.icon className="w-4 h-4" />
                      {tab.label}
                    </button>
                  ))}
                </div>

                {sourceType === "urls" && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">URLs to scrape (one per line)</label>
                    <textarea
                      value={urls}
                      onChange={e => setUrls(e.target.value)}
                      rows={4}
                      className="w-full bg-black/30 border border-white/[0.08] rounded-lg px-4 py-3 text-sm text-white resize-none focus:outline-none focus:border-purple-500/50"
                      placeholder="https://en.wikipedia.org/wiki/Machine_learning"
                    />
                  </div>
                )}
                {sourceType === "huggingface" && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Dataset Name</label>
                      <input
                        value={hfDataset}
                        onChange={e => setHfDataset(e.target.value)}
                        className="w-full bg-black/30 border border-white/[0.08] rounded-lg px-4 py-3 text-sm text-white focus:outline-none focus:border-purple-500/50"
                        placeholder="wikitext"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Subset (optional)</label>
                      <input
                        value={hfSubset}
                        onChange={e => setHfSubset(e.target.value)}
                        className="w-full bg-black/30 border border-white/[0.08] rounded-lg px-4 py-3 text-sm text-white focus:outline-none focus:border-purple-500/50"
                        placeholder="wikitext-2-raw-v1"
                      />
                    </div>
                  </div>
                )}
                {sourceType === "text" && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Paste your training text</label>
                    <textarea
                      value={textContent}
                      onChange={e => setTextContent(e.target.value)}
                      rows={6}
                      className="w-full bg-black/30 border border-white/[0.08] rounded-lg px-4 py-3 text-sm text-white resize-none focus:outline-none focus:border-purple-500/50"
                      placeholder="Paste articles, documents, or any text corpus..."
                    />
                  </div>
                )}

                <div className="flex justify-end mt-6">
                  <Button onClick={() => setStep("model")} variant="primary">
                    Next: Choose Model <ChevronRight className="w-4 h-4" />
                  </Button>
                </div>
              </motion.div>
            )}

            {/* ── Step 2: Model Selection ── */}
            {step === "model" && (
              <motion.div key="model" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                <h3 className="text-lg font-semibold text-white mb-4">Select Model Architecture</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {MODEL_PRESETS.map((m, i) => (
                    <button
                      key={m.name}
                      onClick={() => setSelectedModel(i)}
                      className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
                        selectedModel === i
                          ? "border-purple-500/50 bg-purple-500/10"
                          : "border-white/[0.06] bg-white/[0.02] hover:border-white/10"
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Brain className="w-4 h-4 text-purple-400" />
                        <span className="text-sm font-semibold text-white">{m.label}</span>
                      </div>
                      <p className="text-xs text-gray-500">{m.desc}</p>
                      <span className="inline-block mt-2 text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-gray-400 font-mono">{m.type}</span>
                    </button>
                  ))}
                </div>
                <div className="flex justify-between mt-6">
                  <Button onClick={() => setStep("source")} variant="ghost" size="sm">
                    Back
                  </Button>
                  <Button onClick={() => setStep("config")} variant="primary">
                    Next: Configure <ChevronRight className="w-4 h-4" />
                  </Button>
                </div>
              </motion.div>
            )}

            {/* ── Step 3: Config ── */}
            {step === "config" && (
              <motion.div key="config" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                <h3 className="text-lg font-semibold text-white mb-4">Training Configuration</h3>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {[
                    { label: "Sleep Cycles", value: numCycles, set: setNumCycles, min: 1, max: 5 },
                    { label: "Wake Epochs", value: wakeEpochs, set: setWakeEpochs, min: 1, max: 5 },
                    { label: "Dream Epochs", value: dreamEpochs, set: setDreamEpochs, min: 1, max: 5 },
                    { label: "Nightmare Epochs", value: nightmareEpochs, set: setNightmareEpochs, min: 1, max: 5 },
                    { label: "Batch Size", value: batchSize, set: setBatchSize, min: 1, max: 32 },
                    { label: "Max Samples", value: maxSamples, set: setMaxSamples, min: 50, max: 5000 },
                  ].map(cfg => (
                    <div key={cfg.label}>
                      <label className="block text-xs text-gray-500 mb-1">{cfg.label}</label>
                      <input
                        type="number"
                        value={cfg.value}
                        onChange={e => cfg.set(Number(e.target.value))}
                        min={cfg.min}
                        max={cfg.max}
                        className="w-full bg-black/30 border border-white/[0.08] rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50"
                      />
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Dream Strength: {dreamStrength}</label>
                    <input type="range" min={0.1} max={0.5} step={0.05} value={dreamStrength} onChange={e => setDreamStrength(Number(e.target.value))} className="w-full accent-purple-500" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Nightmare Strength: {nightmareStrength}</label>
                    <input type="range" min={0.5} max={1.0} step={0.05} value={nightmareStrength} onChange={e => setNightmareStrength(Number(e.target.value))} className="w-full accent-red-500" />
                  </div>
                </div>

                {error && (
                  <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" /> {error}
                  </div>
                )}

                <div className="flex justify-between mt-6">
                  <Button onClick={() => setStep("model")} variant="ghost" size="sm">
                    Back
                  </Button>
                  <Button onClick={handleLaunch} disabled={isLaunching} loading={isLaunching} variant="primary">
                    {!isLaunching && <Rocket className="w-4 h-4" />}
                    Launch Pipeline
                  </Button>
                </div>
              </motion.div>
            )}

            {/* ── Step 4: Running ── */}
            {step === "running" && pipelineStatus && (
              <motion.div key="running" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
                <Card
                  title="Pipeline Running"
                  subtitle={runId ?? undefined}
                  glow="neural"
                  className="border-0 bg-transparent p-0 shadow-none"
                >
                <div className="text-center mb-6 -mt-2">
                  <p className="text-sm text-gray-400 font-mono">{runId}</p>
                  <div className="mt-2 flex justify-center">
                    <ConnectionStatus
                      status={wsStatus}
                      attempt={wsAttempt}
                      onReconnect={wsStatus === "disconnected" ? reconnectWs : undefined}
                    />
                  </div>
                </div>

                {/* Phase progress */}
                <div className="flex items-center justify-between mb-8 px-4">
                  {PHASES.map((phase, i) => {
                    const isActive = phase.key === pipelineStatus.current_phase;
                    const isDone = phaseIdx > i;
                    return (
                      <div key={phase.key} className="flex flex-col items-center gap-2 flex-1">
                        <motion.div
                          animate={isActive ? { scale: [1, 1.15, 1] } : {}}
                          transition={{ repeat: Infinity, duration: 1.5 }}
                          className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all ${
                            isActive
                              ? `border-[${phase.color}] bg-[${phase.color}]/20`
                              : isDone
                                ? "border-green-500 bg-green-500/20"
                                : "border-white/10 bg-white/5"
                          }`}
                          style={isActive ? { borderColor: phase.color, backgroundColor: `${phase.color}20` } : isDone ? {} : {}}
                        >
                          {isDone ? (
                            <CheckCircle2 className="w-5 h-5 text-green-400" />
                          ) : (
                            <phase.icon className="w-5 h-5" style={{ color: isActive ? phase.color : "#6b7280" }} />
                          )}
                        </motion.div>
                        <span className={`text-xs font-medium ${isActive ? "text-white" : "text-gray-500"}`}>{phase.label}</span>
                      </div>
                    );
                  })}
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                  <div className="bg-white/[0.03] rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500 mb-1">Status</div>
                    <div className="text-sm font-semibold text-cyan-400 uppercase">{pipelineStatus.status}</div>
                  </div>
                  <div className="bg-white/[0.03] rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500 mb-1">Progress</div>
                    <div className="text-sm font-mono text-white">{pipelineStatus.progress_pct.toFixed(1)}%</div>
                  </div>
                  <div className="bg-white/[0.03] rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500 mb-1">Phase Loss</div>
                    <div className="text-sm font-mono text-white">{pipelineStatus.phase_loss.toFixed(4)}</div>
                  </div>
                  <div className="bg-white/[0.03] rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500 mb-1">Cycle</div>
                    <div className="text-sm font-mono text-white">{pipelineStatus.current_cycle + 1} / {pipelineStatus.total_cycles}</div>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-white/5 rounded-full h-2 mb-4 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-purple-500 to-cyan-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, Math.max(0, pipelineStatus.progress_pct))}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>

                {pipelineStatus.error && (
                  <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-center gap-2 mb-4">
                    <XCircle className="w-4 h-4" /> {pipelineStatus.error}
                  </div>
                )}

                <div className="flex justify-center">
                  <Button onClick={handleCancel} variant="danger" size="sm">
                    Cancel Pipeline
                  </Button>
                </div>
                </Card>
              </motion.div>
            )}

            {/* ── Step 5: Complete ── */}
            {step === "complete" && (
              <motion.div key="complete" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
                <div className="text-center mb-6">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 300 }}
                    className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-500/20 border-2 border-green-500/30 mb-4"
                  >
                    <CheckCircle2 className="w-8 h-8 text-green-400" />
                  </motion.div>
                  <h3 className="text-xl font-bold text-white mb-1">Pipeline Complete!</h3>
                  <p className="text-sm text-gray-400">Your model has been hardened through the NightmareNet sleep cycle.</p>
                </div>

                {/* Training history */}
                {pipelineStatus?.history && pipelineStatus.history.length > 0 && (
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4" /> Training History
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-gray-500 text-xs">
                            <th className="text-left py-2 px-3">Phase</th>
                            <th className="text-right py-2 px-3">Avg Loss</th>
                            <th className="text-right py-2 px-3">Steps</th>
                          </tr>
                        </thead>
                        <tbody>
                          {pipelineStatus.history.map((h, i) => (
                            <tr key={i} className="border-t border-white/5">
                              <td className="py-2 px-3 text-white font-medium capitalize">{String(h.phase ?? "")}</td>
                              <td className="py-2 px-3 text-right font-mono text-gray-300">{Number(h.avg_loss ?? 0).toFixed(4)}</td>
                              <td className="py-2 px-3 text-right font-mono text-gray-400">{String(h.total_steps ?? 0)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Report */}
                {report?.report_md && (
                  <div className="mb-6 bg-black/30 rounded-lg p-4 border border-white/[0.06] max-h-64 overflow-y-auto">
                    <h4 className="text-sm font-semibold text-gray-300 mb-2">Evaluation Report</h4>
                    <pre className="text-xs text-gray-400 whitespace-pre-wrap font-mono">{report.report_md}</pre>
                  </div>
                )}

                <div className="flex justify-center gap-4">
                  <button
                    onClick={() => {
                      stopPolling();
                      disconnectWs();
                      setStep("source");
                      setRunId(null);
                      setPipelineStatus(null);
                      setReport(null);
                      setError(null);
                    }}
                    className="px-6 py-2.5 rounded-lg border border-white/10 text-white text-sm hover:bg-white/5 transition cursor-pointer"
                  >
                    Run Another Pipeline
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  );
}
