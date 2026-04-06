"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Wifi, WifiOff, RefreshCw, Server, Tag, TestTube2, Clock,
  Zap, Moon, Skull, Shield, GitCompareArrows, Settings2, Upload,
  Activity, CheckCircle2, XCircle,
} from "lucide-react";
import { getHealth, type HealthResponse } from "@/lib/api";

type ConnectionState = "idle" | "loading" | "connected" | "error";

interface Endpoint {
  name: string; path: string; icon: React.ComponentType<{ className?: string }>;
  method: "GET" | "POST"; body?: string; isUpload?: boolean; latency: number | null; status: "ok" | "error" | "pending";
}

const DEFS: Omit<Endpoint, "latency" | "status">[] = [
  { name: "Health", path: "/api/v1/health", icon: Server, method: "GET" },
  { name: "Dream", path: "/api/v1/generate/dream", icon: Moon, method: "POST", body: JSON.stringify({ text: "a", strength: 0.1 }) },
  { name: "Nightmare", path: "/api/v1/generate/nightmare", icon: Skull, method: "POST", body: JSON.stringify({ text: "a", strength: 0.1 }) },
  { name: "Robustness", path: "/api/v1/evaluate/robustness", icon: Shield, method: "POST", body: JSON.stringify({ text: "a", strengths: [0.1] }) },
  { name: "Compare", path: "/api/v1/compare", icon: GitCompareArrows, method: "POST", body: JSON.stringify({ text: "a" }) },
  { name: "Train Config", path: "/api/v1/train/config", icon: Settings2, method: "POST", body: JSON.stringify({}) },
  { name: "Upload", path: "/api/v1/upload/text", icon: Upload, method: "POST", body: undefined, isUpload: true },
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Status() {
  const [state, setState] = useState<ConnectionState>("idle");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  const [endpoints, setEndpoints] = useState<Endpoint[]>(
    DEFS.map((d) => ({ ...d, latency: null, status: "pending" as const })),
  );
  const [uptime, setUptime] = useState(0);
  const uptimeRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const check = useCallback(async () => {
    setState("loading");
    setError(null);
    try {
      const res = await getHealth();
      setHealth(res);
      setState("connected");
      setLastCheck(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cannot reach API");
      setState("error");
      setLastCheck(new Date());
    }
  }, []);

  const checkEndpoints = useCallback(async () => {
    const results = await Promise.all(
      DEFS.map(async (def) => {
        const start = performance.now();
        try {
          let fetchOpts: RequestInit;
          if (def.isUpload) {
            const formData = new FormData();
            formData.append("file", new Blob(["test"], { type: "text/plain" }), "probe.txt");
            fetchOpts = { method: "POST", body: formData };
          } else {
            fetchOpts = {
              method: def.method,
              headers: def.body ? { "Content-Type": "application/json" } : undefined,
              body: def.body,
            };
          }
          const res = await fetch(`${API_BASE}${def.path}`, fetchOpts);
          const latency = Math.round(performance.now() - start);
          return { ...def, latency, status: res.status < 500 ? "ok" as const : "error" as const };
        } catch {
          return { ...def, latency: null, status: "error" as const };
        }
      }),
    );
    setEndpoints(results);
  }, []);

  useEffect(() => {
    const init = async () => {
      await check();
      await checkEndpoints();
    };
    init();
    const interval = setInterval(() => { check(); checkEndpoints(); }, 30_000);
    return () => clearInterval(interval);
  }, [check, checkEndpoints]);

  useEffect(() => {
    if (state === "connected") {
      uptimeRef.current = setInterval(() => setUptime((u) => u + 1), 1000);
    } else {
      setTimeout(() => setUptime(0), 0);
      if (uptimeRef.current) clearInterval(uptimeRef.current);
    }
    return () => { if (uptimeRef.current) clearInterval(uptimeRef.current); };
  }, [state]);

  const formatUptime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  const okCount = endpoints.filter((e) => e.status === "ok").length;
  const avgLatency = endpoints.filter((e) => e.latency != null).reduce((sum, e, _, arr) => sum + (e.latency ?? 0) / arr.length, 0);

  const isConnected = state === "connected";
  const isError = state === "error";

  return (
    <section id="status" className="relative py-28 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-[10px] font-mono text-success uppercase tracking-[0.2em] mb-3 block">Live</span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            System <span className="text-gradient-neural">Status</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Real-time endpoint health, latency monitoring, and API diagnostics.
          </p>
        </motion.div>

        {/* Status banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className={`glass-card p-5 mb-5 ${isConnected ? "!border-success/15 box-glow-neural" : isError ? "!border-nightmare/15" : ""}`}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <AnimatePresence mode="wait">
                <motion.div key={state} initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: "spring", stiffness: 400 }}>
                  {isConnected ? (
                    <div className="relative">
                      <Wifi className="w-5 h-5 text-success" />
                      <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-success animate-ping" />
                    </div>
                  ) : isError ? (
                    <WifiOff className="w-5 h-5 text-nightmare" />
                  ) : (
                    <RefreshCw className="w-5 h-5 text-muted animate-spin" />
                  )}
                </motion.div>
              </AnimatePresence>
              <div>
                <p className={`text-sm font-semibold ${isConnected ? "text-success" : isError ? "text-nightmare" : "text-muted"}`}>
                  {isConnected ? "All Systems Operational" : isError ? "API Unreachable" : "Checking..."}
                </p>
                <p className="text-[10px] text-muted font-mono">
                  {lastCheck ? `Last checked: ${lastCheck.toLocaleTimeString()}` : "Waiting..."}
                </p>
              </div>
            </div>
            <button
              onClick={() => { check(); checkEndpoints(); }}
              disabled={state === "loading"}
              className="p-2 rounded-lg hover:bg-white/[0.05] transition-colors disabled:opacity-40 cursor-pointer"
              title="Refresh all"
            >
              <RefreshCw className={`w-4 h-4 text-muted ${state === "loading" ? "animate-spin" : ""}`} />
            </button>
          </div>
        </motion.div>

        {/* Metrics */}
        <AnimatePresence>
          {health && isConnected && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5"
            >
              {[
                { icon: Server, label: "Status", value: health.status.toUpperCase(), accent: "success" },
                { icon: Tag, label: "Version", value: health.version, accent: "neural" },
                { icon: TestTube2, label: "Tests", value: health.tests_passing != null ? String(health.tests_passing) : "N/A", accent: "dream" },
                { icon: Activity, label: "Endpoints", value: `${okCount}/${endpoints.length}`, accent: "success" },
                { icon: Zap, label: "Avg Latency", value: avgLatency > 0 ? `${Math.round(avgLatency)}ms` : "—", accent: "warning" },
                { icon: Clock, label: "Uptime", value: formatUptime(uptime), accent: "neural" },
              ].map((m) => (
                <div key={m.label} className="glass-card p-3 flex items-center gap-2.5">
                  <m.icon className={`w-4 h-4 text-${m.accent} shrink-0`} />
                  <div className="min-w-0">
                    <p className="text-[9px] font-mono text-muted uppercase tracking-wider">{m.label}</p>
                    <p className={`text-sm font-semibold text-${m.accent} truncate`}>{m.value}</p>
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Endpoint matrix */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="glass-card overflow-hidden"
        >
          <div className="px-5 py-3 border-b border-white/[0.04]">
            <p className="text-xs font-mono text-muted uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5" /> Endpoint Health Matrix
            </p>
          </div>
          <div className="divide-y divide-white/[0.03]">
            {endpoints.map((ep, i) => (
              <motion.div
                key={ep.path}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.03 }}
                className="flex items-center gap-3 px-5 py-3 hover:bg-white/[0.02] transition-colors"
              >
                <div className="shrink-0">
                  {ep.status === "ok" ? (
                    <CheckCircle2 className="w-4 h-4 text-success" />
                  ) : ep.status === "error" ? (
                    <XCircle className="w-4 h-4 text-nightmare" />
                  ) : (
                    <RefreshCw className="w-4 h-4 text-muted animate-spin" />
                  )}
                </div>
                <ep.icon className="w-4 h-4 text-muted shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-text-dim truncate">{ep.name}</p>
                  <p className="text-[10px] font-mono text-muted truncate">{ep.method} {ep.path}</p>
                </div>
                <div className="text-right shrink-0">
                  {ep.latency != null ? (
                    <span className={`text-xs font-mono font-bold ${ep.latency < 200 ? "text-success" : ep.latency < 1000 ? "text-warning" : "text-nightmare"}`}>
                      {ep.latency}ms
                    </span>
                  ) : (
                    <span className="text-xs font-mono text-muted">—</span>
                  )}
                </div>
                <div className="w-16 shrink-0 hidden sm:block">
                  <div className="h-1.5 rounded-full bg-void/60 overflow-hidden">
                    {ep.latency != null && (
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min((ep.latency / 2000) * 100, 100)}%` }}
                        transition={{ duration: 0.3 }}
                        className={`h-full rounded-full ${ep.latency < 200 ? "bg-success" : ep.latency < 1000 ? "bg-warning" : "bg-nightmare"}`}
                      />
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Error detail */}
        <AnimatePresence>
          {isError && error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-4 glass-card p-4 !border-nightmare/15">
              <p className="text-nightmare text-xs font-mono mb-1">Error: {error}</p>
              <p className="text-muted text-xs">
                Start the API: <code className="text-neural">uvicorn nightmarenet.api.app:app --reload</code>
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
