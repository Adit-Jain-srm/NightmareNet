"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Panel } from "./Panel";
import { Badge, type BadgeVariant } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Select } from "@/components/ui/Select";
import { Input } from "@/components/ui/Input";
import { useToast } from "@/components/ui/Toast";
import { IconHistory, IconSearch } from "./icons";

type EventKind = "run" | "config" | "deploy" | "alert" | "auth";

interface AuditEvent {
  id: string;
  kind: EventKind;
  actor: string;
  message: string;
  meta?: string;
  ts: string;
}

const VARIANTS: Record<EventKind, BadgeVariant> = {
  run: "neural",
  config: "dream",
  deploy: "success",
  alert: "nightmare",
  auth: "warning",
};

const SAMPLE: AuditEvent[] = [
  { id: "evt-018", kind: "alert",  actor: "system",      message: "Nightmare phase loss spike detected — 1.84 (+0.32 σ)", meta: "exp_4f0a · cycle 4", ts: "10s ago" },
  { id: "evt-017", kind: "run",    actor: "adit",        message: "Started training run wikitext-resilient-v3",            meta: "DistilBERT · 5 cycles",  ts: "12m ago" },
  { id: "evt-016", kind: "config", actor: "adit",        message: "Updated nightmare strength 0.7 → 0.8",                  meta: "config · global",       ts: "14m ago" },
  { id: "evt-015", kind: "deploy", actor: "ci-bot",      message: "Deployed hardened-v3 to staging (model store)",          meta: "tag:hardened-v3.0.4",  ts: "1h ago" },
  { id: "evt-014", kind: "auth",   actor: "adit",        message: "API key rotated · scope=eval-only",                       meta: "rk_2bF…1Jq",           ts: "3h ago" },
  { id: "evt-013", kind: "run",    actor: "scheduler",   message: "Completed run distilgpt2-night-only · robustness 86.1",   meta: "exp_2e80 · 1h 04m",    ts: "5h ago" },
  { id: "evt-012", kind: "config", actor: "adit",        message: "Pruning ratio set to 0.40 for compress phase",            meta: "compress.pruning",     ts: "6h ago" },
  { id: "evt-011", kind: "alert",  actor: "system",      message: "GPU VRAM pressure · 3.4/4.0GB sustained",                meta: "node-01",              ts: "7h ago" },
];

export function AuditTrail() {
  const [filter, setFilter] = useState<EventKind | "all">("all");
  const [query, setQuery] = useState("");
  const toast = useToast();

  const events = useMemo(
    () =>
      SAMPLE.filter((e) => {
        if (filter !== "all" && e.kind !== filter) return false;
        if (!query.trim()) return true;
        const q = query.toLowerCase();
        return (
          e.message.toLowerCase().includes(q) ||
          e.actor.toLowerCase().includes(q) ||
          (e.meta?.toLowerCase().includes(q) ?? false)
        );
      }),
    [filter, query]
  );

  return (
    <Panel
      title="Audit Trail"
      subtitle={`${events.length} events`}
      icon={<IconHistory size={14} />}
      glow="dream"
      toolbar={
        <>
          <Input
            placeholder="Search events…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            leftIcon={<IconSearch size={11} />}
            containerClassName="w-44"
            className="!py-1.5 !text-xs"
          />
          <Select
            size="sm"
            value={filter}
            onChange={(v) => setFilter(v as typeof filter)}
            className="w-28"
            options={[
              { value: "all", label: "All" },
              { value: "run", label: "Runs" },
              { value: "config", label: "Config" },
              { value: "deploy", label: "Deploy" },
              { value: "alert", label: "Alerts" },
              { value: "auth", label: "Auth" },
            ]}
          />
        </>
      }
    >
      {events.length === 0 ? (
        <EmptyState
          icon={<IconHistory size={18} />}
          title="No events recorded"
          description={
            query || filter !== "all"
              ? "Nothing matches the current filter. Try clearing search or expanding the event kind."
              : "Run an experiment or change a setting to start populating the audit trail."
          }
          primary={
            query || filter !== "all"
              ? {
                  label: "Clear filters",
                  onClick: () => {
                    setQuery("");
                    setFilter("all");
                  },
                }
              : undefined
          }
          secondary={{
            label: "Open settings",
            onClick: () => {
              // Surface side-effect via toast since AuditTrail doesn't receive
              // an onSectionChange prop. DashboardRoot listens for the "g s"
              // keyboard shortcut for the real navigation.
              toast.push({
                title: "Audit settings",
                description: "Configure retention and webhooks in Settings.",
                variant: "info",
              });
            },
          }}
        />
      ) : (
      <ol className="relative ml-3 space-y-3 border-l border-white/[0.06] pl-5">
        {events.map((e, idx) => (
          <motion.li
            key={e.id}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2, delay: idx * 0.04 }}
            className="relative"
          >
            <span className="absolute -left-[27px] top-1.5 flex h-3 w-3 items-center justify-center">
              <span
                className={[
                  "h-2 w-2 rounded-full",
                  e.kind === "alert" ? "bg-nightmare" : e.kind === "deploy" ? "bg-emerald-400" : e.kind === "auth" ? "bg-amber-400" : e.kind === "config" ? "bg-dream" : "bg-neural",
                  "shadow-[0_0_8px_currentColor]",
                ].join(" ")}
              />
            </span>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={VARIANTS[e.kind]} size="xs">
                {e.kind}
              </Badge>
              <span className="font-mono text-[10px] text-slate-400">{e.id}</span>
              <span className="text-[10px] text-slate-300">·</span>
              <span className="text-[10px] text-slate-400">{e.ts}</span>
              <span className="ml-auto font-mono text-[10px] text-slate-400">{e.actor}</span>
            </div>
            <p className="mt-1 text-[12px] text-slate-200">{e.message}</p>
            {e.meta && <p className="mt-0.5 font-mono text-[10px] text-slate-400">{e.meta}</p>}
          </motion.li>
        ))}
      </ol>
      )}
    </Panel>
  );
}
