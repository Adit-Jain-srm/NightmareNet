"use client";

import { useState, useEffect } from "react";
import { Panel } from "./Panel";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { useToast } from "@/components/ui/Toast";
import { useSounds } from "@/lib/sounds";
import { IconKey, IconSettings, IconShield, IconWand, IconBell, IconHome } from "./icons";
import { testWebhook, getWebhooks, saveWebhooks } from "@/lib/api";

type Tab = "keys" | "model" | "distortion" | "integrations" | "notifications" | "preferences";

const TABS: { key: Tab; label: string; icon: React.ReactNode }[] = [
  { key: "keys", label: "API Keys", icon: <IconKey size={12} /> },
  { key: "model", label: "Model", icon: <IconShield size={12} /> },
  { key: "distortion", label: "Distortion", icon: <IconWand size={12} /> },
  { key: "integrations", label: "Integrations", icon: <IconSettings size={12} /> },
  { key: "notifications", label: "Notifications", icon: <IconBell size={12} /> },
  { key: "preferences", label: "Preferences", icon: <IconHome size={12} /> },
];

interface WebhookConfig {
  id: string;
  url: string;
  events: string[];
}

interface ApiKey {
  id: string;
  name: string;
  scope: string;
  created: string;
  lastUsed: string;
  preview: string;
}

const KEYS: ApiKey[] = [
  { id: "rk_2bF…1Jq", name: "Production · CI", scope: "eval-only", created: "2026-04-12", lastUsed: "12m ago", preview: "rk_2bF***1Jq" },
  { id: "rk_84a…M0p", name: "Local dev", scope: "full", created: "2026-04-02", lastUsed: "3h ago", preview: "rk_84a***M0p" },
  { id: "rk_19e…Tx2", name: "Notebook · GPU", scope: "training", created: "2026-03-28", lastUsed: "1d ago", preview: "rk_19e***Tx2" },
];

export function SettingsPanel() {
  const [tab, setTab] = useState<Tab>("keys");
  const [model, setModel] = useState("distilbert-base-uncased");
  const [batch, setBatch] = useState("8");
  const [lr, setLr] = useState("5e-5");
  const [dreamStrength, setDreamStrength] = useState("0.25");
  const [nightmareStrength, setNightmareStrength] = useState("0.80");
  const [seed, setSeed] = useState("42");
  const [azureEndpoint, setAzureEndpoint] = useState("");
  const [azureKey, setAzureKey] = useState("");
  const [adaptionLabsKey, setAdaptionLabsKey] = useState("");
  const [huggingfaceToken, setHuggingfaceToken] = useState("");
  const [wandbKey, setWandbKey] = useState("");
  const toast = useToast();
  const sounds = useSounds();

  const [webhooks, setWebhooks] = useState<WebhookConfig[]>([]);

  useEffect(() => {
    getWebhooks()
      .then((res) => {
        setWebhooks(res.webhooks.map((w, i) => ({ ...w, id: `wh-${i}-${Date.now()}` })));
      })
      .catch(() => {});
  }, []);

  return (
    <Panel
      title="Settings"
      subtitle="Workspace · API · Defaults"
      icon={<IconSettings size={14} />}
      glow="dream"
      toolbar={<Badge variant="outline" size="xs">workspace · adit</Badge>}
    >
      {/* Sound preference toggle */}
      <div className="mb-4 flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" className="text-slate-400">
            <path d="M2.5 5.5h2L7 3v8L4.5 8.5h-2a.5.5 0 01-.5-.5V6a.5.5 0 01.5-.5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
            {sounds.enabled && (
              <>
                <path d="M9 5.2a2.5 2.5 0 010 3.6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                <path d="M10.5 3.8a4.5 4.5 0 010 6.4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
              </>
            )}
            {!sounds.enabled && (
              <path d="M9.5 5L12 8M12 5L9.5 8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            )}
          </svg>
          <span className="text-[11px] text-slate-300">UI Sounds</span>
        </div>
        <button
          type="button"
          role="switch"
          aria-label="Toggle UI sounds"
          aria-checked={sounds.enabled}
          onClick={sounds.toggle}
          className={[
            "relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition-colors",
            sounds.enabled ? "bg-neural/40" : "bg-white/[0.08]",
          ].join(" ")}
        >
          <span
            className={[
              "inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform",
              sounds.enabled ? "translate-x-[18px]" : "translate-x-[3px]",
            ].join(" ")}
          />
        </button>
      </div>

      <div className="mb-4 flex flex-wrap gap-1 border-b border-white/[0.06] pb-2">
        {TABS.map((t) => {
          const active = t.key === tab;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              className={[
                "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] cursor-pointer transition-colors",
                active ? "bg-neural/[0.08] text-neural" : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-300",
              ].join(" ")}
              aria-pressed={active}
            >
              {t.icon}
              {t.label}
            </button>
          );
        })}
      </div>

      {tab === "keys" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-[11px] text-slate-400">Manage workspace API keys. Rotate often. Scope down to eval-only for CI.</p>
            <Button variant="primary" size="sm" onClick={() => toast.push({ title: "Generated new key", description: "Visible once — copy now.", variant: "success" })}>
              Generate key
            </Button>
          </div>
          <ul className="space-y-2">
            {KEYS.map((k) => (
              <li key={k.id} className="flex items-center gap-3 rounded-lg border border-white/[0.05] bg-white/[0.02] p-2.5">
                <span className="flex h-7 w-7 items-center justify-center rounded-md bg-white/[0.04] text-slate-400">
                  <IconKey size={12} />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm text-slate-100">{k.name}</p>
                  <p className="font-mono text-[10px] text-slate-400">
                    {k.preview} · scope {k.scope} · created {k.created} · last used {k.lastUsed}
                  </p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => toast.push({ title: "Key copied", variant: "info" })}>
                  Copy
                </Button>
                <Button variant="danger" size="sm" onClick={() => toast.push({ title: "Key revoked", description: k.name, variant: "warning" })}>
                  Revoke
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {tab === "model" && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Select
            label="Default model"
            value={model}
            onChange={setModel}
            options={[
              { value: "distilbert-base-uncased", label: "DistilBERT", hint: "67M" },
              { value: "bert-base-uncased", label: "BERT", hint: "110M" },
              { value: "roberta-base", label: "RoBERTa", hint: "125M" },
              { value: "gpt2", label: "GPT-2", hint: "124M" },
              { value: "distilgpt2", label: "DistilGPT-2", hint: "82M" },
            ]}
          />
          <Input label="Batch size" value={batch} onChange={(e) => setBatch(e.target.value)} type="number" />
          <Input label="Learning rate" value={lr} onChange={(e) => setLr(e.target.value)} hint="Wake/Dream phases" />
          <Input label="Random seed" value={seed} onChange={(e) => setSeed(e.target.value)} type="number" />
          <div className="sm:col-span-2 flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={() => {
              setModel("distilbert-base-uncased");
              setBatch("8");
              setLr("5e-5");
              setSeed("42");
              toast.push({ title: "Defaults restored", variant: "info" });
            }}>Reset</Button>
            <Button variant="primary" size="sm" onClick={() => toast.push({ title: "Settings saved", variant: "success" })}>
              Save defaults
            </Button>
          </div>
        </div>
      )}

      {tab === "distortion" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Input label="Dream strength" value={dreamStrength} onChange={(e) => setDreamStrength(e.target.value)} hint="0.10 — 0.50" />
            <Input label="Nightmare strength" value={nightmareStrength} onChange={(e) => setNightmareStrength(e.target.value)} hint="0.50 — 1.00" />
          </div>
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
            <p className="mb-1.5 text-[10px] uppercase tracking-widest text-slate-400">Distortion mix</p>
            <ul className="space-y-1.5 text-[11px]">
              {[
                { label: "Character (typos · swaps · deletes)", weight: 35 },
                { label: "Word (synonyms · drops)", weight: 30 },
                { label: "Semantic (paraphrase)", weight: 20 },
                { label: "Adversarial (PGD · TextFooler)", weight: 15 },
              ].map((m) => (
                <li key={m.label} className="grid grid-cols-[1fr_120px_40px] items-center gap-3">
                  <span className="truncate text-slate-300">{m.label}</span>
                  <span className="h-1 w-full overflow-hidden rounded-full bg-white/[0.04]">
                    <span className="block h-full rounded-full bg-gradient-to-r from-dream to-neural" style={{ width: `${m.weight}%` }} />
                  </span>
                  <span className="text-right font-mono text-slate-300">{m.weight}%</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="flex justify-end">
            <Button variant="primary" size="sm" onClick={() => toast.push({ title: "Distortion mix saved", variant: "success" })}>
              Save mix
            </Button>
          </div>
        </div>
      )}

      {tab === "integrations" && (
        <div className="space-y-4">
          <p className="text-[11px] text-slate-400">Connect external services for model training, evaluation, and experiment tracking.</p>

          <div className="space-y-3">
            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">Azure OpenAI</p>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <Input label="Endpoint" value={azureEndpoint} onChange={(e) => setAzureEndpoint(e.target.value)} hint="https://your-resource.openai.azure.com" />
                <Input label="API Key" value={azureKey} onChange={(e) => setAzureKey(e.target.value)} type="password" />
              </div>
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => toast.push({ title: "Azure OpenAI connected", variant: "success" })}>
                  Test Connection
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">Adaption Labs</p>
              <Input label="API Key" value={adaptionLabsKey} onChange={(e) => setAdaptionLabsKey(e.target.value)} type="password" />
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => toast.push({ title: "Adaption Labs connected", variant: "success" })}>
                  Test Connection
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">HuggingFace</p>
              <Input label="Access Token" value={huggingfaceToken} onChange={(e) => setHuggingfaceToken(e.target.value)} type="password" hint="hf_***" />
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => toast.push({ title: "HuggingFace connected", variant: "success" })}>
                  Test Connection
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">Weights & Biases</p>
              <Input label="API Key" value={wandbKey} onChange={(e) => setWandbKey(e.target.value)} type="password" />
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => toast.push({ title: "W&B connected", variant: "success" })}>
                  Test Connection
                </Button>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <Button variant="primary" size="sm" onClick={() => toast.push({ title: "Integrations saved", variant: "success" })}>
              Save all
            </Button>
          </div>
        </div>
      )}

      {tab === "notifications" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-[11px] text-slate-400">
              Configure incoming webhooks for real-time Slack, Discord, or MS Teams alerts.
            </p>
            <Button
              variant="primary"
              size="sm"
              onClick={() => {
                const newWh: WebhookConfig = {
                  id: `wh-${Date.now()}`,
                  url: "",
                  events: ["run_complete", "regression_detected", "alert", "deploy"],
                };
                setWebhooks([...webhooks, newWh]);
              }}
            >
              Add Webhook
            </Button>
          </div>

          <div className="space-y-3">
            {webhooks.length === 0 ? (
              <div className="rounded-lg border border-dashed border-white/[0.08] p-6 text-center text-slate-500">
                No webhooks configured. Click &quot;Add Webhook&quot; to get started.
              </div>
            ) : (
              webhooks.map((wh) => (
                <div
                  key={wh.id}
                  className="rounded-lg border border-white/[0.06] bg-white/[0.01] p-3.5 space-y-3"
                >
                  <div className="flex gap-2">
                    <Input
                      label="Webhook URL"
                      placeholder="https://hooks.slack.com/services/..."
                      value={wh.url}
                      onChange={(e) => {
                        const updated = webhooks.map((w) =>
                          w.id === wh.id ? { ...w, url: e.target.value } : w
                        );
                        setWebhooks(updated);
                      }}
                    />
                    <div className="flex items-end pb-0.5">
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => {
                          setWebhooks(webhooks.filter((w) => w.id !== wh.id));
                        }}
                      >
                        Remove
                      </Button>
                    </div>
                  </div>

                  <div>
                    <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                      Subscribed Events
                    </span>
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {[
                        { key: "run_complete", label: "Pipeline Complete" },
                        { key: "regression_detected", label: "Regression" },
                        { key: "alert", label: "VRAM Alert" },
                        { key: "deploy", label: "Benchmark Finish" },
                      ].map((ev) => {
                        const active = wh.events.includes(ev.key);
                        return (
                          <label
                            key={ev.key}
                            className="flex cursor-pointer items-center gap-2 rounded bg-white/[0.02] p-1.5 hover:bg-white/[0.04]"
                          >
                            <input
                              type="checkbox"
                              aria-label={`${ev.label} webhook event`}
                              checked={active}
                              onChange={() => {
                                const newEvents = active
                                  ? wh.events.filter((e) => e !== ev.key)
                                  : [...wh.events, ev.key];
                                const updated = webhooks.map((w) =>
                                  w.id === wh.id ? { ...w, events: newEvents } : w
                                );
                                setWebhooks(updated);
                              }}
                              className="rounded border-white/[0.1] bg-black/40 text-neural focus:ring-0"
                            />
                            <span className="text-[11px] text-slate-300">{ev.label}</span>
                          </label>
                        );
                      })}
                    </div>
                  </div>

                  <div className="flex justify-end gap-2 border-t border-white/[0.04] pt-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!wh.url}
                      onClick={async () => {
                        toast.push({
                          title: "Testing Webhook",
                          description: "Sending test payload...",
                          variant: "info",
                        });
                        try {
                          const testEvent = wh.events[0] || "run_complete";
                          await testWebhook({ url: wh.url, event_type: testEvent });
                          toast.push({
                            title: "Webhook Verified",
                            description: "Test notification sent successfully.",
                            variant: "success",
                          });
                        } catch (err: unknown) {
                          const msg = err instanceof Error ? err.message : "Unknown error.";
                          toast.push({
                            title: "Connection Failed",
                            description: msg,
                            variant: "warning",
                          });
                        }
                      }}
                    >
                      Test Connection
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>

          {webhooks.length > 0 && (
            <div className="flex justify-end pt-2">
              <Button
                variant="primary"
                size="sm"
                onClick={async () => {
                  try {
                    await saveWebhooks({
                      webhooks: webhooks.map(w => ({ url: w.url, events: w.events }))
                    });
                    toast.push({
                      title: "Webhooks Saved",
                      description: "Webhook configuration saved successfully.",
                      variant: "success",
                    });
                  } catch (err: unknown) {
                    toast.push({
                      title: "Failed to save",
                      description: "Could not save webhooks.",
                      variant: "warning",
                    });
                  }
                }}
              >
                Save webhooks
              </Button>
            </div>
          )}
        </div>
      )}

      {tab === "preferences" && (
        <div className="space-y-4">
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">Sidebar Layout</p>
            <p className="text-[11px] text-slate-400 mb-3">Reset the personalized dashboard section ordering back to the default state. This removes custom ordering and visit statistics.</p>
            <Button variant="danger" size="sm" onClick={() => {
              window.dispatchEvent(new CustomEvent("reset-sidebar-prefs"));
              toast.push({ title: "Sidebar layout reset", variant: "info" });
            }}>
              Reset to default order
            </Button>
          </div>
        </div>
      )}
    </Panel>
  );
}
