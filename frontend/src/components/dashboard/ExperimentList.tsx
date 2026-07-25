"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Panel } from "./Panel";
import { Badge, type BadgeVariant } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { SkeletonRows } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { InlineEdit } from "@/components/InlineEdit";
import { searchExperiments, deleteExperiment, updateExperiment, exportExperiment, createPipeline, type PipelineCreateRequest } from "@/lib/api";
import {
  IconBeaker,
  IconDownload,
  IconFilter,
  IconKebab,
  IconPlus,
  IconSearch,
  IconSpinner,
} from "./icons";

interface Experiment {
  id: string;
  name: string;
  model: string;
  status: "running" | "complete" | "failed" | "queued";
  cycles: number;
  robustness: number;
  duration: string;
  createdAt: string;
  config?: PipelineCreateRequest;
}

const SAMPLE: Experiment[] = [
  { id: "exp_4f0a", name: "wikitext-resilient-v3", model: "DistilBERT", status: "running", cycles: 4, robustness: 81.4, duration: "12m 04s", createdAt: "2m ago" },
  { id: "exp_3b91", name: "gpt2-domain-shift", model: "GPT-2", status: "running", cycles: 2, robustness: 67.2, duration: "07m 41s", createdAt: "9m ago" },
  { id: "exp_3a17", name: "roberta-attack-eval", model: "RoBERTa", status: "queued", cycles: 0, robustness: 0, duration: "—", createdAt: "12m ago" },
  { id: "exp_2e80", name: "distilgpt2-night-only", model: "DistilGPT-2", status: "complete", cycles: 5, robustness: 86.1, duration: "1h 04m", createdAt: "2h ago" },
  { id: "exp_2d3c", name: "bert-baseline", model: "BERT", status: "complete", cycles: 3, robustness: 74.6, duration: "32m 18s", createdAt: "6h ago" },
  { id: "exp_2c5b", name: "wiki-stress-typos", model: "DistilBERT", status: "failed", cycles: 1, robustness: 0, duration: "04m 12s", createdAt: "1d ago" },
  { id: "exp_2a12", name: "compress-ratio-2x", model: "GPT-2", status: "complete", cycles: 4, robustness: 79.8, duration: "48m 02s", createdAt: "2d ago" },
  { id: "exp_1f88", name: "char-pgd-eval", model: "RoBERTa", status: "complete", cycles: 2, robustness: 83.5, duration: "21m 56s", createdAt: "3d ago" },
];

const statusVariant: Record<Experiment["status"], BadgeVariant> = {
  running: "neural",
  complete: "success",
  failed: "nightmare",
  queued: "warning",
};

interface ToastApi {
  push: (t: {
    title: string;
    description?: string;
    variant: "info" | "success" | "warning" | "error";
  }) => string;
}

interface RowActionsMenuProps {
  row: Experiment;
  toast: ToastApi;
  onDelete: (id: string) => void;
  onRerun: (row: Experiment) => void;
  onExport: (id: string) => void;
  onCompare: (id: string) => void;
  loading: Set<string>;
}

interface MenuItemDef {
  label: string;
  ariaLabel: string;
  variant?: "default" | "danger";
  onSelect: () => void;
}

/**
 * Floating per-row contextual menu. Appears on row hover for pointer devices
 * and is always visible on touch (no `hover:` class — see container below).
 *
 * Keyboard: opens with Enter/Space on the trigger; closes on Escape and on
 * outside click. The trigger acts as a stable anchor so the popover can
 * absolutely-position relative to the cell without measuring DOM.
 */
function RowActionsMenu({ row, toast, onDelete, onRerun, onExport, onCompare, loading }: RowActionsMenuProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close();
      }
    };
    const onClick = (e: MouseEvent) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(e.target as Node)) close();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [open, close]);

  const items: MenuItemDef[] = useMemo(
    () => [
      {
        label: "Compare to baseline",
        ariaLabel: `Compare ${row.name} to baseline`,
        onSelect: () => {
          onCompare(row.id);
        },
      },
      {
        label: "Re-run with strength × 1.2",
        ariaLabel: `Re-run ${row.name} with strength × 1.2`,
        onSelect: () => {
          onRerun(row);
        },
      },
      {
        label: "Export run report (CSV)",
        ariaLabel: `Export ${row.name} run report as CSV`,
        onSelect: () => {
          onExport(row.id);
        },
      },
      {
        label: "Open in new tab",
        ariaLabel: `Open ${row.name} in a new tab`,
        onSelect: () => {
          toast.push({
            title: "Deep-link pending",
            description: "Run-detail deep links land in the next sprint.",
            variant: "warning",
          });
        },
      },
      {
        label: "Delete",
        ariaLabel: `Delete ${row.name}`,
        variant: "danger",
        onSelect: () => {
          onDelete(row.id);
        },
      },
    ],
    [row, onCompare, onRerun, onExport, onDelete]
  );

  return (
    <div
      ref={wrapperRef}
      className={[
        // Visible only when the row is hovered on pointer devices; always
        // visible on touch (no hover capability). Trigger is also focusable
        // via keyboard, which forces visibility.
        "relative inline-flex opacity-100",
        "hover:opacity-100 focus-within:opacity-100",
        "[@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover/row:opacity-100",
      ].join(" ")}
    >
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Row actions for ${row.name}`}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        disabled={loading.has(row.id)}
        className={[
          "inline-flex h-6 w-6 cursor-pointer items-center justify-center rounded-md",
          "border border-transparent text-slate-400",
          "transition-colors hover:border-white/10 hover:bg-white/[0.05] hover:text-slate-100",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neural/50",
          loading.has(row.id) ? "opacity-50 cursor-not-allowed" : "",
        ].join(" ")}
      >
        {loading.has(row.id) ? <IconSpinner size={14} /> : <IconKebab size={14} />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            key="menu"
            role="menu"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.12, ease: "easeOut" }}
            className={[
              "absolute right-0 top-7 z-30 min-w-[200px] origin-top-right",
              "rounded-lg border border-white/[0.08] bg-abyss/95 p-1 shadow-2xl backdrop-blur-md",
            ].join(" ")}
          >
            {items.map((item, i) => {
              const isDanger = item.variant === "danger";
              const isLast = i === items.length - 1;
              return (
                <button
                  key={item.label}
                  type="button"
                  role="menuitem"
                  aria-label={item.ariaLabel}
                  onClick={() => {
                    item.onSelect();
                    close();
                  }}
                  className={[
                    "flex w-full cursor-pointer items-center justify-between gap-3",
                    "rounded-md px-2.5 py-1.5 text-left text-[11.5px]",
                    "transition-colors focus-visible:outline-none focus-visible:ring-1",
                    isDanger
                      ? "text-nightmare-soft hover:bg-nightmare/[0.12] hover:text-nightmare focus-visible:ring-nightmare/50"
                      : "text-slate-300 hover:bg-white/[0.05] hover:text-slate-100 focus-visible:ring-neural/40",
                    !isLast || !isDanger ? "" : "mt-0.5 border-t border-white/[0.04] pt-2",
                  ].join(" ")}
                >
                  <span>{item.label}</span>
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export interface ExperimentListProps {
  loading?: boolean;
  onSectionChange?: (key: "benchmarks" | "experiments" | "run-detail") => void;
  experiments?: Experiment[];
}

export function ExperimentList({
  loading = false,
  onSectionChange,
  experiments,
}: ExperimentListProps = {}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | Experiment["status"]>("all");
  const [semanticIds, setSemanticIds] = useState<string[] | null>(null);
  const [semanticPending, setSemanticPending] = useState(false);
  const [semanticError, setSemanticError] = useState(false);
  const [loadingActions, setLoadingActions] = useState<Set<string>>(new Set());
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [experimentToDelete, setExperimentToDelete] = useState<string | null>(null);
  const toast = useToast();
  const [localNames, setLocalNames] = useState<Record<string, string>>({});

  // `experiments` defaults to the demo sample — callers can pass `[]` to
  // exercise the empty state, or a real dataset once the runs API is wired.
  const source = useMemo(() => {
    const base = experiments ?? SAMPLE;
    if (Object.keys(localNames).length === 0) return base;
    return base.map((exp) =>
      localNames[exp.id] ? { ...exp, name: localNames[exp.id] } : exp
    );
  }, [experiments, localNames]);

  useEffect(() => {
    const trimmed = query.trim();
    setSemanticIds(null);
    if (trimmed.length < 3) {
      setSemanticPending(false);
      setSemanticError(false);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      const statusFilter =
        filter === "all"
          ? undefined
          : { status: filter === "complete" ? "completed" : filter };
      setSemanticPending(true);
      setSemanticError(false);
      searchExperiments(trimmed, 12, statusFilter)
        .then((response) => {
          if (cancelled) return;
          setSemanticIds(response.results.map((result) => result.run_id));
        })
        .catch(() => {
          if (cancelled) return;
          setSemanticIds(null);
          setSemanticError(true);
        })
        .finally(() => {
          if (!cancelled) setSemanticPending(false);
        });
    }, 250);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, filter]);

  const rows = useMemo(() => {
    const semanticRank = new Map((semanticIds ?? []).map((id, idx) => [id, idx]));
    return source
      .filter((r) => {
        if (filter !== "all" && r.status !== filter) return false;
        if (!query.trim()) return true;
        if (semanticRank.has(r.id)) return true;
        const q = query.toLowerCase();
        return (
          r.name.toLowerCase().includes(q) ||
          r.id.toLowerCase().includes(q) ||
          r.model.toLowerCase().includes(q)
        );
      })
      .sort((a, b) => {
        const aRank = semanticRank.get(a.id);
        const bRank = semanticRank.get(b.id);
        if (aRank === undefined && bRank === undefined) return 0;
        if (aRank === undefined) return 1;
        if (bRank === undefined) return -1;
        return aRank - bRank;
      });
  }, [query, filter, source, semanticIds]);

  const sourceEmpty = source.length === 0;

  const handleStartFirst = useCallback(() => {
    if (onSectionChange) {
      onSectionChange("benchmarks");
      toast.push({
        title: "Opening benchmark suite",
        description: "Pick a benchmark to launch your first run.",
        variant: "info",
      });
    } else {
      toast.push({
        title: "Ready when you are",
        description: "Open the Benchmarks section from the sidebar to start a run.",
        variant: "info",
      });
    }
  }, [onSectionChange, toast]);

  const handleRename = useCallback(async (id: string, newName: string) => {
    // Optimistic update
    setLocalNames(prev => ({ ...prev, [id]: newName }));
    try {
      await updateExperiment(id, { name: newName });
    } catch (error) {
      // Revert on failure
      setLocalNames(prev => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      toast.push({
        title: "Rename failed",
        description: "Failed to update experiment name.",
        variant: "error",
      });
      throw error; // Let InlineEdit know it failed
    }
  }, [toast]);

  const handleDelete = useCallback((id: string) => {
    setExperimentToDelete(id);
    setDeleteConfirmOpen(true);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    if (!experimentToDelete) return;
    
    setLoadingActions((prev) => new Set(prev).add(experimentToDelete));
    
    try {
      await deleteExperiment(experimentToDelete);
      toast.push({
        title: "Experiment deleted",
        description: "The experiment has been successfully removed.",
        variant: "success",
      });
      // Note: List refresh requires architectural changes (experiments passed as props)
      // Consider passing a refresh callback or moving to local state management
    } catch (error) {
      toast.push({
        title: "Delete failed",
        description: error instanceof Error ? error.message : "Failed to delete experiment",
        variant: "error",
      });
    } finally {
      setLoadingActions((prev) => {
        const next = new Set(prev);
        next.delete(experimentToDelete);
        return next;
      });
      setDeleteConfirmOpen(false);
      setExperimentToDelete(null);
    }
  }, [experimentToDelete, toast]);

  const handleRerun = useCallback(async (row: Experiment) => {
    setLoadingActions((prev) => new Set(prev).add(row.id));
    
    try {
      const baseConfig = row.config || {
        source_type: "text" as const,
        text_content: "",
        model_name: row.model,
        model_type: "causal_lm" as const,
        num_cycles: row.cycles,
        dream_strength: 0.25,
        nightmare_strength: 0.8,
      };
      
      // Apply 1.2x strength multiplier for re-run
      const config = {
        ...baseConfig,
        dream_strength: (baseConfig.dream_strength ?? 0.25) * 1.2,
        nightmare_strength: (baseConfig.nightmare_strength ?? 0.8) * 1.2,
      };
      
      await createPipeline(config);
      toast.push({
        title: "Re-run queued",
        description: `${row.name} will run at 1.2× the original strength.`,
        variant: "success",
      });
    } catch (error) {
      toast.push({
        title: "Re-run failed",
        description: error instanceof Error ? error.message : "Failed to queue re-run",
        variant: "error",
      });
    } finally {
      setLoadingActions((prev) => {
        const next = new Set(prev);
        next.delete(row.id);
        return next;
      });
    }
  }, [toast]);

  const handleExport = useCallback(async (id: string) => {
    setLoadingActions((prev) => new Set(prev).add(id));
    
    try {
      const response = await exportExperiment(id, "csv");
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${id}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      
      toast.push({
        title: "Export prepared",
        description: `${id}.csv downloaded successfully.`,
        variant: "info",
      });
    } catch (error) {
      toast.push({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Failed to export experiment",
        variant: "error",
      });
    } finally {
      setLoadingActions((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }, [toast]);

  const handleCompare = useCallback((id: string) => {
    // Navigate to compare page with selected experiment ID
    // For now, we'll show a toast since the compare page doesn't exist yet
    toast.push({
      title: "Comparing to baseline",
      description: `Navigate to /compare?ids=${id} to compare experiments.`,
      variant: "info",
    });
    // In a real implementation: window.location.href = `/compare?ids=${id}`;
  }, [toast]);

  const columns: DataTableColumn<Experiment>[] = [
    {
      key: "name",
      header: "Experiment",
      accessor: (r) => r.name,
      sortable: true,
      cell: (r) => (
        <div className="min-w-0">
          <InlineEdit
            value={r.name}
            onSave={(newName) => handleRename(r.id, newName)}
            className="truncate text-sm text-slate-100 block"
          />
          <p className="font-mono text-[10px] text-slate-400">{r.id}</p>
        </div>
      ),
    },
    {
      key: "model",
      header: "Model",
      accessor: (r) => r.model,
      sortable: true,
      cell: (r) => <span className="text-xs text-slate-300">{r.model}</span>,
    },
    {
      key: "status",
      header: "Status",
      accessor: (r) => r.status,
      sortable: true,
      cell: (r) => (
        <Badge variant={statusVariant[r.status]} size="xs" dot>
          {r.status}
        </Badge>
      ),
    },
    {
      key: "cycles",
      header: "Cycles",
      accessor: (r) => r.cycles,
      sortable: true,
      align: "right",
      cell: (r) => <span className="font-mono text-xs">{r.cycles}</span>,
    },
    {
      key: "robustness",
      header: "Robustness",
      accessor: (r) => r.robustness,
      sortable: true,
      align: "right",
      cell: (r) =>
        r.robustness === 0 ? (
          <span className="text-slate-300">—</span>
        ) : (
          <span
            className={[
              "font-mono text-xs",
              r.robustness >= 80 ? "text-emerald-300" : r.robustness >= 70 ? "text-neural" : "text-amber-300",
            ].join(" ")}
          >
            {r.robustness.toFixed(1)}
          </span>
        ),
    },
    {
      key: "duration",
      header: "Duration",
      accessor: (r) => r.duration,
      sortable: true,
      align: "right",
      cell: (r) => <span className="font-mono text-[11px] text-slate-400">{r.duration}</span>,
    },
    {
      key: "createdAt",
      header: "Created",
      accessor: (r) => r.createdAt,
      align: "right",
      cell: (r) => <span className="text-[11px] text-slate-400">{r.createdAt}</span>,
    },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      accessor: () => "",
      align: "right",
      width: "44px",
      cell: (r): ReactNode => (
        <RowActionsMenu
          row={r}
          toast={toast}
          onDelete={handleDelete}
          onRerun={handleRerun}
          onExport={handleExport}
          onCompare={handleCompare}
          loading={loadingActions}
        />
      ),
    },
  ];

  return (
    <Panel
      title="Experiments"
      subtitle={`${rows.length} of ${source.length} runs`}
      icon={<IconBeaker size={14} />}
      glow="dream"
      toolbar={
        <>
          <Input
            placeholder="Search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            leftIcon={<IconSearch size={12} />}
            containerClassName="w-56"
            className="!py-1.5 !text-xs"
            aria-label="Search experiments"
          />
          <Select
            size="sm"
            value={filter}
            onChange={(v) => setFilter(v as typeof filter)}
            className="w-32"
            options={[
              { value: "all", label: "All states" },
              { value: "running", label: "Running" },
              { value: "complete", label: "Complete" },
              { value: "failed", label: "Failed" },
              { value: "queued", label: "Queued" },
            ]}
          />
          <Button variant="ghost" size="sm" aria-label="Filter" onClick={() => setFilter(filter === "all" ? "running" : "all")} title="Toggle running filter">
            <IconFilter size={12} />
          </Button>
          <Button variant="ghost" size="sm" aria-label="Export" onClick={() => {
            const csv = ["id,name,model,status,cycles,robustness,duration,created"]
              .concat(rows.map(r => `${r.id},${r.name},${r.model},${r.status},${r.cycles},${r.robustness},${r.duration},${r.createdAt}`))
              .join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "experiments.csv";
            a.click();
            URL.revokeObjectURL(url);
            toast.push({ title: "Exported experiments", description: `${rows.length} rows as CSV`, variant: "success" });
          }}>
            <IconDownload size={12} />
          </Button>
          <Button variant="primary" size="sm" onClick={handleStartFirst}>
            <IconPlus size={12} /> New Run
          </Button>
        </>
      }
      bodyClassName={loading || sourceEmpty ? "px-4 py-4" : "px-0 py-0"}
    >
      {semanticPending || semanticError ? (
        <div className="border-b border-white/[0.06] px-4 py-2 text-[11px] text-slate-400">
          {semanticPending
            ? "Searching experiment meaning..."
            : "Semantic search unavailable; using local matches."}
        </div>
      ) : null}
      {loading ? (
        <SkeletonRows rows={6} />
      ) : sourceEmpty ? (
        <EmptyState
          icon={<IconBeaker size={18} />}
          title="No experiments yet"
          description="Kick off your first hardening cycle — a starter benchmark takes about ten minutes on a single GPU and shows you the full Wake → Dream → Nightmare → Compress loop."
          primary={{
            label: "Run your first experiment",
            onClick: handleStartFirst,
          }}
          secondary={{
            label: "Browse benchmarks",
            onClick: handleStartFirst,
          }}
        />
      ) : (
        <>
          <ExperimentTable columns={columns} rows={rows} />
          <ConfirmDialog
            open={deleteConfirmOpen}
            onClose={() => setDeleteConfirmOpen(false)}
            title="Delete experiment"
            subtitle="This action cannot be undone. The experiment and all its data will be permanently removed."
            confirmLabel="Delete"
            cancelLabel="Cancel"
            variant="danger"
            onConfirm={handleConfirmDelete}
            isLoading={loadingActions.has(experimentToDelete || "")}
          />
        </>
      )}
    </Panel>
  );
}

/**
 * Wraps DataTable in a `group/row` context per row so the `RowActionsMenu`
 * can fade in on row hover via Tailwind's group-modifier system without
 * touching the generic DataTable primitive.
 */
function ExperimentTable({
  columns,
  rows,
}: {
  columns: DataTableColumn<Experiment>[];
  rows: Experiment[];
}) {
  return (
    <div className="[&_tbody_tr]:group/row">
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r) => r.id}
        density="compact"
        initialSort={{ key: "createdAt", direction: "desc" }}
        empty={<span>No experiments match your filters.</span>}
      />
    </div>
  );
}
