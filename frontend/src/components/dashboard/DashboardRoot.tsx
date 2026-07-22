"use client";

import { lazy, Suspense, useCallback, useMemo, useState, type ReactNode } from "react";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { AnimatePresence, motion } from "framer-motion";
import { AppShell } from "./AppShell";
import type { DashboardSectionKey } from "./Sidebar";
import {
  Skeleton,
  SkeletonChart,
  SkeletonRows,
  SkeletonStatTile,
} from "@/components/ui/Skeleton";
import { OnboardingOverlay } from "./OnboardingOverlay";
import { WhatsNew } from "./WhatsNew";
import { KeyboardHelp } from "./KeyboardHelp";
import { AskNightmareDock } from "./AskNightmareDock";
import { ToastProvider, useToast } from "../ui/Toast";
import { useGlobalShortcuts } from "./useGlobalShortcuts";
import { useSounds } from "@/lib/sounds";


const CommandCenter = lazy(() =>
  import("./CommandCenter").then((module) => ({
    default: module.CommandCenter,
  })),
);
const ExperimentList = lazy(() =>
  import("./ExperimentList").then((module) => ({
    default: module.ExperimentList,
  })),
);
const RunDetail = lazy(() =>
  import("./RunDetail").then((module) => ({ default: module.RunDetail })),
);
const PhaseVisualizer = lazy(() =>
  import("./PhaseVisualizer").then((module) => ({
    default: module.PhaseVisualizer,
  })),
);
const LiveMetrics = lazy(() =>
  import("./LiveMetrics").then((module) => ({ default: module.LiveMetrics })),
);
const RobustnessRadar = lazy(() =>
  import("./RobustnessRadar").then((module) => ({
    default: module.RobustnessRadar,
  })),
);
const ModelComparison = lazy(() =>
  import("./ModelComparison").then((module) => ({
    default: module.ModelComparison,
  })),
);
const DistortionPreview = lazy(() =>
  import("./DistortionPreview").then((module) => ({
    default: module.DistortionPreview,
  })),
);
const AuditTrail = lazy(() =>
  import("./AuditTrail").then((module) => ({ default: module.AuditTrail })),
);
const BenchmarkSuite = lazy(() =>
  import("./BenchmarkSuite").then((module) => ({
    default: module.BenchmarkSuite,
  })),
);
const CIIntegration = lazy(() =>
  import("./CIIntegration").then((module) => ({
    default: module.CIIntegration,
  })),
);
const SettingsPanel = lazy(() =>
  import("./SettingsPanel").then((module) => ({
    default: module.SettingsPanel,
  })),
);
const DataQuality = lazy(() =>
  import("./DataQuality").then((module) => ({ default: module.DataQuality })),
);

type DashboardPanelFallbackProps = {
  label: string;
  variant?: "chart" | "rows" | "stats" | "panel";
};

function DashboardPanelFallback({
  label,
  variant = "panel",
}: DashboardPanelFallbackProps) {
  return (
    <section
      aria-label={`${label} loading`}
      aria-busy="true"
      className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"
    >
      <span className="sr-only">Loading {label}</span>
      {variant === "chart" && <SkeletonChart />}
      {variant === "rows" && <SkeletonRows rows={5} />}
      {variant === "stats" && <SkeletonStatTile />}
      {variant === "panel" && (
        <div className="space-y-3" aria-hidden="true">
          <Skeleton height={18} width="35%" />
          <Skeleton height={12} width="70%" />
          <Skeleton height={180} width="100%" rounded="lg" />
        </div>
      )}
    </section>
  );
}

type PanelSuspenseProps = DashboardPanelFallbackProps & {
  children: ReactNode;
};

function PanelSuspense({ children, label, variant }: PanelSuspenseProps) {
  return (
    <Suspense
      fallback={<DashboardPanelFallback label={label} variant={variant} />}
    >
      {children}
    </Suspense>
  );
}

type SectionMeta = {
  title: string;
  breadcrumb: { label: string }[];
};

const SECTION_META: Record<DashboardSectionKey, SectionMeta> = {
  "command-center": { title: "Command Center", breadcrumb: [{ label: "Overview" }, { label: "Command Center" }] },
  experiments: { title: "Experiments", breadcrumb: [{ label: "Overview" }, { label: "Experiments" }] },
  "run-detail": { title: "Run Detail", breadcrumb: [{ label: "Overview" }, { label: "Run · wikitext-resilient-v3" }] },
  phases: { title: "Phase Visualizer", breadcrumb: [{ label: "Analytics" }, { label: "Phases" }] },
  metrics: { title: "Live Metrics", breadcrumb: [{ label: "Analytics" }, { label: "Metrics" }] },
  robustness: { title: "Robustness Radar", breadcrumb: [{ label: "Analytics" }, { label: "Radar" }] },
  compare: { title: "Model Comparison", breadcrumb: [{ label: "Analytics" }, { label: "Compare" }] },
  distortions: { title: "Distortion Preview", breadcrumb: [{ label: "Analytics" }, { label: "Distortions" }] },
  "data-quality": { title: "Data Quality", breadcrumb: [{ label: "Analytics" }, { label: "Data Quality" }] },
  audit: { title: "Audit Trail", breadcrumb: [{ label: "Operations" }, { label: "Audit" }] },
  benchmarks: { title: "Benchmark Suite", breadcrumb: [{ label: "Operations" }, { label: "Benchmarks" }] },
  ci: { title: "CI Integration", breadcrumb: [{ label: "Operations" }, { label: "CI" }] },
  settings: { title: "Settings", breadcrumb: [{ label: "Operations" }, { label: "Settings" }] },
};

const stagger = {
  initial: {},
  animate: { transition: { staggerChildren: 0.04, delayChildren: 0.03 } },
};

const fadeIn = {
  initial: { opacity: 0, y: 12 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { type: "spring" as const, stiffness: 300, damping: 30 },
  },
  exit: {
    opacity: 0,
    y: -6,
    transition: { duration: 0.15, ease: "easeIn" as const },
  },
};

function DashboardRootInner() {
  const [section, setSection] = useState<DashboardSectionKey>("command-center");
  const [helpOpen, setHelpOpen] = useState(false);
  const [palettePulse, setPalettePulse] = useState(0); // bumped to ask AppShell to open palette
  const meta = useMemo(() => SECTION_META[section], [section]);
  const toast = useToast();
  const { playTransition } = useSounds();

  const navigate = useCallback((next: DashboardSectionKey) => {
    setSection((prev) => {
      if (prev === next) return prev;
      playTransition();
      return next;
    });
  }, [playTransition]);

  useGlobalShortcuts({
    onPaletteToggle: () => setPalettePulse((n) => n + 1),
    onHelpToggle: () => setHelpOpen((o) => !o),
    onNavigate: (next) => {
      navigate(next);
      toast.push({
        title: `Jumped to ${SECTION_META[next].title}`,
        variant: "info",
        durationMs: 1800,
      });
    },
  });

  return (
    <AppShell
      activeSection={section}
      onSectionChange={navigate}
      title={meta.title}
      breadcrumb={meta.breadcrumb}
      apiStatus="online"
      externalPaletteOpenPulse={palettePulse}
      onPaletteAction={(a) => {
        if (a === "new-run") {
          navigate("experiments");
          toast.push({ title: "Opening experiments", description: "Start a new run here.", variant: "info" });
        }
        if (a === "bench-mr") {
          navigate("benchmarks");
          toast.push({ title: "Benchmark suite", description: "Pick a benchmark to run.", variant: "info" });
        }
        if (a === "export") {
          navigate("run-detail");
          toast.push({ title: "Export ready", description: "Latest run report opened.", variant: "success" });
        }
        if (a === "cancel-run") {
          navigate("run-detail");
          toast.push({ title: "Cancellation queued", description: "The active run will stop after the current step.", variant: "warning" });
        }
      }}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={section}
          variants={stagger}
          initial="initial"
          animate="animate"
          exit={{ opacity: 0 }}
          className="space-y-4"
        >
          {section === "command-center" && (
            <>
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Command Center" variant="stats">
                  <CommandCenter />
                </PanelSuspense>
              </motion.div>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <motion.div variants={fadeIn}>
                  <PanelSuspense label="Phase Visualizer">
                    <PhaseVisualizer activePhase={2} />
                  </PanelSuspense>
                </motion.div>
                <motion.div variants={fadeIn}>
                  <PanelSuspense label="Robustness Radar">
                    <RobustnessRadar />
                  </PanelSuspense>
                </motion.div>
              </div>
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Live Metrics" variant="chart">
                  <LiveMetrics />
                </PanelSuspense>
              </motion.div>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <motion.div variants={fadeIn}>
                  <PanelSuspense label="Distortion Preview">
                    <DistortionPreview />
                  </PanelSuspense>
                </motion.div>
                <motion.div variants={fadeIn}>
                  <PanelSuspense label="Audit Trail" variant="rows">
                    <AuditTrail />
                  </PanelSuspense>
                </motion.div>
              </div>
            </>
          )}

          <ErrorBoundary
            fallbackTitle="Experiments unavailable"
            fallbackMessage="The experiment list failed to render. Retry this panel or report the issue."
          >
            {section === "experiments" && (
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Experiments" variant="rows">
                  <ExperimentList onSectionChange={navigate} />
                </PanelSuspense>
              </motion.div>
            )}
          </ErrorBoundary>

          <ErrorBoundary
            fallbackTitle="Run details unavailable"
            fallbackMessage="The selected run details failed to render. Retry this panel or report the issue."
          >
            {section === "run-detail" && (
              <>
                <motion.div variants={fadeIn}>
                  <PanelSuspense label="Run Detail">
                    <RunDetail />
                  </PanelSuspense>
                </motion.div>
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <motion.div variants={fadeIn}>
                    <PanelSuspense label="Live Metrics" variant="chart">
                      <LiveMetrics />
                    </PanelSuspense>
                  </motion.div>
                  <motion.div variants={fadeIn}>
                    <PanelSuspense label="Robustness Radar">
                      <RobustnessRadar />
                    </PanelSuspense>
                  </motion.div>
                </div>
              </>
            )}
          </ErrorBoundary>

          {section === "phases" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="Phase Visualizer">
                <PhaseVisualizer activePhase={1} />
              </PanelSuspense>
            </motion.div>
          )}

          <ErrorBoundary
            fallbackTitle="Live metrics unavailable"
            fallbackMessage="Live metrics failed to render. Other dashboard panels remain available."
          >
            {section === "metrics" && (
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Live Metrics" variant="chart">
                  <LiveMetrics />
                </PanelSuspense>
              </motion.div>
            )}
          </ErrorBoundary>

          {section === "robustness" && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <motion.div className="lg:col-span-2" variants={fadeIn}>
                <PanelSuspense label="Robustness Radar">
                  <RobustnessRadar />
                </PanelSuspense>
              </motion.div>
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Model Comparison">
                  <ModelComparison />
                </PanelSuspense>
              </motion.div>
            </div>
          )}

          {section === "compare" && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Model Comparison">
                  <ModelComparison />
                </PanelSuspense>
              </motion.div>
              <motion.div variants={fadeIn}>
                <PanelSuspense label="Robustness Radar">
                  <RobustnessRadar />
                </PanelSuspense>
              </motion.div>
            </div>
          )}

          {section === "distortions" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="Distortion Preview">
                <DistortionPreview />
              </PanelSuspense>
            </motion.div>
          )}

          {section === "data-quality" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="Data Quality" variant="rows">
                <DataQuality />
              </PanelSuspense>
            </motion.div>
          )}

          {section === "audit" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="Audit Trail" variant="rows">
                <AuditTrail />
              </PanelSuspense>
            </motion.div>
          )}

          {section === "benchmarks" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="Benchmark Suite" variant="rows">
                <BenchmarkSuite />
              </PanelSuspense>
            </motion.div>
          )}

          {section === "ci" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="CI Integration">
                <CIIntegration />
              </PanelSuspense>
            </motion.div>
          )}

          {section === "settings" && (
            <motion.div variants={fadeIn}>
              <PanelSuspense label="Settings">
                <SettingsPanel />
              </PanelSuspense>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>

      <OnboardingOverlay onNavigate={navigate} />
      <WhatsNew />
      <KeyboardHelp open={helpOpen} onClose={() => setHelpOpen(false)} />
      <AskNightmareDock section={section} onNavigate={navigate} />
    </AppShell>
  );
}

export function DashboardRoot() {
  return (
    <ToastProvider>
      <DashboardRootInner />
    </ToastProvider>
  );
}