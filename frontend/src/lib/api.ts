/**
 * API origin for browser/SSR fetches.
 * - If `NEXT_PUBLIC_API_URL` is set, it wins (e.g. split domains or e2e).
 * - Otherwise in the browser we use same-origin `/api/...` so `next.config` rewrites
 *   can proxy to the Python backend. Non-browser falls back to localhost for tests/SSR.
 */
export function getApiBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_URL;
  if (typeof fromEnv === "string" && fromEnv.length > 0) {
    return fromEnv.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    return "";
  }
  return "http://127.0.0.1:8000";
}

export interface DistortionRequest {
  text: string;
  strength: number;
  seed?: number;
  config?: Record<string, unknown>;
}

export interface DistortionResponse {
  original_text: string;
  distorted_text: string;
  distortion_type: string;
  strength: number;
  seed: number | null;
}

export interface RobustnessRequest {
  text: string;
  strengths: number[];
}

export interface RobustnessResponse {
  original_text: string;
  scores: {
    dream: Record<string, { similarity: number; length_ratio: number }>;
    nightmare: Record<string, { similarity: number; length_ratio: number }>;
  };
  summary: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  tests_passing?: number | null;
}

// --- Upload ---

export interface UploadResponse {
  filename: string;
  file_type: string;
  text_content: string;
  char_count: number;
  word_count: number;
  line_count: number;
  preview: string;
}

// --- Training Config ---

export interface TrainingConfigRequest {
  model_name?: string;
  model_type?: string;
  num_cycles?: number;
  wake_epochs?: number;
  dream_epochs?: number;
  nightmare_epochs?: number;
  learning_rate?: number;
  nightmare_lr_multiplier?: number;
  batch_size?: number;
  dream_strength?: number;
  nightmare_strength?: number;
  pruning_ratio?: number;
  kl_weight?: number;
  early_stopping?: boolean;
  use_learned_adversarial?: boolean;
}

export interface TrainingPhasePreview {
  cycle: number;
  phase: string;
  epochs: number;
  learning_rate: number;
  description: string;
}

export interface TrainingConfigResponse {
  valid: boolean;
  total_phases: number;
  total_epochs: number;
  estimated_phases: TrainingPhasePreview[];
  config_summary: Record<string, unknown>;
  recommendations: string[];
}

// --- Compare ---

export interface CompareRequest {
  text: string;
  baseline_strength?: number;
  challenge_strength?: number;
  seed?: number;
}

export interface DistortionDetail {
  distorted_text: string;
  similarity: number;
  length_ratio: number;
}

export interface CompareResponse {
  original_text: string;
  baseline_strength: number;
  challenge_strength: number;
  dream: { baseline: DistortionDetail; challenge: DistortionDetail };
  nightmare: { baseline: DistortionDetail; challenge: DistortionDetail };
  resilience_score: number;
  analysis: string;
}

function getApiKey(): string | undefined {
  if (typeof window !== "undefined") {
    return localStorage.getItem("nightmarenet-api-key") || undefined;
  }
  return undefined;
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `API error ${res.status}`);
  }
  return res.json();
}

export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/v1/health");
}

export function generateDream(req: DistortionRequest): Promise<DistortionResponse> {
  return apiFetch<DistortionResponse>("/api/v1/generate/dream", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function generateNightmare(req: DistortionRequest): Promise<DistortionResponse> {
  return apiFetch<DistortionResponse>("/api/v1/generate/nightmare", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function evaluateRobustness(req: RobustnessRequest): Promise<RobustnessResponse> {
  return apiFetch<RobustnessResponse>("/api/v1/evaluate/robustness", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function previewTrainingConfig(
  req: TrainingConfigRequest,
): Promise<TrainingConfigResponse> {
  return apiFetch<TrainingConfigResponse>("/api/v1/train/config", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function compareDistortions(req: CompareRequest): Promise<CompareResponse> {
  return apiFetch<CompareResponse>("/api/v1/compare", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function uploadTextFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${getApiBase()}/api/v1/upload/text`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `Upload failed (${res.status})`);
  }
  return res.json();
}

// --- Demo ---

export interface DemoRequest {
  text: string;
  seed?: number;
}

export interface DemoResponse {
  original_text: string;
  dream: DistortionDetail;
  nightmare: DistortionDetail;
  resilience_delta: number;
  insight: string;
}

export function runDemo(req: DemoRequest): Promise<DemoResponse> {
  return apiFetch<DemoResponse>("/api/v1/demo", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// --- Pipeline ---

export interface PipelineCreateRequest {
  source_type: "urls" | "huggingface" | "text";
  urls?: string[];
  hf_dataset?: string;
  hf_subset?: string;
  text_content?: string;
  model_name?: string;
  model_type?: string;
  num_cycles?: number;
  wake_epochs?: number;
  dream_epochs?: number;
  nightmare_epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  max_samples?: number;
  dream_strength?: number;
  nightmare_strength?: number;
  webhooks?: { url: string; events: string[] }[];
}

export interface PipelineStatusResponse {
  run_id: string;
  status: string;
  current_cycle: number;
  total_cycles: number;
  current_phase: string;
  phase_loss: number;
  progress_pct: number;
  eta_seconds: number;
  is_running: boolean;
  error: string | null;
  has_report: boolean;
  history: Record<string, unknown>[];
}

export interface PipelineReportResponse {
  run_id: string;
  report_md: string;
  comparison: Record<string, unknown> | null;
}

export interface PipelineRunsListResponse {
  runs: PipelineStatusResponse[];
  total: number;
  offset: number;
  limit: number;
}

export function createPipeline(
  req: PipelineCreateRequest,
): Promise<PipelineStatusResponse> {
  return apiFetch<PipelineStatusResponse>("/api/v1/pipeline/create", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getPipelineStatus(
  runId: string,
): Promise<PipelineStatusResponse> {
  return apiFetch<PipelineStatusResponse>(
    `/api/v1/pipeline/${runId}/status`,
  );
}

export function cancelPipeline(
  runId: string,
): Promise<PipelineStatusResponse> {
  return apiFetch<PipelineStatusResponse>(
    `/api/v1/pipeline/${runId}/cancel`,
    { method: "POST" },
  );
}

export function getPipelineReport(
  runId: string,
): Promise<PipelineReportResponse> {
  return apiFetch<PipelineReportResponse>(
    `/api/v1/pipeline/${runId}/report`,
  );
}

export function listPipelineRuns(
  offset?: number,
  limit?: number,
): Promise<PipelineRunsListResponse> {
  const params = new URLSearchParams();
  if (offset !== undefined) params.append("offset", offset.toString());
  if (limit !== undefined) params.append("limit", limit.toString());
  const query = params.toString();
  return apiFetch<PipelineRunsListResponse>(
    `/api/v1/pipeline/runs${query ? `?${query}` : ""}`,
  );
}

// --- Copilot ---

export interface CopilotSuggestion {
  label: string;
  action: string;
  detail: string;
}

export interface CopilotDoneEvent {
  done: true;
  suggestions: CopilotSuggestion[];
  model: string;
}

export interface CopilotAskRequest {
  question: string;
  section: string;
  context?: Record<string, unknown>;
  stream?: boolean;
}

export type CopilotStreamEvent = { token: string } | CopilotDoneEvent;

/**
 * Stream a copilot answer as Server-Sent Events.
 *
 * Yields incremental `{ token }` chunks until the terminal
 * `{ done, suggestions, model }` event. The same response shape is emitted
 * by the heuristic and LLM backends, so consumers never branch.
 *
 * Throws on non-200 responses; callers should catch and degrade to a
 * heuristic UI so the dock never appears broken.
 */
export async function* askCopilot(
  question: string,
  section: string,
  context?: Record<string, unknown>,
  signal?: AbortSignal,
): AsyncGenerator<CopilotStreamEvent, void, void> {
  const res = await fetch(`${getApiBase()}/api/v1/copilot/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...authHeaders(),
    },
    body: JSON.stringify({ question, section, context, stream: true }),
    signal,
  });
  if (!res.ok) {
    let detail = `Copilot error ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch {
      // body wasn't JSON; keep status code message
    }
    throw new Error(detail);
  }
  if (!res.body) {
    throw new Error("Copilot returned no stream body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line ("\n\n"). Handle CRLF too.
      let sepIdx = buffer.indexOf("\n\n");
      while (sepIdx !== -1) {
        const raw = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + 2);
        sepIdx = buffer.indexOf("\n\n");

        for (const line of raw.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            const evt = JSON.parse(payload) as CopilotStreamEvent;
            yield evt;
          } catch {
            // ignore malformed events
          }
        }
      }
    }
    } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignored
    }
  }
}

// --- Data optimization (Adaption Labs) ---

export interface BrandControls {
  length?: "minimal" | "concise" | "detailed" | "extensive";
  safety_categories?: string[];
  hallucination_mitigation?: boolean;
  blueprint?: string;
}

export interface RecipeSpecification {
  recipes?: {
    reasoning_traces?: boolean;
    deduplication?: boolean;
    prompt_rephrase?: boolean;
  };
}

export interface JobSpecification {
  max_rows?: number;
  idempotency_key?: string;
}

export interface DataOptimizeRequest {
  texts: string[];
  column_mapping: Record<string, string | string[]>;
  phase?: "wake" | "dream" | "nightmare" | "compress";
  brand_controls?: BrandControls;
  recipe_specification?: RecipeSpecification;
  job_specification?: JobSpecification;
  estimate_only?: boolean;
}

export interface DataImportRequest {
  source: "huggingface" | "kaggle";
  url: string;
  files: string[];
  column_mapping: Record<string, string | string[]>;
  brand_controls?: BrandControls;
  recipe_specification?: RecipeSpecification;
}

export interface DataOptimizeResponse {
  status: string;
  run_id?: string | null;
  optimized_count?: number | null;
  quality?: Record<string, unknown> | null;
  estimate?: { credits?: number; estimated_minutes?: number } | null;
  elapsed_seconds?: number | null;
  before_stats?: DataStats | null;
  after_stats?: DataStats | null;
  quality_delta?: { count_change?: number; avg_length_change?: number } | null;
}

export interface DataStats {
  count: number;
  avg_length: number;
  total_chars: number;
  avg_words: number;
  min_length?: number;
  max_length?: number;
}

export interface OptimizeStreamEvent {
  event: "start" | "progress" | "complete" | "error";
  run_id: string;
  state?: string;
  progress_pct?: number;
  message?: string;
  result?: { optimized_count?: number; quality?: Record<string, unknown> } | null;
  before_stats?: DataStats | null;
  after_stats?: DataStats | null;
  elapsed_seconds?: number;
  error?: string;
}

export function optimizeData(body: DataOptimizeRequest): Promise<DataOptimizeResponse> {
  return apiFetch<DataOptimizeResponse>("/api/v1/data/optimize", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function* optimizeDataStream(
  body: DataOptimizeRequest,
  signal?: AbortSignal,
): AsyncGenerator<OptimizeStreamEvent, void, void> {
  const res = await fetch(`${getApiBase()}/api/v1/data/optimize/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...authHeaders(),
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = `Optimization error ${res.status}`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || errBody.error || detail;
    } catch {
      // non-JSON response
    }
    throw new Error(detail);
  }
  if (!res.body) {
    throw new Error("No stream body returned");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIdx = buffer.indexOf("\n\n");
      while (sepIdx !== -1) {
        const raw = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + 2);
        sepIdx = buffer.indexOf("\n\n");

        for (const line of raw.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            yield JSON.parse(payload) as OptimizeStreamEvent;
          } catch {
            // skip malformed
          }
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignored
    }
  }
}

// --- Config suggestions ---

export interface ConfigSuggestion {
  param: string;
  current: unknown;
  suggested: unknown;
  reason: string;
}

export interface SuggestConfigRequest {
  current_config: Record<string, unknown>;
  last_metrics?: Record<string, unknown>;
  hardware?: string;
}

export interface SuggestConfigResponse {
  suggestions: ConfigSuggestion[];
  model: string;
}

export function suggestConfig(body: SuggestConfigRequest): Promise<SuggestConfigResponse> {
  return apiFetch<SuggestConfigResponse>("/api/v1/suggest/config", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// --- Experiment Search ---

export interface ExperimentSearchResult {
  run_id: string;
  relevance_score: number;
  summary: string;
  metadata: Record<string, unknown>;
}

export interface ExperimentSearchResponse {
  results: ExperimentSearchResult[];
  filters: Record<string, unknown>;
  backend: string;
}

export function searchExperiments(
  query: string,
  topK = 10,
  filters?: Record<string, unknown>,
): Promise<ExperimentSearchResponse> {
  return apiFetch<ExperimentSearchResponse>("/api/v1/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK, filters: filters ?? {} }),
  });
}

// --- Adaption Labs: Import & Estimate ---

export function importAndOptimize(body: DataImportRequest): Promise<DataOptimizeResponse> {
  return apiFetch<DataOptimizeResponse>("/api/v1/data/import", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function estimateOptimization(body: DataOptimizeRequest): Promise<DataOptimizeResponse> {
  return apiFetch<DataOptimizeResponse>("/api/v1/data/optimize/estimate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// --- Notifications & Webhooks ---

export interface TestWebhookRequest {
  url: string;
  event_type: string;
}

export function testWebhook(body: TestWebhookRequest): Promise<{ status: string }> {
  return apiFetch<{ status: string }>("/api/v1/notifications/test-webhook", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export interface WebhookConfig {
  url: string;
  events: string[];
}

export interface WebhookSettingsRequest {
  webhooks: WebhookConfig[];
}

export interface WebhookSettingsResponse {
  webhooks: WebhookConfig[];
}

export function getWebhooks(): Promise<WebhookSettingsResponse> {
  return apiFetch<WebhookSettingsResponse>("/api/v1/settings/webhooks");
}

export function saveWebhooks(body: WebhookSettingsRequest): Promise<WebhookSettingsResponse> {
  return apiFetch<WebhookSettingsResponse>("/api/v1/settings/webhooks", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// --- Experiment Management ---

export interface ExperimentDeleteResponse {
  run_id: string;
  deleted: boolean;
}

export function deleteExperiment(runId: string): Promise<ExperimentDeleteResponse> {
  return apiFetch<ExperimentDeleteResponse>(`/api/v1/experiments/${runId}`, {
    method: "DELETE",
  });
}

export interface ExperimentExportResponse {
  run_id: string;
  format: string;
  data: string;
}

export function exportExperiment(runId: string, format: "csv" | "json" = "csv"): Promise<ExperimentExportResponse> {
  return apiFetch<ExperimentExportResponse>(`/api/v1/experiments/${runId}/export?format=${format}`);
}
