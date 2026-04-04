const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
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
  const res = await fetch(`${API_BASE}/api/v1/upload/text`, {
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

