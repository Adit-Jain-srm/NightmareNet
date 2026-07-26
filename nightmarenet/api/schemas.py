"""Pydantic models for API request/response schemas."""

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

StrengthFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class DistortionRequest(BaseModel):
    """Request body for dream/nightmare text generation endpoints."""

    text: str = Field(..., min_length=1, max_length=50000, description="Input text to distort.")
    strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Distortion strength in [0, 1]. Dream: 0.2-0.3, Nightmare: 0.7-0.9.",
    )
    config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional per-distortion type weight overrides.",
    )
    seed: Optional[int] = Field(
        default=None, ge=0, le=2**31 - 1, description="Random seed for reproducibility."
    )


class DistortionResponse(BaseModel):
    """Response body for dream/nightmare text generation endpoints."""

    original_text: str
    distorted_text: str
    distortion_type: str  # "dream" or "nightmare"
    strength: float
    seed: Optional[int] = None


class RobustnessRequest(BaseModel):
    """Request body for robustness evaluation endpoint."""

    text: str = Field(..., min_length=1, max_length=50000, description="Text to evaluate.")
    strengths: list[StrengthFloat] = Field(
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        min_length=1,
        description="Distortion strengths to test at (each must be in [0, 1]).",
    )


class RobustnessResponse(BaseModel):
    """Response body for robustness evaluation endpoint."""

    original_text: str
    scores: dict[str, Any]
    summary: str


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = "ok"
    version: str
    tests_passing: Optional[int] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None


class AuthErrorResponse(BaseModel):
    """Authentication error response."""

    error: str = "Unauthorized"
    detail: str = "Invalid or missing API key."


# --- Training Config Schemas ---


class TrainingConfigRequest(BaseModel):
    """Request body for training configuration validation and preview."""

    model_name: str = Field(
        default="gpt2",
        description="HuggingFace model name or path.",
    )
    model_type: str = Field(
        default="causal_lm",
        description="Model type: causal_lm, masked_lm, or seq_classification.",
    )
    num_cycles: int = Field(default=3, ge=1, le=100, description="Number of full sleep cycles.")
    wake_epochs: int = Field(default=3, ge=0, le=50, description="Epochs per wake phase.")
    dream_epochs: int = Field(default=2, ge=0, le=50, description="Epochs per dream phase.")
    nightmare_epochs: int = Field(default=1, ge=0, le=50, description="Epochs per nightmare phase.")
    learning_rate: float = Field(default=5e-5, gt=0, le=1.0, description="Base learning rate.")
    nightmare_lr_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Learning rate multiplier for nightmare phase.",
    )
    batch_size: int = Field(default=8, ge=1, le=256, description="Training batch size.")
    dream_strength: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Dream distortion strength."
    )
    nightmare_strength: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Nightmare distortion strength."
    )
    pruning_ratio: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="Fraction of weights to prune."
    )
    kl_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="KL divergence loss weight during dream phase.",
    )
    early_stopping: bool = Field(default=False, description="Stop training when loss plateaus.")
    use_learned_adversarial: bool = Field(
        default=False,
        description="Use learned adversarial distortions (MLM-based). Slower but stronger.",
    )


class TrainingPhasePreview(BaseModel):
    """Preview of a single training phase."""

    cycle: int
    phase: str
    epochs: int
    learning_rate: float
    description: str


class TrainingConfigResponse(BaseModel):
    """Response for training config validation and preview."""

    valid: bool
    total_phases: int
    total_epochs: int
    estimated_phases: list[TrainingPhasePreview]
    config_summary: dict[str, Any]
    recommendations: list[str]


# --- Model Comparison Schemas ---


class CompareRequest(BaseModel):
    """Request body for comparing distortion effects at two strength profiles."""

    text: str = Field(..., min_length=1, max_length=50000, description="Text to evaluate.")
    baseline_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Baseline distortion strength (milder).",
    )
    challenge_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Challenge distortion strength (stronger).",
    )
    seed: Optional[int] = Field(
        default=42, ge=0, le=2**31 - 1, description="Random seed for reproducibility."
    )


class DistortionDetail(BaseModel):
    """Detail of a single distortion result."""

    distorted_text: str
    similarity: float
    length_ratio: float


class CompareResponse(BaseModel):
    """Response for distortion comparison endpoint."""

    original_text: str
    baseline_strength: float
    challenge_strength: float
    dream: dict[str, DistortionDetail]
    nightmare: dict[str, DistortionDetail]
    resilience_score: float = Field(
        description="How well text structure survives escalation (0-1, higher=more resilient).",
    )
    analysis: str


# --- File Upload Schemas ---


class UploadResponse(BaseModel):
    """Response for file upload endpoint."""

    filename: str
    file_type: str
    text_content: str
    char_count: int
    word_count: int
    line_count: int
    preview: str = Field(
        description="First 500 characters of the file content.",
    )


# --- Demo Schemas ---


class DemoRequest(BaseModel):
    """Request body for the interactive demo endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to distort through dream and nightmare modes.",
    )
    seed: Optional[int] = Field(
        default=42,
        ge=0,
        le=2**31 - 1,
        description="Random seed for reproducible demo results.",
    )


class DemoResponse(BaseModel):
    """Response for the interactive demo — dream + nightmare in one call."""

    original_text: str
    dream: DistortionDetail = Field(
        description="Dream distortion result at strength=0.25.",
    )
    nightmare: DistortionDetail = Field(
        description="Nightmare distortion result at strength=0.80.",
    )
    resilience_delta: float = Field(
        description="Similarity drop from dream to nightmare (0-1).",
    )
    insight: str = Field(
        description=("Human-readable explanation of what the distortions reveal."),
    )


# --- Pipeline Schemas ---


class WebhookConfigSchema(BaseModel):
    """Configuration for a webhook subscription."""

    url: str = Field(..., description="The webhook URL.")
    events: list[str] = Field(
        default_factory=list,
        description="Events to subscribe to: run_complete, regression_detected, alert, deploy",
    )


class WebhookSettingsRequest(BaseModel):
    """Request body for saving webhook settings."""

    webhooks: list[WebhookConfigSchema] = Field(
        default_factory=list, description="List of webhook configurations."
    )


class WebhookSettingsResponse(BaseModel):
    """Response body containing webhook settings."""

    webhooks: list[WebhookConfigSchema] = Field(
        default_factory=list, description="List of webhook configurations."
    )


class PipelineCreateRequest(BaseModel):
    """Request body for creating a new end-to-end pipeline run."""

    # Data source (exactly one must be provided)
    source_type: str = Field(
        ...,
        description="One of: 'urls', 'huggingface', 'text'.",
    )
    urls: Optional[list[str]] = Field(
        default=None,
        description="List of URLs to scrape (when source_type='urls').",
    )
    hf_dataset: Optional[str] = Field(
        default=None,
        description="HuggingFace dataset name (when source_type='huggingface').",
    )
    hf_subset: Optional[str] = Field(
        default=None,
        description="HuggingFace dataset subset.",
    )
    text_content: Optional[str] = Field(
        default=None,
        description="Raw text content (when source_type='text').",
    )

    # Model config
    model_name: str = Field(
        default="distilbert-base-uncased",
        description="HuggingFace model name.",
    )
    model_type: str = Field(
        default="masked_lm",
        description="Model type: causal_lm, masked_lm, or seq_classification.",
    )

    # Training config
    num_cycles: int = Field(default=1, ge=1, le=10, description="Sleep cycles.")
    wake_epochs: int = Field(default=1, ge=1, le=10, description="Wake epochs per cycle.")
    dream_epochs: int = Field(default=1, ge=1, le=10, description="Dream epochs per cycle.")
    nightmare_epochs: int = Field(default=1, ge=1, le=10, description="Nightmare epochs per cycle.")
    learning_rate: float = Field(default=5e-5, gt=0, le=1.0)
    batch_size: int = Field(default=8, ge=1, le=64)
    max_samples: Optional[int] = Field(
        default=500,
        description="Max training samples (None for unlimited).",
    )
    dream_strength: float = Field(default=0.25, ge=0.0, le=1.0)
    nightmare_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    webhooks: Optional[list[WebhookConfigSchema]] = Field(
        default=None,
        description="Optional list of webhooks to trigger during pipeline events.",
    )


class PipelineStatusResponse(BaseModel):
    """Current status of a pipeline run."""

    run_id: str
    status: str
    current_cycle: int = 0
    total_cycles: int = 0
    current_phase: str = ""
    phase_loss: float = 0.0
    progress_pct: float = 0.0
    eta_seconds: float = 0.0
    is_running: bool = False
    error: Optional[str] = None
    has_report: bool = False
    history: list[dict[str, Any]] = Field(default_factory=list)


class PipelineReportResponse(BaseModel):
    """Full evaluation report for a completed pipeline."""

    run_id: str
    report_md: str
    comparison: Optional[dict[str, Any]] = None


class PipelineRunsListResponse(BaseModel):
    """Paginated response for listing pipeline runs."""

    runs: list[PipelineStatusResponse]
    total: int
    offset: int
    limit: int


class TestWebhookRequest(BaseModel):
    """Request body for testing a webhook URL."""

    url: str = Field(..., description="The webhook endpoint URL.")
    event_type: str = Field(
        default="run_complete",
        description="Event type to test: run_complete, regression_detected, alert, deploy",
    )


class PipelineTrainRequest(BaseModel):
    """Request body for training a pipeline."""

    config: dict[str, Any] = Field(..., description="Training config")
    model_name: str = Field(default="gpt2", description="Model to train")
    cycles: int = Field(default=1, ge=1, le=100)


class PipelineEvaluateRequest(BaseModel):
    """Request body for evaluating a pipeline."""

    config: dict[str, Any] = Field(..., description="Evaluation config")
    model_name: str = Field(default="gpt2", description="Model to evaluate")


class PipelineCancelRequest(BaseModel):
    """Request body for cancelling a pipeline run."""

    pipeline_id: str = Field(..., description="ID of the pipeline to cancel")
    force: bool = Field(default=False, description="Force cancellation")


class SettingsWebhooksRequest(BaseModel):
    """Request body for updating webhook settings."""

    webhooks: list[WebhookConfigSchema] = Field(..., description="List of webhook configurations")


class WebhookTestResponse(BaseModel):
    """Response for testing webhooks, preserving legacy contract."""

    status: Literal["ok"]
