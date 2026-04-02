"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from typing import Annotated, Any, Optional

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
    tests_passing: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
