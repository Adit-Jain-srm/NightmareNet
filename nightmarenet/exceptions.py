"""
Custom exception hierarchy for NightmareNet.

These exceptions replace bare `except Exception:` handlers in
pipeline.py and pipeline_runner.py. Each carries structured context
(phase, cycle, config path, etc.) so callers can log, recover, or
surface actionable, user-facing error messages instead of generic
tracebacks.
"""

from __future__ import annotations

from typing import Any, Optional


class NightmareNetError(Exception):
    """Base class for all NightmareNet-specific errors.

    Subclasses should set `self.message` to a short, user-facing
    description and may optionally set `self.hint` with a recovery
    suggestion. `__str__` combines both so the exception is readable
    on its own (e.g. in logs or CLI output) without extra formatting.
    """

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} (hint: {self.hint})"
        return self.message


class PipelinePhaseError(NightmareNetError):
    """Raised when a named pipeline phase fails during a training/run cycle."""

    def __init__(
        self,
        phase: str,
        cycle: Optional[int] = None,
        details: Optional[str] = None,
        *,
        hint: Optional[str] = None,
    ) -> None:
        self.phase = phase
        self.cycle = cycle
        self.details = details

        msg = f"Pipeline phase '{phase}' failed"
        if cycle is not None:
            msg += f" on cycle {cycle}"
        if details:
            msg += f": {details}"

        super().__init__(msg, hint=hint or "check the phase logs for the underlying cause")


class ConfigurationError(NightmareNetError):
    """Raised when a config value is missing, malformed, or invalid."""

    def __init__(
        self,
        key_path: str,
        message: str,
        *,
        hint: Optional[str] = None,
    ) -> None:
        self.key_path = key_path
        self.config_message = message

        msg = f"Configuration error at '{key_path}': {message}"
        super().__init__(msg, hint=hint or f"verify the '{key_path}' entry in your config file")


class CheckpointCorruptError(NightmareNetError):
    """Raised when a model checkpoint cannot be loaded or fails validation."""

    def __init__(
        self,
        path: str,
        reason: str,
        *,
        hint: Optional[str] = None,
    ) -> None:
        self.path = path
        self.reason = reason

        msg = f"Checkpoint at '{path}' is corrupt or unreadable: {reason}"
        super().__init__(
            msg,
            hint=hint or "restore from the last known-good checkpoint or re-run from scratch",
        )


class HubUploadError(NightmareNetError):
    """Raised when uploading a model/artifact to a model hub fails."""

    def __init__(
        self,
        model_id: str,
        status: Any,
        *,
        hint: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.status = status

        msg = f"Upload of model '{model_id}' to hub failed (status: {status})"
        super().__init__(
            msg,
            hint=hint or "check hub credentials/connectivity and retry the upload",
        )


class DataIngestError(NightmareNetError):
    """Raised when ingesting data from a source fails or yields invalid data."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        hint: Optional[str] = None,
    ) -> None:
        self.source = source
        self.reason = reason

        msg = f"Failed to ingest data from '{source}': {reason}"
        super().__init__(
            msg,
            hint=hint or "verify the data source is reachable and correctly formatted",
        )


class DSLSyntaxError(ValueError, NightmareNetError):
    """Raised when a distortion DSL string expression cannot be parsed due to invalid syntax."""

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        self.message = message
        self.hint = hint
        ValueError.__init__(self, message)
        NightmareNetError.__init__(
            self,
            message,
            hint=hint or "check DSL expression syntax (e.g., 'typo(0.3) -> swap(0.1)')",
        )
