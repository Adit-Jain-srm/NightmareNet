"""Distortion DSL for composable YAML-defined attack chains."""

from nightmarenet.distortions.dsl.executor import ChainExecutor
from nightmarenet.distortions.dsl.parser import (
    format_dsl_expression,
    parse_chain_config,
    parse_dsl_expression,
)
from nightmarenet.distortions.dsl.preset_loader import list_presets, load_preset
from nightmarenet.distortions.dsl.schema import ChainConfig, ChainStep, Defaults
from nightmarenet.exceptions import DSLSyntaxError

__all__ = [
    "ChainConfig",
    "ChainStep",
    "Defaults",
    "ChainExecutor",
    "parse_chain_config",
    "parse_dsl_expression",
    "format_dsl_expression",
    "list_presets",
    "load_preset",
    "DSLSyntaxError",
]
