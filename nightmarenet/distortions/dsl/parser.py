"""YAML and inline expression parser for distortion chain configurations."""

import re
from pathlib import Path
from typing import List, Tuple, Union

import yaml

from nightmarenet.distortions.dsl.schema import ChainConfig, ChainStep
from nightmarenet.distortions.registry import get_registry
from nightmarenet.exceptions import DSLSyntaxError

_ENGINE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
_MAX_CHAIN_LENGTH = 100
_MAX_EXPRESSION_LENGTH = 10000


def parse_chain_config(
    config_path: Union[str, Path],
    validate_engines: bool = True,
) -> ChainConfig:
    """Parse a YAML distortion chain configuration file.

    Args:
        config_path: Path to the YAML configuration file
        validate_engines: If True, verify all referenced engines are registered

    Returns:
        Validated ChainConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If validation fails (schema or engine validation)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    # Validate with Pydantic
    try:
        chain_config = ChainConfig(**data)
    except Exception as e:
        raise ValueError(f"Schema validation failed: {e}") from e

    # Validate engines if requested
    if validate_engines:
        registry = get_registry()
        for step in chain_config.chain:
            if step.engine not in registry:
                available = ", ".join(registry.engine_names)
                raise ValueError(f"Unknown engine '{step.engine}' in step. Available: {available}")

    return chain_config


def validate_chain_config(config_path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate a distortion chain configuration file without loading it.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parse_chain_config(config_path, validate_engines=True)
        return True, "Configuration is valid"
    except FileNotFoundError as e:
        return False, f"File not found: {e}"
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def format_dsl_expression(steps: Union[List[ChainStep], ChainConfig]) -> str:
    """Format a list of ChainSteps or ChainConfig into a DSL string expression.

    Example:
        `[ChainStep(engine="typo", strength=0.3), ChainStep(engine="swap", strength=0.1)]`
        -> `"typo(0.3) -> swap(0.1)"`
    """
    if isinstance(steps, ChainConfig):
        step_list = steps.chain
    elif isinstance(steps, list):
        step_list = steps
    else:
        raise TypeError("steps must be a ChainConfig or a list of ChainStep objects")

    formatted_parts = []
    for step in step_list:
        if not isinstance(step, ChainStep):
            raise TypeError(f"Expected ChainStep object, got {type(step).__name__}")

        s_str = f"{round(step.strength, 4):.4g}"

        if step.condition and step.condition != "always":
            formatted_parts.append(f"{step.engine}({s_str}, condition={repr(step.condition)})")
        else:
            formatted_parts.append(f"{step.engine}({s_str})")

    return " -> ".join(formatted_parts)


def parse_dsl_expression(
    expression: str,
    validate_engines: bool = False,
) -> List[ChainStep]:
    """Parse a DSL string expression into a list of ChainStep objects.

    Format: `"engine(strength) -> engine2(strength2)"`
    Example: `"typo(0.3) -> swap(0.1) -> delete(0.05)"`

    Args:
        expression: DSL expression string.
        validate_engines: If True, verify engines are registered.

    Returns:
        List of ChainStep objects.

    Raises:
        DSLSyntaxError: If syntax is invalid, empty, or bounds are exceeded.
        ValueError: If validate_engines=True and engine is unregistered.
    """
    if not isinstance(expression, str):
        raise DSLSyntaxError("DSL expression must be a string")

    stripped = expression.strip()
    if not stripped:
        raise DSLSyntaxError("DSL expression cannot be empty or whitespace-only")

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise DSLSyntaxError(
            f"DSL expression exceeds maximum length ({_MAX_EXPRESSION_LENGTH} characters)"
        )

    raw_parts = expression.split("->")
    if len(raw_parts) > _MAX_CHAIN_LENGTH:
        raise DSLSyntaxError(
            f"DSL chain length exceeds maximum allowed steps ({_MAX_CHAIN_LENGTH})"
        )

    steps: List[ChainStep] = []
    for idx, raw_part in enumerate(raw_parts, start=1):
        part = raw_part.strip()
        if not part:
            raise DSLSyntaxError(f"Empty step at position {idx} in DSL expression")

        paren_start = part.find("(")
        if paren_start != -1:
            if not part.endswith(")"):
                raise DSLSyntaxError(f"Unclosed parenthesis in step {idx}: '{part}'")

            engine_name = part[:paren_start].strip()
            args_content = part[paren_start + 1 : -1].strip()

            if "(" in args_content or ")" in args_content:
                raise DSLSyntaxError(f"Nested or mismatched parentheses in step {idx}: '{part}'")
        else:
            engine_name = part
            args_content = None

        if not engine_name or not _ENGINE_NAME_PATTERN.match(engine_name):
            raise DSLSyntaxError(f"Invalid engine name '{engine_name}' in step {idx}")

        strength = 0.5
        condition = "always"

        if args_content is not None and args_content != "":
            arg_tokens = [a.strip() for a in args_content.split(",") if a.strip()]
            if not arg_tokens:
                raise DSLSyntaxError(f"Empty argument block in step {idx}: '{part}'")

            try:
                strength = float(arg_tokens[0])
            except ValueError:
                raise DSLSyntaxError(
                    f"Invalid strength value '{arg_tokens[0]}' in step {idx}"
                ) from None

            if not (0.0 <= strength <= 1.0) or strength != strength:
                raise DSLSyntaxError(
                    f"Strength value {strength} in step {idx} must be in range [0.0, 1.0]"
                )

            for extra_arg in arg_tokens[1:]:
                if extra_arg.startswith("condition="):
                    cond_val = extra_arg.split("=", 1)[1].strip().strip("'\"")
                    condition = cond_val
                else:
                    cond_val = extra_arg.strip("'\"")
                    condition = cond_val

        try:
            step = ChainStep(
                engine=engine_name,
                strength=strength,
                condition=condition,
                description=None,
                config={},
            )
        except ValueError as e:
            raise DSLSyntaxError(f"Invalid step configuration in step {idx}: {e}") from e

        steps.append(step)

    if validate_engines:
        registry = get_registry()
        for step in steps:
            if step.engine not in registry:
                available = ", ".join(registry.engine_names)
                raise ValueError(f"Unknown engine '{step.engine}'. Available: {available}")

    return steps
