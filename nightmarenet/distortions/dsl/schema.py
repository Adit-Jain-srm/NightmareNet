"""Pydantic schema for distortion chain configuration validation."""

import ast
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Defaults(BaseModel):
    """Default configuration values for the chain."""

    seed: int = 42
    preserve_length: bool = False
    max_retries: int = 3


class ChainStep(BaseModel):
    """A single step in a distortion chain."""

    engine: str = Field(..., description="Name of the distortion engine to apply")
    strength: float = Field(..., ge=0.0, le=1.0, description="Strength of distortion (0-1)")
    description: Optional[str] = Field(None, description="Human-readable description of this step")
    condition: Optional[str] = Field(
        "always",
        description="Condition for applying this step (e.g., 'strength > 0.5', 'always')",
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional engine-specific configuration",
    )

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v: Optional[str]) -> str:
        """Validate condition syntax using AST parsing for security."""
        if v is None or v == "":
            return "always"
        v = v.strip()
        if v == "always":
            return v

        # Use AST parsing to validate the condition structure
        try:
            tree = ast.parse(v, mode="eval")
            cls._validate_condition_ast(tree.body)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e

        return v

    @staticmethod
    def _validate_condition_ast(node: ast.AST) -> None:
        """Validate that the AST only contains allowed constructs.

        Only allows simple comparisons: strength OP number
        where OP is one of: >, <, >=, <=, ==, !=

        Args:
            node: AST node to validate

        Raises:
            ValueError: If disallowed constructs are found
        """
        if isinstance(node, ast.Compare):
            # Validate the left side (should be 'strength' variable)
            if not isinstance(node.left, ast.Name) or node.left.id != "strength":
                raise ValueError("Condition must compare 'strength' variable")

            # Validate the right side (should be a number)
            if len(node.comparators) != 1:
                raise ValueError("Condition must have exactly one comparison")

            comparator = node.comparators[0]
            if not isinstance(comparator, (ast.Constant, ast.Num)):
                # ast.Num is for Python < 3.8, ast.Constant >= 3.8
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    raise ValueError("None literal is not allowed")
                raise ValueError("Condition must compare with a numeric literal")

            # Validate the operator
            allowed_ops = {
                ast.Gt: ">",
                ast.Lt: "<",
                ast.GtE: ">=",
                ast.LtE: "<=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            if len(node.ops) != 1 or type(node.ops[0]) not in allowed_ops:
                raise ValueError(f"Condition must use one of: {', '.join(allowed_ops.values())}")
        else:
            raise ValueError("Condition must be a comparison")


class ChainConfig(BaseModel):
    """Complete distortion chain configuration."""

    name: str = Field(..., description="Name of the distortion chain")
    description: Optional[str] = Field(None, description="Human-readable description")
    version: int = Field(1, ge=1, description="Configuration version")
    chain: List[ChainStep] = Field(
        ...,
        min_length=1,
        description="Ordered list of distortion steps",
    )
    defaults: Defaults = Field(default_factory=Defaults, description="Default configuration values")
