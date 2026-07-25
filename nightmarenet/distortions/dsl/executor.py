"""Sequential chain executor for distortion chains."""

import ast
import logging
from typing import Optional

from nightmarenet.distortions.dsl.schema import ChainConfig
from nightmarenet.distortions.registry import get_registry

logger = logging.getLogger(__name__)


class ChainExecutor:
    """Executes distortion chains sequentially with condition evaluation."""

    def __init__(self, registry=None):
        """Initialize the executor.

        Args:
            registry: Optional DistortionRegistry instance. If None, uses global registry.
        """
        self.registry = registry or get_registry()

    def _evaluate_condition(self, condition: Optional[str], strength: float) -> bool:
        """Evaluate a condition string against the current strength.

        Uses AST parsing for safe evaluation - only allows simple comparisons
        with the 'strength' variable and numeric literals.

        Args:
            condition: Condition string (e.g., "strength > 0.5", "always")
            strength: Current strength value to evaluate against

        Returns:
            True if condition passes, False otherwise
        """
        if condition is None or condition == "always":
            return True

        try:
            return self._safe_eval_condition(condition, strength)
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _safe_eval_condition(self, condition: str, strength: float) -> bool:
        """Safely evaluate a condition using AST parsing.

        Only allows: strength comparisons with numeric literals using
        operators: >, <, >=, <=, ==, !=

        Args:
            condition: Condition string to evaluate
            strength: Strength value to compare against

        Returns:
            Boolean result of the comparison

        Raises:
            ValueError: If condition contains disallowed constructs
        """
        # Parse the condition into an AST
        tree = ast.parse(condition, mode="eval")

        # Validate the AST structure
        self._validate_condition_ast(tree.body)

        # Safely evaluate the validated AST
        return self._eval_ast(tree.body, strength)

    def _validate_condition_ast(self, node: ast.AST) -> None:
        """Validate that the AST only contains allowed constructs.

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

    def _eval_ast(self, node: ast.AST, strength: float) -> bool:
        """Evaluate a validated AST node.

        Args:
            node: AST node to evaluate
            strength: Strength value

        Returns:
            Boolean result
        """
        if isinstance(node, ast.Compare):
            left = strength  # We validated this is the 'strength' variable
            right = self._eval_literal(node.comparators[0])
            op = node.ops[0]

            if isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.GtE):
                return left >= right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            else:
                raise ValueError(f"Unsupported operator: {type(op)}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _eval_literal(self, node: ast.AST) -> float:
        """Evaluate a literal node to a float.

        Args:
            node: AST literal node

        Returns:
            Float value
        """
        if isinstance(node, ast.Constant):
            if node.value is None:
                raise ValueError("None literal is not allowed")
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric literals are allowed")
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            if isinstance(node.n, (int, float)):
                return float(node.n)
            raise ValueError(f"Unsupported numeric type in ast.Num: {type(node.n)}")
        else:
            raise ValueError(f"Unsupported literal type: {type(node)}")

    def execute(
        self,
        text: str,
        chain_config: ChainConfig,
        overall_strength: float,
        seed: Optional[int] = None,
    ) -> str:
        """Execute a distortion chain sequentially.

        Each step evaluates its condition, applies if passed,
        and feeds output to the next step. Failed steps are logged but don't abort the chain.

        Args:
            text: Input text to distort
            chain_config: Chain configuration to execute
            overall_strength: Overall strength for the chain (used for condition evaluation)
            seed: Random seed for reproducibility (overrides chain defaults if provided)

        Returns:
            Final distorted text after applying all applicable steps
        """
        effective_seed = seed if seed is not None else chain_config.defaults.seed

        current_text = text
        steps_applied = 0
        steps_skipped = 0
        steps_failed = 0

        logger.info(f"Executing chain '{chain_config.name}' with strength {overall_strength}")

        for i, step in enumerate(chain_config.chain):
            step_num = i + 1

            # Check condition
            if not self._evaluate_condition(step.condition, overall_strength):
                logger.debug(
                    f"Step {step_num} ({step.engine}) skipped: condition '{step.condition}' not met"
                )
                steps_skipped += 1
                continue

            # Apply the distortion
            try:
                logger.debug(
                    f"Step {step_num} ({step.engine}): applying with strength {step.strength}"
                )

                # Use step-specific strength, not overall strength
                step_seed = effective_seed + i if effective_seed is not None else None
                distorted = self.registry.apply(
                    step.engine,
                    current_text,
                    strength=step.strength,
                    seed=step_seed,
                )

                current_text = distorted
                steps_applied += 1
                logger.debug(f"Step {step_num} completed successfully")

            except Exception as e:
                logger.warning(
                    f"Step {step_num} ({step.engine}) failed: {e}. Skipping and continuing."
                )
                steps_failed += 1
                # Don't abort the chain - continue with next step
                continue

        logger.info(
            f"Chain execution complete: {steps_applied} applied, "
            f"{steps_skipped} skipped, {steps_failed} failed"
        )

        return current_text

    def execute_with_trace(
        self,
        text: str,
        chain_config: ChainConfig,
        overall_strength: float,
        seed: Optional[int] = None,
    ) -> dict:
        """Execute a chain and return detailed trace information.

        Useful for debugging and UI visualization of step-by-step transformations.

        Args:
            text: Input text to distort
            chain_config: Chain configuration to execute
            overall_strength: Overall strength for the chain
            seed: Random seed for reproducibility

        Returns:
            Dictionary with trace information including:
            - original: Original text
            - final: Final distorted text
            - steps: List of step results with input/output and status
        """
        effective_seed = seed if seed is not None else chain_config.defaults.seed

        current_text = text
        steps_trace = []

        for i, step in enumerate(chain_config.chain):
            step_num = i + 1
            step_trace = {
                "step": step_num,
                "engine": step.engine,
                "strength": step.strength,
                "condition": step.condition,
                "input": current_text,
                "status": "skipped",
                "output": current_text,
                "error": None,
            }

            # Check condition
            if not self._evaluate_condition(step.condition, overall_strength):
                steps_trace.append(step_trace)
                continue

            # Apply the distortion
            try:
                step_seed = effective_seed + i if effective_seed is not None else None
                distorted = self.registry.apply(
                    step.engine,
                    current_text,
                    strength=step.strength,
                    seed=step_seed,
                )

                step_trace["status"] = "applied"
                step_trace["output"] = distorted
                current_text = distorted

            except Exception as e:
                step_trace["status"] = "failed"
                step_trace["error"] = str(e)

            steps_trace.append(step_trace)

        return {
            "original": text,
            "final": current_text,
            "steps": steps_trace,
            "seed": effective_seed,
        }
