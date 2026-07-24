"""Property-based and fuzzing tests for distortion DSL parser.

Covers:
- Hypothesis roundtrip invariant: parse(format(x)) == x
- Fuzz testing with random text and Unicode strings (no unhandled crashes)
- Explicit edge cases: empty input, whitespace-only, single element, max chain length
- Specific exception handling via DSLSyntaxError
- Fast execution (<30s tuned Hypothesis settings)
"""

from __future__ import annotations

import time

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from nightmarenet.distortions.dsl.parser import format_dsl_expression, parse_dsl_expression
from nightmarenet.distortions.dsl.schema import ChainStep
from nightmarenet.exceptions import DSLSyntaxError, NightmareNetError

# Strategy for generating valid engine names
engine_name_st = st.from_regex(r"^[a-zA-Z][a-zA-Z0-9_]{0,14}$", fullmatch=True)

# Strategy for generating valid ChainStep objects
chain_step_st = st.builds(
    ChainStep,
    engine=engine_name_st,
    strength=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    condition=st.just("always"),
)

# Strategy for valid lists of ChainSteps (1 to 15 steps)
chain_steps_list_st = st.lists(chain_step_st, min_size=1, max_size=15)


class TestDSLParserProperties:
    """Hypothesis property-based tests for DSL parser and formatter."""

    @given(steps=chain_steps_list_st)
    @settings(max_examples=100, deadline=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_invariant(self, steps: list[ChainStep]):
        """Invariant: parse(format(x)) == x for valid ChainSteps."""
        formatted = format_dsl_expression(steps)
        parsed = parse_dsl_expression(formatted)

        assert len(parsed) == len(steps)
        for orig, p in zip(steps, parsed):
            assert orig.engine == p.engine
            assert abs(orig.strength - p.strength) <= 1e-3
            assert p.condition == "always"

    @given(text=st.text())
    @settings(max_examples=200, deadline=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_fuzz_arbitrary_unicode(self, text: str):
        """Fuzz parser with arbitrary Unicode strings; must not crash unexpectedly."""
        try:
            result = parse_dsl_expression(text)
            assert isinstance(result, list)
            for step in result:
                assert isinstance(step, ChainStep)
                assert 0.0 <= step.strength <= 1.0
        except (DSLSyntaxError, ValueError):
            pass


class TestDSLParserEdgeCases:
    """Explicit edge case tests for distortion DSL parser."""

    def test_empty_string_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="empty"):
            parse_dsl_expression("")

    def test_whitespace_only_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="empty"):
            parse_dsl_expression("   \t\n  ")

    def test_non_string_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="must be a string"):
            parse_dsl_expression(12345)  # type: ignore[arg-type]

    def test_single_distortion_default_strength(self):
        steps = parse_dsl_expression("typo")
        assert len(steps) == 1
        assert steps[0].engine == "typo"
        assert steps[0].strength == 0.5

    def test_single_distortion_custom_strength(self):
        steps = parse_dsl_expression("swap(0.35)")
        assert len(steps) == 1
        assert steps[0].engine == "swap"
        assert steps[0].strength == 0.35

    def test_multi_step_chain_parsing(self):
        expression = "typo(0.3) -> swap(0.1) -> delete(0.05)"
        steps = parse_dsl_expression(expression)
        assert len(steps) == 3
        assert [s.engine for s in steps] == ["typo", "swap", "delete"]
        assert [s.strength for s in steps] == [0.3, 0.1, 0.05]

    def test_max_chain_length_exceeded(self):
        too_long = " -> ".join([f"engine{i}(0.1)" for i in range(105)])
        with pytest.raises(DSLSyntaxError, match="exceeds maximum allowed steps"):
            parse_dsl_expression(too_long)

    def test_empty_step_in_chain(self):
        with pytest.raises(DSLSyntaxError, match="Empty step"):
            parse_dsl_expression("typo(0.3) -> -> swap(0.1)")

    def test_trailing_arrow_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="Empty step"):
            parse_dsl_expression("typo(0.3) ->")

    def test_leading_arrow_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="Empty step"):
            parse_dsl_expression("-> typo(0.3)")

    def test_nested_parentheses_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="Nested or mismatched"):
            parse_dsl_expression("typo(((0.5)))")

    def test_unclosed_parenthesis_raises_syntax_error(self):
        with pytest.raises(DSLSyntaxError, match="Unclosed parenthesis"):
            parse_dsl_expression("typo(0.5")

    def test_strength_out_of_range_high(self):
        with pytest.raises(DSLSyntaxError, match="must be in range"):
            parse_dsl_expression("typo(1.5)")

    def test_strength_out_of_range_low(self):
        with pytest.raises(DSLSyntaxError, match="must be in range"):
            parse_dsl_expression("typo(-0.2)")

    def test_non_numeric_strength(self):
        with pytest.raises(DSLSyntaxError, match="Invalid strength value"):
            parse_dsl_expression("typo(abc)")

    def test_dsl_syntax_error_inherits_value_error_and_nightmarenet_error(self):
        err = DSLSyntaxError("Invalid DSL syntax")
        assert isinstance(err, ValueError)
        assert isinstance(err, NightmareNetError)

    def test_test_suite_execution_speed(self):
        start = time.time()
        parse_dsl_expression("typo(0.3) -> swap(0.1)")
        duration = time.time() - start
        assert duration < 1.0
