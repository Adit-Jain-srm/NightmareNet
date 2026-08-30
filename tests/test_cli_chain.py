import pytest
import sys
from io import StringIO
from unittest.mock import patch
from nightmarenet.cli import build_parser, main
from nightmarenet.exceptions import DSLSyntaxError

def test_chain_and_config_mutually_exclusive(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["distort", "--chain", "typo(0.3)", "--config", "test.yaml", "--text", "hello"])
    captured = capsys.readouterr()
    assert "not allowed with argument" in captured.err

@patch("nightmarenet.cli.parse_dsl_expression")
@patch("nightmarenet.distortions.dsl.executor.ChainExecutor.execute")
def test_chain_parsing_and_execution(mock_execute, mock_parse_dsl, capsys):
    mock_parse_dsl.return_value = []
    mock_execute.return_value = "hxllo"

    test_args = ["nightmarenet", "distort", "--chain", "typo(0.3)", "--text", "hello"]
    with patch.object(sys, "argv", test_args):
        ret = main()
        assert ret == 0
        captured = capsys.readouterr()
        assert "Distorted: hxllo" in captured.out
        mock_parse_dsl.assert_called_once_with("typo(0.3)", validate_engines=True)
        mock_execute.assert_called_once()

@patch("nightmarenet.cli.parse_dsl_expression")
def test_chain_invalid_syntax_error(mock_parse_dsl, capsys):
    mock_parse_dsl.side_effect = DSLSyntaxError("Invalid syntax message")

    test_args = ["nightmarenet", "distort", "--chain", "invalid_chain()", "--text", "hello"]
    with patch.object(sys, "argv", test_args):
        ret = main()
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: Invalid syntax message" in captured.err
