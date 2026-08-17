from typing import Optional

import torch

import nightmarenet.cli as cli


def test_benchmark_command_runs_with_tiny_model(tmp_path, monkeypatch, capsys) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(10, 4)

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return self.linear(input_ids.float())

    model = TinyModel()

    class TinyTokenizer:
        pad_token: Optional[str] = "<pad>"
        eos_token: Optional[str] = "</s>"

        def __call__(
            self,
            texts,
            padding=None,
            truncation=None,
            max_length=None,
            return_tensors=None,
        ):
            batch_size = len(texts) if isinstance(texts, list) else 1
            seq_len = max_length or 10
            return {"input_ids": torch.randint(0, 100, (batch_size, seq_len))}

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda _: model,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda _: TinyTokenizer(),
    )

    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """
model:
  name: tiny-model
  type: causal_lm
  max_length: 10
  device: cpu
""",
        encoding="utf-8",
    )

    result = cli.main(
        [
            "benchmark",
            "--model",
            "tiny-model",
            "--config",
            str(config),
            "--batch-sizes",
            "1,2",
            "--output",
            str(tmp_path),
        ]
    )

    assert result == 0

    output = capsys.readouterr().out
    assert "Inference Benchmark Results" in output
    assert "Latency" in output
    assert "Throughput" in output
    assert "Peak Memory" in output

    result_files = list(tmp_path.glob("benchmark_tiny-model_*.json"))
    assert len(result_files) == 1
