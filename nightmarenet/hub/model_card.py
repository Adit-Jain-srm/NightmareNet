import os
import pathlib
import platform
from typing import Any, Dict

import yaml
from huggingface_hub import ModelCard, ModelCardData

DEFAULT_TEMPLATE = """
---
{{ card_data }}
---

# NightmareNet Hardened Model: `{{ repo_id }}`

This model has been robustified using the NightmareNet framework.

## Model Training & Resilience Profile
* **Robustness Score:** {{ robustness_score }}
* **Cycle Details:** Loop execution phase successfully completed (Cycle count: {{ cycle_count }}).
* **Distortion Vectors Defended:** {{ distortion_families }}

## Hardware Information
* **System:** {{ system }}
* **Processor:** {{ processor }}
* **CPU Cores:** {{ cpu_cores }}
* **Memory:** {{ memory }}

## Reproducibility Metadata Configuration
```yaml
{{ config_yaml }}
```
"""

def generate_model_card(repo_id: str, metadata: Dict[str, Any]) -> str:
    """
    Auto-generates an academic/robustness HuggingFace Model Card (README.md)
    complete with standardized frontmatter metadata tags.
    """
    # Extract tags and metric for ModelCardData
    robustness_score = metadata.get("robustness_score", 0.0)


    # Custom tags for nightmarenet
    custom_tags = {}
    for key in ["cycle_count", "final_robustness_score", "distortion_families"]:
        if key in metadata:
            custom_tags[f"nightmarenet_{key}"] = metadata[key]

    card_data = ModelCardData(
        language="en",
        tags=["nightmarenet", "robustness", "adversarial-defense"],
        library_name="transformers",
        pipeline_tag="text-classification",
        metrics=["robustness_score"],
        model_index=[
            {
                "name": repo_id.split("/")[-1],
                "results": [
                    {
                        "task": {"type": "text-classification"},
                        "dataset": {"name": "robustness-evaluation", "type": "evaluation"},
                        "metrics": [
                            {
                                "type": "robustness_score",
                                "value": robustness_score,
                                "name": "Robustness Score",
                            }
                        ],
                    }
                ],
            }
        ],
        **custom_tags
    )

    # Try to extract template from config
    config = metadata.get("config") or {}
    hub_cfg = config.get("hub") or {}
    template_path = hub_cfg.get("model_card_template")

    # System info
    try:
        sys_info = platform.system() + " " + platform.release()
        processor = platform.processor()
        cpu_cores = str(os.cpu_count())
        memory = "N/A"
    except Exception:
        sys_info = "Unknown"
        processor = "Unknown"
        cpu_cores = "Unknown"
        memory = "Unknown"

    kwargs = {
        "repo_id": repo_id,
        "robustness_score": metadata.get("robustness_score", "N/A"),
        "cycle_count": metadata.get("cycle_count", "N/A"),
        "distortion_families": ", ".join(metadata.get("distortion_families", ["None"])),
        "config_yaml": yaml.safe_dump(config, default_flow_style=False) if config else "{}",
        "system": sys_info,
        "processor": processor,
        "cpu_cores": cpu_cores,
        "memory": memory,
    }

    if template_path:
        resolved_path = pathlib.Path(template_path).resolve()
        project_dir = pathlib.Path.cwd().resolve()

        try:
            resolved_path.relative_to(project_dir)
        except ValueError:
            raise ValueError(
                f"Template path escapes the project directory: {template_path}"
            ) from None

        with open(resolved_path, encoding="utf-8") as f:
            template_content = f.read()
    else:
        template_content = DEFAULT_TEMPLATE

    card = ModelCard.from_template(
        card_data,
        template_str=template_content,
        **kwargs
    )

    return str(card)
