"""Preset loader for discovering and loading distortion chain presets."""

from pathlib import Path
from typing import List, Optional

from nightmarenet.distortions.dsl.parser import parse_chain_config
from nightmarenet.distortions.dsl.schema import ChainConfig

# Default preset directories
DEFAULT_PRESET_DIRS = [
    Path("distortions/presets"),
    Path(__file__).parent.parent.parent.parent / "distortions" / "presets",
]


def get_preset_dirs() -> List[Path]:
    """Get list of directories to search for preset files.

    Returns:
        List of directory paths that may contain preset YAML files
    """
    dirs = []
    for dir_path in DEFAULT_PRESET_DIRS:
        if dir_path.exists():
            dirs.append(dir_path)
    return dirs


def discover_presets(preset_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Discover all preset YAML files in the preset directories.

    Args:
        preset_dirs: Optional list of directories to search. If None, uses default dirs.

    Returns:
        List of paths to preset YAML files
    """
    if preset_dirs is None:
        preset_dirs = get_preset_dirs()

    presets = []
    for dir_path in preset_dirs:
        if not dir_path.exists():
            continue
        for yaml_file in dir_path.glob("*.yaml"):
            presets.append(yaml_file)
        for yaml_file in dir_path.glob("*.yml"):
            presets.append(yaml_file)

    return sorted(presets)


def list_presets(preset_dirs: Optional[List[Path]] = None) -> List[dict]:
    """List all available presets with metadata.

    Args:
        preset_dirs: Optional list of directories to search. If None, uses default dirs.

    Returns:
        List of dictionaries with preset metadata (name, description, path, version)
    """
    preset_files = discover_presets(preset_dirs)
    presets_info = []

    for preset_path in preset_files:
        try:
            config = parse_chain_config(preset_path, validate_engines=False)
            presets_info.append(
                {
                    "name": config.name,
                    "description": config.description or "",
                    "path": str(preset_path),
                    "version": config.version,
                    "num_steps": len(config.chain),
                }
            )
        except Exception:
            # Skip invalid presets when listing
            continue

    return presets_info


def load_preset(name: str, preset_dirs: Optional[List[Path]] = None) -> ChainConfig:
    """Load a preset by name.

    Args:
        name: Name of the preset to load (without .yaml extension)
        preset_dirs: Optional list of directories to search. If None, uses default dirs.

    Returns:
        ChainConfig object

    Raises:
        FileNotFoundError: If preset file not found
        ValueError: If preset fails validation
    """
    if preset_dirs is None:
        preset_dirs = get_preset_dirs()

    # Try to find the preset file
    for dir_path in preset_dirs:
        if not dir_path.exists():
            continue

        # Try with .yaml extension
        yaml_path = dir_path / f"{name}.yaml"
        if yaml_path.exists():
            return parse_chain_config(yaml_path, validate_engines=True)

        # Try with .yml extension
        yml_path = dir_path / f"{name}.yml"
        if yml_path.exists():
            return parse_chain_config(yml_path, validate_engines=True)

    # Not found
    available = [p.stem for p in discover_presets(preset_dirs)]
    raise FileNotFoundError(f"Preset '{name}' not found. Available presets: {', '.join(available)}")


def load_preset_from_path(path: str) -> ChainConfig:
    """Load a preset from a specific file path.

    Args:
        path: Path to the preset YAML file

    Returns:
        ChainConfig object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If preset fails validation
    """
    return parse_chain_config(path, validate_engines=True)
