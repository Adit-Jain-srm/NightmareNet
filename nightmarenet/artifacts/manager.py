import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def parse_duration(duration_str: str) -> float:
    """Parse a duration string (e.g. '30d', '12h', '45m', '60s') into seconds.

    If no unit suffix is found, treats the entire string as seconds.
    """
    if not duration_str:
        return 0.0

    duration_str = duration_str.strip()
    unit = duration_str[-1].lower()
    value_str = duration_str[:-1]

    try:
        if unit == "d":
            return float(value_str) * 24 * 3600
        elif unit == "h":
            return float(value_str) * 3600
        elif unit == "m":
            return float(value_str) * 60
        elif unit == "s":
            return float(value_str)
        else:
            return float(duration_str)
    except ValueError as e:
        raise ValueError(f"Invalid duration format: '{duration_str}'") from e


class ArtifactManager:
    """Central registry manager for pipeline artifacts and retention policies.

    Artifact metadata is stored in sidecar '.artifact-meta.json' files next to
    the actual files or directories, maintaining a decentralized registry.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        if root_dir is None:
            self.root_dir = Path.cwd()
        else:
            self.root_dir = Path(root_dir)

    def register(
        self,
        path: Union[str, Path],
        run_id: str,
        artifact_type: str,
        retention_policy: Optional[dict] = None,
    ) -> Path:
        """Register a file or directory as an artifact.

        Creates a sidecar metadata file: `<artifact_path>.artifact-meta.json`.

        Returns:
            The Path to the created sidecar metadata file.
        """
        artifact_path = Path(path)
        # Handle absolute path conversion to relative for portability
        try:
            rel_path = artifact_path.relative_to(self.root_dir)
        except ValueError:
            # If not under root_dir, store as-is or make relative
            if artifact_path.is_absolute():
                rel_path = artifact_path
            else:
                rel_path = artifact_path

        # Calculate size in bytes
        size_bytes = 0
        if artifact_path.exists():
            if artifact_path.is_file():
                size_bytes = artifact_path.stat().st_size
            elif artifact_path.is_dir():
                for root, _, files in os.walk(artifact_path):
                    for file in files:
                        p = Path(root) / file
                        if p.exists() and not p.is_symlink():
                            size_bytes += p.stat().st_size

        # Apply default policy if none specified
        if retention_policy is None:
            retention_policy = {"type": "TIME_BASED", "duration": "30d"}
        elif "type" not in retention_policy:
            retention_policy["type"] = "TIME_BASED"

        if retention_policy["type"] == "TIME_BASED" and "duration" not in retention_policy:
            retention_policy["duration"] = "30d"

        metadata = {
            "artifact_path": str(rel_path.as_posix()),
            "creation_time": datetime.now(timezone.utc).isoformat() + "Z",
            "run_id": run_id,
            "size_bytes": size_bytes,
            "artifact_type": artifact_type,
            "retention_policy": retention_policy,
        }

        # Sidecar file sits next to the artifact
        meta_path = Path(f"{str(artifact_path)}.artifact-meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Registered artifact at %s (type=%s)", rel_path, artifact_type)
        return meta_path

    def list_artifacts(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find and parse all registered artifact sidecar metadata files.

        Optionally filtered by run_id. Returns list of metadata dicts, sorted
        newest first.
        """
        exclude_dirs = {
            ".git",
            "node_modules",
            ".venv",
            ".venv312",
            "__pycache__",
            ".pytest_cache",
            ".nightmarenet_cache",
        }
        artifacts = []

        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith(".artifact-meta.json"):
                    meta_path = Path(root) / file
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            meta = json.load(f)

                        # Enforce schema validity
                        if not all(
                            k in meta
                            for k in (
                                "artifact_path",
                                "creation_time",
                                "run_id",
                                "artifact_type",
                                "retention_policy",
                            )
                        ):
                            continue

                        meta["absolute_metadata_path"] = meta_path
                        # Resolve absolute path to the actual artifact
                        rel_path = Path(meta["artifact_path"])
                        if rel_path.is_absolute():
                            meta["absolute_artifact_path"] = rel_path
                        else:
                            meta["absolute_artifact_path"] = self.root_dir / rel_path

                        if run_id is None or meta.get("run_id") == run_id:
                            artifacts.append(meta)
                    except Exception as e:
                        logger.warning("Failed to parse metadata file %s: %s", meta_path, e)

        # Sort newest first
        artifacts.sort(key=lambda x: x.get("creation_time", ""), reverse=True)
        return artifacts

    def clean(self, older_than: Optional[str] = None) -> List[Path]:
        """Clean up artifacts that violate retention policies or older_than threshold.

        Removes both the actual artifact file/directory and its metadata sidecar.

        Returns:
            List of Paths of deleted artifacts.
        """
        all_artifacts = self.list_artifacts()
        deleted_paths = []
        now = datetime.now(timezone.utc)

        # Override cutoff
        cutoff_time = None
        if older_than:
            seconds = parse_duration(older_than)
            cutoff_time = now - timedelta(seconds=seconds)

        # Group by artifact_type to apply COUNT_BASED globally per type
        type_groups: Dict[str, List[Dict[str, Any]]] = {}
        for art in all_artifacts:
            art_type = art.get("artifact_type", "unknown")
            type_groups.setdefault(art_type, []).append(art)

        to_delete = []

        for _art_type, arts in type_groups.items():
            # Ensure they are sorted newest first
            arts.sort(key=lambda x: x.get("creation_time", ""), reverse=True)

            count_seen = 0
            for art in arts:
                meta_path: Path = art["absolute_metadata_path"]
                art_path: Path = art["absolute_artifact_path"]
                creation_str = art.get("creation_time", "")

                try:
                    # Strip Z and parse as UTC datetime
                    creation_time = datetime.fromisoformat(
                        creation_str.rstrip("Z")
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    # Fallback to filesystem ctime if iso format is corrupted
                    creation_time = datetime.fromtimestamp(
                        meta_path.stat().st_ctime, tz=timezone.utc
                    )

                policy = art.get("retention_policy", {})
                policy_type = policy.get("type", "TIME_BASED")

                should_delete = False
                reason = ""

                # 1. CLI threshold override takes precedence
                if cutoff_time and creation_time < cutoff_time:
                    should_delete = True
                    reason = f"older than clean threshold '{older_than}'"

                # 2. Check individual policy
                if not should_delete:
                    if policy_type == "TIME_BASED":
                        duration_str = policy.get("duration", "30d")
                        policy_seconds = parse_duration(duration_str)
                        policy_cutoff = now - timedelta(seconds=policy_seconds)
                        if creation_time < policy_cutoff:
                            should_delete = True
                            reason = f"violated TIME_BASED policy ({duration_str})"

                    elif policy_type == "COUNT_BASED":
                        limit = int(policy.get("count", 5))
                        count_seen += 1
                        if count_seen > limit:
                            should_delete = True
                            reason = (
                                f"violated COUNT_BASED policy (kept {limit} newest, "
                                f"current index={count_seen})"
                            )

                    elif policy_type == "KEEP_FOREVER":
                        # Checked by global cutoff, otherwise retained
                        pass

                if should_delete:
                    to_delete.append((art_path, meta_path, reason))

        # Perform actual deletions
        for art_path, meta_path, reason in to_delete:
            if art_path.exists():
                try:
                    if art_path.is_file():
                        art_path.unlink()
                    elif art_path.is_dir():
                        import shutil
                        shutil.rmtree(art_path)
                    logger.info("Cleaned artifact: %s (%s)", art_path, reason)
                    deleted_paths.append(art_path)
                except Exception as e:
                    logger.error("Failed to clean artifact at %s: %s", art_path, e)

            if meta_path.exists():
                try:
                    meta_path.unlink()
                except Exception as e:
                    logger.error("Failed to delete sidecar metadata at %s: %s", meta_path, e)

        return deleted_paths
