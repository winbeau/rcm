"""Versioned metadata and integrity helpers for rCM attention artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "rcm.attention.v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: str | Path) -> str | None:
    """Return the current repository revision, or None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_worktree_state(repo: str | Path) -> dict[str, Any] | None:
    """Return a reproducible summary of tracked and untracked source changes."""
    repo = Path(repo)
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        tracked_diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        untracked_raw = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

    untracked = []
    for raw_path in untracked_raw.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        path = repo / relative
        untracked.append(
            {
                "path": relative,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {
        "dirty": bool(status),
        "status_porcelain": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_files": untracked,
    }


def validate_run_metadata(metadata: dict[str, Any]) -> None:
    """Fail fast when a capture manifest is missing protocol-critical fields."""
    required = {
        "schema_version",
        "model",
        "model_size",
        "num_layers",
        "num_heads",
        "layer_indices",
        "prompt",
        "prompt_sha256",
        "seed",
        "sample_count",
        "pixel_geometry",
        "latent_geometry",
        "frame_seq_length",
        "num_frames_latent",
        "capture_protocol",
        "capture_pass",
        "full_horizon",
        "dtype",
        "command_argv",
        "checkpoint",
        "repository",
        "runtime",
        "verification",
        "coverage",
        "artifacts",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"attention run metadata missing required fields: {missing}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported attention schema {metadata['schema_version']!r}")
    if int(metadata["sample_count"]) != 1:
        raise ValueError("attention capture metadata requires sample_count=1")
    layer_indices = [int(layer) for layer in metadata["layer_indices"]]
    if not layer_indices:
        raise ValueError("attention capture metadata must name at least one layer")
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("attention capture metadata contains duplicate layer indices")
    if min(layer_indices) < 0 or max(layer_indices) >= int(metadata["num_layers"]):
        raise ValueError("attention capture metadata contains an out-of-range layer")
    if int(metadata["num_heads"]) <= 0:
        raise ValueError("attention capture metadata requires num_heads > 0")
    if int(metadata["num_frames_latent"]) <= 0:
        raise ValueError("attention capture metadata requires num_frames_latent > 0")
    if int(metadata["frame_seq_length"]) <= 0:
        raise ValueError("attention capture metadata requires frame_seq_length > 0")
    capture_protocol = metadata["capture_protocol"]
    if not isinstance(capture_protocol, str) or not capture_protocol.strip():
        raise ValueError("attention capture metadata requires a non-empty capture_protocol")
    if not metadata["full_horizon"]:
        raise ValueError("attention capture metadata must declare full_horizon=true")
    if metadata["capture_pass"] not in {"append", "denoise0"}:
        raise ValueError(f"unsupported capture pass {metadata['capture_pass']!r}")
    if not metadata["command_argv"]:
        raise ValueError("attention capture metadata must preserve command_argv")
    checkpoint_sha = str(metadata["checkpoint"].get("sha256", ""))
    if len(checkpoint_sha) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha.lower()):
        raise ValueError("attention capture metadata requires a 64-character checkpoint SHA-256")
    if not metadata["repository"].get("revision"):
        raise ValueError("attention capture metadata requires a repository revision")
    if metadata["verification"].get("enabled") and not metadata["verification"].get("stats"):
        raise ValueError("enabled attention verification requires recorded error statistics")
    if not metadata["coverage"].get("complete", False):
        raise ValueError("attention capture metadata coverage is not complete")
    coverage_layers = metadata["coverage"].get("layers", {})
    if set(coverage_layers) != {str(layer) for layer in layer_indices}:
        raise ValueError("attention coverage layers do not match layer_indices")
    artifacts = metadata["artifacts"]
    if {int(row["layer_index"]) for row in artifacts} != set(layer_indices):
        raise ValueError("attention artifacts do not match layer_indices")
    expected_shape = [
        int(metadata["num_heads"]),
        int(metadata["num_frames_latent"]),
        int(metadata["num_frames_latent"]),
    ]
    for artifact in artifacts:
        digest = str(artifact.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("each attention artifact requires a valid SHA-256")
        if list(artifact.get("shape", [])) != expected_shape:
            raise ValueError(
                f"attention artifact shape {artifact.get('shape')} != {expected_shape}"
            )


def validate_run_directory(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a run manifest and every referenced artifact digest."""
    manifest_path = Path(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_run_metadata(metadata)
    for artifact in metadata["artifacts"]:
        artifact_path = manifest_path.parent / artifact["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing attention artifact: {artifact_path}")
        observed = sha256_file(artifact_path)
        if observed != artifact["sha256"]:
            raise ValueError(
                f"attention artifact digest mismatch for {artifact_path}: "
                f"expected {artifact['sha256']}, got {observed}"
            )
    return metadata


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an rCM attention run")
    parser.add_argument("manifest", help="path to run.json")
    args = parser.parse_args()
    metadata = validate_run_directory(args.manifest)
    print(
        f"valid {metadata['capture_protocol']}: "
        f"{len(metadata['artifacts'])} artifacts"
    )


if __name__ == "__main__":
    main()
