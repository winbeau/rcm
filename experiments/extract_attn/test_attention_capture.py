"""CPU gates for rCM attention capture artifacts and coverage."""
from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from rcm.utils.attention_artifact import (
    SCHEMA_VERSION,
    sha256_file,
    validate_run_directory,
    validate_run_metadata,
    write_json,
)
from rcm.utils.frame_attention import (
    FrameAttentionObserver,
    frame_logits_naive,
    frame_logits_pooled,
)


def _attention_pair(*, query_frames: int, key_frames: int):
    generator = torch.Generator().manual_seed(17 + query_frames + key_frames)
    query = torch.randn(1, query_frames * 4, 2, 5, generator=generator)
    key = torch.randn(1, key_frames * 4, 2, 5, generator=generator)
    return query, key


def test_pooled_frame_logits_match_naive():
    query, key = _attention_pair(query_frames=2, key_frames=3)
    scale = query.shape[-1] ** -0.5

    pooled = frame_logits_pooled(query, key, frame_tokens=4, scale=scale)
    naive = frame_logits_naive(query, key, frame_tokens=4, scale=scale)

    torch.testing.assert_close(pooled, naive, rtol=1e-5, atol=1e-6)


def test_verification_runs_once_and_coverage_is_per_layer(monkeypatch):
    observer = FrameAttentionObserver(
        layer_indices=[0, 2],
        frame_tokens=4,
        num_heads=2,
        num_frames=2,
        verify_once=True,
    )
    original = observer.verify_against_naive
    calls = []

    def counted(query, key, *args, **kwargs):
        calls.append((query.shape, key.shape))
        return original(query, key, *args, **kwargs)

    monkeypatch.setattr(observer, "verify_against_naive", counted)
    for layer in [0, 2]:
        query, key = _attention_pair(query_frames=1, key_frames=1)
        observer.observe(layer, 0, query, key, cached_len=0)
        query, key = _attention_pair(query_frames=1, key_frames=2)
        observer.observe(layer, 1, query, key, cached_len=4)

    coverage = observer.coverage(expected_blocks=2)

    assert len(calls) == 1
    assert observer.verification_stats is not None
    assert coverage["complete"] is True
    assert set(coverage["layers"]) == {"0", "2"}
    assert coverage["layers"]["2"]["final_key_frame_end"] == 2


def test_duplicate_observation_hard_fails():
    observer = FrameAttentionObserver([0], 4, 2, 1)
    query, key = _attention_pair(query_frames=1, key_frames=1)
    observer.observe(0, 0, query, key, cached_len=0)

    with pytest.raises(ValueError, match="duplicate attention observation"):
        observer.observe(0, 0, query, key, cached_len=0)


def test_coverage_rejects_missing_layer_block():
    observer = FrameAttentionObserver([0, 1], 4, 2, 2)
    for block, key_frames in [(0, 1), (1, 2)]:
        query, key = _attention_pair(query_frames=1, key_frames=key_frames)
        observer.observe(0, block, query, key, cached_len=(key_frames - 1) * 4)

    with pytest.raises(RuntimeError, match="incomplete attention coverage"):
        observer.coverage(expected_blocks=2)


def test_coverage_rejects_noncausal_key_horizon():
    observer = FrameAttentionObserver([0], 4, 2, 2)
    query, key = _attention_pair(query_frames=1, key_frames=2)
    observer.observe(0, 0, query, key, cached_len=4)
    query, key = _attention_pair(query_frames=1, key_frames=2)
    observer.observe(0, 1, query, key, cached_len=4)

    with pytest.raises(RuntimeError, match="invalid_horizons"):
        observer.coverage(expected_blocks=2)


def _valid_metadata(tmp_path):
    artifact_path = tmp_path / "layer0.pt"
    artifact_path.write_bytes(b"attention-artifact")
    prompt = "A test prompt"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "capture_protocol": "rcm2-denoise0-v1",
        "model": "Causal-rCM Wan2.1 T2V",
        "model_size": "1.3B",
        "num_layers": 2,
        "num_heads": 2,
        "layer_indices": [0],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed": 0,
        "sample_count": 1,
        "pixel_geometry": {"width": 32, "height": 16},
        "latent_geometry": {
            "frames": 2,
            "height": 4,
            "width": 8,
            "patchified_height": 2,
            "patchified_width": 4,
        },
        "frame_seq_length": 8,
        "num_frames_latent": 2,
        "capture_pass": "denoise0",
        "full_horizon": True,
        "dtype": {"model": "bfloat16", "artifact": "float16"},
        "command_argv": ["python", "capture.py"],
        "checkpoint": {"path": "/checkpoint.pt", "sha256": "a" * 64},
        "repository": {"path": "/repo", "revision": "deadbeef"},
        "runtime": {"python": "3.12", "torch": "2.9.1"},
        "verification": {"enabled": False, "stats": None},
        "coverage": {
            "expected_blocks": 2,
            "complete": True,
            "layers": {
                "0": {
                    "observed_blocks": [0, 1],
                    "missing_blocks": [],
                    "extra_blocks": [],
                    "invalid_horizons": [],
                    "final_key_frame_end": 2,
                    "tensor_shape": [2, 2, 2],
                    "complete": True,
                }
            },
        },
        "artifacts": [
            {
                "layer_index": 0,
                "path": artifact_path.name,
                "sha256": sha256_file(artifact_path),
                "shape": [2, 2, 2],
                "dtype": "float16",
            }
        ],
    }
    return metadata, artifact_path


def test_run_metadata_and_artifact_hash_validate(tmp_path):
    metadata, _ = _valid_metadata(tmp_path)
    manifest = tmp_path / "run.json"
    validate_run_metadata(metadata)
    write_json(manifest, metadata)

    loaded = validate_run_directory(manifest)

    assert loaded["capture_protocol"] == "rcm2-denoise0-v1"


def test_run_directory_rejects_artifact_digest_mismatch(tmp_path):
    metadata, artifact_path = _valid_metadata(tmp_path)
    manifest = tmp_path / "run.json"
    write_json(manifest, metadata)
    artifact_path.write_bytes(b"modified")

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_run_directory(manifest)


def test_metadata_requires_capture_protocol(tmp_path):
    metadata, _ = _valid_metadata(tmp_path)
    missing = copy.deepcopy(metadata)
    del missing["capture_protocol"]
    with pytest.raises(ValueError, match="missing required fields.*capture_protocol"):
        validate_run_metadata(missing)

    empty = copy.deepcopy(metadata)
    empty["capture_protocol"] = ""
    with pytest.raises(ValueError, match="non-empty capture_protocol"):
        validate_run_metadata(empty)


def test_metadata_rejects_incomplete_or_unverified_capture(tmp_path):
    metadata, _ = _valid_metadata(tmp_path)
    incomplete = copy.deepcopy(metadata)
    incomplete["coverage"]["complete"] = False
    with pytest.raises(ValueError, match="coverage is not complete"):
        validate_run_metadata(incomplete)

    unverified = copy.deepcopy(metadata)
    unverified["verification"] = {"enabled": True, "stats": None}
    with pytest.raises(ValueError, match="requires recorded error statistics"):
        validate_run_metadata(unverified)
