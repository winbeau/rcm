#!/usr/bin/env python
"""Add the key `jupyter-classify` reads to a tree of rCM attention artifacts.

`jupyter-classify` loads `payload["last_frame_attention_per_head"]` with no
fallback (`classifier/core.py:18`, `classifier/labeling.py:44`,
`classifier/batch_process.py:45`); the rCM extractor writes
`last_block_frame_attention` and `full_frame_attention`. This bridges the two.

**The pooling choice is deliberate, not cosmetic.** The key's name says "last
frame", but the artifact it is usually paired with stores the *mean over the
last block's query rows*. Those coincide only when the block is one frame:

    --chunk_t 1  (c1-1)   block_sizes [1]*72   -> last block == last frame
    --chunk_t 3  (c3-3)   block_sizes [3]*24   -> last block == frames 69,70,71

The classifier's first two cascade steps threshold `pos_rate = mean(a > 0)` on
this vector, and averaging three query rows pulls values toward zero, so the two
choices do not give the same labels. Which one produced the Self-Forcing
artifacts behind `best_labels.csv` is an open provenance question, so this
script refuses to guess: pass `--pooling` explicitly and it records the choice
in the artifact under `last_frame_pooling`.

    python experiments/extract_attn/adapt_for_classifier.py \
        --src cache/attn/mgvb128-c33 --dst cache/attn/mgvb128-c33-classify \
        --pooling last_frame
"""

import argparse
import shutil
from pathlib import Path

import torch

from rcm.utils.attention_artifact import (
    sha256_file,
    validate_run_directory,
    write_json,
)


def adapt_payload(payload: dict, pooling: str) -> dict:
    """Return a copy with `last_frame_attention_per_head` added."""
    full = payload.get("full_frame_attention")
    if full is None:
        raise KeyError("artifact has no full_frame_attention; cannot derive the last-frame row")
    full = full.float()  # [heads, F_q, F_k]

    if pooling == "last_frame":
        per_head = full[:, -1, :]
    elif pooling == "last_block_mean":
        block_sizes = payload.get("block_sizes")
        if not block_sizes:
            raise KeyError("artifact has no block_sizes; cannot locate the last block")
        start = sum(block_sizes[:-1])
        per_head = full[:, start:, :].mean(dim=1)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")

    out = dict(payload)
    out["last_frame_attention_per_head"] = per_head.to(torch.float16)
    out["last_frame_pooling"] = pooling
    # Record what the row actually spans, so a later reader does not have to
    # re-derive it from block_sizes.
    out["last_frame_source_query_frames"] = (
        [int(full.shape[1]) - 1] if pooling == "last_frame" else list(range(sum(payload["block_sizes"][:-1]), int(full.shape[1])))
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="tree of run_<NNN>/layer<N>.pt")
    ap.add_argument("--dst", required=True, help="output tree (created; not written in place)")
    ap.add_argument(
        "--pooling",
        required=True,
        choices=["last_frame", "last_block_mean"],
        help="which query row(s) become last_frame_attention_per_head; see the module docstring",
    )
    ap.add_argument("--overwrite", action="store_true", help="replace --dst if it exists")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"no such directory: {src}")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)

    runs = sorted(p for p in src.glob("run_*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run_* directories under {src}")

    n_files = 0
    for run in runs:
        source_manifest = run / "run.json"
        metadata = validate_run_directory(source_manifest)
        out_run = dst / run.name
        out_run.mkdir(parents=True, exist_ok=True)
        output_artifacts = []
        source_query_frames = {}
        for artifact in metadata["artifacts"]:
            pt = run / artifact["path"]
            payload = torch.load(pt, map_location="cpu", weights_only=False)
            out_path = out_run / pt.name
            adapted = adapt_payload(payload, args.pooling)
            torch.save(adapted, out_path)
            output_artifacts.append(
                {
                    **artifact,
                    "path": out_path.name,
                    "sha256": sha256_file(out_path),
                }
            )
            source_query_frames[str(artifact["layer_index"])] = adapted[
                "last_frame_source_query_frames"
            ]
            n_files += 1

        output_metadata = dict(metadata)
        output_metadata["artifacts"] = output_artifacts
        output_metadata["classifier_adapter"] = {
            "pooling": args.pooling,
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_query_frames": source_query_frames,
        }
        write_json(out_run / "run.json", output_metadata)
        validate_run_directory(out_run / "run.json")
        print(f"  {run.name}: {len(output_artifacts)} layers", flush=True)

    print(f"\nadapted {n_files} artifacts across {len(runs)} runs -> {dst}")
    print(f"pooling = {args.pooling}")

    sample = torch.load(next((dst / runs[0].name).glob("layer*.pt")), map_location="cpu", weights_only=False)
    v = sample["last_frame_attention_per_head"].float()
    print(f"sample shape {tuple(v.shape)}  range [{v.min():.4f}, {v.max():.4f}]")
    print(f"pos_rate per head (mean over frames): min {(v > 0).float().mean(1).min():.3f} max {(v > 0).float().mean(1).max():.3f}")


if __name__ == "__main__":
    main()
