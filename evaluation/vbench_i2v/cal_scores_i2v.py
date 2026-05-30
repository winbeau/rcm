#!/usr/bin/env python3
"""Compute VBench-I2V final scores from an evaluation results directory.

Reads all ``*_eval_results.json`` files, prints a table of all 9 I2V dimension
scores (raw + normalized), and computes Quality Score, I2V Score, and Total Score.

Usage:
    python evaluation/vbench_i2v/cal_scores_i2v.py --result_dir <eval_results_dir>
"""

import argparse
import json
import os

I2V_KEY = {
    "camera_motion": "Video-Text Camera Motion",
    "i2v_subject": "Video-Image Subject Consistency",
    "i2v_background": "Video-Image Background Consistency",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
}

I2V_DIMS = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]

QUALITY_DIMS = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
]

DIM_WEIGHT = {
    "Video-Text Camera Motion": 0.1,
    "Video-Image Subject Consistency": 1,
    "Video-Image Background Consistency": 1,
    "Subject Consistency": 1,
    "Background Consistency": 1,
    "Motion Smoothness": 1,
    "Dynamic Degree": 0.5,
    "Aesthetic Quality": 1,
    "Imaging Quality": 1,
}

NORMALIZE = {
    "Video-Text Camera Motion": (0.0, 1.0),
    "Video-Image Subject Consistency": (0.1462, 1.0),
    "Video-Image Background Consistency": (0.2615, 1.0),
    "Subject Consistency": (0.1462, 1.0),
    "Background Consistency": (0.2615, 1.0),
    "Motion Smoothness": (0.706, 0.9975),
    "Dynamic Degree": (0.0, 1.0),
    "Aesthetic Quality": (0.0, 1.0),
    "Imaging Quality": (0.0, 1.0),
}

I2V_WEIGHT = 1.0
QUALITY_WEIGHT = 1.0


def load_scores(result_dir: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith("_eval_results.json"):
            continue
        with open(os.path.join(result_dir, fname)) as f:
            data = json.load(f)
        for key, val in data.items():
            display_name = I2V_KEY.get(key, key)
            if isinstance(val, list) and len(val) >= 1:
                scores[display_name] = val[0]
            elif isinstance(val, (int, float)):
                scores[display_name] = val
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compute VBench-I2V final scores")
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()

    scores = load_scores(args.result_dir)

    all_dims = I2V_DIMS + QUALITY_DIMS
    missing = [d for d in all_dims if d not in scores]
    if missing:
        print(f"Warning: missing {len(missing)} dimensions: {', '.join(missing)}\n")

    header = f"{'Dimension':<42s} {'Raw':>8s} {'Norm':>8s} {'Weight':>6s} {'Group':<10s}"
    print(header)
    print("-" * len(header))

    norm_scores: dict[str, float] = {}
    for dim in all_dims:
        if dim not in scores:
            continue
        raw = scores[dim]
        lo, hi = NORMALIZE[dim]
        norm = (raw - lo) / (hi - lo) if hi > lo else 0.0
        norm = max(0.0, min(1.0, norm))
        norm_scores[dim] = norm
        w = DIM_WEIGHT[dim]
        group = "I2V" if dim in I2V_DIMS else "Quality"
        print(f"  {dim:<40s} {raw:8.4f} {norm:8.4f} {w:6.1f} {group:<10s}")

    print()

    i2v_num = sum(norm_scores.get(d, 0) * DIM_WEIGHT[d] for d in I2V_DIMS)
    i2v_den = sum(DIM_WEIGHT[d] for d in I2V_DIMS if d in norm_scores)
    i2v_score = i2v_num / i2v_den if i2v_den else 0.0

    quality_num = sum(norm_scores.get(d, 0) * DIM_WEIGHT[d] for d in QUALITY_DIMS)
    quality_den = sum(DIM_WEIGHT[d] for d in QUALITY_DIMS if d in norm_scores)
    quality_score = quality_num / quality_den if quality_den else 0.0

    total = (QUALITY_WEIGHT * quality_score + I2V_WEIGHT * i2v_score) / (QUALITY_WEIGHT + I2V_WEIGHT)

    print(f"  {'I2V Score':<40s} {i2v_score:8.4f}   (weighted avg of {len([d for d in I2V_DIMS if d in norm_scores])} I2V dims)")
    print(f"  {'Quality Score':<40s} {quality_score:8.4f}   (weighted avg of {len([d for d in QUALITY_DIMS if d in norm_scores])} quality dims)")
    print(f"  {'Total Score':<40s} {total:8.4f}   (quality*{QUALITY_WEIGHT} + i2v*{I2V_WEIGHT}) / {QUALITY_WEIGHT + I2V_WEIGHT}")


if __name__ == "__main__":
    main()
