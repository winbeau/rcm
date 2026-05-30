# VBench Text-to-Video Evaluation

## 0. Prerequisites

```bash
pip install vbench

# detectron2 (required by some VBench dimensions)
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

## 1. Prompt JSON

`prompts.json` contains the 944 VBench-T2V prompts with GPT-4o-augmented
versions and VBench `auxiliary_info` (object labels, colors, spatial
relationships, etc.) needed for semantic evaluation dimensions.

Format:

```json
[
  {
    "prompt": "a bird and a cat",
    "dimensions": ["multiple_objects"],
    "augmented_prompt": "In a sunlit garden, a sleek black cat ...",
    "auxiliary_info": {"multiple_objects": {"object": "bird and cat"}}
  }
]
```

## 2. Generate Videos

`sample_videos.py` distributes prompts across GPU ranks via `torchrun` and
generates `--num_samples` videos per prompt.  Two orthogonal flags control the
configuration:

| Flag | Values | Meaning |
|------|--------|---------|
| `--arch` | `bidirectional`, `causal` | Model architecture |
| `--distilled` | (flag) | Few-step distilled sampling; omit for multi-step Euler ODE |

When `--output_dir` is omitted, the script auto-generates a directory from the
run configuration to prevent conflicts, e.g.
`evaluation/vbench_text2video/videos_causal_distilled_4steps_<ckpt>_seed0`.

### Causal + few-step (distilled)

```bash
torchrun --nproc_per_node=8 evaluation/vbench_text2video/sample_videos.py \
    --arch causal --distilled \
    --dit_path path/to/distilled.pth \
    --prompt_json evaluation/vbench_text2video/prompts.json \
    --num_samples 5 --num_steps 4 \
    --first_chunk_t 1 --chunk_t 1 \
    --model_size 1.3B/14B --seed 0
```

### Causal + multi-step (diffusion/teacher)

```bash
torchrun --nproc_per_node=8 evaluation/vbench_text2video/sample_videos.py \
    --arch causal \
    --dit_path path/to/diffusion.pth \
    --prompt_json evaluation/vbench_text2video/prompts.json \
    --num_samples 5 --num_steps 50 \
    --guidance_scale 3.0 --timestep_shift 3.0 \
    --first_chunk_t 1 --chunk_t 4 \
    --model_size 1.3B/14B --seed 0
```

Set `--cp_size=2` for 14B model to avoid OOM with CFG.

### Bidirectional + few-step (distilled)

```bash
torchrun --nproc_per_node=8 evaluation/vbench_text2video/sample_videos.py \
    --arch bidirectional --distilled \
    --dit_path path/to/distilled.pth \
    --prompt_json evaluation/vbench_text2video/prompts.json \
    --num_samples 5 --num_steps 4 \
    --model_size 1.3B/14B --seed 0
```

### Bidirectional + ODE (diffusion/teacher)

```bash
torchrun --nproc_per_node=8 evaluation/vbench_text2video/sample_videos.py \
    --arch bidirectional \
    --dit_path path/to/diffusion.pth \
    --prompt_json evaluation/vbench_text2video/prompts.json \
    --num_samples 5 --num_steps 50 \
    --guidance_scale 5.0 --timestep_shift 3.0 \
    --model_size 1.3B/14B --seed 0
```

### Key flags

| Flag | Description |
|------|-------------|
| `--arch` | `bidirectional` or `causal` |
| `--distilled` | Few-step distilled sampling (omit for Euler ODE) |
| `--num_samples` | Videos per prompt (default 5, VBench standard) |
| `--output_dir` | Saving directory (default: auto-generated from run config) |
| `--no_augmented_prompt` | Disable augmented prompts for generation; use original short prompts |

### Output structure

```
<output_dir>/
  In a still frame, a stop sign-0.mp4
  In a still frame, a stop sign-1.mp4
  ...
  vbench_eval_info.json    # VBench-compatible JSON with prompts, dimensions, auxiliary_info, video_list
```

The script **auto-skips** prompts whose videos already exist on disk.

`vbench_eval_info.json` is a drop-in replacement for `VBench_full_info.json`,
with `video_list` already populated. It includes `auxiliary_info` for all 16
dimensions.

## 3. Evaluate with VBench

Use the generated `vbench_eval_info.json` as `--full_json_dir` in VBench
standard mode. This supports all 16 dimensions including semantic ones
(object_class, color, etc.) that require auxiliary metadata:

### Evaluate all 16 dimensions

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # for compatibility with PyTorch>=2.6.0
export VBENCH_CACHE_DIR=~/.cache/vbench

VIDEO_DIR=<output_dir>
RESULT_DIR=${VIDEO_DIR}_eval_results
EVAL_JSON=$VIDEO_DIR/vbench_eval_info.json

DIMS=(
  subject_consistency background_consistency temporal_flickering
  motion_smoothness dynamic_degree aesthetic_quality imaging_quality
  object_class multiple_objects human_action color
  spatial_relationship scene appearance_style
  overall_consistency temporal_style
)

NGPUS=8

for dim in "${DIMS[@]}"; do
    echo "=== Evaluating: $dim ==="
    vbench evaluate \
        --ngpus=$NGPUS \
        --videos_path "$VIDEO_DIR" \
        --dimension "$dim" \
        --output_path "$RESULT_DIR" \
        --full_json_dir "$EVAL_JSON"
done
```

## 4. Compute Final Scores

```bash
python evaluation/vbench_text2video/cal_scores.py --result_dir "$RESULT_DIR"
```

This prints all 16 dimension scores (raw + normalized), then the aggregated
Quality Score, Semantic Score, and Total Score:

```
Dimension                         Raw     Norm Weight Group
----------------------------------------------------------------
  subject consistency           0.9543   0.9459    1.0 Quality
  background consistency        0.9712   0.9611    1.0 Quality
  ...
  overall consistency           0.2341   0.6432    1.0 Semantic

  Quality Score                 0.8234   (weighted avg of 7 quality dims)
  Semantic Score                0.7456   (weighted avg of 9 semantic dims)
  Total Score                   0.8078   (quality*4 + semantic*1) / 5
```

| Aggregate | Dimensions |
|-----------|-----------|
| **Quality Score** | subject_consistency, background_consistency, temporal_flickering, motion_smoothness, aesthetic_quality, imaging_quality, dynamic_degree |
| **Semantic Score** | object_class, multiple_objects, human_action, color, spatial_relationship, scene, appearance_style, temporal_style, overall_consistency |
| **Total Score** | Weighted average of Quality and Semantic scores |

# VisionReward Evaluation

## Prerequisites

```bash
pip install git+https://github.com/facebookresearch/pytorchvideo.git
pip install transformers==4.42.4
```

The model (`THUDM/VisionReward-Video`) is downloaded automatically from
HuggingFace on first run.

### Run evaluation

```bash
# Single GPU
python evaluation/vbench_text2video/eval_vision_reward.py \
    --eval_info "$VIDEO_DIR/vbench_eval_info.json"

# Multi-GPU (videos are distributed across ranks)
torchrun --nproc_per_node=8 evaluation/vbench_text2video/eval_vision_reward.py \
    --eval_info "$VIDEO_DIR/vbench_eval_info.json"
```

### Output

Results are saved to `<output_dir>/vision_reward_scores.json`:

```json
{
  "mean_score": 0.1234,
  "num_videos": 4720,
  "video_results": [
    {"video_path": "...", "prompt": "...", "score": 0.1500},
    ...
  ]
}
```

# VBench-Long Evaluation

VBench-Long evaluates long videos using a slow-fast approach (within-clip +
cross-clip consistency). It uses the same 16 dimensions and prompt suite as
T2V -- just generate longer videos with more frames.

## Prerequisites

```bash
pip install scenedetect[opencv] --upgrade
```

## Generate long videos

Use the same sampling script with a larger `--num_frames`:

```bash
torchrun --nproc_per_node=8 evaluation/vbench_text2video/sample_videos.py \
    --arch causal --distilled \
    --dit_path path/to/distilled.pth \
    --prompt_json evaluation/vbench_text2video/prompts.json \
    --num_samples 5 --num_steps 4 \
    --num_frames 241 \
    --first_chunk_t 1 --chunk_t 1 \
    --seed 0
```

## Evaluate

VBench-Long handles video splitting and slow-fast evaluation internally:

```bash
git clone https://github.com/Vchitect/VBench.git /tmp/VBench 2>/dev/null || true

VIDEO_DIR=<output_dir>
RESULT_DIR=${VIDEO_DIR}_eval_results_long

DIMS=(
  subject_consistency background_consistency temporal_flickering
  motion_smoothness dynamic_degree aesthetic_quality imaging_quality
  object_class multiple_objects human_action color
  spatial_relationship scene temporal_style appearance_style
  overall_consistency
)

for dim in "${DIMS[@]}"; do
    echo "=== Evaluating: $dim ==="
    python /tmp/VBench/vbench2_beta_long/eval_long.py \
        --videos_path "$VIDEO_DIR" \
        --dimension "$dim" \
        --output_path "$RESULT_DIR" \
        --mode long_vbench_standard \
        --dev_flag
done
```

For `temporal_flickering`, add `--static_filter_flag`.

## Compute scores

```bash
python evaluation/vbench_text2video/cal_scores.py --result_dir "$RESULT_DIR"
```