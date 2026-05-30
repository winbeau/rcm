# VBench-I2V Evaluation (Causal Model)

Image-to-video generation and evaluation using the causal model with
`first_chunk_t=1`. The input image is VAE-encoded into the first latent frame,
its KV cache is prefilled (no denoising), and remaining chunks are causally
generated.

## 0. Prerequisites

```bash
pip install vbench gdown
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
pip install dreamsim
pip install transformers==4.44.0
pip install timm==1.0.11
# camera_motion dimension needs a CoTracker Visualizer stub
# (the pip package is missing vbench2_beta_i2v/third_party/)
VBENCH_I2V_PKG=$(python -c "import vbench2_beta_i2v; import os; print(os.path.dirname(vbench2_beta_i2v.__file__))")
mkdir -p "$VBENCH_I2V_PKG/third_party/cotracker/utils"
touch "$VBENCH_I2V_PKG/third_party/__init__.py"
touch "$VBENCH_I2V_PKG/third_party/cotracker/__init__.py"
touch "$VBENCH_I2V_PKG/third_party/cotracker/utils/__init__.py"
echo "class Visualizer: pass" > "$VBENCH_I2V_PKG/third_party/cotracker/utils/visualizer.py"
```

## 1. Download the VBench-I2V Image Suite

```bash
bash evaluation/vbench_i2v/download_images.sh
```

This downloads 355 images cropped to 4 aspect ratios (`1-1`, `8-5`, `7-4`,
`16-9`) into `evaluation/vbench_i2v/data/crop/` along with
`vbench2_i2v_full_info.json`.

VBench-I2V hardcodes image paths as `vbench2_beta_i2v/data/crop/{resolution}/`.
Create a symlink so VBench can find the images:

```bash
mkdir -p vbench2_beta_i2v
ln -sfn $(realpath evaluation/vbench_i2v/data) vbench2_beta_i2v/data
```

## 2. Generate Videos

Pick the crop folder matching your model's aspect ratio (e.g. `16-9` for
16:9 / 480p). The sampling script reads `vbench2_i2v_full_info.json` directly.

### Causal + few-step (distilled)

```bash
torchrun --nproc_per_node=8 evaluation/vbench_i2v/sample_videos_i2v.py \
    --distilled \
    --dit_path path/to/distilled.pth \
    --prompt_json evaluation/vbench_i2v/data/vbench2_i2v_full_info.json \
    --image_dir evaluation/vbench_i2v/data/crop/16-9 \
    --num_samples 5 --num_steps 4 \
    --chunk_t 1 --seed 0
```

### Causal + ODE (diffusion / teacher)

```bash
torchrun --nproc_per_node=8 evaluation/vbench_i2v/sample_videos_i2v.py \
    --dit_path path/to/teacher.pth \
    --prompt_json evaluation/vbench_i2v/data/vbench2_i2v_full_info.json \
    --image_dir evaluation/vbench_i2v/data/crop/16-9 \
    --num_samples 5 --num_steps 50 \
    --guidance_scale 3.0 --timestep_shift 3.0 \
    --chunk_t 4 --seed 0
```

Set `--cp_size=2` for 14B model to avoid OOM with CFG.

### How it works

1. The input image is resized to the target resolution and VAE-encoded into a
   single latent frame (`first_chunk_t=1`).
2. Chunk 0 is prefilled: a forward pass with `t=0` and `KVCacheMode.APPEND`
   stores the image latent's K/V in the cache -- no denoising is performed.
3. Chunks 1..N are denoised causally, attending to the prefilled image context.
4. The full latent sequence (image frame + generated frames) is decoded and
   saved.

### Output structure

```
<output_dir>/
  a close up of a blue and orange liquid-0.mp4
  a close up of a blue and orange liquid-1.mp4
  ...
  vbench_eval_info.json
```

## 3. Evaluate with VBench-I2V

VBench-I2V has 3 I2V-specific dimensions (`i2v_subject`, `i2v_background`,
`camera_motion`) plus 6 video-quality dimensions.

```bash
VIDEO_DIR=<output_dir>
RESULT_DIR=${VIDEO_DIR}_eval_results
EVAL_JSON=evaluation/vbench_i2v/data/vbench2_i2v_full_info.json
```

### I2V-specific dimensions (single GPU)

```bash
export GITHUB_TOKEN=xxx
python evaluation/vbench_i2v/eval_i2v_dims.py \
    --videos_path "$VIDEO_DIR" \
    --resolution 16-9
```

### Video-quality dimensions (multi-GPU)

```bash
NGPUS=8

QUALITY_DIMS=(
  subject_consistency background_consistency
  motion_smoothness dynamic_degree
  aesthetic_quality imaging_quality
)

for dim in "${QUALITY_DIMS[@]}"; do
    echo "=== Evaluating: $dim ==="
    vbench evaluate \
        --ngpus=$NGPUS \
        --videos_path "$VIDEO_DIR" \
        --dimension "$dim" \
        --output_path "$RESULT_DIR" \
        --full_json_dir "$EVAL_JSON"
done
```

### Compute final scores

```bash
python evaluation/vbench_i2v/cal_scores_i2v.py --result_dir "$RESULT_DIR"
```

This prints all 9 dimensions (raw + normalized) and the aggregated I2V Score,
Quality Score, and Total Score.
