import argparse
import io
import json
import os

import numpy as np
import torch
from decord import VideoReader, cpu, bridge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from imaginaire.utils import distributed

MODEL_PATH = "THUDM/VisionReward-Video"

QUESTIONS = [
    'Does the video meet all the requirements stated in the text "[[prompt]]"?',
    'Does the video meet most of the requirements stated in the text "[[prompt]]"?',
    'Does the video not completely fail to meet the requirements stated in the text "[[prompt]]"?',
    "Is the composition aesthetically pleasing?",
    "Does the composition have no obvious flaws?",
    "Does the camera movement have no obvious flaws?",
    "Are the colors not significantly unattractive?",
    "Is the lighting perfectly accurate?",
    "Does the lighting have no obvious errors?",
    "Is there any lighting present?",
    "Is the lighting exceptionally beautiful?",
    "Is the lighting beautiful?",
    "Is the lighting not unattractive?",
    "Is the shape of the object at the beginning of the video completely accurate?",
    "Does the shape of the object at the beginning have no obvious errors?",
    "Is the shape of the object at the beginning not chaotic?",
    "Is the shape of the object perfectly maintained throughout the video?",
    "Is the shape of the object not chaotic throughout the video?",
    "Is the camera motion highly dynamic?",
    "Is the camera motion not minimal?",
    "Is the smoothness of the object's movement very good?",
    "Is the object's movement completely realistic?",
    "Is the image quality very stable?",
    "Are the details very refined?",
    "Are the details not rough?",
    "Are the details not significantly rough?",
    "Are all the letters correct?",
    "Are there any letters present?",
    "Is the video content part of the physical world?",
]

WEIGHTS = np.array(
    [
        0.9543901856422174,
        0.25239747290239256,
        1.141818673357406,
        0.03495652038170829,
        0.025237463294006605,
        0.12600844108184325,
        0.03221505988621183,
        0.16286819641189937,
        0.21673935360893115,
        0.01970324496671629,
        0.13604019362894557,
        0.09647134683834487,
        0.15490927135496332,
        0.1294164598219855,
        0.09891696198970226,
        0.18839328668539077,
        0.1844335421380767,
        0.2635526157239052,
        0.11168980468489233,
        0.05173789659242723,
        0.02562797122879315,
        0.4389890596048526,
        0.26857694964769424,
        0.42925171836383774,
        0.00846154228462919,
        0.12757277121689847,
        0.05798205026065391,
        0.1446334304609205,
        0.39418111694677266,
    ]
)


# ---------------------------------------------------------------------------
# Video loading & scoring
# ---------------------------------------------------------------------------


def load_video_frames(video_path: str, num_frames: int = 24) -> torch.Tensor:
    bridge.set_bridge("torch")
    with open(video_path, "rb") as f:
        video_data = f.read()
    vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
    total = len(vr)
    timestamps = vr.get_frame_timestamp(np.arange(total))
    timestamps = [t[0] for t in timestamps]
    max_second = round(max(timestamps)) + 1
    frame_ids = []
    for sec in range(max_second):
        closest = min(timestamps, key=lambda x: abs(x - sec))
        frame_ids.append(timestamps.index(closest))
        if len(frame_ids) >= num_frames:
            break
    frames = vr.get_batch(frame_ids)
    return frames.permute(3, 0, 1, 2)  # [C, T, H, W]


def score_video(model, tokenizer, video_path: str, prompt: str, device: str, torch_type) -> float:
    video = load_video_frames(video_path)
    queries = [q.replace("[[prompt]]", prompt) for q in QUESTIONS]

    answers = []
    for query in queries:
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=[],
            template_version="chat",
        )
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
            "images": [[inputs["images"][0].to(device).to(torch_type)]],
        }
        gen_kwargs = {"max_new_tokens": 2048, "pad_token_id": 128002, "top_k": 1, "do_sample": False, "top_p": 0.1, "temperature": 0.1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1]]
        answers.append(tokenizer.decode(outputs[0]))

    scores = np.array([1 if a.strip().lower().startswith("yes") else -1 for a in answers])
    return float(np.mean(scores * WEIGHTS))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="VisionReward evaluation on generated videos")
    p.add_argument("--eval_info", type=str, required=True, help="Path to vbench_eval_info.json")
    p.add_argument("--output", type=str, default=None, help="Output JSON path (default: <eval_info_dir>/vision_reward_scores.json)")
    p.add_argument("--model_path", type=str, default=MODEL_PATH)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    distributed.init()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    with open(args.eval_info) as f:
        eval_info = json.load(f)

    all_items: list[tuple[str, str]] = []
    for entry in eval_info:
        prompt = entry["prompt_en"]
        for vp in entry.get("video_list", []):
            all_items.append((vp, prompt))

    my_items = all_items[rank::world_size]

    if rank == 0:
        print(f"Total videos: {len(all_items)}, this rank: {len(my_items)}, world_size: {world_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if rank == 0:
        print(f"Loading VisionReward model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch_type, trust_remote_code=True).eval().to(device)

    video_results = []
    for video_path, prompt in tqdm(my_items, desc=f"[R{rank}] Scoring", disable=(rank != 0)):
        if not os.path.exists(video_path):
            if rank == 0:
                print(f"  Skipping missing: {video_path}")
            continue
        s = score_video(model, tokenizer, video_path, prompt, device, torch_type)
        video_results.append({"video_path": video_path, "prompt": prompt, "score": s})

    # ---- Save per-rank results ----
    eval_dir = os.path.dirname(args.eval_info)
    rank_path = os.path.join(eval_dir, f"vision_reward_rank{rank}.json")
    with open(rank_path, "w") as f:
        json.dump(video_results, f, indent=2, ensure_ascii=False)

    distributed.barrier()

    # ---- Rank 0: merge results ----
    if rank == 0:
        merged = []
        for r in range(world_size):
            rp = os.path.join(eval_dir, f"vision_reward_rank{r}.json")
            if os.path.exists(rp):
                with open(rp) as f:
                    merged.extend(json.load(f))
                os.remove(rp)

        scores = [v["score"] for v in merged]
        mean_score = float(np.mean(scores)) if scores else 0.0
        output_path = args.output or os.path.join(eval_dir, "vision_reward_scores.json")
        results = {
            "mean_score": mean_score,
            "num_videos": len(merged),
            "video_results": merged,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nVisionReward mean score: {mean_score:.4f} ({len(merged)} videos)")
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
