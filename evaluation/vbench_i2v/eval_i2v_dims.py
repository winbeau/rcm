import argparse
import os

import torch
from vbench2_beta_i2v import VBenchI2V

ALL_I2V_DIMS = ["i2v_subject", "i2v_background", "camera_motion"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate VBench-I2V dimensions")
    parser.add_argument("--videos_path", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="16-9", choices=["1-1", "8-5", "7-4", "16-9"])
    parser.add_argument("--full_json_dir", type=str, default="evaluation/vbench_i2v/data/vbench2_i2v_full_info.json")
    parser.add_argument("--output_path", type=str, default=None, help="Result directory (default: <videos_path>_eval_results)")
    parser.add_argument("--dimension", nargs="+", default=ALL_I2V_DIMS)
    args = parser.parse_args()

    output_path = args.output_path or f"{args.videos_path}_eval_results"
    device = torch.device("cuda")

    for dim in args.dimension:
        result_file = os.path.join(output_path, f"{dim}_eval_results.json")
        if os.path.exists(result_file):
            print(f"Skipping {dim}: {result_file} already exists")
            continue

        print(f"=== Evaluating: {dim} ===")
        my_VBench = VBenchI2V(device, args.full_json_dir, output_path)
        my_VBench.evaluate(
            videos_path=args.videos_path,
            name=dim,
            dimension_list=[dim],
            resolution=args.resolution,
        )


if __name__ == "__main__":
    main()
