#!/usr/bin/env python3

"""

python /home/iitb/Kishan_SpecDec/_spade2/summary/cnnDialyMail/summary_cnnDailyMail.py\
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/cnndailyMail/qwen_k40_cnn_result_point9_gm6_4B_30B_genLen128.json \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/cnnDialyMail/qwen_k40_cnn_summary_point9_gm6_4B_30B_genLen128.json

"""
import json
import argparse
from collections import defaultdict

def safe_append(d, key, value):
    if value is not None:
        d[key].append(value)

def compute_means(values):
    return sum(values) / len(values) if values else None

def main(args):
    with open(args.input, "r") as f:
        data = json.load(f)

    # model_name -> metric_name -> list of values
    stats = defaultdict(lambda: defaultdict(list))

    for item in data:
        for model_name, model_out in item.items():
            # skip non-model keys
            if model_name in ["question_id", "prompt", "label"]:
                continue

            safe_append(stats[model_name], "tokens_generated",
                        model_out.get("tokens_generated"))
            safe_append(stats[model_name], "throughput_toks_per_sec",
                        model_out.get("throughput_toks_per_sec"))
            safe_append(stats[model_name], "mean_skip",
                        model_out.get("mean_skip"))
            safe_append(stats[model_name], "acceptance_rate",
                        model_out.get("acceptance_rate"))
            safe_append(stats[model_name], "target_calls",
                        model_out.get("spec_target_model_calls_reported"))

    # compute means
    output = {}
    for model_name, metrics in stats.items():
        output[model_name] = {
            "mean_tokens_generated":
                compute_means(metrics["tokens_generated"]),
            "mean_throughput_toks_per_sec":
                compute_means(metrics["throughput_toks_per_sec"]),
            "mean_mean_skip":
                compute_means(metrics["mean_skip"]),
            "mean_acceptance_rate":
                compute_means(metrics["acceptance_rate"]),
            "mean_target_calls":
                compute_means(metrics["target_calls"]),
        }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved mean statistics to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mean behavior per model from CNN/DailyMail outputs"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON file"
    )

    args = parser.parse_args()
    main(args)
