"""
python /home/iitb/Kishan_SpecDec/_spade2/summary/specbench/summary_specbench.py \
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/specbench/qwen_specbench_results_gm6_4b_30b_genLen64.jsonl \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/specbench/qwen_specbench_summary_gm6_4b_30b_genLen64.json

"""

#!/usr/bin/env python3
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# -------------------------
# Category remapping
# -------------------------

MULTI_TURN_CATEGORIES = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}

def normalize_category(cat: str) -> str:
    if cat in MULTI_TURN_CATEGORIES:
        return "multi_turn_conversation"
    return cat


# -------------------------
# Aggregation helpers
# -------------------------

def init_metric_store():
    return {
        "tokens": [],
        "throughput": [],
        "skip": [],
        "acceptance": []  # speculative only
    }


# -------------------------
# Main summarization logic
# -------------------------

def summarize(results):
    """
    results: list of SpecBench result dicts
    """
    data = defaultdict(lambda: {
        "target": init_metric_store(),
        "draft": init_metric_store(),
        "speculative": init_metric_store()
    })

    for item in tqdm(results, desc="Aggregating SpecBench metrics"):
        category = normalize_category(item["category"])

        # ---- Target ----
        if "target" in item:
            t = item["target"]
            data[category]["target"]["tokens"].append(t.get("tokens_generated", 0))
            data[category]["target"]["throughput"].append(
                t.get("throughput_toks_per_sec", 0)
            )
            data[category]["target"]["skip"].append(t.get("mean_skip", 0))

        # ---- Draft ----
        if "draft" in item:
            d = item["draft"]
            data[category]["draft"]["tokens"].append(d.get("tokens_generated", 0))
            data[category]["draft"]["throughput"].append(
                d.get("throughput_toks_per_sec", 0)
            )
            data[category]["draft"]["skip"].append(d.get("mean_skip", 0))

        # ---- Speculative ----
        if "speculative" in item:
            s = item["speculative"]
            data[category]["speculative"]["tokens"].append(
                s.get("tokens_generated", 0)
            )
            data[category]["speculative"]["throughput"].append(
                s.get("throughput_toks_per_sec", 0)
            )
            data[category]["speculative"]["skip"].append(
                s.get("mean_skip", 0)
            )
            data[category]["speculative"]["acceptance"].append(
                s.get("acceptance_rate", 0)
            )

    # -------------------------
    # Compute means
    # -------------------------

    summary = {}

    for category, models in data.items():
        summary[category] = {}

        for model_name, metrics in models.items():
            if len(metrics["tokens"]) == 0:
                continue

            model_summary = {
                "mean_tokens_generated": float(np.mean(metrics["tokens"])),
                "mean_throughput_toks_per_sec": float(
                    np.mean(metrics["throughput"])
                ),
                "mean_skip": float(np.mean(metrics["skip"]))
            }

            if model_name == "speculative":
                model_summary["mean_acceptance_rate"] = float(
                    np.mean(metrics["acceptance"])
                )

            summary[category][model_name] = model_summary

    return summary


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Category-wise mean summarization for SpecBench outputs"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SpecBench result JSON"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output summary JSON"
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    summary = summarize(results)

    summary["_meta"] = {
        "input_path": args.input,
        "instances_used": len(results),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ SpecBench summary written to: {args.output}")


if __name__ == "__main__":
    main()
