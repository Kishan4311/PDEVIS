#!/usr/bin/env python3
"""

Example:
python /home/iitb/Kishan_SpecDec/_spade2/summary/ifeval/summary_ifeval.py \
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/ifeval/qwen_ifeval_result_point9_gm6_4B_30B_genLen64.json \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/ifeval/qwen_ifeval_summary_point9_gm6_4B_30B_genLen64.json

"""

import json
import argparse
import numpy as np
from tqdm import tqdm

# -------------------------
# Metric helpers
# -------------------------

def compute_instruction_accuracy(instruction_results: dict):
    """
    instruction_results: dict[str, bool]
    Returns:
      inst_acc: float
      strict_acc: int (0 or 1)
    """
    if not instruction_results:
        return 0.0, 0

    values = list(instruction_results.values())
    inst_acc = float(np.mean(values))
    strict_acc = int(all(values))
    return inst_acc, strict_acc


# -------------------------
# Model evaluation
# -------------------------

def evaluate_model(results, model_key):
    inst_accs = []
    strict_accs = []

    toks, speeds, skips = [], [], []
    acc_rates, spec_calls = [], []

    for item in tqdm(results, desc=f"Evaluating {model_key}"):

        model_out = item.get(model_key, {})
        instruction_results = model_out.get("instruction_results", {})

        inst_acc, strict_acc = compute_instruction_accuracy(
            instruction_results
        )

        inst_accs.append(inst_acc)
        strict_accs.append(strict_acc)

        # --- optional decoding stats ---
        toks.append(model_out.get("tokens_generated", 0))
        speeds.append(model_out.get("throughput_toks_per_sec", 0))
        skips.append(model_out.get("mean_skip", 0))

        if model_key == "speculative":
            acc_rates.append(model_out.get("acceptance_rate", 0))
            spec_calls.append(
                model_out.get("spec_target_model_calls_reported", 0)
            )

    summary = {
        "instruction_accuracy_mean": float(np.mean(inst_accs)),
        "strict_accuracy_mean": float(np.mean(strict_accs)),
        "mean_tokens_generated": float(np.mean(toks)),
        "mean_throughput_toks_per_sec": float(np.mean(speeds)),
        "mean_skip": float(np.mean(skips)),
        "num_instances": len(results)
    }

    if model_key == "speculative":
        summary.update({
            "mean_acceptance_rate": float(np.mean(acc_rates)),
            "mean_spec_target_model_calls": float(np.mean(spec_calls))
        })

    return summary


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize IFEval instruction-following quality"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to IFEval result JSON"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output summary JSON"
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = {
        "Target_only": evaluate_model(results, "target_only"),
        "Drafter_only": evaluate_model(results, "drafter_only"),
        "Speculative": evaluate_model(results, "speculative"),
        "_meta": {
            "input_path": args.input,
            "instances_used": len(results),
            "k_last_question_id": results[-1].get("question_id", None)
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ IFEval summary written to: {args.output}")


if __name__ == "__main__":
    main()
