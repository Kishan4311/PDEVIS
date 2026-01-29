""" 
python /home/iitb/Kishan_SpecDec/_spade2/summary/mgsm/summary_mgsm.py \
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/mgsm/llama_k40_mgsm_result_point9_gm6_3B_70B_genLen8.json \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/mgsm/llama_k40_mgsm_summary_point9_gm6_3B_70B_genLen8.json

"""

#!/usr/bin/env python3
import json
import argparse
import numpy as np
import re
from tqdm import tqdm


# -------------------------
# Helpers
# -------------------------

def normalize_number(x):
    """
    Normalize model output or label to a float if possible.
    Handles:
      - commas: "230,000"
      - currency: "$30,000"
      - whitespace
    Returns None if parsing fails.
    """
    if x is None:
        return None

    if isinstance(x, (int, float)):
        return float(x)

    x = str(x)
    x = x.replace(",", "")
    x = re.sub(r"[^0-9.\-]", "", x)

    try:
        return float(x)
    except ValueError:
        return None


def is_correct(pred, gold, tol=1e-6):
    """
    MGSM exact numeric match (with float tolerance)
    """
    p = normalize_number(pred)
    g = normalize_number(gold)

    if p is None or g is None:
        return False

    return abs(p - g) <= tol


# -------------------------
# Model evaluation
# -------------------------

def evaluate_model(results, model_key):
    total = 0
    correct = 0

    toks, speeds, skips = [], [], []
    acc_rates, spec_calls = [], []

    for item in tqdm(results, desc=f"Evaluating {model_key}"):
        gold = item["label"]
        pred = item[model_key]["output_text"]

        total += 1
        if is_correct(pred, gold):
            correct += 1

        toks.append(item[model_key].get("tokens_generated", 0))
        speeds.append(item[model_key].get("throughput_toks_per_sec", 0))
        skips.append(item[model_key].get("mean_skip", 0))

        if model_key == "speculative":
            acc_rates.append(item[model_key].get("acceptance_rate", 0))
            spec_calls.append(
                item[model_key].get("spec_target_model_calls_reported", 0)
            )

    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "accuracy": float(accuracy),
        "error_rate": float(1.0 - accuracy),
        "total_questions": total,
        "mean_tokens_generated": float(np.mean(toks)),
        "mean_throughput_toks_per_sec": float(np.mean(speeds)),
        "mean_skip": float(np.mean(skips))
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
        description="MGSM evaluation (numeric exact match)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    summary = {
        "target_only": evaluate_model(results, "target_only"),
        "drafter_only": evaluate_model(results, "drafter_only"),
        "speculative": evaluate_model(results, "speculative"),
        "_meta": {
            "instances": len(results),
            "last_question_id": results[-1]["question_id"]
        }
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ MGSM evaluation complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
