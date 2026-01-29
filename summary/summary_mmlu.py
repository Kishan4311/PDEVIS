'''
python /home/iitb/Kishan_SpecDec/_spade2/summary/mmlu/summary_mmlu.py\
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/mmlu/llama_k20_mmlu_result_point9_gm6_3B_70B_genLen64.json \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/mmlu/llama_k20_mmlu_summary_point9_gm6_3B_70B_genLen64.json

'''

#!/usr/bin/env python3
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def evaluate_model(results, model_key):
    total = 0
    correct = 0

    toks, speeds, skips = [], [], []
    acc_rates, spec_calls = [], []

    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)

    for item in tqdm(results, desc=f"Evaluating {model_key}"):
        gold = item["label"].strip()
        pred = item[model_key]["output_text"].strip()

        total += 1
        subject = item.get("subject", "unknown")
        subject_total[subject] += 1

        if pred == gold:
            correct += 1
            subject_correct[subject] += 1

        toks.append(item[model_key].get("tokens_generated", 0))
        speeds.append(item[model_key].get("throughput_toks_per_sec", 0))
        skips.append(item[model_key].get("mean_skip", 0))

        if model_key == "speculative":
            acc_rates.append(item[model_key].get("acceptance_rate", 0))
            spec_calls.append(
                item[model_key].get("spec_target_model_calls_reported", 0)
            )

    accuracy = correct / total if total > 0 else 0.0

    per_subject_accuracy = {
        s: subject_correct[s] / subject_total[s]
        for s in subject_total
    }

    summary = {
        "accuracy": float(accuracy),
        "error_rate": float(1.0 - accuracy),
        "total_questions": total,
        "mean_tokens_generated": float(np.mean(toks)),
        "mean_throughput_toks_per_sec": float(np.mean(speeds)),
        "mean_skip": float(np.mean(skips)),
        "per_subject_accuracy": per_subject_accuracy
    }

    if model_key == "speculative":
        summary.update({
            "mean_acceptance_rate": float(np.mean(acc_rates)),
            "mean_spec_target_model_calls": float(np.mean(spec_calls))
        })

    return summary


def main():
    parser = argparse.ArgumentParser("MMLU evaluation")
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

    print(f"\n✅ Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
