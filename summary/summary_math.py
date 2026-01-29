#!/usr/bin/env python3
"""
Math QA quality & mean summary script

Example:
python /home/iitb/Kishan_SpecDec/_spade2/summary/math/summary_math.py \
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/math/qwen_math_result_point9_gm6_4B_30B_genLen64.json\
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/math/qwen_math_summary_point9_gm6_4B_30B_genLen64.json
"""

import json
import argparse
import numpy as np
import re
from tqdm import tqdm

# -------------------------
# Answer extraction helpers
# -------------------------

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed_answer(text):
    """
    Extracts the last \\boxed{...} answer.
    Returns None if not found.
    """
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(ans):
    """
    Light normalization for math answers.
    """
    if ans is None:
        return None

    ans = ans.strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("\\,", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    return ans


def is_correct(pred_text, label_texts):
    """
    pred_text: model output
    label_texts: list of reference solutions
    """
    pred_ans = normalize_answer(extract_boxed_answer(pred_text))
    if pred_ans is None:
        return False

    for ref in label_texts:
        ref_ans = normalize_answer(extract_boxed_answer(ref))
        if ref_ans is None:
            continue
        if pred_ans == ref_ans:
            return True

    return False


# -------------------------
# Model evaluation
# -------------------------

def evaluate_model(results, model_key):
    correct = []

    toks, speeds, skips = [], [], []
    acc_rates, spec_calls = [], []

    for item in tqdm(results, desc=f"Evaluating {model_key}"):

        label = item.get("label", [])
        model_out = item.get(model_key, {})
        output = model_out.get("output_text", "")

        ok = is_correct(output, label)
        correct.append(int(ok))

        toks.append(model_out.get("tokens_generated", 0))
        speeds.append(model_out.get("throughput_toks_per_sec", 0))
        skips.append(model_out.get("mean_skip", 0))

        if model_key == "speculative":
            acc_rates.append(model_out.get("acceptance_rate", 0))
            spec_calls.append(
                model_out.get("spec_target_model_calls_reported", 0)
            )

    summary = {
        "exact_match_accuracy": float(np.mean(correct)),
        "num_correct": int(np.sum(correct)),
        "num_total": len(results),
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
        description="Summarize math QA quality metrics"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to math result JSON"
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

    print(f"\n✅ Math QA evaluation complete. Summary written to: {args.output}")


if __name__ == "__main__":
    main()
