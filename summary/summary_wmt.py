"""
python /home/iitb/Kishan_SpecDec/_spade2/summary/wmt/summary_wmt.py \
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/wmt/spec_time.json \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/wmt/spec_time_summary.json

"""
#!/usr/bin/env python3
import json
import argparse
import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm

# -------------------------
# Metric helpers
# -------------------------

bleu1_metric = BLEU(max_ngram_order=1, smooth_method="exp")
bleu4_metric = BLEU(max_ngram_order=4, smooth_method="exp")

def compute_bleu(hypothesis, references):
    bleu1 = bleu1_metric.corpus_score(
        [hypothesis],
        [[r] for r in references]
    ).score

    bleu4 = bleu4_metric.corpus_score(
        [hypothesis],
        [[r] for r in references]
    ).score

    return bleu1, bleu4

# def compute_bleu(hypothesis, references):
#     # """
#     # Returns BLEU-1 and BLEU-4 using sentence_bleu (version-safe)
#     # """
#     bleu1 = sacrebleu.sentence_bleu(
#         hypothesis,
#         [references],
#         smooth_method="exp",
#         max_ngram_order=1,
#     ).score

#     bleu4 = sacrebleu.sentence_bleu(
#         hypothesis,
#         [references],
#         smooth_method="exp",
#         max_ngram_order=4,
#     ).score
#     # sb = sacrebleu.sentence_bleu(
#     #     hypothesis,
#     #     references,
#     #     smooth_method="exp"
#     # )

#     # bleu1 = sb.precisions[0]  # BLEU-1 (precision)
#     # bleu4 = sb.score          # BLEU-4

#     return bleu1, bleu4


def compute_rouge(hypothesis, references):
    """
    Returns ROUGE-1 F1 and ROUGE-L F1 (averaged over references)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=True
    )

    r1, rL = [], []
    for ref in references:
        score = scorer.score(ref, hypothesis)
        r1.append(score["rouge1"].fmeasure)
        rL.append(score["rougeL"].fmeasure)

    return np.mean(r1) * 100, np.mean(rL) * 100


# def compute_cider(hypothesis, references, idx, cider_scorer):
#     """
#     Returns CIDEr-D score
#     """
#     gts = {idx: references}
#     res = {idx: [hypothesis]}
#     score, _ = cider_scorer.compute_score(gts, res)
#     return score


# -------------------------
# Model evaluation
# -------------------------

def evaluate_model(results, model_key):
    bleu1s, bleu4s = [], []
    rouge1s, rougeLs = [], []
    ciders = []

    toks, speeds, skips = [], [], []
    acc_rates, spec_calls = [], []

    # cider_scorer = Cider()
    # ---- CIDEr containers ----
    gts = {}
    res = {}

    for i, item in enumerate(tqdm(results, desc=f"Evaluating {model_key}")):
        refs = item["label"]
        output = item[model_key]["output_text"]

        bleu1, bleu4 = compute_bleu(output, refs)
        rouge1, rougeL = compute_rouge(output, refs)
        # cider = compute_cider(output, refs, i, cider_scorer)

        bleu1s.append(bleu1)
        bleu4s.append(bleu4)
        rouge1s.append(rouge1)
        rougeLs.append(rougeL)
        # ciders.append(cider)

        gts[i] = refs
        res[i] = [output]

        toks.append(item[model_key].get("tokens_generated", 0))
        speeds.append(item[model_key].get("throughput_toks_per_sec", 0))
        skips.append(item[model_key].get("mean_skip", 0))

        if model_key == "speculative":
            acc_rates.append(item[model_key].get("acceptance_rate", 0))
            spec_calls.append(
                item[model_key].get("spec_target_model_calls_reported", 0)
            )
    
    # ---- CIDEr-D (CORPUS-LEVEL) ----
    cider_scorer = Cider()
    cider_mean, cider_per_sample = cider_scorer.compute_score(gts, res)

    summary = {
        "bleu1_mean": float(np.mean(bleu1s)),
        "bleu4_mean": float(np.mean(bleu4s)),
        "rouge1_f1_mean": float(np.mean(rouge1s)),
        "rougeL_f1_mean": float(np.mean(rougeLs)),
        "ciderD_mean": float(cider_mean),
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
        description="Evaluate translation outputs and summarize quality metrics"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input results JSON"
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
            "k_last_question_id": results[-1]["question_id"]
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation complete. Summary written to: {args.output}")


if __name__ == "__main__":
    main()
