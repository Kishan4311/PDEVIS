"""

python /home/iitb/Kishan_SpecDec/_spade2/summary/wmt/ablation_summary_wmt.py\
  --input /home/iitb/Kishan_SpecDec/_spade2/evaluation/wmt/ablation_k80_point9_wmt.json \
  --output /home/iitb/Kishan_SpecDec/_spade2/summary/wmt/ablation_k80_point9_wmt_summary.json

"""

#!/usr/bin/env python3
import json
import argparse
import numpy as np
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


def compute_rouge(hypothesis, references):
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


# -------------------------
# Evaluation
# -------------------------

def evaluate_ablation(results):
    bleu1s, bleu4s = [], []
    rouge1s, rougeLs = [], []
    toks, speeds, skips = [], [], []

    gts = {}
    res = {}

    for i, item in enumerate(tqdm(results, desc="Evaluating ablation")):
        # ---- references ----
        refs = [item["label"]]  # wrap string into list

        # ---- hypothesis ----
        output = item["output"]["output_text"]

        bleu1, bleu4 = compute_bleu(output, refs)
        rouge1, rougeL = compute_rouge(output, refs)

        bleu1s.append(bleu1)
        bleu4s.append(bleu4)
        rouge1s.append(rouge1)
        rougeLs.append(rougeL)

        gts[i] = refs
        res[i] = [output]

        toks.append(item["output"].get("tokens_generated", 0))
        speeds.append(item["output"].get("throughput_toks_per_sec", 0))
        skips.append(item["output"].get("mean_skip", 0))

    # ---- CIDEr (corpus-level) ----
    cider_scorer = Cider()
    cider_mean, _ = cider_scorer.compute_score(gts, res)

    summary = {
        "bleu1_mean": float(np.mean(bleu1s)),
        "bleu4_mean": float(np.mean(bleu4s)),
        "rouge1_f1_mean": float(np.mean(rouge1s)),
        "rougeL_f1_mean": float(np.mean(rougeLs)),
        "ciderD_mean": float(cider_mean),
        "mean_tokens_generated": float(np.mean(toks)),
        "mean_throughput_toks_per_sec": float(np.mean(speeds)),
        "mean_skip": float(np.mean(skips)),
    }

    return summary


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize WMT ablation results"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = {
        "Ablation_Qwen_reuse": evaluate_ablation(results),
        "_meta": {
            "input_path": args.input,
            "instances_used": len(results),
            "last_question_id": results[-1]["question_id"]
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Ablation summary written to: {args.output}")


if __name__ == "__main__":
    main()
