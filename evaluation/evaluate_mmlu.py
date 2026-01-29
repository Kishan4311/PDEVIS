#!/usr/bin/env python
# coding=utf-8
"""
python /home/iitb/Kishan_SpecDec/_spade2/evaluation/mmlu/evaluate_mmlu.py \
  --draft_device "cuda:0" \
  --target_device "cuda:3" \
  --dataset "/home/iitb/Kishan_SpecDec/_spade2/Data/mmlu_filtered.json" \
  --code_path "/home/iitb/Kishan_SpecDec/_spade2/appInference2.py" \
  --k 20 \
  --gen_len 64 \
  --gamma 6 \
  --target_model "llama-70b" \
  --drafter_model "llama-3b" \
  --output "/home/iitb/Kishan_SpecDec/_spade2/evaluation/mmlu/llama_k20_mmlu_result_point9_gm6_3B_70B_genLen64.json"

"""


import json
import argparse
import time
import os
from typing import Dict, Any
from tqdm import tqdm
import torch
import numpy as np
import random

# ---------------------------------------------------------
# Import your existing SPADE inference code
# ---------------------------------------------------------
import sys
def load_inference(code_path):
    code_dir = os.path.dirname(code_path)
    sys.path.insert(0, code_dir)
    module_name = os.path.basename(code_path).replace(".py", "")
    return __import__(module_name)

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mmlu(path: str, k: int):
    with open(path, "r") as f:
        data = json.load(f)
    return data[:k]


# ---------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------
def main(args):
    set_seed(42)

    inference_mod = load_inference(args.code_path)

    cli = inference_mod.InferenceCLI(
        device="cuda:0",  # unused if per-model devices provided
        device_target=args.target_device,
        device_drafter=args.draft_device,
        target_model=inference_mod.ModelsCatalog.model_id(args.target_model),
        drafter_model=inference_mod.ModelsCatalog.model_id(args.drafter_model),
    )

    # override generation params
    cli.gen_len = args.gen_len
    cli.gamma = args.gamma
    cli.chat = True

    dataset = load_mmlu(args.dataset, args.k)
    # results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_f = open(args.output, "w")
    out_f.write("[\n")
    first = True

    print(f"Running evaluation on {len(dataset)} samples")

    for item in tqdm(dataset, desc="Evaluating MMLU"):
        qid = item["question_id"]
        prompt = item["prompt"]
        label = item["label"]
        subject = item["subject"]

        entry: Dict[str, Any] = {
            "question_id": qid,
            "subject": subject,
            "prompt": prompt,
            "label": label,
        }

        # -------------------------------------------------
        # Target only
        # -------------------------------------------------
        cli.spec = False
        cli.target_gen = True
        cli.dr = False

        t0 = time.time()
        out = cli.run_once(prompt)
        t1 = time.time()

        tgt_text = out["target"]
        tgt_tokens = len(cli.tokenizer.encode(tgt_text))
        tgt_tp = tgt_tokens / max(1e-6, (t1 - t0))

        entry["target_only"] = {
            "output_text": tgt_text,
            "tokens_generated": tgt_tokens,
            "throughput_toks_per_sec": tgt_tp,
            "mean_skip":0,
        }

        # -------------------------------------------------
        # Drafter only
        # -------------------------------------------------
        cli.spec = False
        cli.target_gen = False
        cli.dr = True

        t0 = time.time()
        out = cli.run_once(prompt)
        t1 = time.time()

        dr_text = out["drafter"]
        dr_tokens = len(cli.tokenizer.encode(dr_text))
        dr_tp = dr_tokens / max(1e-6, (t1 - t0))

        mean_skip = cli.drafter.model.pop_mean_skip()

        entry["drafter_only"] = {
            "output_text": dr_text,
            "tokens_generated": dr_tokens,
            "throughput_toks_per_sec": dr_tp,
            "mean_skip": mean_skip,
        }

        # -------------------------------------------------
        # Speculative
        # -------------------------------------------------
        cli.spec = True
        cli.target_gen = False
        cli.dr = False

        set_seed(42)
        t0 = time.time()
        output_ids, acc_rate, target_calls = inference_mod.speculative_generate(
            cli.tokenizer(
                cli.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                ),
                return_tensors="pt",
            ).input_ids[0].tolist(),
            cli.drafter,
            cli.target,
            tokenizer=cli.tokenizer,
            gamma=cli.gamma,
            max_gen_len=cli.gen_len,
            logits_processor=cli.processor,
            eos_tokens_id=cli.end_tokens,
        )
        t1 = time.time()

        spec_text = cli.tokenizer.decode(output_ids, skip_special_tokens=True)
        spec_tokens = len(output_ids)
        spec_tp = spec_tokens / max(1e-6, (t1 - t0))

        mean_skip = cli.drafter.model.pop_mean_skip()

        entry["speculative"] = {
            "output_text": spec_text,
            "tokens_generated": spec_tokens,
            "throughput_toks_per_sec": spec_tp,
            "acceptance_rate": acc_rate,
            "spec_target_model_calls_reported": target_calls,
            "mean_skip": mean_skip,
        }

        if not first:
            out_f.write(",\n")
        out_f.write(json.dumps(entry, indent=2))
        out_f.flush()           # ensure prompt-level write
        first = False
        

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    out_f.write("\n]\n")
    out_f.close()

    print(f"\nSaved results to {args.output}")

# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("MMLU Evaluation with SPADE")

    parser.add_argument("--draft_device", type=str, required=True)
    parser.add_argument("--target_device", type=str, required=True)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)

    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=6)

    parser.add_argument("--target_model", type=str, required=True,)
    parser.add_argument("--drafter_model", type=str, required=True,)

    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    main(args)
