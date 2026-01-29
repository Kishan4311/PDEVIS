#!/usr/bin/env python3
"""
evaluate_specbench.py

Usage example:
python /home/iitb/Kishan_SpecDec/_spade2/evaluation/specbench/evaluate_specbench.py \
  --code-path "/home/iitb/Kishan_SpecDec/_spade2/appInference2.py" \
  --specbench "/home/iitb/Kishan_SpecDec/_spade2/Data/Specbench_filtered.jsonl" \
  --out "/home/iitb/Kishan_SpecDec/_spade2/evaluation/specbench/qwen_specbench_results_gm6_4b_30b_genLen64.jsonl" \
  --drafter-device cuda:1 \
  --target-device cuda:2  \
  --gamma 6 \
  --max-gen-len 64 \
  --max-examples 200 \
  --target-model "qwen-30b-I" \
  --drafter-model "qwen-4b-I"

"""


import time
import json
import os
import argparse
import importlib.util
import sys
import math
from typing import List, Optional

import random
import torch
import numpy as np
from tqdm import tqdm

# ---------------------------
# Helper: load module by file
# ---------------------------
def load_user_module(module_path: str, mod_name: str = "user_specdec"):
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

# ---------------------------------------------------------
# Wrapper that measures time and call count of model forward
# ---------------------------------------------------------
class ModelWithTimer:
    def __init__(self, model):
        self._model = model
        self.time_sec = 0.0
        self.calls = 0

    # forward the call and measure the wall-clock time
    def __call__(self, *args, **kwargs):
        t0 = time.time()
        out = self._model(*args, **kwargs)
        t1 = time.time()
        self.time_sec += (t1 - t0)
        self.calls += 1
        return out

    # forward attribute access to the underlying model (device, config, etc.)
    def __getattr__(self, name):
        return getattr(self._model, name)

# ---------------------------------------------------------
# Prompt preparation (use tokenizer.apply_chat_template if available)
# ---------------------------------------------------------
def prepare_prompt_from_turns(turns: List[str], tokenizer, chat=True):
    # turns is a list of strings; join them in a way appropriate for your dataset.
    # Best-effort: if tokenizer provides apply_chat_template, use it with the first turn as a user message.
    try:
        if chat and hasattr(tokenizer, "apply_chat_template"):
            # if there are several turns, join them separated by two newlines
            prompt_text = "\n\n".join(turns)
            # The Llama tokenizer used in your code expects a list-of-dicts with role/content.
            # For generality we pass the whole combined text as a single user message.
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False)
    except Exception:
        # fall back to join below
        pass
    # fallback: simply join the turns with blank lines
    return "\n\n".join(turns)

def pop_mean_skip_safe(model):
    """
    Safely extract and reset mean_skip from LlamaModel.
    Works for raw models and ModelWithTimer wrappers.
    """
    try:
        base = model._model if hasattr(model, "_model") else model
        if hasattr(base, "model") and hasattr(base.model, "pop_mean_skip"):
            return float(base.model.pop_mean_skip())
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------
def evaluate(
    code_path: str,
    specbench_path: str,
    out_jsonl: str,
    device: str = "cuda",
    target_device: Optional[str] = None,
    drafter_device: Optional[str] = None,
    gamma: int = 4,
    max_gen_len: int = 64,
    processor_name: str = "greedy",
    processor_args = {"temperature": 1.0},
    max_examples: int | None = None,
    seed: int = 42,
    target_model: Optional[str] = None,
    drafter_model: Optional[str] = None,
):
    assert os.path.exists(code_path), f"Code file not found: {code_path}"
    assert os.path.exists(specbench_path), f"SpecBench file not found: {specbench_path}"

    # 1) import user's file as module
    mod = load_user_module(code_path, "specdec_usercode")

    # 2) instantiate the InferenceCLI (this will load models)
    print("Instantiating InferenceCLI (this will load models) ...")
    cli = mod.InferenceCLI(device=device, 
                           device_target=target_device,
                           device_drafter=drafter_device,
                           target_model=mod.ModelsCatalog.model_id(target_model),
                           drafter_model=mod.ModelsCatalog.model_id(drafter_model))  # this prints info and loads models
    target = cli.target
    drafter = cli.drafter
    tokenizer = cli.tokenizer

    # 3) build processor map (use classes from user module)
    processors = {
        "greedy": mod.GreedyProcessor,
    }
    if processor_name not in processors:
        raise ValueError(f"Processor {processor_name} not available. Choose from {list(processors.keys())}")
    processor_class = processors[processor_name]
    processor = processor_class(**processor_args)

    # 4) load SpecBench dataset
    records = []
    with open(specbench_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    n_total = len(records)
    if max_examples is None:
        max_examples = n_total

    # 5) prepare output file
    out_dir = os.path.dirname(out_jsonl) or "."
    os.makedirs(out_dir, exist_ok=True)
    fout = open(out_jsonl, "w", encoding="utf-8")

    # stats accumulators (kept for summary; per-category aggregation will be done later)
    # baseline_target_throughputs = []
    # baseline_draft_throughputs = []
    # spec_throughputs = []
    # acceptance_rates = []
    # target_calls_list = []
    # target_time_list = []
    # draft_time_list = []

    # successes = 0
    failures = 0

    # helper for deterministic generation
    def set_seed_local(s):
        # your CLI had _set_seed; call it to get same behavior if available
        try:
            cli._set_seed(s)
        except Exception:
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

    # iterate
    for i, rec in enumerate(tqdm(records[:max_examples], desc="Examples")):
        try:
            qid = rec.get("question_id", i)
            category = rec.get("category", "unknown")
            turns = rec.get("turns", [])
            prompt_text = prepare_prompt_from_turns(turns, tokenizer, chat=True)

            # Tokenize to ids (list)
            tokenized = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            # ---------- Baseline (autoregressive on target) ----------
            set_seed_local(seed)
            t0 = time.time()
            ar_target_ids = mod.autoregressive_generate(
                tokenized,
                target,
                max_gen_len=max_gen_len,
                logits_processor=processor,
                eos_tokens_id=cli.end_tokens,
                pad_token_id=pad_id,
            )
            t1 = time.time()
            ar_target_time = t1 - t0
            if ar_target_ids is None:
                ar_target_ids = []
            ar_target_text = tokenizer.decode(ar_target_ids, skip_special_tokens=True)
            ar_target_num_tokens = len(ar_target_ids)
            ar_target_throughput = ar_target_num_tokens / max(ar_target_time, 1e-9)
            target_mean_skip = pop_mean_skip_safe(target)


            # ---------- Baseline (autoregressive on drafter) ----------
            set_seed_local(seed)
            t0 = time.time()
            ar_draft_ids = mod.autoregressive_generate(
                tokenized,
                drafter,
                max_gen_len=max_gen_len,
                logits_processor=processor,
                eos_tokens_id=cli.end_tokens,
                pad_token_id=pad_id,

            )
            t1 = time.time()
            ar_draft_time = t1 - t0
            if ar_draft_ids is None:
                ar_draft_ids = []
            ar_draft_text = tokenizer.decode(ar_draft_ids, skip_special_tokens=True)
            ar_draft_num_tokens = len(ar_draft_ids)
            ar_draft_throughput = ar_draft_num_tokens / max(ar_draft_time, 1e-9)
            draft_mean_skip = pop_mean_skip_safe(drafter)


            # ---------- Speculative (wrap both drafter and target to measure times separately) ----------
            wrapped_target = ModelWithTimer(target)
            wrapped_drafter = ModelWithTimer(drafter)

            set_seed_local(seed)
            t0 = time.time()
            spec_output_ids, accept_rate, target_calls_reported = mod.speculative_generate(
                tokenized,
                wrapped_drafter,
                wrapped_target,
                tokenizer=tokenizer,
                gamma=gamma,
                logits_processor=processor,
                max_gen_len=max_gen_len,
                eos_tokens_id=cli.end_tokens,
                pad_token_id=pad_id,
             
            )
            t1 = time.time()
            spec_total_time = t1 - t0

            if spec_output_ids is None:
                spec_output_ids = []
            spec_text = tokenizer.decode(spec_output_ids, skip_special_tokens=True)
            spec_num_tokens = len(spec_output_ids)
            spec_throughput = spec_num_tokens / max(spec_total_time, 1e-9)

            # measured per-model times & calls during spec
            measured_target_calls = wrapped_target.calls
            # measured_target_time = wrapped_target.time_sec
            # measured_draft_calls = wrapped_drafter.calls
            # measured_draft_time = wrapped_drafter.time_sec
            
            spec_draft_mean_skip = pop_mean_skip_safe(wrapped_drafter)
            # spec_target_mean_skip = pop_mean_skip_safe(wrapped_target)

            # accumulate stats
            # baseline_target_throughputs.append(ar_target_throughput)
            # baseline_draft_throughputs.append(ar_draft_throughput)
            # spec_throughputs.append(spec_throughput)
            # acceptance_rates.append(float(accept_rate) if accept_rate is not None else None)
            # target_calls_list.append(int(measured_target_calls))
            # target_time_list.append(float(measured_target_time))
            # draft_time_list.append(float(measured_draft_time))

            # write per-prompt result
            item = {
                "question_id": qid,
                "category": category,
                "prompt_text": prompt_text,
                "target": {
                    "output_text": ar_target_text,
                    "tokens_generated": ar_target_num_tokens,
                    "throughput_toks_per_sec": ar_target_throughput,
                    "mean_skip": target_mean_skip,
                },
                "draft": {
                    "output_text": ar_draft_text,
                    "tokens_generated": ar_draft_num_tokens,
                    "throughput_toks_per_sec": ar_draft_throughput,
                    "mean_skip": draft_mean_skip,
                },
                "speculative": {
                    "output_text": spec_text,
                    "tokens_generated": spec_num_tokens,
                    "throughput_toks_per_sec": spec_throughput,
                    "acceptance_rate": float(accept_rate) if accept_rate is not None else None,
                    "target_calls_reported_by_algo": int(target_calls_reported) if target_calls_reported is not None else None,
                    "mean_skip": spec_draft_mean_skip,
                },
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
            # successes += 1

        except Exception as e:
            failures += 1
            print(f"[ERROR] example index {i} failed: {e}")
            err_obj = {"question_id": rec.get("question_id", i), "error": str(e)}
            fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            fout.flush()
            continue

    fout.close()

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", required=True, help="Path to your speculative code file (python), e.g. '/mnt/data/llama_3_2_1b_specdec_v2 (1).py'")
    parser.add_argument("--specbench", required=True, help="Path to SpecBench jsonl")
    parser.add_argument("--out", required=True, help="Output JSONL file for per-prompt results")
    parser.add_argument("--device", default="cuda", help="Device to pass to InferenceCLI (cuda/cpu)")
    parser.add_argument("--gamma", type=int, default=4, help="Gamma (drafts) for speculative decoding")
    parser.add_argument("--max-gen-len", type=int, default=64, help="Max generation length")
    parser.add_argument("--processor", default="greedy", choices=["greedy","multinomial","topk","nucleus","topknucleus"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-model", type=str, default=None, help="HuggingFace repo id or local path to use as TARGET model (overrides specdec defaults)")
    parser.add_argument("--drafter-model", type=str, default=None, help="HuggingFace repo id or local path to use as DRAFTER model (overrides specdec defaults)")
    parser.add_argument("--target-device", type=str, default=None,
                    help="Device for target model, e.g. cuda:2")
    parser.add_argument("--drafter-device", type=str, default=None,
                    help="Device for drafter model, e.g. cuda:0")

    args = parser.parse_args()

    evaluate(
        code_path=args.code_path,
        specbench_path=args.specbench,
        out_jsonl=args.out,
        # summary_json=args.summary,
        device=args.device,
        target_device=args.target_device,
        drafter_device=args.drafter_device,
        gamma=args.gamma,
        max_gen_len=args.max_gen_len,
        processor_name=args.processor,
        processor_args={"temperature": 1.0},
        max_examples=args.max_examples,
        seed=args.seed,
        target_model=args.target_model,
        drafter_model=args.drafter_model,
    )
