"""
python /home/iitb/Kishan_SpecDec/_spade2/evaluation/wmt/ablation_wmt.py \
--dataset /home/iitb/Kishan_SpecDec/_spade2/Data/wmt.json \
--model_name "Qwen/Qwen3-4B-Instruct-2507" \
--gen_len 64 \
--k 80 \
--output /home/iitb/Kishan_SpecDec/_spade2/evaluation/wmt/ablation_k80_point92_wmt.json

"""
#!/usr/bin/env python3
import json
import time
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_wmt(path, k=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if k is None else data[:k]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto"
    )
    model.eval()

    # 🔑 ENABLE REUSE / SKIP TRACKING
    assert hasattr(model, "model"), "Unexpected model structure"
    model.model.enable_reuse = True

    dataset = load_wmt(args.dataset, args.k)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Running {len(dataset)} WMT samples with reuse enabled")

    # -------------------------------------------------
    # Open output file (incremental JSON array)
    # -------------------------------------------------
    out_f = open(args.output, "w", encoding="utf-8")
    out_f.write("[\n")
    first = True

    for item in tqdm(dataset):
        prompt = item["prompt"][0]

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            # reset reuse counters per prompt
            model.model._reuse_skip_accum = 0
            model.model._reuse_token_count = 0

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.gen_len,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize()
        t1 = time.time()

        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        tokens_generated = output_ids.numel()
        throughput = tokens_generated / max(1e-6, (t1 - t0))

        mean_skip = model.model.pop_mean_skip()

        entry = {
            "question_id": item["question_id"],
            "category": "wmt",
            "prompt": prompt,
            "label": item["label"][0],
            "output": {
                "output_text": output_text,
                "tokens_generated": tokens_generated,
                "throughput_toks_per_sec": throughput,
                "mean_skip": mean_skip
            }
        }

        # -------------------------------------------------
        # Incremental write
        # -------------------------------------------------
        if not first:
            out_f.write(",\n")
        out_f.write(json.dumps(entry, indent=2))
        out_f.flush()
        first = False

    # -------------------------------------------------
    # Close JSON array
    # -------------------------------------------------
    out_f.write("\n]\n")
    out_f.close()

    print(f"\n✅ Saved results to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WMT Qwen reuse inference")

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    main(args)
