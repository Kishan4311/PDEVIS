### Can't Wait to Verify : Parallel Drafting to Eliminate Verifier Idle Time in Speculative Decoding(PDEVIS)

**PDEVIS** is a plug-and-play acceleration framework for speculative decoding that parallelizes draft generation across transformer layers. By reducing verifier idle time using hidden-state divergence checks, PDEVIS improves GPU utilization and end-to-end latency without retraining, achieving up to **1.7× draft speedup** with minimal accuracy loss.

---

## ✨ Key Features

- Implements speculative decoding with **adaptive layer skipping**
- Introduces a **plug-and-play method** to parallelize transformer layer execution during the drafting stage
- Supports **LLaMA** and **Qwen** family models with configurable draft and target (verifier) models
- Provides evaluation suites for **SpecBench, WMT, MMLU, MGSM, MATH, and IFEval**
- Modular, easy-to-run scripts for **reproducible experiments** and extensibility
- Fully documented and suitable for **research use and production prototyping**

---

## 🚀 Quick Start

### Clone the Repository
```bash
git clone https://github.com/Kishan4311/PDEVIS.git
cd PDEVIS
````

### Install Requirements

```bash
pip install -r requirements.txt
```

### Hugging Face Authentication

```bash
huggingface-cli login
```

or from Python:

```python
from huggingface_hub import login
login("<YOUR_HF_TOKEN>")
```

---

## 💬 Run Interactive Chat

Launch the interactive inference UI:

```bash
python appInference.py
```

You may change the device (CPU or specific GPU) via command-line arguments if required.

---

## 🧠 Supported Models

### LLaMA Family

```text
llama-1b   -> meta-llama/Llama-3.2-1B-Instruct
llama-3b   -> meta-llama/Llama-3.2-3B-Instruct
llama-8b   -> meta-llama/Llama-3.1-8B-Instruct
llama-70b  -> meta-llama/Meta-Llama-3-70B-Instruct
```

### Qwen Family

```text
qwen-0.6b   -> Qwen/Qwen3-0.6B
qwen-1.7b   -> Qwen/Qwen3-1.7B
qwen-4b     -> Qwen/Qwen3-4B
qwen-8b     -> Qwen/Qwen3-8B
qwen-32b    -> Qwen/Qwen3-32B
qwen-4b-I   -> Qwen/Qwen3-4B-Instruct-2507
qwen-30b-I  -> Qwen/Qwen3-30B-A3B-Instruct-2507
qwen-80b-I  -> Qwen/Qwen3-Next-80B-A3B-Instruct
```

---

## 📊 Evaluation

> **Note:** For evaluation, use a larger target (verifier) model (e.g., LLaMA-3-70B or larger) and a smaller draft model. Update dataset paths and output locations as needed.

All evaluation scripts include detailed usage instructions at the top of each file.

---

### Evaluate WMT Dataset

```bash
python evaluation/wmt/evaluate_wmt.py \
  --draft_device cuda:0 \
  --target_device cuda:3 \
  --dataset Data/wmt.json \
  --code_path appInference.py \
  --k 200 \
  --gen_len 64 \
  --gamma 6 \
  --target_model llama-70b \
  --drafter_model llama-3b \
  --output evaluation/wmt/llama_wmt_results.json
```

---

### Evaluate SpecBench Dataset

```bash
python evaluation/specbench/evaluate_specbench.py \
  --code-path appInference.py \
  --specbench Data/Specbench_filtered.jsonl \
  --out evaluation/specbench/qwen_specbench_results.jsonl \
  --drafter-device cuda:1 \
  --target-device cuda:2 \
  --gamma 6 \
  --max-gen-len 64 \
  --max-examples 200 \
  --target-model qwen-30b-I \
  --drafter-model qwen-4b-I
```

---

## 📈 Result Summarization

### WMT Summary

```bash
python summary/wmt/summary_wmt.py \
  --input evaluation/wmt/llama_wmt_results.json \
  --output summary/wmt/llama_wmt_summary.json
```

### SpecBench Summary

```bash
python summary/specbench/summary_specbench.py \
  --input evaluation/specbench/qwen_specbench_results.jsonl \
  --output summary/specbench/qwen_specbench_summary.json
```

---

## ⚙️ Configuration & Model Selection

* Model names and device options are specified via command-line arguments
* For best cost–accuracy trade-offs:

  * Use a **small, fast model** as the drafter
  * Use a **large, high-quality model** as the target/verifier

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.
For major changes, please open an issue first to discuss the design.

---

## 📄 Citation

If you use this work in your research, please cite:

**Cannot Wait to Verify: Parallel Drafting to Eliminate Verifier Idle Time in Speculative Decoding**

```
