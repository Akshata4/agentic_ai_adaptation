# Adaptation of Agentic AI — Code Reproduction

**Paper**: [Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills](https://arxiv.org/abs/2512.16301)  
**Author**: Akshata Madavi  
**Course**: Agentic AI Adaptation (Individual Assignment)

---

## Deliverables

| Deliverable | Link |
|-------------|------|
| Medium article | [Teaching an LLM to Use Tools](https://medium.com/@akshatamadavi/teaching-an-llm-to-use-tools-fine-tuning-llama-3-2-with-a-karpathy-style-autoresearch-loop-7ccb25c43579) |
| Slide deck (Slideshare) | [Slides](https://github.com/Akshata4/agentic_ai_adaptation/blob/main/Automated_LLM_Tool_SFT.pdf) |
| Video walkthrough (YouTube) | https://www.youtube.com/watch?v=d77__gzxjt0 |
| HuggingFace adapter | `adapter_repo` in `config.yaml` |

---

## What This Project Reproduces

This project implements the **A1 paradigm** from the survey — *supervised fine-tuning of an LLM agent* to perform structured tool calling — and wraps it in a **Karpathy-style autoresearch loop** that iterates over hyperparameters and keeps only improvements.

The survey describes four adaptation paradigms for agentic AI:

| ID | Paradigm | What it does |
|----|----------|--------------|
| **A1** | SFT of the agent | Fine-tune the model on task demonstrations ← **reproduced here** |
| A2 | RL of the agent | Reinforce via rewards (RLHF, PPO, GRPO) |
| T1 | Tool/skill modules | Reusable plug-in tools the agent calls |
| T2 | Memory / RAG | Retrieval-augmented generation from a memory store |

---

## Architecture

```
[Colab Notebook]                     [Local Agent]
       |                                    |
Load base model (Llama 3.2 3B)       Load fine-tuned adapter
       |                                    |
Fine-tune on xlam-60k (LoRA)         Feed test questions
       |                                    |
Save adapter → HF Hub  ─────────>   Model generates tool calls
       |                                    |
Log accuracy (metric)                Execute tools, check answers
       |                                    |
       └──────── autoresearch loop ─────────┘
              tweak config.yaml → retrain → re-evaluate → keep/discard
```

---

## Stack

| Component | Choice |
|-----------|--------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuning | LoRA / QLoRA via [unsloth](https://github.com/unslothai/unsloth) + SFTTrainer (trl) |
| Dataset | `Salesforce/xlam-function-calling-60k` |
| Training hardware | Google Colab Pro A100 |
| Inference (local) | HuggingFace `transformers` + `peft` |
| Tools | Wikipedia API, Calculator |
| Metrics | Format accuracy + Tool-name accuracy + Argument accuracy (weighted combined) |
| Autoresearch | Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) |

---

## Repository Structure

```
agentic_ai_adaptation/
├── README.md
│
├── colab/
│   ├── finetune.ipynb       # Colab Pro A100 — QLoRA fine-tuning + eval + HF push
│   └── requirements.txt
│
├── agent/
│   ├── tools.py             # Wikipedia + calculator tool definitions + registry
│   ├── agent.py             # Load adapter, build prompt, generate tool call
│   ├── eval.py              # Run eval on held-out xlam split, print metrics JSON
│   └── requirements.txt
│
├── autoresearch/
│   ├── config.yaml          # Single source of truth — hyperparams for each iteration
│   ├── run_loop.py          # Orchestrator: propose → wait → eval → keep/discard
│   └── results_log.csv      # One row per iteration
│
├── results/
│   ├── generate_chart.py    # Reads results_log.csv → accuracy_chart.png + iteration_table.md
│   ├── accuracy_chart.png   # Generated after runs
│   └── iteration_table.md   # Generated after runs
│
├── slides/                  # Slide deck (PDF / PPTX)
├── medium_link.txt          # Medium article URL
└── video_link.txt           # YouTube + Slideshare URLs
```

---

## Setup

### 1. Colab (training)

1. Open `colab/finetune.ipynb` in [Google Colab](https://colab.research.google.com) with **A100 GPU**
2. Add your HuggingFace token to Colab Secrets as `HF_TOKEN`
3. In **Cell 2**, set `GITHUB_REPO` to this repo's URL
4. Update `config.yaml` with your HF username in `adapter_repo`
5. Run all cells

### 2. Local (agent + autoresearch loop)

Requires [uv](https://docs.astral.sh/uv/) — install once with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
# Install all dependencies into an isolated .venv
uv sync

# Install with optional viz tools (for generate_chart.py)
uv sync --group viz

# Run eval against a trained adapter
uv run python agent/eval.py

# Run the autoresearch loop
uv run python autoresearch/run_loop.py

# Add a new dependency
uv add <package>
```

> **Note**: Local inference requires a GPU or Apple Silicon (MPS). The agent auto-detects the best available device. `bitsandbytes` (4-bit quant) is used only on CUDA; it is safe to ignore install warnings on Mac.

---

## Running the Autoresearch Loop

```bash
python autoresearch/run_loop.py
```

Each iteration:
1. `run_loop.py` updates `config.yaml` with the next hyperparams from the search space
2. You push the repo, run `finetune.ipynb` on Colab
3. Copy the metrics JSON from the last Colab cell, paste into the terminal
4. `run_loop.py` compares combined accuracy and keeps/discards the config
5. Results are appended to `autoresearch/results_log.csv`

After all runs, generate charts:
```bash
pip install matplotlib
python results/generate_chart.py
```

---

## Hyperparameter Search Space

| Iter | lora_r | lr | max_steps | train_samples | Notes |
|------|--------|----|-----------|---------------|-------|
| 1 | 16 | 2e-4 | 500 | 5000 | baseline |
| 2 | 16 | 2e-4 | 500 | 10000 | 2× training data |
| 3 | 16 | 2e-4 | 1000 | 5000 | 2× steps |
| 4 | 16 | 1e-4 | 1000 | 5000 | lower lr + 2× steps |
| 5 | 16 | 2e-4 | 1000 | 10000 | 2× data + 2× steps |
| 6 | 32 | 1e-4 | 1000 | 10000 | lora_r=32 + 2× data + 2× steps |

---

## Results

10 iterations completed. Best combined accuracy: **97.20%** (iteration 3).

| Iter | lora_r | lr | max_steps | Format | Tool Name | Arguments | Combined | Notes |
|------|--------|----|-----------|--------|-----------|-----------|----------|-------|
| 1 | 16 | 2e-4 | 500 | 100.00% | 100.00% | — | 100.00% | baseline (pre-args metric) |
| 2 | 8 | 2e-4 | 500 | 100.00% | 100.00% | — | 100.00% | smaller lora_r=8 (pre-args metric) |
| 3 | 16 | 2e-4 | 500 | 100.00% | 100.00% | 91.50% | **97.20%** | ← best |
| 4 | 16 | 4e-4 | 500 | 99.50% | 99.50% | 90.00% | 96.37% | higher lr=4e-4 |
| 5 | 16 | 2e-4 | 500 | 100.00% | 100.00% | 90.50% | 96.87% | baseline |
| 6 | 16 | 2e-4 | 500 | 99.50% | 99.00% | 88.50% | 95.71% | minimal target modules |
| 7 | 16 | 2e-4 | 500 | 100.00% | 100.00% | 91.00% | 97.03% | baseline |
| 8 | 16 | 2e-4 | 500 | 100.00% | 100.00% | 91.00% | 97.03% | 2× training data |
| 9 | 16 | 2e-4 | 1000 | 100.00% | 100.00% | 90.00% | 96.70% | 2× steps |
| 10 | 16 | 1e-4 | 1000 | 100.00% | 100.00% | 90.50% | 96.87% | lower lr + 2× steps |

**Key findings:**
- Format and tool-name accuracy saturate near 100% across all configs — the model learns the output schema reliably
- Argument accuracy (91.50% best) is the bottleneck — exact value matching is harder than structure
- Higher learning rate (iter 4) and minimal LoRA target modules (iter 6) both hurt argument accuracy
- More training data and longer training did not improve beyond the baseline

See [results/iteration_table.md](results/iteration_table.md) and [results/accuracy_chart.png](results/accuracy_chart.png).


---

## Paper Reference

Jiang, P. et al. (2025). *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills*. arXiv:2512.16301. https://arxiv.org/abs/2512.16301
