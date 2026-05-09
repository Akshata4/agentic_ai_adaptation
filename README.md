# Adaptation of Agentic AI — Code Reproduction

**Paper**: [Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills](https://arxiv.org/abs/2512.16301)  
**Author**: Akshata Madavi  
**Course**: Agentic AI Adaptation (Individual Assignment)

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
| Metrics | Format accuracy + Answer accuracy |
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

| Iter | lora_r | lr | max_steps | Notes |
|------|--------|----|-----------|-------|
| 1 | 16 | 2e-4 | 500 | baseline |
| 2 | 8 | 2e-4 | 500 | smaller adapter |
| 3 | 32 | 1e-4 | 500 | larger adapter, lower lr |
| 4 | 16 | 4e-4 | 500 | higher learning rate |
| 5 | 16 | 2e-4 | 1000 | 2× training steps |
| 6 | 16 | 2e-4 | 500 | minimal target modules |

---

## Results

See [results/iteration_table.md](results/iteration_table.md) and [results/accuracy_chart.png](results/accuracy_chart.png) — populated after running the loop.

---

## Deliverables

| Deliverable | Link |
|-------------|------|
| Medium article | see `medium_link.txt` |
| Slide deck (Slideshare) | see `video_link.txt` |
| Video walkthrough (YouTube) | see `video_link.txt` |
| HuggingFace adapter | `adapter_repo` in `config.yaml` |

---

## Paper Reference

Jiang, P. et al. (2025). *Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills*. arXiv:2512.16301. https://arxiv.org/abs/2512.16301
