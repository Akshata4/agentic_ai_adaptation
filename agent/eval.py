"""
Evaluation script.
Loads the fine-tuned adapter, runs it over the held-out eval split of
Salesforce/xlam-function-calling-60k, and reports:
  - format_accuracy   : % of outputs that are valid {"name":..., "arguments":...} JSON
  - answer_accuracy   : % of outputs where the tool name matches the ground truth
  - combined_accuracy : weighted average of the two (weights from config.yaml)
"""

import json
import sys
import yaml
from pathlib import Path
from datasets import load_dataset

# Allow running from repo root or from /agent
sys.path.insert(0, str(Path(__file__).parent))

from agent import load_model, generate, CONFIG_PATH

CONFIG_PATH = Path(__file__).parent.parent / "autoresearch" / "config.yaml"


def load_eval_split(cfg: dict) -> list:
    dataset = load_dataset(cfg["dataset_name"], split="train")
    n = len(dataset)
    start = int(n * cfg["train_split_ratio"])
    end = min(start + cfg["eval_samples"], n)
    return dataset.select(range(start, end))


def check_format(result: dict) -> bool:
    if not result["parse_ok"]:
        return False
    tc = result["tool_call"]
    return isinstance(tc, dict) and "name" in tc and "arguments" in tc


def check_answer(result: dict, expected_answers: list) -> bool:
    if not check_format(result):
        return False
    predicted_tool = result["tool_call"].get("name", "")
    # Accept if predicted tool matches any expected tool call
    return any(a.get("name", "") == predicted_tool for a in expected_answers)


def run_eval(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[Eval] Iteration {cfg['iteration']} | Adapter: {cfg['adapter_repo']}")
    model, tokenizer, device = load_model(config_path)

    eval_data = load_eval_split(cfg)
    print(f"[Eval] {len(eval_data)} samples from {cfg['dataset_name']}")

    fmt_ok = ans_ok = total = 0

    for i, row in enumerate(eval_data):
        query = row.get("query", "").strip()
        answers_raw = row.get("answers", "[]")
        if not query:
            continue

        expected = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        if not expected:
            continue

        result = generate(model, tokenizer, device, query)
        fmt_ok += int(check_format(result))
        ans_ok += int(check_answer(result, expected))
        total += 1

        if (i + 1) % 25 == 0:
            print(
                f"  [{i+1}/{len(eval_data)}] "
                f"format={fmt_ok/total:.1%}  answer={ans_ok/total:.1%}"
            )

    if total == 0:
        raise RuntimeError("No valid eval samples found.")

    fmt_acc = fmt_ok / total
    ans_acc = ans_ok / total
    combined = cfg["tool_format_weight"] * fmt_acc + cfg["answer_accuracy_weight"] * ans_acc

    metrics = {
        "iteration": cfg["iteration"],
        "adapter_repo": cfg["adapter_repo"],
        "lora_r": cfg["lora_r"],
        "learning_rate": cfg["learning_rate"],
        "max_steps": cfg["max_steps"],
        "format_accuracy": round(fmt_acc, 4),
        "answer_accuracy": round(ans_acc, 4),
        "combined_accuracy": round(combined, 4),
        "eval_samples": total,
        "notes": cfg.get("notes", ""),
    }

    print(f"\n{'─'*50}")
    print(f"  Format accuracy : {fmt_acc:.2%}")
    print(f"  Answer accuracy : {ans_acc:.2%}")
    print(f"  Combined        : {combined:.2%}  (best so far: {cfg['best_combined_accuracy']:.2%})")
    print(f"{'─'*50}")
    # Print as JSON so run_loop.py can parse the last JSON line
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    run_eval()
