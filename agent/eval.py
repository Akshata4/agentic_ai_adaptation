"""
Evaluation script.
Loads the fine-tuned adapter, runs it over the held-out eval split of
Salesforce/xlam-function-calling-60k, and reports:
  - format_accuracy    : % of outputs that are valid {"name":..., "arguments":...} JSON
  - toolname_accuracy  : % of outputs where the tool name matches the ground truth
  - argument_accuracy  : % of outputs where ALL arguments match the ground truth
  - combined_accuracy  : weighted average of all three (weights from config.yaml)
"""

import json
import sys
import yaml
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))

from agent import load_model, generate

CONFIG_PATH = Path(__file__).parent.parent / "autoresearch" / "config.yaml"


def load_eval_split(cfg: dict):
    dataset = load_dataset(cfg["dataset_name"], split="train")
    n = len(dataset)
    start = int(n * cfg["train_split_ratio"])
    end = min(start + cfg["eval_samples"], n)
    return dataset.select(range(start, end))


def _normalize(value):
    """Normalize a value for loose comparison: lowercase strings, parse numeric strings."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value.strip().lower()
    return value


def _args_match(predicted: dict, expected: dict) -> bool:
    """
    Check if predicted arguments match expected arguments.
    All keys in expected must be present in predicted with matching values.
    Extra keys in predicted are ignored (model may add optional params).
    """
    for key, exp_val in expected.items():
        if key not in predicted:
            return False
        if _normalize(predicted[key]) != _normalize(exp_val):
            return False
    return True


def check_format(result: dict) -> bool:
    if not result["parse_ok"]:
        return False
    tc = result["tool_call"]
    return isinstance(tc, dict) and "name" in tc and "arguments" in tc


def check_toolname(result: dict, expected_answers: list) -> bool:
    if not check_format(result):
        return False
    predicted = result["tool_call"].get("name", "")
    return any(a.get("name", "") == predicted for a in expected_answers)


def check_arguments(result: dict, expected_answers: list) -> bool:
    """
    Tool name must match AND all expected arguments must match.
    Tries each expected answer — passes if any one matches fully.
    """
    if not check_format(result):
        return False
    tc = result["tool_call"]
    predicted_name = tc.get("name", "")
    predicted_args = tc.get("arguments", {})

    for answer in expected_answers:
        if answer.get("name", "") != predicted_name:
            continue
        expected_args = answer.get("arguments", {})
        if _args_match(predicted_args, expected_args):
            return True
    return False


def run_eval(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[Eval] Iteration {cfg['iteration']} | Adapter: {cfg['adapter_repo']}")
    model, tokenizer, device = load_model(config_path)

    eval_data = load_eval_split(cfg)
    print(f"[Eval] {len(eval_data)} samples | metrics: format + toolname + arguments\n")

    fmt_ok = name_ok = args_ok = total = 0

    for i, row in enumerate(eval_data):
        query = row.get("query", "").strip()
        answers_raw = row.get("answers", "[]")
        if not query:
            continue

        expected = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        if not expected:
            continue

        result = generate(model, tokenizer, device, query)

        fmt_ok  += int(check_format(result))
        name_ok += int(check_toolname(result, expected))
        args_ok += int(check_arguments(result, expected))
        total   += 1

        if (i + 1) % 25 == 0:
            print(
                f"  [{i+1}/{len(eval_data)}] "
                f"format={fmt_ok/total:.1%}  "
                f"toolname={name_ok/total:.1%}  "
                f"args={args_ok/total:.1%}"
            )

    if total == 0:
        raise RuntimeError("No valid eval samples found.")

    fmt_acc  = fmt_ok  / total
    name_acc = name_ok / total
    args_acc = args_ok / total

    w_fmt  = cfg.get("tool_format_weight", 0.34)
    w_name = cfg.get("answer_accuracy_weight", 0.33)
    w_args = cfg.get("argument_accuracy_weight", 0.33)
    combined = w_fmt * fmt_acc + w_name * name_acc + w_args * args_acc

    metrics = {
        "iteration":         cfg["iteration"],
        "adapter_repo":      cfg["adapter_repo"],
        "lora_r":            cfg["lora_r"],
        "learning_rate":     cfg["learning_rate"],
        "max_steps":         cfg["max_steps"],
        "format_accuracy":   round(fmt_acc,  4),
        "toolname_accuracy": round(name_acc, 4),
        "argument_accuracy": round(args_acc, 4),
        "combined_accuracy": round(combined, 4),
        "eval_samples":      total,
        "notes":             cfg.get("notes", ""),
    }

    print(f"\n{'─'*52}")
    print(f"  Format accuracy   : {fmt_acc:.2%}")
    print(f"  Tool-name accuracy: {name_acc:.2%}")
    print(f"  Argument accuracy : {args_acc:.2%}  ← new harder metric")
    print(f"  Combined          : {combined:.2%}  (best so far: {cfg['best_combined_accuracy']:.2%})")
    print(f"{'─'*52}")
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    run_eval()
