"""
Autoresearch loop orchestrator (local side).

Each iteration:
  1. Show what hyperparams to change in config.yaml
  2. Pause — user runs finetune.ipynb on Colab with the new config
  3. User pastes metrics back here (or run_eval is called automatically if GPU is local)
  4. Keep config if combined_accuracy improved, else revert
  5. Append result to results_log.csv

Inspired by karpathy/autoresearch — the key idea is:
  tweak one thing → measure → keep if better → repeat
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOG_PATH = Path(__file__).parent / "results_log.csv"
AGENT_DIR = Path(__file__).parent.parent / "agent"

# Ordered search space: each dict is one iteration's hyperparams to try.
# The loop cycles through these, picking the next untried config.
SEARCH_SPACE = [
    {"lora_r": 16, "learning_rate": 2.0e-4, "max_steps": 500,  "notes": "baseline"},
    {"lora_r": 8,  "learning_rate": 2.0e-4, "max_steps": 500,  "notes": "smaller lora_r=8"},
    {"lora_r": 32, "learning_rate": 1.0e-4, "max_steps": 500,  "notes": "larger lora_r=32, lower lr"},
    {"lora_r": 16, "learning_rate": 4.0e-4, "max_steps": 500,  "notes": "higher lr=4e-4"},
    {"lora_r": 16, "learning_rate": 2.0e-4, "max_steps": 1000, "notes": "2x steps"},
    {"lora_r": 16, "learning_rate": 2.0e-4, "max_steps": 500,
     "lora_target_modules": ["q_proj", "v_proj"], "notes": "minimal target modules"},
]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def append_log(metrics: dict):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def run_eval_subprocess() -> dict:
    """Run eval.py as a subprocess, parse the last JSON line of its output."""
    result = subprocess.run(
        ["uv", "run", "python", "eval.py"],
        capture_output=True, text=True, cwd=str(AGENT_DIR)
    )
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("eval.py failed — see stderr above.")
    lines = result.stdout.strip().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Could not parse metrics from eval.py output.")


def prompt_manual_metrics(iteration: int) -> dict:
    """Fallback: ask user to paste metrics from Colab output."""
    print("\nPaste the JSON metrics line from your Colab notebook output,")
    print('then press Enter twice (e.g.  {"format_accuracy": 0.82, ...} )')
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    raw = " ".join(lines)
    try:
        m = json.loads(raw)
        m.setdefault("iteration", iteration)
        m.setdefault("timestamp", datetime.now().isoformat())
        return m
    except json.JSONDecodeError as e:
        print(f"Could not parse: {e}. Enter metrics manually.")
        return {
            "iteration": iteration,
            "format_accuracy": float(input("  format_accuracy (0–1): ")),
            "answer_accuracy": float(input("  answer_accuracy (0–1): ")),
            "combined_accuracy": float(input("  combined_accuracy (0–1): ")),
            "timestamp": datetime.now().isoformat(),
        }


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto-eval", action="store_true",
        help="Run eval.py automatically instead of prompting for manual input"
    )
    parser.add_argument("--max-iter", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config()
    best_acc = cfg.get("best_combined_accuracy", 0.0)
    best_cfg = cfg.copy()
    iteration = cfg.get("iteration", 1)

    print(f"\n{'='*60}")
    print(f"  Autoresearch Loop  —  max {args.max_iter} iterations")
    print(f"  Best accuracy so far: {best_acc:.2%}")
    print(f"{'='*60}")

    for _ in range(args.max_iter):
        idx = (iteration - 1) % len(SEARCH_SPACE)
        tweaks = SEARCH_SPACE[idx]

        # Build new config from best known config + tweaks
        new_cfg = best_cfg.copy()
        new_cfg.update(tweaks)
        new_cfg["iteration"] = iteration
        save_config(new_cfg)

        print(f"\n{'─'*60}")
        print(f"  Iteration {iteration}  |  Config: {tweaks['notes']}")
        print(f"    lora_r={new_cfg['lora_r']}  lr={new_cfg['learning_rate']}  steps={new_cfg['max_steps']}")
        print(f"{'─'*60}")

        if not args.auto_eval:
            print("\nSteps:")
            print("  1. Commit + push this repo so Colab can pull the latest config.yaml")
            print("  2. Open colab/finetune.ipynb in Google Colab (Pro A100)")
            print("  3. Run all cells — adapter will be pushed to HF Hub")
            print("  4. Copy the metrics JSON printed at the end of the notebook")
            input("\nPress ENTER when Colab training is done...\n")

        # Get metrics
        metrics = {}
        if args.auto_eval:
            try:
                metrics = run_eval_subprocess()
            except Exception as e:
                print(f"Auto eval failed: {e}")
                metrics = prompt_manual_metrics(iteration)
        else:
            metrics = prompt_manual_metrics(iteration)

        metrics.setdefault("timestamp", datetime.now().isoformat())
        metrics.setdefault("lora_r", new_cfg["lora_r"])
        metrics.setdefault("learning_rate", new_cfg["learning_rate"])
        metrics.setdefault("max_steps", new_cfg["max_steps"])
        metrics.setdefault("notes", tweaks["notes"])

        combined = float(metrics.get("combined_accuracy", 0))
        append_log(metrics)

        print(f"\n  Result: combined={combined:.2%}  (best={best_acc:.2%})")

        if combined > best_acc:
            print("  ✓ IMPROVEMENT — keeping this config as new best")
            best_acc = combined
            best_cfg = new_cfg.copy()
            best_cfg["best_combined_accuracy"] = best_acc
            save_config(best_cfg)
        else:
            print("  ✗ No improvement — reverting to previous best config")
            revert = best_cfg.copy()
            revert["iteration"] = iteration + 1
            save_config(revert)

        iteration += 1

        cont = input("\nContinue to next iteration? (y/n): ").strip().lower()
        if cont != "y":
            break

    print(f"\nAutoresearch complete.  Best combined accuracy: {best_acc:.2%}")
    print(f"Results saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
