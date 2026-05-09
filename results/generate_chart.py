"""
Run after autoresearch iterations to produce:
  results/accuracy_chart.png  — line chart of metrics per iteration
  results/iteration_table.md  — markdown table summary
"""

import csv
import sys
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "autoresearch" / "results_log.csv"
CHART_PATH = Path(__file__).parent / "accuracy_chart.png"
TABLE_PATH = Path(__file__).parent / "iteration_table.md"


def load_results() -> list[dict]:
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        sys.exit(f"No results yet — run run_loop.py first. ({LOG_PATH})")
    with open(LOG_PATH) as f:
        return list(csv.DictReader(f))


def make_chart(rows: list[dict]):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not installed — skipping chart. Run: pip install matplotlib")
        return

    iters = [int(r["iteration"]) for r in rows]
    fmt   = [float(r["format_accuracy"]) * 100  for r in rows]
    ans   = [float(r["answer_accuracy"]) * 100   for r in rows]
    comb  = [float(r["combined_accuracy"]) * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, fmt,  marker="o", label="Format accuracy")
    ax.plot(iters, ans,  marker="s", label="Answer accuracy")
    ax.plot(iters, comb, marker="^", label="Combined", linewidth=2, linestyle="--")

    ax.set_xlabel("Autoresearch Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Tool-Call Accuracy Across Autoresearch Iterations\n(Llama 3.2 3B + LoRA on xlam-function-calling-60k)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(iters)

    # Annotate best combined
    best_idx = comb.index(max(comb))
    ax.annotate(
        f"Best: {max(comb):.1f}%",
        (iters[best_idx], comb[best_idx]),
        textcoords="offset points", xytext=(5, 8),
        fontsize=9, color="tab:green"
    )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"Chart saved: {CHART_PATH}")


def make_table(rows: list[dict]):
    lines = [
        "# Autoresearch Results\n",
        "| Iter | lora_r | lr | steps | Format | Answer | Combined | Notes |",
        "|------|--------|----|-------|--------|--------|----------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['iteration']} "
            f"| {r.get('lora_r','-')} "
            f"| {r.get('learning_rate','-')} "
            f"| {r.get('max_steps','-')} "
            f"| {float(r['format_accuracy']):.1%} "
            f"| {float(r['answer_accuracy']):.1%} "
            f"| {float(r['combined_accuracy']):.1%} "
            f"| {r.get('notes','')} |"
        )

    best = max(rows, key=lambda x: float(x["combined_accuracy"]))
    lines.append(f"\n**Best iteration**: {best['iteration']} — "
                 f"combined={float(best['combined_accuracy']):.1%}  "
                 f"({best.get('notes','')})")

    TABLE_PATH.write_text("\n".join(lines))
    print(f"Table saved:  {TABLE_PATH}")


if __name__ == "__main__":
    rows = load_results()
    make_chart(rows)
    make_table(rows)
