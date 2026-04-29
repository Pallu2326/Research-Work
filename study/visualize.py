import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["provider"], row["mode"])
        if key not in stats:
            stats[key] = {"count": 0.0, "ucr": 0.0, "cov": 0.0}
        stats[key]["count"] += 1.0
        stats[key]["ucr"] += float(row["unsupported_claim_rate"])
        stats[key]["cov"] += float(row["expected_coverage"])

    for key, values in stats.items():
        count = values["count"] or 1.0
        values["ucr"] /= count
        values["cov"] /= count
    return stats


def plot_dashboard(stats: dict[tuple[str, str], dict[str, float]], output_path: Path) -> None:
    labels = [f"{provider}\n{mode}" for provider, mode in sorted(stats)]
    ucr_values = [stats[key]["ucr"] for key in sorted(stats)]
    cov_values = [stats[key]["cov"] for key in sorted(stats)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(labels, ucr_values, color="#d9534f")
    axes[0].set_title("Unsupported Claim Rate (Lower is Better)")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(labels, cov_values, color="#5cb85c")
    axes[1].set_title("Expected Coverage (Higher is Better)")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("LLM Legal Drafting Evaluation Summary")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    csv_path = Path("results/eval_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError("results/eval_results.csv not found. Run python main.py first.")

    rows = load_rows(csv_path)
    stats = aggregate(rows)
    output_path = Path("results/eval_dashboard.png")
    plot_dashboard(stats=stats, output_path=output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
