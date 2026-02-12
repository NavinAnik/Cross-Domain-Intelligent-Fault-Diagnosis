import os
import matplotlib.pyplot as plt


def plot_graph(x, y, title, xlabel, ylabel, filename, labels=None, output_dir="results"):
    """Plot and save a graph. Use output_dir to save per-model (e.g. results/dinov2)."""
    plt.figure(figsize=(10, 6))

    # Check if y contains multiple series
    if isinstance(y[0], (list, tuple)):
        for i, y_series in enumerate(y):
            label = labels[i] if labels else None
            plt.plot(x, y_series, marker='o', linestyle='-', label=label)
    else:
        plt.plot(x, y, marker='o', linestyle='-', label=labels[0] if labels else ylabel)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if labels:
        plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
