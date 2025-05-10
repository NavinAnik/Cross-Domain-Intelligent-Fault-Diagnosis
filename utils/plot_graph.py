import matplotlib.pyplot as plt


def plot_graph(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend([ylabel])
    cyberpunk.add_glow_effects()
    plt.savefig(f"/results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

