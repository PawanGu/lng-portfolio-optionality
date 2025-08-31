# src/viz/plots.py
import matplotlib.pyplot as plt
import numpy as np

def plot_pnl_hist(series_list, labels, outfile: str):
    plt.figure()
    for s, label in zip(series_list, labels):
        s = np.asarray(s)
        plt.hist(s, bins=50, alpha=0.5, label=label, density=True)
    plt.legend()
    plt.xlabel("PnL")
    plt.ylabel("Density")
    plt.title("PnL distribution")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
