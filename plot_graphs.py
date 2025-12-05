import matplotlib.pyplot as plt

# CTR values from your results
ctr_data = {
    "xgboost": {"control": 0.094877, "treatment": 0.093023},
    "mf":      {"control": 0.085185, "treatment": 0.050000},
    "bpr":     {"control": 0.112523, "treatment": 0.060134},
    "linucb":  {"control": 0.106306, "treatment": 0.076404},
    "fm":      {"control": 0.086042, "treatment": 0.094340},
}

for method, values in ctr_data.items():
    plt.figure(figsize=(5,4))

    groups = list(values.keys())  # ["control", "treatment"]
    ctrs = list(values.values())

    plt.bar(groups, ctrs)
    plt.title(f"CTR Comparison - {method}")
    plt.ylabel("CTR")
    plt.ylim(0, max(ctrs) * 1.3)

    # Print CTR values above bars
    for i, v in enumerate(ctrs):
        plt.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"plots/{method}.png")
    plt.show()

