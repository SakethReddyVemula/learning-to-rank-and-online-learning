import matplotlib.pyplot as plt

# CTR and Dwell Time values
data = {
    "xgboost": {
        "ctr": {"control": 0.094877, "treatment": 0.093023},
        "dwell": {"control": 6.182508, "treatment": 7.932516},
    },
    "mf": {
        "ctr": {"control": 0.085185, "treatment": 0.050000},
        "dwell": {"control": 7.543835, "treatment": 0.938800},
    },
    "bpr": {
        "ctr": {"control": 0.112523, "treatment": 0.060134},
        "dwell": {"control": 7.668545, "treatment": 2.618345},
    },
    "linucb": {
        "ctr": {"control": 0.106306, "treatment": 0.076404},
        "dwell": {"control": 4.932329, "treatment": 5.656883},
    },
    "fm": {
        "ctr": {"control": 0.086042, "treatment": 0.094340},
        "dwell": {"control": 6.870435, "treatment": 7.190797},
    },
}

# Generate 5 figures
for method, vals in data.items():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- CTR subplot ---
    groups = list(vals["ctr"].keys())  # ['control', 'treatment']
    ctrs = list(vals["ctr"].values())

    axes[0].bar(groups, ctrs)
    axes[0].set_title(f"{method.upper()} - CTR")
    axes[0].set_ylabel("CTR")
    axes[0].set_ylim(0, max(ctrs) * 1.3)

    # Value annotations
    for i, v in enumerate(ctrs):
        axes[0].text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)

    # --- Dwell time subplot ---
    dwell_groups = list(vals["dwell"].keys())
    dwell_vals = list(vals["dwell"].values())

    axes[1].bar(dwell_groups, dwell_vals)
    axes[1].set_title(f"{method.upper()} - Average Dwell Time")
    axes[1].set_ylabel("Dwell Time (s)")
    axes[1].set_ylim(0, max(dwell_vals) * 1.3)

    # Value annotations
    for i, v in enumerate(dwell_vals):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"plots/{method}.png")
    plt.show()
