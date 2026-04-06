import json
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "new_jenny"   # folder directly in root
N_TRIALS = 3
COLOR = "darkorange"

def load_json_file(filepath):
    with open(filepath, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        data = []
        for chunk in raw:
            data.extend(chunk["data"])
        gain = raw[0].get("gain", None)
    else:
        data = raw["data"]
        gain = raw.get("gain", None)

    return np.array(data, dtype=float), gain


def load_files():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])[:N_TRIALS]
    trials = []

    for fname in files:
        try:
            data, gain = load_json_file(os.path.join(DATA_DIR, fname))
            trials.append((fname, data, gain))
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    return trials


# load trials
trials = load_files()

# compute axis limits
all_data = [data for _, data, _ in trials]

global_ymin = min(d.min() for d in all_data)
global_ymax = max(d.max() for d in all_data)
global_xmax = max(len(d) for d in all_data)

y_pad = (global_ymax - global_ymin) * 0.05
global_ymin -= y_pad
global_ymax += y_pad


fig, axes = plt.subplots(1, N_TRIALS, figsize=(16, 4), sharex=True, sharey=True)

for col in range(N_TRIALS):
    ax = axes[col]

    if col < len(trials):
        fname, data, gain = trials[col]

        ax.plot(data, color=COLOR, linewidth=0.6)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        clip_idx = np.where(data >= 4090)[0]
        if len(clip_idx):
            ax.scatter(clip_idx, data[clip_idx], color="red", s=2, zorder=5)

        title = f"Trial {col+1}"
        if gain is not None:
            title += f" (gain={gain:.1f})"

        ax.set_title(title, fontsize=10)
    else:
        ax.axis("off")

    ax.set_xlim(0, global_xmax)
    ax.set_ylim(global_ymin, global_ymax)

    ax.set_xlabel("Sample", fontsize=8)
    if col == 0:
        ax.set_ylabel("ADC value", fontsize=8)

plt.suptitle("Geophone Signals — new_josh", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("new_josh_signals.png", dpi=130, bbox_inches="tight")

print("Saved new_josh_signals.png")
plt.show()