import json
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "step_data"
PERSONS = ["jenny", "josh", "tim"]
COLORS = {"jenny": "steelblue", "josh": "darkorange", "tim": "seagreen"}
N_TRIALS = 3

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

def load_person_files(person, n=N_TRIALS):
    folder = os.path.join(DATA_DIR, person)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])[:n]
    trials = []
    for fname in files:
        try:
            data, gain = load_json_file(os.path.join(folder, fname))
            trials.append((fname, data, gain))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    return trials

# load all data first so we can compute global axis limits
all_trials = {p: load_person_files(p) for p in PERSONS}

all_data = [data for trials in all_trials.values() for _, data, _ in trials]
global_ymin = min(d.min() for d in all_data)
global_ymax = max(d.max() for d in all_data)
global_xmax = max(len(d) for d in all_data)

# add a little padding
y_pad = (global_ymax - global_ymin) * 0.05
global_ymin -= y_pad
global_ymax += y_pad

fig, axes = plt.subplots(len(PERSONS), N_TRIALS,
                         figsize=(16, 4 * len(PERSONS)),
                         sharex=True, sharey=True)

for row, person in enumerate(PERSONS):
    trials = all_trials[person]
    for col in range(N_TRIALS):
        ax = axes[row][col]
        if col < len(trials):
            fname, data, gain = trials[col]
            ax.plot(data, color=COLORS[person], linewidth=0.6)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

            # mark clipping in red
            clip_idx = np.where(data >= 4090)[0]
            if len(clip_idx):
                ax.scatter(clip_idx, data[clip_idx], color="red", s=2, zorder=5)

            title = f"{person.capitalize()} — Trial {col+1}"
            if gain is not None:
                title += f"  (gain={gain:.1f})"
            ax.set_title(title, fontsize=10)
        else:
            ax.axis("off")

        ax.set_xlim(0, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)

        # only label outer edges
        if row == len(PERSONS) - 1:
            ax.set_xlabel("Sample", fontsize=8)
        if col == 0:
            ax.set_ylabel("ADC value", fontsize=8)

plt.suptitle("Geophone Signals — First 3 Trials per Person", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("signals_overview.png", dpi=130, bbox_inches="tight")
print("Saved signals_overview.png")
plt.show()