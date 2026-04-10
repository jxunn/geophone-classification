import json
import glob
import numpy as np
from scipy.signal import butter, filtfilt

signals = []
for filepath in glob.glob('nothing/*.json'):
    with open(filepath) as f:
        packets = json.load(f)
    signal = np.array([x for packet in packets for x in packet['data']])
    b, a = butter(4, 35 / 2500, btype='low')
    filtered = filtfilt(b, a, signal)
    signals.append(filtered)

all_samples = np.concatenate(signals)
print(f"Noise std: {np.std(all_samples):.4f}")
print(f"Noise mean: {np.mean(all_samples):.4f}")