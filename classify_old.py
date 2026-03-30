import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import ndimage

import tsfel
import pandas as pd
import numpy as np

# get default feature config (temporal + statistical + spectral)
cfg = tsfel.get_features_by_domain()

def process_file(filepath):
    with open(filepath) as f:
        packets = json.load(f)

    signal = np.array([x for packet in packets for x in packet['data']])

    # plt.plot(signal)
    # plt.title('Josh')
    # plt.show()

    #event detection

    mean = np.mean(signal)
    std = np.std(signal)

    k = 1.5 # tunable threshold multiplier
    threshold = k * std #this threshold is adaptive

    is_active = np.abs(signal - mean) > threshold #array of bools (true=prob SE, false=prob not SE)

    # plt.plot(signal - mean)
    # plt.axhline(threshold, color='r', linestyle='--', label='threshold')
    # plt.axhline(-threshold, color='r', linestyle='--')
    # plt.title('Josh - with threshold')
    # plt.xticks(np.arange(0, len(signal), 500))  # tick every 500 samples
    # plt.legend()
    # plt.show()

    #group together "probably a SE" samples
    # label connected regions of active samples
    labeled, num_events = ndimage.label(is_active)
    print(f"Found {num_events} events")

    # get the start and end index of each event
    event_slices = ndimage.find_objects(labeled)

    for i, s in enumerate(event_slices):
        start = s[0].start
        end = s[0].stop
        #print(f"Event {i+1}: samples {start} to {end}, length {end-start}")

    #dealing w/fragmentation

    min_length = 10 #min # samples for a footstep
    merge_gap = 50 #if 2 events within 50 samples of each other, they're part of the same footstep

    # merge gaps between active regions
    filled = ndimage.binary_closing(is_active, structure=np.ones(merge_gap))

    # re-label after merging
    labeled, num_events = ndimage.label(filled)
    event_slices = ndimage.find_objects(labeled)

    # filter out short events
    event_slices = [s for s in event_slices if s[0].stop - s[0].start >= min_length]

    #prob dont need this check
    if len(event_slices) < 5:
        print(f"Skipping {filepath} - only {len(event_slices)} events found")
        return None

    print(f"Found {len(event_slices)} events")
    for i, s in enumerate(event_slices):
        start = s[0].start
        end = s[0].stop
        #print(f"Event {i+1}: samples {start} to {end}, length {end-start}")

    #keep 5 best footsteps
    #note: tried using energy with a fixed window for each but that didn't work bc it would get all the noise around the step
    events = []
    for s in event_slices:
        start = s[0].start
        end = s[0].stop
        window = signal[start:end]
        energy = np.sum(window ** 2)
        events.append((energy, start, end, window))

    events.sort(key=lambda x: x[0], reverse=True)
    top5 = events[:5]

    for i, (energy, start, end, window) in enumerate(top5):
        print(f"Selected event {i+1}: samples {start} to {end}, energy={energy:.1f}")

    # #see which were selected
    # plt.plot(signal - mean)

    # # all detected events in light green
    # for s in event_slices:
    #     plt.axvspan(s[0].start, s[0].stop, alpha=0.2, color='green')

    # # top 5 selected in orange
    # for energy, start, end, window in top5:
    #     plt.axvspan(start, end, alpha=0.5, color='orange')

    # plt.title('All detected (green) vs top 5 selected (orange)')
    # plt.show()

    #normalize
    normalized = []
    for energy, start, end, window in top5:
        norm_window = window / energy
        normalized.append(norm_window)

    #plot to check they have same-ish amplitude
    # plt.figure()
    # for i, norm_window in enumerate(normalized):
    #     plt.plot(norm_window, label=f'Event {i+1}')
    # plt.title('Normalized events')
    # plt.legend()
    # plt.show()

    #truncate all steps to be same window (dont wanna pad with 0s bc spectral leakage, will mess with freq domain features)
    min_len = min(len(w) for w in normalized)
    truncated = [w[:min_len] for w in normalized]

    #TSFEL
    #extract features from each SE
    rows = []
    for window in truncated:
        df = pd.DataFrame(window, columns=['signal'])
        feats = tsfel.time_series_features_extractor(cfg, df, fs=525, verbose=0)
        rows.append(feats)
    
    return rows


# loop over all files and build dataset
all_rows = []
all_labels = []

for person in ['jenny', 'josh', 'tim']:
    for filepath in glob.glob(f'step_data/{person}/*.json'):
        rows = process_file(filepath)
        if rows is None:
            continue
        for row in rows:
            all_rows.append(row)
            all_labels.append(person)

X = pd.concat(all_rows).reset_index(drop=True)
y = all_labels
print(f"Dataset shape: {X.shape}, labels: {len(y)}")

from collections import Counter
print(Counter(y))

#train classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scale features - important for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train SVM with RBF kernel like the paper
clf = SVC(kernel='rbf', probability=True)  # probability=True needed for confidence scores later
clf.fit(X_train_scaled, y_train)

# step level accuracy
step_accuracy = clf.score(X_test_scaled, y_test)
print(f"Step level accuracy: {step_accuracy:.3f}")

# trace level evaluation
trace_predictions = []
trace_true = []

for tid in np.unique(trace_ids[test_idx]):
    # get all steps belonging to this trace
    mask = trace_ids[test_idx] == tid
    X_trace = X_test_scaled[mask]
    y_trace = y_test[mask]
    
    # get confidence scores for each step
    probas = clf.predict_proba(X_trace)
    
    # pick the step with highest confidence
    best_step = np.argmax(probas.max(axis=1))
    predicted_label = clf.classes_[np.argmax(probas[best_step])]
    
    trace_predictions.append(predicted_label)
    trace_true.append(y_trace[0])  # all steps in trace have same true label

trace_accuracy = np.mean(np.array(trace_predictions) == np.array(trace_true))
print(f"Trace level accuracy: {trace_accuracy:.3f}")
print(classification_report(trace_true, trace_predictions))