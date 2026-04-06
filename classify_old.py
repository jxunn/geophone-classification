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
    signal = signal[:len(signal)//2]

    # plt.plot(signal)
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
    #print(f"Found {num_events} events")

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

    #print(f"Found {len(event_slices)} events")
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

    #for i, (energy, start, end, window) in enumerate(top5):
        #print(f"Selected event {i+1}: samples {start} to {end}, energy={energy:.1f}")

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
        feats = tsfel.time_series_features_extractor(cfg, df, fs=500, verbose=0)
        rows.append(feats)
    
    return rows


# loop over all files and build dataset
all_rows = []
all_labels = []
all_trace_ids = []

trace_id = 0
for person in ['jenny', 'josh', 'tim']:
    for filepath in glob.glob(f'step_data_new/{person}/*.json'):
        rows = process_file(filepath)
        if rows is None:
            continue
        for row in rows:
            all_rows.append(row)
            all_labels.append(person)
            all_trace_ids.append(trace_id)  # all 5 steps from same trace share same id
        trace_id += 1

X = pd.concat(all_rows).reset_index(drop=True)
y = np.array(all_labels)
trace_ids = np.array(all_trace_ids)

print(f"Dataset shape: {X.shape}, labels: {len(y)}")
from collections import Counter
print(Counter(y))

#train/test split: ensure no trace is split across train and test, that would make it biased. 
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)  # 5 fold cross validation

step_accs = []
trace_accs = []

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=trace_ids)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_trace_ids = trace_ids[test_idx]

    # manually chosen features - half time domain, half frequency domain
    chosen_features = [
        'signal_Standard deviation', 'signal_Mean absolute diff',
        'signal_Sum absolute diff', 'signal_Entropy', 'signal_Zero crossing rate',
        'signal_Spectral centroid', 'signal_Spectral decrease',
        'signal_Spectral entropy', 'signal_Spectral roll-off', 'signal_Spectral spread'
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[chosen_features])
    X_test_scaled = scaler.transform(X_test[chosen_features])

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train_scaled, y_train) # model learning from training data
    # ... rest stays the same

    step_accs.append(clf.score(X_test_scaled, y_test))

    correct = 0
    total = 0
    for tid in np.unique(test_trace_ids):
        mask = test_trace_ids == tid
        proba = clf.predict_proba(X_test_scaled[mask])
        true_label = y_test[mask][0]
        
        # print each step's prediction and confidence
        #print(f"\n  Trace {tid} (true={true_label}):")
        for step_i, prob_row in enumerate(proba):
            predicted = clf.classes_[np.argmax(prob_row)]
            confidence = prob_row.max()
            #print(f"    Step {step_i+1}: predicted={predicted}, confidence={confidence:.3f}, all_probs={dict(zip(clf.classes_, prob_row.round(3)))}")
        
        best_step = np.argmax(proba.max(axis=1))
        predicted_label = clf.classes_[np.argmax(proba[best_step])]
        #print(f"  → Trace prediction: {predicted_label} ({'✓' if predicted_label == true_label else '✗'})")

        # keep counting for accuracy
        if predicted_label == true_label:
            correct += 1
        total += 1
    trace_accs.append(correct / total)
    print(f"Fold {fold+1}: step={step_accs[-1]:.3f}, trace={trace_accs[-1]:.3f}")

print(f"\nMean step accuracy: {np.mean(step_accs):.3f} +/- {np.std(step_accs):.3f}")
print(f"Mean trace accuracy: {np.mean(trace_accs):.3f} +/- {np.std(trace_accs):.3f}")