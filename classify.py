import json
import numpy as np
import glob
from scipy import ndimage

import tsfel
import pandas as pd
from statistics import mode

cfg = tsfel.get_features_by_domain()

def process_file(filepath):
    with open(filepath) as f:
        packets = json.load(f)

    # flatten packets into signal (same as np.concatenate in detect())
    signal = np.array([x for packet in packets for x in packet['data']])

    mean = np.mean(signal)
    std = np.std(signal)

    ## configure threshold above noise floor
    k = 1.5
    threshold = k * std # this threshold is adaptive

    footstep_detected = np.abs(signal - mean) > threshold # array of bools (true=prob SE, false=prob not SE)

    ## group samples that are likely part of same SE
    # tunable parameters
    min_length = 10 # min # samples for a footstep
    merge_gap = 50 # if 2 events within 50 samples of each other, they're part of same SE

    # merge gaps between active regions (make groups, each group a SE)
    filled = ndimage.binary_closing(footstep_detected, structure=np.ones(merge_gap))

    labeled, num_events = ndimage.label(filled) # label each SE
    event_slices = ndimage.find_objects(labeled) # gives stop & start for each SE

    # remove SEs too short to be footstep
    event_slices = [s for s in event_slices if s[0].stop - s[0].start >= min_length]

    if len(event_slices) < 5:
        print(f"Skipping {filepath} - only {len(event_slices)} events found")
        return None

    ## keep 5 best footsteps
    event_energies = []
    for SE in event_slices:
        start = SE[0].start
        end = SE[0].stop
        window = signal[start:end]
        energy = np.sum(window ** 2) # compute energy of each footstep
        event_energies.append((energy, start, end, window))

    event_energies.sort(key=lambda x: x[0], reverse=True)
    best = event_energies[:5] # get 5 best footsteps

    ## normalize footstep (by dividing by energy) to remove magnitude
    normalized = []
    for energy, start, end, window in best:
        norm_window = window / energy
        normalized.append(norm_window)

    ## truncate each footstep so they all have same window size (dont wanna pad with 0s bc spectral leakage, will mess with freq domain features)
    window_size = min(len(w) for w in normalized)
    truncated = [w[:window_size] for w in normalized]

    ## extract features from each footstep
    features = [] # list of dfs (1 df per footstep) containing features
    for SE in truncated:
        df = pd.DataFrame(SE, columns=['signal']) # convert footstep signal into df for TSFEL library
        extracted = tsfel.time_series_features_extractor(cfg, df, fs=500, verbose=0)
        features.append(extracted)

    return features


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
            all_trace_ids.append(trace_id)
        trace_id += 1

X = pd.concat(all_rows).reset_index(drop=True)
y = np.array(all_labels)
trace_ids = np.array(all_trace_ids)

print(f"Dataset shape: {X.shape}, labels: {len(y)}")
from collections import Counter
print(Counter(y))

from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

gkf = GroupKFold(n_splits=5)

step_accs = []
trace_accs = []

chosen_features = [
    'signal_Standard deviation', 'signal_Mean absolute diff',
    'signal_Sum absolute diff', 'signal_Entropy', 'signal_Zero crossing rate',
    'signal_Spectral centroid', 'signal_Spectral decrease',
    'signal_Spectral entropy', 'signal_Spectral roll-off', 'signal_Spectral spread'
]

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=trace_ids)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_trace_ids = trace_ids[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[chosen_features])
    X_test_scaled = scaler.transform(X_test[chosen_features])

    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train_scaled, y_train)

    step_accs.append(clf.score(X_test_scaled, y_test))

    correct = 0
    total = 0
    for tid in np.unique(test_trace_ids):
        mask = test_trace_ids == tid
        proba = clf.predict_proba(X_test_scaled[mask])
        true_label = y_test[mask][0]

        # choose highest confidence person for each footstep
        per_step_pred = [] # array w/ prediction for each step (based on who had highest confidence)
        for per_step_confidences in proba: # array of 3 confidence levels specific to a step (1 per person)
            highest_confidence = per_step_confidences.max()

            # if the highest confidence for that step is too low, say it's unidentified
            if highest_confidence < 0.5:
                #print("unconfident\n")
                per_step_pred.append("unidentified")
                continue

            highest_confidence_idx = np.argmax(per_step_confidences)
            highest_confidence_label = clf.classes_[highest_confidence_idx]

            print(f"  step: {highest_confidence_label} ({highest_confidence:.3f})")

            per_step_pred.append(highest_confidence_label)
        print(f"Step predictions: {per_step_pred}")
        trace_pred = mode(per_step_pred)

        if trace_pred == true_label:
            correct += 1
        total += 1
    trace_accs.append(correct / total)
    print(f"Fold {fold+1}: step={step_accs[-1]:.3f}, trace={trace_accs[-1]:.3f}")

print(f"\nMean step accuracy: {np.mean(step_accs):.3f} +/- {np.std(step_accs):.3f}")
print(f"Mean trace accuracy: {np.mean(trace_accs):.3f} +/- {np.std(trace_accs):.3f}")

# train final model on all data and save
import joblib

scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X[chosen_features])
clf_final = SVC(kernel='rbf', probability=True)
clf_final.fit(X_all_scaled, y)

joblib.dump(clf_final, 'model.pkl')
joblib.dump(scaler_final, 'scaler.pkl')
print("Model and scaler saved.")