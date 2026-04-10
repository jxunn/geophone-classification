import json
import numpy as np
import glob
from scipy import ndimage

import tsfel
import pandas as pd
from statistics import mode

from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

cfg = tsfel.get_features_by_domain()

def process_file(filepath):
    with open(filepath) as f:
        packets = json.load(f)

    # flatten packets into signal (same as np.concatenate in detect())
    signal = np.array([x for packet in packets for x in packet['data']])

    print(signal[:20])

    ## apply LPF
    b, a = butter(4, 35 / 2500, btype='low')
    filtered = filtfilt(b, a, signal)
    #filtered = signal

    mean = np.mean(filtered)
    std = np.std(filtered)



    # ## spectral analysis plot
    # fft_vals = np.fft.rfft(filtered - np.mean(filtered))
    # fft_freqs = np.fft.rfftfreq(len(filtered), d=1/5000)
    # power = np.abs(fft_vals) ** 2

    # plt.figure(figsize=(15, 4))
    # plt.plot(fft_freqs, power)
    # plt.title("Power Spectral Density")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power")
    # plt.xlim(0, 500)  # adjust as needed
    # plt.show()





    ## configure threshold above noise floor
    k = 3.3
    #threshold = k * std # this threshold is adaptive
    threshold = k * 58 #this is hardcoded for now bc we had the same noise floor for data collection. in real-time it will be adaptive

    footstep_detected = np.abs(filtered - mean) > threshold # array of bools (true=prob SE, false=prob not SE)

    ## group samples that are likely part of same SE
    # tunable parameters
    min_length = 10 # min # samples for a footstep
    merge_gap = 500 # if 2 events within 500 samples of each other, they're part of same SE

    # merge gaps between active regions (make groups, each group a SE)
    filled = ndimage.binary_closing(footstep_detected, structure=np.ones(merge_gap))



    # plt.figure(figsize=(15,6))

    # plt.subplot(3,1,1)
    # plt.plot(signal, color='gray')
    # plt.title("Original Signal")

    # plt.subplot(3,1,2)
    # plt.plot(footstep_detected.astype(int))
    # plt.title("Raw Threshold Detection (footstep_detected)")

    # plt.subplot(3,1,3)
    # plt.plot(filled.astype(int))
    # plt.title("After Gap Filling (binary_closing)")

    # plt.tight_layout()
    # plt.show()




    labeled, num_events = ndimage.label(filled) # label each SE
    event_slices = ndimage.find_objects(labeled) # gives stop & start for each SE

    # remove SEs too short to be footstep
    event_slices = [s for s in event_slices if s[0].stop - s[0].start >= min_length]

    #note: min_length no longer filters out random noise spikes, although the noise spikes rarely reach the magnitude of the footstep spike
    #we can run the peaks through the matched filter to check if it resembles noise, then filter those out





    # plt.figure(figsize=(15,4))
    # plt.plot(filtered, color='gray', label="Filtered signal")

    # # threshold lines
    # plt.axhline(mean + threshold, color='red', linestyle='--', label='Upper threshold')
    # plt.axhline(mean - threshold, color='red', linestyle='--', label='Lower threshold')

    # for i, s in enumerate(event_slices):
    #     start = s[0].start
    #     end = s[0].stop
    #     plt.axvspan(start, end, alpha=0.3, label=f"Event {i}" if i < 5 else None)

    # plt.title("Detected Event Slices with Threshold")
    # plt.xlabel("Sample index")
    # plt.ylabel("ADC value")
    # plt.legend()
    # plt.show()






    #note: instead of using energy, use magnitude to choose the best footsteps!
    #upon plot observation magnitude is the most reliable indicator (consistently in the middle of good footsteps, whereas energy can be weird and highlight less strong footsteps)
    event_peaks = []
    for event in event_slices:
        data = filtered[event[0].start : event[0].stop]
        peak_idx = np.argmax(np.abs(data)) + event[0].start
        peak = filtered[peak_idx]
        event_peaks.append((peak, peak_idx))

    event_peaks.sort()
    highest_peaks = event_peaks[-5:] # get the 5 steps with the highest peak

    best = []
    for peak, peak_idx in highest_peaks:
        start = peak_idx - 700
        stop = peak_idx + 700
        best.append((start, stop))






    # plt.figure(figsize=(15,4))
    # plt.plot(filtered, color='gray', label="Filtered signal")

    # plt.axhline(mean + threshold, color='red', linestyle='--', label='Upper threshold')
    # plt.axhline(mean - threshold, color='red', linestyle='--', label='Lower threshold')

    # for i, (start, stop) in enumerate(best):
    #     plt.axvspan(start, stop, alpha=0.3, color='blue', label=f"Best {i+1}" if i < 5 else None)

    # plt.title("Top 5 Selected Footsteps")
    # plt.xlabel("Sample index")
    # plt.ylabel("ADC value")
    # plt.legend()
    # plt.show()




    ## normalize footstep (by dividing by energy) to remove magnitude
    normalized = []
    for start, stop in best:
        window = filtered[start : stop]
        energy = np.sum(window ** 2)
        normalized.append(window / energy)


    ## extract features from each footstep
    features = [] # list of dfs (1 df per footstep) containing features
    for SE in normalized:
        df = pd.DataFrame(SE, columns=['signal']) # convert footstep signal into df for TSFEL library
        extracted = tsfel.time_series_features_extractor(cfg, df, fs=5000, verbose=0)
        features.append(extracted)

    return features


# loop over all files and build dataset
all_rows = []
all_labels = []
all_trace_ids = []

trace_id = 0
for person in ['josh', 'tim', 'jenny']:
    for filepath in glob.glob(f'step_data_newest/{person}/*.json'):
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
                #per_step_pred.append("unidentified")
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