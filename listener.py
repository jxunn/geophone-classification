import paho.mqtt.client as mqtt
import json
import numpy as np
import pandas as pd
import time

from scipy import ndimage
import tsfel

BROKER_IP = "172.20.10.5"
BROKER_PORT = 1883

recording = False
start_time = -1
recorded_data = []

cfg = tsfel.get_features_by_domain() # import TSFEL features
picked_features = [
    'signal_Standard deviation', 'signal_Mean absolute diff',
    'signal_Sum absolute diff', 'signal_Entropy', 'signal_Zero crossing rate',
    'signal_Spectral centroid', 'signal_Spectral decrease',
    'signal_Spectral entropy', 'signal_Spectral roll-off', 'signal_Spectral spread'
]

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("geoscope/node1/#")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global recording, start_time, recorded_data

    #print(msg.topic+" "+str(msg.payload))
    packet = json.loads(msg.payload)

    uuid = packet["uuid"]
    data = packet["data"]
    #print(f"[{uuid}] samples={len(data)}")

    if not recording:
        if np.any(np.array(data) > 2400) or np.any(np.array(data) < 1000):
            print("detected message!\n")
            recording = True
            start_time = time.time()
            recorded_data.append(data)
    else:
        recorded_data.append(data)
        print(f"{time.time() - start_time:.1f}s recorded", end="\r")

        if time.time() - start_time > 5: # recording for 5 seconds
            print("finished recording data\n")
            #client.disconnect()

            # choose 5 best footsteps & run model here
            features = detect(recorded_data)
            person = classify(features)

            # reset globals for next walk
            recorded_data = []
            recording = False
            start_time = -1

            # send detected person's name to LCD


def detect(recorded_data):
    signal = np.concatenate(recorded_data)

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
    # note: maybe move to previous loop
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


def classify(features):
    
    return


mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message

mqttc.connect("172.20.10.5", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
mqttc.loop_forever()