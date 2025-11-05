import mne
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

'''
# Load EDF file
edf_file_path = 'SN001.edf'

raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Select EEG channels for plotting
eeg_channels = raw.pick_types(eeg=True)
# print(eeg_channels)

# Select EEG channels for anomaly detection
eeg_data = raw.get_data(picks='eeg')
print(type(eeg_data))
# selected_channels = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
# eeg_data = raw.get_data(picks=selected_channels)


print("-----------")
# print(raw," ",raw.ch_names,"  ",raw.annotations)
print("-----------")

# for i in range(8):
#     for j in range(5):
        # print(eeg_data[i][j])
    # print("//////////")

# print(len(eeg_data),len(eeg_data[0]))


import pandas as pd

# Assuming 'eeg_data' is your numpy.ndarray with shape (n_channels, n_samples)
# You might need to transpose the data if channels are in rows and time points in columns
eeg_data_transposed = eeg_data[:100].T

# Assuming 'channel_names' is a list of channel names (adjust as needed)
channel_names = ['EEG Channel 1', 'EEG Channel 2', 'EEG Channel 3', 'EEG Channel 4', 'EEG Channel 5', 'EEG Channel 6', 'EEG Channel 7', 'EEG Channel 8']

# Create a DataFrame
df = pd.DataFrame(data=eeg_data_transposed, columns=channel_names)

# Print the DataFrame
print(df[:20])
'''


'''
# Load PSG EDF+ file with annotations
edf_file_path = 'SN001_sleepscoring.edf'
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Extract sleep staging annotations
annotations = mne.read_annotations(edf_file_path)
print(type(annotations))
# l = annotations.split()
# print(l[:5])

# Print sleep stages and their corresponding time intervals
# for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
#     print(f"Sleep Stage {description} from {onset} to {onset + duration} seconds.")

annotations_text = annotations.description
print(annotations_text[:15])

# Extract the 5th word from each description and store in a list
# sleep_stages = [match.group(1) for description in annotations_text for j in range(4) if (match := re.search(r'Sleep stage (\w+)', description))]


# Print the list of sleep stages
# print(sleep_stages[:5],len(sleep_stages))
'''

'''

# Extract statistical features (mean, standard deviation) as an example
features = [eeg_data.mean(axis=1), eeg_data.std(axis=1)]
print("FEATURES : ")
print(features[:5],len(features))
print(" ")
# Transpose the feature matrix
X = np.array(features).T

# Train Isolation Forest model
clf = IsolationForest(contamination=0.05)  # Adjust the contamination parameter as needed
clf.fit(X)

# Predict anomalies

predictions = clf.predict(X)
print("PREDICTIONS : ")
print(predictions[:5],len(predictions))
print(" ")

# Identify anomalies (instances with -1 prediction)
anomalies = X[predictions == -1]
print(" ANOMALIES: ")
print(anomalies[:5],"   ",len(anomalies))
print(" ")
'''
# Visualize anomalies or take further actions based on your needs

# Plot EEG signals
# raw.plot(n_channels=len(eeg_channels), scalings='auto', title='EEG signals')

# plt.show()
