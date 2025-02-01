import wfdb as wf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import pywt
import statsmodels.api as sm
import scipy.fftpack

def butter_bandpass_filter(input_signal, low_cutoff, high_cutoff, sampling_rate, order):
    nyq = 0.5 * sampling_rate
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    numerator, denominator = butter(order, [low, high], btype='band')
    filtered = filtfilt(numerator, denominator, input_signal)
    return filtered

def process_signal(signal):
    filtered_signal = butter_bandpass_filter(signal, 1.0, 40.0, 1000, 2)
    return filtered_signal[:115200]

patient_paths = [
    "C:\\Users\\SCH\\Downloads\\patient173\\s0305lre",
    "C:\\Users\\SCH\\Downloads\\patient182\\s0308lre",
    "C:\\Users\\SCH\\Downloads\\patient234\\s0460_re",
    "C:\\Users\\SCH\\Downloads\\patient238\\s0466_re"
]
data = {}
for idx, path in enumerate(patient_paths):
    signal_array, fields = wf.rdsamp(path)
    sig = signal_array[:, 1]
    if len(sig) > 115200:
        sig = sig[:115200]
    processed_signal = process_signal(sig)
    data[f"sub_{idx + 1}"] = processed_signal

df = pd.DataFrame(data)
print(df.head())
print(df.shape)

def get_r_peaks(signal):
    peaks, _ = find_peaks(signal, distance=100)
    threshold = np.max(signal) * 0.7
    r_peaks = [peak for peak in peaks if signal[peak] > threshold]
    return r_peaks

r_peaks_dict = {key: get_r_peaks(df[key]) for key in df.columns}

def segment_signal(signal, r_peaks):
    segments = []
    for i in range(1, len(r_peaks) - 1):
        start = r_peaks[i] - 50
        end = r_peaks[i] + 100
        segments.append(signal[start:end])
    return segments
segmented_signals = {key: segment_signal(df[key], r_peaks_dict[key]) for key in df.columns}

def extract_features(signal_segment):
    # Extract wavelet features
    wavelet = pywt.Wavelet('db4')
    decomp_levels = 5
    decomp = pywt.wavedec(signal_segment, wavelet, level=decomp_levels)
    ca5, _, _, _, _, _ = decomp
    wavelet_features = ca5[:41]

    # Extract autocorrelation features
    acc = sm.tsa.acf(signal_segment, nlags=1000)
    acc_segment = acc[:1000]
    dct_features = scipy.fftpack.dct(acc_segment, type=2)

    return np.concatenate([wavelet_features, dct_features])

features = []
labels = []

for key in segmented_signals.keys():
    for segment in segmented_signals[key]:
        features.append(extract_features(segment))
        labels.append(key)

features_df = pd.DataFrame(features)
features_df['label'] = labels

X = features_df.drop('label', axis=1)
y = features_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines for both classifiers
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Define parameter grids for GridSearchCV
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30]
}

param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf', 'poly']
}

# Perform GridSearchCV for RandomForest
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Perform GridSearchCV for SVM
grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)


# Print classification reports
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

print("SVM Classification Report")
print(classification_report(y_test, y_pred_svm))

# Accuracy comparison
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Random Forest Accuracy:", accuracy_rf)
print("SVM Accuracy:", accuracy_svm)

# Subject identification logic
def identify_subject(predictions, threshold=0.5):
    subject_counts = pd.Series(predictions).value_counts(normalize=True)
    identified_subject = subject_counts.idxmax() if subject_counts.max() > threshold else "unidentified"
    return identified_subject

# Apply identification logic to test set predictions
subjects_identified_rf = identify_subject(y_pred_rf)
subjects_identified_svm = identify_subject(y_pred_svm)

print("Identified Subject (Random Forest):", subjects_identified_rf)
print("Identified Subject (SVM):", subjects_identified_svm)
