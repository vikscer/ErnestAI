import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from src.wake_word_detection.feature_extraction import extract_features

# Paths
wake_word_path = '../wake_word_data/wake_word'
background_noise_path = '../wake_word_data/background_noise'

# Verify directories
if not os.path.exists(wake_word_path) or not os.path.exists(background_noise_path):
    raise FileNotFoundError("One or both of the specified directories do not exist.")

# Load wake_word_data and labels
features = []
labels = []

# Helper function to add features and labels with error handling
def add_features_from_directory(directory, label):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Label wake words as 1 and background noise as 0
add_features_from_directory(wake_word_path, label=1)
add_features_from_directory(background_noise_path, label=0)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Prepare training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Cross-validation on the full dataset
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=skf)
print("Cross-Validation Accuracy:", np.mean(scores))

# Save the trained model
joblib.dump(clf, '../wake_word_detection/wake_word_model.pkl')
print("Model saved successfully.")
