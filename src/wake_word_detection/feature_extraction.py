import librosa
import numpy as np


def extract_features(file_path, sample_rate=16000, n_mfcc=13):
    """
    Extract MFCC features from an audio file.

    Parameters:
    - file_path (str): Path to the audio file.
    - sample_rate (int): Sampling rate to load the audio file.
    - n_mfcc (int): Number of MFCC coefficients to extract.

    Returns:
    - np.ndarray: Mean MFCC coefficients of the audio file.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Take the mean of the MFCC coefficients across time
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


def extract_features_from_audio(audio_data, sample_rate=16000, n_mfcc=13):
    """
    Extract MFCC features from raw audio wake_word_data (used for real-time detection).

    Parameters:
    - audio_data (np.ndarray): Raw audio wake_word_data.
    - sample_rate (int): Sampling rate of the audio wake_word_data.
    - n_mfcc (int): Number of MFCC coefficients to extract.

    Returns:
    - np.ndarray: Mean MFCC coefficients of the audio wake_word_data.
    """
    # Compute MFCCs directly from the audio wake_word_data
    mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=n_mfcc)

    # Take the mean of the MFCC coefficients across time
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean
