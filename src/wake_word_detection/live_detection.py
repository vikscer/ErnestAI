import sounddevice as sd
import joblib
import numpy as np
import soundfile as sf
from datetime import datetime
from src.wake_word_detection.feature_extraction import extract_features
from src.utils.config_loader import load_config
import os

# Load configuration
config = load_config()
listening_tone_path = config['static']['listening_tone_path'] + config['assistant']['character'] + '.wav'

# Load the trained model
clf = joblib.load('src/wake_word_detection/wake_word_model.pkl')
print("Model loaded successfully. Classes:", clf.classes_)


def play_listening_tone():
    """Play a tone to indicate the assistant is listening."""
    data, fs = sf.read(listening_tone_path)
    sd.play(data, fs)
    sd.wait()


def detect_wake_word(audio_data, threshold=0.8):
    """Detect if the given audio contains the wake word based on a confidence threshold."""
    # Save audio data to a temporary file to extract features
    sf.write("temp/temp.wav", audio_data, 16000)
    features = extract_features("temp/temp.wav")

    # Get probability score from the classifier
    prob = clf.predict_proba([features])[0][1]  # Probability of the wake word class

    # Debug information
    #print(f"Wake word probability: {prob:.2f}")

    # Return True if the probability exceeds the threshold, otherwise False
    return prob >= threshold

def listen_for_wake_word(threshold=0.8, window_duration=0.7, hop_duration=0.1):
    wake_word_samples_dir = "src/wake_word_data/wake_word"
    """Listens for the wake word and saves detected snippets as samples."""
    print("Listening for wake word...")

    # Record a 3-second audio clip
    audio_data = sd.rec(int(2 * 16000), samplerate=16000, channels=1)
    sd.wait()

    # Define window and hop size in samples
    window_length = int(window_duration * 16000)
    hop_length = int(hop_duration * 16000)

    # Slide over the 3-second audio data in 0.7-second windows
    for start in range(0, len(audio_data) - window_length + 1, hop_length):
        # Extract a 0.7-second segment
        window_data = audio_data[start:start + window_length]

        # Check if the wake word is detected in this window
        if detect_wake_word(window_data, threshold=threshold):
            # Save the detected wake word snippet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            sample_path = os.path.join(wake_word_samples_dir, f"wake_word_sample_{timestamp}.wav")
            sf.write(sample_path, window_data, 16000)
            print(f"Wake word detected! Sample saved as {sample_path}")
            # You can add a break here if you want to stop after the first detection
            play_listening_tone()
            return True  # Proceed to recognition

    print("No wake word detected in the 3-second clip.")
    return False