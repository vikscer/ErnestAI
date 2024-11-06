import sounddevice as sd
import numpy as np
import soundfile as sf
import os
from google.cloud import speech
import time
from src.utils.config_loader import load_config

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/googlauth.json"

config = load_config()
listening_tone_path = config['static']['listening_tone']
threshold = 0.02  # Adjust based on testing
recording_duration = 3  # Duration to record after detecting the wake word

def play_listening_tone():
    """Play a tone to indicate the assistant is listening."""
    data, fs = sf.read(listening_tone_path)
    sd.play(data, fs)
    sd.wait()

def listen_for_wake_word():
    """Continuously listens for the wake word."""
    print("Listening for wake word...")
    while True:
        audio_data = sd.rec(int(recording_duration * 16000), samplerate=16000, channels=1)
        sd.wait()

        # Analyze audio to detect wake word (simple threshold-based method)
        if np.max(np.abs(audio_data)) > threshold:
            print("Wake word detected!")
            play_listening_tone()
            # Save the recorded audio to a temporary file
            sf.write("temp/wake_word_temp.wav", audio_data, 16000)  # Changed from sd.write to sf.write
            return True  # Proceed to recognition
