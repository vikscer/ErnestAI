import sounddevice as sd
import numpy as np
import io
import os
from google.cloud import speech
import soundfile as sf

# Set the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/googlauth.json"

def record_audio(duration=5, sample_rate=16000, device_index=1):
    """Record audio from a specific device."""
    print("Recording started...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device_index)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio_data

def save_audio(audio_data, filename="temp_audio.wav", sample_rate=16000):
    """Save recorded audio to a file."""
    sf.write(filename, audio_data, sample_rate)
    print(f"Audio saved to {filename}")
    return filename

def recognize_speech(audio_file_path):
    """Capture audio from a file and recognize Slovak speech."""
    client = speech.SpeechClient()

    # Read the audio file and handle potential issues
    try:
        with io.open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()
            if not content:
                print(f"Error: No audio content found in {audio_file_path}")
                return None
    except Exception as e:
        print(f"Failed to read audio file: {e}")
        return None

    # Create RecognitionAudio and RecognitionConfig objects
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=32000,
        language_code="sk-SK"
    )

    try:
        # Make the API call for speech recognition
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            print("No speech was recognized.")
            return None

        # Join the transcripts of each result
        result_text = " ".join([result.alternatives[0].transcript for result in response.results])
        return result_text
    except Exception as e:
        print(f"Speech recognition failed: {e}")
        return None