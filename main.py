import os
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
from src.wake_word_detection.live_detection import listen_for_wake_word  # Custom wake word detection
from src.voice_recognition import recognize_speech
from src.openai_response import generate_response
from src.text_to_speech import text_to_speech
from src.utils.config_loader import load_config

# Set the microphone input device index (replace with your chosen device index)
device_index = 1  # Update this index based on the device list output

# Load configuration
config = load_config()
listening_tone_path = config['static']['listening_tone']

def save_text_to_file(text, filename):
    """Save the text to a specified file."""
    with open("temp/" + filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {filename}")

def record_audio_until_silence(duration=2, threshold=0.02, device_index=0):
    """Record audio from the specified device until silence is detected for a specified duration."""
    print("Recording started...")
    recorded_data = []
    silence_start_time = None

    while True:
        # Record for 1 second at a time, using the specified device
        audio_data = sd.rec(int(16000), samplerate=16000, channels=1, device=device_index)
        sd.wait()

        # Check if audio exceeds the silence threshold
        if np.max(np.abs(audio_data)) > threshold:
            if silence_start_time is not None:
                silence_start_time = None  # Reset silence timer if speech is detected
            recorded_data.append(audio_data)
        else:
            if silence_start_time is None:
                silence_start_time = time.time()

            # If silence lasts for the specified duration, stop recording
            if silence_start_time and time.time() - silence_start_time >= duration:
                print("Silence detected. Stopping recording.")
                break

    # Combine all recorded chunks into one
    recorded_audio = np.concatenate(recorded_data, axis=0)

    # Save the recorded audio to a temporary file
    audio_file_path = "temp/temp_audio.wav"
    sf.write(audio_file_path, recorded_audio, 16000)
    return audio_file_path

def main():
    while True:
        # Listen for the custom wake word
        if listen_for_wake_word():
            print("Wake word detected! You can start speaking...")

            # Record audio until silence is detected for 2 seconds, using the specified device index
            audio_file_path = record_audio_until_silence(device_index=device_index)

            # Call the speech recognition function
            recognized_text = recognize_speech(audio_file_path)

            if recognized_text:
                print("Recognized Text:", recognized_text)
                # Save recognized text to a file
                save_text_to_file(recognized_text, "recognized_text.txt")

                # Generate response using GPT-4o Turbo
                response_text = generate_response(recognized_text)
                print("Response Text:", response_text)
                # Save response text to a file
                save_text_to_file(response_text, "response_text.txt")

                # Convert the response text to speech and play it
                text_to_speech(response_text)

if __name__ == "__main__":
    main()
