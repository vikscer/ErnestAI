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
listening_tone_path = config['static']['listening_tone_path'] + config['assistant']['character'] + '.wav'

def save_text_to_file(text, filename):
    """Save the text to a specified file."""
    with open("temp/" + filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {filename}")

def record_audio_until_silence(duration=2, threshold=0.02, device_index=1):
    """Record audio from the specified device until silence is detected for a specified duration."""
    recorded_data = []
    silence_start_time = None
    buffer_duration = 2  # Record in 2-second chunks
    sample_rate = 32000
    buffer_samples = int(buffer_duration * sample_rate)

    while True:
        audio_data = sd.rec(buffer_samples, samplerate=sample_rate, channels=1, device=device_index)
        sd.wait()

        if np.max(np.abs(audio_data)) > threshold:
            if silence_start_time is not None:
                silence_start_time = None  # Reset silence timer if speech is detected
            recorded_data.append(audio_data)
        else:
            if silence_start_time is None:
                silence_start_time = time.time()
            if silence_start_time and time.time() - silence_start_time >= duration:
                break

    try:
        recorded_audio = np.concatenate(recorded_data, axis=0)
        audio_file_path = "temp/temp_audio.wav"
        sf.write(audio_file_path, recorded_audio, sample_rate)
        print(f"Audio saved to {audio_file_path}")
        return audio_file_path
    except Exception as e:
        print(f"Error recording audio: {e}")
        return False

def wait_for_follow_up(threshold=0.02, wait_duration=1):
    """Wait briefly after responding to detect if the user wants to continue speaking."""
    print("Waiting for follow-up...")

    silence_start_time = None
    sample_rate = 32000
    buffer_samples = int(0.5 * sample_rate)  # Check in 0.5-second chunks

    while True:
        audio_data = sd.rec(buffer_samples, samplerate=sample_rate, channels=1, device=device_index)
        sd.wait()

        if np.max(np.abs(audio_data)) > threshold:
            return True  # User started talking again
        else:
            if silence_start_time is None:
                silence_start_time = time.time()
            if silence_start_time and time.time() - silence_start_time >= wait_duration:
                return False  # No input from the user

def main():
    while True:
        # Listen for the custom wake word
        if listen_for_wake_word():
            print("Wake word detected! You can start speaking...")

            while True:
                # Record audio until silence is detected for 2 seconds
                recorded_audio = record_audio_until_silence(device_index=device_index)
                if recorded_audio:
                    recognized_text = recognize_speech(recorded_audio)

                    if recognized_text:
                        print("Recognized Text:", recognized_text)
                        save_text_to_file(recognized_text, "recognized_text.txt")

                        # Generate and respond
                        response_text = generate_response(recognized_text)
                        print("Response Text:", response_text)
                        save_text_to_file(response_text, "response_text.txt")
                        text_to_speech(response_text)

                        # Check if the user wants to continue talking within 2 seconds
                        if not wait_for_follow_up():
                            print("No follow-up detected. Returning to wake word detection.")
                            break  # Exit the loop if no follow-up detected

if __name__ == "__main__":
    main()
