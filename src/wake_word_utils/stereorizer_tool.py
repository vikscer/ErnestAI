import os
import soundfile as sf
import numpy as np

def process_audio_file(file_path):
    # Load the audio file
    data, samplerate = sf.read(file_path)

    # Check if the audio is stereo
    if len(data.shape) == 2:
        # If stereo, keep only the left channel
        left_channel = data[:, 0]
        # Create a new stereo signal with right channel empty
        modified_data = np.column_stack((left_channel, np.zeros_like(left_channel)))
    else:
        # If mono, create a stereo signal with right channel empty
        modified_data = np.column_stack((data, np.zeros_like(data)))

    # Save the modified audio, overwriting the original file
    sf.write(file_path, modified_data, samplerate)
    print(f"Processed {file_path}")

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            process_audio_file(file_path)

# Example usage
directory = "../wake_word_data/background_noise"
process_directory(directory)
