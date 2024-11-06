import os
import soundfile as sf
import numpy as np


def process_wav_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            # Read the stereo audio file
            data, sample_rate = sf.read(file_path)

            # Check if the file is stereo (2 channels)
            if data.shape[1] == 2:
                # Extract the left channel, duplicate it to create mono data
                left_channel = data[:, 0]
                mono_data = np.column_stack([left_channel, left_channel])

                # Write back as mono by overwriting the original file
                sf.write(file_path, mono_data, sample_rate)
                print(f"Processed {filename} - converted to mono.")
            else:
                print(f"{filename} is already mono, skipping.")


# Specify the directory containing the .wav files
directory = "../wake_word_data/wake_word"
process_wav_files(directory)
