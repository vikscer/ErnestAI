import os
import librosa
import numpy as np
import soundfile as sf
import random


def split_audio(file_path, output_dir, min_duration=0.6, max_duration=0.9, sample_rate=16000):
    """Split audio into segments of random lengths between min_duration and max_duration."""
    y, sr = librosa.load(file_path, sr=sample_rate)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Only split if the audio is longer than 0.8 seconds
    if total_duration > 0.8:
        start = 0
        segment_count = 1

        while start < total_duration:
            # Random segment duration between min_duration and max_duration
            segment_duration = random.uniform(min_duration, max_duration)
            end = start + segment_duration

            # Ensure we don’t exceed the total duration of the audio
            if end > total_duration:
                end = total_duration

            # Convert time in seconds to sample indices
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # Extract segment and save as a new file
            segment = y[start_sample:end_sample]
            output_file = os.path.join(output_dir,
                                       f"{os.path.splitext(os.path.basename(file_path))[0]}_segment_{segment_count}.wav")
            sf.write(output_file, segment, sr)
            print(f"Saved segment {segment_count} for {file_path} to {output_file}")

            segment_count += 1
            start = end


def process_directory(directory, min_duration=0.6, max_duration=0.9, sample_rate=16000):
    """Process all .wav files in a directory, splitting each into randomized segments."""
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            split_audio(file_path, directory, min_duration, max_duration, sample_rate)


# Specify the directory containing the .wav files
directory = "../wake_word_data/background_noise"
process_directory(directory)
