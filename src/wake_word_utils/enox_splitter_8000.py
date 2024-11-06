import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks


def split_enox_segments(file_path, output_dir, sample_rate=16000, segment_duration=0.7, pre_segment_buffer=0.1):
    """
    Splits segments where "enox" is spoken in an audio file and saves them as separate files.

    Parameters:
    - file_path (str): Path to the audio file.
    - output_dir (str): Directory where split segments will be saved.
    - sample_rate (int): Sampling rate for loading the audio file.
    - segment_duration (float): Duration of each "enox" segment in seconds.
    - pre_segment_buffer (float): Additional buffer before each segment in seconds.
    """
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=sample_rate)

    # Compute short-term energy as an indicator of speech segments
    frame_length = int(0.025 * sr)  # 25 ms per frame
    hop_length = int(0.01 * sr)  # 10 ms hop between frames
    energy = np.array([
        sum(abs(audio[i:i + frame_length] ** 2))
        for i in range(0, len(audio), hop_length)
    ])

    # Detect peaks in the energy signal to locate "enox" segments
    peaks, _ = find_peaks(energy, height=np.mean(energy), distance=int(0.5 * sr / hop_length))
    segment_samples = int(segment_duration * sr)
    pre_buffer_samples = int(pre_segment_buffer * sr)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split and save each segment
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    segment_count = 1
    for peak in peaks:
        start_sample = max(0, peak * hop_length - pre_buffer_samples)
        end_sample = start_sample + segment_samples
        segment_audio = audio[start_sample:end_sample]

        # Save the segment as a new file
        segment_file_path = os.path.join(output_dir, f"{base_name}_segment_{segment_count}.wav")
        sf.write(segment_file_path, segment_audio, sample_rate)
        print(f"Saved segment: {segment_file_path}")
        segment_count += 1


def process_directory(input_dir, output_dir, sample_rate=16000, segment_duration=0.7, pre_segment_buffer=0.1):
    """
    Processes all .wav files in a directory to split "enox" segments.

    Parameters:
    - input_dir (str): Directory containing the input audio files.
    - output_dir (str): Directory where split segments will be saved.
    - sample_rate (int): Sampling rate for loading audio files.
    - segment_duration (float): Duration of each "enox" segment in seconds.
    - pre_segment_buffer (float): Additional buffer before each segment in seconds.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            split_enox_segments(file_path, output_dir, sample_rate, segment_duration, pre_segment_buffer)


# Example usage
input_directory = "../wake_word_detection/enoxkokot/"
output_directory = "../wake_word_detection/splitter_results/"
process_directory(input_directory, output_directory)
